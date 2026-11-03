import json
import logging
import re
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.config import get_settings
from src.core.key_manager import key_manager
from src.modules.graph.graph_schema import (
    EntityType,
    RelationType,
    TripleItem,
    VALID_RELATION_CONSTRAINTS,
    normalize_label,
)

logger = logging.getLogger(__name__)

GRAPH_EXTRACTOR_SYSTEM_PROMPT = """Bạn là Chuyên gia Khai phá Tri thức (Knowledge Graph Extractor) về Thảo Dược và Y học Cổ truyền Việt Nam.
Nhiệm vụ của bạn là đọc kỹ văn bản tài liệu về thảo dược và trích xuất danh sách các bộ ba quan hệ (Triples) bám sát 100% nội dung tài liệu.

--- BỘ QUY TẮC NHÃN THỰC THỂ (ENTITY TYPES) ĐƯỢC PHÉP ---
1. Herb: Tên loài thảo dược (ví dụ: Cà Gai Leo, Sâm Ngọc Linh, Kim Ngân Hoa).
2. PlantPart: Bộ phận dùng làm thuốc (ví dụ: Thân lá, Rễ củ, Nụ hoa, Quả...).
3. Compound: Hoạt chất hóa học / Dược chất (ví dụ: Glycoalcaloid, Saponin, Solamargine, Flavonoid...).
4. Disease: Bệnh lý, chứng bệnh hoặc triệu chứng được hỗ trợ điều trị (ví dụ: Viêm Gan B, Men Gan Cao, Giải Độc Gan, Phong Thấp, Mụn Nhọt, Cảm Sốt...).
5. Contraindication: Đối tượng chống chỉ định, kiêng kỵ sử dụng (ví dụ: Phụ Nữ Mang Thai, Trẻ Em Dưới 6 Tuổi, Người Thể Hư Hàn...).

--- BỘ QUAN HỆ KHÉP KÍN (RELATION TYPES) ĐƯỢC PHÉP ---
1. (Herb) -[USES_PART]-> (PlantPart): Cây dùng bộ phận nào làm thuốc.
2. (Herb) -[CONTAINS_COMPOUND]-> (Compound): Cây chứa hoạt chất hoặc dược chất nào.
3. (Herb) -[TREATS_DISEASE]-> (Disease): Cây hỗ trợ điều trị hoặc cải thiện bệnh lý/triệu chứng gì.
4. (Compound) -[TREATS_DISEASE]-> (Disease): Hoạt chất cụ thể có tác dụng điều trị bệnh lý nào.
5. (Herb) -[CONTRAINDICATED_FOR]-> (Contraindication): Cây kiêng cữ cho đối tượng nào.

TUYỆT ĐỐI TUÂN THỦ:
- Không bịa đặt hoặc suy diễn thông tin ngoài tài liệu.
- Không sử dụng bất kỳ Relation Type hoặc Entity Type nào ngoài danh sách trên.
- Viết tên thực thể súc tích, rõ ràng, không kèm câu văn dài dòng.

ĐẦU RA:
Trả về DUY NHẤT một chuỗi JSON hợp lệ theo định dạng sau (không có markdown giải thích bên ngoài):
{{
  "triples": [
    {{
      "source_label": "Cà Gai Leo",
      "source_type": "Herb",
      "relation": "CONTAINS_COMPOUND",
      "target_label": "Glycoalcaloid",
      "target_type": "Compound"
    }},
    {{
      "source_label": "Cà Gai Leo",
      "source_type": "TREATS_DISEASE",
      "relation": "TREATS_DISEASE",
      "target_label": "Viêm Gan B",
      "target_type": "Disease"
    }}
  ]
}}
"""


class KnowledgeGraphExtractor:
    """Trích xuất đồ thị tri thức từ văn bản tài liệu thảo dược bằng LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or getattr(self.settings, "DEFAULT_LLM_MODEL", "gemini-2.5-flash")
        self._llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            chat_key = key_manager.get_key_for_chat(self.model_name)
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=chat_key,
                temperature=0.1,
            )
        return self._llm

    def extract_triples(self, document_text: str, doc_source: str = "") -> List[TripleItem]:
        """Trích xuất và chuẩn hóa các Triples từ văn bản tài liệu."""
        doc_clean = document_text.strip()
        if not doc_clean:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", GRAPH_EXTRACTOR_SYSTEM_PROMPT),
            ("human", "Văn bản thảo dược cần trích xuất tri thức:\n\n{text}"),
        ])

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            raw_output = chain.invoke({"text": doc_clean})
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"```$", "", cleaned)
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            raw_triples = data.get("triples", [])

            validated_triples: List[TripleItem] = []
            for item in raw_triples:
                s_label = normalize_label(str(item.get("source_label", "")))
                s_type = str(item.get("source_type", "")).strip()
                rel = str(item.get("relation", "")).strip()
                t_label = normalize_label(str(item.get("target_label", "")))
                t_type = str(item.get("target_type", "")).strip()

                if not s_label or not t_label:
                    continue

                # Kiểm tra tính hợp lệ của ràng buộc ontology (SourceType, Relation, TargetType)
                if (s_type, rel, t_type) in VALID_RELATION_CONSTRAINTS:
                    validated_triples.append(
                        TripleItem(
                            source_label=s_label,
                            source_type=s_type,
                            relation=rel,
                            target_label=t_label,
                            target_type=t_type,
                        )
                    )
                else:
                    logger.debug(
                        f"Bỏ qua triple không hợp lệ ontology: ({s_type}, {rel}, {t_type})"
                    )

            logger.info(
                f"✓ Đã trích xuất {len(validated_triples)} triples hợp lệ từ '{doc_source or 'text'}'"
            )
            return validated_triples

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất đồ thị tri thức từ '{doc_source}': {e}")
            return []

