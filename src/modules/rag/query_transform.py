import json
import logging
import re
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.config import get_settings
from src.core.key_manager import key_manager

logger = logging.getLogger(__name__)

TRANSFORM_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về kỹ thuật truy xuất thông tin (Information Retrieval) trong lĩnh vực Thảo dược & Y học cổ truyền Việt Nam.

Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và thực hiện 2 việc:
1. HyDE (Hypothetical Document Embeddings): Viết một đoạn tài liệu y học giả định ngắn (khoảng 3-4 câu) mô tả chi tiết tên các cây thuốc dân gian tiềm năng, bộ phận dùng, tính vị và công dụng trị liệu phù hợp với triệu chứng/câu hỏi được nêu.
2. Query Expansion (Multi-Query): Phân rã câu hỏi thành 2-3 câu truy vấn tìm kiếm phụ (sub-queries) ngắn gọn, tập trung vào các khía cạnh khác nhau:
   - Tên thảo dược, thành phần dược tính hoặc bộ phận dùng
   - Công dụng chữa bệnh hoặc bài thuốc dân gian
   - Liều lượng, cách sắc uống hoặc kiêng kỵ

YÊU CẦU ĐẦU RA:
Trả về DUY NHẤT một đối tượng JSON hợp lệ (không kèm giải thích markdown ngoài khối JSON) có cấu trúc sau:
{{
  "hypothetical_doc": "Đoạn văn bản y học giả định...",
  "sub_queries": [
    "câu truy vấn phụ 1",
    "câu truy vấn phụ 2",
    "câu truy vấn phụ 3"
  ]
}}
"""


class QueryTransformer:
    """Xử lý mở rộng và biến đổi câu truy vấn: HyDE (Hypothetical Document Embeddings) và Multi-Query Generation."""

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or getattr(self.settings, "DEFAULT_LLM_MODEL", "gemini-2.5-flash")
        self._llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Khởi tạo LLM với key xoay vòng tự động."""
        if self._llm is None:
            chat_key = key_manager.get_key_for_chat(self.model_name)
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=chat_key,
                temperature=0.3,
            )
        return self._llm

    def transform(self, query: str) -> Dict[str, Any]:
        """Thực hiện song song HyDE và Multi-Query Expansion trong 1 lượt gọi LLM tối ưu độ trễ."""
        query_clean = query.strip()
        if not query_clean:
            return {
                "hypothetical_doc": "",
                "sub_queries": [],
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system", TRANSFORM_SYSTEM_PROMPT),
            ("human", 'Câu hỏi của người dùng: "{query}"'),
        ])

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            raw_output = chain.invoke({"query": query_clean})

            # Làm sạch chuỗi JSON nếu có bọc code block ```json ... ```
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"```$", "", cleaned)
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            hypo_doc = parsed.get("hypothetical_doc", "").strip()
            sub_queries = [str(q).strip() for q in parsed.get("sub_queries", []) if str(q).strip()]

            logger.info(
                f"✓ QueryTransformer thành công cho '{query_clean[:30]}...': "
                f"{len(sub_queries)} sub-queries, HyDE doc: {len(hypo_doc)} ký tự."
            )
            return {
                "hypothetical_doc": hypo_doc,
                "sub_queries": sub_queries,
            }
        except Exception as e:
            logger.warning(f"Lỗi khi biến đổi query bằng LLM: {e}. Sử dụng query gốc.")
            return {
                "hypothetical_doc": query_clean,
                "sub_queries": [query_clean],
            }

    def generate_hypothetical_document(self, query: str) -> str:
        """Tạo đoạn văn bản giả định phục vụ HyDE retrieval."""
        res = self.transform(query)
        return res.get("hypothetical_doc") or query

    def generate_sub_queries(self, query: str) -> List[str]:
        """Tạo các câu hỏi phụ phục vụ Multi-query retrieval."""
        res = self.transform(query)
        return res.get("sub_queries") or [query]
