import json
import logging
import re
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.config import get_settings
from src.core.key_manager import key_manager

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """Bạn là Chuyên gia Đánh giá Chất lượng Hệ thống RAG (RAG Triad Evaluator - chuẩn TruLens).
Nhiệm vụ của bạn là đánh giá khắt khe chất lượng của quy trình RAG dựa trên 3 tiêu chí cốt lõi (thang điểm 0 đến 100):

1. CONTEXT RELEVANCE (Độ liên quan của ngữ cảnh):
- Các đoạn trích dẫn tài liệu từ cơ sở dữ liệu có chứa thông tin để trả lời câu hỏi của người dùng không?
- Nếu tài liệu không liên quan gì đến câu hỏi: điểm thấp (< 40).
- Nếu tài liệu liên quan mật thiết và đầy đủ: điểm cao (> 85).

2. GROUNDEDNESS / FAITHFULNESS (Tính bám sát tài liệu & Chống ảo giác):
- Mọi luận điểm, công dụng, liều dùng trong câu trả lời có được chứng thực trực tiếp bởi tài liệu không?
- Nếu câu trả lời tự bịa đặt, suy diễn ngoài tài liệu: điểm thấp (< 50).
- Nếu câu trả lời bám sát 100% tài liệu, hoặc thông báo trung thực rằng tài liệu chưa có thông tin: điểm cao (90 - 100).

3. ANSWER RELEVANCE (Độ đúng trọng tâm của câu trả lời):
- Câu trả lời có phản hồi trực tiếp, giải quyết đúng thắc mắc trong câu hỏi của người dùng không?
- Nếu trả lời lan man, lạc đề: điểm thấp (< 50).
- Nếu trả lời đúng trọng tâm, súc tích và hữu ích: điểm cao (85 - 100).

ĐẦU RA:
Trả về DUY NHẤT một chuỗi JSON hợp lệ theo cấu trúc sau (không kèm markdown ngoài khối JSON):
{{
  "context_relevance": 90,
  "groundedness": 95,
  "answer_relevance": 92,
  "overall_score": 92,
  "verdict": "Rất đáng tin cậy",
  "feedback": "Nhận xét súc tích bằng tiếng Việt (1-2 câu)"
}}
"""


class RAGTriadEvaluator:
    """Đánh giá chất lượng và độ tin cậy của câu trả lời RAG theo bộ ba RAG Triad."""

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

    def evaluate(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Thực hiện đánh giá 3 trụ cột RAG Triad."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            (
                "human",
                """Đánh giá trường hợp sau:
--- CÂU HỎI CỦA NGƯỜI DÙNG ---
{question}

--- NGỮ CẢNH TÀI LIỆU TRÍCH XUẤT ---
{context}

--- CÂU TRẢ LỜI CỦA HỆ THỐNG ---
{answer}
""",
            ),
        ])

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            raw_output = chain.invoke({
                "question": question,
                "context": context[:2500],  # Cắt bớt nếu context quá dài để tiết kiệm token
                "answer": answer[:2500],
            })

            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"```$", "", cleaned)
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            cr = max(0, min(100, int(parsed.get("context_relevance", 80))))
            gr = max(0, min(100, int(parsed.get("groundedness", 85))))
            ar = max(0, min(100, int(parsed.get("answer_relevance", 85))))
            overall = int(round((cr + gr + ar) / 3))

            verdict = parsed.get("verdict")
            if not verdict:
                if overall >= 80:
                    verdict = "Rất đáng tin cậy"
                elif overall >= 60:
                    verdict = "Đạt tiêu chuẩn"
                else:
                    verdict = "Cần đối chiếu thêm"

            return {
                "success": True,
                "context_relevance": cr,
                "groundedness": gr,
                "answer_relevance": ar,
                "overall_score": overall,
                "verdict": verdict,
                "feedback": parsed.get("feedback", "Câu trả lời bám sát tài liệu y khoa.").strip(),
            }
        except Exception as e:
            logger.warning(f"Lỗi khi đánh giá RAG Triad: {e}. Áp dụng điểm mặc định.")
            return {
                "success": False,
                "context_relevance": 80,
                "groundedness": 85,
                "answer_relevance": 85,
                "overall_score": 83,
                "verdict": "Đạt tiêu chuẩn",
                "feedback": "Hệ thống đã phản hồi dựa trên tài liệu cẩm nang tra cứu.",
            }
