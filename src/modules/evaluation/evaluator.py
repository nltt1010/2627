import json
import logging
import time
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.config import get_settings
from src.core.key_manager import key_manager

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """Bạn là chuyên gia thẩm định dữ liệu RAG Dược Liệu.
Nhiệm vụ: Đánh giá xem câu trả lời (Answer) có hoàn toàn dựa trên ngữ cảnh (Context) trích xuất được hay không.

[NGỮ CẢNH TRÍCH XUẤT]:
{context}

[CÂU TRẢ LỜI CỦA HỆ THỐNG]:
{answer}

Thang điểm từ 0.0 đến 1.0:
- 1.0: Mọi thông tin trong câu trả lời đều có bằng chứng trực tiếp từ ngữ cảnh, không tự bịa đặt hay suy diễn sai lệch.
- 0.5: Trả lời đúng một phần, có chi tiết không nằm trong ngữ cảnh.
- 0.0: Bịa đặt hoặc hoàn toàn không khớp với ngữ cảnh.

Trả về duy nhất chuỗi JSON hợp lệ theo định dạng:
{{"score": 1.0, "reason": "Giải thích ngắn gọn"}}
"""

RELEVANCE_PROMPT = """Bạn là chuyên gia thẩm định độ liên quan câu trả lời RAG.
Nhiệm vụ: Đánh giá xem câu trả lời (Answer) có đi thẳng vào trọng tâm câu hỏi (Question) hay không.

[CÂU HỎI]:
{question}

[CÂU TRẢ LỜI]:
{answer}

Thang điểm từ 0.0 đến 1.0:
- 1.0: Trả lời trúng đích, đầy đủ ý chính, không vòng vo lạc đề.
- 0.5: Trả lời đúng hướng nhưng còn thiếu ý hoặc hơi lan man.
- 0.0: Hoàn toàn lạc đề.

Trả về duy nhất chuỗi JSON hợp lệ theo định dạng:
{{"score": 1.0, "reason": "Giải thích ngắn gọn"}}
"""


class RAGEvaluator:
    """Đánh giá chất lượng RAG (Faithfulness & Answer Relevancy) có hỗ trợ xoay vòng Key khi chạm rate limit."""

    def __init__(self):
        self.settings = get_settings()
        self.parser = StrOutputParser()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lấy instance LLM với key hiện tại từ key_manager."""
        key = key_manager.get_key_for_chat(self.settings.DEFAULT_LLM_MODEL)
        return ChatGoogleGenerativeAI(
            model=self.settings.DEFAULT_LLM_MODEL,
            google_api_key=key,
            temperature=0.0,
        )

    def _execute_with_retry(self, chain, inputs: Dict[str, Any], max_retries: int = 4) -> str:
        """Thực thi chain, tự động xoay key và thử lại nếu chạm 429."""
        for attempt in range(max_retries):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"Key hiện tại chạm giới hạn 429. Đang chuyển key kế tiếp... (Lần {attempt + 1}/{max_retries})")
                    # Xoay key
                    key_manager.rotate_key()
                    # Khởi tạo lại chain với key mới
                    new_llm = self._get_llm()
                    prompt_template = chain.first
                    chain = prompt_template | new_llm | self.parser
                    time.sleep(2)
                else:
                    raise e
        raise RuntimeError("Tất cả các API key đều chạm ngưỡng hạn ngạch hoặc gặp lỗi!")

    def _parse_score(self, text: str) -> Dict[str, Any]:
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            return {"score": 0.0, "reason": "Lỗi định dạng JSON kết quả"}

    def evaluate_pipeline(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        total_faithfulness = 0.0
        total_relevance = 0.0

        for idx, tc in enumerate(test_cases, 1):
            context_str = "\n".join(tc["contexts"])

            # 1. Chấm điểm Faithfulness
            f_chain = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT) | self._get_llm() | self.parser
            f_res = self._execute_with_retry(f_chain, {"context": context_str, "answer": tc["answer"]})
            f_eval = self._parse_score(f_res)
            
            # Nghỉ nhẹ 2 giây tránh nghẽn RPM
            time.sleep(2)

            # 2. Chấm điểm Relevance
            r_chain = ChatPromptTemplate.from_template(RELEVANCE_PROMPT) | self._get_llm() | self.parser
            r_res = self._execute_with_retry(r_chain, {"question": tc["question"], "answer": tc["answer"]})
            r_eval = self._parse_score(r_res)

            f_score = float(f_eval.get("score", 0.0))
            r_score = float(r_eval.get("score", 0.0))

            total_faithfulness += f_score
            total_relevance += r_score

            results.append({
                "question": tc["question"],
                "faithfulness": f_score,
                "relevance": r_score,
                "reasons": {
                    "faithfulness": f_eval.get("reason", ""),
                    "relevance": r_eval.get("reason", ""),
                },
            })
            print(f"  + Hoàn tất đánh giá câu hỏi [{idx}/{len(test_cases)}]")
            time.sleep(2)

        n = len(test_cases)
        return {
            "mean_faithfulness": round(total_faithfulness / n, 3) if n > 0 else 0.0,
            "mean_relevance": round(total_relevance / n, 3) if n > 0 else 0.0,
            "details": results,
        }