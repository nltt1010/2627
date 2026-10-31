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

FORMATTER_SYSTEM_PROMPT = """Bạn là chuyên gia Dược điển & Biên tập viên cẩm nang Thảo dược Việt Nam.
Nhiệm vụ của bạn là đọc nội dung văn bản thô, bài báo, tài liệu y học hoặc ghi chép dân gian về một loài cây thuốc và CHUẨN HÓA CẤU TRÚC theo đúng định dạng chuẩn của cẩm nang VietHerb.

QUY TẮC ĐỊNH DẠNG BẮT BUỘC:
Dòng 1: [Tên tiếng Việt chuẩn] (danh pháp khoa học: [Tên Latin in nghiêng])
Các phần tiếp theo ngăn cách rõ ràng bằng tiêu đề:
- Tên gọi khác:
- Đặc điểm thực vật:
- Bộ phận dùng:
- Dược tính và thành phần hóa học:
- Công dụng chính:
- Bài thuốc ứng dụng & Liều dùng:
- Kiêng kỵ & Lưu ý:

Nếu thông tin đầu vào thiếu phần nào, hãy dựa trên kiến thức dược liệu học chuẩn để bổ sung súc tích, chính xác hoặc ghi rõ "Đang cập nhật".

ĐẦU RA:
Trả về DUY NHẤT một chuỗi JSON hợp lệ (không có văn bản ngoài JSON) theo định dạng:
{{
  "herb_name": "Tên tiếng Việt viết hoa chữ cái đầu",
  "scientific_name": "Tên danh pháp khoa học Latin",
  "formatted_content": "Toàn bộ nội dung hoàn chỉnh đã chuẩn hóa (bắt đầu bằng dòng 1: Tên (danh pháp khoa học: ...))",
  "summary": "Tóm tắt ngắn gọn 1-2 câu về loài cây này"
}}
"""


class DocumentFormattingAgent:
    """Tác nhân AI tự động phân tích cú pháp, trích xuất thực thể và chuẩn hóa tài liệu thảo dược."""

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
                temperature=0.2,
            )
        return self._llm

    def format_document(
        self,
        raw_text: str,
        herb_name: str = "",
        scientific_name: str = "",
    ) -> Dict[str, Any]:
        """Chuẩn hóa văn bản thô thành định dạng cẩm nang chuẩn VietHerb."""
        raw_clean = raw_text.strip()
        if not raw_clean:
            return {
                "success": False,
                "error": "Nội dung văn bản trống!",
            }

        user_input_prompt = f"Văn bản thô cần chuẩn hóa:\n{raw_clean}"
        if herb_name:
            user_input_prompt += f"\nGợi ý tên cây: {herb_name}"
        if scientific_name:
            user_input_prompt += f"\nGợi ý danh pháp: {scientific_name}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", FORMATTER_SYSTEM_PROMPT),
            ("human", "{text}"),
        ])

        chain = prompt | self._get_llm() | StrOutputParser()

        try:
            raw_output = chain.invoke({"text": user_input_prompt})
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"```$", "", cleaned)
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            return {
                "success": True,
                "herb_name": parsed.get("herb_name", herb_name).strip(),
                "scientific_name": parsed.get("scientific_name", scientific_name).strip(),
                "formatted_content": parsed.get("formatted_content", raw_clean).strip(),
                "summary": parsed.get("summary", "").strip(),
            }
        except Exception as e:
            logger.error(f"Lỗi khi AI chuẩn hóa tài liệu: {e}")
            return {
                "success": False,
                "error": f"Lỗi xử lý AI: {str(e)}",
                "raw_text": raw_clean,
            }
