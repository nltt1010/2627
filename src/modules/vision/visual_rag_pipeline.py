import base64
import io
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from src.core.config import get_settings
from src.core.key_manager import key_manager
from src.modules.vision.image_store import ImageVectorStore
from src.modules.rag.hybrid_retriever import HybridRetrieverManager

logger = logging.getLogger(__name__)

VISUAL_RAG_PROMPT = """Bạn là trợ lý chuyên gia dược liệu VietHerb-AI.
Dưới đây là hình ảnh do người dùng cung cấp và thông tin trích xuất từ kho tài liệu chuẩn xác về loài cây đã được nhận diện.

--- KẾT QUẢ ĐỊNH DANH HÌNH ẢNH BAN ĐẦU ---
Tên cây nhận diện: {herb_name} ({scientific_name})
Độ tương đồng đặc trưng ảnh: {similarity_score}%

--- NGỮ CẢNH TÀI LIỆU DƯỢC TÍNH (CHUẨN) ---
{context}

--- YÊU CẦU ---
1. Quan sát hình ảnh đính kèm và xác nhận nhanh các đặc điểm nhận dạng nhìn thấy trong ảnh (màu sắc, hình dạng lá, thân hoặc củ) có khớp với mô tả loài cây này không.
2. Dựa vào NGỮ CẢNH TÀI LIỆU, trả lời câu hỏi của người dùng một cách chính xác, súc tích và khách quan.
3. Nếu người dùng không hỏi cụ thể, hãy tóm tắt: Tên khoa học, đặc điểm, công dụng chính và liều dùng/lưu ý.
4. Ghi rõ nguồn tài liệu tham khảo ở cuối câu trả lời.
"""


class VisualRAGPipeline:
    """Pipeline Visual RAG kết hợp Image Vector Search cục bộ và Gemini 2.5 Flash."""

    def __init__(
        self,
        image_store: ImageVectorStore,
        text_retriever: HybridRetrieverManager,
        similarity_threshold: float = 0.70,
    ):
        self.settings = get_settings()
        self.image_store = image_store
        self.text_retriever = text_retriever
        self.threshold = similarity_threshold
        self.parser = StrOutputParser()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        key = key_manager.get_key_for_chat(self.settings.DEFAULT_LLM_MODEL)
        return ChatGoogleGenerativeAI(
            model=self.settings.DEFAULT_LLM_MODEL,
            google_api_key=key,
            temperature=0.2,
        )

    def process_image_query(
        self,
        image_input: Union[str, Path, Image.Image],
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Xử lý truy vấn ảnh có kiểm tra trạng thái dữ liệu ảnh mẫu."""
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")

        # 1. Kiểm tra kho ảnh có dữ liệu không
        if self.image_store.total_images == 0:
            return {
                "matched": False,
                "empty_store": True,
                "herb_name": None,
                "similarity_score": 0.0,
                "answer": "Hệ thống hiện tại chưa có dữ liệu hình ảnh mẫu nào trong kho tri thức để đối chiếu. Vui lòng nạp ảnh mẫu cho các loài thảo dược trước khi tra cứu bằng hình ảnh.",
                "sources": [],
            }

        # 2. So khớp ảnh cục bộ (0 API call)
        match_result = self.image_store.search_similar_herb(
            query_image=pil_image,
            threshold=self.threshold,
        )

        if not match_result or not match_result.get("matched"):
            score = match_result.get("similarity_score", 0.0) if match_result else 0.0
            available_herbs = self.image_store.get_available_herbs()
            herbs_str = ", ".join(available_herbs) if available_herbs else "Chưa có"

            return {
                "matched": False,
                "empty_store": False,
                "herb_name": None,
                "similarity_score": score,
                "answer": (
                    f"Không nhận diện được thảo dược từ ảnh cung cấp (Độ khớp cao nhất: {score * 100:.1f}%, dưới ngưỡng an toàn {self.threshold * 100:.0f}%).\n\n"
                    f"📌 *Lưu ý: Kho hình ảnh hiện tại mới chỉ có mẫu đối chiếu cho các loài: {herbs_str}.* "
                    f"Nếu cây này chưa nằm trong danh sách trên, vui lòng cập nhật thêm ảnh mẫu vào hệ thống."
                ),
                "sources": [],
            }

        herb_name = match_result["herb_name"]
        scientific_name = match_result.get("scientific_name", "")
        sim_score = round(match_result["similarity_score"] * 100, 2)

        # 2. Truy xuất tài liệu văn bản bằng Hybrid Search
        query_text = user_question.strip() if user_question and user_question.strip() else f"Đặc điểm, công dụng và dược tính của cây {herb_name}"
        retrieved_docs = self.text_retriever.get_relevant_documents(query=f"{herb_name} {query_text}", top_k=3)

        context_chunks = []
        sources = set()
        for idx, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source", "Không rõ")
            sources.add(src)
            context_chunks.append(f"[Đoạn {idx} - Nguồn: {src}]\n{doc.page_content.strip()}")
        context_str = "\n\n".join(context_chunks)

        # 3. Chuẩn bị ảnh dạng base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt_text = VISUAL_RAG_PROMPT.format(
            herb_name=herb_name,
            scientific_name=scientific_name,
            similarity_score=sim_score,
            context=context_str,
        )
        if user_question and user_question.strip():
            prompt_text += f"\n--- CÂU HỎI BỔ SUNG CỦA NGƯỜI DÙNG ---\n{user_question.strip()}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
        )

        # 4. Gọi Gemini duy nhất 1 lần để sinh câu trả lời
        llm = self._get_llm()
        response = llm.invoke([message])

        return {
            "matched": True,
            "herb_name": herb_name,
            "scientific_name": scientific_name,
            "similarity_score": sim_score,
            "answer": response.content,
            "sources": list(sources),
        }