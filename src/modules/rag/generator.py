import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.core.config import get_settings
from src.core.key_manager import key_manager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về dược liệu và nông nghiệp Việt Nam (VietHerb-AI).

--- THỐNG KÊ KHO DỮ LIỆU THỰC TẾ CỦA HỆ THỐNG ---
- Tổng số loài thảo dược hiện có: {total_count} loài.
- Danh sách cụ thể các loài có trong hệ thống: {herb_list}.
* QUY TẮC BẮT BUỘC: Nếu người dùng hỏi về số lượng ("có bao nhiêu loài", "có mấy cây", "liệt kê danh sách các cây"), bạn PHẢI dựa trực tiếp vào mục thống kê trên để trả lời chính xác con số {total_count} và kể tên các loài cây tương ứng.

--- QUY TẮC PHẢN HỒI Y KHOA / DƯỢC TÍNH ---
1. Chỉ sử dụng thông tin có trong phần 'NGỮ CẢNH TÀI LIỆU'. Tuyệt đối không tự bịa đặt hoặc suy diễn các công dụng y học/liều lượng không có trong tài liệu.
2. Nếu ngữ cảnh không đủ thông tin để trả lời câu hỏi, hãy thẳng thắn thông báo: "Tài liệu hiện tại chưa có thông tin về vấn đề này."
3. Cuối câu trả lời, luôn liệt kê rõ ràng các nguồn tài liệu đã tham khảo (Metadata: source).
4. Giữ giọng văn khách quan, khoa học và súc tích. Định dạng rõ ràng bằng Markdown.

--- NGỮ CẢNH TÀI LIỆU ---
{context}
"""


class RAGGenerator:
    """Xử lý kết hợp ngữ cảnh từ Hybrid Retriever và sinh câu trả lời bằng Gemini LLM."""

    def __init__(self, hybrid_retriever: HybridRetrieverManager, image_store: Optional[Any] = None):
        self.settings = get_settings()
        self.retriever = hybrid_retriever
        self.image_store = image_store

        # Sử dụng đúng cấu hình DEFAULT_LLM_MODEL của dự án
        model_name = getattr(self.settings, "DEFAULT_LLM_MODEL", "gemini-2.5-flash")

        # Lấy key hợp lệ đã được kiểm duyệt cho Chat LLM
        selected_key = key_manager.get_key_for_chat(model_name)

        # Khởi tạo mô hình Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=selected_key,
            temperature=0.2,  # Nhiệt độ thấp để giảm tối đa ảo giác
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.output_parser = StrOutputParser()

    def _get_live_database_summary(self) -> Dict[str, Any]:
        """Quét và tổng hợp chính xác dữ liệu hiện hữu từ kho raw và vector store."""
        raw_dir = Path("data/raw")
        herbs_in_system = []

        if raw_dir.exists():
            for f in raw_dir.glob("*.txt"):
                herb_name = f.stem.replace("_", " ").title()
                herbs_in_system.append(herb_name)

        # Lấy thêm các loài cây có trong kho ảnh ChromaDB nếu có image_store
        image_herbs = self.image_store.get_available_herbs() if self.image_store else []
        all_unique_herbs = sorted(list(set(herbs_in_system + image_herbs)))

        return {
            "total_count": len(all_unique_herbs),
            "herb_list": all_unique_herbs,
        }

    def _format_context(self, docs: List[Document]) -> str:
        """Định dạng các đoạn văn bản trích xuất kèm nguồn tham khảo."""
        formatted_chunks = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Không rõ")
            formatted_chunks.append(f"[Đoạn {idx} - Nguồn: {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Quy trình RAG hoàn chỉnh: Lấy thống kê kho -> Truy xuất ngữ cảnh -> Đưa vào Prompt -> Sinh câu trả lời."""
        # 1. Lấy dữ liệu thống kê trực tiếp từ kho
        db_summary = self._get_live_database_summary()
        herb_list_str = ", ".join(db_summary["herb_list"]) if db_summary["herb_list"] else "Chưa có loài nào"

        # 2. Truy xuất tài liệu liên quan bằng Hybrid Search
        relevant_docs = self.retriever.get_relevant_documents(query=question, top_k=top_k)

        # 3. Định dạng ngữ cảnh
        if relevant_docs:
            context_str = self._format_context(relevant_docs)
            sources = list({doc.metadata.get("source", "Không rõ") for doc in relevant_docs})
        else:
            context_str = "Hiện không tìm thấy đoạn trích dẫn cụ thể nào khớp với từ khóa tìm kiếm."
            sources = []

        # 4. Tạo chuỗi thực thi (LCEL Chain)
        chain = self.prompt_template | self.llm | self.output_parser

        # 5. Sinh phản hồi
        answer = chain.invoke({
            "total_count": db_summary["total_count"],
            "herb_list": herb_list_str,
            "context": context_str,
            "question": question,
        })

        return {
            "answer": answer,
            "sources": sources,
            "db_stats": db_summary,
        }