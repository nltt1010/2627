import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.core.config import get_settings
from src.core.key_manager import key_manager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về dược liệu và nông nghiệp Việt Nam (VietHerb-AI).
Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng dựa TRỰC TIẾP và DUY NHẤT vào ngữ cảnh tài liệu được cung cấp dưới đây.

--- QUY TẮC PHẢN HỒI ---
1. Chỉ sử dụng thông tin có trong phần 'NGỮ CẢNH TÀI LIỆU'. Tuyệt đối không tự bịa đặt hoặc suy diễn các công dụng y học/liều lượng không có trong tài liệu.
2. Nếu ngữ cảnh không đủ thông tin để trả lời câu hỏi, hãy thẳng thắn thông báo: "Tài liệu hiện tại chưa có thông tin về vấn đề này."
3. Cuối câu trả lời, luôn liệt kê rõ ràng các nguồn tài liệu đã tham khảo (Metadata: source).
4. Giữ giọng văn khách quan, khoa học và súc tích.

--- NGỮ CẢNH TÀI LIỆU ---
{context}
"""


class RAGGenerator:
    """Xử lý kết hợp ngữ cảnh từ Hybrid Retriever và sinh câu trả lời bằng Gemini LLM."""

    def __init__(self, hybrid_retriever: HybridRetrieverManager):
        self.settings = get_settings()
        self.retriever = hybrid_retriever

        # Lấy key hợp lệ đã được kiểm duyệt cho Chat LLM
        selected_key = key_manager.get_key_for_chat(self.settings.DEFAULT_LLM_MODEL)

        # Khởi tạo mô hình Gemini 2.5 Flash
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.DEFAULT_LLM_MODEL,
            google_api_key=selected_key,
            temperature=0.2,  # Nhiệt độ thấp để giảm tối đa ảo giác
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.output_parser = StrOutputParser()

    def _format_context(self, docs: List[Document]) -> str:
        """Định dạng các đoạn văn bản trích xuất kèm nguồn tham khảo."""
        formatted_chunks = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Không rõ")
            formatted_chunks.append(f"[Đoạn {idx} - Nguồn: {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Quy trình RAG hoàn chỉnh: Truy xuất ngữ cảnh -> Đưa vào Prompt -> Sinh câu trả lời."""
        # 1. Truy xuất tài liệu liên quan bằng Hybrid Search
        relevant_docs = self.retriever.get_relevant_documents(query=question, top_k=top_k)

        if not relevant_docs:
            return {
                "answer": "Không tìm thấy dữ liệu liên quan trong kho tri thức.",
                "sources": [],
            }

        # 2. Định dạng ngữ cảnh
        context_str = self._format_context(relevant_docs)

        # 3. Tạo chuỗi thực thi (LCEL Chain)
        chain = self.prompt_template | self.llm | self.output_parser

        # 4. Sinh phản hồi
        answer = chain.invoke({
            "context": context_str,
            "question": question,
        })

        # 5. Thu thập danh sách nguồn duy nhất
        sources = list({doc.metadata.get("source", "Không rõ") for doc in relevant_docs})

        return {
            "answer": answer,
            "sources": sources,
        }