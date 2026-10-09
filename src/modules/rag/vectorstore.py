import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from src.core.config import get_settings
from src.core.key_manager import key_manager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Quản lý ChromaDB và tích hợp Google Cloud Embedding API thông qua Key Manager."""

    def __init__(self, collection_name: str = "vietherb_knowledge"):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = self.settings.CHROMA_PERSIST_DIR

        # Đảm bảo các key đã được quét năng lực trước khi sử dụng
        if not key_manager._key_registry:
            key_manager.discover_keys_capabilities()

        # Lấy key hợp lệ đã được kiểm chứng hỗ trợ embedding
        selected_key = key_manager.get_key_for_embedding(self.settings.DEFAULT_EMBEDDING_MODEL)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings.DEFAULT_EMBEDDING_MODEL,
            google_api_key=selected_key,
        )

        self._vectorstore: Optional[Chroma] = None

    def get_vectorstore(self) -> Chroma:
        """Khởi tạo hoặc nạp instance ChromaDB đã lưu trữ trên đĩa."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """Đưa danh sách document chunks vào cơ sở dữ liệu vector."""
        if not documents:
            logger.warning("Không có documents nào để thêm vào VectorStore.")
            return

        vectorstore = self.get_vectorstore()
        logger.info(f"Đang tạo embeddings và lưu {len(documents)} chunks vào ChromaDB...")
        vectorstore.add_documents(documents=documents)
        logger.info("✓ Hoàn tất lưu trữ vector.")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Truy vấn tìm kiếm các đoạn văn bản tương đồng ngữ nghĩa nhất."""
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search(query=query, k=k)