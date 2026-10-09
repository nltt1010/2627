import logging
from typing import List, Dict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.modules.rag.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRetrieverManager:
    """Quản lý tìm kiếm kết hợp (Hybrid Search): ChromaDB (Dense) + BM25 (Sparse)."""

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        documents: List[Document],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.vs_manager = vectorstore_manager
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # 1. Sparse BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 3

        # 2. Dense VectorStore
        self.chroma_vs = self.vs_manager.get_vectorstore()
        logger.info("✓ Khởi tạo thành công Hybrid Retriever (Dense + BM25).")

    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """Truy xuất và hợp nhất kết quả từ Dense và Sparse bằng thuật toán chấm điểm xếp hạng."""
        # Lấy kết quả từ Dense
        dense_docs = self.chroma_vs.similarity_search(query=query, k=top_k)

        # Lấy kết quả từ Sparse BM25
        sparse_docs = self.bm25_retriever.invoke(query)

        # Hợp nhất và loại trừ trùng lặp dựa trên page_content
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Chấm điểm kết quả từ Dense
        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            doc_map[content] = doc
            # Tính điểm theo thứ hạng (RRF ranking)
            score = self.dense_weight * (1.0 / (rank + 1))
            doc_scores[content] = doc_scores.get(content, 0.0) + score

        # Chấm điểm kết quả từ Sparse
        for rank, doc in enumerate(sparse_docs):
            content = doc.page_content
            doc_map[content] = doc
            score = self.sparse_weight * (1.0 / (rank + 1))
            doc_scores[content] = doc_scores.get(content, 0.0) + score

        # Sắp xếp các đoạn văn theo điểm từ cao xuống thấp
        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)

        return [doc_map[c] for c in sorted_contents[:top_k]]