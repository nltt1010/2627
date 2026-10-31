import logging
from typing import List, Dict, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.reranker import CrossEncoderReranker
from src.modules.rag.query_transform import QueryTransformer

logger = logging.getLogger(__name__)


class HybridRetrieverManager:
    """Quản lý tìm kiếm kết hợp nâng cao (Advanced Hybrid Retrieval):
    ChromaDB (Dense) + BM25 (Sparse) + HyDE + Multi-Query + Cross-Encoder Reranker.
    """

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        documents: List[Document],
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        reranker: Optional[CrossEncoderReranker] = None,
        query_transformer: Optional[QueryTransformer] = None,
    ):
        self.vs_manager = vectorstore_manager
        self.documents = documents
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # 1. Sparse BM25
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5

        # 2. Dense VectorStore
        self.chroma_vs = self.vs_manager.get_vectorstore()

        # 3. Thành phần RAG Nâng cao
        self.reranker = reranker or CrossEncoderReranker()
        self.query_transformer = query_transformer or QueryTransformer()

        logger.info("✓ Khởi tạo thành công Advanced Hybrid Retriever (Dense + BM25 + HyDE + Multi-Query + Reranker).")

    def get_basic_hybrid_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Truy xuất cơ bản dựa trên xếp hạng RRF giữa Dense và Sparse."""
        dense_docs = self.chroma_vs.similarity_search(query=query, k=top_k)
        sparse_docs = self.bm25_retriever.invoke(query)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            doc_map[content] = doc
            score = self.dense_weight * (1.0 / (rank + 1))
            doc_scores[content] = doc_scores.get(content, 0.0) + score

        for rank, doc in enumerate(sparse_docs):
            content = doc.page_content
            doc_map[content] = doc
            score = self.sparse_weight * (1.0 / (rank + 1))
            doc_scores[content] = doc_scores.get(content, 0.0) + score

        sorted_contents = sorted(doc_scores.keys(), key=lambda c: doc_scores[c], reverse=True)
        return [doc_map[c] for c in sorted_contents[:top_k]]

    def get_relevant_documents(
        self,
        query: str,
        top_k: int = 3,
        use_advanced: bool = True,
    ) -> List[Document]:
        """Truy xuất tài liệu nâng cao:
        1. Query Expansion (Multi-Query) & HyDE: sinh tài liệu giả định và các câu hỏi phụ.
        2. Tìm kiếm đa diện trong Dense (vector câu hỏi + vector HyDE) và Sparse (BM25 câu hỏi + sub-queries).
        3. Gom tập hợp ứng viên và khử trùng lặp.
        4. Cross-Encoder Reranker chạy cục bộ để chấm điểm tương quan thực tế và chọn top_k chuẩn xác nhất.
        """
        if not use_advanced or len(self.documents) <= top_k:
            return self.get_basic_hybrid_documents(query=query, top_k=top_k)

        try:
            # 1. BIẾN ĐỔI QUERY: HyDE & Multi-Query
            transform_res = self.query_transformer.transform(query)
            hyde_doc = transform_res.get("hypothetical_doc", "")
            sub_queries = transform_res.get("sub_queries", [])

            candidate_map: Dict[str, Document] = {}

            # 2. DENSE RETRIEVAL (ChromaDB)
            # 2.1 Truy xuất theo query gốc
            for doc in self.chroma_vs.similarity_search(query=query, k=4):
                candidate_map[doc.page_content] = doc

            # 2.2 Truy xuất theo HyDE (tài liệu giả định)
            if hyde_doc and hyde_doc != query:
                for doc in self.chroma_vs.similarity_search(query=hyde_doc, k=4):
                    candidate_map[doc.page_content] = doc

            # 3. SPARSE RETRIEVAL (BM25)
            # 3.1 Truy xuất theo query gốc
            for doc in self.bm25_retriever.invoke(query):
                candidate_map[doc.page_content] = doc

            # 3.2 Truy xuất theo các sub-queries
            for sq in sub_queries:
                for doc in self.bm25_retriever.invoke(sq):
                    candidate_map[doc.page_content] = doc

            candidate_docs = list(candidate_map.values())
            logger.info(f"Thu thập được {len(candidate_docs)} ứng viên cho câu hỏi '{query[:35]}...'")

            # 4. CROSS-ENCODER RERANKING
            # Đưa toàn bộ ứng viên qua Reranker chấm điểm tương quan thực tế
            reranked_docs = self.reranker.rerank(
                query=query,
                documents=candidate_docs,
                top_k=top_k,
            )

            return reranked_docs

        except Exception as e:
            logger.error(f"Lỗi trong quy trình Advanced Retrieval: {e}. Fallback về Basic Hybrid.")
            return self.get_basic_hybrid_documents(query=query, top_k=top_k)
