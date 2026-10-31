import logging
import math
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Chuyển đổi điểm logit của Reranker về xác suất xác thực (0.0 đến 1.0)."""
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


class CrossEncoderReranker:
    """Mô hình Cross-Encoder Reranker chạy cục bộ để tái xếp hạng các đoạn văn bản truy xuất từ BM25 và VectorStore."""

    _instance: Optional["CrossEncoderReranker"] = None

    def __new__(cls, *args, **kwargs):
        # Triển khai Singleton để tránh tải lại trọng số mô hình nhiều lần vào bộ nhớ
        if cls._instance is None:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.model = None
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Khởi tạo mô hình CrossEncoder từ cache hoặc tải về an toàn."""
        try:
            logger.info(f"Đang nạp mô hình Cross-Encoder Reranker: {self.model_name}...")
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            logger.info(f"✓ Khởi tạo thành công Cross-Encoder Reranker: {self.model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi nạp CrossEncoder ({self.model_name}): {e}. Hệ thống sẽ sử dụng fallback.")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Document]:
        """Tái xếp hạng danh sách documents dựa trên tương quan ngữ nghĩa thực tế với câu hỏi.

        Args:
            query: Câu hỏi của người dùng.
            documents: Danh sách documents từ bộ tìm kiếm cơ sở.
            top_k: Số lượng kết quả tốt nhất cần lấy.

        Returns:
            Danh sách documents đã được sắp xếp lại theo mức độ liên quan giảm dần.
        """
        if not documents:
            return []

        if len(documents) == 1:
            documents[0].metadata["rerank_score"] = 1.0
            return documents

        if self.model is None:
            logger.warning("Reranker chưa sẵn sàng, giữ nguyên thứ tự ban đầu.")
            return documents[:top_k]

        try:
            # Tạo các cặp (query, doc_content) để đưa vào Cross-Encoder
            pairs = [[query, doc.page_content.strip()] for doc in documents]
            scores = self.model.predict(pairs)

            ranked_docs = []
            for doc, raw_score in zip(documents, scores):
                prob = sigmoid(float(raw_score))
                # Gắn điểm đánh giá vào metadata để tiện kiểm tra và đo lường độ tin cậy
                doc.metadata["rerank_score"] = round(prob, 4)
                doc.metadata["raw_rerank_score"] = round(float(raw_score), 4)
                ranked_docs.append(doc)

            # Sắp xếp giảm dần theo điểm tương quan
            ranked_docs.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)

            logger.info(
                f"✓ Đã rerank {len(documents)} tài liệu cho query '{query[:30]}...'. "
                f"Điểm cao nhất: {ranked_docs[0].metadata.get('rerank_score', 0):.4f}"
            )
            return ranked_docs[:top_k]
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện reranking: {e}. Fallback về thứ tự gốc.")
            return documents[:top_k]

