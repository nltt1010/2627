import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Xử lý đọc tài liệu từ thư mục và cắt nhỏ (chunking) tối ưu cho tiếng Việt."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.settings = get_settings()

        # Bộ tách từ ưu tiên bảo toàn cấu trúc văn bản tự nhiên
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_text_files(self, directory: str = None) -> List[Document]:
        target_dir = Path(directory or self.settings.RAW_DATA_DIR)
        documents = []

        if not target_dir.exists():
            logger.warning(f"Thư mục không tồn tại: {target_dir}")
            return documents

        for file_path in target_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path.name},
                        )
                        documents.append(doc)
            except Exception as e:
                logger.error(f"Lỗi khi đọc file {file_path}: {e}")

        logger.info(f"Đã đọc {len(documents)} tài liệu thô từ {target_dir}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            f"Đã phân tách {len(documents)} tài liệu thành {len(chunks)} chunks nhỏ."
        )
        return chunks