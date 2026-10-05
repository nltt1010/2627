import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.rag.ingestion import DocumentIngestion


def test_ingestion_pipeline():
    print("--- KIỂM TRA MODULE INGESTION & CHUNKING ---")
    ingestion = DocumentIngestion(chunk_size=250, chunk_overlap=30)

    # 1. Đọc file
    docs = ingestion.load_text_files()
    assert len(docs) > 0, "Không đọc được file tài liệu nào trong data/raw!"
    print(f"✓ Đọc thành công {len(docs)} tài liệu.")
    print(f"  Tên tài liệu: {docs[0].metadata.get('source')}")

    # 2. Cắt chunk
    chunks = ingestion.split_documents(docs)
    assert len(chunks) > 1, "Văn bản chưa được chia thành nhiều chunk!"
    print(f"✓ Cắt văn bản thành công: {len(chunks)} chunks.")
    print(f"  Nội dung chunk mẫu 1:\n  \"{chunks[0].page_content.strip()[:100]}...\"")
    print(f"  Metadata chunk: {chunks[0].metadata}")


if __name__ == "__main__":
    test_ingestion_pipeline()