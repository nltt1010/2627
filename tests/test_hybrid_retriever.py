import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager


def test_hybrid_search():
    print("--- KIỂM TRA MODULE HYBRID RETRIEVAL (DENSE + SPARSE BM25) ---")

    # 1. Chuẩn bị dữ liệu chunks
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    # 2. Khởi tạo VectorStore và nạp chunks nếu chưa có
    vs_manager = VectorStoreManager(collection_name="hybrid_test_collection")
    vs_manager.add_documents(chunks)

    # 3. Khởi tạo Hybrid Retriever
    hybrid_manager = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks
    )

    # 4. Thử nghiệm truy vấn bằng từ khóa kỹ thuật chính xác (thế mạnh BM25)
    query = "Panax vietnamensis"
    print(f"\n>> Đang tìm kiếm từ khóa chính xác: '{query}'")
    results = hybrid_manager.get_relevant_documents(query)

    assert len(results) > 0, "Không tìm thấy kết quả phù hợp!"
    print(f"✓ Tìm thấy {len(results)} kết quả phù hợp nhất từ Hybrid Search:")
    for idx, doc in enumerate(results[:2], 1):
        print(f"\n[Kết quả {idx}] (Nguồn: {doc.metadata.get('source')}):")
        print(f"\"{doc.page_content.strip()}\"")


if __name__ == "__main__":
    test_hybrid_search()