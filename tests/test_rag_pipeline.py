import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.generator import RAGGenerator


def test_full_rag_pipeline():
    print("--- KIỂM TRA TOÀN DIỆN RAG PIPELINE (HYBRID RETRIEVAL + GEMINI FLASH) ---")

    # 1. Chuẩn bị dữ liệu và retriever
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    vs_manager = VectorStoreManager(collection_name="full_rag_collection")
    vs_manager.add_documents(chunks)

    hybrid_retriever = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks,
    )

    # 2. Khởi tạo generator
    generator = RAGGenerator(hybrid_retriever=hybrid_retriever)

    # Câu hỏi 1: Có trong tài liệu
    question_1 = "Sâm Ngọc Linh sống ở độ cao nào và chứa bao nhiêu hợp chất saponin?"
    print(f"\n[Câu hỏi 1]: {question_1}")
    result_1 = generator.answer_question(question_1)
    print(f"\n[VietHerb-AI Trả lời]:\n{result_1['answer']}")
    print(f"\n[Nguồn trích dẫn]: {result_1['sources']}")

    # Câu hỏi 2: Không có trong tài liệu (Kiểm tra chống bịa đặt/hallucination)
    question_2 = "Cây Đinh Lăng chữa được bệnh ung thư gan không?"
    print(f"\n--------------------------------------------------")
    print(f"[Câu hỏi 2 - Ngoài tài liệu]: {question_2}")
    result_2 = generator.answer_question(question_2)
    print(f"\n[VietHerb-AI Trả lời]:\n{result_2['answer']}")


if __name__ == "__main__":
    test_full_rag_pipeline()