import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.generator import RAGGenerator
from src.modules.evaluation.evaluator import RAGEvaluator


def run_evaluation():
    print("--- KHỞI TẠO HỆ THỐNG RAG ĐỂ THU THẬP DỮ LIỆU ĐÁNH GIÁ ---")
    # 1. Nạp tri thức
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    vs_manager = VectorStoreManager(collection_name="ragas_eval_collection")
    vs_manager.add_documents(chunks)

    retriever = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks,
    )
    generator = RAGGenerator(hybrid_retriever=retriever)

    # 2. Bộ câu hỏi thử nghiệm (Golden Test Cases)
    test_questions = [
        "Sâm Ngọc Linh chứa bao nhiêu hợp chất saponin và phân bố ở độ cao nào?",
        "Cà Gai Leo có công dụng chính là gì đối với gan?",
        "Cây Sâm Ngọc Linh cần điều kiện độ che phủ rừng tự nhiên là bao nhiêu?",
    ]

    eval_data = []

    print("\n>> Đang chạy suy luận qua từng câu hỏi...")
    for q in test_questions:
        # Lấy ngữ cảnh từ Hybrid Retriever
        retrieved_docs = retriever.get_relevant_documents(query=q, top_k=3)
        contexts = [d.page_content for d in retrieved_docs]

        # Lấy câu trả lời từ Generator
        result = generator.answer_question(q)
        answer = result["answer"]

        eval_data.append({
            "question": q,
            "contexts": contexts,
            "answer": answer,
        })
        print(f"✓ Hoàn tất thu thập: '{q[:40]}...'")

    # 3. Đánh giá chất lượng với Ragas
    print("\n--- BẮT ĐẦU CHẤM ĐIỂM VỚI RAGAS ---")
    evaluator = RAGEvaluator()
    score_report = evaluator.evaluate_pipeline(eval_data)

    print("\n================= KẾT QUẢ ĐÁNH GIÁ =================")
    print(score_report)
    print("====================================================")


if __name__ == "__main__":
    run_evaluation()