import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.vision.image_store import ImageVectorStore
from src.modules.vision.visual_rag_pipeline import VisualRAGPipeline


def test_visual_rag():
    print("--- KIỂM TRA LUỒNG TÍCH HỢP VISUAL RAG PIPELINE (1 API CALL) ---")

    # 1. Nạp tài liệu văn bản
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    vs_manager = VectorStoreManager(collection_name="full_rag_collection")
    vs_manager.add_documents(chunks)

    text_retriever = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks,
    )

    # 2. Khởi tạo kho ảnh mẫu
    image_store = ImageVectorStore(collection_name="test_herb_images")

    # 3. Khởi tạo pipeline
    visual_rag = VisualRAGPipeline(
        image_store=image_store,
        text_retriever=text_retriever,
        similarity_threshold=0.70,
    )

    # 4. Giả lập ảnh test: ảnh xanh lá (Cà Gai Leo)
    test_img = Image.new("RGB", (200, 200), color=(38, 142, 38))
    question = "Cây này có dùng để giải độc gan và hạ men gan được không?"

    print(f"\n>> Gửi ảnh kiểm tra kèm câu hỏi: '{question}'")
    result = visual_rag.process_image_query(image_input=test_img, user_question=question)

    print("\n================= KẾT QUẢ PHẢN HỒI =================")
    print(f"Trạng thái nhận diện: {result['matched']}")
    print(f"Cây nhận diện:       {result['herb_name']} ({result.get('scientific_name')})")
    print(f"Độ khớp vector ảnh:  {result['similarity_score']}%")
    print(f"\n[VietHerb-AI]:\n{result['answer']}")
    print(f"\n[Nguồn]: {result['sources']}")
    print("====================================================")


if __name__ == "__main__":
    test_visual_rag()