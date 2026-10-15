import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.vision.image_store import ImageVectorStore


def test_image_vectorstore():
    print("--- KIỂM TRA LOCAL IMAGE EMBEDDER & IMAGE VECTORSTORE (0 API CALL) ---")

    # 1. Tạo thư mục tạm và 2 ảnh mẫu giả lập
    test_img_dir = Path("data/sample_images")
    test_img_dir.mkdir(parents=True, exist_ok=True)

    img_sam_path = test_img_dir / "sample_sam_ngoc_linh.jpg"
    img_ca_path = test_img_dir / "sample_ca_gai_leo.jpg"

    # Tạo ảnh mẫu: Sâm Ngọc Linh (tone màu nâu củ) và Cà Gai Leo (tone màu xanh lá)
    Image.new("RGB", (200, 200), color=(150, 100, 50)).save(img_sam_path)
    Image.new("RGB", (200, 200), color=(34, 139, 34)).save(img_ca_path)

    # 2. Khởi tạo kho ảnh và nạp 2 ảnh mẫu
    image_store = ImageVectorStore(collection_name="test_herb_images")

    print("\n>> Đang nạp ảnh mẫu vào ChromaDB...")
    image_store.add_herb_image(
        image_input=img_sam_path,
        image_id="img_sam_01",
        herb_name="Sâm Ngọc Linh",
        scientific_name="Panax vietnamensis",
    )
    image_store.add_herb_image(
        image_input=img_ca_path,
        image_id="img_cagai_01",
        herb_name="Cà Gai Leo",
        scientific_name="Solanum procumbens",
    )

    # 3. Thử nghiệm so khớp ảnh (Giả lập người dùng gửi 1 ảnh màu xanh tương tự Cà Gai Leo)
    print("\n>> Giả lập người dùng gửi ảnh lá cây màu xanh...")
    query_green = Image.new("RGB", (200, 200), color=(40, 145, 40))
    match_result = image_store.search_similar_herb(query_image=query_green, threshold=0.70)

    print(f"✓ Kết quả nhận diện: {match_result}")
    assert match_result is not None
    assert match_result["matched"] is True
    assert match_result["herb_name"] == "Cà Gai Leo"
    print(f"✓ Nhận diện chính xác cây: {match_result['herb_name']} với độ khớp: {match_result['similarity_score'] * 100:.2f}%")


if __name__ == "__main__":
    test_image_vectorstore()