import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.modules.vision.image_store import ImageVectorStore

def test_check_data():
    store = ImageVectorStore(collection_name="test_herb_images")
    
    print(f"Tổng số ảnh mẫu hiện có: {store.total_images}")
    available = store.get_available_herbs()
    print(f"Các loài cây đã có ảnh mẫu: {available}")
    
    # Kiểm tra từng cây
    print(f"- Sâm Ngọc Linh có ảnh mẫu không? -> {store.has_herb_image('Sâm Ngọc Linh')}")
    print(f"- Cà Gai Leo có ảnh mẫu không?    -> {store.has_herb_image('Cà Gai Leo')}")
    print(f"- Đinh Lăng có ảnh mẫu không?     -> {store.has_herb_image('Đinh Lăng')}")

if __name__ == "__main__":
    test_check_data()