import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import chromadb
from PIL import Image

from src.core.config import get_settings
from src.modules.vision.image_embedder import local_image_embedder

logger = logging.getLogger(__name__)


class ImageVectorStore:
    """Quản lý kho vector ảnh mẫu thảo dược trên ChromaDB có hỗ trợ thư viện ảnh thực tế."""

    def __init__(self, collection_name: str = "herb_images_collection"):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = self.settings.CHROMA_PERSIST_DIR
        self.images_root = Path("data/herb_images")
        self.images_root.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def total_images(self) -> int:
        """Tổng số vector ảnh hiện có."""
        return self.collection.count()

    def get_available_herbs(self) -> List[str]:
        """Danh sách tên các loài cây đã có ảnh mẫu."""
        if self.total_images == 0:
            return []
        all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
        herbs: Set[str] = {m["herb_name"] for m in all_metadata if m and "herb_name" in m}
        return sorted(list(herbs))

    def get_herb_gallery(self, herb_name: str) -> List[Dict[str, Any]]:
        """Lấy toàn bộ danh sách ảnh mẫu của một loài cây cụ thể."""
        if self.total_images == 0:
            return []
        
        results = self.collection.get(
            where={"herb_name": herb_name},
            include=["metadatas"]
        )
        items = []
        if results and results["ids"]:
            for img_id, meta in zip(results["ids"], results["metadatas"]):
                items.append({
                    "id": img_id,
                    "file_path": meta.get("file_path", ""),
                    "image_type": meta.get("image_type", "general"),
                    "scientific_name": meta.get("scientific_name", ""),
                })
        return items

    def _normalize_image(self, image: Image.Image, max_size: int = 800) -> Image.Image:
        """Chuẩn hóa ảnh: chuyển sang RGB và resize về kích thước tối đa max_size giữ nguyên tỉ lệ."""
        # Chuyển đổi mọi định dạng (RGBA, Grayscale, WebP) sang RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize giữ nguyên tỉ lệ nếu cạnh lớn hơn max_size
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    def add_herb_image(
        self,
        image_input: Union[str, Path, Image.Image],
        image_id: str,
        herb_name: str,
        scientific_name: str = "",
        image_type: str = "general",
        save_file: bool = True,
    ) -> str:
        """Chuẩn hóa kích thước, chuyển sang .jpg và lưu vector vào ChromaDB."""
        herb_dir = self.images_root / herb_name.strip().lower().replace(" ", "_")
        herb_dir.mkdir(parents=True, exist_ok=True)

        # 1. Đọc ảnh và đưa qua bộ chuẩn hóa RGB + Resize
        if isinstance(image_input, (str, Path)):
            raw_image = Image.open(image_input)
        else:
            raw_image = image_input

        processed_img = self._normalize_image(raw_image, max_size=800)

        # 2. Luôn lưu định dạng .jpg duy nhất
        saved_path = herb_dir / f"{image_id}.jpg"
        if save_file:
            processed_img.save(saved_path, format="JPEG", quality=85, optimize=True)

        # 3. Trích xuất vector cục bộ qua mô hình CLIP
        vector = local_image_embedder.embed_image(processed_img)

        # 4. Lưu metadata và vector vào ChromaDB
        self.collection.upsert(
            ids=[image_id],
            embeddings=[vector],
            metadatas=[{
                "herb_name": herb_name.strip(),
                "scientific_name": scientific_name.strip(),
                "image_type": image_type,
                "file_path": str(saved_path),
            }],
        )
        logger.info(f"✓ Đã chuẩn hóa và lưu ảnh JPEG: {saved_path}")
        return str(saved_path)

    def delete_herb_image(self, image_id: str) -> None:
        """Xóa vector ảnh khỏi ChromaDB và xóa file vật lý nếu có."""
        record = self.collection.get(ids=[image_id], include=["metadatas"])
        if record and record["metadatas"]:
            file_path = record["metadatas"][0].get("file_path")
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                except Exception as e:
                    logger.error(f"Lỗi khi xóa file vật lý: {e}")

        self.collection.delete(ids=[image_id])
        logger.info(f"✓ Đã xóa ảnh ID: {image_id}")

    def search_similar_herb(
        self,
        query_image: Union[str, Path, Image.Image],
        threshold: float = 0.70,
        top_k: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """So khớp ảnh người dùng với kho ảnh mẫu."""
        if self.total_images == 0:
            return {
                "matched": False,
                "empty_store": True,
                "similarity_score": 0.0,
                "message": "Kho dữ liệu hình ảnh hiện đang rỗng. Chưa có ảnh mẫu nào để đối chiếu.",
            }

        query_vector = local_image_embedder.embed_image(query_image)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = round(1.0 - distance, 4)
        metadata = results["metadatas"][0][0]
        matched_id = results["ids"][0][0]

        if similarity >= threshold:
            return {
                "matched": True,
                "empty_store": False,
                "herb_name": metadata["herb_name"],
                "scientific_name": metadata.get("scientific_name", ""),
                "similarity_score": similarity,
                "matched_image_id": matched_id,
                "file_path": metadata.get("file_path", ""),
            }

        return {
            "matched": False,
            "empty_store": False,
            "similarity_score": similarity,
            "message": f"Ảnh không khớp với loài cây nào trong kho ({similarity * 100:.2f}% < {threshold * 100:.0f}%).",
        }

    def get_text_herbs(self) -> List[Dict[str, str]]:
        """Quét toàn bộ file .txt trong data/raw để lấy danh sách thảo dược đã có tài liệu văn bản."""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            return []

        text_herbs = []
        for file_path in raw_dir.glob("*.txt"):
            herb_name = file_path.stem.replace("_", " ").title()
            scientific_name = ""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    # Tự động trích xuất tên tiếng Việt nếu dòng đầu có định dạng "Tên (danh pháp...)"
                    if "(" in first_line and ")" in first_line:
                        parts = first_line.split("(")
                        candidate_name = parts[0].strip()
                        if candidate_name:
                            herb_name = candidate_name
                        # Trích xuất danh pháp khoa học nếu có
                        inner = parts[1].split(")")[0]
                        if "khoa học:" in inner:
                            scientific_name = inner.split("khoa học:")[1].strip()
            except Exception as e:
                logger.warning(f"Không thể đọc metadata từ file {file_path}: {e}")

            text_herbs.append({
                "herb_name": herb_name,
                "file_name": file_path.name,
                "scientific_name": scientific_name,
            })
        return text_herbs

    def get_herbs_missing_images(self) -> List[Dict[str, str]]:
        """Tìm các loài cây đã có file văn bản nhưng CHƯA CÓ bất kỳ ảnh mẫu nào trong ChromaDB."""
        text_herbs = self.get_text_herbs()
        available_herbs_lower = {h.lower() for h in self.get_available_herbs()}

        missing = []
        for item in text_herbs:
            if item["herb_name"].lower() not in available_herbs_lower:
                missing.append(item)
        return missing

    