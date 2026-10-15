import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import chromadb
from PIL import Image

from src.core.config import get_settings
from src.modules.vision.image_embedder import local_image_embedder

logger = logging.getLogger(__name__)


class ImageVectorStore:
    """Quản lý kho vector ảnh mẫu thảo dược trên ChromaDB có kiểm soát tình trạng dữ liệu."""

    def __init__(self, collection_name: str = "herb_images_collection"):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = self.settings.CHROMA_PERSIST_DIR

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def total_images(self) -> int:
        """Tổng số lượng vector ảnh mẫu hiện có trong kho."""
        return self.collection.count()

    def get_available_herbs(self) -> List[str]:
        """Lấy danh sách tên các loài cây đã có ảnh mẫu trong kho."""
        if self.total_images == 0:
            return []
        
        all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
        herbs: Set[str] = {m["herb_name"] for m in all_metadata if m and "herb_name" in m}
        return sorted(list(herbs))

    def has_herb_image(self, herb_name: str) -> bool:
        """Kiểm tra một loài cây cụ thể đã được nạp dữ liệu ảnh mẫu hay chưa."""
        available = self.get_available_herbs()
        return any(herb_name.lower() == h.lower() for h in available)

    def add_herb_image(
        self,
        image_input: Union[str, Path, Image.Image],
        image_id: str,
        herb_name: str,
        scientific_name: str = "",
        image_type: str = "general",
    ) -> None:
        """Đưa một ảnh mẫu vào kho vector kèm metadata định danh."""
        vector = local_image_embedder.embed_image(image_input)
        
        self.collection.upsert(
            ids=[image_id],
            embeddings=[vector],
            metadatas=[{
                "herb_name": herb_name,
                "scientific_name": scientific_name,
                "image_type": image_type,
            }],
        )
        logger.info(f"✓ Đã lưu vector ảnh mẫu [{image_id}] cho cây: {herb_name}")

    def search_similar_herb(
        self,
        query_image: Union[str, Path, Image.Image],
        threshold: float = 0.70,
        top_k: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        So khớp ảnh người dùng với kho ảnh mẫu.
        Có kiểm tra điều kiện dữ liệu rỗng.
        """
        # Kiểm tra điều kiện: Kho ảnh đã có dữ liệu hay chưa
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
            return {
                "matched": False,
                "empty_store": False,
                "similarity_score": 0.0,
                "message": "Không tìm thấy mẫu ảnh phù hợp trong kho.",
            }

        distance = results["distances"][0][0]
        similarity = round(1.0 - distance, 4)
        metadata = results["metadatas"][0][0]
        matched_id = results["ids"][0][0]

        logger.info(f"Kết quả so khớp ảnh: {metadata['herb_name']} (Độ khớp: {similarity * 100:.2f}%)")

        if similarity >= threshold:
            return {
                "matched": True,
                "empty_store": False,
                "herb_name": metadata["herb_name"],
                "scientific_name": metadata.get("scientific_name", ""),
                "similarity_score": similarity,
                "matched_image_id": matched_id,
            }

        return {
            "matched": False,
            "empty_store": False,
            "similarity_score": similarity,
            "message": f"Ảnh không khớp với loài cây nào trong kho (Độ tương đồng cao nhất: {similarity * 100:.2f}% < ngưỡng {threshold * 100:.0f}%).",
        }