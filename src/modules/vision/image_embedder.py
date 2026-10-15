import logging
from pathlib import Path
from typing import List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalImageEmbedder:
    """Trích xuất vector đặc trưng hình ảnh cục bộ (CLIP) không dùng API, tối ưu RAM."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model: Chỉ nạp vào RAM khi thực sự có thao tác embedding ảnh."""
        if self._model is None:
            logger.info(f"Đang nạp mô hình trích xuất vector ảnh cục bộ ({self.model_name})...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_image(self, image_input: Union[str, Path, Image.Image]) -> List[float]:
        """Chuyển đổi 1 bức ảnh thành mảng vector số thực."""
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")

        # Encode ảnh thành vector (512 chiều)
        embedding = self.model.encode(pil_image)
        return embedding.tolist()


# Instance dùng chung
local_image_embedder = LocalImageEmbedder()