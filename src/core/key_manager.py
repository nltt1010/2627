import logging
from typing import List, Optional
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Quản lý danh sách API Key và tự động xoay vòng khi chạm ngưỡng hạn ngạch."""

    def __init__(self, keys: Optional[List[str]] = None):
        settings = get_settings()
        self._keys: List[str] = keys if keys is not None else settings.gemini_api_keys
        self._current_index: int = 0

        if not self._keys:
            logger.warning("Không tìm thấy GEMINI_API_KEYS trong cấu hình!")

    @property
    def total_keys(self) -> int:
        """Tổng số key khả dụng."""
        return len(self._keys)

    def get_current_key(self) -> str:
        """Lấy key đang được kích hoạt hiện tại."""
        if not self._keys:
            raise ValueError("Danh sách API Key rỗng. Hãy kiểm tra file .env!")
        return self._keys[self._current_index]

    def rotate_key(self) -> str:
        """Chuyển sang key kế tiếp theo cơ chế xoay vòng."""
        if not self._keys:
            raise ValueError("Không có API Key nào để xoay vòng!")

        old_index = self._current_index
        self._current_index = (self._current_index + 1) % len(self._keys)
        logger.info(
            f"Đã xoay vòng API Key từ vị trí {old_index} sang {self._current_index}"
        )
        return self.get_current_key()


# Khởi tạo instance mặc định để dùng chung trong toàn bộ hệ thống
key_manager = APIKeyManager()