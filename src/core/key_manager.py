import logging
from typing import Dict, List, Optional, Set
import google.generativeai as genai
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class APIKeyInfo:
    """Lưu trữ thông tin và danh sách các mô hình mà 1 API Key hỗ trợ."""

    def __init__(self, key: str):
        self.key: str = key
        self.is_valid: bool = False
        self.supported_models: Set[str] = set()
        self.supports_embedding: bool = False
        self.supports_chat: bool = False


class APIKeyManager:
    """Quản lý danh sách API Key, tự động quét mô hình khả dụng và xoay vòng theo năng lực."""

    def __init__(self, keys: Optional[List[str]] = None):
        settings = get_settings()
        self._raw_keys: List[str] = (
            keys if keys is not None else settings.gemini_api_keys
        )
        self._key_registry: Dict[str, APIKeyInfo] = {}

        # Con trỏ chỉ mục xoay vòng riêng cho từng tác vụ
        self._chat_cursor: int = 0
        self._embed_cursor: int = 0

    def discover_keys_capabilities(self) -> None:
        """Quét toàn bộ danh sách key để kiểm tra tính hợp lệ và danh sách model được hỗ trợ."""
        if not self._raw_keys:
            logger.warning("Không có API Key nào được khai báo trong cấu hình!")
            return

        print("\n--- ĐANG QUÉT DANH SÁCH MÔ HÌNH CỦA CÁC API KEY ---")
        for idx, key in enumerate(self._raw_keys, 1):
            info = APIKeyInfo(key=key)
            masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "******"
            try:
                genai.configure(api_key=key)
                models = genai.list_models()

                for m in models:
                    model_name = m.name  # Dạng "models/gemini-..." hoặc "models/text-embedding-..."
                    info.supported_models.add(model_name)

                    methods = m.supported_generation_methods
                    if "embedContent" in methods:
                        info.supports_embedding = True
                    if "generateContent" in methods:
                        info.supports_chat = True

                info.is_valid = True
                print(
                    f"✓ Key [{idx}] ({masked_key}): HỢP LỆ | "
                    f"Hỗ trợ {len(info.supported_models)} models | "
                    f"Chat: {'Có' if info.supports_chat else 'Không'} | "
                    f"Embedding: {'Có' if info.supports_embedding else 'Không'}"
                )
            except Exception as e:
                info.is_valid = False
                print(f"✗ Key [{idx}] ({masked_key}): LỖI / VÔ HIỆU HÓA ({e})")

            self._key_registry[key] = info

    def get_valid_keys_for_model(self, model_name: str) -> List[str]:
        """Lấy danh sách các key hỗ trợ cụ thể model_name."""
        # Chuẩn hóa để so khớp dù chuỗi có hoặc không có tiền tố "models/"
        target = model_name if model_name.startswith("models/") else f"models/{model_name}"

        valid_keys = [
            k
            for k, info in self._key_registry.items()
            if info.is_valid and (target in info.supported_models or model_name in info.supported_models)
        ]
        return valid_keys

    def get_key_for_embedding(self, model_name: Optional[str] = None) -> str:
        """Lấy key khả dụng cho tác vụ embedding theo thứ tự xoay vòng."""
        settings = get_settings()
        target_model = model_name or settings.DEFAULT_EMBEDDING_MODEL

        eligible_keys = self.get_valid_keys_for_model(target_model)
        if not eligible_keys:
            # Fallback lấy key nào có cờ supports_embedding
            eligible_keys = [
                k for k, info in self._key_registry.items() if info.supports_embedding
            ]

        if not eligible_keys:
            raise ValueError(
                f"Không có API Key nào hỗ trợ mô hình embedding: {target_model}!"
            )

        self._embed_cursor = (self._embed_cursor + 1) % len(eligible_keys)
        return eligible_keys[self._embed_cursor]

    def get_key_for_chat(self, model_name: Optional[str] = None) -> str:
        """Lấy key khả dụng cho tác vụ Chat LLM theo thứ tự xoay vòng."""
        settings = get_settings()
        target_model = model_name or settings.DEFAULT_LLM_MODEL

        eligible_keys = self.get_valid_keys_for_model(target_model)
        if not eligible_keys:
            eligible_keys = [
                k for k, info in self._key_registry.items() if info.supports_chat
            ]

        if not eligible_keys:
            raise ValueError(
                f"Không có API Key nào hỗ trợ mô hình chat: {target_model}!"
            )

        self._chat_cursor = (self._chat_cursor + 1) % len(eligible_keys)
        return eligible_keys[self._chat_cursor]


key_manager = APIKeyManager()