from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "VietHerb-AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    APP_PORT: int = 8000

    GEMINI_API_KEYS_RAW: str = Field(default="", alias="GEMINI_API_KEYS")


    CHROMA_PERSIST_DIR: str = "chroma_db"
    RAW_DATA_DIR: str = "data/raw"

    DEFAULT_LLM_MODEL: str = "gemini-2.5-flash"
    DEFAULT_EMBEDDING_MODEL: str = "models/text-embedding-004"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def gemini_api_keys(self) -> List[str]:
        """Tách chuỗi các API key phân tách bởi dấu phẩy thành danh sách."""
        if not self.GEMINI_API_KEYS_RAW:
            return []
        return [key.strip() for key in self.GEMINI_API_KEYS_RAW.split(",") if key.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()