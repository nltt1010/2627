import re
from functools import lru_cache
from typing import List
from dotenv import dotenv_values
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
        keys: List[str] = []

       
        if self.GEMINI_API_KEYS_RAW:
            for k in self.GEMINI_API_KEYS_RAW.split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)

        
        env_dict = dotenv_values(".env")
        pattern = re.compile(r"^GEMINI_API_KEYS?_?\d+$", re.IGNORECASE)

        for env_name, env_val in env_dict.items():
            if env_val and pattern.match(env_name):
                val_clean = env_val.strip()
                if val_clean and val_clean not in keys:
                    keys.append(val_clean)

        return keys


@lru_cache()
def get_settings() -> Settings:
    return Settings()