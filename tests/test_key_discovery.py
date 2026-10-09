import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.key_manager import key_manager
from src.core.config import get_settings


def test_discovery():
    settings = get_settings()
    # 1. Quét toàn bộ model của các key
    key_manager.discover_keys_capabilities()

    # 2. Kiểm tra lấy key cho Embedding
    try:
        embed_key = key_manager.get_key_for_embedding(settings.DEFAULT_EMBEDDING_MODEL)
        print(f"\n✓ Key được chọn cho Embedding ({settings.DEFAULT_EMBEDDING_MODEL}): {embed_key[:6]}******")
    except Exception as e:
        print(f"\n✗ Lỗi tìm key cho Embedding: {e}")

    # 3. Kiểm tra lấy key cho Chat LLM
    try:
        chat_key = key_manager.get_key_for_chat(settings.DEFAULT_LLM_MODEL)
        print(f"✓ Key được chọn cho Chat ({settings.DEFAULT_LLM_MODEL}): {chat_key[:6]}******")
    except Exception as e:
        print(f"✗ Lỗi tìm key cho Chat: {e}")


if __name__ == "__main__":
    test_discovery()