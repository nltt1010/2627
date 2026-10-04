import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.key_manager import APIKeyManager, key_manager


def test_key_rotation_mock():
    """Kiểm tra cơ chế xoay vòng với danh sách key giả lập."""
    mock_keys = ["KEY_A", "KEY_B", "KEY_C"]
    manager = APIKeyManager(keys=mock_keys)

    assert manager.get_current_key() == "KEY_A"
    assert manager.rotate_key() == "KEY_B"
    assert manager.rotate_key() == "KEY_C"
    # Quay trở lại đầu danh sách
    assert manager.rotate_key() == "KEY_A"
    print("✓ Test xoay vòng giả lập: THÀNH CÔNG")


def test_real_key_loaded():
    """Kiểm tra key thật từ file .env."""
    total = key_manager.total_keys
    print(f"✓ Số lượng key thực tế nạp được từ .env: {total}")
    if total > 0:
        print(f"✓ Key hiện tại sẵn sàng: {key_manager.get_current_key()[:6]}******")


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE KEY MANAGER ---")
    test_key_rotation_mock()
    test_real_key_loaded()