from src.core.config import get_settings

def test_load_settings():
    settings = get_settings()
    print("--- KIỂM TRA CẤU HÌNH THÀNH CÔNG ---")
    print(f"Tên ứng dụng: {settings.APP_NAME}")
    print(f"Mô hình LLM mặc định: {settings.DEFAULT_LLM_MODEL}")
    print(f"Số lượng Gemini API Key nạp được: {len(settings.gemini_api_keys)}")

if __name__ == "__main__":
    test_load_settings()