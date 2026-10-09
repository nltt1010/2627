import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import google.generativeai as genai
from src.core.key_manager import key_manager

# Đảm bảo có key
key = key_manager._raw_keys[0]
genai.configure(api_key=key)

print("--- DANH SÁCH MÔ HÌNH HỖ TRỢ EMBEDDING (embedContent) ---")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f"Name: {m.name} | Methods: {m.supported_generation_methods}")