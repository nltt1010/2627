import sys
from pathlib import Path

# Thêm đường dẫn gốc dự án vào PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.generator import RAGGenerator

# Thiết lập cấu hình trang giao diện
st.set_page_config(
    page_title="VietHerb-AI | Trợ lý Dược Liệu Việt Nam",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 VietHerb-AI")
st.caption("Hệ thống RAG tra cứu tri thức Dược liệu & Thảo dược Việt Nam (Chống ảo giác y khoa)")


@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Khởi tạo và nạp pipeline RAG duy nhất một lần để tối ưu RAM."""
    # 1. Ingestion
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    # 2. VectorStore
    vs_manager = VectorStoreManager(collection_name="vietherb_app_collection")
    vs_manager.add_documents(chunks)

    # 3. Hybrid Retriever
    hybrid_retriever = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks,
    )

    # 4. Generator
    generator = RAGGenerator(hybrid_retriever=hybrid_retriever)
    return generator


# Nạp hệ sinh thái RAG với Spinner hiển thị trạng thái
with st.spinner("Đang khởi tạo chỉ mục tri thức và kết nối mô hình..."):
    rag_engine = initialize_rag_system()

# Quản lý lịch sử tin nhắn qua Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là VietHerb-AI. Bạn cần tra cứu thông tin gì về dược tính, phân bố hay kỹ thuật nuôi trồng thảo dược?",
            "sources": [],
        }
    ]

# Hiển thị các tin nhắn trước đó
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Nguồn tài liệu tham khảo"):
                for src in msg["sources"]:
                    st.write(f"- `{src}`")

# Ô nhập câu hỏi của người dùng
user_input = st.chat_input("Nhập câu hỏi (ví dụ: Sâm Ngọc Linh có công dụng gì?)...")

if user_input:
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Sinh câu trả lời qua RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang trích xuất tri thức và tổng hợp..."):
            result = rag_engine.answer_question(user_input)
            answer_text = result["answer"]
            sources_list = result["sources"]

            st.markdown(answer_text)
            if sources_list:
                with st.expander("📚 Nguồn tài liệu tham khảo"):
                    for src in sources_list:
                        st.write(f"- `{src}`")

    # 3. Lưu vào lịch sử phiên
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_list,
    })