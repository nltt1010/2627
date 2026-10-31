import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from PIL import Image

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.generator import RAGGenerator
from src.modules.vision.image_store import ImageVectorStore
from src.modules.vision.visual_rag_pipeline import VisualRAGPipeline

st.set_page_config(
    page_title="VietHerb-AI | Quản trị Dược Liệu & Visual RAG",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 VietHerb-AI")
st.caption("Hệ thống RAG Multimodal & Quản lý Kho Tri thức Dược liệu Việt Nam")


@st.cache_resource(show_spinner=False)
def load_vietherb_system():
    ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
    raw_docs = ingestion.load_text_files()
    chunks = ingestion.split_documents(raw_docs)

    vs_manager = VectorStoreManager(collection_name="vietherb_app_collection")
    vs_manager.add_documents(chunks)

    hybrid_retriever = HybridRetrieverManager(
        vectorstore_manager=vs_manager,
        documents=chunks,
    )

    generator = RAGGenerator(hybrid_retriever=hybrid_retriever)
    image_store = ImageVectorStore(collection_name="herb_images_collection")
    visual_pipeline = VisualRAGPipeline(
        image_store=image_store,
        text_retriever=hybrid_retriever,
        similarity_threshold=0.70,
    )

    return generator, image_store, visual_pipeline


with st.spinner("Đang khởi tạo hệ sinh thái tri thức và các bộ chỉ mục vector..."):
    text_generator, image_store, visual_rag = load_vietherb_system()

# Quản lý trạng thái cây đang được chọn để xem chi tiết ảnh
if "selected_herb_admin" not in st.session_state:
    st.session_state.selected_herb_admin = None

# --- SIDEBAR: CẬP NHẬT TÀI LIỆU VĂN BẢN NHANH ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    st.caption(f"📊 **Kho ảnh:** {image_store.total_images} vectors")
    st.caption(f"🌿 **Số loài có ảnh:** {len(image_store.get_available_herbs())}")
    
    st.divider()
    st.subheader("Nạp tài liệu văn bản (.txt)")
    uploaded_text_files = st.file_uploader(
        "Tải lên file văn bản mô tả thảo dược",
        type=["txt"],
        accept_multiple_files=True,
    )
    if uploaded_text_files and st.button("Lưu & Re-index văn bản", use_container_width=True):
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_text_files:
            with open(raw_dir / uf.name, "wb") as f:
                f.write(uf.getbuffer())
        st.cache_resource.clear()
        st.success("Đã cập nhật kho văn bản!")
        st.rerun()

# --- GIAO DIỆN CHÍNH: 3 PHÂN HỆ ---
tab_gallery, tab_vision, tab_text = st.tabs([
    "🖼️ Quản lý Thư viện Ảnh (Admin)",
    "📷 Nhận diện qua ảnh (Visual RAG)",
    "💬 Tra cứu văn bản chuyên sâu",
])

# ==============================================================================
# TAB 1: THƯ VIỆN ẢNH ADMIN (HORIZONTAL CARDS & DETAIL VIEW)
# ==============================================================================
with tab_gallery:
    st.subheader("Bộ Sưu Tập Thảo Dược Đã Lưu Trữ")
    available_herbs = image_store.get_available_herbs()
    missing_herbs = image_store.get_herbs_missing_images()

    # --- KHU VỰC 1: CẢNH BÁO CÁC CÂY CÓ TEXT NHƯNG THIẾU ẢNH ---
    if missing_herbs:
        with st.container(border=True):
            st.markdown("##### ⚠️ Thảo dược đã có tài liệu nhưng CHƯA CÓ ảnh mẫu:")
            st.caption("Các loài cây này đã có trong kho RAG văn bản, nhưng tính năng nhận diện ảnh chưa hỗ trợ vì thiếu vector đối chiếu.")

            m_cols = st.columns(max(len(missing_herbs), 3))
            for m_idx, m_item in enumerate(missing_herbs):
                m_col = m_cols[m_idx % 3]
                with m_col:
                    st.warning(f"📄 **{m_item['herb_name']}**")
                    if m_item['scientific_name']:
                        st.caption(f"*{m_item['scientific_name']}*")
                    st.caption(f"Nguồn: `{m_item['file_name']}`")
                    
                    # Nút chọn để bổ sung ảnh ngay cho cây này
                    if st.button("➕ Bổ sung ảnh mẫu", key=f"btn_add_img_{m_item['herb_name']}", use_container_width=True):
                        st.session_state.selected_herb_admin = m_item['herb_name']
                        st.session_state.target_scientific_name = m_item['scientific_name']
                        st.rerun()

    st.markdown("---")

    # --- KHU VỰC 2: CÁC CÂY ĐÃ CÓ ẢNH ĐẦY ĐỦ (HIỂN THỊ DẠNG THẺ NGANG) ---
    st.markdown("##### 🌿 Thảo dược đã có ảnh mẫu trong kho:")
    if not available_herbs:
        st.info("Hiện chưa có loài cây nào có ảnh mẫu. Hãy chọn một cây ở danh sách cảnh báo phía trên để nạp ảnh!")
    else:
        cols = st.columns(max(min(len(available_herbs), 4), 1))
        for idx, herb_name in enumerate(available_herbs):
            col = cols[idx % 4]
            gallery = image_store.get_herb_gallery(herb_name)
            first_img_path = gallery[0]["file_path"] if gallery else None

            with col:
                with st.container(border=True):
                    if first_img_path and Path(first_img_path).exists():
                        st.image(first_img_path, use_container_width=True)
                    else:
                        st.image("https://placehold.co/300x200?text=Chua+co+anh", use_container_width=True)

                    st.markdown(f"**{herb_name}**")
                    sci_name = gallery[0]["scientific_name"] if gallery and gallery[0]["scientific_name"] else "Chưa rõ"
                    st.caption(f"*{sci_name}* • ({len(gallery)} ảnh)")

                    if st.button("Xem bộ ảnh ➔", key=f"btn_view_{herb_name}", use_container_width=True):
                        st.session_state.selected_herb_admin = herb_name
                        st.session_state.target_scientific_name = sci_name
                        st.rerun()

    st.divider()

    # --- KHU VỰC 3: CHI TIẾT ALBUM / FORM BỔ SUNG ẢNH VÀO CÂY ĐƯỢC CHỌN ---
    if st.session_state.selected_herb_admin:
        cur_herb = st.session_state.selected_herb_admin
        cur_sci = getattr(st.session_state, "target_scientific_name", "")
        gallery = image_store.get_herb_gallery(cur_herb)

        st.subheader(f"🔍 Quản lý ảnh của: **{cur_herb}** ({len(gallery)} ảnh mẫu hiện có)")
        if st.button("✕ Đóng / Quay lại", type="secondary"):
            st.session_state.selected_herb_admin = None
            st.rerun()

        # Hiển thị các ảnh đã có nếu có
        if gallery:
            img_cols = st.columns(4)
            for i, item in enumerate(gallery):
                c = img_cols[i % 4]
                with c:
                    with st.container(border=True):
                        f_path = item["file_path"]
                        if Path(f_path).exists():
                            st.image(f_path, use_container_width=True)
                        else:
                            st.warning("File vật lý bị thiếu")
                        st.caption(f"ID: `{item['id']}`")
                        st.caption(f"Phần: *{item['image_type']}*")
                        if st.button("🗑️ Xóa ảnh", key=f"del_{item['id']}", use_container_width=True):
                            image_store.delete_herb_image(item["id"])
                            st.cache_resource.clear()
                            st.success(f"Đã xóa ảnh {item['id']}")
                            st.rerun()

        # Form nạp ảnh (Bổ sung vào cây có sẵn)
        st.markdown(f"##### 📥 Nạp thêm ảnh mẫu cho **{cur_herb}**:")
        with st.form(f"add_more_img_{cur_herb}", clear_on_submit=True):
            f_upload = st.file_uploader("Chọn file ảnh (lá, thân, củ, hoa)", type=["jpg", "jpeg", "png"])
            f_type = st.selectbox("Bộ phận hiển thị trong ảnh", ["Toàn thân", "Lá cây", "Củ / Rễ", "Hoa / Quả", "Khác"])
            sub_btn = st.form_submit_button("Lưu ảnh vào cây này", use_container_width=True)

            if sub_btn and f_upload:
                img_id = f"img_{cur_herb.lower().replace(' ', '_')}_{image_store.total_images + 1}"
                image_store.add_herb_image(
                    image_input=Image.open(f_upload),
                    image_id=img_id,
                    herb_name=cur_herb,
                    scientific_name=cur_sci,
                    image_type=f_type,
                )
                st.cache_resource.clear()
                st.success(f"✓ Đã bổ sung ảnh mẫu thành công cho cây {cur_herb}!")
                st.rerun()

    else:
        # Form thêm mới một loài cây hoàn toàn chưa có trong cả kho text lẫn kho ảnh
        with st.expander("✨ Thêm mới một loài thảo dược chưa có trong kho"):
            with st.form("create_brand_new_herb_form", clear_on_submit=True):
                new_h_name = st.text_input("Tên thảo dược mới")
                new_sci_name = st.text_input("Tên khoa học")
                new_img = st.file_uploader("Ảnh mẫu đầu tiên", type=["jpg", "jpeg", "png"])
                new_type = st.selectbox("Bộ phận", ["Toàn thân", "Lá cây", "Củ / Rễ", "Hoa / Quả"])
                btn_new = st.form_submit_button("Khởi tạo loài cây mới", use_container_width=True)

                if btn_new:
                    if not new_h_name.strip() or not new_img:
                        st.error("Vui lòng điền tên cây và chọn ảnh!")
                    else:
                        img_id = f"img_{new_h_name.strip().lower().replace(' ', '_')}_1"
                        image_store.add_herb_image(
                            image_input=Image.open(new_img),
                            image_id=img_id,
                            herb_name=new_h_name.strip(),
                            scientific_name=new_sci_name.strip(),
                            image_type=new_type,
                        )
                        st.cache_resource.clear()
                        st.success(f"✓ Đã thêm mới thành công cây: {new_h_name.strip()}!")
                        st.rerun()
# ==============================================================================
# TAB 2: VISUAL RAG TRA CỨU
# ==============================================================================
with tab_vision:
    st.subheader("Nhận diện thảo dược qua ảnh & Đối chiếu tri thức")
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown("##### 1. Tải ảnh thảo dược cần kiểm tra")
        query_file = st.file_uploader(
            "Tải ảnh lá, thân, hoa hoặc củ cây",
            type=["jpg", "jpeg", "png"],
            key="user_query_img",
        )
        custom_question = st.text_input(
            "Câu hỏi bổ sung (Tùy chọn)",
            placeholder="Ví dụ: Cây này hỗ trợ bệnh gì và dùng như thế nào?",
        )
        run_vision_btn = st.button("🔍 Tiến hành phân tích & Tra cứu", type="primary", use_container_width=True)

        if query_file:
            st.image(query_file, caption="Ảnh bạn cung cấp", use_container_width=True)

    with col_right:
        st.markdown("##### 2. Kết quả phân tích từ VietHerb-AI")
        if query_file and run_vision_btn:
            with st.spinner("Đang trích xuất vector ảnh cục bộ (0 API call) và đối chiếu..."):
                pil_query = Image.open(query_file)
                result = visual_rag.process_image_query(
                    image_input=pil_query,
                    user_question=custom_question,
                )

                if result.get("matched"):
                    st.success(
                        f"**Nhận diện thành công:** {result['herb_name']} "
                        f"(*{result.get('scientific_name', '')}*) — Độ khớp: **{result['similarity_score']}%**"
                    )
                    st.markdown("---")
                    st.markdown(result["answer"])
                    if result.get("sources"):
                        with st.expander("📚 Tài liệu chuẩn được trích xuất"):
                            for s in result["sources"]:
                                st.write(f"- `{s}`")
                else:
                    st.warning(result["answer"])

# ==============================================================================
# TAB 3: TRA CỨU VĂN BẢN CHUYÊN SÂU
# ==============================================================================
with tab_text:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Xin chào! Tôi là VietHerb-AI. Bạn có thể hỏi về liều dùng, dược tính hoặc chuyển sang tab ảnh để nhận diện thảo dược.",
                "sources": [],
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Nguồn tài liệu tham khảo"):
                    for src in msg["sources"]:
                        st.write(f"- `{src}`")

    if user_prompt := st.chat_input("Nhập câu hỏi thảo dược..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm tri thức và tổng hợp..."):
                res = text_generator.answer_question(user_prompt)
                st.markdown(res["answer"])
                if res["sources"]:
                    with st.expander("📚 Nguồn tài liệu tham khảo"):
                        for src in res["sources"]:
                            st.write(f"- `{src}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "sources": res["sources"],
        })