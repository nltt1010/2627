import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io

from src.modules.rag.ingestion import DocumentIngestion
from src.modules.rag.vectorstore import VectorStoreManager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.generator import RAGGenerator
from src.modules.vision.image_store import ImageVectorStore
from src.modules.vision.visual_rag_pipeline import VisualRAGPipeline

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Tối đa 16MB

# --- KHỞI TẠO HỆ THỐNG RAG VÀ VISION DUY NHẤT 1 LẦN ---
print(">> Đang khởi tạo hệ thống VietHerb-AI (Text RAG + Visual Vector Store)...")
ingestion = DocumentIngestion(chunk_size=300, chunk_overlap=30)
raw_docs = ingestion.load_text_files()
chunks = ingestion.split_documents(raw_docs)

vs_manager = VectorStoreManager(collection_name="vietherb_flask_collection")
vs_manager.add_documents(chunks)

hybrid_retriever = HybridRetrieverManager(
    vectorstore_manager=vs_manager,
    documents=chunks,
)
image_store = ImageVectorStore(collection_name="herb_images_collection")
text_generator = RAGGenerator(hybrid_retriever=hybrid_retriever)
image_store = ImageVectorStore(collection_name="herb_images_collection")
visual_rag = VisualRAGPipeline(
    image_store=image_store,
    text_retriever=hybrid_retriever,
    similarity_threshold=0.70,
)
print("✓ Khởi tạo hệ thống hoàn tất!")


# --- ROUTE GIAO DIỆN CHÍNH ---
@app.route("/")
def index():
    return render_template("index.html")


# --- API QUẢN TRỊ ẢNH (ADMIN) ---
@app.route("/api/herbs", methods=["GET"])
def get_herbs_summary():
    """Lấy danh sách các cây đã có ảnh và các cây trong kho text nhưng chưa có ảnh."""
    available_herbs_names = image_store.get_available_herbs()
    available_herbs = []

    for name in available_herbs_names:
        gallery = image_store.get_herb_gallery(name)
        first_img = gallery[0]["file_path"] if gallery else ""
        sci_name = gallery[0]["scientific_name"] if gallery else ""
        available_herbs.append({
            "name": name,
            "scientific_name": sci_name,
            "image_count": len(gallery),
            "thumbnail": f"/api/image/view?path={first_img}" if first_img else "",
        })

    missing_herbs = image_store.get_herbs_missing_images()
    return jsonify({
        "available_herbs": available_herbs,
        "missing_herbs": missing_herbs,
        "total_images": image_store.total_images,
    })


@app.route("/api/herb/gallery", methods=["GET"])
def get_herb_gallery():
    """Lấy chi tiết album ảnh của 1 loài cây."""
    herb_name = request.args.get("name", "").strip()
    if not herb_name:
        return jsonify({"error": "Thiếu tên cây"}), 400

    gallery = image_store.get_herb_gallery(herb_name)
    items = []
    for item in gallery:
        items.append({
            "id": item["id"],
            "image_type": item["image_type"],
            "url": f"/api/image/view?path={item['file_path']}",
        })
    return jsonify({"herb_name": herb_name, "images": items})


@app.route("/api/herb/image/upload", methods=["POST"])
def upload_herb_image():
    """Upload một hoặc nhiều ảnh mẫu cho loài cây, trích xuất vector bằng CLIP và lưu trữ."""
    herb_name = request.form.get("herb_name", "").strip()
    scientific_name = request.form.get("scientific_name", "").strip()
    image_type = request.form.get("image_type", "Toàn thân")

    # Lấy danh sách nhiều file từ input 'files' hoặc 'files[]'
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    
    if not herb_name or not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        return jsonify({"error": "Thiếu tên cây hoặc chưa chọn file ảnh!"}), 400

    saved_count = 0
    errors = []

    for idx, f in enumerate(uploaded_files):
        if not f or f.filename == "":
            continue
        try:
            pil_img = Image.open(f.stream)
            img_id = f"img_{herb_name.lower().replace(' ', '_')}_{image_store.total_images + 1}_{idx + 1}"

            image_store.add_herb_image(
                image_input=pil_img,
                image_id=img_id,
                herb_name=herb_name,
                scientific_name=scientific_name,
                image_type=image_type,
            )
            saved_count += 1
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")

    if saved_count == 0:
        return jsonify({"error": f"Không thể xử lý ảnh: {', '.join(errors)}"}), 500

    return jsonify({
        "success": True,
        "message": f"Đã lưu và vector hóa thành công {saved_count} ảnh cho {herb_name}!",
        "saved_count": saved_count,
        "errors": errors,
    })


@app.route("/api/herb/image/delete", methods=["POST"])
def delete_herb_image():
    """Xóa ảnh mẫu."""
    data = request.get_json() or {}
    image_id = data.get("image_id")
    if not image_id:
        return jsonify({"error": "Thiếu image_id"}), 400

    image_store.delete_herb_image(image_id)
    return jsonify({"success": True, "message": f"Đã xóa ảnh {image_id}"})


@app.route("/api/image/view", methods=["GET"])
def view_image():
    """Hiển thị file ảnh vật lý lên trình duyệt an toàn bằng đường dẫn tuyệt đối."""
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        return "Missing image path", 400

    # Chuyển đổi thành đường dẫn tuyệt đối bắt nguồn từ thư mục gốc ROOT_DIR
    target_path = Path(raw_path)
    if not target_path.is_absolute():
        target_path = (ROOT_DIR / target_path).resolve()

    if target_path.exists() and target_path.is_file():
        # Trả về file dựa trên đường dẫn tuyệt đối đã được xác thực
        return send_file(str(target_path), mimetype="image/jpeg")

    return f"Image not found at: {target_path}", 404


# --- API TRA CỨU RAG ---
@app.route("/api/rag/text", methods=["POST"])
@app.route("/api/rag/text", methods=["POST"])
def query_text_rag():
    """Endpoint trò chuyện: Chỉ gửi kèm ảnh khi xác định rõ ràng loài cây có trong image_store."""
    question = request.form.get("question", "").strip()
    image_file = request.files.get("image")

    if not question and not image_file:
        return jsonify({"error": "Vui lòng nhập câu hỏi hoặc gửi kèm ảnh!"}), 400

    attached_herb_image = None
    available_herbs = image_store.get_available_herbs()  # Danh sách tên chuẩn: ['Cà Gai Leo', 'Sâm Ngọc Linh', ...]

    # =========================================================================
    # TRƯỜNG HỢP 1: NGƯỜI DÙNG GỬI ẢNH ĐỂ NHẬN DIỆN
    # =========================================================================
    if image_file and image_file.filename != "":
        try:
            pil_img = Image.open(image_file.stream)
            v_res = visual_rag.process_image_query(
                image_input=pil_img, 
                user_question=question or "Nhận diện loài thảo dược này và nêu công dụng y học chính."
            )
            answer_text = v_res.get("answer", "")
            sources = v_res.get("sources", [])
            matched_herb = v_res.get("herb_name", "")
            sim_score = v_res.get("similarity_score", 0)

            # CHỈ ĐÍNH KÈM ẢNH KHI:
            # - Có loài cây khớp trong kho (matched_herb thuộc available_herbs)
            # - Độ tương đồng vector đạt ngưỡng (>= 70%)
            # - Gemini KHÔNG nói "không khớp" hoặc "không phải" loài cây này trong câu trả lời
            negative_keywords = ["không khớp", "không phải", "khác với", "nhầm lẫn", "không phù hợp"]
            is_rejected_by_gemini = any(kw in answer_text.lower() for kw in negative_keywords)

            if matched_herb in available_herbs and sim_score >= 70.0 and not is_rejected_by_gemini:
                gallery = image_store.get_herb_gallery(matched_herb)
                if gallery:
                    attached_herb_image = {
                        "herb_name": matched_herb,
                        "url": f"/api/image/view?path={gallery[0]['file_path']}",
                        "scientific_name": gallery[0].get("scientific_name", "")
                    }

            return jsonify({
                "answer": answer_text,
                "sources": sources,
                "attached_image": attached_herb_image  # Sẽ là None nếu không khớp chuẩn
            })
        except Exception as e:
            return jsonify({"error": f"Lỗi xử lý ảnh: {str(e)}"}), 500

    # =========================================================================
    # TRƯỜNG HỢP 2: NGƯỜI DÙNG HỎI DẠNG VĂN BẢN (TEXT)
    # =========================================================================
    res = text_generator.answer_question(question)
    answer_text = res.get("answer", "")
    sources = res.get("sources", [])

    # CHỈ ĐÍNH KÈM ẢNH KHI:
    # Người dùng hỏi đích danh loài cây có trong image_store (kiểm tra regex ranh giới từ để tránh khớp nhầm)
    matched_herb_in_question = None
    for herb in available_herbs:
        # Kiểm tra xem tên cây có xuất hiện trọn vẹn trong câu hỏi của người dùng không
        pattern = r'\b' + re.escape(herb.lower()) + r'\b'
        if re.search(pattern, question.lower()):
            matched_herb_in_question = herb
            break

    # Nếu người dùng hỏi đúng tên cây có trong kho ảnh -> lấy ảnh minh họa
    if matched_herb_in_question:
        gallery = image_store.get_herb_gallery(matched_herb_in_question)
        if gallery:
            attached_herb_image = {
                "herb_name": matched_herb_in_question,
                "url": f"/api/image/view?path={gallery[0]['file_path']}",
                "scientific_name": gallery[0].get("scientific_name", "")
            }

    return jsonify({
        "answer": answer_text,
        "sources": sources,
        "attached_image": attached_herb_image  # Chỉ có ảnh nếu hỏi đích danh cây trong kho
    })

@app.route("/api/rag/vision", methods=["POST"])
def query_vision_rag():
    """Nhận diện qua ảnh (Visual RAG)."""
    if "file" not in request.files:
        return jsonify({"error": "Thiếu file ảnh"}), 400

    file = request.files["file"]
    custom_question = request.form.get("question", "")
    pil_img = Image.open(file.stream)

    res = visual_rag.process_image_query(image_input=pil_img, user_question=custom_question)
    return jsonify(res)

@app.route("/api/rag/text/upload", methods=["POST"])
def upload_text_document():
    """Tải lên file tài liệu văn bản (.txt) mới và re-index lại kho tri thức RAG."""
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        return jsonify({"error": "Vui lòng chọn ít nhất 1 file .txt"}), 400

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for f in uploaded_files:
        if f and f.filename.endswith(".txt"):
            save_path = raw_dir / secure_filename(f.filename)
            f.save(save_path)
            saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "Chỉ chấp nhận các file định dạng .txt!"}), 400

    # Tự động Re-index kho tài liệu vào VectorStore và HybridRetriever
    global chunks, vs_manager, hybrid_retriever, text_generator, visual_rag
    try:
        raw_docs = ingestion.load_text_files()
        chunks = ingestion.split_documents(raw_docs)
        vs_manager.add_documents(chunks)
        hybrid_retriever = HybridRetrieverManager(
            vectorstore_manager=vs_manager,
            documents=chunks,
        )
        text_generator = RAGGenerator(
            hybrid_retriever=hybrid_retriever,
            image_store=image_store
        )
        visual_rag.text_retriever = hybrid_retriever
    except Exception as e:
        return jsonify({"error": f"Lỗi re-index tài liệu: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": f"Đã lưu thành công {saved_count} file tài liệu và đồng bộ hóa lại tri thức RAG!",
        "saved_count": saved_count,
    })

import re

def slugify_vietnamese(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt thành tên file an toàn (vd: 'Đinh Lăng' -> 'dinh_lang')."""
    text = text.lower().strip()
    # Chuyển các ký tự có dấu phổ biến
    patterns = {
        '[àáạảãâầấậẩẫăằắặẳẵ]': 'a',
        '[èéẹẻẽêềếệểễ]': 'e',
        '[ìíịỉĩ]': 'i',
        '[òóọỏõôồốộổỗơờớợởỡ]': 'o',
        '[ùúụủũưừứựửữ]': 'u',
        '[ỳýỵỷỹ]': 'y',
        '[đ]': 'd',
    }
    for regex, replace in patterns.items():
        text = re.sub(regex, replace, text)
    # Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ, số và khoảng trắng
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text or "thao_duoc"

@app.route("/api/rag/text/create", methods=["POST"])
def create_text_document():
    """Tạo trực tiếp file .txt từ văn bản nhập trên giao diện và re-index lại kho tri thức."""
    data = request.get_json() or {}
    herb_name = data.get("herb_name", "").strip()
    scientific_name = data.get("scientific_name", "").strip()
    content = data.get("content", "").strip()

    if not herb_name or not content:
        return jsonify({"error": "Vui lòng nhập tên thảo dược và nội dung mô tả!"}), 400

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Đặt tên file tự động dựa theo tên cây (vd: dinh_lang.txt)
    base_slug = slugify_vietnamese(herb_name)
    file_name = f"{base_slug}.txt"
    file_path = raw_dir / file_name

    # Chuẩn hóa định dạng chuẩn để hàm quét dữ liệu tự bóc tách được danh pháp
    # Dòng 1: Tên (danh pháp khoa học: ...)
    full_text = f"{herb_name}"
    if scientific_name:
        full_text += f" (danh pháp khoa học: {scientific_name})"
    full_text += f"\n\n{content}\n"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)
    except Exception as e:
        return jsonify({"error": f"Lỗi ghi file: {str(e)}"}), 500

    # Tự động Re-index kho tài liệu vào VectorStore và HybridRetriever
    global chunks, vs_manager, hybrid_retriever, text_generator, visual_rag
    try:
        raw_docs = ingestion.load_text_files()
        chunks = ingestion.split_documents(raw_docs)
        vs_manager.add_documents(chunks)
        hybrid_retriever = HybridRetrieverManager(
            vectorstore_manager=vs_manager,
            documents=chunks,
        )
        text_generator = RAGGenerator(
            hybrid_retriever=hybrid_retriever,
            image_store=image_store
        )
        visual_rag.text_retriever = hybrid_retriever
    except Exception as e:
        return jsonify({"error": f"Lỗi re-index tài liệu: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": f"Đã tạo file {file_name} và cập nhật tri thức RAG thành công!",
        "file_name": file_name,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)