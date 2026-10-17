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
def query_text_rag():
    """Tra cứu văn bản chuyên sâu."""
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Thiếu câu hỏi"}), 400

    res = text_generator.answer_question(question)
    return jsonify(res)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)