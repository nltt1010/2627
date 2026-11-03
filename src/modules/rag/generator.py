import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.core.config import get_settings
from src.core.key_manager import key_manager
from src.modules.rag.hybrid_retriever import HybridRetrieverManager
from src.modules.rag.evaluator import RAGTriadEvaluator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về dược liệu và nông nghiệp Việt Nam (VietHerb-AI).

--- THỐNG KÊ KHO DỮ LIỆU THỰC TẾ CỦA HỆ THỐNG ---
- Tổng số loài thảo dược hiện có: {total_count} loài.
- Danh sách cụ thể các loài có trong hệ thống: {herb_list}.
* QUY TẮC BẮT BUỘC: Nếu người dùng hỏi về số lượng ("có bao nhiêu loài", "có mấy cây", "liệt kê danh sách các cây"), bạn PHẢI dựa trực tiếp vào mục thống kê trên để trả lời chính xác con số {total_count} và kể tên các loài cây tương ứng.

--- QUY TẮC PHẢN HỒI Y KHOA / DƯỢC TÍNH (GRAPHRAG) ---
1. Kết hợp thông tin từ 'MẠNG LƯỚI QUAN HỆ ĐỒ THỊ TRI THỨC (GRAPH CONTEXT)' và 'NGỮ CẢNH TÀI LIỆU TRÍCH XUẤT (TEXT CHUNKS)'. Tuyệt đối không tự bịa đặt hoặc suy diễn các công dụng y học/liều lượng không có trong cơ sở dữ liệu.
2. Nếu câu hỏi yêu cầu so sánh, tìm cây cùng chữa bệnh, tìm hoạt chất hoặc tìm mối quan hệ liên kết bắc cầu giữa các loài cây, hãy tận dụng triệt để các quan hệ đã được kiểm chứng trong Đồ thị tri thức.
3. Nếu cả ngữ cảnh tài liệu và đồ thị đều không đủ thông tin, hãy thẳng thắn thông báo: "Tài liệu hiện tại chưa có thông tin về vấn đề này."
4. Cuối câu trả lời, luôn liệt kê rõ ràng các nguồn tài liệu đã tham khảo (Metadata: source).
5. Giữ giọng văn khách quan, khoa học và súc tích. Định dạng rõ ràng bằng Markdown.

--- MẠNG LƯỚI QUAN HỆ ĐỒ THỊ TRI THỨC (GRAPH CONTEXT) ---
{graph_context}

--- NGỮ CẢNH TÀI LIỆU TRÍCH XUẤT (TEXT CHUNKS) ---
{context}
"""


class RAGGenerator:
    """Xử lý kết hợp ngữ cảnh từ Advanced Hybrid Retriever + Knowledge Graph (GraphRAG) và sinh câu trả lời bằng Gemini LLM."""

    def __init__(
        self,
        hybrid_retriever: HybridRetrieverManager,
        image_store: Optional[Any] = None,
        evaluator: Optional[RAGTriadEvaluator] = None,
        graph_manager: Optional[Any] = None,
    ):
        self.settings = get_settings()
        self.retriever = hybrid_retriever
        self.image_store = image_store
        self.evaluator = evaluator or RAGTriadEvaluator()
        self.graph_manager = graph_manager

        model_name = getattr(self.settings, "DEFAULT_LLM_MODEL", "gemini-2.5-flash")
        selected_key = key_manager.get_key_for_chat(model_name)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=selected_key,
            temperature=0.2,
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self.output_parser = StrOutputParser()

    def _get_live_database_summary(self) -> Dict[str, Any]:
        """Quét và tổng hợp chính xác dữ liệu hiện hữu từ kho raw và vector store."""
        raw_dir = Path("data/raw")
        herbs_in_system = []

        if raw_dir.exists():
            for f in raw_dir.glob("*.txt"):
                herb_name = f.stem.replace("_", " ").title()
                herbs_in_system.append(herb_name)

        image_herbs = self.image_store.get_available_herbs() if self.image_store else []
        all_unique_herbs = sorted(list(set(herbs_in_system + image_herbs)))

        return {
            "total_count": len(all_unique_herbs),
            "herb_list": all_unique_herbs,
        }

    def _format_context(self, docs: List[Document]) -> str:
        """Định dạng các đoạn văn bản trích xuất kèm nguồn tham khảo và điểm rerank."""
        formatted_chunks = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Không rõ")
            score_info = ""
            if "rerank_score" in doc.metadata:
                score_info = f" | Độ tương quan: {doc.metadata['rerank_score']*100:.1f}%"
            formatted_chunks.append(f"[Đoạn {idx} - Nguồn: {source}{score_info}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted_chunks)

    def _get_graph_context(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Trích xuất tri thức đồ thị có cấu trúc (Graph Context) tương ứng với câu hỏi."""
        if not self.graph_manager or not hasattr(self.graph_manager, "graph"):
            return "Không có module đồ thị tri thức kết nối.", []

        q_lower = question.lower()
        graph = self.graph_manager.graph
        facts = []
        structured_items = []

        # 1. Phát hiện các cây có trong đồ thị xuất hiện trong câu hỏi
        matched_herbs = []
        for n, data in graph.nodes(data=True):
            if data.get("type") == "Herb":
                label = data.get("label", "")
                if label.lower() in q_lower or re.search(r'\b' + re.escape(label.lower()) + r'\b', q_lower):
                    matched_herbs.append((n, label))

        # Kiểm tra câu hỏi có ý định tìm cây cùng chữa bệnh hay không
        is_sharing_query = any(k in q_lower for k in ["cùng", "giống", "thay thế", "khác", "nhóm"])

        # 2. Xử lý câu hỏi tìm cây có chung công dụng
        if is_sharing_query and matched_herbs:
            for n_id, h_label in matched_herbs:
                sharing = self.graph_manager.find_herbs_sharing_disease(h_label)
                if sharing:
                    for item in sharing:
                        facts.append(
                            f"- Cây '{item['herb_name']}' CÓ CHUNG CÔNG DỤNG với '{h_label}' trong việc điều trị: {', '.join(item['shared_diseases'])}."
                        )
                        structured_items.append({
                            "type": "shared_disease",
                            "herb_a": h_label,
                            "herb_b": item["herb_name"],
                            "diseases": item["shared_diseases"],
                        })

        # 3. Xử lý đường đi liên kết giữa 2 cây nếu có 2 cây trở lên
        if len(matched_herbs) >= 2:
            paths_res = self.graph_manager.get_paths_between(matched_herbs[0][1], matched_herbs[1][1])
            if paths_res.get("found"):
                facts.append(f"- Phát hiện mối liên kết giữa '{matched_herbs[0][1]}' và '{matched_herbs[1][1]}' trong đồ thị qua {len(paths_res['paths'])} đường dẫn.")

        # 4. Trích xuất các quan hệ trực tiếp xung quanh các cây được nhắc đến
        for n_id, h_label in matched_herbs:
            ego = self.graph_manager.get_ego_graph(h_label, hops=1)
            parts = []
            compounds = []
            diseases = []
            contraindications = []

            for edge in ego.get("edges", []):
                tgt_id = edge["target"]
                rel = edge["relation"]
                tgt_node = graph.nodes.get(tgt_id, {})
                tgt_label = tgt_node.get("label", tgt_id)

                if rel == "USES_PART":
                    parts.append(tgt_label)
                elif rel == "CONTAINS_COMPOUND":
                    compounds.append(tgt_label)
                elif rel == "TREATS_DISEASE":
                    diseases.append(tgt_label)
                elif rel == "CONTRAINDICATED_FOR":
                    contraindications.append(tgt_label)

            if parts:
                facts.append(f"- Thảo dược '{h_label}' dùng bộ phận: {', '.join(set(parts))}.")
            if compounds:
                facts.append(f"- Thảo dược '{h_label}' chứa hoạt chất: {', '.join(set(compounds))}.")
            if diseases:
                facts.append(f"- Thảo dược '{h_label}' hỗ trợ điều trị: {', '.join(set(diseases))}.")
            if contraindications:
                facts.append(f"- Thảo dược '{h_label}' chống chỉ định/kiêng kỵ cho: {', '.join(set(contraindications))}.")

        # 5. Nếu chưa có cây nào khớp, tìm xem có bệnh nào xuất hiện trong câu hỏi không
        if not matched_herbs:
            for n, data in graph.nodes(data=True):
                if data.get("type") == "Disease":
                    d_label = data.get("label", "")
                    if len(d_label) >= 4 and d_label.lower() in q_lower:
                        herbs_found = self.graph_manager.find_herbs_for_disease(d_label)
                        if herbs_found:
                            h_names = [hf["herb_name"] for hf in herbs_found]
                            facts.append(f"- Theo Đồ thị tri thức, bệnh/triệu chứng '{d_label}' có thể được điều trị bằng các cây: {', '.join(set(h_names))}.")

        if facts:
            return "\n".join(facts), structured_items
        return "Không có dữ liệu quan hệ đồ thị tương ứng cho câu hỏi này.", []

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        evaluate_rag: bool = True,
    ) -> Dict[str, Any]:
        """Quy trình GraphRAG hoàn chỉnh:
        1. Lấy thống kê kho
        2. Truy xuất tài liệu bằng Advanced Retriever (HyDE + Multi-Query + CrossEncoder Reranker)
        3. Truy xuất mạng lưới quan hệ từ Knowledge Graph (Graph Context)
        4. Định dạng ngữ cảnh kép (Text + Graph) & sinh câu trả lời bằng Gemini
        5. Tự động đánh giá chất lượng bằng RAG Triad Evaluator (Context Relevance, Groundedness, Answer Relevance)
        """
        # 1. Lấy dữ liệu thống kê trực tiếp từ kho
        db_summary = self._get_live_database_summary()
        herb_list_str = ", ".join(db_summary["herb_list"]) if db_summary["herb_list"] else "Chưa có loài nào"

        # 2. Truy xuất tài liệu liên quan bằng Advanced Hybrid Search
        relevant_docs = self.retriever.get_relevant_documents(query=question, top_k=top_k, use_advanced=True)

        # 3. Truy xuất đồ thị tri thức (Graph Context)
        graph_context_str, graph_items = self._get_graph_context(question)

        # 4. Định dạng ngữ cảnh văn bản
        if relevant_docs:
            context_str = self._format_context(relevant_docs)
            sources = list({doc.metadata.get("source", "Không rõ") for doc in relevant_docs})
        else:
            context_str = "Hiện không tìm thấy đoạn trích dẫn cụ thể nào khớp với từ khóa tìm kiếm."
            sources = []

        # 5. Tạo chuỗi thực thi (LCEL Chain)
        chain = self.prompt_template | self.llm | self.output_parser

        # 6. Sinh phản hồi kết hợp Graph Context + Text Context
        combined_context_for_eval = f"GRAPH CONTEXT:\n{graph_context_str}\n\nTEXT CHUNKS:\n{context_str}"
        answer = chain.invoke({
            "total_count": db_summary["total_count"],
            "herb_list": herb_list_str,
            "graph_context": graph_context_str,
            "context": context_str,
            "question": question,
        })

        # 7. Đánh giá chất lượng bằng RAG Triad
        evaluation = None
        if evaluate_rag and (relevant_docs or graph_items):
            try:
                evaluation = self.evaluator.evaluate(
                    question=question,
                    context=combined_context_for_eval,
                    answer=answer,
                )
            except Exception as e:
                logger.warning(f"Lỗi khi đánh giá RAG Triad: {e}")

        # Chuẩn bị danh sách chi tiết các tài liệu đã trích xuất
        retrieved_details = []
        for d in relevant_docs:
            retrieved_details.append({
                "source": d.metadata.get("source", "Tài liệu"),
                "rerank_score": d.metadata.get("rerank_score"),
                "snippet": d.page_content[:150] + "..." if len(d.page_content) > 150 else d.page_content,
            })

        return {
            "answer": answer,
            "sources": sources,
            "db_stats": db_summary,
            "evaluation": evaluation,
            "retrieved_docs": retrieved_details,
            "graph_context": graph_items,
        }
