import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import networkx as nx

from src.modules.graph.graph_schema import (
    EntityType,
    RelationType,
    TripleItem,
    GraphNode,
    GraphEdge,
    KnowledgeGraphData,
    generate_node_id,
    normalize_label,
)

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path("data/graph/knowledge_graph.json")


class KnowledgeGraphManager:
    """Quản lý đồ thị tri thức thảo dược bằng NetworkX và lưu trữ tại data/graph/knowledge_graph.json."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_GRAPH_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self.load_graph()

    def load_graph(self) -> None:
        """Đọc file knowledge_graph.json và nạp vào NetworkX DiGraph."""
        self.graph.clear()
        if not self.storage_path.exists():
            logger.info(f"Chưa có file đồ thị tại {self.storage_path}. Khởi tạo đồ thị rỗng.")
            self.save_graph()
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            nodes_data = raw_data.get("nodes", {})
            for node_id, n_info in nodes_data.items():
                self.graph.add_node(
                    node_id,
                    id=node_id,
                    label=n_info.get("label", node_id),
                    type=n_info.get("type", "Unknown"),
                    properties=n_info.get("properties", {}),
                )

            edges_data = raw_data.get("edges", [])
            for e_info in edges_data:
                src = e_info.get("source")
                tgt = e_info.get("target")
                rel = e_info.get("relation")
                doc = e_info.get("doc_source", "")
                if src and tgt and self.graph.has_node(src) and self.graph.has_node(tgt):
                    self.graph.add_edge(src, tgt, relation=rel, doc_source=doc)

            logger.info(
                f"✓ Đã nạp đồ thị tri thức: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges từ {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Lỗi khi đọc file đồ thị {self.storage_path}: {e}")

    def save_graph(self) -> None:
        """Ghi toàn bộ trạng thái đồ thị ra file knowledge_graph.json."""
        try:
            nodes_dict = {}
            for node_id, data in self.graph.nodes(data=True):
                nodes_dict[node_id] = {
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "type": data.get("type", "Unknown"),
                    "properties": data.get("properties", {}),
                }

            edges_list = []
            for u, v, data in self.graph.edges(data=True):
                edges_list.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", ""),
                    "doc_source": data.get("doc_source", ""),
                })

            export_data = {
                "nodes": nodes_dict,
                "edges": edges_list,
            }

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"✓ Đã lưu đồ thị vào {self.storage_path} ({len(nodes_dict)} nodes, {len(edges_list)} edges)"
            )
        except Exception as e:
            logger.error(f"Lỗi khi lưu đồ thị ra {self.storage_path}: {e}")

    def add_herb_graph(self, triples: List[TripleItem], doc_source: str) -> Dict[str, int]:
        """Thêm các Triples của một file tài liệu vào đồ thị.

        - Nếu node chưa có: tạo mới.
        - Nếu node đã có: giữ nguyên.
        - Thêm các edges kèm thuộc tính doc_source.
        """
        if not triples:
            return {"added_nodes": 0, "added_edges": 0}

        added_nodes = 0
        added_edges = 0

        for t in triples:
            s_label = normalize_label(t.source_label)
            t_label = normalize_label(t.target_label)

            s_id = generate_node_id(t.source_type, s_label)
            t_id = generate_node_id(t.target_type, t_label)

            if not self.graph.has_node(s_id):
                self.graph.add_node(s_id, id=s_id, label=s_label, type=t.source_type, properties={})
                added_nodes += 1

            if not self.graph.has_node(t_id):
                self.graph.add_node(t_id, id=t_id, label=t_label, type=t.target_type, properties={})
                added_nodes += 1

            if not self.graph.has_edge(s_id, t_id):
                self.graph.add_edge(s_id, t_id, relation=t.relation, doc_source=doc_source)
                added_edges += 1
            else:
                # Nếu cạnh đã có, cập nhật thêm doc_source nếu cần
                current_docs = self.graph[s_id][t_id].get("doc_source", "")
                if doc_source and doc_source not in current_docs:
                    self.graph[s_id][t_id]["doc_source"] = f"{current_docs}, {doc_source}".strip(", ")

        self.save_graph()
        logger.info(
            f"✓ Hoàn tất cập nhật đồ thị từ '{doc_source}': thêm {added_nodes} nodes mới, {added_edges} edges mới."
        )
        return {"added_nodes": added_nodes, "added_edges": added_edges}

    def delete_by_source(self, doc_source: str) -> Dict[str, int]:
        """Xóa toàn bộ các cạnh gắn với doc_source và dọn dẹp các node mồ côi."""
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            edge_docs = data.get("doc_source", "")
            if doc_source in edge_docs:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)

        # Xóa các node mồ côi (không còn bất kỳ cạnh nào liên kết)
        isolated_nodes = list(nx.isolates(self.graph))
        for n in isolated_nodes:
            self.graph.remove_node(n)

        self.save_graph()
        logger.info(
            f"✓ Đã dọn dẹp đồ thị nguồn '{doc_source}': xóa {len(edges_to_remove)} edges, {len(isolated_nodes)} nodes mồ côi."
        )
        return {"removed_edges": len(edges_to_remove), "removed_nodes": len(isolated_nodes)}

    def find_node_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[str]:
        """Tìm ID của node dựa trên tên / nhãn."""
        name_clean = name.strip().lower()
        if entity_type:
            direct_id = generate_node_id(entity_type, name)
            if self.graph.has_node(direct_id):
                return direct_id

        for node_id, data in self.graph.nodes(data=True):
            if data.get("label", "").lower() == name_clean:
                if not entity_type or data.get("type") == entity_type:
                    return node_id
        return None

    def get_ego_graph(self, herb_name_or_id: str, hops: int = 1) -> Dict[str, Any]:
        """Trích xuất mạng lưới liên kết cục bộ xung quanh một thực thể (bán kính 1 hoặc 2 bước nhảy)."""
        target_id = None
        if self.graph.has_node(herb_name_or_id):
            target_id = herb_name_or_id
        else:
            target_id = self.find_node_by_name(herb_name_or_id, EntityType.HERB.value)
            if not target_id:
                target_id = self.find_node_by_name(herb_name_or_id)

        if not target_id:
            return {"center_node": None, "nodes": [], "edges": []}

        # Tạo ego graph từ đồ thị vô hướng để lấy cả cạnh vào và cạnh ra
        undirected = self.graph.to_undirected()
        sub_nodes = set(nx.ego_graph(undirected, target_id, radius=hops).nodes())

        nodes_list = []
        for n in sub_nodes:
            nd = self.graph.nodes[n]
            nodes_list.append({
                "id": n,
                "label": nd.get("label", n),
                "type": nd.get("type", "Unknown"),
                "is_center": (n == target_id),
            })

        edges_list = []
        for u in sub_nodes:
            for v in sub_nodes:
                if self.graph.has_edge(u, v):
                    ed = self.graph[u][v]
                    edges_list.append({
                        "source": u,
                        "target": v,
                        "relation": ed.get("relation", ""),
                        "doc_source": ed.get("doc_source", ""),
                    })

        return {
            "center_node": target_id,
            "center_label": self.graph.nodes[target_id].get("label", target_id),
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def expand_node(self, node_id: str) -> Dict[str, Any]:
        """Lấy tất cả láng giềng trực tiếp (in và out edges) của node để bổ sung vào đồ thị UI."""
        if not self.graph.has_node(node_id):
            return {"nodes": [], "edges": []}

        neighbors: Set[str] = set()
        neighbors.update(self.graph.successors(node_id))
        neighbors.update(self.graph.predecessors(node_id))
        neighbors.add(node_id)

        nodes_list = []
        for n in neighbors:
            nd = self.graph.nodes[n]
            nodes_list.append({
                "id": n,
                "label": nd.get("label", n),
                "type": nd.get("type", "Unknown"),
            })

        edges_list = []
        for n in neighbors:
            if self.graph.has_edge(node_id, n):
                ed = self.graph[node_id][n]
                edges_list.append({
                    "source": node_id,
                    "target": n,
                    "relation": ed.get("relation", ""),
                    "doc_source": ed.get("doc_source", ""),
                })
            if self.graph.has_edge(n, node_id):
                ed = self.graph[n][node_id]
                edges_list.append({
                    "source": n,
                    "target": node_id,
                    "relation": ed.get("relation", ""),
                    "doc_source": ed.get("doc_source", ""),
                })

        return {"nodes": nodes_list, "edges": edges_list}

    def get_paths_between(self, herb_a: str, herb_b: str) -> Dict[str, Any]:
        """Tìm mối quan hệ bắc cầu giữa 2 cây (cùng hoạt chất, cùng trị bệnh...)."""
        id_a = self.find_node_by_name(herb_a, EntityType.HERB.value) or self.find_node_by_name(herb_a)
        id_b = self.find_node_by_name(herb_b, EntityType.HERB.value) or self.find_node_by_name(herb_b)

        if not id_a or not id_b:
            return {"found": False, "paths": [], "nodes": [], "edges": []}

        undirected = self.graph.to_undirected()
        try:
            raw_paths = list(nx.all_simple_paths(undirected, source=id_a, target=id_b, cutoff=3))
        except Exception:
            raw_paths = []

        if not raw_paths:
            return {"found": False, "paths": [], "nodes": [], "edges": []}

        all_nodes_in_paths = set()
        for p in raw_paths:
            all_nodes_in_paths.update(p)

        nodes_list = []
        for n in all_nodes_in_paths:
            nd = self.graph.nodes[n]
            nodes_list.append({
                "id": n,
                "label": nd.get("label", n),
                "type": nd.get("type", "Unknown"),
            })

        edges_list = []
        for u in all_nodes_in_paths:
            for v in all_nodes_in_paths:
                if self.graph.has_edge(u, v):
                    ed = self.graph[u][v]
                    edges_list.append({
                        "source": u,
                        "target": v,
                        "relation": ed.get("relation", ""),
                        "doc_source": ed.get("doc_source", ""),
                    })

        return {
            "found": True,
            "paths": raw_paths,
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def find_herbs_sharing_disease(self, herb_name: str) -> List[Dict[str, Any]]:
        """Tìm các loài thảo dược khác có chung công dụng trị bệnh với loài cây được chỉ định."""
        herb_id = self.find_node_by_name(herb_name, EntityType.HERB.value)
        if not herb_id:
            return []

        # 1. Lấy danh sách các bệnh mà cây này điều trị
        diseases: Set[str] = set()
        for tgt in self.graph.successors(herb_id):
            edge_data = self.graph[herb_id][tgt]
            if edge_data.get("relation") == RelationType.TREATS_DISEASE.value:
                diseases.add(tgt)

        if not diseases:
            return []

        # 2. Tìm các cây khác cùng điều trị các bệnh này
        sharing_map: Dict[str, Set[str]] = {}
        for d_id in diseases:
            d_label = self.graph.nodes[d_id].get("label", d_id)
            for src in self.graph.predecessors(d_id):
                if src != herb_id and self.graph.nodes[src].get("type") == EntityType.HERB.value:
                    if src not in sharing_map:
                        sharing_map[src] = set()
                    sharing_map[src].add(d_label)

        results = []
        for h_id, shared_d in sharing_map.items():
            results.append({
                "herb_id": h_id,
                "herb_name": self.graph.nodes[h_id].get("label", h_id),
                "shared_diseases": list(shared_d),
            })
        return results

    def find_herbs_for_disease(self, disease_name: str) -> List[Dict[str, Any]]:
        """Tìm các cây thuốc hỗ trợ điều trị một bệnh lý / triệu chứng cụ thể."""
        d_id = self.find_node_by_name(disease_name, EntityType.DISEASE.value)
        if not d_id:
            # Tìm kiếm mờ nếu không khớp chính xác
            d_name_lower = disease_name.lower().strip()
            for nid, data in self.graph.nodes(data=True):
                if data.get("type") == EntityType.DISEASE.value and (
                    d_name_lower in data.get("label", "").lower() or data.get("label", "").lower() in d_name_lower
                ):
                    d_id = nid
                    break

        if not d_id:
            return []

        direct_herbs = []
        for src in self.graph.predecessors(d_id):
            if self.graph.nodes[src].get("type") == EntityType.HERB.value:
                direct_herbs.append({
                    "herb_id": src,
                    "herb_name": self.graph.nodes[src].get("label", src),
                    "via_compound": None,
                })
            elif self.graph.nodes[src].get("type") == EntityType.COMPOUND.value:
                compound_label = self.graph.nodes[src].get("label", src)
                for h_src in self.graph.predecessors(src):
                    if self.graph.nodes[h_src].get("type") == EntityType.HERB.value:
                        direct_herbs.append({
                            "herb_id": h_src,
                            "herb_name": self.graph.nodes[h_src].get("label", h_src),
                            "via_compound": compound_label,
                        })

        return direct_herbs

    def get_full_graph(self) -> Dict[str, Any]:
        """Lấy toàn bộ dữ liệu đồ thị tri thức kèm thống kê."""
        nodes_list = []
        type_counts: Dict[str, int] = {}
        for n, data in self.graph.nodes(data=True):
            ntype = data.get("type", "Unknown")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1
            nodes_list.append({
                "id": n,
                "label": data.get("label", n),
                "type": ntype,
            })

        edges_list = []
        relation_counts: Dict[str, int] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
            edges_list.append({
                "source": u,
                "target": v,
                "relation": rel,
                "doc_source": data.get("doc_source", ""),
            })

        return {
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list),
            "type_counts": type_counts,
            "relation_counts": relation_counts,
            "nodes": nodes_list,
            "edges": edges_list,
        }

