import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    HERB = "Herb"
    PLANT_PART = "PlantPart"
    COMPOUND = "Compound"
    DISEASE = "Disease"
    CONTRAINDICATION = "Contraindication"


class RelationType(str, Enum):
    USES_PART = "USES_PART"
    CONTAINS_COMPOUND = "CONTAINS_COMPOUND"
    TREATS_DISEASE = "TREATS_DISEASE"
    CONTRAINDICATED_FOR = "CONTRAINDICATED_FOR"


# Bộ quy tắc ràng buộc các cặp (SourceType, Relation, TargetType) được phép tồn tại
VALID_RELATION_CONSTRAINTS: Set[Tuple[str, str, str]] = {
    (EntityType.HERB.value, RelationType.USES_PART.value, EntityType.PLANT_PART.value),
    (EntityType.HERB.value, RelationType.CONTAINS_COMPOUND.value, EntityType.COMPOUND.value),
    (EntityType.HERB.value, RelationType.TREATS_DISEASE.value, EntityType.DISEASE.value),
    (EntityType.COMPOUND.value, RelationType.TREATS_DISEASE.value, EntityType.DISEASE.value),
    (EntityType.HERB.value, RelationType.CONTRAINDICATED_FOR.value, EntityType.CONTRAINDICATION.value),
}


def slugify_vietnamese(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt thành slug chuẩn ID (ví dụ: 'Cà Gai Leo' -> 'ca_gai_leo')."""
    text = text.lower().strip()
    patterns = {
        r'[àáạảãâầấậẩẫăằắặẳẵ]': 'a',
        r'[èéẹẻẽêềếệểễ]': 'e',
        r'[ìíịỉĩ]': 'i',
        r'[òóọỏõôồốộổỗơờớợởỡ]': 'o',
        r'[ùúụủũưừứựửữ]': 'u',
        r'[ỳýỵỷỹ]': 'y',
        r'[đ]': 'd',
    }
    for regex, replace in patterns.items():
        text = re.sub(regex, replace, text)
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'[\s\-]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or "unknown"


def normalize_label(label: str) -> str:
    """Chuẩn hóa nhãn thực thể: Xóa khoảng trắng thừa, viết hoa chữ cái đầu (Title Case)."""
    clean = " ".join(label.strip().split())
    # Chuyển đổi thành Title Case nhưng giữ nguyên các từ viết hoa viết tắt (ví dụ B, K5, B1...)
    words = clean.split()
    capitalized_words = []
    for w in words:
        if len(w) <= 2 and w.isupper():
            capitalized_words.append(w)
        else:
            capitalized_words.append(w.capitalize())
    return " ".join(capitalized_words)


def generate_node_id(entity_type: str, label: str) -> str:
    """Tạo ID duy nhất có tiền tố loại thực thể (ví dụ: 'herb_ca_gai_leo', 'disease_viem_gan_b')."""
    type_prefix = slugify_vietnamese(entity_type)
    label_slug = slugify_vietnamese(label)
    return f"{type_prefix}_{label_slug}"


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: Optional[Dict[str, str]] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    doc_source: str


class TripleItem(BaseModel):
    source_label: str
    source_type: str
    relation: str
    target_label: str
    target_type: str


class KnowledgeGraphData(BaseModel):
    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)

