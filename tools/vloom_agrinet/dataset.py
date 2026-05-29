import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from vloom.config import DatasetConfig
from vloom.dataset.base import BaseDataset, DatasetItem

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INSECT_KEYWORDS = [
    "ant",
    "aphid",
    "apis",
    "arachnid",
    "araneae",
    "bee",
    "beetle",
    "bittacus",
    "bombus",
    "bug",
    "butterfly",
    "calopteryx",
    "chrysoperla",
    "coccinell",
    "danaus",
    "dragonfly",
    "fly",
    "hymenoptera",
    "insect",
    "ladybug",
    "mantis",
    "moth",
    "osmia",
    "pantala",
    "spider",
    "vespa",
    "vespula",
    "wasp",
]


def _norm_label(label: str) -> str:
    return label.strip().replace(" ", "_")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL row %s:%s: %s", path, line_no, exc)
    return rows


def _resolve_path(path: Optional[Path], base: Path) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    base_path = base / path
    return base_path


class AgriNetContrastDataset(BaseDataset):
    """AgriNet-1K image dataset for VLooM contrastive reasoning-chain sampling."""

    def __init__(
        self,
        name: str,
        description: str,
        data_root: Path,
        class_file: Path = Path("AgriNet-wds-cls.txt"),
        image_root: Optional[Path] = None,
        knowledge_file: Optional[Path] = None,
        pairs_file: Optional[Path] = None,
        sample_file: Optional[Path] = None,
        include_keywords: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        candidates_per_item: int = 4,
        include_references: bool = True,
        include_wiki: bool = True,
        allow_missing_data: bool = False,
        recursive_fallback: bool = False,
    ):
        super().__init__(name, description)
        self.data_root = Path(data_root)
        self.class_file = _resolve_path(class_file, self.data_root) or self.data_root / "AgriNet-wds-cls.txt"
        self.image_root = Path(image_root) if image_root else self.data_root
        if image_root is None:
            self.image_root = self.data_root
        elif not self.image_root.is_absolute():
            self.image_root = self.data_root / self.image_root
        self.knowledge_file = _resolve_path(knowledge_file, self.data_root) if knowledge_file else self.data_root / "metadata/classes_knowledge.jsonl"
        self.pairs_file = _resolve_path(pairs_file, self.data_root) if pairs_file else self.data_root / "metadata/finegrained_pairs.jsonl"
        self.sample_file = _resolve_path(sample_file, self.data_root) if sample_file else None
        self.include_keywords = [kw.lower() for kw in (include_keywords or DEFAULT_INSECT_KEYWORDS)]
        self.max_samples = max_samples
        self.candidates_per_item = max(2, candidates_per_item)
        self.include_references = include_references
        self.include_wiki = include_wiki
        self.allow_missing_data = allow_missing_data
        self.recursive_fallback = recursive_fallback

        self.classes: List[str] = []
        self.knowledge_by_label: Dict[str, Dict[str, Any]] = {}
        self.similar_by_label: Dict[str, List[str]] = {}
        self.load_index()

    def load_index(self):
        self.classes = self._load_classes()
        self.knowledge_by_label = self._load_knowledge()
        self.similar_by_label = self._load_pairs()
        sample_items = self._load_samples()
        if sample_items:
            self.items = sample_items[: self.max_samples] if self.max_samples else sample_items
            for item in self.items:
                label = item.get("final_label") or item.get("label")
                if label and label not in self.classes:
                    self.classes.append(label)
            logger.info("AgriNetContrastDataset loaded %d manifest samples", len(self.items))
            return

        if not self.classes:
            message = f"No AgriNet classes found at {self.class_file}"
            if self.allow_missing_data:
                logger.warning("%s; using placeholder dry-run item", message)
                self.classes = ["Apis_mellifera", "Bombus_hypocrita", "Vespula_vulgaris", "ladybugs"]
                self.items = [
                    {
                        "img_path": None,
                        "img_name": "placeholder",
                        "label": "Apis_mellifera",
                    }
                ]
                return
            raise FileNotFoundError(message)

        image_items = self._scan_images()
        if not image_items:
            message = f"No image files found under {self.image_root}"
            if self.allow_missing_data:
                logger.warning("%s; using placeholder dry-run item", message)
                image_items = [
                    {
                        "img_path": None,
                        "img_name": "placeholder",
                        "label": "Apis_mellifera",
                    }
                ]
                for label in ["Apis_mellifera", "Bombus_hypocrita", "Vespula_vulgaris", "ladybugs"]:
                    if label not in self.classes:
                        self.classes.append(label)
                self.similar_by_label.setdefault("Apis_mellifera", ["Bombus_hypocrita", "Vespula_vulgaris", "ladybugs"])
            else:
                raise FileNotFoundError(message)

        if self.max_samples:
            image_items = image_items[: self.max_samples]
        self.items = image_items
        logger.info("AgriNetContrastDataset loaded %d items", len(self.items))

    def _load_classes(self) -> List[str]:
        if not self.class_file.exists():
            return []
        if self.class_file.suffix == ".jsonl":
            labels = []
            for row in _read_jsonl(self.class_file):
                label = row.get("code") or row.get("label") or row.get("class") or row.get("class_name")
                if label:
                    labels.append(_norm_label(str(label)))
            return labels
        labels: List[str] = []
        with self.class_file.open("r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if not label:
                    continue
                if " " in label and label.split()[0].isdigit():
                    label = " ".join(label.split()[1:])
                labels.append(_norm_label(label))
        return labels

    def _load_knowledge(self) -> Dict[str, Dict[str, Any]]:
        rows = _read_jsonl(self.knowledge_file)
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            label = row.get("code") or row.get("label") or row.get("class") or row.get("class_name")
            if label:
                out[_norm_label(str(label))] = row
        return out

    def _load_pairs(self) -> Dict[str, List[str]]:
        rows = _read_jsonl(self.pairs_file)
        out: Dict[str, List[str]] = {}
        for row in rows:
            label = row.get("label") or row.get("class") or row.get("anchor")
            similar = row.get("similar_classes") or row.get("candidates") or row.get("positives") or []
            if not similar and row.get("hard_negatives"):
                similar = [item.get("code") for item in row["hard_negatives"] if item.get("code")]
            if isinstance(similar, str):
                similar = [similar]
            if label and similar:
                out[_norm_label(str(label))] = [_norm_label(str(x)) for x in similar]
        return out

    def _load_samples(self) -> List[Dict[str, Any]]:
        if not self.sample_file:
            return []
        rows = _read_jsonl(self.sample_file)
        items: List[Dict[str, Any]] = []
        for row in rows:
            query_image = row.get("query_image")
            label = row.get("final_label") or row.get("label")
            if not query_image or not label:
                continue
            items.append(row)
        return items

    def _filtered_classes(self) -> List[str]:
        filtered = []
        for label in self.classes:
            haystack = label.lower().replace("_", " ")
            knowledge = self.knowledge_by_label.get(label, {})
            domain_group = str(knowledge.get("domain_group", "")).lower()
            roles = " ".join(str(x).lower() for x in knowledge.get("agricultural_role", []))
            if any(kw in haystack or kw in domain_group or kw in roles for kw in self.include_keywords):
                filtered.append(label)
        return filtered or self.classes

    def _scan_images(self) -> List[Dict[str, Any]]:
        target_classes = set(self._filtered_classes())
        if not self.image_root.exists():
            return []

        items: List[Dict[str, Any]] = []
        for label in sorted(target_classes):
            class_dir = self.image_root / label
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                items.append({"img_path": path, "img_name": path.stem, "label": label})
                if self.max_samples and len(items) >= self.max_samples:
                    return items
        if items:
            return items
        if not self.recursive_fallback:
            return []

        for path in sorted(self.image_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = self._infer_label(path, target_classes)
            if not label:
                continue
            items.append({"img_path": path, "img_name": path.stem, "label": label})
            if self.max_samples and len(items) >= self.max_samples:
                return items
        return items

    def _infer_label(self, path: Path, target_classes: set[str]) -> Optional[str]:
        parts = [_norm_label(part) for part in path.relative_to(self.image_root).parts[:-1]]
        for part in reversed(parts):
            if part in target_classes:
                return part
        stem = _norm_label(path.stem)
        for label in target_classes:
            if stem == label or stem.startswith(f"{label}_") or stem.startswith(f"{label}-"):
                return label
        return None

    def _candidate_labels(self, label: str) -> List[str]:
        candidates = [label]
        for candidate in self.similar_by_label.get(label, []):
            if candidate not in self.classes:
                self.classes.append(candidate)
            if candidate not in candidates:
                candidates.append(candidate)

        filtered = self._filtered_classes()
        for candidate in filtered:
            if len(candidates) >= self.candidates_per_item:
                break
            if candidate != label and candidate not in candidates:
                candidates.append(candidate)
        return candidates[: self.candidates_per_item]

    async def get_item(self, index: int) -> DatasetItem:
        item = self.items[index]
        if "query_image" in item:
            label = _norm_label(str(item["final_label"]))
            candidate_objects = item.get("candidate_labels", [])
            candidate_codes = [
                _norm_label(str(candidate.get("code", candidate)))
                for candidate in candidate_objects
            ]
            if label not in candidate_codes:
                candidate_codes.insert(0, label)
            positives = item.get("positive_reference_images", [])
            negatives = item.get("negative_reference_images", [])
            if not self.include_references:
                positives = []
                negatives = []
            negative_paths = [entry.get("image_path") for entry in negatives if entry.get("image_path")]
            image_paths = positives + negative_paths
            wiki_evidence = item.get("wiki_evidence", []) if self.include_wiki else ["insufficient wiki knowledge"]
            wiki_available = bool(item.get("wiki_available")) if self.include_wiki else False
            template_vars = {
                "img_name": item.get("sample_id") or Path(item["query_image"]).stem,
                "task_domain": item.get("task_domain", "unknown"),
                "gt_label": label,
                "gt_label_zh": item.get("final_label_zh", ""),
                "candidate_labels": candidate_codes,
                "candidate_label_details": candidate_objects,
                "query_image": item["query_image"],
                "positive_reference_images": positives,
                "negative_reference_images": negatives,
                "image_paths": image_paths,
                "class_knowledge": {"wiki_evidence": wiki_evidence},
                "wiki_evidence": wiki_evidence,
                "knowledge_available": wiki_available,
                "wiki_available": wiki_available,
            }
            return DatasetItem(
                idx=index,
                img_name=template_vars["img_name"],
                img_path=Path(item["query_image"]),
                template_vars=template_vars,
                valid=True,
            )

        label = item["label"]
        knowledge = self.knowledge_by_label.get(label, {})
        candidates = self._candidate_labels(label)
        template_vars = {
            "img_name": item["img_name"],
            "gt_label": label,
            "candidate_labels": candidates,
            "class_knowledge": knowledge,
            "similar_classes": [c for c in candidates if c != label],
            "knowledge_available": bool(knowledge),
        }
        return DatasetItem(
            idx=index,
            img_name=item["img_name"],
            img_path=item["img_path"],
            template_vars=template_vars,
            valid=True,
        )


@DatasetConfig.register_subclass("agrinet_contrast")
@dataclass
class AgriNetContrastConfig(DatasetConfig):
    data_root: Path = Path("datasets/AgriNet-1K")
    class_file: Path = Path("AgriNet-wds-cls.txt")
    image_root: Optional[Path] = None
    knowledge_file: Optional[Path] = None
    pairs_file: Optional[Path] = None
    sample_file: Optional[Path] = None
    include_keywords: Optional[List[str]] = field(default_factory=lambda: DEFAULT_INSECT_KEYWORDS.copy())
    max_samples: Optional[int] = 10
    candidates_per_item: int = 4
    include_references: bool = True
    include_wiki: bool = True
    allow_missing_data: bool = True
    recursive_fallback: bool = False

    def dataset_class(self) -> Type[AgriNetContrastDataset]:
        return AgriNetContrastDataset
