"""MMNER jsonl dataset loader for future full GMNER training."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import box_iou
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX, normalize_entity_type, strip_bio_prefix
from gmner.data.graph_builders import TextGraphBuilder
from gmner.data.tokenization import encode_words_with_alignment
from gmner.utils.io import read_jsonl


class MMNERJsonDataset(Dataset):
    """Loads jsonl records with tokens/ner_tags/image metadata."""

    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        graph_builder: TextGraphBuilder,
        max_length: int = 128,
        label2id: Optional[dict[str, int]] = None,
        grounding_enabled: bool = False,
        expand_entities_for_grounding: bool = True,
        image_feature_dir: Optional[str] = None,
        image_annotation_dir: Optional[str] = None,
        max_regions: int = 16,
        region_feature_dim: int = 2048,
        grounding_iou_threshold: float = 0.5,
        add_null_region: bool = True,
        graph_cache_dir: Optional[str] = None,
        groundability_type_priors: Optional[str] = None,
        groundability_mention_priors: Optional[str] = None,
        region_min_score: float = 0.0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.graph_builder = graph_builder
        self.max_length = max_length
        self.label2id = label2id or DEFAULT_LABEL2ID
        self.id2label = {value: key for key, value in self.label2id.items()}
        self.records = read_jsonl(self.jsonl_path)
        self.subtype_label2id = self._build_subtype_label2id(self.records)
        self.grounding_enabled = grounding_enabled
        self.expand_entities_for_grounding = expand_entities_for_grounding
        self.image_feature_dir = Path(image_feature_dir) if image_feature_dir else None
        self.image_annotation_dir = Path(image_annotation_dir) if image_annotation_dir else None
        self.max_regions = max_regions
        self.region_feature_dim = region_feature_dim
        self.grounding_iou_threshold = grounding_iou_threshold
        self.add_null_region = add_null_region
        self.region_min_score = float(region_min_score)
        self.use_dependency_graph = bool(getattr(graph_builder, "config", None) and graph_builder.config.use_dependency_graph)
        self.graph_cache_dir: Optional[Path] = None
        if self.use_dependency_graph:
            cache_base = (
                Path(graph_cache_dir)
                if graph_cache_dir
                else self.jsonl_path.parent / "graph_cache" / self.jsonl_path.stem
            )
            cache_base.mkdir(parents=True, exist_ok=True)
            self.graph_cache_dir = cache_base
        self.groundability_type_null_prior: Dict[str, float] = {}
        self.groundability_mention_type_null_prior: Dict[tuple[str, str], float] = {}
        self._load_groundability_priors(
            type_prior_path=groundability_type_priors,
            mention_prior_path=groundability_mention_priors,
        )
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def _convert_tags(self, tags: List) -> List[int]:
        converted: List[int] = []
        for tag in tags:
            if isinstance(tag, int):
                converted.append(tag)
            else:
                converted.append(self.label2id.get(str(tag), IGNORE_INDEX))
        return converted

    @staticmethod
    def _build_subtype_label2id(records: List[dict]) -> Dict[str, int]:
        id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
        labels: set[tuple[int, str]] = set()
        for record in records:
            fine_tags = record.get("fine_ner_tags")
            if not isinstance(fine_tags, list):
                continue
            ner_tags = record.get("ner_tags", [])
            for idx, tag in enumerate(fine_tags):
                subtype = strip_bio_prefix(str(tag))
                if subtype != "O":
                    coarse_label = id2label.get(int(ner_tags[idx]), "O") if idx < len(ner_tags) else "O"
                    coarse_type = coarse_label.split("-", 1)[1] if "-" in coarse_label else "OTHER"
                    labels.add((ENTITY_TYPE2ID.get(coarse_type, ENTITY_TYPE2ID["OTHER"]), subtype))
        return {label: idx for idx, (_, label) in enumerate(sorted(labels))}

    def _tags_to_labels(self, tags: List[int]) -> List[str]:
        labels = []
        for tag in tags:
            if isinstance(tag, int):
                labels.append(self.id2label.get(tag, "O"))
            else:
                labels.append(str(tag))
        return labels

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower()

    def _extract_entities(
        self,
        tokens: List[str],
        tags: List[int],
        fine_tags: Optional[List[str]] = None,
    ) -> List[Tuple[int, int, str, str, str]]:
        labels = self._tags_to_labels(tags)
        entities: List[Tuple[int, int, str, str, str]] = []
        start = None
        current_type = None

        def span_subtype(span_start: int, span_end: int, fallback_type: str) -> str:
            if not fine_tags:
                return fallback_type
            for fine_label in fine_tags[span_start:span_end]:
                subtype = strip_bio_prefix(fine_label)
                if subtype != "O":
                    return subtype
            return fallback_type

        for idx, label in enumerate(labels):
            if label == "O":
                if start is not None:
                    entity_text = " ".join(tokens[start:idx])
                    entity_type = current_type or "O"
                    entities.append((start, idx, entity_text, entity_type, span_subtype(start, idx, entity_type)))
                    start = None
                    current_type = None
                continue

            if label.startswith("B-"):
                if start is not None:
                    entity_text = " ".join(tokens[start:idx])
                    entity_type = current_type or "O"
                    entities.append((start, idx, entity_text, entity_type, span_subtype(start, idx, entity_type)))
                start = idx
                current_type = label[2:]
                continue

            if label.startswith("I-"):
                ent_type = label[2:]
                if start is None or current_type != ent_type:
                    start = idx
                    current_type = ent_type
                continue

        if start is not None:
            entity_text = " ".join(tokens[start:len(tokens)])
            entity_type = current_type or "O"
            entities.append(
                (start, len(tokens), entity_text, entity_type, span_subtype(start, len(tokens), entity_type))
            )

        return entities

    def _extract_evidence_entities(
        self,
        tokens: List[str],
        record: dict,
    ) -> List[Tuple[int, int, str, str, str, bool, Optional[str]]]:
        """Read predicted-span evidence entities from a prepared Stage 2 file.

        Each item may contain ``start``, ``end``, ``text``, ``target_type`` and
        ``target_subtype``. ``force_null_region`` is supported for future false
        positive training samples, but exact-span matched samples normally keep
        it false and use XML/VinVL matching.
        """

        raw_entities = record.get("evidence_entities")
        if not isinstance(raw_entities, list):
            return []

        entities: List[Tuple[int, int, str, str, str, bool, Optional[str]]] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start"))
                end = int(item.get("end"))
            except Exception:
                continue
            if start < 0 or end <= start or end > len(tokens):
                continue
            entity_type = normalize_entity_type(
                item.get("target_type", item.get("type", "OTHER"))
            )
            entity_subtype = str(item.get("target_subtype") or entity_type)
            entity_text = str(item.get("text") or " ".join(tokens[start:end]))
            force_null_region = bool(item.get("force_null_region", False))
            raw_predicted_type = item.get("predicted_type", item.get("type"))
            predicted_type = (
                normalize_entity_type(raw_predicted_type)
                if raw_predicted_type is not None
                else None
            )
            if predicted_type == "O":
                predicted_type = None
            entities.append(
                (
                    start,
                    end,
                    entity_text,
                    entity_type,
                    entity_subtype,
                    force_null_region,
                    predicted_type,
                )
            )
        return entities

    def _load_groundability_priors(
        self,
        type_prior_path: Optional[str],
        mention_prior_path: Optional[str],
    ) -> None:
        if type_prior_path:
            path = Path(type_prior_path)
            if path.exists():
                try:
                    for entry in read_jsonl(path):
                        entity_type = str(entry.get("entity_type", ""))
                        if entity_type:
                            self.groundability_type_null_prior[entity_type] = float(entry.get("null_prior", 0.5))
                except Exception:
                    self.groundability_type_null_prior = {}

        if mention_prior_path:
            path = Path(mention_prior_path)
            if path.exists():
                try:
                    for entry in read_jsonl(path):
                        mention = self._normalize_name(str(entry.get("mention", "")))
                        entity_type = str(entry.get("entity_type", ""))
                        if mention and entity_type:
                            self.groundability_mention_type_null_prior[(mention, entity_type)] = float(
                                entry.get("null_prior", 0.5)
                            )
                except Exception:
                    self.groundability_mention_type_null_prior = {}

    def _get_grounding_null_prior(self, entity_text: str, entity_type: str) -> float:
        mention_key = self._normalize_name(entity_text)
        value = self.groundability_mention_type_null_prior.get((mention_key, entity_type))
        if value is None:
            value = self.groundability_type_null_prior.get(entity_type, 0.5)
        return float(min(max(value, 1e-4), 1.0 - 1e-4))

    @staticmethod
    def _build_word_spans(tokens: List[str]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            start = cursor
            end = start + len(token)
            spans.append((start, end))
            cursor = end + 1
        return spans

    def _read_xml_boxes(self, image_id: str) -> Dict[str, List[List[int]]]:
        if not self.image_annotation_dir:
            return {}
        xml_path = self.image_annotation_dir / f"{image_id}.xml"
        if not xml_path.exists():
            return {}

        boxes_by_name: Dict[str, List[List[int]]] = {}
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        for obj in root.findall("object"):
            name_node = obj.find("name")
            box_node = obj.find("bndbox")
            if name_node is None or box_node is None:
                continue
            name = self._normalize_name(name_node.text or "")
            try:
                xmin = int(box_node.findtext("xmin", default="0"))
                ymin = int(box_node.findtext("ymin", default="0"))
                xmax = int(box_node.findtext("xmax", default="0"))
                ymax = int(box_node.findtext("ymax", default="0"))
            except Exception:
                continue
            boxes_by_name.setdefault(name, []).append([xmin, ymin, xmax, ymax])

        return boxes_by_name

    def _load_region_features(
        self,
        image_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
        features = np.zeros((self.max_regions, self.region_feature_dim), dtype=np.float32)
        boxes = np.zeros((self.max_regions, 4), dtype=np.float32)
        mask = np.zeros((self.max_regions,), dtype=np.float32)
        scores = np.zeros((self.max_regions,), dtype=np.float32)
        object_labels = [""] * self.max_regions
        object_attributes = [""] * self.max_regions
        image_size = np.zeros((2,), dtype=np.float32)

        if self.image_feature_dir:
            npz_path = self.image_feature_dir / f"{image_id}.jpg.npz"
            if npz_path.exists():
                data = np.load(str(npz_path), allow_pickle=True)
                num_boxes = int(data.get("num_boxes", 0))
                image_size[0] = float(data.get("image_h", 0))
                image_size[1] = float(data.get("image_w", 0))
                raw_scores = np.asarray(data.get("scores", np.ones((num_boxes,), dtype=np.float32)))
                selected_indices = np.flatnonzero(raw_scores[:num_boxes] >= self.region_min_score)
                selected_indices = selected_indices[: self.max_regions]
                final_num = len(selected_indices)
                if "box_features" in data:
                    features[:final_num] = data["box_features"][selected_indices]
                if "bounding_boxes" in data:
                    boxes[:final_num] = data["bounding_boxes"][selected_indices]
                scores[:final_num] = raw_scores[selected_indices]
                if "objects" in data:
                    values = data["objects"][selected_indices]
                    object_labels[:final_num] = [str(item) for item in values]
                if "attr_obj" in data:
                    values = data["attr_obj"][selected_indices]
                    object_attributes[:final_num] = [str(item) for item in values]
                mask[:final_num] = 1.0

        if self.add_null_region:
            features = np.concatenate([features, np.zeros((1, self.region_feature_dim), dtype=np.float32)], axis=0)
            boxes = np.concatenate([boxes, np.zeros((1, 4), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.ones((1,), dtype=np.float32)], axis=0)
            scores = np.concatenate([scores, np.ones((1,), dtype=np.float32)], axis=0)
            object_labels.append("NULL")
            object_attributes.append("")

        return features, boxes, mask, scores, object_labels, object_attributes, image_size

    def _select_region_label(
        self,
        entity_text: str,
        boxes_by_name: Dict[str, List[List[int]]],
        candidate_boxes: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> int:
        name_key = self._normalize_name(entity_text)
        if name_key not in boxes_by_name:
            return self.max_regions if self.add_null_region else IGNORE_INDEX

        gt_boxes = torch.tensor(boxes_by_name[name_key], dtype=torch.float32)
        valid_count = int(candidate_mask.sum())
        if self.add_null_region and valid_count > 0:
            valid_count -= 1
        if valid_count == 0:
            return self.max_regions if self.add_null_region else IGNORE_INDEX

        candidates = torch.tensor(candidate_boxes[:valid_count], dtype=torch.float32)
        if candidates.numel() == 0 or gt_boxes.numel() == 0:
            return self.max_regions if self.add_null_region else IGNORE_INDEX

        ious = box_iou(gt_boxes, candidates)
        best_ious, _ = ious.max(dim=0)
        best_idx = int(best_ious.argmax().item())
        best_iou = float(best_ious.max().item())

        if best_iou >= self.grounding_iou_threshold:
            return best_idx
        return self.max_regions if self.add_null_region else IGNORE_INDEX

    def _select_region_targets(
        self,
        entity_text: str,
        boxes_by_name: Dict[str, List[List[int]]],
        candidate_boxes: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Return the CE target, positive mask, and continuous IoU quality."""

        positive_mask = np.zeros((candidate_mask.shape[0],), dtype=np.float32)
        iou_targets = np.zeros((candidate_mask.shape[0],), dtype=np.float32)
        null_index = self.max_regions if self.add_null_region else IGNORE_INDEX
        name_key = self._normalize_name(entity_text)
        if name_key not in boxes_by_name:
            if self.add_null_region:
                positive_mask[null_index] = 1.0
                iou_targets[null_index] = 1.0
            return null_index, positive_mask, iou_targets

        gt_boxes = torch.tensor(boxes_by_name[name_key], dtype=torch.float32)
        valid_count = int(candidate_mask.sum())
        if self.add_null_region and valid_count > 0:
            valid_count -= 1
        if valid_count == 0:
            if self.add_null_region:
                positive_mask[null_index] = 1.0
                iou_targets[null_index] = 1.0
            return null_index, positive_mask, iou_targets

        candidates = torch.tensor(candidate_boxes[:valid_count], dtype=torch.float32)
        if candidates.numel() == 0 or gt_boxes.numel() == 0:
            if self.add_null_region:
                positive_mask[null_index] = 1.0
                iou_targets[null_index] = 1.0
            return null_index, positive_mask, iou_targets

        ious = box_iou(gt_boxes, candidates)
        best_ious, _ = ious.max(dim=0)
        iou_targets[:valid_count] = best_ious.clamp(0.0, 1.0).to(dtype=torch.float32).numpy()
        best_iou = float(best_ious.max().item())
        if best_iou >= self.grounding_iou_threshold:
            positive_regions = best_ious >= self.grounding_iou_threshold
            positive_mask[:valid_count] = positive_regions.to(dtype=torch.float32).numpy()
            return int(best_ious.argmax().item()), positive_mask, iou_targets

        if self.add_null_region:
            positive_mask[null_index] = 1.0
            iou_targets[null_index] = 1.0
        return null_index, positive_mask, iou_targets

    def _build_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        iterator = self.records
        if self.use_dependency_graph:
            iterator = tqdm(self.records, desc="Building samples", disable=not sys.stderr.isatty())

        for record_idx, record in enumerate(iterator):
            record_id = record.get("id", record_idx)
            record_id_str = str(record_id)
            tokens: List[str] = record["tokens"]
            tags = self._convert_tags(record.get("ner_tags", []))
            fine_tags = record.get("fine_ner_tags")
            if not isinstance(fine_tags, list) or len(fine_tags) != len(tokens):
                fine_tags = None

            text = " ".join(tokens)
            encoding, word_ids = encode_words_with_alignment(
                self.tokenizer,
                tokens,
                max_length=self.max_length,
            )
            offsets = None
            if word_ids is not None:
                word_spans = self._build_word_spans(tokens)
                offsets = []
                for word_id in word_ids:
                    if word_id is None or word_id >= len(word_spans):
                        offsets.append((0, 0))
                    else:
                        offsets.append(word_spans[word_id])
            ner_labels: List[int] = []
            if word_ids is None:
                ner_labels = [IGNORE_INDEX] * len(encoding["input_ids"])
            else:
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        ner_labels.append(IGNORE_INDEX)
                    elif word_idx != previous_word_idx:
                        ner_labels.append(tags[word_idx] if word_idx < len(tags) else IGNORE_INDEX)
                    else:
                        ner_labels.append(IGNORE_INDEX)
                    previous_word_idx = word_idx

            attention_mask = encoding["attention_mask"]
            adjacency = None
            cache_path = None
            if self.graph_cache_dir is not None:
                record_id = record.get("id", record_idx)
                safe_id = str(record_id).replace("/", "_").replace("\\", "_")
                graph_signature = (
                    f"{text}|{attention_mask}|{self.graph_builder.config}"
                ).encode("utf-8")
                fingerprint = hashlib.sha1(graph_signature).hexdigest()[:12]
                cache_path = self.graph_cache_dir / f"{safe_id}_{fingerprint}.pt"
                if cache_path.exists():
                    try:
                        cached = torch.load(cache_path, map_location="cpu")
                        if isinstance(cached, torch.Tensor) and cached.size(0) == len(attention_mask):
                            adjacency = cached
                    except Exception:
                        adjacency = None

            if adjacency is None:
                adjacency = self.graph_builder.build(
                    text=text,
                    offsets=offsets,
                    attention_mask=attention_mask,
                )
                if cache_path is not None:
                    try:
                        torch.save(adjacency.cpu(), cache_path)
                    except Exception:
                        pass

            base_sample = {
                "sample_id": record_id_str,
                "record_id": record_id_str,
                "input_ids": encoding["input_ids"],
                "attention_mask": attention_mask,
                "token_type_ids": encoding.get("token_type_ids"),
                "adjacency": adjacency,
                "ner_labels": ner_labels,
                "image_path": str(self.image_dir / record["image"]),
                "tokens": tokens,
                "word_ids": word_ids,
                "text": text,
                "image_id": Path(record["image"]).stem,
                "ner_loss_weight": 1.0,
                "num_entities_in_record": 1,
            }

            if not self.grounding_enabled:
                base_sample["target_mask"] = [1 if value > 0 else 0 for value in attention_mask]
                samples.append(base_sample)
                continue

            image_id = Path(record["image"]).stem
            (
                region_features,
                region_boxes,
                region_mask,
                region_scores,
                region_object_labels,
                region_object_attributes,
                image_size,
            ) = self._load_region_features(image_id)
            boxes_by_name = self._read_xml_boxes(image_id)
            has_evidence_entities = isinstance(record.get("evidence_entities"), list)
            evidence_entities = self._extract_evidence_entities(tokens, record)
            if has_evidence_entities:
                entities = evidence_entities
            else:
                entities = [
                    (start, end, text, entity_type, subtype, False, None)
                    for start, end, text, entity_type, subtype in self._extract_entities(
                        tokens,
                        tags,
                        fine_tags=fine_tags,
                    )
                ]
            base_sample["num_entities_in_record"] = len(entities)

            if not entities:
                sample = dict(base_sample)
                sample["target_mask"] = [1 if value > 0 else 0 for value in attention_mask]
                sample["region_features"] = region_features
                sample["region_mask"] = region_mask
                sample["region_scores"] = region_scores
                sample["region_labels"] = self.max_regions if self.add_null_region else IGNORE_INDEX
                region_positive_mask = np.zeros((region_mask.shape[0],), dtype=np.float32)
                region_iou_targets = np.zeros((region_mask.shape[0],), dtype=np.float32)
                if 0 <= sample["region_labels"] < region_positive_mask.shape[0]:
                    region_positive_mask[sample["region_labels"]] = 1.0
                    region_iou_targets[sample["region_labels"]] = 1.0
                sample["region_positive_mask"] = region_positive_mask
                sample["region_iou_targets"] = region_iou_targets
                sample["region_boxes"] = region_boxes
                sample["image_size"] = image_size
                sample["gt_boxes_by_name"] = boxes_by_name
                sample["target_entity_type"] = "O"
                sample["target_subtype"] = "O"
                sample["target_subtype_id"] = IGNORE_INDEX
                sample["target_type_id"] = ENTITY_TYPE2ID["O"]
                sample["grounding_null_prior"] = 1.0 - 1e-4
                sample["region_object_labels"] = region_object_labels
                sample["region_object_attributes"] = region_object_attributes
                samples.append(sample)
                continue

            if not self.expand_entities_for_grounding:
                entities = entities[:1]

            for entity_idx, (
                start,
                end,
                entity_text,
                entity_type,
                entity_subtype,
                force_null_region,
                base_predicted_type,
            ) in enumerate(entities):
                target_mask = []
                if word_ids is None:
                    target_mask = [1 if value > 0 else 0 for value in attention_mask]
                else:
                    for word_idx in word_ids:
                        target_mask.append(1 if word_idx is not None and start <= word_idx < end else 0)

                if sum(target_mask) == 0:
                    target_mask = [1 if value > 0 else 0 for value in attention_mask]

                if force_null_region:
                    region_label = self.max_regions if self.add_null_region else IGNORE_INDEX
                    region_positive_mask = np.zeros((region_mask.shape[0],), dtype=np.float32)
                    region_iou_targets = np.zeros((region_mask.shape[0],), dtype=np.float32)
                    if self.add_null_region:
                        region_positive_mask[region_label] = 1.0
                        region_iou_targets[region_label] = 1.0
                else:
                    region_label, region_positive_mask, region_iou_targets = self._select_region_targets(
                        entity_text=entity_text,
                        boxes_by_name=boxes_by_name,
                        candidate_boxes=region_boxes,
                        candidate_mask=region_mask,
                    )

                sample = dict(base_sample)
                sample["ner_loss_weight"] = 1.0 / max(len(entities), 1)
                sample["target_mask"] = target_mask
                sample["region_features"] = region_features
                sample["region_mask"] = region_mask
                sample["region_scores"] = region_scores
                sample["region_labels"] = region_label
                sample["region_positive_mask"] = region_positive_mask
                sample["region_iou_targets"] = region_iou_targets
                sample["target_text"] = entity_text
                sample["target_start"] = start
                sample["target_end"] = end
                sample["target_entity_type"] = entity_type
                sample["target_subtype"] = entity_subtype
                sample["target_subtype_id"] = self.subtype_label2id.get(entity_subtype, IGNORE_INDEX)
                sample["target_type_id"] = ENTITY_TYPE2ID.get(entity_type, ENTITY_TYPE2ID["O"])
                sample["base_predicted_type"] = base_predicted_type
                sample["base_predicted_type_id"] = (
                    ENTITY_TYPE2ID.get(base_predicted_type, IGNORE_INDEX)
                    if base_predicted_type is not None
                    else IGNORE_INDEX
                )
                sample["grounding_null_prior"] = self._get_grounding_null_prior(entity_text, entity_type)
                sample["region_boxes"] = region_boxes
                sample["image_size"] = image_size
                sample["region_object_labels"] = region_object_labels
                sample["region_object_attributes"] = region_object_attributes
                sample["gt_boxes_by_name"] = boxes_by_name
                samples.append(sample)

        return samples

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]
