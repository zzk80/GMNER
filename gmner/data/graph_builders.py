"""Graph construction utilities for text/image nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torchvision.ops import box_iou


@dataclass
class GraphBuilderConfig:
    use_dependency_graph: bool = False
    dependency_backend: str = "spacy"
    dependency_model: str = "en_core_web_sm"
    window_size: int = 2


class TextGraphBuilder:
    """Builds token-level adjacency matrices with optional dependency edges."""

    def __init__(self, config: GraphBuilderConfig):
        self.config = config
        self._parser_loaded = False
        self._parser = None

    def _load_parser(self):
        if self._parser_loaded:
            return self._parser

        self._parser_loaded = True
        if not self.config.use_dependency_graph:
            return None

        if self.config.dependency_backend.lower() != "spacy":
            return None

        try:
            import spacy

            self._parser = spacy.load(self.config.dependency_model)
        except Exception:
            self._parser = None

        return self._parser

    @staticmethod
    def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) < min(a_end, b_end)

    @staticmethod
    def _normalize(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency / degree

    def _build_local_graph(self, attention_mask: Sequence[int], seq_len: int) -> torch.Tensor:
        adjacency = torch.eye(seq_len, dtype=torch.float32)
        valid_indices = [idx for idx, value in enumerate(attention_mask) if value > 0]
        window_size = max(1, self.config.window_size)

        for i, src in enumerate(valid_indices):
            for j in range(max(0, i - window_size), min(len(valid_indices), i + window_size + 1)):
                dst = valid_indices[j]
                adjacency[src, dst] = 1.0
                adjacency[dst, src] = 1.0

        return adjacency

    def build(
        self,
        text: Optional[str],
        offsets: Optional[Sequence[Tuple[int, int]]],
        attention_mask: Sequence[int],
    ) -> torch.Tensor:
        seq_len = len(attention_mask)
        adjacency = self._build_local_graph(attention_mask=attention_mask, seq_len=seq_len)

        parser = self._load_parser()
        if parser is None or not text or not offsets:
            return self._normalize(adjacency)

        try:
            doc = parser(text)
            token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]

            token_to_subwords: List[List[int]] = []
            for span_start, span_end in token_spans:
                aligned: List[int] = []
                for subword_idx, (sub_start, sub_end) in enumerate(offsets):
                    if sub_end <= sub_start:
                        continue
                    if self._overlap(span_start, span_end, sub_start, sub_end):
                        aligned.append(subword_idx)
                token_to_subwords.append(aligned)

            for token in doc:
                src_subwords = token_to_subwords[token.i]
                dst_subwords = token_to_subwords[token.head.i]
                for src in src_subwords:
                    for dst in dst_subwords:
                        if src < seq_len and dst < seq_len:
                            adjacency[src, dst] = 1.0
                            adjacency[dst, src] = 1.0
        except Exception:
            pass

        return self._normalize(adjacency)


def build_image_adjacency(
    batch_size: int,
    num_nodes: int,
    device: torch.device,
    boxes: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if boxes is None:
        adjacency = torch.ones((batch_size, num_nodes, num_nodes), device=device, dtype=torch.float32)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency / degree

    boxes = boxes.to(device=device, dtype=torch.float32)
    if mask is not None:
        mask = mask.to(device=device)

    adjacency = torch.eye(num_nodes, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    for batch_idx in range(batch_size):
        if mask is None:
            valid = torch.ones((num_nodes,), dtype=torch.bool, device=device)
        else:
            valid = mask[batch_idx] > 0

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 1:
            continue

        valid_boxes = boxes[batch_idx, valid_idx]
        if valid_boxes.numel() == 0:
            continue

        ious = box_iou(valid_boxes, valid_boxes)
        adj_valid = (ious >= iou_threshold).float()
        adj_valid.fill_diagonal_(1.0)
        adjacency[batch_idx][valid_idx[:, None], valid_idx[None, :]] = adj_valid

    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree
