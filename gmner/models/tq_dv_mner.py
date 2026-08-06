"""Type-query dual-visual MNER with joint typed-span decoding."""

from __future__ import annotations

import bisect
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.config import GMNERConfig
from gmner.models.text_encoder import TextEncoder


class TextPreservingVisualResidual(nn.Module):
    """Zero-initialized visual residual that cannot overwrite text at init."""

    def __init__(
        self,
        text_size: int,
        visual_size: int,
        *,
        dropout: float,
        gate_initial_bias: float,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_size, visual_size)
        self.feature = nn.Sequential(
            nn.LayerNorm(visual_size * 4),
            nn.Linear(visual_size * 4, visual_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(visual_size, text_size)
        self.gate = nn.Linear(visual_size, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_initial_bias))

    def forward(
        self,
        text_state: torch.Tensor,
        visual_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.text_projection(text_state)
        visual = visual_state.expand_as(text)
        interaction = torch.cat(
            [text, visual, text * visual, (text - visual).abs()], dim=-1
        )
        hidden = self.feature(interaction)
        delta = torch.tanh(self.delta(hidden))
        gate = torch.sigmoid(self.gate(hidden))
        return gate * delta, gate.squeeze(-1)


class TQDualVisualMNER(nn.Module):
    """Four type queries jointly generate typed spans from Train/Dev records."""

    def __init__(self, config: GMNERConfig) -> None:
        super().__init__()
        if not bool(config.model.tq_enabled):
            raise ValueError("TQDualVisualMNER requires model.tq_enabled=true.")
        if int(config.model.tq_type_count) != 4:
            raise ValueError("TQ-DV-MNER currently requires four coarse types.")
        self.config = config
        self.use_clip = bool(config.model.tq_use_clip)
        self.use_vinvl = bool(config.model.tq_use_vinvl)
        self.visual_enabled = True
        self.text_encoder = TextEncoder(
            config.model.text_model_name,
            dropout=float(config.model.dropout),
        )
        text_size = int(self.text_encoder.hidden_size)
        visual_size = int(config.model.tq_visual_dim)
        heads = int(config.model.cross_attention_heads)
        if visual_size % heads:
            raise ValueError("tq_visual_dim must be divisible by attention heads.")

        self.clip_global_projection = nn.Sequential(
            nn.LayerNorm(int(config.model.tq_clip_feature_dim)),
            nn.Linear(int(config.model.tq_clip_feature_dim), visual_size),
        )
        self.clip_patch_projection = nn.Sequential(
            nn.LayerNorm(int(config.model.tq_clip_feature_dim)),
            nn.Linear(int(config.model.tq_clip_feature_dim), visual_size),
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(int(config.model.region_feature_dim)),
            nn.Linear(int(config.model.region_feature_dim), visual_size),
        )
        self.query_visual_projection = nn.Linear(text_size, visual_size)
        self.clip_retrieval = nn.MultiheadAttention(
            visual_size, heads, dropout=float(config.model.dropout), batch_first=True
        )
        self.region_retrieval = nn.MultiheadAttention(
            visual_size, heads, dropout=float(config.model.dropout), batch_first=True
        )
        self.visual_fusion = nn.Sequential(
            nn.LayerNorm(visual_size * 3),
            nn.Linear(visual_size * 3, visual_size),
            nn.GELU(),
        )
        self.visual_residual = TextPreservingVisualResidual(
            text_size,
            visual_size,
            dropout=float(config.model.dropout),
            gate_initial_bias=float(config.model.tq_gate_initial_bias),
        )
        self.existence_state = nn.Sequential(
            nn.LayerNorm(text_size * 2 + visual_size),
            nn.Linear(text_size * 2 + visual_size, text_size),
            nn.GELU(),
        )
        self.existence_head = nn.Linear(text_size, 1)
        self.existence_to_word = nn.Linear(text_size, text_size)
        self.word_norm = nn.LayerNorm(text_size)
        self.start_head = nn.Linear(text_size, 1)
        self.end_head = nn.Linear(text_size, 1)
        span_size = max(64, visual_size)
        self.span_start_projection = nn.Linear(text_size, span_size)
        self.span_end_projection = nn.Linear(text_size, span_size)
        self.qg_query_projection = nn.Linear(text_size, visual_size)

    def set_visual_enabled(self, enabled: bool) -> None:
        self.visual_enabled = bool(enabled)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        input_ids = batch["query_input_ids"]
        batch_size, type_count, query_length = input_ids.shape
        flat_shape = (batch_size * type_count, query_length)
        token_states, _ = self.text_encoder(
            input_ids=input_ids.reshape(flat_shape),
            attention_mask=batch["query_attention_mask"].reshape(flat_shape),
            token_type_ids=(
                None
                if "query_token_type_ids" not in batch
                else batch["query_token_type_ids"].reshape(flat_shape)
            ),
        )
        hidden_size = int(token_states.size(-1))
        token_states = token_states.reshape(
            batch_size, type_count, query_length, hidden_size
        )
        query_summary = _masked_mean(
            token_states, batch["query_token_mask"].unsqueeze(-1)
        )
        word_states = _gather_word_states(
            token_states,
            batch["query_first_subword_indices"],
            batch["query_word_mask"],
        )
        sentence_summary = _masked_mean(
            word_states, batch["query_word_mask"].unsqueeze(-1)
        )
        visual_state, region_states, query_region_logits = self._visual_states(
            query_summary=query_summary,
            batch=batch,
        )
        if self.visual_enabled and (self.use_clip or self.use_vinvl):
            visual_delta, visual_gate = self.visual_residual(
                word_states, visual_state.unsqueeze(2)
            )
        else:
            visual_delta = torch.zeros_like(word_states)
            visual_gate = word_states.new_zeros(word_states.shape[:-1])
        fused_words = word_states + visual_delta
        existence_state = self.existence_state(
            torch.cat([query_summary, sentence_summary, visual_state], dim=-1)
        )
        existence_logits = self.existence_head(existence_state).squeeze(-1)
        head_words = self.word_norm(
            fused_words + self.existence_to_word(existence_state).unsqueeze(2)
        )
        start_logits = self.start_head(head_words).squeeze(-1)
        end_logits = self.end_head(head_words).squeeze(-1)
        span_start = self.span_start_projection(head_words)
        span_end = self.span_end_projection(head_words)
        span_logits = torch.einsum(
            "btih,btjh->btij", span_start, span_end
        ) / math.sqrt(float(span_start.size(-1)))
        word_mask = batch["query_word_mask"].bool()
        start_logits = start_logits.masked_fill(~word_mask, -1e4)
        end_logits = end_logits.masked_fill(~word_mask, -1e4)
        span_logits = span_logits.masked_fill(
            ~batch["query_span_valid_mask"].bool(), -1e4
        )
        return {
            "query_token_states": token_states,
            "query_summary": query_summary,
            "word_states": word_states,
            "visual_state": visual_state,
            "visual_delta": visual_delta,
            "visual_gate": visual_gate,
            "existence_logits": existence_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "span_logits": span_logits,
            "region_states": region_states,
            "query_region_logits": query_region_logits,
        }

    def _visual_states(
        self,
        *,
        query_summary: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, type_count, _ = query_summary.shape
        query = self.query_visual_projection(query_summary).reshape(
            batch_size * type_count, 1, -1
        )
        clip_patches = self.clip_patch_projection(
            batch["clip_patch_features"]
        )
        clip_global = self.clip_global_projection(
            batch["clip_global_features"]
        )
        region_states = self.region_projection(batch["region_features"])
        region_real_mask = batch["region_mask"].bool() & ~batch[
            "region_is_null"
        ].bool()

        if self.use_clip and self.visual_enabled:
            clip_keys = _repeat_by_type(clip_patches, type_count)
            clip_mask = _repeat_by_type(
                batch["clip_patch_mask"].bool(), type_count
            )
            clip_context, _ = self.clip_retrieval(
                query,
                clip_keys,
                clip_keys,
                key_padding_mask=~clip_mask,
                need_weights=False,
            )
            clip_context = clip_context.reshape(batch_size, type_count, -1)
            global_context = clip_global.unsqueeze(1).expand(-1, type_count, -1)
        else:
            clip_context = query_summary.new_zeros(
                batch_size, type_count, self.config.model.tq_visual_dim
            )
            global_context = torch.zeros_like(clip_context)

        if self.use_vinvl and self.visual_enabled:
            safe_regions, safe_mask = _ensure_nonempty_regions(
                region_states, region_real_mask
            )
            region_keys = _repeat_by_type(safe_regions, type_count)
            repeated_mask = _repeat_by_type(safe_mask, type_count)
            region_context, _ = self.region_retrieval(
                query,
                region_keys,
                region_keys,
                key_padding_mask=~repeated_mask,
                need_weights=False,
            )
            region_context = region_context.reshape(batch_size, type_count, -1)
        else:
            region_context = torch.zeros_like(clip_context)

        visual = self.visual_fusion(
            torch.cat([clip_context, region_context, global_context], dim=-1)
        )
        qg_query = self.qg_query_projection(query_summary.detach())
        query_region_logits = torch.einsum(
            "bth,brh->btr", qg_query, region_states
        ) / math.sqrt(float(region_states.size(-1)))
        query_region_logits = query_region_logits.masked_fill(
            ~region_real_mask.unsqueeze(1), -1e4
        )
        return visual, region_states, query_region_logits

    @torch.no_grad()
    def decode(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> list[list[dict[str, Any]]]:
        existence = outputs["existence_logits"]
        start = outputs["start_logits"]
        end = outputs["end_logits"]
        match = outputs["span_logits"]
        valid = batch["query_span_valid_mask"].bool()
        decoded: list[list[dict[str, Any]]] = []
        for row in range(existence.size(0)):
            candidates: list[dict[str, Any]] = []
            for type_id in range(existence.size(1)):
                existence_probability = torch.sigmoid(existence[row, type_id])
                if float(existence_probability.item()) < float(
                    self.config.model.tq_existence_threshold
                ):
                    continue
                score_matrix = (
                    start[row, type_id].unsqueeze(1)
                    + end[row, type_id].unsqueeze(0)
                    + match[row, type_id]
                    + float(self.config.model.tq_existence_score_weight)
                    * F.logsigmoid(existence[row, type_id])
                ).masked_fill(~valid[row, type_id], -1e4)
                flat = score_matrix.flatten()
                count = min(
                    int(self.config.model.tq_decode_top_k_per_type),
                    int(valid[row, type_id].sum().item()),
                )
                if count <= 0:
                    continue
                values, indices = torch.topk(flat, count)
                width = int(score_matrix.size(1))
                for value, index in zip(values.tolist(), indices.tolist()):
                    if value <= float(self.config.model.tq_span_score_threshold):
                        continue
                    word_start = int(index // width)
                    word_end = int(index % width) + 1
                    candidates.append(
                        {
                            "span": [word_start, word_end],
                            "type_id": type_id,
                            "score": float(value),
                        }
                    )
            decoded.append(_weighted_interval_decode(candidates))
        return decoded

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kind": "tq_dv_mner",
            "format_version": 1,
            "independent_training": True,
            "clip_encoder_in_model": False,
            "clip_fully_frozen": True,
            "primary_metric": "dev_mner_f1",
            "test_accessed": False,
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _gather_word_states(
    token_states: torch.Tensor,
    indices: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    safe = indices.clamp_min(0)
    gather = safe.unsqueeze(-1).expand(-1, -1, -1, token_states.size(-1))
    result = token_states.gather(2, gather)
    return result * word_mask.unsqueeze(-1).to(result.dtype)


def _repeat_by_type(values: torch.Tensor, type_count: int) -> torch.Tensor:
    return values.unsqueeze(1).expand(-1, type_count, *values.shape[1:]).reshape(
        values.size(0) * type_count, *values.shape[1:]
    )


def _ensure_nonempty_regions(
    states: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_states = states.clone()
    safe_mask = mask.clone()
    empty = ~safe_mask.any(dim=-1)
    if empty.any():
        safe_states[empty, 0] = 0.0
        safe_mask[empty, 0] = True
    return safe_states, safe_mask


def _weighted_interval_decode(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["span"][1]),
            int(item["span"][0]),
            int(item["type_id"]),
            -float(item["score"]),
        ),
    )
    ends = [int(item["span"][1]) for item in ordered]
    best_scores = [0.0]
    best_sets: list[list[dict[str, Any]]] = [[]]
    for index, candidate in enumerate(ordered):
        start = int(candidate["span"][0])
        previous = bisect.bisect_right(ends, start, hi=index) - 1
        include_score = best_scores[previous + 1] + float(candidate["score"])
        include_set = best_sets[previous + 1] + [candidate]
        exclude_score = best_scores[-1]
        exclude_set = best_sets[-1]
        if include_score > exclude_score + 1e-8:
            best_scores.append(include_score)
            best_sets.append(include_set)
        elif exclude_score > include_score + 1e-8:
            best_scores.append(exclude_score)
            best_sets.append(exclude_set)
        else:
            include_key = _prediction_signature(include_set)
            exclude_key = _prediction_signature(exclude_set)
            best_scores.append(include_score)
            best_sets.append(include_set if include_key < exclude_key else exclude_set)
    return sorted(
        best_sets[-1],
        key=lambda item: (
            int(item["span"][0]), int(item["span"][1]), int(item["type_id"])
        ),
    )


def _prediction_signature(items: list[dict[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted(
            (
                int(item["span"][0]),
                int(item["span"][1]),
                int(item["type_id"]),
            )
            for item in items
        )
    )
