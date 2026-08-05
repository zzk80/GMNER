# Final-chain OOF Source Inventory

**Status:** `BLOCKED_NO_VALID_SOURCE`

This inventory is metadata-only. It did not deserialize model, cache,
dataset, Dev, or Test payloads and did not compute Oracle labels.

| Source | Status | Key blocker |
| --- | --- | --- |
| `compact_null_release_full_chain_oof` | `INCOMPLETE` | word_space_final_spans, final_m33a_prediction_identity, final_coarse_type_logits, stage1_span_type_identity, ... |
| `d1_strict_stage1_oof` | `INCOMPLETE` | final_m33a_prediction_identity, final_coarse_type_logits, final_region_null_decision, r36_candidate_identity_scores, ... |
| `p4_r0b_regenerated_full_chain_oof` | `SEMANTICALLY_INVALID` | semantic mismatch in folds 2, 4, 5, 7 |
| `other_full_chain_fold_artifacts` | `MISSING` | artifact set not found |

## Decision

No historical source satisfies the frozen minimum row contract.
B1/A1 population materialization and training remain unauthorized.
The only admissible next step is a newly generated single-fold
full-chain OOF dry run under the frozen protocol.
