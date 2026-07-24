# Knowledge Assets

This directory stores train-only statistics and semantic prototype assets.

```text
offline/
  entity_occurrences.jsonl
  mention_inventory.jsonl
  ambiguous_mentions.jsonl

grounding/
  groundability_by_type.jsonl
  groundability_by_mention_type.jsonl

semantic/
  stage1_span_features.pt
  semantic_prototypes.pt
  semantic_prototypes.json
```

`semantic_prototypes.pt` must be built from the best Stage 1 checkpoint and train gold spans. Dev/test entities must never participate in prototype construction.

