# Scene Analyzer Audit

**Status:** `NO_GO`
**Scope:** Dev only
**Formal M3.3A decoder changed:** no
**Test accessed:** no

## Corrected conclusion

The historical Scene Analyzer result of `1.0000` accuracy is invalid as a
deployable result. `scripts/train_scene_analyzer.py` used the gold entity count
both as an input feature and as the single/multi label:

```text
feature: num_entities = len(gold_entities)
label:   multi iff len(gold_entities) > 1
```

The result is deterministic target leakage, not scene recognition. The script
now requires an explicit `--allow-gold-count-leakage-audit` flag and labels the
mode as a historical audit.

## Leakage-free contract

Scene routing uses the frozen formal M3.3A prediction only:

```text
single_or_empty: predicted entity count <= 1
multi:           predicted entity count >= 2
```

Gold entity count is retained only for Dev validation and never participates
in `predicted_scene`.

On 1,500 Dev records:

| Gold / predicted | single_or_empty | multi |
| --- | ---: | ---: |
| single_or_empty | 656 | 92 |
| multi | 74 | 678 |

```text
Accuracy = (656 + 678) / 1500 = 0.889333
Required Gate = 0.950000
Gate = failed
```

The gold distribution is:

```text
zero entities: 87
one entity:    661
multi entity: 752
```

The `single_or_empty` definition is used for routing every record. Diagnostic
reports must still show zero-entity and exactly-one-entity strata separately.

## Strict OOF check

The deployable feature hypothesis was also tested with all 7,000 strict
10-fold full-chain OOF Train records. Logistic regression, tree models, and a
text TF-IDF plus deployment-statistics classifier did not improve the formal
Dev result beyond `0.889333`.

This confirms that the gap is not fixed by replacing the deterministic count
rule with the originally proposed lightweight classifier.

## Decision

The preregistered Scene Analyzer gate requires accuracy at least `0.95`.
Because the leakage-free result is `0.889333`:

1. Scene-conditioned decoding is not connected to the formal evaluator.
2. The 96/1,215 threshold combinations are not searched.
3. No Test artifact is read.
4. Formal Dev GMNER remains `0.621316`.
5. Formal Test GMNER remains `0.615294`.

This avoids fitting six decode thresholds to routing errors that already fail
the first gate.

## Reproduction

Generate and validate the Dev routing artifact:

```bash
PYTHONPATH=. python scripts/generate_scene_predictions.py \
  --input outputs/diagnostics/m33a_entity_count/records.jsonl \
  --output outputs/scene_analyzer/dev_scene_predictions.jsonl \
  --report outputs/scene_analyzer/validation_report.json \
  --expected-records 1500 \
  --required-accuracy 0.95

PYTHONPATH=. python scripts/validate_scene_predictions.py \
  --predictions outputs/scene_analyzer/dev_scene_predictions.jsonl \
  --output-report outputs/scene_analyzer/validation_report.recheck.json \
  --expected-records 1500 \
  --required-accuracy 0.95
```

The generator exits successfully because the artifact itself is valid. The
method decision is recorded in `report["gate"]["passed"]`, which is `false`.
