# M3.3A-P3 Conditional Same-Type Region Resolver

## Status

This branch is the Seed42 C1 MVP selected by the Dev-only M3.3A
assignment Oracle. It changes only the real-region index for selected,
final-visible PER entities in records containing at least two such
entities.

It does not change:

- selected spans;
- coarse entity types;
- final visibility or NULL decisions;
- the per-entity Fine candidate set;
- Hierarchical, Coarse, Fine, or Evidence model parameters.

Test data is not exposed by either the training or evaluation entry
point.

## Frozen Input Contract

For each entity, C1 uses its existing full Fine mask:

```text
Base Top-8 + Learned Top-8 -> 16 real candidates
```

There is no Top-K union between entities and NULL is excluded. Other PER
entities contribute competition probabilities only; they cannot add a
region to the current entity's candidate set.

The resolver reuses the frozen 256-dimensional
`span_grounding_state` and `region_grounding_state`. Its residual head
receives:

```text
span
region
span * region
abs(span - region)
11-dimensional competition projection
```

The final layer is zero initialized, so epoch 0 exactly reproduces the
frozen M3.3A chain.

## Registered Decisions

```text
alpha/residual_scale = 1.0
C1 override margin   = 0.0 with strict gain > 1e-6
C2 override margin   = 0.2
```

C2 is allowed only when C1 has positive GMNER net correction while
damaging more than five base-correct trigger triples. No other override
margin is registered.

The loss is:

```text
1.0 * multi-positive correction
+ 1.0 * preservation KL
+ 0.5 * preservation margin
+ 0.05 * residual L1
```

Candidate-missing samples do not enter correction supervision.

## Dev Baseline Gate

Before training, epoch 0 must reproduce:

```text
Dev GMNER = 0.6213161081953977
override count = 0
span/type/visibility/selected-span changes = 0
candidate and NULL violations = 0
```

Run the disabled resolver check:

```bash
PYTHONPATH=. python scripts/evaluate_same_type_region_resolver.py \
  --config configs/fmnerg_twitter10000_same_type_region_resolver_c1.yaml \
  --disable-resolver \
  --output outputs/fmnerg_roberta128_same_type_region_resolver_c1/dev_disabled.json
```

Train C1:

```bash
PYTHONPATH=. python scripts/train_same_type_region_resolver.py \
  --config configs/fmnerg_twitter10000_same_type_region_resolver_c1.yaml
```

Evaluate the selected checkpoint:

```bash
PYTHONPATH=. python scripts/evaluate_same_type_region_resolver.py \
  --config configs/fmnerg_twitter10000_same_type_region_resolver_c1.yaml \
  --checkpoint outputs/fmnerg_roberta128_same_type_region_resolver_c1/best_model.pt \
  --output outputs/fmnerg_roberta128_same_type_region_resolver_c1/dev.json
```

The formal method decision must use correction/damage counts, not only
GMNER:

```text
GMNER net correction > 0
TO_REAL corrected > damaged
non-trigger changes = 0
candidate/NULL violations = 0
span/type/visibility identity exact
```

## Seed42 C1 Result

The preregistered Seed42 run completed on Dev without accessing Test.
Epoch 0 exactly reproduced the frozen M3.3A chain, and early stopping
selected epoch 0 after epochs 1-3 produced no Dev override.

```text
Frozen baseline GMNER = 0.6213161081953977
C1 best GMNER         = 0.6213161081953977
GMNER delta           = 0
corrected / damaged   = 0 / 0
override count        = 0
trigger count         = 190
Dev correction targets / preservation targets = 40 / 97
```

All frozen-output and candidate-contract checks passed:

```text
non-trigger region changes = 0
span/type/visibility changes = 0
candidate-contract violations = 0
NULL-candidate violations = 0
base-correct trigger preservation = 100%
test_accessed = false
```

The current in-sample Train cache contains only 9 correction examples
against 815 preservation examples among 824 valid trigger examples.
This mismatch explains why the residual remained conservative, but it
does not alter the registered decision:

```text
C1 engineering implementation = pass
C1 Seed42 method gate         = no-go
C2 margin 0.2                 = not triggered
formal M3.3A chain            = unchanged
```
