# GMNER / FMNERG

This repository contains two frozen formal systems for Twitter10000:

```text
Model-G: M3.3A -> GMNER
Model-F: F3 subtype encoder on frozen M3.3A predictions -> FMNERG
```

Historical experiments are not part of the formal prediction path. Their
terminal conclusions are indexed in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

## Formal Results

| Split | Span F1 | MNER F1 | Fine MNER F1 | EEG F1 | GMNER F1 | FMNERG F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 0.872830 | 0.816714 | 0.68039 +/- 0.00297 | 0.660880 | 0.621316 | 0.52052 +/- 0.00219 |
| Test | 0.869804 | 0.818431 | 0.66510 +/- 0.00160 | 0.652157 | 0.615294 | 0.50431 +/- 0.00111 |

F3 values are mean +/- sample standard deviation over preregistered seeds
41/42/43. Test was not used for checkpoint or threshold selection.

## Model-G: M3.3A

```text
RoBERTa Typed-BIO Stage1
-> R16 formal candidates
-> R36 expanded regions
-> Hierarchical Record Verifier
-> Coarse Region Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> record-level decode
```

Formal configurations:

```text
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
```

Formal checkpoints:

```text
outputs/fmnerg_stage1_roberta128/best_model.pt
outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt
outputs/fmnerg_roberta128_coarse_selector/best_model.pt
outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt
outputs/fmnerg_roberta128_evidence_visibility/best_model.pt
```

Evaluate the frozen endpoint:

```bash
PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split dev
```

Architecture details are in
[`docs/HIERARCHICAL_RECORD_VERIFIER.md`](docs/HIERARCHICAL_RECORD_VERIFIER.md).

## Model-F: F3

F3 predicts one of 51 subtypes for each frozen Model-G entity. It cannot
change span, coarse type, region, NULL, ordering, EEG, or GMNER.

```text
frozen M3.3A entity
-> start/end/mean RoBERTa span representation
-> parent-masked subtype encoder
-> Fine MNER / FMNERG
```

Winner configuration:

```text
sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml
```

Formal protocol and commands are documented in
[`sidecars/fmnerg_subtype/README.md`](sidecars/fmnerg_subtype/README.md).

## Final-Chain OOF Research Phase

The repository now contains the code and contracts used to generate a strict
10-fold final-chain OOF Train population. Every record was held out from all
five supervised M3.3A stages.

```text
records                         7,000
record IDs unique               7,000
all folds sealed                true
all replay digests exact        true
NaN / Inf                       0
Dev / Test accessed             false

B1 exact-span rows              10,259
B1 base-wrong rows                 875
raw replacement actions         39,063
strict A1 boundary actions      31,138
strict A1 FIX / NEUTRAL / DAMAGE
                                286 / 4,128 / 26,724
```

The source mother set is retained outside Git:

```text
knowledge/final_chain_oof/ten_fold_population/gold_free_rows.jsonl
knowledge/final_chain_oof/ten_fold_population/supervision_sidecar.jsonl
```

### B1-T0

The text-only exact-span type corrector learned moderate ranking and target
type signals, but folds 0-7 could not freeze a stable action threshold under
the required precision and preservation Gates. All three seeds therefore
executed zero locked actions on folds 8-9.

```text
B1-T0 = NO_GO / SEALED
```

### A1-T0

The observable-tabular grouped boundary selector used the strict
`286 / 31,138` population. Candidate source improved diagnostic AUPRC, but no
seed produced a development prefix satisfying cross-fold coverage, precision,
positive net correction, and formal preservation. The one-time folds 8-9
evaluation executed zero actions for all three seeds.

```text
A1-T0 = NO_GO / SEALED
Observable post-hoc correction = CLOSED
```

The complete phase record is in
[`docs/experiments/FINAL_CHAIN_OOF_POSTHOC_PHASE_SUMMARY.md`](docs/experiments/FINAL_CHAIN_OOF_POSTHOC_PHASE_SUMMARY.md).

## Next Research Boundary

The next candidate direction is not another post-hoc controller. The proposed
architecture moves risk supervision before final decoding:

```text
Typed-BIO candidate lattice
-> hypothesis-conditioned text evidence
-> base-candidate counterfactual comparison
-> risk-aware structured set decoding
-> optional visual/region expansion only after text-only validation
```

The phased J0-J3 roadmap is documented in
[`docs/experiments/CANDIDATE_CONDITIONED_STRUCTURED_DECODER_ROADMAP.md`](docs/experiments/CANDIDATE_CONDITIONED_STRUCTURED_DECODER_ROADMAP.md).

J0-A is now complete. A gold-free R36 typed-span lattice was sealed before
Train supervision was attached. Under the frozen final budget (Top-4 per
group, 32 non-control hypotheses per record, non-overlap, at most one ADD),
the exact OOF Oracle gained 973 correct MNER entities, equivalent to 208.5 per
1500 records. The preregistered continuation floor was 308 OOF gains, or 66
per 1500 records.

```text
J0-A candidate capacity       PASSED
J0-B latent rematerialization NOT AUTHORIZED
J1 training                   NOT AUTHORIZED
Dev / Test                    LOCKED
```

The compact result is in
[`docs/experiments/j0_a_candidate_lattice_oracle_result.json`](docs/experiments/j0_a_candidate_lattice_oracle_result.json).
Generated lattice and supervision rows remain outside Git under
`knowledge/candidate_conditioned_decoder/j0_a/`.

## Retained Visual Controls

DVH and TQ-DV are archived Dev controls, not formal models. DVH reached MNER
`0.799355`; TQ-DV reached `0.810275`. Fixed-span TQ-DV replay improved MNER by
only seven correct typed spans (`0.814740 -> 0.817559`) and did not justify a
downstream rebuild. Their compact method records remain under
`docs/experiments/`.

TP/J3 and PA1 implementation snapshots are retained through the immutable
`archive/tp-clip-j3-20260806` and
`archive/protected-region-mner-pa1-20260806` tags, not through active branches
or the formal runtime surface.

## Repository Layout

```text
gmner/       formal models plus reproducible audit components
scripts/     training, evaluation, cache, and audit entry points
configs/     formal Model-G and retained control configurations
sidecars/    formal F3 subtype implementation
tools/       orchestration and resource monitors
tests/       regression and protocol-contract tests
docs/        formal results, compact archives, and research protocols
```

Pretrained models, checkpoints, caches, images, and generated OOF tensors are
excluded from Git. The formal runtime paths and result-inspection commands are
listed in
[`docs/experiments/CURRENT_CHAIN_RUNBOOK.md`](docs/experiments/CURRENT_CHAIN_RUNBOOK.md).

## Environment And Validation

```bash
conda activate gmner
pip install -r requirements.txt
export PYTHONPATH=.
python -m pytest -q
```

Formal result selection rules are recorded in
[`docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md`](docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md).
