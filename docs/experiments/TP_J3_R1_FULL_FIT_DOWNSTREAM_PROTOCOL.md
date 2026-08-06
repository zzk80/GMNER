# TP J3-r1 Full-fit Downstream Rebuild

Status: completed. `METHOD_NO_GO_TEST_GENERALIZATION`.

The full-fit downstream rebuild and its single frozen Test evaluation are
complete. Dev GMNER improved from `0.6213161082` to `0.6295095257`, but Test
GMNER decreased from `0.6152941176` to `0.6086956522`. The rebuilt chain is
rejected and formal M3.3A remains unchanged. See
`TP_J3_R1_FULL_FIT_DOWNSTREAM_RESULT.md` and
`tp_j3_r1_full_fit_downstream_result.json`.

## Fixed Inputs

- Stage1: J3-r1 Seed43 best checkpoint, selected on Dev before downstream work.
- Candidate policy: the unchanged M3.3A R16/R36 settings.
- Downstream architecture, losses, decode thresholds, and checkpoint selection:
  unchanged from M3.3A.
- Training cache: full-fit Train predictions, not OOF.
- Checkpoint selection: Dev only.
- Test: one execution after all stages and Test cache commands are frozen.

The experiment must not select a Stage1 seed, downstream checkpoint, decode
threshold, or architecture using Test. The reported Test result is accepted
without a parameter change or rerun.

## Chain

```text
J3-r1 Seed43 Stage1
-> independent Train/Dev R16
-> anchored Train/Dev R36
-> Hierarchical Record Verifier
-> Coarse Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> freeze
-> build Test R16/R36
-> one-time Test evaluation
```

All artifacts use the `tp_j3_r1_seed43` namespace and must not overwrite the
formal M3.3A caches or checkpoints.
