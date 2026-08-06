# TP J3-r1 Full-fit Downstream Result

Status: `METHOD_NO_GO_TEST_GENERALIZATION`

## Experiment Contract

- Training regime: ordinary full-fit Train training with Dev checkpoint selection.
- OOF: not used.
- Stage1: J3-r1 Seed43, selected before downstream rebuilding.
- Downstream chain: unchanged M3.3A R16/R36, Hierarchical Verifier, Coarse
  Selector, Fine Adapter, and Evidence Visibility.
- Test policy: one frozen evaluation after architecture, checkpoints, caches,
  and thresholds were sealed. No Test-driven tuning or rerun is permitted.

## Dev Result

| Metric | Formal M3.3A | J3-r1 rebuilt chain | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.8728300363 | 0.8747466559 | +0.0019166195 |
| MNER F1 | 0.8167137667 | 0.8204296717 | +0.0037159050 |
| EEG F1 | 0.6608800969 | 0.6672071342 | +0.0063270373 |
| GMNER F1 | 0.6213161082 | 0.6295095257 | +0.0081934175 |

The rebuilt Dev chain progressed from Stage1 `0.6198263679`, through
Hierarchical `0.6218078638` and Fine `0.6278881232`, to final Evidence
Visibility `0.6295095257` GMNER.

## One-time Test Result

| Metric | Formal M3.3A | J3-r1 rebuilt chain | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.8698039216 | 0.8707456227 | +0.0009417011 |
| MNER F1 | 0.8184313725 | 0.8184143223 | -0.0000170502 |
| EEG F1 | 0.6521568627 | 0.6460751525 | -0.0060817102 |
| GMNER F1 | 0.6152941176 | 0.6086956522 | -0.0065984654 |

The Evidence head improved its Test Fine baseline from `0.6059413732` to
`0.6086956522` GMNER (`+0.0027542790`, net `+7` triples). The failure is
therefore not primarily caused by the final Evidence head.

The new Stage1 Test GMNER was `0.5936520376`, compared with the old Stage1
value `0.5916813122` (`+0.0019707254`). The Stage1 signal generalized weakly,
but the retrained intermediate grounding chain erased it: the new Fine
baseline `0.6059413732` was below the old formal Fine baseline
`0.6133333333` by `0.0073919601`.

## Decision

The J3-r1 full-fit downstream chain is rejected because its one-time Test
GMNER is below the formal M3.3A result despite a strong Dev gain. The formal
best remains M3.3A with Test GMNER `0.6152941176`.

This Test result is frozen. It must not be used to select another seed,
checkpoint, threshold, architecture, or rerun of the same experiment.
