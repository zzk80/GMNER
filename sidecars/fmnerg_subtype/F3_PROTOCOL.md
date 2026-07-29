# FMNERG F3-P1 Controlled Learning-Rate Protocol

## Status

F3-P1 passed its preregistered Dev Gate. The frozen winner is
`lr6_lower_double`, which changes only the lower RoBERTa learning rate from
`1e-6` to `2e-6`.

Dev selection did not access Test and did not modify the frozen GMNER chain.

## Fixed Baseline

F3 starts from the accepted F2 full-encoder model family:

```text
Train: gold span + gold parent
Dev: formal predicted span + predicted parent
Encoder scope: all RoBERTa layers
Subtype head: shared_hard
Selection metric: Dev FMNERG F1
```

The paired F2 Dev FMNERG baselines are immutable:

| Seed | F2 FMNERG F1 |
| ---: | ---: |
| 41 | 0.5171578522406137 |
| 42 | 0.5183689947517158 |
| 43 | 0.5163504238998788 |
| Mean | 0.5172924236307361 |

F2 is not retrained during F3. The formal Dev GMNER value must remain exactly
`0.6213161081953977` within the configured numerical tolerance.

## P1 Candidates

The F2 learning rates are:

```text
subtype head:   1e-4
upper backbone: 5e-6
lower backbone: 1e-6
```

Each P1 candidate changes exactly one group by either `0.5x` or `2.0x`.
Architecture, data, seed protocol, loss, epochs, warmup, early stopping, and
all other learning rates remain fixed.

The authoritative contract is:

```text
sidecars/fmnerg_subtype/f3_p1_protocol.yaml
```

The six candidate configurations are under:

```text
sidecars/fmnerg_subtype/configs/f3_p1_*.yaml
```

## Selection

All six candidates first run only with Seed42. A candidate passes screening
when:

```text
candidate FMNERG - paired F2 Seed42 FMNERG >= 0.002
```

At most one candidate advances. The largest paired delta wins. Candidates
within `0.000404` of the top delta are resolved by the preregistered
conservative order in the machine protocol. No result-dependent second
candidate may advance.

The winner then runs Seeds 41 and 43; its existing Seed42 result is reused.
The three-seed Gate passes only when all conditions hold:

```text
mean paired FMNERG delta >= 0.003
every paired FMNERG delta > 0
population std of paired deltas <= 0.002
GMNER identity exact for every seed
formal Stage1 not mutated
test_accessed = false
```

Passing P1 freezes F3 and stops model selection. Failure advances the research
protocol to P2; P1, P2, and P3 deltas are never added together.

## Frozen Dev Result

Seed42 was the only screening seed. Only `lr6_lower_double` reached the
required `+0.002` paired improvement:

```text
Seed42 FMNERG = 0.5211949939442875
paired delta = +0.0028259991925716488
```

The paired three-seed confirmation result is:

| Seed | FMNERG F1 | Paired delta | Fine MNER F1 |
| ---: | ---: | ---: | ---: |
| 41 | 0.522809850626 | +0.005651998385 | 0.683891804602 |
| 42 | 0.521194993944 | +0.002825999193 | 0.680662091239 |
| 43 | 0.517561566411 | +0.001211142511 | 0.676624949536 |

```text
mean FMNERG = 0.520522136994
mean paired delta = +0.003229713363
paired-delta population std = 0.001835309070
mean Fine MNER = 0.680392948459
GMNER = 0.621316108195 for every seed
test_accessed = false
```

All Gate checks passed. P2 and P3 model selection are stopped.

## Execution

Cloud Linux:

```bash
cd ~/gmner

nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cuda \
  bash tools/run_fmnerg_subtype_f3_p1.sh \
  > fmnerg_subtype_f3_p1.log 2>&1 &

echo $!
tail -f fmnerg_subtype_f3_p1.log
```

The runner is serial and restartable. It skips only summaries that satisfy the
expected seed, full-encoder scope, candidate-config SHA-256, frozen-GMNER
identity, and no-Test contract.

Outputs:

```text
outputs/fmnerg_subtype_f3_p1/
  <candidate>/seed42/
  screen_seed42.json
  <winner>/seed41/
  <winner>/seed43/
  final_dev_summary.json
```

If no Seed42 candidate passes, `screen_seed42.json` is the terminal P1 result
and no additional seed is trained.

## Test Boundary

The current runner has no Test path and must not be extended in place.

The frozen Test protocol is:

```text
sidecars/fmnerg_subtype/f3_final_test.yaml
```

It reuses the existing atomic one-time runner. The runner must execute from
the exact `f3-dev-frozen` tag with a clean tracked worktree. It writes the
access seal before loading any Test artifact, records the known F2 Test result
as prior information, and forbids a completed rerun.

## Frozen Test Result

The one-time Test run completed under the frozen tag and protocol:

| Seed | Fine MNER F1 | FMNERG F1 | Paired FMNERG delta vs F2 |
| ---: | ---: | ---: | ---: |
| 41 | 0.663137254902 | 0.502745098039 | +0.000784313725 |
| 42 | 0.667058823529 | 0.505098039216 | +0.005490196078 |
| 43 | 0.665098039216 | 0.505098039216 | +0.002352941176 |

```text
Fine MNER = 0.665098039216 +/- 0.001600973688
FMNERG = 0.504313725490 +/- 0.001109187108
mean FMNERG delta vs F2 = +0.002875816993
MNER = 0.818431372549
EEG = 0.652156862745
GMNER = 0.615294117647
test_accessed = true
F3 test_access_count = 1
repository_test_access_count = 2
```

No seed was selected using Test. The completed access seal forbids rerunning
the method. The compact frozen result is stored at:

```text
sidecars/fmnerg_subtype/f3_final_test_result.json
```

## OOF Boundary

F3-P1 is not an OOF experiment. It uses the accepted F2 supervised contract:
gold spans and gold parents on Train, formal predictions on Dev. Existing
retained full-chain OOF features do not contain the span boundaries required
to train this encoder path and are not relabeled as F3 training data.
