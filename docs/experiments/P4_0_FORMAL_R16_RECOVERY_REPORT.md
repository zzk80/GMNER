# P4.0 Phase B Formal R16 Recovery Report

## Status

```text
P4.0 Phase A                         PASS
P4.0 Phase B formal artifact set    BLOCKED
Source manifest                     BLOCKED_UNSEALED
Formal-span sidecars                NOT GENERATED
Preservation validation             NOT RUN
Promotion score                     UNFROZEN
Oracle                              NOT RUN
Folds 8-9                           LOCKED
Dev                                 LOCKED
P4.1                                NOT AUTHORIZED
Test                                LOCKED
```

The formal status is:

```text
P4_0_FORMAL_ARTIFACT_RECOVERY_BLOCKED
```

## Exact Recovery Result

The archive proof contains one authoritative `artifact_sha256.formal_cache`
value for each development fold. Searches covered the retained local
workspace, the local OOF checkpoint archive, other local drives, and the
known `server4090` backup locations.

No direct `heldout_r16.pt` file or archive containing one was found. Therefore:

```text
folds with an exact SHA256 match     0 / 8
missing exact artifact folds        0,1,2,3,4,5,6,7
candidate files hashed              0
candidate payloads deserialized     0
formal-span sidecars generated      0
```

The retained fold manifests explicitly describe hash-only artifact retention
after compact heldout feature validation and cleanup. They do not record a
backup destination for the deleted formal R16 caches.

## Safety Gate

The Phase B implementation hashes every candidate before `torch.load`.
Recovery is atomic across folds 0-7: if one exact hash is absent, no candidate
payload is deserialized. Calibration folds, Dev, Test, labels, and Oracle
statistics are not opened.

The following substitutions remain prohibited:

```text
D1 span coordinates
cross-cache candidate row indices
approximate decode
newly regenerated artifacts without separate authorization
```

## Provenance

```text
generator commit:
2920dea879bc1dfe483b7aba1449f1582dce024d

machine report canonical digest:
77dd70ebb0dd2eb6ea3eaee6d2c9a298fd2a10b6ea1399006974339d98dca2aa

machine report file SHA256:
4025c507afe2a759e3bd006792b39627c1ea7e262cf6354cf9eb7764473c6eae

external search inventory SHA256:
4c5787292401feb62563aeec4776c769ad29e523bcdab28af31def1a817534f7
```

The machine-readable report is
`docs/experiments/p4_0_formal_r16_recovery_report.json`.

## Decision

P4.0 cannot proceed to source-manifest sealing or actionability Oracle with
the retained artifacts. Continuing requires a separately authorized,
reproducible regeneration of the full-chain OOF formal R16 source. Until then,
the correct terminal state is `P4_0_FORMAL_ARTIFACT_RECOVERY_BLOCKED`.
