# M3.3A Entity Count Diagnostic Analysis

**Date:** 2024-07-27  
**Script:** `scripts/diagnose_entity_count.py`  
**Model:** Evidence Visibility (M3.3A)  
**Checkpoint:** `outputs/fmnerg_roberta128_evidence_visibility/best_model.pt`  
**Dataset:** FMNERg Dev (1500 records)

---

## Purpose

Diagnose Evidence Visibility model (M3.3A) performance by analyzing predictions stratified by **gold entity count** to identify systematic biases.

---

## Key Findings

### 1. **Single-Entity Performance Degradation**

**Single-Entity Records (661, 44.1%):**
- GMNER F1: **60.10%** ⬇️
- Predictions: 750 (gold: 661)
- **Over-prediction: +13.5%**

**Multi-Entity Records (752, 50.1%):**
- GMNER F1: **63.17%** ⬆️
- Predictions: 1741 (gold: 1789)
- **Under-prediction: -2.7%**

**Gap:** 3.07 points worse on single-entity records despite simpler ground truth

---

### 2. **Precision-Recall Imbalance**

| Slice | Precision | Recall | Behavior |
|-------|-----------|--------|----------|
| Single-Entity | 56.53% | 64.15% | Over-triggers (high recall, low precision) |
| Multi-Entity | 64.04% | 62.33% | Conservative (balanced) |

**Interpretation:** Model lacks false positive suppression in simple scenarios

---

### 3. **Performance Drop Analysis**

**Span → GMNER F1 Drop:**
- Single-Entity: 86.46% → 60.10% (**-26.36 points**)
- Multi-Entity: 87.93% → 63.17% (**-24.76 points**)

Larger drop in single-entity scenarios suggests:
- Region grounding errors
- Evidence visibility miscalibration
- Type classification drift

---

## Root Cause Hypotheses

### H1: **Threshold Miscalibration**

Evidence visibility thresholds may be too permissive for sparse candidate sets:
- `visible_from_null_threshold: 0.8` (promotes null → visible)
- `null_from_visible_threshold: 0.2` (demotes visible → null)

**Test:** Re-run with `null_from_visible_threshold: 0.15` to increase false positive suppression

---

### H2: **Training Distribution Bias**

Model optimized for multi-entity scenarios:
- Multi-entity records are more common or weighted higher during training
- Candidate competition improves precision in dense scenarios
- Sparse scenarios lack inhibition mechanism

**Solution:** Re-balance training data or add loss weight for low-entity-count records

---

### H3: **Candidate Promotion Over-Sensitivity**

Fine grounding adapter may promote too many candidates when:
- Few visual regions exist (sparse scenes)
- No strong negative evidence suppresses weak candidates

**Mitigation:** Gate candidate promotion on global scene density or confidence calibration

---

## Recommendations

### Immediate (M3.3B)

1. **Analyze False Positives**
   ```bash
   # Extract single-entity over-predictions from records.jsonl
   jq 'select(.gold_entity_count == 1 and .pred_entity_count > 1)' \
      outputs/diagnostics/m33a_entity_count/records.jsonl | head -20
   ```

2. **Threshold Sensitivity Study**
   - Test `null_from_visible_threshold` in [0.15, 0.20, 0.25]
   - Measure impact on single-entity precision/recall

3. **Error Pattern Mining**
   - Cluster false positives by entity type, visual scene, text length
   - Identify if errors concentrate in specific entity categories

---

### Long-Term (M3.4+)

1. **Dynamic Thresholding**
   - Condition visibility thresholds on `fine_candidate_count`
   - Lower threshold for high-density scenes, raise for sparse scenes

2. **Training Re-balancing**
   - Up-sample single-entity records 2×
   - Add per-sample loss weight: `w = 1.0 + (1.0 / gold_entity_count)`

3. **Calibration Study**
   - Plot precision-recall curves per entity count bin [1, 2, 3, 4+]
   - Find optimal operating points per slice

---

## Script Usage

```bash
python scripts/diagnose_entity_count.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --formal-cache knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical.pt \
  --expanded-cache knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical_r36.pt \
  --output-dir outputs/diagnostics/m33a_entity_count
```

**Outputs:**
- `records.jsonl` — Per-record predictions with gold/pred entity counts
- `summary.json` — Aggregate metrics by slice
- `protocol.json` — Evaluation parameters

---

## Results Summary

| Metric | Overall | Single-Entity | Multi-Entity |
|--------|---------|---------------|--------------|
| Records | 1500 | 661 (44.1%) | 752 (50.1%) |
| **GMNER F1** | **62.13%** | **60.10%** | **63.17%** |
| MNER F1 | 81.67% | 79.66% | 82.78% |
| EEG F1 | 66.09% | 64.35% | 67.03% |
| Span F1 | 87.28% | 86.46% | 87.93% |
| Pred Count | 2504 | 750 (+13.5%) | 1741 (-2.7%) |

---

## Next Steps

1. **Run follow-up analysis** on false positive patterns in `records.jsonl`
2. **Test threshold adjustments** to improve single-entity precision
3. **Profile training data** to verify entity count distribution
4. **Design M3.3B improvements** targeting single-entity scenarios

---

## Related Files

- Script: `scripts/diagnose_entity_count.py`
- Results: `outputs/diagnostics/m33a_entity_count/` (local only, not in git)
- Analysis: `outputs/diagnostics/m33a_entity_count/ANALYSIS.md` (local only)

---

## Conclusion

M3.3A Evidence Visibility model shows systematic **over-prediction in single-entity scenarios** (+13.5%), leading to 3-point GMNER F1 degradation compared to multi-entity records. 

**Root cause:** Insufficient false positive suppression when candidate competition is low.

**Impact:** Model achieves 60.10% GMNER F1 on single-entity records vs 63.17% on multi-entity, despite single-entity being structurally simpler.

**Action:** Investigate threshold calibration and training data balance for M3.3B iteration.
