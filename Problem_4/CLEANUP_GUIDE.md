# Cleanup Guide - Files Safe to Delete

This guide identifies old, obsolete, or testing files that can be safely deleted now that the final comprehensive analysis is complete.

---

## ✅ KEEP (Active/Final Files)

### Analysis Scripts:
- **`analysis/comprehensive_grid_search.py`** ✓ KEEP
  - Final grid search with all n-gram sizes (3, 4, 5)
  - Generates all metrics (P, R, F1, FPR, ROC-AUC, PR-AUC)
  - Creates plots (detector coverage, precision-recall curves)
  - Result: 274 configurations tested

- **`analysis/analyze_results.py`** ✓ KEEP
  - Quick results viewer for post-analysis
  - Useful for examining comprehensive_results.csv

### Results Files:
- **`results/comprehensive_results.csv`** ✓ KEEP
  - Complete results: 274 configs across n-grams 3, 4, 5
  - Primary data file for paper

- **`results/auc_results.csv`** ✓ KEEP
  - ROC-AUC and PR-AUC metrics per detector set
  - Required for paper metrics

- **`results/detector_cache_bits/`** ✓ KEEP
  - All 22 cached detector sets (n-grams 3, 4, 5)
  - Enables instant re-running of experiments
  - Example files:
    - `det_1k_r7_n3.pkl`, `det_2k_r8_n4.pkl`, `det_3k_r8_n5.pkl`, etc.

- **`results/plots/`** ✓ KEEP
  - `detector_coverage_curve.png` - Shows recall vs. detector count
  - `precision_recall_curve.png` - Shows P/R trade-offs
  - Both required for paper

---

## ❌ DELETE (Old/Obsolete Files)

### Analysis Scripts:

1. **`analysis/bit_based_grid_search.py`** ❌ DELETE
   - **Status**: Obsolete - superseded by comprehensive_grid_search.py
   - **Description**: Initial bit-based grid search (64 configs, n-gram=4 only)
   - **Why delete**: 
     - Only tested n-gram=4
     - Limited hyperparameter ranges
     - Produced `bit_based_results.csv` which is now obsolete
     - All functionality replaced by comprehensive_grid_search.py

2. **`analysis/test_bit_matching.py`** ❌ DELETE
   - **Status**: Testing/debugging script
   - **Description**: Quick test to verify bit-based r-contiguous implementation
   - **Why delete**:
     - Was used during development to test the implementation
     - No longer needed - implementation is verified and working
     - Not used in final analysis

3. **`analysis/ngram_comparison.py`** ❌ DELETE
   - **Status**: Empty file
   - **Description**: Placeholder file (0 bytes)
   - **Why delete**: 
     - Never implemented
     - N-gram comparison functionality is now in comprehensive_grid_search.py

### Results Files:

4. **`results/bit_based_results.csv`** ❌ DELETE
   - **Status**: Obsolete - superseded by comprehensive_results.csv
   - **Description**: Results from initial bit_based_grid_search.py run
   - **Why delete**:
     - Only contains n-gram=4 results with limited configs
     - All data re-generated and included in comprehensive_results.csv
     - No unique information

5. **`results/detector_cache/`** (entire directory) ❌ DELETE
   - **Status**: Obsolete - old word-based detectors
   - **Description**: Cached detectors from pre-bit-based implementation
   - **Files**: 
     - `det_2k_r0.005.pkl`
     - `det_3k_r0.003.pkl`
     - `det_5k_r0.003.pkl`
     - `det_7k_r0.002.pkl`
     - `det_10k_r0.002.pkl`
     - `det_15k_r0.001.pkl`
   - **Why delete**:
     - These use old word-based representation (not bit-based)
     - File naming uses old ratio parameter format (e.g., `r0.005`)
     - Not compatible with current bit-based implementation
     - All replaced by `detector_cache_bits/`

---

## Cleanup Commands

### Option 1: Safe Deletion (Review First)
```bash
cd /Users/mats/ACIT4610-exam/Problem_4

# Review files before deletion
ls -lh analysis/bit_based_grid_search.py
ls -lh analysis/test_bit_matching.py
ls -lh analysis/ngram_comparison.py
ls -lh results/bit_based_results.csv
ls -lh results/detector_cache/

# Delete obsolete files
rm analysis/bit_based_grid_search.py
rm analysis/test_bit_matching.py
rm analysis/ngram_comparison.py
rm results/bit_based_results.csv
rm -rf results/detector_cache/
```

### Option 2: Archive Before Deletion
```bash
cd /Users/mats/ACIT4610-exam/Problem_4

# Create archive of old files
mkdir -p archive
mv analysis/bit_based_grid_search.py archive/
mv analysis/test_bit_matching.py archive/
mv analysis/ngram_comparison.py archive/
mv results/bit_based_results.csv archive/
mv results/detector_cache/ archive/

# Later, delete archive when confirmed unnecessary
# rm -rf archive/
```

---

## Space Savings

Estimated disk space to be freed:

- `bit_based_grid_search.py`: ~8 KB
- `test_bit_matching.py`: ~4 KB
- `ngram_comparison.py`: 0 KB (empty)
- `bit_based_results.csv`: ~6 KB
- `detector_cache/` directory: ~50-100 MB (6 detector files)

**Total savings**: ~50-100 MB (mostly from old detector cache)

---

## Post-Cleanup File Structure

```
Problem_4/
├── analysis/
│   ├── analyze_results.py          ✓ (utility)
│   └── comprehensive_grid_search.py ✓ (main analysis)
├── results/
│   ├── comprehensive_results.csv    ✓ (274 configs)
│   ├── auc_results.csv              ✓ (ROC/PR metrics)
│   ├── detector_cache_bits/         ✓ (22 detector sets)
│   │   ├── det_1k_r7_n3.pkl
│   │   ├── det_2k_r8_n4.pkl
│   │   ├── det_3k_r8_n5.pkl
│   │   └── ... (19 more)
│   └── plots/                       ✓ (2 PNG files)
│       ├── detector_coverage_curve.png
│       └── precision_recall_curve.png
└── src/
    ├── nsa_optimized.py             ✓ (bit-based NSA)
    ├── preprocessing.py             ✓ (data loading)
    └── constants.py                 ✓ (configuration)
```

---

## Summary

**Files to Keep**: 5 files + 2 directories
- 2 Python scripts (analysis)
- 2 CSV files (results)
- 22 detector cache files (detector_cache_bits/)
- 2 plot files (plots/)

**Files to Delete**: 3 files + 1 directory
- 3 obsolete Python scripts
- 1 obsolete CSV file
- 1 old detector cache directory (6 files)

**Recommendation**: Use Option 2 (archive first) if unsure, then delete archive after confirming everything works correctly with the final analysis.
