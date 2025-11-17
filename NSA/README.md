# Problem 4: Spam Detection Using Negative Selection Algorithm (NSA)

Implementation of a bit-based Negative Selection Algorithm for SMS spam detection using pure self-learning (trained only on legitimate messages).


analysis/comprehensive_grid_search.py is the MAIN file to run the algorithm  
## 📁 Project Structure

```
NSA/
├── README.md                   # This file
│
├── data/                       # Dataset storage
│   ├── sms_spam.tsv           # SMS Spam Collection (5,574 messages)
│   └── enron1/                # Enron email dataset (alternative)
│
├── src/                        # Core implementation
│   ├── constants.py           # Configuration constants (cleaned)
│   ├── preprocessing.py       # Data loading and text preprocessing
│   ├── nsa_optimized.py       # NSA classifier with bit-based matching
│   └── __init__.py
│
├── analysis/                   # Experiment scripts
│   ├── comprehensive_grid_search.py    # MAIN FILE
│   ├── detector_coverage_analysis.py   # Generates detector scaling plots
│   └── detector_statistics.py          # Calculates detector matching stats
│
├── results/                    # Generated outputs
│   ├── comprehensive_results.csv       # All 274 configuration results
│   ├── detector_cache_bits/      # Cached detectors (speeds up for set seed)
│   └── plots/                          # Generated figures
│       ├── detector_coverage_curve.png
│       ├── detector_f1_comparison.png
│       ├── pareto_front.png
│       └── precision_recall_curve.png

```
