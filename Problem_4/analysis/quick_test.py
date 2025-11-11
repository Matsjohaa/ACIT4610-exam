"""
Quick test of pure NSA with relaxed ham tolerance
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import load_data, train_test_split
from nsa_optimized import NegativeSelectionClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from constants import DATA_PATH

# Load data
texts, labels = load_data(DATA_PATH)
X_train, y_train, X_test, y_test = train_test_split(texts, labels)

print("="*70)
print("QUICK TEST: Pure NSA with Relaxed Ham Tolerance")
print("="*70)

# Test r-contiguous configuration that showed promise
config = {
    'representation': 'vocabulary',
    'matching_rule': 'r_contiguous',
    'num_detectors': 700,
    'vocab_size': 1000,
    'detector_size': 4,
    'min_activations': 1,
    'max_ham_match_ratio': 0.05  # Will be tripled to 0.15 internally
}

print(f"\nConfiguration: {config}")
print("-"*70)

detector = NegativeSelectionClassifier(**config)
detector.fit(X_train, y_train)

y_pred = detector.predict(X_test)

# Calculate metrics (labels are 0/1, not 'ham'/'spam')
f1 = f1_score(y_test, y_pred, pos_label=1)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"F1 Score:    {f1:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Accuracy:    {accuracy:.4f}")
print("="*70)

# Check if improvement
if f1 > 0.02:
    print("\n✓ IMPROVEMENT! F1 > 0.02 (was 0.0144)")
else:
    print("\n✗ Still too low, need more adjustments")
