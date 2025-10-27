# Problem 4: NSA Spam Detection

This project implements a **Negative Selection Algorithm (NSA)** for SMS spam detection, inspired by biological immune systems. The algorithm learns to identify spam messages by creating detectors that ignore legitimate messages (ham) but activate on suspicious patterns.

## 🧬 Algorithm Introduction

### Biological Inspiration
The Negative Selection Algorithm mimics how your immune system works:
- **cells** are trained to recognize "non-self" (foreign threats) while ignoring "self" (healthy tissue)
- During development, cells that react to healthy tissue are **eliminated**
- Only cells that ignore healthy tissue but respond to threats **survive**
- These detectors patrol the body, activating when encountering foreign patterns

### NSA Implementation for Spam Detection

**Core Principle**: Create detectors that **ignore ham messages** but **activate on spam patterns**

#### Step-by-Step Process:

1. **Text Preprocessing**: Convert SMS messages into sets of token IDs
   ```
   "Free money now!" → {45, 123, 891}
   ```

2. **Random Detector Generation**: Create candidate detectors as random token combinations
   ```python
   detector = {token_45, token_123, token_234}  # Random 3-5 tokens
   ```

3. **Negative Selection Training**: 
   - Test each detector against **all ham messages**
   - **Reject** detectors that activate on ham (they're too general)
   - **Keep** detectors that ignore ham but might catch spam
   ```python
   if detector_matches_any_ham_message(detector):
       reject_detector()  # Too general - would cause false positives
   else:
       keep_detector()    # Good - ignores ham, might catch spam
   ```

4. **Classification**: 
   - Count how many detectors activate on a new message
   - If activation count ≥ threshold → **Spam**
   - If activation count < threshold → **Ham**

#### Why This Works
- **Ham messages** use consistent, natural language patterns
- **Spam messages** contain unusual combinations: "FREE", "URGENT", "CALL NOW", excessive punctuation
- **NSA detectors** learn these spam-specific anomalous patterns while avoiding normal conversation patterns

### Data Splitting Strategy

To ensure reliable model evaluation and prevent overfitting, the dataset is split into three distinct sets:

#### **Training Set (70%)**
- **Purpose**: Train the NSA algorithm by generating and selecting detectors
- **Usage**: Used in the "negative selection" phase where detectors are tested against ham messages
- **Key Point**: Only ham messages from training set are used to reject detectors
- **Why Important**: Ensures detectors learn what "normal" (ham) patterns look like

#### **Validation Set (10%)**  
- **Purpose**: Parameter tuning and model selection
- **Usage**: Test different NSA hyperparameters (detector size, overlap threshold, etc.)
- **Key Point**: Used to find optimal parameter combinations without touching test data
- **Why Important**: Prevents overfitting to test set during parameter search

#### **Test Set (20%)**
- **Purpose**: Final, unbiased performance evaluation
- **Usage**: Only used once at the end to report final model performance
- **Key Point**: Never seen during training or parameter tuning
- **Why Important**: Gives honest estimate of how well the model will work on new, unseen data

#### **Data Flow**
```
SMS Dataset (5,572 messages)
├── Training Set (70%) → Train detectors, reject those matching ham
├── Validation Set (10%) → Tune hyperparameters, select best model  
└── Test Set (20%) → Final evaluation, report results
```

**Critical Rule**: Information must never "leak" between sets. The algorithm should never see test or validation data during training, and parameter selection should never use test data.

## 📁 Source Code Structure (`/src`)

### Core Files

- **`nsa.py`** - **Main Algorithm Implementation**
  - `NegativeSelectionClassifier` class
  - Detector generation and negative selection training
  - Pattern matching logic and classification
  - Activation counting for scoring

- **`preprocessing.py`** - **Data Pipeline**
  - Load SMS spam dataset from TSV format
  - Text cleaning and normalization
  - Train/validation/test split
  - Vocabulary building and token ID conversion
  - Convert text messages to token ID sets

- **`evaluation.py`** - **Evaluation Pipeline**
  - Main entry point for running experiments
  - Loads data, trains NSA model, evaluates performance
  - Generates classification reports and confusion matrices
  - Saves misclassified examples for analysis

- **`utils.py`** - **Utility Functions**
  - Reproducibility helpers (`set_seed()`)
  - Confusion matrix computation
  - Performance metrics calculation (precision, recall, F1)
  - Classification report generation

- **`visualization.py`** - **Plotting Functions**
  - Confusion matrix visualization
  - Performance plotting utilities
  - Matplotlib configuration and styling

- **`constants.py`** - **Configuration Hub**
  - All hyperparameters and settings
  - File paths and directory structure
  - Experiment configuration options

## ⚙️ Configuration Variables (`constants.py`)

### Data Configuration
- **`SEED`**: Random seed for reproducibility (None = random)
- **`DATA_PATH`**: Path to SMS spam dataset (`data/sms_spam.tsv`)
- **`TEST_RATIO`**: Fraction of data for testing (0.2 = 20%)
- **`VAL_RATIO`**: Fraction of data for validation (0.1 = 10%)

### Text Processing
- **`VOCAB_MIN_FREQ`**: Minimum token frequency to include in vocabulary (2)
- **`VOCAB_MAX_SIZE`**: Maximum vocabulary size (4000 tokens)

### NSA Hyperparameters
- **`NSA_NUM_DETECTORS`**: Number of detectors to generate (1000)
  - *More detectors* = better coverage but slower training
  
- **`NSA_DETECTOR_SIZE`**: Tokens per detector (4)
  - *Smaller* (3) = more general, may over-activate
  - *Larger* (5) = more specific, may miss variants
  
- **`NSA_OVERLAP_THRESHOLD`**: Minimum token overlap for activation (1)
  - *Low* (1) = sensitive but noisy
  - *High* (3) = conservative but may miss spam
  
- **`NSA_MAX_ATTEMPTS`**: Maximum attempts to find valid detectors (60,000)
  - Higher = more thorough search for non-self detectors
  
- **`NSA_MIN_ACTIVATIONS`**: Detectors needed to classify as spam (2)
  - *Low* (1) = any single detector triggers spam classification
  - *High* (2+) = multiple detectors must agree

### Advanced Features
- **`NSA_USE_SPAM_WEIGHTS`**: Enable spam-guided detector sampling (True)
- **`NSA_WEIGHT_EXP`**: Weight exponent for spam token preference (1.5)
  - Biases detector generation toward tokens more common in spam

### Output Configuration
- **`RESULTS_DIR`**: Output directory for results (`results/`)
- **`MISCLASS_*_FILENAME`**: Files for saving classification errors
- **`TRUE_*_FILENAME`**: Files for saving correct classifications

## 🔬 Analysis Script (`/analysis/run_nsa_analysis.py`)

This comprehensive analysis script performs parameter optimization and generates detailed visualizations.

### Main Functions

#### **Parameter Grid Search**
```python
# Tests 54 parameter combinations (3×3×3×2)
num_detectors = [500, 1000, 2000]
detector_size = [3, 4, 5] 
overlap_threshold = [1, 2, 3]
min_activations = [1, 2]
```

#### **Analysis Components**

1. **`load_and_preprocess_data()`**
   - Loads SMS spam dataset (5,572 messages)
   - Splits into train/validation/test sets
   - Builds vocabulary and converts texts to token sets

2. **`run_parameter_search()`**
   - Tests all parameter combinations
   - Evaluates each on validation set
   - Tracks F1-score, ROC-AUC, precision, recall
   - Finds optimal parameters

3. **`plot_baseline_analysis()`**
   - ROC curve with AUC score
   - Precision-Recall curve with AP score  
   - Confusion matrix heatmap
   - Shows performance of best model

4. **`plot_parameter_effects()`**
   - **Detector Size Effects**: How detector size impacts F1-score
   - **Overlap Threshold Effects**: Sensitivity vs specificity trade-offs
   - **Number of Detectors**: Coverage vs computational cost
   - **Activation Threshold**: Conservative vs aggressive classification

5. **`plot_best_model_evaluation()`**
   - Detailed confusion matrix for optimal parameters
   - Performance metrics breakdown
   - Error analysis visualization

6. **`plot_detector_coverage_curves()`**
   - Shows detector activation patterns across dataset
   - Visualizes how many detectors activate for spam vs ham
   - Helps understand decision boundaries

7. **`plot_detector_statistics()`**
   - Detector activation frequency distribution
   - Most/least active detectors
   - Coverage analysis across message types

### Generated Outputs

#### **Visualizations** (`results/plots/`)
- `baseline_analysis.png` - ROC, PR curves, confusion matrix
- `parameter_effects.png` - Parameter sensitivity analysis  
- `best_model_evaluation.png` - Optimal model performance
- `detector_coverage_curves.png` - Activation pattern analysis
- `detector_statistics.png` - Detector usage statistics

#### **Data Files** (`results/`)
- `detector_stats.json` - Detector performance metrics
- `coverage_curve.json` - Coverage analysis data
- `pr_curve.tsv` - Precision-recall curve data
- `roc_curve.tsv` - ROC curve data
- `false_positives.tsv` - Ham messages classified as spam
- `false_negatives.tsv` - Spam messages classified as ham
- `true_positives.tsv` - Correctly identified spam
- `true_negatives.tsv` - Correctly identified ham

### Performance Results

The analysis typically achieves:
- **F1-Score**: ~0.84 (excellent balance of precision/recall)
- **ROC-AUC**: ~0.92 (excellent discrimination ability)
- **Precision**: ~0.86 (low false positive rate)
- **Recall**: ~0.83 (good spam detection rate)

## 🚀 Usage

### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete analysis
cd analysis/
python run_nsa_analysis.py
```

### Basic Evaluation
```bash
# Run single experiment with default parameters
cd src/
python evaluation.py
```

## 📊 Understanding the Results

- **ROC Curve**: Shows true positive rate vs false positive rate trade-offs
- **PR Curve**: Shows precision vs recall trade-offs (better for imbalanced data)
- **Confusion Matrix**: Detailed breakdown of correct/incorrect classifications
- **Parameter Effects**: Helps understand which settings matter most
- **Coverage Analysis**: Shows how detectors respond to different message types

The NSA approach is particularly effective for spam detection because spam messages often contain distinctive, non-natural language patterns that normal conversation avoids!
