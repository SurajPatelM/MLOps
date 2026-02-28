# Feature Selection Lab — IE7374

> **Important:** The original `Feature_Selection.ipynb` notebook has not been modified. All changes, additions, and experiments are contained entirely in the new `feature_selection.py` script. The notebook is included as-is for reference.

## Overview

This lab explores feature selection techniques on the [Breast Cancer Wisconsin Dataset](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) using a RandomForestClassifier. Starting from the original notebook, the lab was restructured into a single configurable Python script that compares **10 feature selection approaches** across three categories:

**Filter Methods** (model-independent, statistical ranking):
1. Correlation with target (Pearson)
2. Remove inter-correlated features
3. Univariate F-test (ANOVA)
4. Chi-squared
5. Mutual Information

**Wrapper Methods** (model-dependent, iterative):
6. Recursive Feature Elimination (RFE)

**Embedded Methods** (built into model training):
7. Feature Importance (RandomForest)
8. L1 Regularization (LinearSVC)
9. Lasso Regularization (LogisticRegression)

## Changes from the Original Lab Notebook

### 1. Converted Notebook to Python Script
The original `Feature_Selection.ipynb` was used as a reference to create a new standalone `feature_selection.py` script. No changes were made to the original notebook.

### 2. Added New Feature Selection Methods
The original notebook included Correlation, F-test, RFE, Feature Importance, and L1 Reg (LinearSVC). The script adds three more:
- **Chi-squared** — filter method using non-negative feature scores via MinMaxScaler, captures different feature relationships than Pearson correlation
- **Mutual Information** — non-linear filter method that detects dependencies Pearson correlation and F-test may miss
- **Lasso (LogisticRegression L1)** — alternative L1 regularization approach using LogisticRegression instead of LinearSVC

### 3. Automated Inter-correlation Removal
The original notebook manually identified and hardcoded which correlated features to drop (`radius_worst`, `perimeter_worst`, `area_worst`). The script automates this using a configurable threshold, systematically scanning the correlation matrix and removing features above the threshold.

### 4. CLI Configuration
All parameters are configurable via command-line arguments:
- `--mode` — run baseline, filter, wrapper, embedded, or all
- `--corr-threshold` — Pearson correlation threshold for target relevance
- `--intercorr-threshold` — threshold for removing inter-correlated features
- `--k-best` — number of features for SelectKBest methods
- `--rfe-n-features` — number of features for RFE
- `--importance-threshold` — threshold for RandomForest feature importance
- `--l1-C` — regularization strength for L1 methods

### 5. Results Logging & Plots
All results are appended to `results.txt` with timestamps and a summary comparison table. Correlation matrices and feature importance plots are saved to `./plots/`.

## Project Structure

```
Feature_Selection/
├── Feature_Selection.ipynb   # Original notebook (unmodified)
├── feature_selection.py      # Main script (all methods)
├── requirements.txt          # Dependencies
├── data/
│   └── breast_cancer_data.csv
├── results.txt               # Experiment results (auto-generated)
├── plots/                    # Saved plots (auto-generated)
│   ├── correlation_full.png
│   ├── correlation_target_relevant.png
│   └── feature_importances.png
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Everything
```bash
python feature_selection.py --mode all
```

### Run Individual Method Groups
```bash
python feature_selection.py --mode baseline
python feature_selection.py --mode filter
python feature_selection.py --mode wrapper
python feature_selection.py --mode embedded
```

### Custom Thresholds
```bash
python feature_selection.py --mode all \
    --corr-threshold 0.3 \
    --intercorr-threshold 0.85 \
    --k-best 15 \
    --rfe-n-features 15 \
    --importance-threshold 0.02
```

## Results

### Experiment: Default Parameters

| Method | Accuracy | ROC AUC | Precision | Recall | F1 Score | Features |
|--------|----------|---------|-----------|--------|----------|----------|
| **Baseline (All)** | 0.9649 | 0.9673 | 0.9318 | 0.9762 | 0.9535 | 30 |
| **Corr w/ Target** | **0.9737** | **0.9742** | **0.9535** | 0.9762 | **0.9647** | 25 |
| **Remove Intercorr** | 0.9298 | 0.9395 | 0.8542 | 0.9762 | 0.9111 | 15 |
| **F-test** | **0.9737** | **0.9742** | **0.9535** | 0.9762 | **0.9647** | 20 |
| **Chi-squared** | 0.9561 | 0.9603 | 0.9111 | 0.9762 | 0.9425 | 20 |
| **Mutual Info** | **0.9737** | **0.9742** | **0.9535** | 0.9762 | **0.9647** | 20 |
| **RFE** | 0.9649 | 0.9673 | 0.9318 | 0.9762 | 0.9535 | 20 |
| **Feature Importance** | 0.9649 | 0.9673 | 0.9318 | 0.9762 | 0.9535 | **14** |
| **L1 Reg (SVC)** | 0.9298 | 0.9296 | 0.8864 | 0.9286 | 0.9070 | 18 |
| **Lasso (LR)** | 0.9474 | 0.9534 | 0.8913 | 0.9762 | 0.9318 | 15 |

### Key Observations

1. **Filter methods (F-test, Mutual Info, Correlation) achieved the best performance** at 97.4% accuracy and 0.9647 F1, outperforming the baseline (96.5%) while using fewer features. F-test and Mutual Info are the most efficient of these, achieving top results with only 20 features.

2. **Feature Importance is the efficiency winner.** It matched baseline accuracy (96.5%) with just 14 features — a 53% reduction in feature count. This makes it the best choice when minimizing model complexity is the priority.

3. **Recall stayed consistently high (97.6%) across most methods.** In a medical diagnosis context, this means the model rarely misses malignant cases regardless of which feature selection method is used.

4. **Precision is where methods diverge.** The top filter methods achieved 95.4% precision while L1 Reg (SVC) dropped to 88.6%. This means weaker methods produce more false positive malignant diagnoses.

5. **Automated inter-correlation removal (threshold=0.9) was too aggressive**, dropping 10 features including important ones like `concave points_mean` and `area_worst`, causing a significant performance drop to 93.0%. A higher threshold (0.95) improved this to 94.7% by retaining 3 more features.

6. **L1 regularization methods underperformed** (93.0% for LinearSVC, 94.7% for Lasso). These methods select features based on linear separability, which may not align well with the non-linear decision boundaries of the RandomForest evaluation model.

### Recommendations

- **Best overall performance**: F-test or Mutual Information (97.4% accuracy, 20 features)
- **Best efficiency**: Feature Importance (96.5% accuracy, 14 features)
- **Best for production**: F-test — highest accuracy with moderate feature count, simple to implement, and model-independent

## Feature Selection Methods Compared

| Category | Method | How It Works | Pros | Cons |
|----------|--------|-------------|------|------|
| **Filter** | Correlation | Pearson correlation with target | Simple, fast, interpretable | Only captures linear relationships |
| **Filter** | F-test | ANOVA F-values | Statistical rigor, widely used | Assumes normal distribution |
| **Filter** | Chi-squared | Chi-squared statistic | Good for categorical features | Requires non-negative values |
| **Filter** | Mutual Info | Information-theoretic measure | Captures non-linear dependencies | Computationally heavier |
| **Wrapper** | RFE | Recursive elimination using model | Uses model feedback | Slow, model-dependent |
| **Embedded** | Feature Importance | Tree-based importance scores | Built-in, efficient | Biased toward high-cardinality features |
| **Embedded** | L1 Reg (SVC) | L1 penalty zeros out weights | Automatic feature selection | Assumes linear separability |
| **Embedded** | Lasso (LR) | L1 penalty on logistic regression | Probabilistic output | Same linear assumption as SVC |

## Dependencies

- `pandas>=1.5.0` — data processing
- `numpy>=1.23.0` — numerical operations
- `scikit-learn>=1.2.0` — ML models, feature selection, metrics
- `seaborn>=0.12.0` — correlation heatmaps
- `matplotlib>=3.6.0` — plotting