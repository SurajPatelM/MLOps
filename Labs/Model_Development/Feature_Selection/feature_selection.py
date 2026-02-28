"""
IE7374 - Feature Selection Lab
Breast Cancer Dataset: Compare Filter, Wrapper, and Embedded methods

Methods included:
  Filter:
    1. Correlation with target (Pearson)
    2. Remove inter-correlated features
    3. Univariate F-test (SelectKBest)
    4. Chi-squared (SelectKBest)
    5. Mutual Information (SelectKBest)
  Wrapper:
    6. Recursive Feature Elimination (RFE)
  Embedded:
    7. Feature Importance (RandomForest)
    8. L1 Regularization (LinearSVC)
    9. Lasso Regularization (LogisticRegression L1)

Usage:
  python feature_selection.py --mode all
  python feature_selection.py --mode filter
  python feature_selection.py --mode wrapper
  python feature_selection.py --mode embedded
  python feature_selection.py --mode baseline
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import (
    RFE, SelectKBest, SelectFromModel,
    chi2, f_classif, mutual_info_classif,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Configuration

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Selection Lab: Filter vs Wrapper vs Embedded on Breast Cancer Dataset"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "filter", "wrapper", "embedded", "all"],
                        help="Which methods to run (default: all)")

    # Data
    parser.add_argument("--data-path", type=str, default="./data/breast_cancer_data.csv",
                        help="Path to dataset CSV (default: ./data/breast_cancer_data.csv)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")

    # Filter settings
    parser.add_argument("--corr-threshold", type=float, default=0.2,
                        help="Correlation threshold for target relevance (default: 0.2)")
    parser.add_argument("--intercorr-threshold", type=float, default=0.9,
                        help="Threshold to remove inter-correlated features (default: 0.9)")
    parser.add_argument("--k-best", type=int, default=20,
                        help="Number of features for SelectKBest methods (default: 20)")

    # Wrapper settings
    parser.add_argument("--rfe-n-features", type=int, default=20,
                        help="Number of features for RFE (default: 20)")

    # Embedded settings
    parser.add_argument("--importance-threshold", type=float, default=0.013,
                        help="Threshold for feature importance selection (default: 0.013)")
    parser.add_argument("--l1-C", type=float, default=1.0,
                        help="Regularization parameter C for L1 methods (default: 1.0)")

    # Output
    parser.add_argument("--output-file", type=str, default="results.txt",
                        help="File to append results to (default: results.txt)")
    parser.add_argument("--plot-dir", type=str, default="./plots",
                        help="Directory to save plots (default: ./plots)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


# Logging

def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


# Data loading & preprocessing

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    # Drop unwanted columns
    cols_to_drop = [c for c in ["Unnamed: 32", "id"] if c in df.columns]
    df.drop(cols_to_drop, axis=1, inplace=True)

    # Integer encode diagnosis: M=1, B=0
    df["diagnosis_int"] = (df["diagnosis"] == "M").astype("int")
    df.drop(["diagnosis"], axis=1, inplace=True)

    X = df.drop("diagnosis_int", axis=1)
    Y = df["diagnosis_int"]

    return df, X, Y


# Model training & evaluation

def train_and_evaluate(X, Y, args):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, stratify=Y, random_state=123
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(criterion="entropy", random_state=47)
    model.fit(X_train_scaled, Y_train)

    y_pred = model.predict(X_test_scaled)

    return {
        "Accuracy": accuracy_score(Y_test, y_pred),
        "ROC AUC": roc_auc_score(Y_test, y_pred),
        "Precision": precision_score(Y_test, y_pred),
        "Recall": recall_score(Y_test, y_pred),
        "F1 Score": f1_score(Y_test, y_pred),
        "Features": X.shape[1],
    }


def log_result(name, metrics, f):
    log(f"\n{'='*55}", f)
    log(f" {name}", f)
    log(f"{'='*55}", f)
    for k, v in metrics.items():
        if isinstance(v, float):
            log(f"  {k:12s}: {v:.4f}", f)
        else:
            log(f"  {k:12s}: {v}", f)
    log("", f)


# Plotting helpers

def save_correlation_matrix(df, feature_names, title, filename, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(16, 14))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=plt.cm.PuBu, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=150)
    plt.close()


def save_feature_importance_plot(importances, feature_names, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 12))
    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp.sort_values(ascending=False).plot(kind="barh")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_importances.png"), dpi=150)
    plt.close()


# 1. Baseline (all features)

def run_baseline(df, X, Y, args, f, results):
    log("\n>>> [BASELINE] Training with all features...", f)
    metrics = train_and_evaluate(X, Y, args)
    log_result("BASELINE (All Features)", metrics, f)
    results.append(("All Features", metrics))

    # Save full correlation matrix
    save_correlation_matrix(
        df, list(X.columns) + ["diagnosis_int"],
        "Full Correlation Matrix", "correlation_full.png", args.plot_dir
    )
    log(f"    Saved: {args.plot_dir}/correlation_full.png", f)


# 2. Filter Methods

def run_filter_methods(df, X, Y, args, f, results):

    # ── 2a. Correlation with target ──
    log(f"\n>>> [FILTER] Correlation with target (threshold={args.corr_threshold})...", f)
    cor = df.corr()
    cor_target = abs(cor["diagnosis_int"])
    relevant = cor_target[cor_target > args.corr_threshold]
    names = [idx for idx in relevant.index if idx != "diagnosis_int"]
    log(f"    Selected {len(names)} features", f)

    metrics = train_and_evaluate(df[names], Y, args)
    log_result(f"FILTER — Correlation (threshold={args.corr_threshold})", metrics, f)
    results.append(("Corr w/ Target", metrics))

    # Save correlation matrix for selected features
    save_correlation_matrix(
        df, names, "Correlation — Target-Relevant Features",
        "correlation_target_relevant.png", args.plot_dir
    )

    # ── 2b. Remove inter-correlated features ──
    log(f"\n>>> [FILTER] Removing inter-correlated features (threshold={args.intercorr_threshold})...", f)
    corr_matrix = df[names].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > args.intercorr_threshold)]
    subset_names = [n for n in names if n not in to_drop]
    log(f"    Removed {len(to_drop)} features: {to_drop}", f)
    log(f"    Remaining: {len(subset_names)} features", f)

    metrics = train_and_evaluate(df[subset_names], Y, args)
    log_result(f"FILTER — Remove Inter-correlated (>{args.intercorr_threshold})", metrics, f)
    results.append(("Remove Intercorr", metrics))

    # ── 2c. Univariate F-test ──
    log(f"\n>>> [FILTER] Univariate F-test (k={args.k_best})...", f)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    selector = SelectKBest(f_classif, k=args.k_best)
    selector.fit(X_train_scaled, Y_train)
    ftest_names = X.columns[selector.get_support()].tolist()
    log(f"    Selected features: {ftest_names}", f)

    metrics = train_and_evaluate(df[ftest_names], Y, args)
    log_result(f"FILTER — F-test (k={args.k_best})", metrics, f)
    results.append(("F-test", metrics))

    # ── 2d. Chi-squared ──
    log(f"\n>>> [FILTER] Chi-squared (k={args.k_best})...", f)
    # Chi-squared requires non-negative values, so use MinMaxScaler
    mm_scaler = MinMaxScaler().fit(X_train)
    X_train_mm = mm_scaler.transform(X_train)

    chi2_selector = SelectKBest(chi2, k=args.k_best)
    chi2_selector.fit(X_train_mm, Y_train)
    chi2_names = X.columns[chi2_selector.get_support()].tolist()
    log(f"    Selected features: {chi2_names}", f)

    metrics = train_and_evaluate(df[chi2_names], Y, args)
    log_result(f"FILTER — Chi-squared (k={args.k_best})", metrics, f)
    results.append(("Chi-squared", metrics))

    # ── 2e. Mutual Information ──
    log(f"\n>>> [FILTER] Mutual Information (k={args.k_best})...", f)
    mi_selector = SelectKBest(mutual_info_classif, k=args.k_best)
    mi_selector.fit(X_train_scaled, Y_train)
    mi_names = X.columns[mi_selector.get_support()].tolist()
    log(f"    Selected features: {mi_names}", f)

    metrics = train_and_evaluate(df[mi_names], Y, args)
    log_result(f"FILTER — Mutual Information (k={args.k_best})", metrics, f)
    results.append(("Mutual Info", metrics))


# 3. Wrapper Methods

def run_wrapper_methods(df, X, Y, args, f, results):

    # ── 3a. Recursive Feature Elimination ──
    log(f"\n>>> [WRAPPER] RFE (n_features={args.rfe_n_features})...", f)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = RandomForestClassifier(criterion="entropy", random_state=47)
    rfe = RFE(model, n_features_to_select=args.rfe_n_features)
    rfe.fit(X_train_scaled, Y_train)
    rfe_names = X.columns[rfe.get_support()].tolist()
    log(f"    Selected features: {rfe_names}", f)

    metrics = train_and_evaluate(df[rfe_names], Y, args)
    log_result(f"WRAPPER — RFE (n={args.rfe_n_features})", metrics, f)
    results.append(("RFE", metrics))


# 4. Embedded Methods

def run_embedded_methods(df, X, Y, args, f, results):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, stratify=Y, random_state=123
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # ── 4a. Feature Importance (RandomForest) ──
    log(f"\n>>> [EMBEDDED] Feature Importance (threshold={args.importance_threshold})...", f)
    rf = RandomForestClassifier(random_state=47)
    rf.fit(X_train_scaled, Y_train)

    save_feature_importance_plot(rf.feature_importances_, X.columns, args.plot_dir)
    log(f"    Saved: {args.plot_dir}/feature_importances.png", f)

    sfm = SelectFromModel(rf, prefit=True, threshold=args.importance_threshold)
    fi_names = X.columns[sfm.get_support()].tolist()
    log(f"    Selected {len(fi_names)} features: {fi_names}", f)

    metrics = train_and_evaluate(df[fi_names], Y, args)
    log_result(f"EMBEDDED — Feature Importance (>{args.importance_threshold})", metrics, f)
    results.append(("Feature Importance", metrics))

    # ── 4b. L1 Regularization (LinearSVC) ──
    log(f"\n>>> [EMBEDDED] L1 Regularization — LinearSVC (C={args.l1_C})...", f)
    lsvc = LinearSVC(C=args.l1_C, penalty="l1", dual=False, random_state=args.seed, max_iter=10000)
    lsvc.fit(X_train_scaled, Y_train)

    sfm_l1 = SelectFromModel(lsvc, prefit=True)
    l1_names = X.columns[sfm_l1.get_support()].tolist()
    log(f"    Selected {len(l1_names)} features: {l1_names}", f)

    metrics = train_and_evaluate(df[l1_names], Y, args)
    log_result(f"EMBEDDED — L1 Reg / LinearSVC (C={args.l1_C})", metrics, f)
    results.append(("L1 Reg (SVC)", metrics))

    # ── 4c. Lasso Regularization (LogisticRegression L1) ──
    log(f"\n>>> [EMBEDDED] Lasso Regularization — LogisticRegression (C={args.l1_C})...", f)
    lasso = LogisticRegression(
        C=args.l1_C, penalty="l1", solver="liblinear",
        random_state=args.seed, max_iter=10000
    )
    lasso.fit(X_train_scaled, Y_train)

    sfm_lasso = SelectFromModel(lasso, prefit=True)
    lasso_names = X.columns[sfm_lasso.get_support()].tolist()
    log(f"    Selected {len(lasso_names)} features: {lasso_names}", f)

    metrics = train_and_evaluate(df[lasso_names], Y, args)
    log_result(f"EMBEDDED — Lasso / LogisticRegression (C={args.l1_C})", metrics, f)
    results.append(("Lasso (LR)", metrics))


# Summary table

def print_summary_table(results_list, f):
    log(f"\n{'='*80}", f)
    log(f" SUMMARY — All Methods Compared", f)
    log(f"{'='*80}", f)
    header = f"  {'Method':<22s} {'Acc':>7s} {'ROC':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'#Feat':>6s}"
    log(header, f)
    log(f"  {'─'*68}", f)
    for name, m in results_list:
        row = (f"  {name:<22s} {m['Accuracy']:>7.4f} {m['ROC AUC']:>7.4f} "
               f"{m['Precision']:>7.4f} {m['Recall']:>7.4f} {m['F1 Score']:>7.4f} "
               f"{m['Features']:>6d}")
        log(row, f)
    log("", f)


# Main

def main():
    args = parse_args()
    np.random.seed(args.seed)

    f = open(args.output_file, "a")
    header = (f"\n{'#'*60}\n"
              f"# Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"# Mode: {args.mode}\n"
              f"# Args: {vars(args)}\n"
              f"{'#'*60}")
    log(header, f)

    log("Loading dataset...", f)
    df, X, Y = load_and_preprocess(args.data_path)
    log(f"  Shape: {df.shape} | Features: {X.shape[1]} | Target distribution: {dict(Y.value_counts())}", f)

    results_list = []

    if args.mode in ("baseline", "all"):
        run_baseline(df, X, Y, args, f, results_list)

    if args.mode in ("filter", "all"):
        run_filter_methods(df, X, Y, args, f, results_list)

    if args.mode in ("wrapper", "all"):
        run_wrapper_methods(df, X, Y, args, f, results_list)

    if args.mode in ("embedded", "all"):
        run_embedded_methods(df, X, Y, args, f, results_list)

    # Print summary table
    if len(results_list) > 1:
        print_summary_table(results_list, f)

    f.close()
    log(f"\n Results appended to {args.output_file}", None)
    log(f" Plots saved to {args.plot_dir}/", None)


if __name__ == "__main__":
    main()