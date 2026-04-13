"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay,
                             mean_absolute_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Task 1: Load Data
# ---------------------------------------------------------------------------

def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    df = pd.read_csv(filepath)

    print("=== Task 1: Basic EDA ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts()}")
    churn_rate = df['churned'].mean()
    print(f"Churn rate: {churn_rate:.2%}  (imbalanced — ~{churn_rate*100:.0f}% positive class)")

    return df


# ---------------------------------------------------------------------------
# Task 2: Split Data
# ---------------------------------------------------------------------------

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    For continuous targets (e.g. monthly_charges), stratification is skipped.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Use stratification only for binary/categorical targets
    is_classification = y.nunique() <= 10
    stratify = y if is_classification else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print(f"\n=== Task 2: Train/Test Split (target='{target_col}') ===")
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    if is_classification:
        print(f"Churn rate — Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task 3: Logistic Regression Pipeline
# ---------------------------------------------------------------------------

def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Task 3: Logistic Regression Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Logistic Regression")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png")

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics


# ---------------------------------------------------------------------------
# Task 4: Ridge Regression Pipeline
# ---------------------------------------------------------------------------

def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])
    return pipeline


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n=== Task 4: Ridge Regression Evaluation ===")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

    return {"mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Task 5: Lasso Regularization Comparison (not autograder-tested)
# ---------------------------------------------------------------------------

def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=0.1, max_iter=10000))
    ])
    return pipeline


def compare_ridge_lasso(X_train, X_test, y_train, y_test, feature_names):
    """Fit Ridge and Lasso on the same data and compare coefficients."""
    ridge = build_ridge_pipeline()
    lasso = build_lasso_pipeline()

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_coefs = ridge.named_steps["regressor"].coef_
    lasso_coefs = lasso.named_steps["regressor"].coef_

    print("\n=== Task 5: Ridge vs Lasso Coefficients ===")
    print(f"{'Feature':<22} {'Ridge':>12} {'Lasso':>12} {'Zeroed?':>10}")
    print("-" * 58)
    for feat, rc, lc in zip(feature_names, ridge_coefs, lasso_coefs):
        zeroed = "YES" if abs(lc) < 1e-6 else ""
        print(f"{feat:<22} {rc:>12.4f} {lc:>12.4f} {zeroed:>10}")

    # Features Lasso drives to zero are typically those with low marginal
    # predictive power for monthly_charges once correlated features are
    # already in the model (e.g., binary flags like has_partner /
    # has_dependents carry little extra signal beyond tenure & total_charges).


# ---------------------------------------------------------------------------
# Task 6: Cross-Validation
# ---------------------------------------------------------------------------

def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv_splitter, scoring="accuracy")

    print("\n=== Task 6: Cross-Validation (5-fold) ===")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  Mean: {scores.mean():.4f}  ±  {scores.std():.4f}")

    return scores


# ---------------------------------------------------------------------------
# Task 7: Summary of Findings (not autograder-tested)
# ---------------------------------------------------------------------------

"""
=== Task 7: Summary of Findings ===

1. Most important features for predicting churn:
   - tenure: Customers with shorter tenure churn far more — they haven't
     yet formed loyalty. This is typically the single strongest predictor.
   - num_support_calls: High support-call volume signals frustration and
     is a reliable early-warning indicator of churn.
   - monthly_charges: Higher bills correlate with higher churn risk,
     especially on Month-to-month contracts.
   - total_charges: Largely driven by tenure × monthly_charges; adds
     signal when tenure is short but spending is already high.
   - has_partner / has_dependents: Customers with family ties tend to be
     more stable (lower churn rate).

2. Model performance:
   The Logistic Regression with class_weight="balanced" deliberately
   trades precision for recall. On an imbalanced dataset (~16% churn),
   an unweighted model achieves high accuracy by predicting "no churn"
   almost always — making accuracy a misleading metric.
   → Recall is MORE concerning here. Missing an actual churner (false
     negative) means losing a customer with zero opportunity for
     intervention. A false alarm (false positive) wastes a retention
     offer but costs far less. We therefore prioritise recall while
     monitoring F1 to avoid excessive false positives.

3. Recommended next steps:
   - Add a DummyClassifier baseline (Thursday's Integration Task) to
     quantify how much the model beats random chance.
   - Try threshold tuning (e.g., 0.3–0.4) to further boost recall at an
     acceptable precision cost.
   - Engineer features: contract_type and internet_service are strong
     predictors in the full dataset but were excluded here as non-numeric;
     encode them (one-hot) and re-run.
   - Explore tree-based models (Random Forest, Gradient Boosting) which
     handle non-linear interactions without manual feature engineering.
   - Use SMOTE or other resampling instead of class_weight to address
     imbalance and compare results.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Task 1 ----
    df = load_data()
    if df is None:
        raise SystemExit("load_data() returned None — check file path.")

    print(f"\nLoaded {len(df)} rows, {df.shape[1]} columns")

    # Numeric features used for both tasks
    numeric_features = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen",
                        "has_partner", "has_dependents"]

    # ---- Task 2 + 3: Classification ----
    df_cls = df[numeric_features + ["churned"]].dropna()
    X_train, X_test, y_train, y_test = split_data(df_cls, "churned")

    pipe_log = build_logistic_pipeline()
    metrics = evaluate_classifier(pipe_log, X_train, X_test, y_train, y_test)
    print(f"\nLogistic Regression metrics: {metrics}")

    # ---- Task 6: Cross-Validation ----
    scores = run_cross_validation(pipe_log, X_train, y_train)
    print(f"CV Mean ± Std: {scores.mean():.3f} ± {scores.std():.3f}")

    # ---- Task 4: Ridge Regression ----
    reg_features = ["tenure", "total_charges", "num_support_calls",
                    "senior_citizen", "has_partner", "has_dependents"]
    df_reg = df[reg_features + ["monthly_charges"]].dropna()
    X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")

    ridge_pipe = build_ridge_pipeline()
    reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
    print(f"\nRidge Regression metrics: {reg_metrics}")

    # ---- Task 5: Lasso Comparison ----
    compare_ridge_lasso(X_tr, X_te, y_tr, y_te, reg_features)