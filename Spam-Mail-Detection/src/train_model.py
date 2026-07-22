"""
train_model.py -- Model training, evaluation, cross-validation, and persistence.

Pipeline:
  1. split_data()            -- stratified 80/20 train-test split
  2. train_model()           -- Multinomial Naive Bayes with progress bar
  3. cross_validate_model()  -- 5-fold stratified cross-validation
  4. evaluate_model()        -- accuracy, precision, recall, F1, confusion matrix
  5. plot_roc_curve()        -- ROC + AUC
  6. plot_pr_curve()         -- Precision-Recall curve
  7. plot_confusion_matrix() -- heatmap
  8. plot_accuracy_graph()   -- train vs test accuracy bar
  9. save_model()            -- pickle model + vectorizer
"""

import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection        import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes            import MultinomialNB
from sklearn.metrics                import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_logger, get_model_path, get_notebook_path,
    save_metrics_to_csv, save_metrics_to_json,
    print_section, print_metrics
)

logger = setup_logger("train_model")


# ---------------------------------------------
# 1. Train / test split
# ---------------------------------------------

def split_data(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Stratified train/test split.

    Args:
        X            : Feature series (clean text).
        y            : Label series (0/1).
        test_size    : Fraction for the test set.
        random_state : Reproducibility seed.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    print_section("Train / Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"  Total samples  : {len(X)}")
    print(f"  Training set   : {len(X_train)}  ({100*(1-test_size):.0f}%)")
    print(f"  Test set       : {len(X_test)}   ({100*test_size:.0f}%)")
    print(f"  Spam in train  : {y_train.sum()}  ({100*y_train.mean():.1f}%)")
    print(f"  Spam in test   : {y_test.sum()}   ({100*y_test.mean():.1f}%)")

    logger.info(f"Split done -- train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------
# 2. Model training
# ---------------------------------------------

def train_model(X_train_tfidf, y_train, alpha: float = 0.1) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes classifier with a simulated progress bar.

    Args:
        X_train_tfidf : Sparse TF-IDF training matrix.
        y_train       : Training labels.
        alpha         : Laplace smoothing parameter.

    Returns:
        Fitted MultinomialNB model.
    """
    print_section("Training -- Multinomial Naive Bayes")

    steps = ["Initialising model", "Fitting on training data", "Computing priors",
             "Computing likelihoods", "Model ready"]

    with tqdm(total=len(steps), desc="  Training", unit="step",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:

        model = MultinomialNB(alpha=alpha)
        for step in steps:
            pbar.set_postfix_str(step)
            if step == "Fitting on training data":
                model.fit(X_train_tfidf, y_train)
            time.sleep(0.15)   # brief pause so progress bar is visible
            pbar.update(1)

    print(f"\n  [OK] Model trained  (alpha={alpha})")
    print(f"  Classes          : {model.classes_}")
    print(f"  Feature count    : {model.feature_count_.shape[1]}")

    logger.info("MultinomialNB training complete.")
    return model


# ---------------------------------------------
# 3. Cross-validation
# ---------------------------------------------

def cross_validate_model(
    model: MultinomialNB,
    X_tfidf,
    y: pd.Series,
    cv: int = 5
) -> dict:
    """
    Stratified k-fold cross-validation.

    Args:
        model  : Fitted or unfitted sklearn estimator.
        X_tfidf: Sparse feature matrix (full dataset).
        y      : Labels (full dataset).
        cv     : Number of folds.

    Returns:
        Dictionary of mean CV metrics.
    """
    print_section(f"{cv}-Fold Stratified Cross-Validation")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision", "recall", "f1"]
    cv_results: dict = {}

    for metric in tqdm(scoring, desc="  Cross-validating", unit="metric"):
        scores = cross_val_score(model, X_tfidf, y, cv=skf, scoring=metric, n_jobs=-1)
        mean_s = scores.mean()
        std_s  = scores.std()
        cv_results[metric] = {"mean": mean_s, "std": std_s, "scores": scores.tolist()}
        print(f"  {metric:<12} : {mean_s:.4f} +/- {std_s:.4f}")

    logger.info("Cross-validation complete.")
    return cv_results


# ---------------------------------------------
# 4. Evaluation
# ---------------------------------------------

def evaluate_model(
    model:        MultinomialNB,
    X_train_tfidf,
    X_test_tfidf,
    y_train:      pd.Series,
    y_test:       pd.Series
) -> dict:
    """
    Compute and display all evaluation metrics.

    Args:
        model         : Fitted MultinomialNB.
        X_train_tfidf : Sparse training matrix.
        X_test_tfidf  : Sparse test matrix.
        y_train       : Training labels.
        y_test        : Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    print_section("Model Evaluation")

    y_pred       = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    train_acc = model.score(X_train_tfidf, y_train)
    test_acc  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,        zero_division=0)

    metrics = {
        "train_accuracy" : round(train_acc, 4),
        "test_accuracy"  : round(test_acc,  4),
        "precision"      : round(precision, 4),
        "recall"         : round(recall,    4),
        "f1_score"       : round(f1,        4),
    }

    print_metrics(metrics)

    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Save metrics
    save_metrics_to_csv(metrics)
    save_metrics_to_json(metrics)

    return metrics, y_pred, y_pred_proba


# ---------------------------------------------
# 5. ROC Curve
# ---------------------------------------------

def plot_roc_curve(y_test: pd.Series, y_pred_proba: np.ndarray) -> None:
    """
    Plot and save the ROC curve with AUC annotation.

    Args:
        y_test       : True labels.
        y_pred_proba : Predicted spam probabilities.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", linestyle="--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = get_notebook_path("08_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved --> {path}")
    print(f"  AUC Score : {roc_auc:.4f}")


# ---------------------------------------------
# 6. Precision-Recall Curve
# ---------------------------------------------

def plot_pr_curve(y_test: pd.Series, y_pred_proba: np.ndarray) -> None:
    """
    Plot and save the Precision-Recall curve.

    Args:
        y_test       : True labels.
        y_pred_proba : Predicted spam probabilities.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="#3498db", lw=2,
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color="#3498db")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = get_notebook_path("09_pr_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved --> {path}")
    print(f"  PR-AUC Score : {pr_auc:.4f}")


# ---------------------------------------------
# 7. Confusion matrix
# ---------------------------------------------

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Plot and save a styled confusion matrix heatmap.

    Args:
        y_test : True labels.
        y_pred : Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Ham (0)", "Spam (1)"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = get_notebook_path("10_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved --> {path}")


# ---------------------------------------------
# 8. Accuracy graph
# ---------------------------------------------

def plot_accuracy_graph(train_acc: float, test_acc: float) -> None:
    """
    Plot and save a simple train vs. test accuracy comparison bar chart.

    Args:
        train_acc : Training accuracy (0-1).
        test_acc  : Test accuracy (0-1).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Training Accuracy", "Test Accuracy"],
        [train_acc * 100, test_acc * 100],
        color=["#3498db", "#2ecc71"],
        edgecolor="white",
        linewidth=1.5,
        width=0.4
    )
    for bar, val in zip(bars, [train_acc, test_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val*100:.2f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold"
        )
    ax.set_ylim(80, 102)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training vs. Test Accuracy", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = get_notebook_path("11_accuracy_graph.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Accuracy graph saved --> {path}")


# ---------------------------------------------
# 9. Save model
# ---------------------------------------------

def save_model(model: MultinomialNB, vectorizer, suffix: str = "") -> None:
    """
    Persist the trained model and TF-IDF vectorizer to disk using joblib.

    Args:
        model      : Fitted MultinomialNB.
        vectorizer : Fitted TfidfVectorizer.
        suffix     : Optional filename suffix (e.g., timestamp).
    """
    print_section("Saving Model")

    model_path = get_model_path(f"spam_model{suffix}.pkl")
    vec_path   = get_model_path(f"tfidf_vectorizer{suffix}.pkl")

    joblib.dump(model,      model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"  [OK] Model saved      --> {model_path}")
    print(f"  [OK] Vectorizer saved --> {vec_path}")
    logger.info(f"Model saved      --> {model_path}")
    logger.info(f"Vectorizer saved --> {vec_path}")


# ---------------------------------------------
# Load model helper (used by predict.py & app.py)
# ---------------------------------------------

def load_model(model_name: str = "spam_model.pkl", vec_name: str = "tfidf_vectorizer.pkl"):
    """
    Load a saved model and vectorizer from disk.

    Args:
        model_name : Filename of the model pickle.
        vec_name   : Filename of the vectorizer pickle.

    Returns:
        Tuple (model, vectorizer).

    Raises:
        FileNotFoundError: If the model files are not found.
    """
    model_path = get_model_path(model_name)
    vec_path   = get_model_path(vec_name)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run main.py first to train and save the model."
        )
    if not vec_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {vec_path}. "
            "Please run main.py first to train and save the model."
        )

    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    logger.info(f"Model loaded      --> {model_path}")
    logger.info(f"Vectorizer loaded --> {vec_path}")
    return model, vectorizer
