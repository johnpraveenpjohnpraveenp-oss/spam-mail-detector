"""
main.py -- Orchestrator for the Spam Mail Detection pipeline.

Run this file to execute the complete workflow:
  1. Load & explore the dataset
  2. Clean and preprocess text
  3. Engineer features
  4. Run EDA (save charts)
  5. Vectorise with TF-IDF
  6. Train / test split
  7. Train Multinomial Naive Bayes
  8. Cross-validate
  9. Evaluate (accuracy, precision, recall, F1)
 10. Plot ROC, PR, Confusion Matrix, Accuracy graph
 11. Save model + vectorizer to disk

Usage:
    python main.py
    python main.py --skip-eda     # skip EDA charts
    python main.py --no-cross-val # skip cross-validation
"""

# -- Windows UTF-8 fix (must be before any print) ------------
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
import argparse
import logging
from pathlib import Path

# --- Ensure src/ is importable --------------
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import setup_logger, print_section, ensure_directories
from data_preprocessing import (
    download_nltk_resources,
    load_dataset,
    preprocess_dataframe,
    engineer_features,
    run_eda,
    vectorize_text,
)
from train_model import (
    split_data,
    train_model,
    cross_validate_model,
    evaluate_model,
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_accuracy_graph,
    save_model,
)

logger = setup_logger("main")


# ---------------------------------------------
# Full pipeline
# ---------------------------------------------

def run_pipeline(skip_eda: bool = False, skip_cross_val: bool = False) -> None:
    """
    Execute the complete spam detection pipeline end-to-end.

    Args:
        skip_eda       : If True, skip EDA chart generation.
        skip_cross_val : If True, skip cross-validation step.
    """
    start_time = time.time()

    print("\n" + "=" * 62)
    print("   [*]  Spam Mail Detection System                        ")
    print("         Multinomial Naive Bayes Classifier               ")
    print("=" * 62 + "\n")

    # -- Step 0: Setup ------------------------
    ensure_directories()
    download_nltk_resources()
    logger.info("Pipeline started.")

    # -- Step 1: Load dataset -----------------
    print_section("Step 1 -- Load Dataset")
    df = load_dataset()

    # -- Step 2: Preprocess -------------------
    print_section("Step 2 -- Data Cleaning & Preprocessing")
    df = preprocess_dataframe(df)

    # -- Step 3: Feature engineering ----------
    print_section("Step 3 -- Feature Engineering")
    df = engineer_features(df)

    # -- Step 4: EDA --------------------------
    if not skip_eda:
        print_section("Step 4 -- Exploratory Data Analysis")
        run_eda(df)
    else:
        logger.info("EDA skipped (--skip-eda flag).")

    # -- Step 5: Train/test split -------------
    print_section("Step 5 -- Train / Test Split")
    X = df["clean_message"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    # -- Step 6: TF-IDF -----------------------
    print_section("Step 6 -- TF-IDF Vectorisation")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    # -- Step 7: Train model ------------------
    print_section("Step 7 -- Model Training")
    model = train_model(X_train_tfidf, y_train)

    # -- Step 8: Cross-validation -------------
    if not skip_cross_val:
        print_section("Step 8 -- Cross-Validation")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        # Re-vectorise full dataset for CV
        cv_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True
        )
        X_full_tfidf = cv_vectorizer.fit_transform(X)
        cv_model = MultinomialNB(alpha=0.1)
        cv_results = cross_validate_model(cv_model, X_full_tfidf, y)
    else:
        logger.info("Cross-validation skipped (--no-cross-val flag).")

    # -- Step 9: Evaluate ---------------------
    print_section("Step 9 -- Model Evaluation")
    metrics, y_pred, y_pred_proba = evaluate_model(
        model, X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # -- Step 10: Plot evaluation charts ------
    print_section("Step 10 -- Generating Evaluation Charts")
    plot_roc_curve(y_test, y_pred_proba)
    plot_pr_curve(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_accuracy_graph(metrics["train_accuracy"], metrics["test_accuracy"])
    print("  All charts saved to notebooks/")

    # -- Step 11: Save model ------------------
    print_section("Step 11 -- Saving Model")
    save_model(model, vectorizer)

    # -- Done ---------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 62)
    print(f"   [OK]  Pipeline completed in {elapsed:.1f}s")
    print(f"   [**]  Test Accuracy : {metrics['test_accuracy']*100:.2f}%")
    print(f"   [**]  F1 Score      : {metrics['f1_score']*100:.2f}%")
    print(f"   [**]  Model saved   : models/spam_model.pkl")
    print(f"   [>>]  Run UI        : streamlit run app.py")
    print("=" * 62 + "\n")

    logger.info(
        f"Pipeline complete -- acc={metrics['test_accuracy']:.4f}, "
        f"f1={metrics['f1_score']:.4f}, elapsed={elapsed:.1f}s"
    )


# ---------------------------------------------
# Entry point
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spam Mail Detection System -- Full Training Pipeline"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA chart generation to speed up training."
    )
    parser.add_argument(
        "--no-cross-val", action="store_true",
        help="Skip cross-validation step."
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            skip_eda=args.skip_eda,
            skip_cross_val=args.no_cross_val
        )
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
