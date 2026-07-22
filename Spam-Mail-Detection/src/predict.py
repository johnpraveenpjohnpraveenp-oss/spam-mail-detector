"""
predict.py -- Prediction module for the Spam Mail Detection System.

Usage (CLI):
    python src/predict.py

Usage (import):
    from src.predict import predict_email
    result = predict_email("Congratulations! You won a prize.")
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils         import setup_logger
from train_model   import load_model
from data_preprocessing import clean_text, download_nltk_resources

logger = setup_logger("predict")


# ---------------------------------------------
# Core prediction function
# ---------------------------------------------

def predict_email(
    raw_text:   str,
    model=None,
    vectorizer=None
) -> dict:
    """
    Predict whether an email / SMS is Spam or Ham.

    Args:
        raw_text   : The raw email text to classify.
        model      : Pre-loaded MultinomialNB (optional; loaded from disk if None).
        vectorizer : Pre-loaded TfidfVectorizer (optional; loaded from disk if None).

    Returns:
        Dictionary with keys:
          - prediction   : "Spam" or "Ham"
          - label        : 1 (spam) or 0 (ham)
          - spam_prob    : float probability of spam (0-1)
          - ham_prob     : float probability of ham (0-1)
          - clean_text   : Cleaned version of the input text

    Raises:
        ValueError      : If raw_text is empty.
        FileNotFoundError: If model files are not found on disk.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Email text cannot be empty.")

    # Lazy-load model if not provided
    if model is None or vectorizer is None:
        logger.info("Loading model from disk ...")
        model, vectorizer = load_model()

    # Clean the input text
    cleaned = clean_text(raw_text)

    # Vectorise
    features = vectorizer.transform([cleaned])

    # Predict
    label        = int(model.predict(features)[0])
    proba        = model.predict_proba(features)[0]
    spam_prob    = float(proba[1])
    ham_prob     = float(proba[0])
    prediction   = "Spam" if label == 1 else "Ham"

    result = {
        "prediction" : prediction,
        "label"      : label,
        "spam_prob"  : round(spam_prob * 100, 2),
        "ham_prob"   : round(ham_prob  * 100, 2),
        "clean_text" : cleaned,
    }

    logger.info(
        f"Prediction: {prediction}  "
        f"(spam={spam_prob:.2%}, ham={ham_prob:.2%})"
    )
    return result


# ---------------------------------------------
# CLI interface
# ---------------------------------------------

def _print_result(result: dict) -> None:
    """Pretty-print a prediction result to stdout."""
    bar = "-" * 50
    print(f"\n{'='*50}")
    if result["prediction"] == "Spam":
        print(f"  ?  RESULT : SPAM  ?")
    else:
        print(f"  [OK]  RESULT : NOT SPAM (Ham)  [OK]")
    print(f"{'='*50}")
    print(f"  Spam Probability : {result['spam_prob']:.2f}%")
    print(f"  Ham  Probability : {result['ham_prob']:.2f}%")
    print(f"{bar}")


def _run_cli() -> None:
    """Interactive CLI loop for email prediction."""
    # Ensure NLTK data is present
    download_nltk_resources()

    print("\n" + "="*55)
    print("   Spam Mail Detection System -- Prediction CLI")
    print("="*55)
    print("  Type an email message below and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")

    # Pre-load model once
    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print(f"\n[ERR] Error: {e}")
        print("  Please run 'python main.py' first to train the model.")
        sys.exit(1)

    while True:
        try:
            print()
            raw = input("  Enter Email: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! ?")
            break

        if raw.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! ?")
            break

        if not raw:
            print("  [!]  Please enter some text.")
            continue

        try:
            result = predict_email(raw, model=model, vectorizer=vectorizer)
            _print_result(result)
        except Exception as exc:
            print(f"  [ERR] Prediction error: {exc}")


# ---------------------------------------------
# Sample batch prediction (for testing)
# ---------------------------------------------

def run_sample_predictions() -> None:
    """Run a set of known spam/ham examples and display results."""
    samples = [
        ("CONGRATULATIONS! You have won a FREE iPhone. Click here to claim now!", "Spam"),
        ("Hey, are you coming to dinner tonight? I made your favourite pasta.", "Ham"),
        ("URGENT: Your bank account has been compromised. Call 0800 immediately!", "Spam"),
        ("The project report is due tomorrow. Can you send me your section?", "Ham"),
        ("You are a WINNER! Reply WIN to claim GBP1000 cash prize. No purchase needed.", "Spam"),
        ("Happy birthday! Hope you have a wonderful day surrounded by loved ones.", "Ham"),
        ("Free entry to win a brand new car! Text WIN to 80085.", "Spam"),
        ("I'll be 10 minutes late to the meeting. Go ahead and start without me.", "Ham"),
    ]

    try:
        model, vectorizer = load_model()
    except FileNotFoundError as e:
        print(f"\n[ERR] {e}")
        return

    print("\n" + "="*65)
    print("   Sample Predictions")
    print("="*65)
    print(f"  {'Text':<45} {'Expected':<8} {'Got':<8} {'Spam%':>6}")
    print("  " + "-"*61)

    for text, expected in samples:
        result = predict_email(text, model=model, vectorizer=vectorizer)
        status = "[OK]" if result["prediction"] == expected else "[ERR]"
        short_text = text[:43] + ".." if len(text) > 43 else text
        print(f"  {status} {short_text:<45} {expected:<8} {result['prediction']:<8} {result['spam_prob']:>5.1f}%")

    print("="*65)


# ---------------------------------------------
# Entry point
# ---------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spam Mail Detection -- Prediction CLI")
    parser.add_argument(
        "--sample", action="store_true",
        help="Run sample predictions instead of interactive mode."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Predict a single email passed as a command-line argument."
    )
    args = parser.parse_args()

    download_nltk_resources()

    if args.sample:
        run_sample_predictions()
    elif args.text:
        try:
            model, vectorizer = load_model()
            result = predict_email(args.text, model=model, vectorizer=vectorizer)
            _print_result(result)
        except FileNotFoundError as e:
            print(f"\n[ERR] {e}")
            sys.exit(1)
    else:
        _run_cli()
