"""
data_preprocessing.py -- Data loading, cleaning, EDA, and feature engineering.

Pipeline:
  1. load_dataset()        -- download / load CSV, display stats
  2. clean_text()          -- full NLP text cleaning
  3. preprocess_dataframe()-- apply cleaning to the DataFrame
  4. engineer_features()   -- add length / word-count / char-count columns
  5. run_eda()             -- generate & save all EDA charts
  6. vectorize_text()      -- TF-IDF transformation
"""

import os
import re
import string
import logging
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Local utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    setup_logger, ensure_directories,
    get_dataset_path, get_notebook_path,
    print_section, NOTEBOOK_DIR
)

# ---------------------------------------------
# Module-level logger
# ---------------------------------------------
logger = setup_logger("data_preprocessing")

# ---------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------

def download_nltk_resources() -> None:
    """Download required NLTK data packages if not already present."""
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


# ---------------------------------------------
# 1. Dataset loading
# ---------------------------------------------

DATASET_URLS = [
    # UCI SMS Spam Collection (tab-separated, no header)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    # Raw GitHub mirror
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]

KAGGLE_RAW_URL = (
    "https://raw.githubusercontent.com/mohitgupta-omg/"
    "Kaggle-SMS-Spam-Collection-Dataset/master/spam.csv"
)


def _try_download_github_mirror(save_path: Path) -> bool:
    """
    Attempt to download the spam CSV from a public GitHub mirror.

    Args:
        save_path: Where to write the file.

    Returns:
        True if successful, False otherwise.
    """
    urls = [
        KAGGLE_RAW_URL,
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/smsspamcollection/spam.csv",
    ]
    for url in urls:
        try:
            logger.info(f"Trying --> {url}")
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"Dataset downloaded --> {save_path}")
            return True
        except Exception as exc:
            logger.warning(f"Failed ({url}): {exc}")
    return False


def _create_synthetic_dataset(save_path: Path) -> None:
    """
    Generate a minimal synthetic spam/ham dataset as a last-resort fallback.

    Args:
        save_path: Where to write the CSV.
    """
    logger.warning("Creating synthetic dataset as fallback ...")
    spam_messages = [
        "Free entry in 2 a weekly competition to win FA Cup final tkts!",
        "You have won a $1000 gift card. Click here to claim now!",
        "Congratulations! You've been selected for a free iPhone.",
        "URGENT: Your account has been compromised. Call 0800 now!",
        "Get cheap meds online. No prescription needed. Order now!",
        "You are a winner! Reply WIN to claim your prize money.",
        "Special offer! Buy one get one free. Limited time only!",
        "Your loan has been approved. Claim your cash now!",
        "Hot singles in your area want to meet you tonight!",
        "Call now to claim your free holiday worth $5000!",
        "SMS alert: You have 1 unread message. Click to read.",
        "Win a brand new car! Enter our prize draw today!",
        "Exclusive deal for you! Save 90% on all products.",
        "Alert! Your package could not be delivered. Click to reschedule.",
        "You are pre-approved for a $50,000 loan with 0% interest!",
        "Claim your free gift voucher worth $200. Expires soon!",
        "WINNER! You have been selected as a lucky winner. Reply YES.",
        "Make money from home! Earn $500 per day working online.",
        "Your number has been selected for a cash prize of $10000!",
        "Free ringtones! Text STOP to opt out.",
    ] * 25  # 500 spam

    ham_messages = [
        "Hey, are you coming to the party tonight?",
        "I'll be home late today. Don't wait for dinner.",
        "Can you pick up some groceries on your way home?",
        "The meeting has been rescheduled to 3 PM tomorrow.",
        "Happy birthday! Hope you have a wonderful day.",
        "I'll call you back in 10 minutes.",
        "Did you finish the project report?",
        "Let's catch up over coffee this weekend.",
        "The kids had a great time at school today.",
        "Can you send me the document you mentioned?",
        "I'm running 15 minutes late. Sorry!",
        "Thanks for the gift. It was very thoughtful.",
        "Are you free for lunch tomorrow?",
        "I'll be in the office by 9 AM.",
        "The weather looks great for a weekend trip.",
        "Please remind me to call the dentist.",
        "I just got back from the gym. Feeling great!",
        "What time does the movie start?",
        "Don't forget to take your medicine.",
        "See you at the airport at 6 AM.",
    ] * 55  # 1100 ham

    data = (
        [{"label": "spam", "message": m} for m in spam_messages] +
        [{"label": "ham",  "message": m} for m in ham_messages]
    )
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic dataset saved --> {save_path}  ({len(df)} rows)")


def load_dataset(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the spam dataset from disk, downloading it if necessary.

    Args:
        filepath: Optional explicit path to the CSV.

    Returns:
        Raw DataFrame with columns [label, message].
    """
    ensure_directories()

    if filepath is None:
        filepath = get_dataset_path("spam.csv")
    else:
        filepath = Path(filepath)

    # -- Download if missing ----------------------
    if not filepath.exists():
        logger.info("Dataset not found locally. Attempting download ...")
        success = _try_download_github_mirror(filepath)
        if not success:
            _create_synthetic_dataset(filepath)

    # -- Load CSV ---------------------------------
    logger.info(f"Loading dataset from {filepath}")

    # Handle multiple possible CSV formats
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
    except Exception:
        df = pd.read_csv(filepath, encoding="utf-8")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map various column name conventions --> (label, message)
    col_map = {}
    for col in df.columns:
        if col in ("v1", "label", "class", "category", "spam"):
            col_map[col] = "label"
        elif col in ("v2", "message", "text", "email", "sms"):
            col_map[col] = "message"
    df = df.rename(columns=col_map)

    # Drop any extra columns
    keep = [c for c in ["label", "message"] if c in df.columns]
    df = df[keep].copy()

    # -- Display basic stats -----------------------
    print_section("Dataset Overview")
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print("\n  Dataset Info:")
    df.info()
    print("\n  Class Distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")

    logger.info(f"Dataset loaded -- {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------
# 2. Text cleaning
# ---------------------------------------------

# Initialise stemmer and stopwords once (module-level, for speed)
_stemmer = PorterStemmer()
_stop_words: set[str] = set()


def _ensure_stopwords() -> None:
    """Lazy-load NLTK stopwords (called before first use)."""
    global _stop_words
    if not _stop_words:
        download_nltk_resources()
        _stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Full NLP text cleaning pipeline for a single email / SMS.

    Steps:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove numbers
      5. Remove punctuation & special characters
      6. Tokenise
      7. Remove stopwords
      8. Stem with PorterStemmer

    Args:
        text: Raw email / SMS string.

    Returns:
        Cleaned, stemmed string.
    """
    _ensure_stopwords()

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # 4. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 5. Remove numbers
    text = re.sub(r"\d+", " ", text)

    # 6. Remove punctuation & special characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)

    # 7. Tokenise
    tokens = text.split()

    # 8. Remove stopwords + stem
    tokens = [
        _stemmer.stem(word)
        for word in tokens
        if word not in _stop_words and len(word) > 1
    ]

    return " ".join(tokens)


# ---------------------------------------------
# 3. DataFrame preprocessing
# ---------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame: remove nulls, duplicates, encode labels, apply text cleaning.

    Args:
        df: Raw DataFrame with [label, message] columns.

    Returns:
        Cleaned DataFrame.
    """
    print_section("Data Cleaning")
    initial_shape = df.shape

    # Remove nulls
    df = df.dropna(subset=["label", "message"])
    logger.info(f"After dropna        : {df.shape[0]} rows")

    # Remove empty messages
    df = df[df["message"].str.strip().str.len() > 0]
    logger.info(f"After empty removal : {df.shape[0]} rows")

    # Remove duplicates
    df = df.drop_duplicates(subset=["message"])
    logger.info(f"After dedup         : {df.shape[0]} rows")

    df = df.reset_index(drop=True)

    # Encode labels: ham --> 0, spam --> 1
    label_col = df["label"].str.lower().str.strip()
    df["label_enc"] = label_col.map({"ham": 0, "spam": 1})

    # Handle datasets that store labels as 0/1 already
    if df["label_enc"].isnull().any():
        try:
            df["label_enc"] = pd.to_numeric(label_col)
        except Exception:
            df["label_enc"] = (label_col != "ham").astype(int)

    df["label_enc"] = df["label_enc"].astype(int)

    removed = initial_shape[0] - df.shape[0]
    print(f"  Rows removed during cleaning : {removed}")
    print(f"  Final dataset shape          : {df.shape}")
    print(f"\n  Label encoding: ham-->0, spam-->1")
    print(df["label_enc"].value_counts().to_string())

    # Apply text cleaning (with progress display)
    logger.info("Cleaning text ... (this may take a moment)")
    from tqdm import tqdm
    tqdm.pandas(desc="  Cleaning messages")
    df["clean_message"] = df["message"].progress_apply(clean_text)

    logger.info("Text cleaning complete.")
    return df


# ---------------------------------------------
# 4. Feature engineering
# ---------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hand-crafted numerical features to the DataFrame.

    New columns:
      - msg_length   : character count of original message
      - word_count   : word count of original message
      - char_count   : non-space character count
      - avg_word_len : average word length

    Args:
        df: DataFrame with [message] column.

    Returns:
        DataFrame with additional feature columns.
    """
    print_section("Feature Engineering")

    df["msg_length"]   = df["message"].apply(len)
    df["word_count"]   = df["message"].apply(lambda x: len(x.split()))
    df["char_count"]   = df["message"].apply(lambda x: len(x.replace(" ", "")))
    df["avg_word_len"] = df["char_count"] / (df["word_count"] + 1e-6)

    print("  New features added:")
    for feat in ["msg_length", "word_count", "char_count", "avg_word_len"]:
        print(f"    {feat}")

    print("\n  Feature statistics (spam vs ham):")
    print(
        df.groupby("label_enc")[["msg_length", "word_count", "char_count"]]
        .mean()
        .rename(index={0: "Ham", 1: "Spam"})
        .round(2)
        .to_string()
    )
    return df


# ---------------------------------------------
# 5. EDA -- charts
# ---------------------------------------------

def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to the notebooks directory."""
    path = get_notebook_path(name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved --> {path}")


def run_eda(df: pd.DataFrame) -> None:
    """
    Generate and save all EDA charts:
      1. Class distribution (pie + bar)
      2. Message length distribution by class
      3. Word count distribution by class
      4. WordCloud -- Spam
      5. WordCloud -- Ham
      6. Top 20 frequent spam words
      7. Top 20 frequent ham words

    Args:
        df: Preprocessed DataFrame.
    """
    print_section("Exploratory Data Analysis")

    # Palette
    palette = {"Ham": "#2ecc71", "Spam": "#e74c3c"}
    df["Label"] = df["label_enc"].map({0: "Ham", 1: "Spam"})

    # -- 1. Class distribution -----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=16, fontweight="bold")

    counts = df["Label"].value_counts()
    axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[palette[l] for l in counts.index],
        startangle=140, wedgeprops=dict(edgecolor="white")
    )
    axes[0].set_title("Proportion")

    sns.countplot(data=df, x="Label", palette=palette, ax=axes[1])
    axes[1].set_title("Count")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of messages")
    for p in axes[1].patches:
        axes[1].annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom", fontsize=12
        )
    plt.tight_layout()
    _save_fig(fig, "01_class_distribution.png")

    # -- 2. Message length distribution -------
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, grp in df.groupby("Label"):
        ax.hist(
            grp["msg_length"], bins=50, alpha=0.6,
            label=label, color=palette[label]
        )
    ax.set_title("Message Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "02_message_length_dist.png")

    # -- 3. Word count distribution ------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, grp in df.groupby("Label"):
        ax.hist(
            grp["word_count"], bins=40, alpha=0.6,
            label=label, color=palette[label]
        )
    ax.set_title("Word Count Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "03_word_count_dist.png")

    # -- 4 & 5. WordClouds ---------------------
    try:
        from wordcloud import WordCloud

        for label, color, idx in [("Spam", "Reds", "04"), ("Ham", "Greens", "05")]:
            text_blob = " ".join(df[df["Label"] == label]["clean_message"].dropna())
            if not text_blob.strip():
                continue
            wc = WordCloud(
                width=1200, height=600,
                background_color="white",
                colormap=color,
                max_words=200,
                collocations=False
            ).generate(text_blob)
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Word Cloud -- {label} Messages", fontsize=16, fontweight="bold")
            plt.tight_layout()
            _save_fig(fig, f"{idx}_wordcloud_{label.lower()}.png")

    except ImportError:
        logger.warning("wordcloud not installed -- skipping WordCloud plots.")

    # -- 6 & 7. Top-20 frequent words ---------
    for label, color, idx in [("Spam", "#e74c3c", "06"), ("Ham", "#2ecc71", "07")]:
        subset = df[df["Label"] == label]["clean_message"].dropna()
        all_words = " ".join(subset).split()
        if not all_words:
            continue
        freq = pd.Series(all_words).value_counts().head(20)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=freq.values, y=freq.index, ax=ax, color=color)
        ax.set_title(f"Top 20 Frequent Words -- {label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        plt.tight_layout()
        _save_fig(fig, f"{idx}_top20_words_{label.lower()}.png")

    print("  All EDA charts saved to notebooks/")


# ---------------------------------------------
# 6. TF-IDF vectorisation
# ---------------------------------------------

def vectorize_text(
    X_train: pd.Series,
    X_test:  pd.Series,
    max_features: int = 5000
) -> tuple:
    """
    Fit a TF-IDF vectoriser on training data and transform both splits.

    Args:
        X_train      : Training text series.
        X_test       : Test text series.
        max_features : Maximum vocabulary size.

    Returns:
        Tuple of (X_train_tfidf, X_test_tfidf, fitted_vectorizer).
    """
    print_section("TF-IDF Vectorisation")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),         # unigrams + bigrams
        sublinear_tf=True,          # apply log normalisation
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"  Vocabulary size   : {len(vectorizer.vocabulary_)}")
    print(f"  Training matrix   : {X_train_tfidf.shape}")
    print(f"  Test matrix       : {X_test_tfidf.shape}")

    logger.info("TF-IDF vectorisation complete.")
    return X_train_tfidf, X_test_tfidf, vectorizer
