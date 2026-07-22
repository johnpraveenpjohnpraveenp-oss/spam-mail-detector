"""
utils.py -- Utility functions for the Spam Mail Detection System.

Provides:
  - Logger setup
  - Directory creation helpers
  - Progress display utilities
  - Results persistence
"""

import os
import logging
import json
import csv
from datetime import datetime
from pathlib import Path


# ---------------------------------------------
# Constants
# ---------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR   = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
LOG_DIR     = BASE_DIR / "logs"

# ---------------------------------------------
# Logger
# ---------------------------------------------

def setup_logger(name: str = "spam_detector", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger that writes to both console and a log file.

    Args:
        name  : Logger name.
        level : Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    ensure_directories()

    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            "%(asctime)s  [%(levelname)-8s]  %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_fmt)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s  [%(levelname)-8s]  %(name)s  --  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------
# Directory helpers
# ---------------------------------------------

def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for directory in [DATASET_DIR, MODEL_DIR, NOTEBOOK_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_dataset_path(filename: str = "spam.csv") -> Path:
    """Return the absolute path to a dataset file."""
    return DATASET_DIR / filename


def get_model_path(filename: str = "spam_model.pkl") -> Path:
    """Return the absolute path to a model file."""
    return MODEL_DIR / filename


def get_notebook_path(filename: str) -> Path:
    """Return the absolute path to a notebooks artefact."""
    return NOTEBOOK_DIR / filename


# ---------------------------------------------
# Results persistence
# ---------------------------------------------

def save_metrics_to_csv(metrics: dict, filename: str = "evaluation_results.csv") -> None:
    """
    Persist evaluation metrics to a CSV file inside the models/ directory.

    Args:
        metrics  : Dictionary of metric_name --> value.
        filename : Output CSV filename.
    """
    output_path = MODEL_DIR / filename
    fieldnames  = list(metrics.keys())

    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"\n[OK] Evaluation results saved --> {output_path}")


def save_metrics_to_json(metrics: dict, filename: str = "evaluation_results.json") -> None:
    """
    Persist evaluation metrics to a JSON file inside the models/ directory.

    Args:
        metrics  : Dictionary of metric_name --> value.
        filename : Output JSON filename.
    """
    output_path = MODEL_DIR / filename
    with open(output_path, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=4)
    print(f"[OK] Metrics JSON saved       --> {output_path}")


# ---------------------------------------------
# Pretty printing helpers
# ---------------------------------------------

def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to stdout."""
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print_section("Model Evaluation Metrics")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25} {value:.4f}")
        else:
            print(f"  {key:<25} {value}")
    print()
