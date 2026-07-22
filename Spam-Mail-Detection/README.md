# 🛡️ Spam Mail Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25-brightgreen)

An end-to-end **Machine Learning** application that classifies emails and SMS messages as **Spam** or **Ham (Not Spam)** using the **Multinomial Naive Bayes** algorithm with **TF-IDF** feature extraction.

---

## 📌 Table of Contents
1. [Project Description](#-project-description)
2. [Features](#-features)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [How to Run](#-how-to-run)
7. [Model Performance](#-model-performance)
8. [Screenshots](#-screenshots)
9. [Future Improvements](#-future-improvements)
10. [License](#-license)

---

## 📖 Project Description

This project implements a complete spam detection pipeline:

- **Data Collection** — Automatically downloads the SMS Spam Collection dataset (UCI)
- **Text Preprocessing** — Lowercasing, URL/HTML removal, stopword filtering, stemming
- **Feature Engineering** — TF-IDF vectorisation + handcrafted features (length, word count)
- **Model Training** — Multinomial Naive Bayes classifier
- **Evaluation** — Accuracy, Precision, Recall, F1 Score, ROC Curve, Confusion Matrix
- **Web Interface** — Interactive Streamlit app with real-time predictions

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Model** | Multinomial Naive Bayes with Laplace smoothing |
| 📊 **TF-IDF** | Bigram + Unigram features, 5,000 vocabulary |
| 🧹 **Text Cleaning** | URL/HTML removal, stemming, stopword filtering |
| 📈 **EDA Charts** | WordClouds, class distribution, top-word frequency |
| 📉 **Evaluation** | ROC Curve, PR Curve, Confusion Matrix, Cross-validation |
| 🌐 **Streamlit UI** | Dark theme, probability gauge, sample email loader |
| 💾 **Model Saving** | Persistent `.pkl` files using joblib |
| 📋 **Logging** | Full run logs saved to `logs/` directory |
| 📄 **CSV Export** | Evaluation metrics exported to `models/evaluation_results.csv` |
| ⬇️ **Auto-download** | Dataset downloaded automatically if not found locally |

---

## 📂 Dataset

**SMS Spam Collection Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Size: ~5,574 messages
- Balance: ~87% Ham, ~13% Spam
- Columns: `label` (ham/spam), `message` (text)

The dataset is downloaded automatically on first run. If unavailable, the system creates a synthetic dataset to ensure the pipeline runs end-to-end.

---

## 🗂️ Project Structure

```
Spam-Mail-Detection/
│
├── dataset/
│   └── spam.csv                    # SMS Spam Collection dataset
│
├── models/
│   ├── spam_model.pkl              # Trained Multinomial NB model
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│   ├── evaluation_results.csv      # Saved evaluation metrics
│   └── evaluation_results.json     # Metrics in JSON format
│
├── notebooks/
│   ├── 01_class_distribution.png   # EDA charts
│   ├── 02_message_length_dist.png
│   ├── 03_word_count_dist.png
│   ├── 04_wordcloud_spam.png
│   ├── 05_wordcloud_ham.png
│   ├── 06_top20_words_spam.png
│   ├── 07_top20_words_ham.png
│   ├── 08_roc_curve.png
│   ├── 09_pr_curve.png
│   ├── 10_confusion_matrix.png
│   └── 11_accuracy_graph.png
│
├── src/
│   ├── utils.py                    # Logging, directory helpers
│   ├── data_preprocessing.py       # Data loading, cleaning, EDA, TF-IDF
│   ├── train_model.py              # Training, evaluation, model saving
│   └── predict.py                  # Prediction module & CLI
│
├── logs/                           # Auto-generated run logs
│
├── app.py                          # Streamlit web application
├── main.py                         # Pipeline orchestrator
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone or navigate to the project directory
cd "Spam-Mail-Detection"

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the model

```bash
python main.py
```

This will:
1. Automatically download the dataset
2. Clean and preprocess the text
3. Generate all EDA charts (saved to `notebooks/`)
4. Train the Multinomial Naive Bayes model
5. Run 5-fold cross-validation
6. Evaluate and display all metrics
7. Save the model to `models/spam_model.pkl`

Optional flags:
```bash
python main.py --skip-eda        # Skip EDA chart generation
python main.py --no-cross-val    # Skip cross-validation
```

---

### Step 2 — Launch the Streamlit web app

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

### Step 3 — CLI prediction (optional)

```bash
# Interactive mode
python src/predict.py

# Single email
python src/predict.py --text "Congratulations! You won a free iPhone!"

# Run sample predictions
python src/predict.py --sample
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~98.0% |
| **Precision** | ~97.5% |
| **Recall** | ~95.2% |
| **F1 Score** | ~96.3% |
| **AUC-ROC** | ~99.1% |

> *Results may vary slightly depending on dataset version and random seed.*

---

## 🖼️ Screenshots

| | |
|--|--|
| **Web App — Ham Result** | **Web App — Spam Result** |
| *(Green result card, low spam probability)* | *(Red result card, high spam probability)* |

---

## 🔮 Future Improvements

- [ ] **Deep Learning** — Implement LSTM / BERT for improved accuracy
- [ ] **Multi-language support** — Extend to non-English spam detection
- [ ] **Real-time email integration** — Connect to Gmail/Outlook API
- [ ] **Active learning** — Allow user feedback to retrain the model
- [ ] **Ensemble methods** — Combine NB + SVM + XGBoost
- [ ] **Explainability** — LIME/SHAP for prediction explanations
- [ ] **Docker** — Containerise the application for easy deployment
- [ ] **REST API** — FastAPI backend for third-party integrations
- [ ] **Model versioning** — MLflow experiment tracking
- [ ] **Dashboard** — Admin panel with live prediction logs

---

## 📚 Dependencies

```
numpy, pandas, scikit-learn, nltk, matplotlib, seaborn,
wordcloud, tqdm, joblib, streamlit, requests
```

Install all with: `pip install -r requirements.txt`

---

## 👨‍💻 Author

**Spam Mail Detection System**  
College Mini Project — Machine Learning  
Algorithm: Multinomial Naive Bayes  
Framework: scikit-learn + Streamlit

---

## 📄 License

This project is licensed under the **MIT License** — free for academic and personal use.
