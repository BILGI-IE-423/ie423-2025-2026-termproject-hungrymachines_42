# IE 423 2025–2026 Term Project — Sentiment Analysis of IMDB Movie Reviews Using NLP and Machine Learning

## Live Website
https://bilgi-ie-423.github.io/ie423-2025-2026-termproject-hungrymachines_42/

## Team Members

* Bersu Yılmaz — 123203069
* Emirhan Karaca — 122203009
* Mert Ada Demirbaş — 123203026

## Dataset

* **Name:** IMDB Dataset of 50K Movie Reviews
* **Source:** [Kaggle — lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* See `data/README.md` for download and placement instructions.

## Project Objective

This project builds an end-to-end Natural Language Processing pipeline for classifying IMDB movie reviews as positive or negative, and uses that pipeline to answer three research questions:

1. **RQ1 — Reliable & interpretable classification.** Can traditional machine learning algorithms reliably and interpretably classify sentiment to support first-stage quality control in customer feedback loops?
2. **RQ2 — Length vs. sentiment.** Is there a statistically significant correlation between review length and sentiment polarity, indicating whether dissatisfied viewers write more exhaustive reviews?
3. **RQ3 — Error patterns.** Are misclassified reviews concentrated in specific structural or behavioral patterns, such as review length, use of rare words, or emotionally mixed content?

The pipeline covers text cleaning, negation tagging, lemmatization, statistical feature engineering, exploratory analysis, supervised modeling with hyperparameter tuning, statistical testing, and error analysis on the best model.

## Repository Structure

```
├── README.md                          → main project overview (this file)
├── index.html                         → published website (GitHub Pages)
├── requirements.txt                   → Python dependencies
├── .gitignore                         → excludes raw/processed CSVs and OS/IDE files
├── data/
│   ├── raw/                           → raw dataset (not committed; see data/README.md)
│   ├── processed/                     → cleaned dataset produced by 02_preprocess_data.py
│   └── README.md                      → dataset access and placement instructions
├── scripts/
│   ├── 01_load_data.py                → loads the raw dataset and prints basic info
│   ├── 02_preprocess_data.py          → text cleaning, lemmatization, feature engineering
│   ├── 03_basic_eda.py                → summary tables and exploratory visualizations
│   ├── 04_modeling.py                 → RQ1: baseline + 3 ML models with grid-search CV
│   ├── 05_length_sentiment.py         → RQ2: statistical tests on length vs. sentiment
│   └── 06_error_analysis.py           → RQ3: structural/behavioral patterns of errors
├── models/
│   ├── best_model.pkl                 → best fitted classifier (used by 06_error_analysis.py)
│   ├── best_vectorizer.pkl            → matching fitted vectorizer
│   └── best_model_meta.csv            → summary metadata of the chosen configuration
├── visuals/
│   ├── figures/                       → all PNG figures produced by the scripts
│   └── tables/                        → all CSV summary tables produced by the scripts

```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended. The first run will download a few small NLTK resources (`wordnet`, `stopwords`, `punkt`, etc.) automatically.

## Running the Scripts

All scripts are run from the repository root so that relative paths resolve correctly:

```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
python scripts/04_modeling.py
python scripts/05_length_sentiment.py
python scripts/06_error_analysis.py
```

The scripts are designed to run in this order. Each step writes its outputs to disk and the next step reads from those outputs:

* `01_load_data.py` validates that `data/raw/IMDB Dataset.csv` is present and prints structural information.
* `02_preprocess_data.py` writes `data/processed/cleaned_data_set.csv` (49,582 rows after deduplication and empty-text filtering).
* `03_basic_eda.py` writes 6 figures and 2 tables to `visuals/`.
* `04_modeling.py` trains a baseline `DummyClassifier` and three ML models (Logistic Regression, Multinomial Naive Bayes, Linear SVM) on two text representations (TF-IDF and Bag-of-Words) with 5-fold stratified grid-search cross-validation; it writes 4 figures, 3 tables, and saves the best model + vectorizer under `models/`. Approximate runtime: ~5 minutes on a modern laptop.
* `05_length_sentiment.py` runs Pearson, Spearman, point-biserial, and Mann-Whitney U tests on review length vs. sentiment, plus a single-feature logistic regression baseline; writes 3 figures and 2 tables.
* `06_error_analysis.py` reloads the saved best model, reproduces the test split, and compares structural and behavioral feature distributions across correct vs. misclassified reviews (and across TP / TN / FP / FN groups); writes 3 figures and 1 table.

## Results Summary

**RQ1 — Reliable & interpretable classification.** Linear SVM with TF-IDF features was the best configuration, reaching **89.85% test accuracy**, **F1 = 0.8999**, and **ROC-AUC = 0.9645** — roughly 40 percentage points above the most-frequent-class baseline (50.19%). Logistic Regression performed competitively (F1 = 0.8986) and its coefficients yielded a directly interpretable lexicon of the 20 most positive and 20 most negative predictive words.

**RQ2 — Length vs. sentiment.** All three correlation tests were statistically significant at α = 0.05 (Pearson r = +0.011, Spearman r = −0.009, point-biserial r = +0.011), but their effect sizes are practically negligible. A logistic regression using only `word_count` reached **49.4% test accuracy** — worse than chance. We conclude that review length carries essentially no usable signal about sentiment, and that the significant p-values are an artifact of the large sample size rather than a meaningful association.

**RQ3 — Error patterns.** Misclassified reviews concentrate in two detectable behavioral patterns. Compared to correctly classified reviews, they use about **25% more rare words** (tokens in the bottom 5% of training document frequency) and score about **15% higher on a mixed-sentiment metric** that measures co-occurrence of top-positive and top-negative lexicon terms. Surface-level structural features (length, lexical diversity, ALL-CAPS count) do **not** discriminate between correct and incorrect predictions, reinforcing the RQ2 finding that surface structure is uninformative.
