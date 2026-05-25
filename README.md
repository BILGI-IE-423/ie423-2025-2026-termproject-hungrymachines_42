# IE 423 2025-2026 Term Project Proposal — Sentiment Analysis of IMDB Movie Reviews Using NLP and Machine Learning

## Team Members

- Bersu Yılmaz — 123203069
- Emirhan Karaca — 122203009
- Mert Ada Demirbaş — 123203026

## Dataset

- **Name:** IMDB Dataset of 50K Movie Reviews
- **Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- See [`data/README.md`](data/README.md) for download and placement instructions.

## Project Objective

This project builds an end-to-end Natural Language Processing pipeline for classifying IMDB movie reviews as positive or negative. The focus is on understanding how different text preprocessing techniques — lemmatization, stop-word removal, and negation tagging — affect the performance of machine learning models. We will compare multiple algorithms and analyze feature importance to identify which words and phrases contribute most to sentiment classification.

## Repository Structure

```
├── README.md                          → main project overview
├── requirements.txt                   → Python dependencies
├── data/
│   ├── raw/                           → raw dataset (not committed; see data/README.md)
│   ├── processed/                     → cleaned dataset produced by 02_preprocess_data.py
│   └── README.md                      → dataset access and placement instructions
├── scripts/
│   ├── 01_load_data.py                → loads the raw dataset and prints basic info
│   ├── 02_preprocess_data.py          → text cleaning, lemmatization, feature engineering
│   └── 03_basic_eda.py                → generates summary tables and visualizations
├── outputs/
│   ├── figures/                       → PNG charts produced by 03_basic_eda.py
│   └── tables/                        → CSV summary tables produced by 03_basic_eda.py
└── docs/
    └── ResearchProposalPreprocessing.md   → main proposal and presentation document
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Scripts

Run from the **repository root** (so that relative paths resolve correctly):

```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

`01_load_data.py` validates the raw dataset, `02_preprocess_data.py` produces `data/processed/cleaned_data_set.csv`, and `03_basic_eda.py` writes all figures to `outputs/figures/` and all summary tables to `outputs/tables/`.

## Proposal Document

The full proposal — research questions, preprocessing walkthrough, and initial outputs — is in [`docs/ResearchProposalPreprocessing.md`](docs/ResearchProposalPreprocessing.md).
