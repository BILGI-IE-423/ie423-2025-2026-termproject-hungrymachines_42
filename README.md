# IE 423 2025-2026 Term Project Proposal - Sentiment Analysis of IMDB Movie Reviews Using NLP and Machine Learning

# Team Members

* Bersu Yılmaz - 123203069

* Emirhan Karaca - 122203009

* Mert Ada Demirbaş - 123203026

# Dataset

Dataset: IMDB Dataset of 50K Movie Reviews

Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Project Objective
This project aims to build an effective Natural Language Processing pipeline for analyzing and classifying IMDB movie reviews as positive or negative. The primary focus of our work is to understand how different text preprocessing techniques, such as lemmatization and stop-word removal, impact the performance of machine learning models. We will evaluate and compare various algorithms to find the best fit for our textual data. Additionally, we plan to analyze the feature importance to discover which specific words or phrases play the biggest role in sentiment classification.

# Repository Structure

```text
├── README.md                  → main project overview
├── requirements.txt           → list of pyhton dependencies
├── data/
│   ├── raw/                   → raw dataset 
│   ├── processed/             → cleaned and preprocessed data
│   └── README.md              → dataset access and instructions for fetching the data
├── scripts/
│   ├── 01_load_data.py        → handles data reading, checks paths configurations
│   ├── 02_preprocess_data.py  → executes text cleaning and saves the final dataset
│   └── 03_basic_eda.py        → code for generating charts and summaries
├── outputs/
│   ├── figures/               → generated viusalizations and graphs
│   └── tables/                → generated summary tables
└── docs/
└── ResearchProposalPreprocessing.md   → main proposal and presentation document
```
# Installation
```text
pip install -r requirements.txt
```
# Running the Codes
```text
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```
# Proposal Document
 For our main proposal and presentation please click: [docs/ResearchProposalPreprocessing.md](docs/ResearchProposalPreprocessing.md)
