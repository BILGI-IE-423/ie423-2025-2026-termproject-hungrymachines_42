# -*- coding: utf-8 -*-
"""
02_preprocess_data.py
---------------------
Performs text cleaning, negation tagging, lemmatization, sentiment
encoding, and statistical feature engineering on the raw IMDB 50K
dataset. The output is saved to:

    data/processed/cleaned_data_set.csv

Run from the repository root:

    python scripts/02_preprocess_data.py
"""

import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------
# Paths (resolved relative to the project root)
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "cleaned_data_set.csv"


# --------------------------------------------------------------------
# NLTK resources (downloaded quietly; idempotent across runs)
# --------------------------------------------------------------------
for resource in ("wordnet", "omw-1.4", "stopwords", "punkt", "averaged_perceptron_tagger"):
    nltk.download(resource, quiet=True)


# --------------------------------------------------------------------
# Vocabulary setup
# --------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

negation_words = {
    # Core negations
    "not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "none",
    # Contractions
    "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't", "ma",
    # Intensity adverbs
    "very", "really", "extremely", "absolutely", "completely",
    "totally", "highly", "deeply", "incredibly", "terribly",
    "barely", "hardly", "scarcely", "slightly", "somewhat",
    "rather", "fairly", "quite", "almost", "nearly",
    # Contrast conjunctions
    "but", "however", "although", "though", "despite", "yet",
    "nevertheless", "nonetheless", "even", "still", "unless",
    "except", "without", "instead", "whereas", "while",
    # Degree words
    "too", "only", "just", "few", "less", "least", "against",
    "any", "anyone", "anything", "whether",
}

# Negation words are preserved by removing them from the stop-words list.
custom_stop_words = stop_words - negation_words


# --------------------------------------------------------------------
# Statistical feature functions (computed on RAW reviews)
# --------------------------------------------------------------------
def get_word_count(text: str) -> int:
    text = re.sub(r"<.*?>", " ", text)
    return len(text.split())


def get_avg_word_length(text: str) -> float:
    text = re.sub(r"<.*?>", " ", text)
    words = re.findall(r"[a-zA-Z]+", text)
    return round(sum(len(w) for w in words) / len(words), 2) if words else 0


def get_exclamation_count(text: str) -> int:
    return text.count("!")


def get_all_caps_count(text: str) -> int:
    return len(re.findall(r"[A-Z]{2,}", text))


def get_lexical_diversity(text: str) -> float:
    text = re.sub(r"<.*?>", " ", text)
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return round(len(set(words)) / len(words), 4) if words else 0


# --------------------------------------------------------------------
# Negation tagging
# --------------------------------------------------------------------
def apply_negation(tokens, negation_trigger_words, window: int = 3):
    """Prefix the next ``window`` tokens after a negation trigger with NEG_."""
    result = []
    negate = 0
    punctuation = {".", ",", "!", "?", ";", ":"}

    for token in tokens:
        if token in punctuation:
            negate = 0
            result.append(token)
        elif token in negation_trigger_words:
            result.append(token)
            negate = window
        elif negate > 0:
            result.append(f"NEG_{token}")
            negate -= 1
        else:
            result.append(token)

    return result


# --------------------------------------------------------------------
# Main text cleaning function
# --------------------------------------------------------------------
def clean_text(text: str) -> str:
    # 1. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Tokenize, preserving punctuation
    tokens = re.findall(r"[a-z']+|[.!?,;:]", text)

    # 4. Negation tagging (BEFORE stop-word removal)
    tokens = apply_negation(tokens, negation_words, window=3)

    # 5. Stop-word removal + lemmatization
    cleaned = []
    for w in tokens:
        if w.startswith("NEG_"):
            root = w.replace("NEG_", "")
            if len(root) > 2:
                cleaned.append(f"NEG_{lemmatizer.lemmatize(root)}")
        elif w not in custom_stop_words and len(w) > 2 and w.isalpha():
            cleaned.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned)


# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------
def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"ERROR: File not found at '{RAW_DATA_PATH}'.\n"
            f"Please download 'IMDB Dataset.csv' from Kaggle and place it in 'data/raw/'."
        )

    # Load
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original Dataset Size: {df.shape}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Dataset Size After Removing Duplicates: {df.shape}")

    # Statistical features on raw text
    print("\nCalculating statistical features on raw reviews...")
    df["word_count"] = df["review"].apply(get_word_count)
    df["avg_word_length"] = df["review"].apply(get_avg_word_length)
    df["exclamation_count"] = df["review"].apply(get_exclamation_count)
    df["all_caps_count"] = df["review"].apply(get_all_caps_count)
    df["lexical_diversity"] = df["review"].apply(get_lexical_diversity)
    print("Statistical features calculated successfully.")

    # Text cleaning
    print("\nText cleaning and lemmatization in progress...")
    print("(This may take approximately 1-3 minutes.)")
    df["cleaned_review"] = df["review"].apply(clean_text)
    print("Text cleaning completed.")

    # Drop rows whose cleaned text is empty
    before = len(df)
    df = df[df["cleaned_review"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with empty cleaned text.")

    # Sentiment encoding
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    print("Sentiment encoded (positive -> 1, negative -> 0).")

    # Column ordering
    df = df[[
        "review", "cleaned_review", "sentiment",
        "word_count", "avg_word_length", "lexical_diversity",
        "exclamation_count", "all_caps_count",
    ]]

    # Preview
    print("\nFirst 3 rows of the processed dataset:")
    print(df.head(3))

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nProcessed dataset saved to: {PROCESSED_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)
