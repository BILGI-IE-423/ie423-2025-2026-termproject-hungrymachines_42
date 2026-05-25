# -*- coding: utf-8 -*-
"""
01_load_data.py
---------------
Loads the raw IMDB 50K movie reviews dataset, validates the file path,
and prints basic structural information (shape, dtypes, sentiment
distribution, head). This script is the entry point of the pipeline
and is intended to be run from the repository root:

    python scripts/01_load_data.py
"""

import os
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------
# Project root is the parent of the "scripts/" folder, regardless of the
# directory the user runs the script from. This makes the pipeline
# reproducible from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"


def load_imdb_data(file_path: Path) -> pd.DataFrame:
    """Load the IMDB dataset and print basic structural information."""

    # 1. Validate file path
    if not file_path.exists():
        raise FileNotFoundError(
            f"ERROR: File not found at '{file_path}'.\n"
            f"Please download 'IMDB Dataset.csv' from Kaggle\n"
            f"(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)\n"
            f"and place it inside 'data/raw/'."
        )

    print(f"File found! Loading data from '{file_path}'...\n")

    # 2. Load the dataset
    df = pd.read_csv(file_path)

    # 3. Print basic dataset information
    print("-" * 40)
    print("BASIC DATASET INFORMATION")
    print("-" * 40)

    print(f"Shape (Rows, Columns): {df.shape}\n")

    if "sentiment" in df.columns:
        print("Sentiment Distribution:")
        print(df["sentiment"].value_counts().to_string())
        print()
    else:
        print("WARNING: 'sentiment' column not found in the dataset.\n")

    print("Column Information (info):")
    df.info()
    print()

    print("First 3 Rows of the Dataset:")
    print(df.head(3))
    print("-" * 40)

    return df


if __name__ == "__main__":
    try:
        df_raw = load_imdb_data(RAW_DATA_PATH)
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)
