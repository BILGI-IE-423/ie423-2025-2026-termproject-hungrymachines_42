# Dataset Access

## Source

- **Dataset name:** IMDB Dataset of 50K Movie Reviews
- **Primary source (Kaggle):** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Mirror (Google Drive — raw):** https://drive.google.com/file/d/1op2yoX-aYIY4KRVdcSuNXiNOyfsqahOB/view?usp=share_link
- **Mirror (Google Drive — already processed):** https://drive.google.com/file/d/189aRaVuXKPx3Z0_r4TFE6Xi75TQZy-iF/view?usp=share_link

The dataset is **not committed to this repository** because of GitHub file size limits and licensing considerations on Kaggle. It must be downloaded manually.

## About the Dataset

The dataset contains **50,000 movie reviews** labeled for binary sentiment classification (`positive` / `negative`). Classes are perfectly balanced (25,000 of each). Each row has two columns: `review` (free text, may contain HTML tags) and `sentiment` (string label).

## Required File Placement

After downloading, the raw file must be placed at exactly this path:

```
data/raw/IMDB Dataset.csv
```

The filename must be **`IMDB Dataset.csv`** (with a space, as distributed by Kaggle). The preprocessing pipeline expects this exact name and will raise `FileNotFoundError` otherwise.

The processed dataset is created automatically at `data/processed/cleaned_data_set.csv` when you run `python scripts/02_preprocess_data.py`. You do not need to download it; it is provided only as a convenience mirror.

## Manual Download Steps

1. Open the Kaggle link above and click **Download** (requires a free Kaggle account).
2. Unzip the downloaded archive — it contains a single file named `IMDB Dataset.csv`.
3. Move that file into the `data/raw/` folder of this repository.
4. From the repository root, run `python scripts/01_load_data.py` to verify the file is readable.
