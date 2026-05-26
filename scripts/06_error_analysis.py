# -*- coding: utf-8 -*-
"""
06_error_analysis.py
--------------------
Research Question 3:
    Are misclassified reviews concentrated in specific structural or
    behavioral patterns, such as review length, use of rare words,
    or emotionally mixed content?

Approach:
    - Reload the best model + vectorizer saved by 04_modeling.py.
    - Reproduce the identical 80/20 stratified split (random_state=42).
    - For each test-set review, compute:
        * Structural features: word_count, lexical_diversity,
          exclamation_count, all_caps_count
        * Behavioral features:
            - rare_word_ratio   — share of tokens whose document
              frequency falls in the bottom 5% of the train corpus.
            - mixed_sentiment_score — share of tokens that appear in
              BOTH the "top positive" and "top negative" word lists from
              Logistic Regression (overlap signal of emotional mixing).
    - Split test reviews into Correct vs Misclassified, then further into
      TP / TN / FP / FN, and compare the distributions of all six features.

Reads:
    data/processed/cleaned_data_set.csv
    models/best_model.pkl
    models/best_vectorizer.pkl
    visuals/tables/top_features_logreg.csv

Writes:
    visuals/tables/error_analysis_groups.csv
    visuals/figures/error_feature_distributions.png
    visuals/figures/rare_word_ratio_by_group.png
    visuals/figures/mixed_sentiment_by_group.png

Run from the repository root:

    python scripts/06_error_analysis.py

Approx. runtime: ~1-2 minutes.
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data_set.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "visuals" / "figures"
TABLES_DIR = PROJECT_ROOT / "visuals" / "tables"
TOP_FEATURES_PATH = TABLES_DIR / "top_features_logreg.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Style
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")
COLOR_POS = "#2E8B57"     # correct / positive class
COLOR_NEG = "#DC3545"     # error / negative class
COLOR_NEUTRAL = "#4A6FA5"
BG_COLOR = "#F8F9FA"

RANDOM_STATE = 42

# --------------------------------------------------------------------
# 1. Pre-flight: required artifacts must exist
# --------------------------------------------------------------------
print("=" * 60)
print("  IMDB SENTIMENT ANALYSIS  —  RQ3: ERROR ANALYSIS")
print("=" * 60)

required = {
    "Processed dataset": PROCESSED_PATH,
    "Best model": MODELS_DIR / "best_model.pkl",
    "Best vectorizer": MODELS_DIR / "best_vectorizer.pkl",
    "Top features CSV": TOP_FEATURES_PATH,
}
missing = [name for name, path in required.items() if not path.exists()]
if missing:
    print("\nERROR: The following required files are missing:")
    for name in missing:
        print(f"   - {name}: {required[name]}")
    print("\nPlease run 02_preprocess_data.py and 04_modeling.py first.")
    sys.exit(1)

# --------------------------------------------------------------------
# 2. Load dataset and recreate the EXACT same train/test split as 04
# --------------------------------------------------------------------
df = pd.read_csv(PROCESSED_PATH)
df = df.dropna(subset=["cleaned_review"]).reset_index(drop=True)
df["cleaned_review"] = df["cleaned_review"].astype(str)

X_all = df["cleaned_review"].values
y_all = df["sentiment"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all,
)

# We also need the FULL row info for the test split so we can join the
# per-row structural features. Recreate the same split on the index.
idx_all = np.arange(len(df))
_, idx_test = train_test_split(
    idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all,
)
test_df = df.iloc[idx_test].reset_index(drop=True).copy()
print(f"\nTest set size: {len(test_df):,}")

# Sanity check: the test labels we just rebuilt should match y_test.
assert (test_df["sentiment"].values == y_test).all(), "Test split mismatch!"

# --------------------------------------------------------------------
# 3. Load the persisted best model + vectorizer
# --------------------------------------------------------------------
print("\nLoading saved model + vectorizer ...")
best_model = joblib.load(MODELS_DIR / "best_model.pkl")
best_vec = joblib.load(MODELS_DIR / "best_vectorizer.pkl")
print(f"   Model:      {type(best_model).__name__}")
print(f"   Vectorizer: {type(best_vec).__name__}")

# --------------------------------------------------------------------
# 4. Predict on the test set + build the error frame
# --------------------------------------------------------------------
print("\nVectorizing test set and predicting ...")
X_test_vec = best_vec.transform(test_df["cleaned_review"].values)
y_pred = best_model.predict(X_test_vec)

test_df["y_true"] = test_df["sentiment"]
test_df["y_pred"] = y_pred
test_df["is_correct"] = test_df["y_true"] == test_df["y_pred"]


def error_group(row):
    """Label each test row as TP / TN / FP / FN."""
    yt, yp = row["y_true"], row["y_pred"]
    if yt == 1 and yp == 1:
        return "TP"
    if yt == 0 and yp == 0:
        return "TN"
    if yt == 0 and yp == 1:
        return "FP"
    return "FN"


test_df["group"] = test_df.apply(error_group, axis=1)

n_correct = int(test_df["is_correct"].sum())
n_wrong = len(test_df) - n_correct
print(f"\nCorrect:        {n_correct:,} ({n_correct / len(test_df):.2%})")
print(f"Misclassified:  {n_wrong:,} ({n_wrong / len(test_df):.2%})")
group_counts = test_df["group"].value_counts().reindex(["TP", "TN", "FP", "FN"])
print("\nGroup counts:")
for g, c in group_counts.items():
    print(f"   {g}: {c:,}")

# --------------------------------------------------------------------
# 5. Behavioral feature 1 — Rare word ratio
# --------------------------------------------------------------------
# Estimate document frequency (DF) of every token in the TRAIN set, then
# mark a token as "rare" if its DF is in the bottom 5% of the distribution
# of DF values. The rare_word_ratio of a review is the share of its
# tokens (with duplicates) that fall in this rare set.
print("\n" + "=" * 60)
print("  STEP 1 — RARE WORD RATIO (computed from training corpus)")
print("=" * 60)

print("   Computing document frequencies on training corpus ...")
train_token_lists = [s.split() for s in X_train]
df_counter = Counter()
for toks in train_token_lists:
    df_counter.update(set(toks))   # one count per document
n_train_docs = len(train_token_lists)

dfs = np.array(list(df_counter.values()))
rare_threshold = np.percentile(dfs, 5)
rare_words = {w for w, c in df_counter.items() if c <= rare_threshold}
print(f"   Train vocabulary size      : {len(df_counter):,}")
print(f"   Rare threshold (DF <= {int(rare_threshold)}) : "
      f"{len(rare_words):,} words")


def rare_word_ratio(text):
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in rare_words) / len(toks)


test_df["rare_word_ratio"] = test_df["cleaned_review"].apply(rare_word_ratio)

# --------------------------------------------------------------------
# 6. Behavioral feature 2 — Mixed sentiment score
# --------------------------------------------------------------------
# We treat the top-N positive words and top-N negative words from the
# Logistic Regression coefficients (computed in 04) as a coarse lexicon.
# For each review we then compute how "mixed" it is, i.e. how often it
# uses BOTH sides of the lexicon. A review that only uses one side gets
# a low mixed score; a review balancing pos and neg words gets a high
# mixed score.
#
# Concretely, if a review has p positive tokens and n negative tokens:
#     mixed_sentiment_score = (2 * min(p, n)) / (p + n + 1)
#
# The +1 in the denominator stabilises reviews with very few lexicon
# hits. The factor 2 means a perfectly balanced review (p == n)
# approaches 1 as p+n grows, and a one-sided review stays near 0.
print("\n" + "=" * 60)
print("  STEP 2 — MIXED SENTIMENT SCORE (using LR top-features)")
print("=" * 60)

top_feat_df = pd.read_csv(TOP_FEATURES_PATH)
pos_lex = set(top_feat_df["positive_word"].astype(str))
neg_lex = set(top_feat_df["negative_word"].astype(str))
print(f"   Positive lexicon size: {len(pos_lex)}")
print(f"   Negative lexicon size: {len(neg_lex)}")


def mixed_sentiment_score(text):
    toks = text.split()
    if not toks:
        return 0.0
    p = sum(1 for t in toks if t in pos_lex)
    n = sum(1 for t in toks if t in neg_lex)
    if p + n == 0:
        return 0.0
    return (2 * min(p, n)) / (p + n + 1)


test_df["mixed_sentiment_score"] = test_df["cleaned_review"].apply(
    mixed_sentiment_score
)

# --------------------------------------------------------------------
# 7. Group-level summary table
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 3 — GROUP-LEVEL FEATURE COMPARISON")
print("=" * 60)

features = [
    "word_count", "lexical_diversity",
    "exclamation_count", "all_caps_count",
    "rare_word_ratio", "mixed_sentiment_score",
]

# (a) Correct vs Misclassified
correct_summary = (
    test_df.groupby("is_correct")[features]
    .mean()
    .round(4)
)
correct_summary.index = ["Misclassified", "Correct"]
print("\nMean feature values — Correct vs Misclassified:")
print(correct_summary.to_string())

# (b) TP / TN / FP / FN
group_order = ["TP", "TN", "FP", "FN"]
group_summary = (
    test_df.groupby("group")[features]
    .mean()
    .reindex(group_order)
    .round(4)
)
group_summary.insert(0, "n", group_counts.reindex(group_order).values)
print("\nMean feature values — TP / TN / FP / FN:")
print(group_summary.to_string())

# Save the combined summary
out_path = TABLES_DIR / "error_analysis_groups.csv"
group_summary.to_csv(out_path)
print(f"\nSaved: {out_path}")

# --------------------------------------------------------------------
# 8. Visualization 1 — 4 structural features, Correct vs Misclassified
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 4 — VISUALIZATIONS")
print("=" * 60)
print("\n[1/3] Structural feature distributions (Correct vs Misclassified) ...")

struct_feats = [
    ("word_count",         "Word Count"),
    ("lexical_diversity",  "Lexical Diversity"),
    ("exclamation_count",  "Exclamation Count"),
    ("all_caps_count",     "ALL-CAPS Count"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor(BG_COLOR)
axes = axes.flatten()

palette = {True: COLOR_POS, False: COLOR_NEG}
labels_map = {True: "Correct", False: "Misclassified"}

for ax, (col, title) in zip(axes, struct_feats):
    ax.set_facecolor(BG_COLOR)

    # Clip extreme outliers for readability (visual only).
    upper = np.percentile(test_df[col], 99)
    data_lo = test_df[col].clip(upper=upper)

    plot_df = pd.DataFrame({
        "value": data_lo,
        "group": test_df["is_correct"].map(labels_map),
    })

    sns.boxplot(
        data=plot_df, x="group", y="value", order=["Correct", "Misclassified"],
        palette={"Correct": COLOR_POS, "Misclassified": COLOR_NEG},
        width=0.45, ax=ax,
        boxprops=dict(alpha=0.85, edgecolor="white"),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
        flierprops=dict(marker="o", markersize=2,
                        markerfacecolor="#AAAAAA", markeredgecolor="none", alpha=0.4),
    )

    ax.set_title(title, fontsize=12, fontweight="bold", color="#2D2D2D", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=10)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")

    # Annotate medians.
    for i, lbl in enumerate(["Correct", "Misclassified"]):
        med = plot_df.loc[plot_df["group"] == lbl, "value"].median()
        ax.text(i, med + upper * 0.02 if upper > 0 else med + 0.02,
                f"med={med:.2f}" if med < 10 else f"med={int(med)}",
                ha="center", va="bottom",
                fontsize=9, color="#333333", fontweight="bold")

plt.suptitle("Structural Features — Correct vs Misclassified Reviews",
             fontsize=15, fontweight="bold", color="#2D2D2D", y=1.00)
plt.tight_layout()
out_path = FIGURES_DIR / "error_feature_distributions.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")

# --------------------------------------------------------------------
# 9. Visualization 2 — Rare word ratio by group (TP/TN/FP/FN)
# --------------------------------------------------------------------
print("\n[2/3] Rare word ratio by error group ...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

group_palette = {
    "TP": COLOR_POS,
    "TN": COLOR_NEUTRAL,
    "FP": "#E07B00",
    "FN": COLOR_NEG,
}

sns.boxplot(
    data=test_df, x="group", y="rare_word_ratio", order=group_order,
    palette=group_palette, width=0.5, ax=ax,
    boxprops=dict(alpha=0.85, edgecolor="white"),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#666666"),
    capprops=dict(color="#666666"),
    flierprops=dict(marker="o", markersize=2.5,
                    markerfacecolor="#999999", markeredgecolor="none", alpha=0.5),
)

ax.set_title("Rare Word Ratio by Error Group",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("Group", fontsize=12, color="#444444", labelpad=10)
ax.set_ylabel("Rare Word Ratio\n(share of tokens in bottom-5% DF set)",
              fontsize=11, color="#444444", labelpad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=11)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")

# Annotate group medians.
for i, g in enumerate(group_order):
    sub = test_df.loc[test_df["group"] == g, "rare_word_ratio"]
    med = sub.median()
    n = len(sub)
    ax.text(i, med + 0.003, f"med={med:.3f}\nn={n:,}",
            ha="center", va="bottom", fontsize=9, color="#333333",
            fontweight="bold")

plt.tight_layout()
out_path = FIGURES_DIR / "rare_word_ratio_by_group.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")

# --------------------------------------------------------------------
# 10. Visualization 3 — Mixed sentiment score by group
# --------------------------------------------------------------------
print("\n[3/3] Mixed sentiment score by error group ...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

sns.boxplot(
    data=test_df, x="group", y="mixed_sentiment_score", order=group_order,
    palette=group_palette, width=0.5, ax=ax,
    boxprops=dict(alpha=0.85, edgecolor="white"),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#666666"),
    capprops=dict(color="#666666"),
    flierprops=dict(marker="o", markersize=2.5,
                    markerfacecolor="#999999", markeredgecolor="none", alpha=0.5),
)

ax.set_title("Mixed Sentiment Score by Error Group",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("Group", fontsize=12, color="#444444", labelpad=10)
ax.set_ylabel("Mixed Sentiment Score\n(higher = both pos & neg lexicon used)",
              fontsize=11, color="#444444", labelpad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=11)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")

for i, g in enumerate(group_order):
    sub = test_df.loc[test_df["group"] == g, "mixed_sentiment_score"]
    med = sub.median()
    mean = sub.mean()
    ax.text(i, med + 0.01, f"med={med:.3f}\nmean={mean:.3f}",
            ha="center", va="bottom", fontsize=9, color="#333333",
            fontweight="bold")

plt.tight_layout()
out_path = FIGURES_DIR / "mixed_sentiment_by_group.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")

# --------------------------------------------------------------------
# Done
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RQ3 ERROR ANALYSIS — COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"  Test set size:       {len(test_df):,}")
print(f"  Correct:             {n_correct:,} ({n_correct / len(test_df):.2%})")
print(f"  Misclassified:       {n_wrong:,} ({n_wrong / len(test_df):.2%})")
print()
print(f"  TP / TN / FP / FN:   "
      f"{group_counts['TP']:,} / {group_counts['TN']:,} / "
      f"{group_counts['FP']:,} / {group_counts['FN']:,}")
print()
print("  Key feature means (Misclassified vs Correct):")
for f in features:
    m_wrong = test_df.loc[~test_df["is_correct"], f].mean()
    m_right = test_df.loc[test_df["is_correct"], f].mean()
    print(f"    {f:<25} wrong={m_wrong:.4f}   correct={m_right:.4f}")
print("=" * 60)