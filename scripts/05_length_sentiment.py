# -*- coding: utf-8 -*-
"""
05_length_sentiment.py
----------------------
Research Question 2:
    Is there a statistically significant correlation between review length
    (word count) and sentiment polarity, indicating whether dissatisfied
    viewers write more exhaustive reviews?
 
Approach:
    - Three correlation tests on (word_count, sentiment):
        * Pearson         — linear association
        * Spearman        — monotonic (rank) association
        * Point-biserial  — proper test for continuous vs binary
    - Mann-Whitney U test — does the length distribution differ between
      positive and negative groups?
    - Length-bin analysis — sentiment rate within short / medium / long /
      very-long bins.
    - Single-feature baseline — logistic regression using ONLY word_count.
      This isolates how much predictive power length carries by itself.
 
Reads:
    data/processed/cleaned_data_set.csv
 
Writes:
    visuals/tables/length_sentiment_stats.csv
    visuals/tables/length_bins.csv
    visuals/figures/length_distribution_by_sentiment.png
    visuals/figures/sentiment_rate_by_length_bin.png
    visuals/figures/length_only_logreg_roc.png
 
Run from the repository root:
 
    python scripts/05_length_sentiment.py
 
Approx. runtime: ~30 seconds.
"""
 
import sys
import warnings
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
 
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data_set.csv"
FIGURES_DIR = PROJECT_ROOT / "visuals" / "figures"
TABLES_DIR = PROJECT_ROOT / "visuals" / "tables"
 
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
 
# --------------------------------------------------------------------
# Style (matches 03_basic_eda.py and 04_modeling.py)
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")
COLOR_POS = "#2E8B57"
COLOR_NEG = "#DC3545"
COLOR_NEUTRAL = "#4A6FA5"
BG_COLOR = "#F8F9FA"
 
RANDOM_STATE = 42
 
# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------
print("=" * 60)
print("  IMDB SENTIMENT ANALYSIS  —  RQ2: LENGTH vs SENTIMENT")
print("=" * 60)
 
if not PROCESSED_PATH.exists():
    print(
        f"ERROR: Processed file not found at '{PROCESSED_PATH}'.\n"
        f"Please run 'python scripts/02_preprocess_data.py' first."
    )
    sys.exit(1)
 
df = pd.read_csv(PROCESSED_PATH)
df = df.dropna(subset=["word_count", "sentiment"]).reset_index(drop=True)
df["word_count"] = df["word_count"].astype(int)
df["sentiment"] = df["sentiment"].astype(int)
df["sentiment_label"] = df["sentiment"].map({1: "Positive", 0: "Negative"})
print(f"\nDataset loaded — shape: {df.shape}")
 
# --------------------------------------------------------------------
# 2. Descriptive: length by sentiment
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 1 — DESCRIPTIVE STATISTICS")
print("=" * 60)
 
desc = df.groupby("sentiment_label")["word_count"].describe().round(2)
print("\nWord count by sentiment:")
print(desc.to_string())
 
# --------------------------------------------------------------------
# 3. Correlation tests
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 2 — CORRELATION TESTS")
print("=" * 60)
 
stats_rows = []
 
# 3.1 Pearson
pearson_r, pearson_p = stats.pearsonr(df["word_count"], df["sentiment"])
stats_rows.append({
    "test": "Pearson correlation",
    "statistic": round(pearson_r, 6),
    "p_value": pearson_p,
    "interpretation": "Linear correlation between word_count and sentiment",
})
print(f"\nPearson r  = {pearson_r:+.4f}   p = {pearson_p:.2e}")
 
# 3.2 Spearman
spearman_r, spearman_p = stats.spearmanr(df["word_count"], df["sentiment"])
stats_rows.append({
    "test": "Spearman correlation",
    "statistic": round(spearman_r, 6),
    "p_value": spearman_p,
    "interpretation": "Monotonic (rank-based) correlation",
})
print(f"Spearman r = {spearman_r:+.4f}   p = {spearman_p:.2e}")
 
# 3.3 Point-biserial (textbook test for continuous vs binary)
pb_r, pb_p = stats.pointbiserialr(df["sentiment"], df["word_count"])
stats_rows.append({
    "test": "Point-biserial correlation",
    "statistic": round(pb_r, 6),
    "p_value": pb_p,
    "interpretation": "Correlation between binary sentiment and continuous word_count",
})
print(f"Point-bis. = {pb_r:+.4f}   p = {pb_p:.2e}")
 
# --------------------------------------------------------------------
# 4. Mann-Whitney U: do the length distributions differ?
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 3 — DISTRIBUTION TEST (Mann-Whitney U)")
print("=" * 60)
 
pos_lengths = df.loc[df["sentiment"] == 1, "word_count"].values
neg_lengths = df.loc[df["sentiment"] == 0, "word_count"].values
 
u_stat, u_p = stats.mannwhitneyu(pos_lengths, neg_lengths, alternative="two-sided")
stats_rows.append({
    "test": "Mann-Whitney U (two-sided)",
    "statistic": round(u_stat, 2),
    "p_value": u_p,
    "interpretation": "Whether the word_count distributions differ between Pos and Neg",
})
print(f"\nU = {u_stat:.2f}   p = {u_p:.2e}")
print(f"Median (Positive) = {np.median(pos_lengths):.1f}")
print(f"Median (Negative) = {np.median(neg_lengths):.1f}")
 
# Save all statistical results
stats_df = pd.DataFrame(stats_rows)
stats_df["p_value"] = stats_df["p_value"].apply(lambda p: f"{p:.3e}")
stats_path = TABLES_DIR / "length_sentiment_stats.csv"
stats_df.to_csv(stats_path, index=False)
print(f"\nSaved: {stats_path}")
 
# --------------------------------------------------------------------
# 5. Length bins — sentiment rate per bin
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 4 — LENGTH-BIN ANALYSIS")
print("=" * 60)
 
bin_edges = [0, 100, 200, 400, df["word_count"].max() + 1]
bin_labels = ["Short (0–100)", "Medium (101–200)",
              "Long (201–400)", "Very Long (>400)"]
df["length_bin"] = pd.cut(df["word_count"], bins=bin_edges, labels=bin_labels,
                          include_lowest=True)
 
bin_summary = df.groupby("length_bin", observed=True).agg(
    total=("sentiment", "count"),
    positive=("sentiment", "sum"),
).reset_index()
bin_summary["negative"] = bin_summary["total"] - bin_summary["positive"]
bin_summary["positive_rate"] = (bin_summary["positive"] / bin_summary["total"]).round(4)
bin_summary["negative_rate"] = (bin_summary["negative"] / bin_summary["total"]).round(4)
 
print("\nSentiment composition per length bin:")
print(bin_summary.to_string(index=False))
 
bin_path = TABLES_DIR / "length_bins.csv"
bin_summary.to_csv(bin_path, index=False)
print(f"\nSaved: {bin_path}")
 
# --------------------------------------------------------------------
# 6. Single-feature baseline: Logistic Regression using only word_count
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 5 — SINGLE-FEATURE BASELINE (word_count only)")
print("=" * 60)
 
X = df[["word_count"]].values.astype(float)
y = df["sentiment"].values
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)
 
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_score = lr.predict_proba(X_test)[:, 1]
 
lr_acc = accuracy_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)
lr_auc = roc_auc_score(y_test, y_score)
 
print(f"\nLogistic Regression using only word_count:")
print(f"   Test accuracy = {lr_acc:.4f}")
print(f"   Test F1       = {lr_f1:.4f}")
print(f"   Test ROC-AUC  = {lr_auc:.4f}")
print(f"   Coefficient   = {lr.coef_[0][0]:+.6f}")
print(f"   Intercept     = {lr.intercept_[0]:+.6f}")
print("\nFor context: the full RQ1 model reached ~0.90 accuracy.")
print("Anything close to 0.50 here means length alone carries almost no signal.")
 
# --------------------------------------------------------------------
# 7. Visualization 1 — Length distribution by sentiment (boxplot, clipped)
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 6 — VISUALIZATIONS")
print("=" * 60)
print("\n[1/3] Length distribution by sentiment ...")
 
# Clip to 99th percentile for readable boxplots (extreme outliers exist).
clip = int(np.percentile(df["word_count"], 99))
plot_df = df[df["word_count"] <= clip].copy()
 
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
 
palette = {"Negative": COLOR_NEG, "Positive": COLOR_POS}
sns.boxplot(
    data=plot_df, x="sentiment_label", y="word_count",
    order=["Negative", "Positive"],
    palette=palette, width=0.45, ax=ax,
    boxprops=dict(alpha=0.85, edgecolor="white"),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#666666"),
    capprops=dict(color="#666666"),
    flierprops=dict(marker="o", markersize=2.5,
                    markerfacecolor="#999999", markeredgecolor="none", alpha=0.5),
)
 
ax.set_title("Review Length Distribution by Sentiment",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("")
ax.set_ylabel("Word Count", fontsize=12, color="#444444", labelpad=10)
ax.set_ylim(0, clip * 1.02)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=11)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")
 
# Annotate medians on the plot
for i, lbl in enumerate(["Negative", "Positive"]):
    med = plot_df.loc[plot_df["sentiment_label"] == lbl, "word_count"].median()
    ax.text(i, med + clip * 0.02, f"median = {int(med)}",
            ha="center", va="bottom", fontsize=10, color="#333333", fontweight="bold")
 
# p-value annotation top-right
p_label = f"Mann-Whitney p = {u_p:.2e}\nPoint-biserial r = {pb_r:+.4f}"
ax.text(0.98, 0.97, p_label, transform=ax.transAxes,
        fontsize=10, color="#333333", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.95))
 
plt.tight_layout()
out_path = FIGURES_DIR / "length_distribution_by_sentiment.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# 8. Visualization 2 — Sentiment rate per length bin
# --------------------------------------------------------------------
print("\n[2/3] Sentiment rate per length bin ...")
 
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
 
x = np.arange(len(bin_summary))
width = 0.36
 
bars_pos = ax.bar(x - width/2, bin_summary["positive_rate"], width=width,
                  color=COLOR_POS, alpha=0.9, edgecolor="white", linewidth=0.5,
                  label="Positive rate")
bars_neg = ax.bar(x + width/2, bin_summary["negative_rate"], width=width,
                  color=COLOR_NEG, alpha=0.9, edgecolor="white", linewidth=0.5,
                  label="Negative rate")
 
for bars in (bars_pos, bars_neg):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                f"{h:.1%}", ha="center", va="bottom",
                fontsize=10, color="#333333")
 
# Reference line at 50%
ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, zorder=1)
ax.text(len(bin_summary) - 0.4, 0.51, "50% balance",
        ha="right", va="bottom", fontsize=9, color="#666666")
 
ax.set_title("Sentiment Composition by Review Length",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=10, color="#333333", ha="center")
ax.set_xlabel("Word Count Range", fontsize=12, color="#444444", labelpad=10)
ax.set_ylabel("Proportion of Reviews", fontsize=12, color="#444444", labelpad=10)
ax.set_ylim(0, max(0.75, bin_summary[["positive_rate", "negative_rate"]].values.max() + 0.1))
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")
ax.legend(fontsize=11, framealpha=0.95, edgecolor="#CCCCCC", facecolor="white")
 
# Add n labels inside the plot above x-axis
for i, row in bin_summary.iterrows():
    ax.text(i, 0.02, f"n={int(row['total']):,}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="#666666")
 
plt.tight_layout()
out_path = FIGURES_DIR / "sentiment_rate_by_length_bin.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# 9. Visualization 3 — ROC curve for length-only logistic regression
# --------------------------------------------------------------------
print("\n[3/3] ROC curve for length-only baseline ...")
 
fpr, tpr, _ = roc_curve(y_test, y_score)
 
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
 
ax.plot(fpr, tpr, color=COLOR_NEUTRAL, linewidth=2.5,
        label=f"Logistic Regression (word_count only)\nAUC = {lr_auc:.3f}")
ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2,
        label="Random classifier (AUC = 0.5)")
 
ax.set_title("ROC Curve — Single-Feature Baseline (word_count only)",
             fontsize=14, fontweight="bold", color="#2D2D2D", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=11, color="#444444")
ax.set_ylabel("True Positive Rate", fontsize=11, color="#444444")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(color="#E0E0E0", linewidth=0.7, linestyle="--")
ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
          edgecolor="#CCCCCC", facecolor="white")
 
# Annotation explaining the result
ax.text(0.05, 0.97,
        f"Test accuracy: {lr_acc:.3f}\n"
        f"Test F1:       {lr_f1:.3f}\n"
        f"Test ROC-AUC:  {lr_auc:.3f}",
        transform=ax.transAxes, fontsize=10, family="monospace",
        color="#333333", ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.95))
 
plt.tight_layout()
out_path = FIGURES_DIR / "length_only_logreg_roc.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# Done
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RQ2 ANALYSIS — COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"  Pearson r       = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
print(f"  Spearman r      = {spearman_r:+.4f}  (p = {spearman_p:.2e})")
print(f"  Point-biserial  = {pb_r:+.4f}  (p = {pb_p:.2e})")
print(f"  Mann-Whitney U  = {u_stat:.2f}  (p = {u_p:.2e})")
print(f"  Length-only LR  = {lr_acc:.4f} accuracy, AUC = {lr_auc:.4f}")
print("=" * 60)
