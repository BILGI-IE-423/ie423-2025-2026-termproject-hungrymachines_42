# -*- coding: utf-8 -*-
"""
03_basic_eda.py
---------------
Exploratory Data Analysis: produces summary statistics tables (CSV)
and visualizations (PNG) from the cleaned dataset.

Reads:
    data/processed/cleaned_data_set.csv

Writes:
    visuals/tables/descriptive_stats.csv
    visuals/tables/summary_stats_by_sentiment.csv
    visuals/figures/sentiment_distribution.png
    visuals/figures/word_count_distribution.png
    visuals/figures/top_words_by_sentiment.png
    visuals/figures/correlation_matrix.png
    visuals/figures/lexical_diversity_vs_wordcount.png
    visuals/figures/lexical_diversity_segments.png

Run from the repository root:

    python scripts/03_basic_eda.py
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba

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
# Style
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------
print("=" * 50)
print("  IMDB SENTIMENT ANALYSIS - BASIC EDA")
print("=" * 50)

if not PROCESSED_PATH.exists():
    print(
        f"ERROR: Processed file not found at '{PROCESSED_PATH}'.\n"
        f"Please run 'python scripts/02_preprocess_data.py' first."
    )
    sys.exit(1)

df = pd.read_csv(PROCESSED_PATH)
df["sentiment_label"] = df["sentiment"].map({1: "Positive", 0: "Negative"})
print(f"\nDataset loaded successfully. Shape: {df.shape}\n")

# --------------------------------------------------------------------
# 2. Initial summaries
# --------------------------------------------------------------------
print("=" * 50)
print("  SECTION 1 - INITIAL SUMMARIES")
print("=" * 50)

stat_cols = [
    "word_count", "avg_word_length", "exclamation_count",
    "all_caps_count", "lexical_diversity",
]

# 2.1 Sentiment distribution
print("\nSentiment Distribution (%):")
print("-" * 40)
dist = df["sentiment_label"].value_counts(normalize=True) * 100
for label, pct in dist.items():
    print(f"   {label:<12}: {pct:.2f}%")

# 2.2 General dataset info
print("\nDataset Overview:")
print("-" * 40)
print(f"   Total reviews     : {len(df):,}")
print(f"   Positive reviews  : {(df['sentiment'] == 1).sum():,}")
print(f"   Negative reviews  : {(df['sentiment'] == 0).sum():,}")
print(f"   Total columns     : {df.shape[1]}")
print(f"   Missing values    : {df.isnull().sum().sum()}")

# 2.3 Grouped mean + median
print("\nAverage & Median Statistics by Sentiment:")
print("-" * 40)
summary_mean = df.groupby("sentiment_label")[stat_cols].mean().round(3)
summary_median = df.groupby("sentiment_label")[stat_cols].median().round(3)
summary_mean.index = [f"{i} - Mean" for i in summary_mean.index]
summary_median.index = [f"{i} - Median" for i in summary_median.index]
summary_stats = pd.concat([summary_mean, summary_median]).sort_index()
print(summary_stats.to_string())

summary_path = TABLES_DIR / "summary_stats_by_sentiment.csv"
summary_stats.to_csv(summary_path, index=True)
print(f"   Saved to: {summary_path}")

# 2.4 Descriptive statistics
print("\nDescriptive Statistics (Behavioral Features):")
print("-" * 40)
desc_stats = df[stat_cols].describe().round(3)
print(desc_stats.to_string())

desc_path = TABLES_DIR / "descriptive_stats.csv"
desc_stats.to_csv(desc_path, index=True)
print(f"   Saved to: {desc_path}")
print()

# --------------------------------------------------------------------
# 3. Visualization 1: Sentiment distribution bar chart
# --------------------------------------------------------------------
print("=" * 50)
print("  SECTION 2 - VISUALIZATIONS")
print("=" * 50)
print("\n[1/6] Sentiment Distribution Bar Chart...")

counts = df["sentiment"].value_counts()
total = len(df)
percentages = (counts / total) * 100
labels = ["Positive", "Negative"]
values = [counts[1], counts[0]]
pct_values = [percentages[1], percentages[0]]

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

bars = ax.bar([0, 0.5], values, color=["#2E8B57", "#DC3545"], width=0.35, alpha=0.85, zorder=3)
for bar, pct in zip(bars, pct_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(values) * 0.02,
        f"{pct:.1f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333333",
    )

ax.set_title("Sentiment Distribution of IMDB Reviews", fontsize=15, fontweight="bold", color="#2D2D2D", pad=20)
ax.set_ylabel("Number of Reviews", fontsize=12, color="#444444", labelpad=12)
ax.set_xticks([0, 0.5])
ax.set_xticklabels(labels)
ax.set_xlim(-0.4, 0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(axis="x", colors="#333333", labelsize=13)
ax.tick_params(axis="y", colors="#555555", labelsize=11)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--", zorder=0)
plt.tight_layout()

out_path = FIGURES_DIR / "sentiment_distribution.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# 4. Visualization 2: Word count distribution
# --------------------------------------------------------------------
print("\n[2/6] Word Count Distribution...")

positive = df[df["sentiment_label"] == "Positive"]["word_count"].clip(upper=1000)
negative = df[df["sentiment_label"] == "Negative"]["word_count"].clip(upper=1000)
bins = np.linspace(0, 1000, 41)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

ax.hist(negative, bins=bins, alpha=0.45, color="#DC3545", edgecolor="white", linewidth=0.4, zorder=2)
ax.hist(positive, bins=bins, alpha=0.38, color="#2E8B57", edgecolor="white", linewidth=0.4, zorder=3)

ax.set_title("Word Count Distribution by Sentiment", fontsize=16, fontweight="bold", pad=20, color="#2D2D2D")
ax.set_xlabel("Word Count", fontsize=13, color="#444444", labelpad=10)
ax.set_ylabel("Frequency", fontsize=13, color="#444444", labelpad=10)
ax.tick_params(colors="#555555", labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, linestyle="--")
ax.set_xlim(0, 1000)

pos_patch = mpatches.Patch(facecolor=to_rgba("#2E8B57", 0.45), label="Positive")
neg_patch = mpatches.Patch(facecolor=to_rgba("#DC3545", 0.45), label="Negative")
ovlp_patch = mpatches.Patch(facecolor=to_rgba("#6B8C5A", 0.65), label="Overlap")
ax.legend(
    handles=[pos_patch, neg_patch, ovlp_patch],
    title="Sentiment", fontsize=11, title_fontsize=12,
    framealpha=0.9, edgecolor="#CCCCCC", facecolor="white", loc="upper right",
)

plt.tight_layout()
out_path = FIGURES_DIR / "word_count_distribution.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# 5. Visualization 3: Top 15 most frequent words
# --------------------------------------------------------------------
print("\n[3/6] Top 15 Most Frequent Words...")

def get_top_words(data, n=15):
    all_words = " ".join(data.dropna()).split()
    all_words = [w for w in all_words if not w.startswith("NEG_")]
    return Counter(all_words).most_common(n)

pos_words_data = get_top_words(df[df["sentiment"] == 1]["cleaned_review"])
neg_words_data = get_top_words(df[df["sentiment"] == 0]["cleaned_review"])
pos_words, pos_counts = zip(*pos_words_data)
neg_words, neg_counts = zip(*neg_words_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8F9FA")
for ax in (ax1, ax2):
    ax.set_facecolor("#F8F9FA")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11, colors="#444444")
    ax.xaxis.set_visible(False)
    ax.yaxis.grid(False)

bars1 = ax1.barh(pos_words, pos_counts, color="#2E8B57", alpha=0.85, edgecolor="white", linewidth=0.5, height=0.6, zorder=3)
ax1.set_title("Most Frequent Words in Positive Reviews", fontsize=13, fontweight="bold", color="#2D2D2D", pad=15)
ax1.invert_yaxis()
for bar, count in zip(bars1, pos_counts):
    ax1.text(bar.get_width() + max(pos_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{count:,}", va="center", ha="left", fontsize=9, color="#555555")
ax1.set_xlim(0, max(pos_counts) * 1.18)

bars2 = ax2.barh(neg_words, neg_counts, color="#DC3545", alpha=0.85, edgecolor="white", linewidth=0.5, height=0.6, zorder=3)
ax2.set_title("Most Frequent Words in Negative Reviews", fontsize=13, fontweight="bold", color="#2D2D2D", pad=15)
ax2.invert_yaxis()
for bar, count in zip(bars2, neg_counts):
    ax2.text(bar.get_width() + max(neg_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{count:,}", va="center", ha="left", fontsize=9, color="#555555")
ax2.set_xlim(0, max(neg_counts) * 1.18)

plt.suptitle("Top 15 Most Frequent Words by Sentiment", fontsize=15, fontweight="bold", color="#2D2D2D", y=1.02)
plt.tight_layout()
out_path = FIGURES_DIR / "top_words_by_sentiment.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# 6. Visualization 4: Correlation matrix
# --------------------------------------------------------------------
print("\n[4/6] Correlation Matrix...")

corr = df[stat_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", vmin=-1, vmax=1, center=0,
    square=True, linewidths=3, linecolor="#F8F9FA",
    annot_kws={"size": 12, "weight": "bold"},
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
    ax=ax,
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10, colors="#444444")
cbar.set_label("Correlation Coefficient", fontsize=11, color="#444444", labelpad=12)
cbar.outline.set_edgecolor("#CCCCCC")

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11, color="#333333")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11, color="#333333")

for text in ax.texts:
    try:
        val = float(text.get_text())
        text.set_color("white" if abs(val) >= 0.5 else "#333333")
    except ValueError:
        pass

ax.set_title("Correlation Between Behavioral Statistics", fontsize=15, fontweight="bold", color="#2D2D2D", pad=20)
plt.tight_layout()
out_path = FIGURES_DIR / "correlation_matrix.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# 7. Visualization 5: Lexical diversity vs word count
# --------------------------------------------------------------------
print("\n[5/6] Lexical Diversity vs Word Count...")

upper_limit = 1200
df_filtered = df[df["word_count"] <= upper_limit]
df_sample = df_filtered.sample(n=min(3000, len(df_filtered)), random_state=42)

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

for label, color in {"Positive": "#2E8B57", "Negative": "#DC3545"}.items():
    subset = df_sample[df_sample["sentiment_label"] == label]
    ax.scatter(subset["word_count"], subset["lexical_diversity"],
               color=color, alpha=0.25, s=18, label=label, zorder=3)

z = np.polyfit(df_sample["word_count"], df_sample["lexical_diversity"], 1)
p = np.poly1d(z)
x_line = np.linspace(df_sample["word_count"].min(), upper_limit, 300)
ax.plot(x_line, p(x_line), color="#333333", linewidth=2, linestyle="--", label="Trend", zorder=5)

ax.set_title("Lexical Diversity vs Word Count", fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("Word Count", fontsize=12, color="#444444", labelpad=10)
ax.set_ylabel("Lexical Diversity", fontsize=12, color="#444444", labelpad=10)
ax.set_xlim(0, 1100)
ax.set_xticks(np.arange(0, 1001, 200))
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(color="#E0E0E0", linewidth=0.7, linestyle="--")

corr_val = df_sample["word_count"].corr(df_sample["lexical_diversity"])
ax.text(0.98, 0.95, f"r = {corr_val:.2f}", transform=ax.transAxes,
        fontsize=12, color="#333333", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))
ax.legend(title="Sentiment", fontsize=11, title_fontsize=12,
          framealpha=0.9, edgecolor="#CCCCCC", facecolor="white",
          loc="upper right", bbox_to_anchor=(0.98, 0.88))

plt.tight_layout()
out_path = FIGURES_DIR / "lexical_diversity_vs_wordcount.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# 8. Visualization 6: Sentiment by lexical diversity segment
# --------------------------------------------------------------------
print("\n[6/6] Sentiment by Lexical Diversity Segment...")

df["ld_segment"] = pd.cut(
    df["lexical_diversity"],
    bins=[0, 0.4, 0.7, 1.0],
    labels=["Low\n(0.0 - 0.4)", "Medium\n(0.4 - 0.7)", "High\n(0.7 - 1.0)"],
)

segment_counts = df.groupby(["ld_segment", "sentiment_label"]).size().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

x = np.arange(len(segment_counts.index))
width = 0.35

bars1 = ax.bar(x - width / 2, segment_counts["Positive"], width=width,
               color="#2E8B57", alpha=0.85, edgecolor="white", linewidth=0.5, label="Positive")
bars2 = ax.bar(x + width / 2, segment_counts["Negative"], width=width,
               color="#DC3545", alpha=0.85, edgecolor="white", linewidth=0.5, label="Negative")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom",
            fontsize=10, color="#333333")

ax.set_title("Sentiment Distribution by Lexical Diversity Segment",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("Lexical Diversity Segment", fontsize=12, color="#444444", labelpad=10)
ax.set_ylabel("Number of Reviews", fontsize=12, color="#444444", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(segment_counts.index, fontsize=11, color="#333333")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, linestyle="--")
ax.legend(title="Sentiment", fontsize=11, title_fontsize=12,
          framealpha=0.9, edgecolor="#CCCCCC", facecolor="white")

plt.tight_layout()
out_path = FIGURES_DIR / "lexical_diversity_segments.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved to: {out_path}")

# --------------------------------------------------------------------
# Done
# --------------------------------------------------------------------
print("\n" + "=" * 50)
print("  ALL EDA TASKS COMPLETED SUCCESSFULLY")
print(f"  6 figures saved to: {FIGURES_DIR}")
print(f"  2 tables  saved to: {TABLES_DIR}")
print("=" * 50)
