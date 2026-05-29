# -*- coding: utf-8 -*-
"""
04_modeling.py
--------------
Research Question 1:
    Can traditional machine learning algorithms reliably and interpretably
    classify sentiment to support first-stage quality control in customer
    feedback loops?
 
Approach:
    - Two feature representations: TF-IDF and Bag-of-Words.
    - Three ML models with hyperparameter tuning (5-fold stratified CV grid
      search): Logistic Regression, Multinomial Naive Bayes, Linear SVM.
    - One baseline: DummyClassifier(strategy="most_frequent").
    - Interpretability: top positive / top negative features from the best
      Logistic Regression model.
 
Reads:
    data/processed/cleaned_data_set.csv
 
Writes:
    visuals/tables/model_comparison.csv
    visuals/tables/best_hyperparameters.csv
    visuals/figures/model_comparison_bars.png
    visuals/figures/confusion_matrices.png
    visuals/figures/roc_curves.png
    visuals/figures/top_features_logreg.png
    models/best_model.pkl
    models/best_vectorizer.pkl
 
Run from the repository root:
 
    python scripts/04_modeling.py
 
Approx. runtime: 12-18 minutes on an Apple M1 / 16 GB.
"""
 
import sys
import time
import warnings
from pathlib import Path
 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
 
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data_set.csv"
FIGURES_DIR = PROJECT_ROOT / "visuals" / "figures"
TABLES_DIR = PROJECT_ROOT / "visuals" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
 
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
# --------------------------------------------------------------------
# Style (matches the EDA script colors)
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")
COLOR_POS = "#2E8B57"   # green for positive / good outcome
COLOR_NEG = "#DC3545"   # red for negative / errors
COLOR_NEUTRAL = "#4A6FA5"  # blue for neutral / baseline
BG_COLOR = "#F8F9FA"
 
RANDOM_STATE = 42
 
# --------------------------------------------------------------------
# 1. Load processed data
# --------------------------------------------------------------------
print("=" * 60)
print("  IMDB SENTIMENT ANALYSIS  —  RQ1: MODELING")
print("=" * 60)
 
if not PROCESSED_PATH.exists():
    print(
        f"ERROR: Processed file not found at '{PROCESSED_PATH}'.\n"
        f"Please run 'python scripts/02_preprocess_data.py' first."
    )
    sys.exit(1)
 
df = pd.read_csv(PROCESSED_PATH)
df = df.dropna(subset=["cleaned_review"]).reset_index(drop=True)
print(f"\nDataset loaded — shape: {df.shape}")
print(f"Positive: {(df['sentiment'] == 1).sum():,}   "
      f"Negative: {(df['sentiment'] == 0).sum():,}")
 
X = df["cleaned_review"].astype(str).values
y = df["sentiment"].astype(int).values
 
# --------------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# --------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)
print(f"\nTrain size: {len(X_train):,}   Test size: {len(X_test):,}")
 
# --------------------------------------------------------------------
# 3. Vectorizers
# --------------------------------------------------------------------
# We restrict to unigrams + bigrams; min_df=5 removes extremely rare
# terms that only add noise; max_features=20000 keeps memory bounded.
VECTORIZER_KWARGS = dict(ngram_range=(1, 2), max_features=20000, min_df=5)
 
vectorizers = {
    "TF-IDF": TfidfVectorizer(**VECTORIZER_KWARGS),
    "BoW": CountVectorizer(**VECTORIZER_KWARGS),
}
 
# --------------------------------------------------------------------
# 4. Model + grid definitions
# --------------------------------------------------------------------
# Each entry: model_name -> (estimator, param_grid)
MODEL_GRID = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {"alpha": [0.1, 0.5, 1.0, 2.0]},
    ),
    "Linear SVM": (
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000),
        {"C": [0.01, 0.1, 1.0, 10.0]},
    ),
}
 
# --------------------------------------------------------------------
# 5. Baseline (DummyClassifier — must beat this)
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 1 — BASELINE")
print("=" * 60)
print("\nTraining DummyClassifier(strategy='most_frequent')...")
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
# DummyClassifier ignores X, but it needs a 2D array, so we use a constant.
dummy.fit(np.zeros((len(X_train), 1)), y_train)
y_pred_dummy = dummy.predict(np.zeros((len(X_test), 1)))
 
baseline_results = {
    "model": "Baseline (most_frequent)",
    "vectorizer": "—",
    "best_params": "—",
    "cv_f1_mean": np.nan,
    "cv_f1_std": np.nan,
    "test_accuracy": accuracy_score(y_test, y_pred_dummy),
    "test_precision": precision_score(y_test, y_pred_dummy, zero_division=0),
    "test_recall": recall_score(y_test, y_pred_dummy, zero_division=0),
    "test_f1": f1_score(y_test, y_pred_dummy, zero_division=0),
    "test_roc_auc": np.nan,
}
print(f"   Baseline accuracy: {baseline_results['test_accuracy']:.4f}")
 
# --------------------------------------------------------------------
# 6. Grid search loop: 2 vectorizers × 3 models
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 2 — GRID SEARCH (2 vectorizers × 3 models, 5-fold CV)")
print("=" * 60)
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results_rows = [baseline_results]
fitted_models = {}      # (vec_name, model_name) -> (vectorizer, best_estimator)
predictions = {}        # (vec_name, model_name) -> (y_pred, y_score)
 
overall_start = time.time()
 
for vec_name, vectorizer in vectorizers.items():
    print(f"\n>>> Vectorizing with {vec_name} ...")
    t0 = time.time()
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    print(f"    matrix shape: {Xtr.shape}  ({time.time() - t0:.1f}s)")
 
    for model_name, (estimator, grid) in MODEL_GRID.items():
        print(f"\n    Grid-searching: {model_name} on {vec_name} ...")
        t0 = time.time()
 
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(Xtr, y_train)
 
        best = gs.best_estimator_
        y_pred = best.predict(Xte)
 
        # Decision scores for ROC: LinearSVC has decision_function; others
        # may have predict_proba. We prefer predict_proba when available.
        if hasattr(best, "predict_proba"):
            y_score = best.predict_proba(Xte)[:, 1]
        else:
            y_score = best.decision_function(Xte)
 
        row = {
            "model": model_name,
            "vectorizer": vec_name,
            "best_params": str(gs.best_params_),
            "cv_f1_mean": gs.cv_results_["mean_test_score"][gs.best_index_],
            "cv_f1_std": gs.cv_results_["std_test_score"][gs.best_index_],
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_score),
        }
        results_rows.append(row)
        fitted_models[(vec_name, model_name)] = (vectorizer, best)
        predictions[(vec_name, model_name)] = (y_pred, y_score)
 
        print(
            f"      done in {time.time() - t0:.1f}s — "
            f"best={gs.best_params_}, "
            f"test_acc={row['test_accuracy']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, "
            f"roc_auc={row['test_roc_auc']:.4f}"
        )
 
print(f"\nTotal grid-search time: {(time.time() - overall_start) / 60:.1f} min")
 
# --------------------------------------------------------------------
# 7. Save comparison table + best-hyperparameter table
# --------------------------------------------------------------------
results_df = pd.DataFrame(results_rows)
 
# Round numeric columns for readability.
num_cols = ["cv_f1_mean", "cv_f1_std",
            "test_accuracy", "test_precision", "test_recall",
            "test_f1", "test_roc_auc"]
results_df[num_cols] = results_df[num_cols].round(4)
 
comparison_path = TABLES_DIR / "model_comparison.csv"
results_df.to_csv(comparison_path, index=False)
print(f"\nSaved: {comparison_path}")
 
hyper_df = results_df[results_df["best_params"] != "—"][
    ["model", "vectorizer", "best_params", "cv_f1_mean", "cv_f1_std"]
].reset_index(drop=True)
hyper_path = TABLES_DIR / "best_hyperparameters.csv"
hyper_df.to_csv(hyper_path, index=False)
print(f"Saved: {hyper_path}")
 
# --------------------------------------------------------------------
# 8. Identify the overall best (model, vectorizer) by test F1
# --------------------------------------------------------------------
ml_only = results_df[results_df["vectorizer"] != "—"].copy()
best_idx = ml_only["test_f1"].idxmax()
best_row = ml_only.loc[best_idx]
BEST_VEC = best_row["vectorizer"]
BEST_MODEL = best_row["model"]
print(f"\nBest configuration: {BEST_MODEL} + {BEST_VEC}  "
      f"(test F1 = {best_row['test_f1']:.4f})")
 
# --------------------------------------------------------------------
# 9. Persist best model + its vectorizer (RQ3 will reload these)
# --------------------------------------------------------------------
best_vec, best_est = fitted_models[(BEST_VEC, BEST_MODEL)]
joblib.dump(best_est, MODELS_DIR / "best_model.pkl")
joblib.dump(best_vec, MODELS_DIR / "best_vectorizer.pkl")
 
# Also persist a small metadata file so RQ3 / the website can pick it up.
meta = {
    "best_model_name": BEST_MODEL,
    "best_vectorizer_name": BEST_VEC,
    "best_params": best_row["best_params"],
    "test_accuracy": float(best_row["test_accuracy"]),
    "test_f1": float(best_row["test_f1"]),
    "test_roc_auc": float(best_row["test_roc_auc"]),
    "random_state": RANDOM_STATE,
}
pd.Series(meta).to_csv(MODELS_DIR / "best_model_meta.csv", header=False)
print(f"Saved best model to: {MODELS_DIR / 'best_model.pkl'}")
print(f"Saved best vectorizer to: {MODELS_DIR / 'best_vectorizer.pkl'}")
 
# --------------------------------------------------------------------
# 10. Visualization 1 — Model comparison bar chart
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  STEP 3 — VISUALIZATIONS")
print("=" * 60)
print("\n[1/4] Model comparison bar chart ...")
 
plot_df = results_df.copy()
plot_df["label"] = plot_df.apply(
    lambda r: f"{r['model']}\n({r['vectorizer']})" if r["vectorizer"] != "—"
    else r["model"], axis=1,
)
plot_df = plot_df.sort_values("test_f1", ascending=True).reset_index(drop=True)
 
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
 
colors = []
for _, r in plot_df.iterrows():
    if r["vectorizer"] == "—":
        colors.append("#9AA0A6")        # gray for baseline
    elif r["model"] == BEST_MODEL and r["vectorizer"] == BEST_VEC:
        colors.append(COLOR_POS)        # green for winner
    else:
        colors.append(COLOR_NEUTRAL)    # blue for the rest
 
bars = ax.barh(plot_df["label"], plot_df["test_f1"],
               color=colors, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
for bar, val in zip(bars, plot_df["test_f1"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=10, color="#333333")
 
ax.set_title("Model Comparison — Test F1 Score",
             fontsize=15, fontweight="bold", color="#2D2D2D", pad=18)
ax.set_xlabel("F1 Score (test set)", fontsize=12, color="#444444", labelpad=10)
ax.set_xlim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(False)
 
plt.tight_layout()
out_path = FIGURES_DIR / "model_comparison_bars.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# 11. Visualization 2 — Confusion matrices (3 ML models, best vectorizer only)
# --------------------------------------------------------------------
print("\n[2/4] Confusion matrices (best vectorizer per model) ...")
 
# For each model, pick its better vectorizer by test F1.
model_best_vec = {}
for m in MODEL_GRID:
    sub = ml_only[ml_only["model"] == m]
    pick = sub.loc[sub["test_f1"].idxmax(), "vectorizer"]
    model_best_vec[m] = pick
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_COLOR)
 
for ax, model_name in zip(axes, MODEL_GRID.keys()):
    vec_name = model_best_vec[model_name]
    y_pred, _ = predictions[(vec_name, model_name)]
    cm = confusion_matrix(y_test, y_pred)
 
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        cbar=False, square=True, linewidths=2, linecolor=BG_COLOR,
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_facecolor(BG_COLOR)
    ax.set_title(f"{model_name}\n({vec_name})",
                 fontsize=12, fontweight="bold", color="#2D2D2D", pad=10)
    ax.set_xlabel("Predicted", fontsize=10, color="#444444")
    ax.set_ylabel("Actual", fontsize=10, color="#444444")
    ax.tick_params(colors="#555555", labelsize=10)
 
plt.suptitle("Confusion Matrices (best vectorizer per model)",
             fontsize=14, fontweight="bold", color="#2D2D2D", y=1.05)
plt.tight_layout()
out_path = FIGURES_DIR / "confusion_matrices.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# 12. Visualization 3 — ROC curves (best vectorizer per model)
# --------------------------------------------------------------------
print("\n[3/4] ROC curves ...")
 
fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
 
palette = {
    "Logistic Regression": COLOR_POS,
    "Naive Bayes": "#E07B00",   # orange
    "Linear SVM": COLOR_NEUTRAL,
}
 
for model_name in MODEL_GRID.keys():
    vec_name = model_best_vec[model_name]
    _, y_score = predictions[(vec_name, model_name)]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, linewidth=2.2, color=palette[model_name],
            label=f"{model_name} ({vec_name}) — AUC = {auc:.3f}", zorder=3)
 
ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2,
        label="Random (AUC = 0.5)", zorder=2)
 
ax.set_title("ROC Curves — Best Vectorizer per Model",
             fontsize=14, fontweight="bold", color="#2D2D2D", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=11, color="#444444")
ax.set_ylabel("True Positive Rate", fontsize=11, color="#444444")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#CCCCCC")
ax.tick_params(colors="#555555", labelsize=10)
ax.grid(color="#E0E0E0", linewidth=0.7, linestyle="--")
ax.legend(loc="lower right", fontsize=10, framealpha=0.95,
          edgecolor="#CCCCCC", facecolor="white")
 
plt.tight_layout()
out_path = FIGURES_DIR / "roc_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# --------------------------------------------------------------------
# 13. Visualization 4 — Top features (interpretability)
# --------------------------------------------------------------------
print("\n[4/4] Top features for Logistic Regression (interpretability) ...")
 
# Use Logistic Regression + its better vectorizer (chosen by test F1).
lr_vec_name = model_best_vec["Logistic Regression"]
lr_vec, lr_model = fitted_models[(lr_vec_name, "Logistic Regression")]
 
feature_names = np.array(lr_vec.get_feature_names_out())
coefs = lr_model.coef_[0]
 
top_n = 20
top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
top_neg_idx = np.argsort(coefs)[:top_n]
 
pos_words = feature_names[top_pos_idx]
pos_coefs = coefs[top_pos_idx]
neg_words = feature_names[top_neg_idx]
neg_coefs = coefs[top_neg_idx]
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG_COLOR)
for ax in (ax1, ax2):
    ax.set_facecolor(BG_COLOR)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, colors="#444444")
    ax.grid(False)
 
# Positive side
bars1 = ax1.barh(pos_words, pos_coefs, color=COLOR_POS, alpha=0.9,
                 edgecolor="white", linewidth=0.5, height=0.7)
ax1.invert_yaxis()
ax1.set_title("Top 20 Words Predicting POSITIVE Sentiment",
              fontsize=13, fontweight="bold", color="#2D2D2D", pad=12)
ax1.set_xlabel("Logistic Regression Coefficient",
               fontsize=10, color="#444444", labelpad=8)
for bar, val in zip(bars1, pos_coefs):
    ax1.text(val + max(pos_coefs) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va="center", ha="left", fontsize=8, color="#555555")
 
# Negative side (plot absolute value so bars grow left-to-right visually)
neg_abs = np.abs(neg_coefs)
bars2 = ax2.barh(neg_words, neg_abs, color=COLOR_NEG, alpha=0.9,
                 edgecolor="white", linewidth=0.5, height=0.7)
ax2.invert_yaxis()
ax2.set_title("Top 20 Words Predicting NEGATIVE Sentiment",
              fontsize=13, fontweight="bold", color="#2D2D2D", pad=12)
ax2.set_xlabel("|Logistic Regression Coefficient|",
               fontsize=10, color="#444444", labelpad=8)
for bar, val, raw in zip(bars2, neg_abs, neg_coefs):
    ax2.text(val + max(neg_abs) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{raw:.2f}", va="center", ha="left", fontsize=8, color="#555555")
 
plt.suptitle(f"Most Predictive Words — Logistic Regression ({lr_vec_name})",
             fontsize=15, fontweight="bold", color="#2D2D2D", y=1.02)
plt.tight_layout()
out_path = FIGURES_DIR / "top_features_logreg.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved: {out_path}")
 
# Also persist the top-features list as a CSV (useful for the website).
top_features_df = pd.DataFrame({
    "rank": list(range(1, top_n + 1)),
    "positive_word": pos_words,
    "positive_coef": pos_coefs.round(4),
    "negative_word": neg_words,
    "negative_coef": neg_coefs.round(4),
})
top_features_df.to_csv(TABLES_DIR / "top_features_logreg.csv", index=False)
print(f"   Saved: {TABLES_DIR / 'top_features_logreg.csv'}")
 
# --------------------------------------------------------------------
# Done
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RQ1 MODELING — COMPLETED SUCCESSFULLY")
print(f"  Best model: {BEST_MODEL} + {BEST_VEC}")
print(f"  Test accuracy: {best_row['test_accuracy']:.4f}")
print(f"  Test F1:       {best_row['test_f1']:.4f}")
print(f"  Test ROC-AUC:  {best_row['test_roc_auc']:.4f}")
print("=" * 60)
