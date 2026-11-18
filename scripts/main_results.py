"""Main results table and figure for all models, from one script (validation split only).

Every model is refitted on the train split with the settings chosen on the validation split
by the tuning scripts, then scored on the validation split, so the table regenerates from code:
  - NB:  best row of results/tune_nb_ngram.csv   (scripts/tune_nb.py, tune_nb_ngram.py)
  - LR:  results/best_lr.csv                     (scripts/tune_lr.py)
  - kNN: best k in results/tune_knn_labse.csv    (scripts/knn_labse.py; LaBSE embeddings cached)
Majority and stratified-random are the naive benchmarks (stratified: mean of seeds 3052-3054).

The test split is NOT used here; it stays locked until the final report.
Run from the repository root, after the tuning scripts have run:
    python scripts/main_results.py
Writes results/main_results_valid.csv (table) and results/main_results_valid.png (figure).
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load_split  # noqa: E402
from train_baselines import SEEDS, evaluate  # noqa: E402
from tune_lr import lr_pipeline  # noqa: E402

TASKS = ("sentiment", "ld")
MODEL_ORDER = ["majority", "stratified (mean of 3 seeds)", "tfidf + NB (tuned)",
               "tfidf + LR (tuned)", "LaBSE + kNN"]
RESULTS = ROOT / "results"


def best_row(csv_name, task):
    """Highest validation macro-F1 row of a tuning CSV for one task."""
    table = pd.read_csv(RESULTS / csv_name, dtype={"alpha": str})
    table = table[table["task"] == task]
    return table.loc[table["macro_f1"].idxmax()]


def knn_embeddings(task, split, sentences):
    """Cached LaBSE embeddings (same cache as scripts/knn_labse.py); encode only if missing."""
    cache = ROOT / "embeddings" / f"labse_{task}_{split}.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(sentences):
            return emb
    from knn_labse import get_embeddings  # imports sentence-transformers: only when needed
    from sentence_transformers import SentenceTransformer
    return get_embeddings(SentenceTransformer("sentence-transformers/LaBSE"), task, split, sentences)


def knn_metrics(task, train, valid, k):
    x_train = knn_embeddings(task, "train", train["sentence"].tolist())
    x_valid = knn_embeddings(task, "valid", valid["sentence"].tolist())
    pred = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(x_train, train["label"]).predict(x_valid)
    y = valid["label"]
    return {"accuracy": round(accuracy_score(y, pred), 3),
            "macro_f1": round(f1_score(y, pred, average="macro"), 3),
            "micro_f1": round(f1_score(y, pred, average="micro"), 3),
            "weighted_f1": round(f1_score(y, pred, average="weighted"), 3)}


def run_task(task):
    train, valid = load_split(task, "train"), load_split(task, "valid")
    rows = []

    def add(model, metrics, setting=""):
        rows.append({"task": task, "model": model, "setting": setting, **metrics})
        print(rows[-1], flush=True)

    add("majority", evaluate(DummyClassifier(strategy="most_frequent"), train, valid))

    strat = pd.DataFrame([evaluate(DummyClassifier(strategy="stratified", random_state=s), train, valid)
                          for s in SEEDS])
    add("stratified (mean of 3 seeds)", strat.mean().round(3).to_dict(), f"seeds {SEEDS[0]}-{SEEDS[-1]}")

    nb = best_row("tune_nb_ngram.csv", task)
    ngram, alpha = ast.literal_eval(nb["ngram_range"]), float(nb["alpha"])
    nb_model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=ngram), MultinomialNB(alpha=alpha))
    add("tfidf + NB (tuned)", evaluate(nb_model, train, valid), f"alpha={alpha:g}, ngram={ngram}")

    lr = best_row("best_lr.csv", task)
    ngram, c = ast.literal_eval(lr["ngram_range"]), float(lr["C"])
    add("tfidf + LR (tuned)", evaluate(lr_pipeline(c, ngram), train, valid), f"C={c:g}, ngram={ngram}")

    k = int(best_row("tune_knn_labse.csv", task)["k"])
    add("LaBSE + kNN", knn_metrics(task, train, valid, k), f"k={k}")
    return rows


def plot(table, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(MODEL_ORDER)
    x = np.arange(len(TASKS))
    for i, model in enumerate(MODEL_ORDER):
        vals = [table[(table.task == t) & (table.model == model)]["macro_f1"].iloc[0] for t in TASKS]
        bars = ax.bar(x + (i - (len(MODEL_ORDER) - 1) / 2) * width, vals, width, label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, ["Sentiment", "Language detection"])
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("All models on the validation split (hyperparameters tuned on validation)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    table = pd.DataFrame([r for task in TASKS for r in run_task(task)])
    table["task"] = pd.Categorical(table["task"], TASKS, ordered=True)
    table["model"] = pd.Categorical(table["model"], MODEL_ORDER, ordered=True)
    table = table.sort_values(["task", "model"])

    out_csv = RESULTS / "main_results_valid.csv"
    out_png = RESULTS / "main_results_valid.png"
    table.to_csv(out_csv, index=False, float_format="%.3f")
    plot(table, out_png)
    print(f"\nSaved {out_csv}\nSaved {out_png}")


if __name__ == "__main__":
    main()
