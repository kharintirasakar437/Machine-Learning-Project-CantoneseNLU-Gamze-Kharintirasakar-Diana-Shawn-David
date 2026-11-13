"""k-NN on frozen LaBSE sentence embeddings, tuned on the validation split.

LaBSE is used as a fixed feature extractor (no fine-tuning), so there is no
vectoriser to fit inside a Pipeline here: the embeddings do not depend on the
train split at all, and cannot leak test/validation information the way a
TF-IDF vocabulary could.

Embeddings are computed once per task/split and cached under embeddings/
(git-ignored; see .gitignore) so repeated tuning runs do not re-encode text.

Search space: number of neighbours k.
Selection rule: highest validation macro-F1. The test split is NOT used.
Run from the repository root:
    python scripts/knn_labse.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, load_split  # noqa: E402

SEED = 3052
TASKS = ("sentiment", "ld")
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]

EMB_DIR = ROOT / "embeddings"
EMB_DIR.mkdir(exist_ok=True)


def get_embeddings(model, task, split, sentences):
    """Return cached LaBSE embeddings for one task/split, computing them on first use."""
    cache_path = EMB_DIR / f"labse_{task}_{split}.npy"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if len(embeddings) == len(sentences):
            return embeddings
    embeddings = model.encode(
        sentences, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    np.save(cache_path, embeddings)
    return embeddings


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "micro_f1": round(f1_score(y_true, y_pred, average="micro"), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 3),
    }


def main():
    model = SentenceTransformer("sentence-transformers/LaBSE")

    rows = []
    curves = {}
    for task in TASKS:
        train = load_split(task, "train")
        valid = load_split(task, "valid")

        x_train = get_embeddings(model, task, "train", train["sentence"].tolist())
        x_valid = get_embeddings(model, task, "valid", valid["sentence"].tolist())

        task_rows = []
        for k in K_VALUES:
            clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            clf.fit(x_train, train["label"])
            pred = clf.predict(x_valid)
            result = {"task": task, "k": k, **evaluate(valid["label"], pred)}
            print(result)
            rows.append(result)
            task_rows.append(result)
        curves[task] = task_rows

    table = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "tune_knn_labse.csv"
    table.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"\nSaved {out_csv}")

    best = table.loc[table.groupby("task")["macro_f1"].idxmax()]
    print("\nBest k per task (by validation macro-F1):")
    print(best.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for task, task_rows in curves.items():
        task_table = pd.DataFrame(task_rows)
        ax.plot(task_table["k"], task_table["macro_f1"], marker="o", label=task)
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("k-NN on frozen LaBSE embeddings: validation macro-F1 vs. k")
    ax.legend(title="Task")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = ROOT / "results" / "tune_knn_labse.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
