"""Tune L2 logistic regression on the validation split.

Search space: inverse regularisation strength C and the character n-gram range.
Selection rule: highest validation macro-F1 (ties: smaller C, then smaller n-gram range).
The test split is NOT used.

Two stages, like scripts/tune_nb.py -> scripts/tune_nb_ngram.py, because one LR fit on the
sentiment task takes minutes (results/m4_train_vs_valid.csv) and the full grid would take hours:
  1. C sweep with the default n-gram range (1, 3)
  2. n-gram sweep at the best C from stage 1 and its two neighbours

Run from the repository root:
    python scripts/tune_lr.py
Writes results/tune_lr.csv (every fit) and results/best_lr.csv (winner per task, read by
scripts/main_results.py).
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, load_split  # noqa: E402
from train_baselines import evaluate  # noqa: E402

TASKS = ("sentiment", "ld")
C_VALUES = [0.1, 0.3, 1, 3, 10, 30, 100]  # C = 1 is the untuned baseline
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (1, 4)]
DEFAULT_NGRAM = (1, 3)


def lr_pipeline(c, ngram_range):
    # The vectoriser sits inside the Pipeline, so it is fitted on train only.
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=ngram_range),
        LogisticRegression(C=c, max_iter=1000),
    )


def neighbours(values, best):
    """The best value and the values next to it in the (ordered) search grid."""
    i = values.index(best)
    return values[max(i - 1, 0): i + 2]


def main():
    rows = []
    for task in TASKS:
        train = load_split(task, "train")
        valid = load_split(task, "valid")

        def run(stage, c, ngram_range):
            result = {"task": task, "stage": stage, "C": c, "ngram_range": str(ngram_range),
                      **evaluate(lr_pipeline(c, ngram_range), train, valid)}
            print(result, flush=True)
            rows.append(result)
            return result

        # Stage 1: C at the default n-gram range
        stage1 = [run(1, c, DEFAULT_NGRAM) for c in C_VALUES]
        best_c = max(stage1, key=lambda r: (r["macro_f1"], -r["C"]))["C"]

        # Stage 2: n-gram range at the best C and its neighbours
        for c in neighbours(C_VALUES, best_c):
            for ngram_range in NGRAM_RANGES:
                if ngram_range == DEFAULT_NGRAM:
                    continue  # already fitted in stage 1
                run(2, c, ngram_range)

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "tune_lr.csv"
    table.to_csv(out, index=False, float_format="%.3f")
    print(f"\nSaved {out}")

    # Winner per task: highest macro-F1; ties go to the smaller C (stronger regularisation)
    best = (table.sort_values(["macro_f1", "C"], ascending=[False, True])
                 .groupby("task", sort=False).head(1).sort_values("task"))
    out_best = ROOT / "results" / "best_lr.csv"
    best.to_csv(out_best, index=False, float_format="%.3f")
    print("\nBest setting per task (by validation macro-F1):")
    print(best.to_string(index=False))
    print(f"Saved {out_best}")


if __name__ == "__main__":
    main()
