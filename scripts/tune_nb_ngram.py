"""Tune Multinomial Naive Bayes: character n-gram range x alpha, on the validation split.

Second stage of the search: alpha values come from the neighbourhood of the best
alpha in scripts/tune_nb.py (results/tune_nb_alpha.csv). The test split is NOT used.
Run from the repository root:
    python scripts/tune_nb_ngram.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, load_split  # noqa: E402
from train_baselines import evaluate  # noqa: E402

# Best alpha from the first stage and its two neighbours, per task
ALPHAS = {"sentiment": [0.003, 0.01, 0.03], "ld": [0.001, 0.003, 0.01]}
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (1, 4)]

rows = []
for task in ("sentiment", "ld"):
    train = load_split(task, "train")
    valid = load_split(task, "valid")
    for ngram_range in NGRAM_RANGES:
        for alpha in ALPHAS[task]:
            model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=ngram_range),
                MultinomialNB(alpha=alpha),
            )
            result = {"task": task, "ngram_range": str(ngram_range), "alpha": str(alpha),
                      **evaluate(model, train, valid)}
            print(result)
            rows.append(result)

table = pd.DataFrame(rows)
out = ROOT / "results" / "tune_nb_ngram.csv"
table.to_csv(out, index=False, float_format="%.3f")
print(f"\nSaved {out}")
