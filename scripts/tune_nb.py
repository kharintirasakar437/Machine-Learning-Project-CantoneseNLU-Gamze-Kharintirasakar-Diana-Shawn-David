"""Tune Multinomial Naive Bayes on the validation split.

Search space: smoothing alpha and the character n-gram range.
Selection rule: highest validation macro-F1. The test split is NOT used.
Run from the repository root:
    python scripts/tune_nb.py
"""

import sys
from pathlib import Path
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Make src/ importable so we can reuse our data loader and evaluation function
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, load_split  # noqa: E402
from train_baselines import evaluate  # noqa: E402

ALPHAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

rows = []
for task in ("sentiment", "ld"):
    train = load_split(task, "train")
    valid = load_split(task, "valid")
    for alpha in ALPHAS:
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3)),
            MultinomialNB(alpha=alpha),
        )
        result = {"task": task, "alpha": alpha, **evaluate(model, train, valid)}
        print(result)
        rows.append(result)

table = pd.DataFrame(rows)
table["alpha"] = table["alpha"].astype(str)  # keep small alphas like 0.0001 readable in the CSV
out = ROOT / "results" / "tune_nb_alpha.csv"

table.to_csv(out, index=False, float_format="%.3f")
print(f"\nSaved {out}")
