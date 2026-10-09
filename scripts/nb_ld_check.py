"""Diagnostic for the M3 memo: what does TF-IDF + Naive Bayes predict on language detection?

Run from the repository root:  python scripts/nb_ld_check.py
Uses the training and validation splits only.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# make src/ importable when this script is run from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import load_split  # noqa: E402
from train_baselines import tfidf  # noqa: E402

train = load_split("ld", "train")
valid = load_split("ld", "valid")

# fit_prior=False removes the class prior, to test whether the prior alone explains the predictions
for fit_prior in (True, False):
    nb = make_pipeline(tfidf(), MultinomialNB(fit_prior=fit_prior))
    nb.fit(train["sentence"], train["label"])
    counts = pd.Series(nb.predict(valid["sentence"])).value_counts().sort_index()
    print(f"fit_prior={fit_prior}: predicted label counts {counts.to_dict()}")
