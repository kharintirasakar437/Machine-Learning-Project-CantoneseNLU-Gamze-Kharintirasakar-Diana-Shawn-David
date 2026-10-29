"""Extra numbers and figures for the Milestone 4 progress report (validation split only).

For each task and each TF-IDF model it reports:
  - macro-F1 on train and on validation (the gap shows over- or underfitting),
  - training time of one fit (seconds, this machine) and the number of TF-IDF features,
  - per-class F1 on validation,
and saves validation confusion matrices for logistic regression as a figure.

The test split is NOT touched. Run from the repository root:
    python scripts/m4_report_extras.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import LABELS, ROOT, load_split  # noqa: E402
from train_baselines import tfidf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

# Readable names for the language-detection labels (DATA_CARD.md)
NAMES = {"sentiment": None, "ld": ["Cantonese", "Mandarin", "Code-mixed"]}

rows, per_class = [], []
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, task in zip(axes, ("sentiment", "ld")):
    train = load_split(task, "train")
    valid = load_split(task, "valid")
    models = {
        "tfidf + NB": make_pipeline(tfidf(), MultinomialNB()),
        "tfidf + LR": make_pipeline(tfidf(), LogisticRegression(max_iter=1000)),
    }
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train["sentence"], train["label"])
        seconds = time.perf_counter() - start

        train_pred = model.predict(train["sentence"])
        valid_pred = model.predict(valid["sentence"])
        rows.append({
            "task": task, "model": name,
            "train_macro_f1": f1_score(train["label"], train_pred, average="macro"),
            "valid_macro_f1": f1_score(valid["label"], valid_pred, average="macro"),
            "fit_seconds": seconds,
            "n_features": len(model[0].vocabulary_),  # character n-grams seen in train
        })
        labels = sorted(train["label"].unique())
        for label, f1 in zip(labels, f1_score(valid["label"], valid_pred, average=None, labels=labels)):
            per_class.append({"task": task, "model": name, "label": label, "valid_f1": f1})

        if name == "tfidf + LR":
            ConfusionMatrixDisplay.from_predictions(
                valid["label"], valid_pred, labels=labels, display_labels=NAMES[task],
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title({"sentiment": "Sentiment", "ld": "Language detection"}[task])

out_dir = ROOT / "results"
summary = pd.DataFrame(rows)
summary.to_csv(out_dir / "m4_train_vs_valid.csv", index=False, float_format="%.3f")
pd.DataFrame(per_class).to_csv(out_dir / "m4_per_class_valid.csv", index=False, float_format="%.3f")
fig.tight_layout()
fig.savefig(out_dir / "m4_confusion_lr_valid.png", dpi=200)

print(summary.round(3).to_string(index=False))
print(pd.DataFrame(per_class).round(3).to_string(index=False))
print("Label names:", LABELS)
