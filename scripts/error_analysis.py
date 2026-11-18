"""Error analysis for the best CantoNLU validation models.

Milestone 5:
- validation split only
- identify misclassified examples
- summarize error types
- save errors for later qualitative review

Do NOT use the test split here.

Run from repository root:
    python scripts/error_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import ROOT, load_split  # noqa: E402


def sentiment_model():
    """
    Best M5 validation model for sentiment.
    """
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 2)),
        LogisticRegression(C=30, max_iter=1000),
    )

def ld_model():
    """Best M5 validation model for language detection."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 2)),
        LogisticRegression(C=30, max_iter=1000),
    )


def analyse_task(task, model):
    train = load_split(task, "train")
    valid = load_split(task, "valid")

    model.fit(train["sentence"], train["label"])
    pred = model.predict(valid["sentence"])

    results = valid.copy().reset_index().rename(
        columns={"index": "validation_row"}
    )
    results["predicted_label"] = pred

    errors = results[
        results["label"] != results["predicted_label"]
    ].copy()

    errors = errors.rename(columns={"label": "true_label"})
    errors["task"] = task

    error_counts = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    candidate_groups = []

    for _, group in errors.groupby(
        ["true_label", "predicted_label"], sort=False
    ):
        sample_size = min(8, len(group))
        candidate_groups.append(
            group.sample(n=sample_size, random_state=3052)
        )

    candidates = pd.concat(candidate_groups, ignore_index=True)

    return errors, error_counts, candidates


def main():
    models = {
        "sentiment": sentiment_model(),
        "ld": ld_model(),
    }

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    for task, model in models.items():
        errors, counts, candidates = analyse_task(task, model)

        errors_path = output_dir / f"error_analysis_{task}.csv"
        counts_path = output_dir / f"error_counts_{task}.csv"
        candidates_path = output_dir / f"error_candidates_{task}.csv"

        errors.to_csv(errors_path, index=False)
        counts.to_csv(counts_path, index=False)
        candidates.to_csv(candidates_path, index=False)

        print(f"\n{task.upper()}")
        print(f"Misclassified validation examples: {len(errors)}")
        print("\nError types:")
        print(counts.to_string(index=False))

        print(f"\nSaved {errors_path}")
        print(f"Saved {counts_path}")
        print(f"Saved {candidates_path}")

if __name__ == "__main__":
    main()