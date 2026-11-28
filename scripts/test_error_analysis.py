"""Error analysis for the locked Milestone 6 final-test predictions.

This script does NOT train, tune, or evaluate models.
It only reads prediction CSVs produced by scripts/final_test.py.

Usage after the official one-time final test run:

    python scripts/test_error_analysis.py
"""

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
PRED_COL = "pred: tfidf + LR (tuned)"


def analyse_task(task):
    input_path = RESULTS_DIR / f"final_test_predictions_{task}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist.\n"
            "Do not run final_test.py yourself. Wait for the official final-test "
            "predictions to be produced and merged."
        )

    df = pd.read_csv(input_path)

    required = {"sentence", "label", PRED_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {sorted(missing)}"
        )

    errors = df[df["label"] != df[PRED_COL]].copy()
    errors = errors.rename(columns={PRED_COL: "prediction"})

    errors.insert(0, "row", errors.index)

    counts = (
        errors.groupby(["label", "prediction"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    errors_path = RESULTS_DIR / f"test_error_analysis_{task}.csv"
    counts_path = RESULTS_DIR / f"test_error_counts_{task}.csv"

    errors[["row", "sentence", "label", "prediction"]].to_csv(
        errors_path, index=False
    )
    counts.to_csv(counts_path, index=False)

    print(f"\n{task.upper()}")
    print("-" * 50)
    print(f"Total examples: {len(df)}")
    print(f"Total errors:   {len(errors)}")
    print("\nError directions:")
    print(counts.to_string(index=False))

    return errors, counts


def sentiment_summary(errors):
    involves_ok = (
        (errors["label"] == "ok")
        | (errors["prediction"] == "ok")
    ).sum()

    total = len(errors)
    pct = 100 * involves_ok / total if total else 0

    direct_smile_cry = (
        ((errors["label"] == "smile") & (errors["prediction"] == "cry"))
        | ((errors["label"] == "cry") & (errors["prediction"] == "smile"))
    ).sum()

    print("\nSentiment pattern check")
    print(f"Errors involving ok: {involves_ok}/{total} ({pct:.1f}%)")
    print(f"Direct smile <-> cry errors: {direct_smile_cry}")

    return {
        "task": "sentiment",
        "total_errors": total,
        "focus_class": "ok",
        "errors_involving_focus_class": involves_ok,
        "pct_involving_focus_class": pct,
        "direct_pure_class_confusions": direct_smile_cry,
    }


def ld_summary(errors):
    involves_mixed = (
        (errors["label"] == 2)
        | (errors["prediction"] == 2)
    ).sum()

    total = len(errors)
    pct = 100 * involves_mixed / total if total else 0

    direct_0_1 = (
        ((errors["label"] == 0) & (errors["prediction"] == 1))
        | ((errors["label"] == 1) & (errors["prediction"] == 0))
    ).sum()

    print("\nLanguage-detection pattern check")
    print(f"Errors involving code-mixed class 2: {involves_mixed}/{total} ({pct:.1f}%)")
    print(f"Direct Cantonese <-> Mandarin errors: {direct_0_1}")

    return {
        "task": "ld",
        "total_errors": total,
        "focus_class": "2",
        "errors_involving_focus_class": involves_mixed,
        "pct_involving_focus_class": pct,
        "direct_pure_class_confusions": direct_0_1,
    }


def main():
    sentiment_errors, _ = analyse_task("sentiment")
    ld_errors, _ = analyse_task("ld")

    summary = pd.DataFrame(
        [
            sentiment_summary(sentiment_errors),
            ld_summary(ld_errors),
        ]
    )

    output_path = RESULTS_DIR / "test_error_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()