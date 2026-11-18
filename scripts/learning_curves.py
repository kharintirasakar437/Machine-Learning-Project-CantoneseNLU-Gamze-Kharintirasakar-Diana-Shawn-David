"""Learning curves: validation macro-F1 vs. training-set size (sensitivity analysis).

For each task, the tuned TF-IDF models (NB and LR, same settings as scripts/uncertainty.py
and scripts/main_results.py) are trained on 10%, 25%, 50% and 100% of the train split and
scored on the full validation split. Each fraction below 100% is drawn three times
(seeds 3052, 3053, 3054) as a label-stratified random subset, so the curve shows the mean
and the spread across seeds. At 100% every seed gives the same training set and both models
are deterministic, so that point is fitted once (spread 0).

Train macro-F1 (on the subset the model was fitted on) is also recorded, so the gap between
the train and validation curves shows whether more data would still help (overfitting) or
not (underfitting).

The test split is NOT used. Run from the repository root:
    python scripts/learning_curves.py            # 4 parallel fits
    python scripts/learning_curves.py --jobs 1   # less memory
Writes results/learning_curves.csv (every fit), results/learning_curves_summary.csv
(mean and std over seeds) and results/learning_curves.png.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, load_split  # noqa: E402
from train_baselines import SEEDS  # noqa: E402
from uncertainty import TASK_TITLES, TASKS, tuned_models  # noqa: E402

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
RESULTS = ROOT / "results"
COLORS = {"NB": "#1f77b4", "LR": "#d62728"}


def subsample(train, fraction, seed):
    """Label-stratified random subset of the train split."""
    if fraction >= 1.0:
        return train
    part, _ = train_test_split(train, train_size=fraction, stratify=train["label"],
                               random_state=seed)
    return part


def fit_one(task, model_name, setting, model, train, valid, fraction, seed):
    part = subsample(train, fraction, seed)
    model = clone(model)
    start = time.perf_counter()
    model.fit(part["sentence"], part["label"])
    seconds = time.perf_counter() - start
    row = {
        "task": task, "model": model_name, "setting": setting, "fraction": fraction,
        "seed": seed if fraction < 1.0 else "all", "n_train": len(part),
        "train_macro_f1": f1_score(part["label"], model.predict(part["sentence"]), average="macro"),
        "valid_macro_f1": f1_score(valid["label"], model.predict(valid["sentence"]), average="macro"),
        "fit_seconds": seconds,
    }
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def plot(summary, out):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.3))
    for ax, task in zip(axes, TASKS):
        t = summary[summary["task"] == task]
        for model in t["model"].unique():
            m = t[t["model"] == model].sort_values("fraction")
            color = COLORS["NB" if "NB" in model else "LR"]
            x = m["fraction"] * 100
            ax.errorbar(x, m["valid_mean"], yerr=m["valid_std"], marker="o", capsize=3,
                        color=color, label=f"{model} — validation")
            ax.plot(x, m["train_mean"], linestyle="--", marker=".", color=color, alpha=0.6,
                    label=f"{model} — train subset")
        ax.set_xticks([f * 100 for f in FRACTIONS], [f"{int(f * 100)}%" for f in FRACTIONS])
        ax.set_xlabel("Share of the training split used")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1.02)
        n_full = int(t[t["fraction"] == 1.0]["n_train"].iloc[0])
        ax.set_title(f"{TASK_TITLES[task]} (100% = {n_full:,} sentences)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Learning curves: mean of seeds 3052-3054 (error bars = std across seeds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4, help="parallel fits (memory-bound)")
    args = parser.parse_args()

    jobs = []
    for task in TASKS:
        train, valid = load_split(task, "train"), load_split(task, "valid")
        for name, (model, setting) in tuned_models(task).items():
            for fraction in FRACTIONS:
                for seed in (SEEDS if fraction < 1.0 else SEEDS[:1]):
                    jobs.append(delayed(fit_one)(task, name, setting, model, train, valid,
                                                 fraction, seed))
    # Largest fits first, so the slow 100% sentiment LR fit does not run last on its own
    jobs.sort(key=lambda j: -j[1][6])
    rows = Parallel(n_jobs=args.jobs)(jobs)

    table = pd.DataFrame(rows).sort_values(["task", "model", "fraction", "seed"],
                                           key=lambda s: s.astype(str))
    summary = (table.groupby(["task", "model", "setting", "fraction"], sort=False)
                    .agg(n_train=("n_train", "first"), n_seeds=("seed", "size"),
                         valid_mean=("valid_macro_f1", "mean"),
                         valid_std=("valid_macro_f1", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                         train_mean=("train_macro_f1", "mean"),
                         train_std=("train_macro_f1", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                         fit_seconds_mean=("fit_seconds", "mean"))
                    .reset_index())
    summary["gap_train_minus_valid"] = summary["train_mean"] - summary["valid_mean"]
    summary["task"] = pd.Categorical(summary["task"], TASKS, ordered=True)
    summary = summary.sort_values(["task", "model", "fraction"])

    out_csv, out_sum = RESULTS / "learning_curves.csv", RESULTS / "learning_curves_summary.csv"
    out_png = RESULTS / "learning_curves.png"
    table.to_csv(out_csv, index=False, float_format="%.3f")
    summary.to_csv(out_sum, index=False, float_format="%.3f")
    plot(summary, out_png)
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nSaved {out_csv}\nSaved {out_sum}\nSaved {out_png}")


if __name__ == "__main__":
    main()
