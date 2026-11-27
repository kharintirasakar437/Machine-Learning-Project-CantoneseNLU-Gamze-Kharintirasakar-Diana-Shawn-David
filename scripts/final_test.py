"""Final evaluation of the tuned models (Milestone 6).

Every model is fitted once on the training split with the settings selected on validation in M5
(read from the tuning CSVs, never typed in by hand) and scored on one held-out split, with
bootstrap 95% confidence intervals and paired differences (same method as scripts/uncertainty.py).

    python scripts/final_test.py                  # VALIDATION split: dry run, writes results/final_valid*
    python scripts/final_test.py --final --knn    # TEST split, run ONCE: writes results/final_test*

The test split is locked: --final refuses to run if results/final_test.csv already exists.
"""
import os

# Fix the number of BLAS/OpenMP threads BEFORE numpy is imported (a precaution; the main fix for
# the run-to-run variation of tuned LR is the converged solver in final_models).
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_split  # noqa: E402
from train_baselines import SEED, SEEDS  # noqa: E402
from uncertainty import (RESULTS, TASKS, best_row, boot_scores, bootstrap_indices, ci,  # noqa: E402
                         load_groups, lr_setting, macro_f1_from_codes, nb_pipeline, nb_setting)

N_BOOT = 1000
TASK_TITLES = {"sentiment": "Sentiment analysis", "ld": "Language detection"}
SPLIT_NAMES = {"valid": "Validation", "test": "Test"}


# ---------------------------------------------------------------- models

def final_models(task):
    """Tuned settings from the M5 tuning CSVs. LR is run to convergence (tol=1e-6): with the
    default tol=1e-4 its result varied by up to 0.010 with rounding noise (see experiments/LOG.md)."""
    ngram, alpha = nb_setting(task)
    c, lr_ngram, _ = lr_setting(task)
    lr = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=lr_ngram),
                       LogisticRegression(C=c, tol=1e-6, max_iter=5000))
    return {
        "tfidf + NB (tuned)": (nb_pipeline(ngram, alpha), f"alpha={alpha:g}, ngram={ngram}"),
        "tfidf + LR (tuned)": (lr, f"C={c:g}, ngram={lr_ngram}, tol=1e-6"),
    }


def knn_predict(task, train, held, split):
    """LaBSE + kNN with the tuned k; uses the embeddings/ cache, encodes with LaBSE if missing."""
    from sklearn.neighbors import KNeighborsClassifier
    from main_results import knn_embeddings
    k = int(best_row("tune_knn_labse.csv", task)["k"])
    print(f"[{task}] LaBSE + kNN (k={k}) ...", flush=True)
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    model.fit(knn_embeddings(task, "train", train["sentence"].tolist()), train["label"])
    return model.predict(knn_embeddings(task, split, held["sentence"].tolist())), f"k={k}"


def predictions(task, split, use_knn):
    """Fit every model once on train; return the held-out split, predictions and settings."""
    train, held = load_split(task, "train"), load_split(task, split)
    x_train, y_train, x = train["sentence"], train["label"], held["sentence"]

    preds = {"majority": DummyClassifier(strategy="most_frequent").fit(x_train, y_train).predict(x)}
    settings = {"majority": ""}
    for seed in SEEDS:
        name = f"stratified (seed {seed})"
        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        preds[name], settings[name] = dummy.fit(x_train, y_train).predict(x), f"seed={seed}"
    for name, (model, setting) in final_models(task).items():
        print(f"[{task}] fitting {name} ({setting}) ...", flush=True)
        start = time.time()
        preds[name], settings[name] = model.fit(x_train, y_train).predict(x), setting
        print(f"[{task}]   done in {time.time() - start:.0f} s", flush=True)
    if use_knn:
        preds["LaBSE + kNN"], settings["LaBSE + kNN"] = knn_predict(task, train, held, split)
    return held, preds, settings


# ---------------------------------------------------------------- scores and uncertainty

def evaluate_task(task, split, use_knn, rng):
    held, preds, settings = predictions(task, split, use_knn)
    classes = sorted(held["label"].unique())
    code = {c: i for i, c in enumerate(classes)}
    y = held["label"].map(code).to_numpy()
    coded = {name: pd.Series(p).map(code).to_numpy() for name, p in preds.items()}

    # Bootstrap resamples (LD: whole source_id groups, as in M5); the same resamples for every model
    groups = load_groups(task, split)
    idx = bootstrap_indices(len(y), groups, rng, N_BOOT)
    boots = boot_scores(y, coded, idx, len(classes))

    rows = []
    for name, p in coded.items():
        f1_lo, f1_hi = ci(boots[name]["macro_f1"])
        acc_lo, acc_hi = ci(boots[name]["accuracy"])
        rows.append({"task": task, "split": split, "model": name, "setting": settings[name],
                     "macro_f1": macro_f1_from_codes(y[None], p[None], len(classes))[0],
                     "macro_f1_lo": f1_lo, "macro_f1_hi": f1_hi,
                     "accuracy": (y == p).mean(), "accuracy_lo": acc_lo, "accuracy_hi": acc_hi,
                     "resample_unit": "source_id" if groups is not None else "sentence",
                     "n_boot": N_BOOT})
    table = pd.DataFrame(rows)

    # Paired differences: best model (by point macro-F1) minus every other non-stratified model
    point = table.set_index("model")["macro_f1"]
    best = point.idxmax()
    paired = []
    for name in coded:
        if name == best or name.startswith("stratified"):
            continue
        diff = boots[best]["macro_f1"] - boots[name]["macro_f1"]
        lo, hi = ci(diff)
        paired.append({"task": task, "split": split, "best_model": best, "other_model": name,
                       "diff_macro_f1": point[best] - point[name], "diff_lo": lo, "diff_hi": hi,
                       "p_best_not_better": (diff <= 0).mean(), "n_boot": N_BOOT})

    # Per-sentence predictions, for the error analysis
    out = held.copy()
    for name, p in preds.items():
        if not name.startswith("stratified"):
            out[f"pred: {name}"] = p
    return table, pd.DataFrame(paired), out


def summary_table(full):
    """Main table: the three stratified seeds collapsed into mean ± SD, other models with their CI."""
    rows = []
    for task, t in full.groupby("task", sort=False):
        strat = t[t["model"].str.startswith("stratified")]
        for _, r in t.iterrows():
            if r["model"] == strat["model"].iloc[0]:  # put the stratified summary in its place
                rows.append({"task": task, "model": "stratified (mean of 3 seeds)",
                             "setting": "seeds 3052-3054", "macro_f1": strat["macro_f1"].mean(),
                             "macro_f1_sd": strat["macro_f1"].std(ddof=1),
                             "accuracy": strat["accuracy"].mean()})
            if r["model"].startswith("stratified"):
                continue
            rows.append({"task": task, "model": r["model"], "setting": r["setting"],
                         "macro_f1": r["macro_f1"], "macro_f1_lo": r["macro_f1_lo"],
                         "macro_f1_hi": r["macro_f1_hi"], "accuracy": r["accuracy"]})
    cols = ["task", "model", "setting", "macro_f1", "macro_f1_lo", "macro_f1_hi", "macro_f1_sd",
            "accuracy"]
    return pd.DataFrame(rows)[cols]


def plot(summary, split, out):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.4))
    for ax, task in zip(axes, TASKS):
        t = summary[summary["task"] == task].reset_index(drop=True)
        x = np.arange(len(t))
        lo = (t["macro_f1"] - t["macro_f1_lo"]).fillna(t["macro_f1_sd"])
        hi = (t["macro_f1_hi"] - t["macro_f1"]).fillna(t["macro_f1_sd"])
        ax.bar(x, t["macro_f1"], color="#9ecae1", edgecolor="#3182bd")
        ax.errorbar(x, t["macro_f1"], yerr=[lo, hi], fmt="none", ecolor="black", capsize=4)
        for xi, v, h in zip(x, t["macro_f1"], hi):
            ax.text(xi, v + h + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, t["model"].str.replace(" (mean of 3 seeds)", "", regex=False),
                      rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{SPLIT_NAMES[split]} macro-F1")
        ax.set_title(TASK_TITLES[task], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{SPLIT_NAMES[split]} macro-F1 with 95% bootstrap CI "
                 f"(stratified: mean ± SD over 3 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="evaluate ONCE on the locked test split (default: validation)")
    parser.add_argument("--knn", action="store_true",
                        help="include LaBSE + kNN (uses the embeddings/ cache, else encodes with LaBSE)")
    args = parser.parse_args()

    split = "test" if args.final else "valid"
    prefix = RESULTS / f"final_{split}"
    if args.final:
        if Path(f"{prefix}.csv").exists():
            sys.exit(f"{prefix}.csv already exists: the test split has been used. Not running again.")
        print("WARNING: evaluating on the TEST split. This is done once, for the final report.\n")

    rng = np.random.default_rng(SEED)
    tables, paired = [], []
    for task in TASKS:
        table, pair, preds = evaluate_task(task, split, args.knn, rng)
        tables.append(table)
        paired.append(pair)
        preds.to_csv(f"{prefix}_predictions_{task}.csv", index=False)

    full, paired = pd.concat(tables), pd.concat(paired)
    summary = summary_table(full)
    full.to_csv(f"{prefix}_all.csv", index=False, float_format="%.3f")
    paired.to_csv(f"{prefix}_paired.csv", index=False, float_format="%.3f")
    summary.to_csv(f"{prefix}.csv", index=False, float_format="%.3f")
    plot(summary, split, f"{prefix}.png")

    cols = ["task", "model", "macro_f1", "macro_f1_lo", "macro_f1_hi", "macro_f1_sd", "accuracy"]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print("\n" + paired.drop(columns=["split", "n_boot"]).round(3).to_string(index=False))
    print(f"\nSaved {prefix}.csv, {prefix}_all.csv, {prefix}_paired.csv, {prefix}.png "
          f"and {prefix}_predictions_*.csv")


if __name__ == "__main__":
    main()
