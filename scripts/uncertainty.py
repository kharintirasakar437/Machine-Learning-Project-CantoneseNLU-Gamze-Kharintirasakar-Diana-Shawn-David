"""Bootstrap confidence intervals for validation macro-F1 (validation split only).

Each model is fitted once on the train split with the settings chosen by the tuning scripts,
then its validation predictions are resampled with replacement B = 1000 times. The same
resamples are used for every model, so we also get *paired* intervals for the difference
between the best model and each other model (a fair comparison on identical samples).

Resampling unit:
  - sentiment: one review (the 999 validation reviews are independent).
  - LD: one `source_id` group (cluster bootstrap). The LD validation split has 2425 sentences
    from only 455 source sentences (a Cantonese, a Mandarin and several code-mixed versions
    of the same text), so the sentences are not independent and a sentence-level bootstrap
    could give intervals that are too narrow. The sentence-level interval is also written
    to the CSV (columns *_sentence) for comparison.

Model settings (read from the tuning outputs, the same way as scripts/main_results.py):
  - NB:  best row of results/tune_nb_ngram.csv
  - LR:  results/best_lr.csv; if that file is not there yet, the untuned M3 default
         (C = 1, char 1-3-grams) is used and a warning is printed
  - kNN: best k in results/tune_knn_labse.csv; only run when the cached LaBSE embeddings
         exist in embeddings/ (run scripts/knn_labse.py first), or with --knn
Majority is the naive reference.

The test split is NOT used. Run from the repository root:
    python scripts/uncertainty.py            # B = 1000, seed 3052
    python scripts/uncertainty.py --knn      # also encode with LaBSE if the cache is missing
Writes results/uncertainty_ci.csv, results/uncertainty_paired.csv, results/uncertainty_ci.png.
"""

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import DATA_DIR, FILES, ROOT, load_split  # noqa: E402
from train_baselines import SEED  # noqa: E402

TASKS = ("sentiment", "ld")
RESULTS = ROOT / "results"
N_BOOT = 1000
ALPHA = 0.05  # 95% percentile intervals
TASK_TITLES = {"sentiment": "Sentiment", "ld": "Language detection"}


# ---------------------------------------------------------------- tuned model settings

def best_row(csv_name, task):
    """Highest validation macro-F1 row of a tuning CSV for one task (None if file missing)."""
    path = RESULTS / csv_name
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype={"alpha": str})
    table = table[table["task"] == task]
    return table.loc[table["macro_f1"].idxmax()]


def nb_setting(task):
    row = best_row("tune_nb_ngram.csv", task)
    return ast.literal_eval(row["ngram_range"]), float(row["alpha"])


def lr_setting(task):
    """Tuned (C, ngram_range) from results/best_lr.csv, or the M3 default if not tuned yet."""
    row = best_row("best_lr.csv", task)
    if row is None:
        print(f"WARNING: results/best_lr.csv not found; using the untuned LR default "
              f"(C=1, ngram=(1, 3)) for {task}. Re-run once scripts/tune_lr.py has run.")
        return 1.0, (1, 3), False
    return float(row["C"]), ast.literal_eval(row["ngram_range"]), True


def nb_pipeline(ngram_range, alpha):
    # Vectoriser inside the Pipeline: fitted on the (sub)train split only
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=ngram_range),
                         MultinomialNB(alpha=alpha))


def lr_pipeline(c, ngram_range):
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=ngram_range),
                         LogisticRegression(C=c, max_iter=1000))


def tuned_models(task):
    """{display name: (pipeline, setting string)} for the TF-IDF models, tuned settings."""
    ngram, alpha = nb_setting(task)
    c, lr_ngram, tuned = lr_setting(task)
    lr_name = "tfidf + LR (tuned)" if tuned else "tfidf + LR (default)"
    return {
        "tfidf + NB (tuned)": (nb_pipeline(ngram, alpha), f"alpha={alpha:g}, ngram={ngram}"),
        lr_name: (lr_pipeline(c, lr_ngram), f"C={c:g}, ngram={lr_ngram}"),
    }


# ---------------------------------------------------------------- kNN on cached LaBSE

def labse_cache(task, split):
    return ROOT / "embeddings" / f"labse_{task}_{split}.npy"


def knn_predictions(task, train, valid, allow_encode):
    """Validation predictions of kNN on LaBSE, or None if embeddings are unavailable."""
    row = best_row("tune_knn_labse.csv", task)
    if row is None:
        return None
    cached = all(labse_cache(task, s).exists() for s in ("train", "valid"))
    if not cached and not allow_encode:
        print(f"Skipping LaBSE + kNN for {task}: no cached embeddings in embeddings/ "
              f"(run scripts/knn_labse.py once, or pass --knn).")
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from main_results import knn_embeddings  # reuses the same cache as knn_labse.py
    k = int(row["k"])
    x_train = knn_embeddings(task, "train", train["sentence"].tolist())
    x_valid = knn_embeddings(task, "valid", valid["sentence"].tolist())
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(x_train, train["label"])
    return model.predict(x_valid), f"k={k}"


# ---------------------------------------------------------------- fast metrics

def macro_f1_from_codes(y, p, n_classes):
    """Macro-F1 for integer-coded labels; vectorised over a batch of resamples.

    y, p: arrays of shape (n_boot, n). Classes absent from both y and p in a resample
    get F1 = 0, the same as sklearn's default (zero_division=0 with labels fixed).
    """
    f1s = []
    for c in range(n_classes):
        tp = ((y == c) & (p == c)).sum(axis=1)
        fp = ((y != c) & (p == c)).sum(axis=1)
        fn = ((y == c) & (p != c)).sum(axis=1)
        denom = 2 * tp + fp + fn
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(f1s, axis=0)


def bootstrap_indices(n, groups, rng, n_boot):
    """n_boot resamples of row indices; whole groups are resampled if groups is given."""
    if groups is None:
        return rng.integers(0, n, size=(n_boot, n))
    group_ids, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(group_ids))]
    picks = rng.integers(0, len(group_ids), size=(n_boot, len(group_ids)))
    # Groups differ in size, so resamples differ in length: return a list of index arrays
    return [np.concatenate([members[g] for g in row]) for row in picks]


def boot_scores(y, preds, idx, n_classes):
    """Bootstrap macro-F1 and accuracy for every model on the same resamples."""
    out = {}
    for name, p in preds.items():
        if isinstance(idx, np.ndarray):
            yb, pb = y[idx], p[idx]
            f1 = macro_f1_from_codes(yb, pb, n_classes)
            acc = (yb == pb).mean(axis=1)
        else:  # list of variable-length resamples (cluster bootstrap)
            f1 = np.array([macro_f1_from_codes(y[i][None], p[i][None], n_classes)[0] for i in idx])
            acc = np.array([(y[i] == p[i]).mean() for i in idx])
        out[name] = {"macro_f1": f1, "accuracy": acc}
    return out


def ci(values):
    lo, hi = np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2])
    return lo, hi


# ---------------------------------------------------------------- main

def load_groups(task, split):
    """source_id per row for LD (load_split keeps only sentence and label)."""
    if task != "ld":
        return None
    df = pd.read_json(DATA_DIR / FILES[task][split], lines=True)
    return df["source_id"].to_numpy()


def run_task(task, rng, n_boot, allow_encode):
    train, valid = load_split(task, "train"), load_split(task, "valid")
    classes = sorted(train["label"].unique())
    code = {c: i for i, c in enumerate(classes)}
    y = valid["label"].map(code).to_numpy()

    preds, settings = {}, {}
    majority = DummyClassifier(strategy="most_frequent").fit(train["sentence"], train["label"])
    preds["majority"] = majority.predict(valid["sentence"])
    settings["majority"] = ""
    for name, (model, setting) in tuned_models(task).items():
        print(f"[{task}] fitting {name} ({setting}) ...", flush=True)
        model.fit(train["sentence"], train["label"])
        preds[name], settings[name] = model.predict(valid["sentence"]), setting
    knn = knn_predictions(task, train, valid, allow_encode)
    if knn is not None:
        preds["LaBSE + kNN"], settings["LaBSE + kNN"] = knn
    preds = {k: pd.Series(v).map(code).to_numpy() for k, v in preds.items()}

    n = len(y)
    groups = load_groups(task, "valid")
    idx_sent = bootstrap_indices(n, None, rng, n_boot)
    boots = {"sentence": boot_scores(y, preds, idx_sent, len(classes))}
    if groups is not None:
        boots["source_id"] = boot_scores(y, preds, bootstrap_indices(n, groups, rng, n_boot),
                                         len(classes))
    primary = "source_id" if groups is not None else "sentence"

    point = {name: {"macro_f1": macro_f1_from_codes(y[None], p[None], len(classes))[0],
                    "accuracy": (y == p).mean()} for name, p in preds.items()}

    ci_rows = []
    for name in preds:
        row = {"task": task, "model": name, "setting": settings[name],
               "macro_f1": point[name]["macro_f1"]}
        for unit, b in boots.items():
            lo, hi = ci(b[name]["macro_f1"])
            suffix = "" if unit == primary else f"_{unit}"
            row[f"macro_f1_lo{suffix}"], row[f"macro_f1_hi{suffix}"] = lo, hi
            row[f"macro_f1_se{suffix}"] = b[name]["macro_f1"].std(ddof=1)
        row["accuracy"] = point[name]["accuracy"]
        row["accuracy_lo"], row["accuracy_hi"] = ci(boots[primary][name]["accuracy"])
        row["resample_unit"], row["n_boot"] = primary, n_boot
        ci_rows.append(row)

    # Paired differences: best model (by point macro-F1) minus each other model
    best = max(point, key=lambda m: point[m]["macro_f1"])
    paired_rows = []
    for name in preds:
        if name == best:
            continue
        diff = boots[primary][best]["macro_f1"] - boots[primary][name]["macro_f1"]
        lo, hi = ci(diff)
        paired_rows.append({
            "task": task, "best_model": best, "other_model": name,
            "diff_macro_f1": point[best]["macro_f1"] - point[name]["macro_f1"],
            "diff_lo": lo, "diff_hi": hi,
            "p_best_not_better": (diff <= 0).mean(),  # share of resamples where best does not win
            "resample_unit": primary, "n_boot": n_boot,
        })
    return ci_rows, paired_rows


def plot(table, out):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.2), sharex=False)
    for ax, task in zip(axes, TASKS):
        t = table[table["task"] == task].reset_index(drop=True)
        ypos = np.arange(len(t))[::-1]
        err = np.vstack([t["macro_f1"] - t["macro_f1_lo"], t["macro_f1_hi"] - t["macro_f1"]])
        ax.errorbar(t["macro_f1"], ypos, xerr=err, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                    capsize=4)
        for yp, (_, r) in zip(ypos, t.iterrows()):
            ax.annotate(f"{r.macro_f1:.3f} [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}]",
                        (r.macro_f1_hi, yp), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.set_yticks(ypos, t["model"])
        ax.set_xlim(0, 1.25)
        ax.set_ylim(-0.5, len(t) - 0.5)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Validation macro-F1 (dot = point estimate, bar = 95% CI)")
        unit = "review" if t["resample_unit"].iloc[0] == "sentence" else "source_id group"
        ax.set_title(f"{TASK_TITLES[task]}\nresampled by {unit}, B = {t['n_boot'].iloc[0]}",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Validation macro-F1 with 95% bootstrap confidence intervals", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--knn", action="store_true",
                        help="encode with LaBSE if the embedding cache is missing (slow)")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    ci_rows, paired_rows = [], []
    for task in TASKS:
        c, p = run_task(task, rng, args.n_boot, args.knn)
        ci_rows += c
        paired_rows += p

    table, paired = pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)
    out_ci, out_pair = RESULTS / "uncertainty_ci.csv", RESULTS / "uncertainty_paired.csv"
    out_png = RESULTS / "uncertainty_ci.png"
    table.to_csv(out_ci, index=False, float_format="%.3f")
    paired.to_csv(out_pair, index=False, float_format="%.3f")
    plot(table, out_png)

    cols = ["task", "model", "macro_f1", "macro_f1_lo", "macro_f1_hi", "macro_f1_se"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\n" + paired.drop(columns=["n_boot"]).round(3).to_string(index=False))
    print(f"\nSaved {out_ci}\nSaved {out_pair}\nSaved {out_png}")


if __name__ == "__main__":
    main()
