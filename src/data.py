"""Load the CantoNLU sentiment and language-detection splits.

The data files are not in the repository; see data/README.md for how to copy them.
"""
from pathlib import Path

import pandas as pd

# Repository root = the folder above src/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# File of each split, per task (official CantoNLU splits, used unchanged)
FILES = {
    "sentiment": {"train": "sentiment/train.jsonl", "valid": "sentiment/valid.jsonl", "test": "sentiment/test.jsonl"},
    "ld": {"train": "ld/ld_train.jsonl", "valid": "ld/ld_val.jsonl", "test": "ld/ld_test.jsonl"},
}

# Labels we expect in each task (see DATA_CARD.md)
LABELS = {"sentiment": {"smile", "ok", "cry"}, "ld": {0, 1, 2}}


def load_split(task, split):
    """Return one split as a DataFrame with the columns `sentence` and `label`."""
    path = DATA_DIR / FILES[task][split]
    df = pd.read_json(path, lines=True)

    # Basic checks: a wrong or corrupted file should fail loudly, not silently
    assert {"sentence", "label"} <= set(df.columns), f"missing columns in {path}"
    assert df["sentence"].notna().all(), f"empty sentence in {path}"
    assert set(df["label"]) <= LABELS[task], f"unexpected label in {path}"

    return df[["sentence", "label"]]


if __name__ == "__main__":
    # Quick check: python src/data.py  (train and valid only; test stays locked)
    for task in FILES:
        for split in ("train", "valid"):
            df = load_split(task, split)
            print(task, split, df.shape, sorted(df["label"].unique()))
