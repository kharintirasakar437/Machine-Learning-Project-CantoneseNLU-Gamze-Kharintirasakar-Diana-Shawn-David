"""Train the baselines and evaluate them on the validation split.

Usage (from the repository root):  python src/train_baselines.py
"""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

from data import ROOT, load_split


SEED = 3052
SEEDS = [SEED, SEED + 1, SEED + 2]  # for models with randomness

def tfidf():
    """Character n-grams of length 1-3, so no Cantonese word segmentation is needed."""
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 3))


def evaluate(model, train, valid):
    """Fit on train, predict valid, return the metrics."""
    model.fit(train["sentence"], train["label"])
    pred = model.predict(valid["sentence"])
    return {
        "accuracy": round(accuracy_score(valid["label"], pred), 3),
        "macro_f1": round(f1_score(valid["label"], pred, average="macro"), 3),
        "micro_f1": round(f1_score(valid["label"], pred, average="micro"), 3),
        "weighted_f1": round(f1_score(valid["label"], pred, average="weighted"), 3),
    }



if __name__ == "__main__":
    rows = []
    for task in ("sentiment", "ld"):
        train = load_split(task, "train")
        valid = load_split(task, "valid")

        models = {"majority": DummyClassifier(strategy="most_frequent")}
        for seed in SEEDS:
            models[f"stratified (seed {seed})"] = DummyClassifier(strategy="stratified", random_state=seed)
        # Course-level baselines (default hyperparameters; tuning on valid comes later)
        models["tfidf + NB"] = make_pipeline(tfidf(), MultinomialNB())
        models["tfidf + LR"] = make_pipeline(tfidf(), LogisticRegression(max_iter=1000))

        for name, model in models.items():
            result = {"task": task, "model": name, **evaluate(model, train, valid)}
            print(result)
            rows.append(result)

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "baseline_valid.csv"
    out.parent.mkdir(exist_ok=True)  # create results/ if it does not exist
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")
