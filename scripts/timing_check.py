import time

import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


for task, path in [
    ("sentiment", "data/sentiment/train.jsonl"),
    ("ld", "data/ld/ld_train.jsonl"),
]:
    train = pd.read_json(path, lines=True)
    for name, clf in [
        ("NB", MultinomialNB()),
        ("LR", LogisticRegression(max_iter=1000)),
    ]:
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3)), clf
        )
        start = time.time()
        pipe.fit(train["sentence"], train["label"])
        ram = psutil.Process().memory_info().rss / 1e6
        print(f"{task} {name}: {time.time() - start:.1f} s, RAM {ram:.0f} MB")
