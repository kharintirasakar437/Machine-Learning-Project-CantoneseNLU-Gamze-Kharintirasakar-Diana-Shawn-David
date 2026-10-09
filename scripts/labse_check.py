import time

import pandas as pd
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/LaBSE")
for task, path in [
    ("sentiment", "data/sentiment/train.jsonl"),
    ("ld", "data/ld/ld_train.jsonl"),
]:
    sample = (
        pd.read_json(path, lines=True)["sentence"]
        .sample(500, random_state=3052)
        .tolist()
    )
    start = time.time()
    model.encode(sample, batch_size=32)
    print(f"{task}: {time.time() - start:.1f} s for 500 sentences")
