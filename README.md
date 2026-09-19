# Evaluating Classical Machine Learning Baselines on Cantonese NLU

CSCI 3052U — Machine Learning I, Fall 2026 · **Group 3, Team CantoneseNLU**
Kharintirasakar Uthayanan · David Zajac · Diana Riyahi · Shawn Xiao · Gamze Esen-Erdemir

Cantonese is under-resourced compared with higher-resource languages, and the
[CantoNLU benchmark](https://arxiv.org/abs/2510.20670) evaluates it across seven NLU tasks using
pretrained language models. This project adds the layer that benchmark is missing: **reproducible
naive and classical baselines**. We work on **sentiment analysis** (openrice-senti) and **language
detection** (Cantonese vs. Mandarin), with a third task added only if those two finish. Published
CantoNLU transformer scores are quoted as reference context — we never re-train them.

The goal is not to beat pretrained models. It is to establish a clear reference point and quantify
how much performance actually comes from large-scale pretrained representations.

## Research questions

- **RQ1.** How close do naive and classical feature-based models come to published CantoNLU
  pretrained-model results on the selected tasks?
- **RQ2.** Does a fixed multilingual embedding representation improve over character-level TF-IDF
  without any fine-tuning?
- **RQ3.** How does performance change with training-set size, and which examples are most often
  misclassified?

## Models

| Tier | Model | Status |
|---|---|---|
| Naive | Majority-class, stratified-random | committed |
| Classical | Character n-gram (1–3) TF-IDF + Multinomial Naive Bayes | committed |
| Classical | Character n-gram (1–3) TF-IDF + L2 logistic regression | committed |
| Embedding | kNN on frozen multilingual sentence embeddings (LaBSE, inference only) | committed |
| Complex | Gradient boosting, MLP | optional, only if feasibility is confirmed |
| Stretch | Pilot zero-/few-shot evaluation of one generative LLM on one task | stretch goal |

## Experimental protocol

These rules come from the M1 proposal and are not negotiable per-experiment:

- **The test split is used once**, for the final reported evaluation. All model selection and
  hyperparameter tuning happen on the **validation** split.
- **Fixed seed 3052** everywhere; at least three seeds where results are stochastic, with
  variability reported.
- Vectorisers and scalers are fitted on **training data only**, inside an sklearn `Pipeline`.
- **Primary metric: macro-F1** — chosen for cross-task consistency and per-class visibility, *not*
  for class imbalance (the sentiment set is deliberately balanced, so accuracy is reported
  alongside). Secondary: accuracy, per-class precision/recall, confusion matrices.
- One sensitivity analysis is required: a learning curve at 10 / 25 / 50 / 100% of the training set.
- Every number in a report, slide or table must regenerate from a script in this repository.

## Repository layout

```
.
├── data/                # datasets - NOT committed, see data/README.md
├── src/                 # pipeline and training scripts (lands at M3)
├── requirements.txt     # Python dependencies
├── CONTRIBUTIONS.md     # who did what, per milestone
└── AI_USE.md            # AI assistance log, kept at time of use
```

## Setup

Python 3.11+ (developed on 3.13).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Benchmark data is **not redistributed in this repository**. The upstream repository ships no LICENSE
file, so we keep the data local, request written course-use permission from the authors, and record
the outcome in the data card. Fetch it yourself:

```bash
git clone https://github.com/Aatlantise/canto-nlu.git
cp canto-nlu/openrice-senti/*.tsv data/
```

Sentiment (openrice-senti) ships official splits: **train 9,999 / valid 999 / test 999**, TSV with
header `label\ttext_a`, three deliberately balanced classes (`smile` / `ok` / `cry`).

Language detection has **no ready-made split** upstream — it is derived from public sources
(Cantonese Wikipedia, `raptorkwok/cantonese_sentences`, Wu Wikipedia). We are requesting the
authors' exact split; if it is unavailable we will construct a documented stratified split and
submit it for instructor approval at M2.

See [data/README.md](data/README.md) for details.

## Running the baselines

Planned interface, landing at M3 (internal deadline Oct 13):

```bash
# naive + classical baselines, evaluated on the validation split
python src/train_baselines.py

# final evaluation on the locked test split - run this ONCE, for the report
python src/train_baselines.py --final
```

`--final` will print a warning by design: it touches the locked test split.

## Results

Baseline results land at M3, together with the script that regenerates them. Nothing is reported
here until it can be reproduced from a command in this repository.

## Milestones

| | Deliverable | Due |
|---|---|---|
| M1 | Proposal, team contract, repository setup, topic declaration | Sept 18, 2026 |
| M2 | Data cards, licences, EDA, split and metric plan, approval | Oct 2, 2026 |
| M3 | Pipeline, naive + NB/LR baselines, environment spec, baseline table | Oct 19, 2026 |
| M4 | Progress presentation, theory, sentiment results, pilot LLM run (stretch) | Oct 30, 2026 |
| M5 | Full comparison (≥3 seeds), ablations, learning curves, error analysis | Nov 20, 2026 |
| M6 | 6–8 page report, repository release regenerating the main table | Dec 4, 2026 |
| M7 | Individual reflection and oral defence | Dec 7, 2026 |

Every deliverable is finished at least 48 hours before its official milestone date. Team meets
Mondays 20:00–21:00; day-to-day communication on Discord.

## Working practices

- Substantial changes go through a pull request reviewed by at least one other member; milestone
  changes are reviewed by the whole team.
- No API keys or credentials in the repository — secrets live in a git-ignored `.env`.
- AI assistance is logged in [AI_USE.md](AI_USE.md) at the time of use. Generative models studied as
  an *object of the experiment* are logged separately from AI assistance.
- Contributions are recorded in [CONTRIBUTIONS.md](CONTRIBUTIONS.md) as the work happens.

## Scope limits

A third task is added only if sentiment and language detection are completed successfully.
Human-performance measurement is out of scope — it belongs to the Lee Language Lab's parallel
annotation effort. The pilot LLM evaluation is a stretch goal and does not affect the success
criterion.

## References

- Min, J., Ng, Y. H., Chan, S., Zhao, H. S., & Lee, E.-S. A. (2025). *CantoNLU: A benchmark for
  Cantonese natural language understanding.* arXiv:2510.20670.
- Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). *Language-agnostic BERT sentence
  embedding.* ACL 2022, 878–891.
- Lee, J. L., Chen, L., Lam, C., Lau, C. M., & Tsui, T.-H. (2022). *PyCantonese: Cantonese
  linguistics and NLP in Python.* LREC 2022.
- Zhang, Z., Ye, Q., Zhang, Z., & Li, Y. (2011). *Sentiment classification of internet restaurant
  reviews written in Cantonese.* Expert Systems with Applications, 38(6), 7674–7682.
- Xiang, R., et al. (2025). *Cantonese natural language processing in the transformers era: A survey
  and current challenges.* Language Resources and Evaluation, 59(2), 1747–1773.
