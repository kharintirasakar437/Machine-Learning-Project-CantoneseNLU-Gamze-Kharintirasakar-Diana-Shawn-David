# Evaluating Classical Machine Learning Baselines on Cantonese NLU

CSCI 3052U — Machine Learning I  
Fall 2026 — **Group 3, Team CantoneseNLU**

## Team Members

- Kharintirasakar Uthayanan
- David Zajac
- Diana Riyahi
- Shawn Xiao
- Gamze Esen-Erdemir

## Project Overview

Cantonese remains relatively under-resourced in natural language processing compared with higher-resource languages. The [CantoNLU benchmark](https://arxiv.org/abs/2510.20670) evaluates Cantonese natural language understanding across seven tasks using pretrained language models.

This project investigates how well simpler and more traditional machine learning approaches perform on selected CantoNLU tasks. Our main goal is to establish clear and reproducible naive and classical baselines, then compare them with published pretrained-model results where a fair comparison is possible.

The project currently focuses on:

- **Sentiment analysis**
- **Language detection**

The goal is not necessarily to outperform pretrained language models. Instead, we want to determine how much performance can be achieved using simpler methods and where pretrained representations provide a clear advantage.

## Research Questions

- **RQ1.** How close do naive and classical feature-based models come to published CantoNLU pretrained-model results on the selected tasks?
- **RQ2.** Does a fixed multilingual embedding representation improve performance over character-level TF-IDF without requiring full fine-tuning?
- **RQ3.** How does performance change with training-set size, and what types of examples are most often misclassified?

## Planned Models

| Tier | Model | Status |
|---|---|---|
| Naive | Majority-class prediction | committed |
| Naive | Stratified-random prediction | committed |
| Classical | Character n-gram (1-3) TF-IDF + Multinomial Naive Bayes | committed |
| Classical | Character n-gram (1-3) TF-IDF + L2 logistic regression | committed |
| Embedding | kNN on fixed multilingual sentence embeddings (LaBSE, inference only) | committed |
| Complex | Gradient boosting or MLP | Optional, only if feasability is confirmed |
| Stretch | Pilot zero-/few-shot evaluation of one generative LLM on one task | Stretch goal |

## Experimental Protocol

These rules come from the M1 proposal and are not negotiable per-experiment:

- The **training split** is used to fit models.
- The **validation split** is used for model selection and hyperparameter tuning.
- **The test split is used once**, for the final reported evaluation. All model selection and hyperparameter tuning happen on the **validation** split.
- TF-IDF vectorizers, scalers, and other fitted preprocessing tools will be fitted using training data only.
- The proposed primary metric is **macro-F1**.
- Secondary measures include:
  - accuracy
  - per-class precision
  - per-class recall
  - confusion matrices
- Where randomness affects results, multiple runs or seeds will be used and variability will be reported.
- At least one sensitivity, robustness, or ablation analysis will be performed.
- A likely sensitivity analysis is a learning curve using different percentages of the training data.
- Reported results should be reproducible from scripts and files contained in this repository.

## Data

The project currently uses data from the CantoNLU repository:

https://github.com/Aatlantise/canto-nlu

Raw dataset files are stored locally and are **not committed to this repository**.

See:

- [`data/README.md`](data/README.md)
- [`DATA_CARD.md`](DATA_CARD.md)

for dataset organization, provenance, known issues, and planned treatment.

### Sentiment Analysis

The sentiment-analysis task uses OpenRice restaurant reviews.

Local files:

    data/sentiment/train.jsonl
    data/sentiment/valid.jsonl
    data/sentiment/test.jsonl

Current split sizes:

| Split | Rows |
|---|---:|
| Train | 9,999 |
| Validation | 999 |
| Test | 999 |

Labels:

- `smile`
- `ok`
- `cry`

The three classes are balanced in each split.

Initial inspection found no duplicate examples within the sentiment splits and no overlap between the train, validation, and test sets.

### Language Detection

The language-detection dataset currently contains three classes:

- `0` = Cantonese
- `1` = Mandarin
- `2` = code-mixed / corrupted sentence

Local files:

    data/ld/ld_train.jsonl
    data/ld/ld_val.jsonl
    data/ld/ld_test.jsonl

Current split sizes:

| Split | Rows |
|---|---:|
| Train | 42,753 |
| Validation | 2,425 |
| Test | 2,341 |

Our original proposal described this task as Cantonese vs. Mandarin. During Milestone 2, the team will decide whether to:

1. keep all three classes, or
2. restrict the task to Cantonese and Mandarin only.

Initial inspection identified possible data-quality issues, including:

- identical sentences with different labels
- a small amount of overlap between splits
- possible label noise in code-mixed examples
- inconsistencies between some documentation and dataset-generation code

These issues will be independently verified before any cleaning decisions are made.

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   ├── sentiment/              # local only, ignored by Git
│   └── ld/                     # local only, ignored by Git
├── notebooks/                  # exploratory analysis and later experiments
├── src/                        # training and evaluation scripts
├── results/                    # generated tables and figures
├── AI_USE.md                   # AI assistance log
├── CONTRIBUTIONS.md            # team contribution record
├── DATA_CARD.md                # dataset documentation
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
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

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Getting the Data

The raw data is not redistributed in this repository.

Clone the upstream CantoNLU repository separately:

```bash
git clone https://github.com/Aatlantise/canto-nlu.git
```

The relevant files can then be copied locally into this project's `data/` directory.

Expected local structure:

```text
data/
├── sentiment/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── ld/
    ├── ld_train.jsonl
    ├── ld_val.jsonl
    └── ld_test.jsonl
```

The raw data files are excluded from Git using `.gitignore`.

Exact source attribution and licensing details are being verified as part of Milestone 2 and will be recorded in `DATA_CARD.md`.

## Running the Baselines

The baseline training interface will be added during Milestone 3.

Planned usage:

```bash
# Train and evaluate naive and classical baselines on the validation split
python src/train_baselines.py
```

Final evaluation will use the locked test split only after model selection is complete:

```bash
python src/train_baselines.py --final
```

The exact command structure may change as the implementation is developed.

## Results

Baseline results will be added during Milestone 3 together with the scripts required to reproduce them.

No result should be treated as final unless it can be regenerated from code in this repository.

## Current Status

Milestone 1 has been completed and submitted.

Current work is focused on **Milestone 2: Dataset and Problem Approval**.

Current tasks include:

- verifying dataset sources and licenses
- examining dataset structure and quality
- performing initial exploratory data analysis
- confirming the final language-detection task definition
- documenting train/validation/test splits
- confirming the evaluation plan
- documenting known dataset limitations
- confirming compute feasibility

## Milestones

| Milestone | Deliverable | Due |
|---|---|---|
| M1 | Proposal, team contract, repository setup, topic declaration | Sept. 18, 2026 |
| M2 | Dataset/data card, EDA, split and metric plan, approval | Oct. 2, 2026 |
| M3 | Pipeline, naive and classical baselines, environment specification, baseline table | Oct. 19, 2026 |
| M4 | Progress presentation, theory, preliminary results, revised plan | Oct. 30, 2026 |
| M5 | Final model comparison, tuning, robustness/sensitivity analysis, error analysis | Nov. 20, 2026 |
| M6 | Final report, reproducible repository, presentation/demo | Dec. 4, 2026 |
| M7 | Individual reflection and oral/code defence | Dec. 7, 2026 |

## Working Practices

- Substantial changes should go through pull requests and be reviewed by at least one other group member.
- Milestone submissions should be reviewed by the full group.
- No API keys, passwords, or credentials should be committed to the repository.
- Secrets should be stored in a Git-ignored `.env` file if needed.
- AI assistance is recorded in [`AI_USE.md`](AI_USE.md).
- Team contributions are recorded in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).
- Raw datasets should not be manually edited.
- Any cleaning or filtering should be performed through reproducible code.

## Reproducibility

The project will maintain:

- fixed train/validation/test splits
- documented random seeds
- reproducible preprocessing
- recorded package dependencies
- experiment logs
- contribution records
- AI-use records

Any reported result should be reproducible from the files and scripts contained in this repository.

## Scope Limits

A third CantoNLU task will only be added if sentiment analysis and language detection are completed successfully.

Human-performance measurement is outside the scope of this CSCI 3052U course project.

A small pilot evaluation of one generative LLM on one task remains a stretch goal and does not affect the project's core success criteria.

## References

- Min, J., Ng, Y. H., Chan, S., Zhao, H. S., & Lee, E.-S. A. (2025). *CantoNLU: A benchmark for Cantonese natural language understanding.* arXiv:2510.20670.
- Feng, F., Yang, Y., Cer, D., Arivazhagan, N., & Wang, W. (2022). *Language-agnostic BERT sentence embedding.* ACL 2022, 878–891.
- Lee, J. L., Chen, L., Lam, C., Lau, C. M., & Tsui, T.-H. (2022). *PyCantonese: Cantonese linguistics and NLP in Python.* LREC 2022.
- Zhang, Z., Ye, Q., Zhang, Z., & Li, Y. (2011). *Sentiment classification of internet restaurant reviews written in Cantonese.* Expert Systems with Applications, 38(6), 7674–7682.
- Xiang, R., et al. (2025). *Cantonese natural language processing in the transformers era: A survey and current challenges.* Language Resources and Evaluation, 59(2), 1747–1773.
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
