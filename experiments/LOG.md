# Experiment Log — Milestone 5

One entry per experiment, newest last. All experiments use the **validation split only**;
the test split stays locked until the final report (`--final`).

Shared settings unless an entry says otherwise:
- Data: official CantoNLU splits, upstream commit `2283874`
- Features: TF-IDF of character 1–3-grams, fitted on train only (scikit-learn Pipeline)
- Seeds: 3052 (and 3053, 3054 where results are random)
- Primary metric: macro-F1; accuracy also reported

## Entry template

### YYYY-MM-DD — <short name> (<member>)
- **Goal:** what question this experiment answers
- **Script / command:** `python scripts/....py`
- **Task(s):** sentiment / LD
- **Model and settings:** model, fixed hyperparameters
- **Search space (if tuning):** values tried, e.g. α ∈ {0.01, 0.03, 0.1, 0.3, 1}
- **Selection rule:** e.g. highest validation macro-F1
- **Result:** best setting and its validation scores; output file in `results/`
- **Observation:** one or two sentences — what we learned, anything surprising

## Entries

### 2026-10-09 — Default baselines (Gamze Esen-Erdemir)
- **Goal:** first naive and classical baselines (Milestone 3)
- **Script / command:** `python src/train_baselines.py`
- **Task(s):** sentiment, LD
- **Model and settings:** majority; stratified (3 seeds); TF-IDF + MultinomialNB (α = 1); TF-IDF + LogisticRegression (C = 1, max_iter = 1000)
- **Search space:** none (defaults)
- **Result:** best = LR, validation macro-F1 0.751 (sentiment), 0.599 (LD); `results/baseline_valid.csv`
- **Observation:** NB predicts "code-mixed" for every LD sentence; removing the class prior changes only 11 predictions (`scripts/nb_ld_check.py`).

### 2026-10-29 — Train vs. validation diagnostics (Gamze Esen-Erdemir)
- **Goal:** check over- and underfitting of the default models (Milestone 4)
- **Script / command:** `python scripts/m4_report_extras.py`
- **Task(s):** sentiment, LD
- **Model and settings:** TF-IDF + NB / LR, defaults as above
- **Result:** LR train/valid macro-F1 0.953 / 0.751 (sentiment), 0.623 / 0.599 (LD); `results/m4_*.csv`, `results/m4_confusion_lr_valid.png`
- **Observation:** sentiment overfits (large train–validation gap); LD underfits (low score even on train). Most sentiment errors involve "ok"; in LD, Cantonese and Mandarin are mostly predicted as code-mixed.
