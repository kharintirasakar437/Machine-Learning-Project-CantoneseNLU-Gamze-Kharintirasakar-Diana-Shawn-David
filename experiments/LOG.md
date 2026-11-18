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

### 2026-11-13 — k-NN on frozen LaBSE embeddings (Shawn Xiao)
- **Goal:** tune k for k-NN classification on frozen LaBSE sentence embeddings (Milestone 5)
- **Script / command:** `python scripts/knn_labse.py`
- **Task(s):** sentiment, LD
- **Model and settings:** LaBSE embeddings (frozen, L2-normalized, cached once per task/split under `embeddings/`), k-NN with cosine metric. No vectoriser to fit inside a Pipeline here: LaBSE is a fixed pretrained extractor, not trained on our train split, so there is no TF-IDF-style leakage risk.
- **Search space:** k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31, 41}
- **Selection rule:** highest validation macro-F1
- **Result:** sentiment best k=41, macro-F1 0.644, accuracy 0.643; LD best k=1, macro-F1 0.382, accuracy 0.506; `results/tune_knn_labse.csv`, `results/tune_knn_labse.png`
- **Observation:** on LD, accuracy keeps rising with k (up to 0.672 at k=41) while macro-F1 falls — larger k collapses predictions toward the majority class, the same effect Gamze saw with NB on LD. On sentiment, kNN-on-LaBSE (0.644) still trails TF-IDF+LR (0.751).

### 2026-11-17 — Bootstrap confidence intervals for macro-F1 (Diana Riyahi)
- **Goal:** how much of the validation macro-F1 differences between models is real rather than sampling noise
- **Script / command:** `python scripts/uncertainty.py`
- **Task(s):** sentiment, LD
- **Model and settings:** majority; TF-IDF + NB with tuned settings (best row of `results/tune_nb_ngram.csv`: α = 0.01 sentiment, α = 0.003 LD, char 1–3-grams); TF-IDF + LR with the untuned default (C = 1, char 1–3-grams) because `results/best_lr.csv` is not on `main` yet (the script reads it automatically once it is). LaBSE + kNN is skipped unless the embedding cache exists. Each model is fitted once on train; validation predictions are resampled B = 1000 times (seed 3052), the same resamples for every model, 95% percentile intervals. Sentiment resamples reviews; LD resamples whole `source_id` groups (455 groups for 2425 sentences), since the variants of one source sentence are not independent.
- **Search space:** none (uses the settings chosen by the tuning scripts)
- **Result:** sentiment: NB 0.749 [0.724, 0.776], LR 0.751 [0.725, 0.778]; paired LR − NB = +0.003 [−0.020, +0.025]. LD: NB 0.646 [0.621, 0.669], LR 0.599 [0.571, 0.626]; paired NB − LR = +0.047 [+0.020, +0.075]. Majority: 0.167 (sentiment), 0.268 (LD). `results/uncertainty_ci.csv`, `results/uncertainty_paired.csv`, `results/uncertainty_ci.png`
- **Observation:** on sentiment, NB and LR cannot be told apart (the paired interval contains 0 and LR wins only 60% of resamples), while on LD tuned NB beats default LR in every resample. The intervals are about ±0.025, so differences under ~0.03 between single validation scores should not be reported as improvements. For LD, the group-level and sentence-level bootstraps gave almost the same width (SE 0.012 vs. 0.012 for NB).

### 2026-11-17 — Learning curves (Diana Riyahi)
- **Goal:** sensitivity of each model to the amount of training data; would more data help?
- **Script / command:** `python scripts/learning_curves.py`
- **Task(s):** sentiment, LD
- **Model and settings:** same NB and LR settings as the bootstrap entry above. Train on a label-stratified random 10%, 25%, 50% of the train split (seeds 3052, 3053, 3054) and on 100% (one fit: same data for every seed, deterministic models); score on the full validation split. Train macro-F1 is measured on the subset used for fitting.
- **Search space:** training fraction ∈ {10%, 25%, 50%, 100%}
- **Result:** validation macro-F1 (mean ± std over seeds) at 10 / 25 / 50 / 100%: sentiment LR 0.673±0.012 / 0.717±0.014 / 0.736±0.009 / 0.751, NB 0.691±0.013 / 0.721±0.002 / 0.738±0.002 / 0.749; LD LR 0.299±0.005 / 0.377±0.019 / 0.469±0.003 / 0.599, NB 0.436±0.012 / 0.477±0.010 / 0.518±0.007 / 0.646. `results/learning_curves.csv`, `results/learning_curves_summary.csv`, `results/learning_curves.png`
- **Observation:** sentiment is flattening (+0.015 from 50% to 100%) with a train–validation gap of 0.20–0.25, so more data of the same kind would help only a little and the models mainly overfit; LD is still rising steeply (+0.13 from 50% to 100% for both models), so LD is data-limited, and LR on LD barely fits its own training data (train 0.62), which matches the underfitting found in M4. Seed-to-seed spread (≤ 0.02) is smaller than the bootstrap intervals.
- **Date:** 2026-11-17
- **Model and settings:** TF-IDF + L2 Logistic Regression; two-stage validation search over C and character n-gram range
- **Result:** best setting for both tasks was C=30, ngram=(1, 2). Sentiment validation macro-F1 = 0.769, accuracy = 0.767. LD validation macro-F1 = 0.744, accuracy = 0.793. `results/tune_lr.csv`, `results/best_lr.csv`
- **Observation:** tuned LR was the strongest validation model on both tasks. Test split was not used.
