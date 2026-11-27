# Experiment Log — Milestone 5

One entry per experiment, newest last. All experiments use the **validation split only**;
the test split stays locked until the final report (`--final`).

Shared settings unless an entry says otherwise:
- Data: official CantoNLU splits, upstream commit `2283874`
- Features: TF-IDF of character 1–3-grams, fitted on train only (scikit-learn Pipeline)
- Seeds: 3052 (and 3053, 3054 where results are random)
- Primary metric: macro-F1; accuracy also reported

## Source of the final M5 numbers

All final M5 result files on `main` (main table, bootstrap intervals, learning curves, error
analysis) were produced on one machine by David Zajac's verification runs (E10, E13, E14), so they
are consistent with each other. Shared environment: scikit-learn 1.7.2, numpy 2.2.6,
scipy 1.18.1, pandas 2.3.3 (pinned in `requirements.txt`); CPU only; data: CantoNLU commit `2283874`.

**Numerical variation of tuned LR.** With weak regularisation (C = 30), LR is deterministic within
one process but changes slightly with the numerical environment: on Gamze's machine (Python 3.13.14,
Intel i7-1165G7) the same scripts gave 0.766 / 0.745 (sentiment / LD) instead of 0.769 / 0.744
(that rerun was not committed); learning curves run in parallel processes gave 0.747 on LD, the serial
run 0.744 (E13); and three refits of LD LR in one process all gave 0.7448, but 0.7458 with one BLAS
thread (281 instead of 211 iterations). The effect is at most 0.003 macro-F1, far smaller than the
bootstrap intervals (about ±0.02). For the final test run (M6) we plan to fix the number of threads.

## Run index

One row per run or search; every configuration of a search is in the listed CSV. Val = validation
macro-F1 (sentiment / LD). Test = locked until the final report. Runtime = one fit on the full
training split, TF-IDF included (from `results/learning_curves_summary.csv`, E13).

| Run | Date | Member | Model / experiment | Hyperparameters | Seeds | Val (SA / LD) | Test | Runtime | Output | Observations |
|---|---|---|---|---|---|---|---|---|---|---|
| E01 | Oct 9 | Gamze | Majority class | — | — | 0.167 / 0.268 | – | < 1 s | `baseline_valid.csv` | Lower bound |
| E02 | Oct 9 | Gamze | Stratified random | — | 3052–3054 | 0.327 / 0.335 (mean) | – | < 1 s | `baseline_valid.csv` | ≈ 1/3, as expected |
| E03 | Oct 9 | Gamze | TF-IDF + NB (default) | α = 1, char 1–3 | — | 0.686 / 0.268 | – | — | `baseline_valid.csv` | **Failed on LD:** predicts code-mixed for every sentence |
| E04 | Oct 9 | Gamze | TF-IDF + LR (default) | C = 1, char 1–3 | — | 0.751 / 0.599 | – | — | `baseline_valid.csv` | M3 baseline |
| E05 | Oct 29 | Gamze | NB, LR (default): train vs. validation | as E03, E04 | — | — | – | — | `m4_*.csv` | Sentiment overfits, LD underfits |
| E06 | Nov 7 | Gamze | NB, α search (stage 1) | α ∈ {1e-4 … 1}, 9 values, char 1–3 | — | best 0.749 / 0.646 | – | — | `tune_nb_alpha.csv` | Small α fixes the LD collapse |
| E07 | Nov 7 | Gamze | NB, n-gram × α (stage 2) | 4 n-gram ranges × 3 α per task | — | **α 0.01 / 0.003, char 1–3:** 0.749 / 0.646 | – | 22 s / 7 s | `tune_nb_ngram.csv` | Unigrams too weak; 4-grams no gain |
| E08 | Nov 13 | Shawn | LaBSE + kNN (cosine) | k ∈ {1 … 41}, 10 values | — | **k 41 / 1:** 0.644 / 0.382 | – | — | `tune_knn_labse.csv` | **Weak on LD:** large k collapses to the majority class |
| E09 | Nov 17 | Kharintirasakar, David | LR, C search (stage 1) | C ∈ {0.1 … 100}, 7 values, char 1–3 | — | best 0.766 / 0.741 | – | — | `tune_lr.csv` | Larger C (weaker L2) helps both tasks |
| E10 | Nov 17 | Kharintirasakar, David | LR, n-gram × C (stage 2) | 3 n-gram ranges × 3 C per task | — | **C 30, char 1–2:** 0.769 / 0.744 | – | 46 s / 31 s | `tune_lr.csv`, `best_lr.csv`, `main_results_valid.*` | Best model on both tasks |
| E11 | Nov 17 | Diana | Bootstrap CIs, paired differences | B = 1000, seed 3052 | 3052 | — | – | — | `uncertainty_*.csv` | **Superseded by E13:** LR was still the C = 1 default |
| E12 | Nov 17 | Diana | Learning curves | 10 / 25 / 50 / 100% of train | 3052–3054 | — | – | — | `learning_curves*.csv` | **Superseded by E13:** LR was still the C = 1 default |
| E13 | Nov 18 | David | Bootstrap and learning curves with tuned LR | tuned settings of E07, E10; serial run | 3052–3054 | LR 0.769 [0.743, 0.792] / 0.744 [0.725, 0.763] | – | — | `uncertainty_*`, `learning_curves*` | Final M5 uncertainty and sensitivity numbers |
| E14 | Nov 17 | David | Error analysis of tuned LR | C = 30, char 1–2 | — | 233 / 501 errors | – | — | `error_*.csv`, `m5_error_analysis_notes.md` | Neutral class (sentiment) and code-mixed boundary (LD) |
| E15 | Nov 27 | Gamze | LR solver tolerance check | C ∈ {10, 30, 100} × char (1,2), (1,3); tol = 1e-6; 1 and 8 threads | — | C 30, char 1–2: 0.765 / 0.743 | – | 59 s / 53 s | (this entry) | M5 variation came from the loose default tol; converged LR is identical across threads; M5 selection kept |
| E16 | Nov 27 | Gamze | `final_test.py` dry run on validation | settings of E07, E10; LR tol = 1e-6 | 3052–3054 | NB 0.749 / 0.646; LR 0.765 [0.739, 0.789] / 0.743 [0.725, 0.761] | – | 148 s / 71 s (LR) | `final_valid*` | Validation reference for the test run |

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

### 2026-11-07 — Naive Bayes tuning (Gamze Esen-Erdemir)
*Runs E06 (stage 1) and E07 (stage 2).*
- **Goal:** tune NB smoothing and n-gram range; test the M3 hypothesis that NB fails on LD because of smoothing
- **Script / command:** `python scripts/tune_nb.py` (stage 1), `python scripts/tune_nb_ngram.py` (stage 2)
- **Task(s):** sentiment, LD
- **Model and settings:** TF-IDF (character n-grams) + MultinomialNB, fit_prior=True
- **Search space:** stage 1: α ∈ {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}, char 1–3-grams. Stage 2 (coarse-to-fine): n-gram range ∈ {(1,1), (1,2), (1,3), (1,4)} × the best α and its two neighbours per task
- **Selection rule:** highest validation macro-F1
- **Result:** sentiment: (1,3), α = 0.01, macro-F1 0.749 (default 0.686). LD: (1,3), α = 0.003, macro-F1 0.646 (default 0.268). `results/tune_nb_alpha.csv`, `results/tune_nb_ngram.csv`
- **Observation:** the hypothesis holds: with α = 1 the fake counts over a very large vocabulary (1.6 M n-grams for sentiment, 0.46 M for LD) drown the real n-gram counts; smaller α fixes the LD collapse. On LD, performance falls again below α = 0.003 (too little smoothing), so the search brackets the optimum. Unigrams alone are not enough (LD 0.416); 4-grams add nothing over (1,3). α = 1 reproduces the M3 numbers exactly.

### 2026-11-13 — k-NN on frozen LaBSE embeddings (Shawn Xiao)
*Run E08.*
- **Goal:** tune k for k-NN classification on frozen LaBSE sentence embeddings (Milestone 5)
- **Script / command:** `python scripts/knn_labse.py`
- **Task(s):** sentiment, LD
- **Model and settings:** LaBSE embeddings (frozen, L2-normalized, cached once per task/split under `embeddings/`), k-NN with cosine metric. No vectoriser to fit inside a Pipeline here: LaBSE is a fixed pretrained extractor, not trained on our train split, so there is no TF-IDF-style leakage risk.
- **Search space:** k ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31, 41}
- **Selection rule:** highest validation macro-F1
- **Result:** sentiment best k=41, macro-F1 0.644, accuracy 0.643; LD best k=1, macro-F1 0.382, accuracy 0.506; `results/tune_knn_labse.csv`, `results/tune_knn_labse.png`
- **Observation:** on LD, accuracy keeps rising with k (up to 0.672 at k=41) while macro-F1 falls — larger k collapses predictions toward the majority class, the same effect Gamze saw with NB on LD. On sentiment, kNN-on-LaBSE (0.644) still trails TF-IDF+LR (0.751).

### 2026-11-17 — Bootstrap confidence intervals for macro-F1 (Diana Riyahi)
*Run E11. Superseded by E13 (same script, re-run with tuned LR); kept here as the original record.*
- **Goal:** how much of the validation macro-F1 differences between models is real rather than sampling noise
- **Script / command:** `python scripts/uncertainty.py`
- **Task(s):** sentiment, LD
- **Model and settings:** majority; TF-IDF + NB with tuned settings (best row of `results/tune_nb_ngram.csv`: α = 0.01 sentiment, α = 0.003 LD, char 1–3-grams); TF-IDF + LR with the untuned default (C = 1, char 1–3-grams) because `results/best_lr.csv` is not on `main` yet (the script reads it automatically once it is). LaBSE + kNN is skipped unless the embedding cache exists. Each model is fitted once on train; validation predictions are resampled B = 1000 times (seed 3052), the same resamples for every model, 95% percentile intervals. Sentiment resamples reviews; LD resamples whole `source_id` groups (455 groups for 2425 sentences), since the variants of one source sentence are not independent.
- **Search space:** none (uses the settings chosen by the tuning scripts)
- **Result:** sentiment: NB 0.749 [0.724, 0.776], LR 0.751 [0.725, 0.778]; paired LR − NB = +0.003 [−0.020, +0.025]. LD: NB 0.646 [0.621, 0.669], LR 0.599 [0.571, 0.626]; paired NB − LR = +0.047 [+0.020, +0.075]. Majority: 0.167 (sentiment), 0.268 (LD). `results/uncertainty_ci.csv`, `results/uncertainty_paired.csv`, `results/uncertainty_ci.png`
- **Observation:** on sentiment, NB and LR cannot be told apart (the paired interval contains 0 and LR wins only 60% of resamples), while on LD tuned NB beats default LR in every resample. The intervals are about ±0.025, so differences under ~0.03 between single validation scores should not be reported as improvements. For LD, the group-level and sentence-level bootstraps gave almost the same width (SE 0.012 vs. 0.012 for NB). This was the initial run using default LR because the tuned LR results were not yet available; the analysis was rerun with the final tuned LR settings on 2026-11-18 (see the verification entry below).

### 2026-11-17 — Learning curves (Diana Riyahi)
*Run E12. Superseded by E13 (same script, re-run with tuned LR); kept here as the original record.*
- **Goal:** sensitivity of each model to the amount of training data; would more data help?
- **Script / command:** `python scripts/learning_curves.py`
- **Task(s):** sentiment, LD
- **Model and settings:** same NB and LR settings as the bootstrap entry above. Train on a label-stratified random 10%, 25%, 50% of the train split (seeds 3052, 3053, 3054) and on 100% (one fit: same data for every seed, deterministic models); score on the full validation split. Train macro-F1 is measured on the subset used for fitting.
- **Search space:** training fraction ∈ {10%, 25%, 50%, 100%}
- **Result:** validation macro-F1 (mean ± std over seeds) at 10 / 25 / 50 / 100%: sentiment LR 0.673±0.012 / 0.717±0.014 / 0.736±0.009 / 0.751, NB 0.691±0.013 / 0.721±0.002 / 0.738±0.002 / 0.749; LD LR 0.299±0.005 / 0.377±0.019 / 0.469±0.003 / 0.599, NB 0.436±0.012 / 0.477±0.010 / 0.518±0.007 / 0.646. `results/learning_curves.csv`, `results/learning_curves_summary.csv`, `results/learning_curves.png`
- **Observation:** sentiment is flattening (+0.015 from 50% to 100%) with a train–validation gap of 0.20–0.25, so more data of the same kind would help only a little and the models mainly overfit; LD is still rising steeply (+0.13 from 50% to 100% for both models), so LD is data-limited, and LR on LD barely fits its own training data (train 0.62), which matches the underfitting found in M4. Seed-to-seed spread (≤ 0.02) is smaller than the bootstrap intervals. This was the initial run using default LR; the learning curves were rerun with the final tuned LR settings on 2026-11-18 (see the verification entry below).

### 2026-11-17 — Logistic-regression tuning (David Zajac)
*Runs E09 (stage 1) and E10 (stage 2).*
- **Goal:** tune the logistic-regression baseline for both tasks and identify the strongest validation configuration for the Milestone 5 model comparison
- **Script / command:** `python scripts/tune_lr.py`; `python scripts/main_results.py`
- **Task(s):** sentiment, LD
- **Model and settings:** TF-IDF character features + L2 LogisticRegression; final selected configuration C = 30, char 1–2-grams
- **Search space:** C ∈ {0.1, 0.3, 1, 3, 10, 30, 100}; character n-gram range ∈ {(1,1), (1,2), (1,3), (1,4)}
- **Selection rule:** highest validation macro-F1; ties prefer smaller C
- **Result:** best setting for both tasks was C = 30, ngram = (1, 2). Sentiment validation macro-F1 = 0.769, accuracy = 0.767; LD validation macro-F1 = 0.744, accuracy = 0.793. Outputs: `results/tune_lr.csv`, `results/best_lr.csv`, `results/main_results_valid.csv`, `results/main_results_valid.png`
- **Observation:** tuned LR was the strongest validation model on both tasks and improved over the default LR configuration, especially on language detection. The test split was not used.

### 2026-11-18 — Tuned-model uncertainty and learning-curve verification (David Zajac)
*Run E13.*
- **Goal:** verify that Diana’s uncertainty and learning-curve analyses use the final tuned LR settings and produce results consistent with the main validation comparison
- **Script / command:** `python scripts/uncertainty.py`; `python scripts/learning_curves.py --jobs 1`
- **Task(s):** sentiment, LD
- **Model and settings:** tuned TF-IDF + NB and tuned TF-IDF + LR; LR uses C = 30, char 1–2-grams
- **Search space:** training fraction ∈ {10%, 25%, 50%, 100%} for learning curves; bootstrap B = 1000 for uncertainty analysis
- **Selection rule:** fixed hyperparameters from prior tuning; no new model selection performed
- **Result:** sentiment LR macro-F1 = 0.769 with 95% bootstrap CI [0.743, 0.792]; LD LR macro-F1 = 0.744 with 95% bootstrap CI [0.725, 0.763]. Paired LR − NB difference = 0.020 [−0.006, 0.044] for sentiment and 0.098 [0.076, 0.122] for LD. The serial learning-curve run reproduced the final full-data scores exactly: sentiment 0.769 and LD 0.744.
- **Observation:** the serial rerun resolved the earlier LD 0.747 vs 0.744 discrepancy. Tuned LR remained strongest on both tasks; the sentiment LR–NB gap was small and its paired interval crossed zero, while the LD advantage was larger and clearly above zero. The test split was not used.

### 2026-11-17 — Error analysis of the tuned models (David Zajac)
*Run E14. Entry written by Gamze from David's notes (`results/m5_error_analysis_notes.md`).*
- **Goal:** identify and explain representative errors of the best model on each task
- **Script / command:** `python scripts/error_analysis.py`
- **Task(s):** sentiment, LD
- **Model and settings:** tuned TF-IDF + LR (C = 30, char 1–2-grams) on both tasks
- **Search space:** none
- **Result:** sentiment: 233 validation errors, 221 of them involve the neutral `ok` class; LD: 501 errors, 496 involve the code-mixed class, only 5 direct Cantonese↔Mandarin confusions. Nine representative examples with notes in `results/error_selected_examples.csv`; full lists in `results/error_analysis_*.csv`, counts in `results/error_counts_*.csv`
- **Observation:** sentiment errors are mostly mixed reviews (praise and criticism together) pulled towards or away from `ok`, but a few strong errors exist. In LD, the pure languages are well separated; the boundary to synthetic code-mixed text, sometimes only a subtle change from a pure sentence, is the main difficulty. Notes and translations were not checked by a Cantonese speaker.

### 2026-11-27 — LR solver tolerance and final-evaluation dry run (Gamze Esen-Erdemir)
*Runs E15 and E16.*
- **Goal:** before the single test run, check why tuned LR varied between runs in M5, and check the final evaluation script on validation
- **Script / command:** tolerance check: ad-hoc refits of the tuned LR pipeline (not committed); dry run: `python scripts/final_test.py`
- **Task(s):** sentiment, LD
- **Model and settings:** TF-IDF + LR, C = 30, char 1–2-grams (M5 selection), lbfgs; default tol = 1e-4 vs. tol = 1e-6 with max_iter = 5000; 1 vs. 8 BLAS threads
- **Search space:** check only: C ∈ {10, 30, 100} × n-gram range ∈ {(1,2), (1,3)} with tol = 1e-6, to see whether the M5 selection still holds
- **Result:** with the default tol, the same sentiment model gave 0.759 (104 iterations, inside the Pipeline) or 0.766 (116 iterations, vectoriser fitted separately), a 0.007 difference caused only by floating-point rounding; with tol = 1e-6 both gave 0.7652 (about 300 iterations), with 1 and with 8 threads. LD: 0.7426 in every setting. Converged grid: sentiment (1,2): 0.763 / **0.765** / 0.765 for C = 10 / 30 / 100, (1,3): 0.768 / 0.764 / 0.764; LD (1,2): 0.739 / **0.743** / 0.743, (1,3): 0.734 / 0.737 / 0.734. Dry run of `final_test.py` on validation: majority 0.167 / 0.268, stratified 0.328 ± 0.006 / 0.335 ± 0.013, NB 0.749 [0.724, 0.776] / 0.646 [0.622, 0.669], LR 0.765 [0.739, 0.789] / 0.743 [0.725, 0.761]; paired LR − NB +0.016 [−0.009, +0.040] (sentiment), +0.097 [+0.074, +0.121] (LD). `results/final_valid*.csv`, `results/final_valid.png`
- **Observation:** the M5 run-to-run variation (up to 0.003, and 0.010 in this check) was not a thread effect but an unconverged solver: with weak regularisation the training data are almost separable, the loss is very flat near the optimum, and lbfgs stopped at tol = 1e-4 at a point that depends on rounding noise. Running to convergence removes the variation. The M5 selection (C = 30, char 1–2) is kept: on LD it is the converged optimum, on sentiment the best converged setting (char 1–3, C = 10) is only 0.002 higher, far inside the bootstrap interval (±0.025). The final evaluation therefore uses the M5 settings with tol = 1e-6; the converged validation scores (0.765 / 0.743) are the reference for the test run. NB and the naive baselines are unchanged. The stratified mean is 0.328 instead of 0.327 because it now averages unrounded seed scores.
