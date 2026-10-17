# Contributions

Course: CSCI 3052U Machine Learning I — Group 3 (Team CantoneseNLU).

This file is the team's contribution record, required at M6 and used as evidence for the M7
individual defence. Student numbers are deliberately **not** listed here; they are in the signed
teamwork contract. Keep entries short, factual and dated, and add them as the work happens rather
than at the end of a milestone.

## Team Members

- Kharintirasakar Uthayanan
- David Zajac
- Diana Riyahi
- Shawn Xiao
- Gamze Esen-Erdemir

## Principal roles (teamwork contract, Section B)

| Role | Member(s) | Responsibilities |
|---|---|---|
| Data Lead and Progress Presentation | Gamze Esen-Erdemir, Kharintirasakar Uthayanan, David Zajac | Cleaning, EDA, train/validation/test splits, leakage checks on features; preparing presentations |
| Modelling Lead | Gamze Esen-Erdemir, Kharintirasakar Uthayanan | Naive benchmarks, model selection, comparison models, theory sections |
| Evaluation and Analysis Lead | Diana Riyahi, Shawn Xiao | Metric justification, validation protocol, seeds/environment spec, experiment log, main-result regeneration script |
| Project Coordinator | Kharintirasakar Uthayanan | Agendas, GitHub project board, milestone submissions, timeline tracking, TA/instructor liaison |
| Communication Lead | Diana Riyahi, Kharintirasakar Uthayanan | Slides, report drafting and integration, figures, demo, revision-response document |

Primary roles identify each member's main area of responsibility; all members review and contribute
to the complete project, and roles may rotate or overlap across milestones so that everyone gains
experience with data preparation, modelling, training, evaluation, documentation and presentation.

## How to record a contribution

- One row per unit of work, newest last, under the milestone it belongs to.
- **Evidence** must be something a reader can open: a PR number, a commit hash, a file path, a
  notebook, a meeting date, a code review, or a document link. "Helped with modelling" is not evidence.
- Substantial pull requests are reviewed by at least one other member before merging; record the
  reviewer, because review is a contribution too.
- If AI assistance was used for a task, the task is also logged in [AI_USE.md](AI_USE.md); this
  file records who was responsible for the work and who verified it.

## M1 — Proposal, team contract, repository setup (due Sept 18, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| 2026-09-18 | Gamze Esen-Erdemir | Repository setup; baseline starter kit (`src/`, `requirements.txt`, first validation results in `results_valid.csv`, internal project notes kept outside the repository), produced with AI assistance — to be logged in `AI_USE.md` | `requirements.txt`, `results_valid.csv` | David Zajac |
| 2026-09-18 | Gamze Esen-Erdemir | Contract/proposal consistency audit and proposed contract revisions; created `AI_USE.md` and this file | Contract revision notes (shared with the team, kept outside the repository), `AI_USE.md`, `CONTRIBUTIONS.md` | Kharintirasakar Uthayanan |

Outstanding M1 items: Lee Language Lab topic declaration, GitHub project board, and applying the
agreed edits from the contract revision notes to the contract.

## M2 — Dataset and problem approval (due Oct 2, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| 2026-09-23 | Gamze Esen-Erdemir | Resolved merge conflicts and removed duplicated sections after PR #4 (`README.md`, `AI_USE.md`, `CONTRIBUTIONS.md`, `requirements.txt`) | Branch `fix-duplicate-docs` | Gamze Esen-Erdemir |
| 2026-09-28 | David Zajac | Sentiment-analysis EDA: verified split sizes, class balance, missing values, duplicate rows/texts, split overlap, review-length distribution, long-review outliers, and sample examples; updated sentiment findings in the data card | `notebooks/sentiment_eda.ipynb`, `DATA_CARD.md`, sentiment EDA PR | Gamze Esen-Erdemir |
| 2026-09-30 | Gamze Esen-Erdemir | Language-detection EDA: class balance, `source_id` group structure, duplicates and conflicting labels, overlap between splits, sentence length, missing values; summary of findings and open questions for the 2-class vs. 3-class decision | `notebooks/language_detection_eda.ipynb`, `DATA_CARD.md` branch `m2-ld-eda` | Kharintirasakar Uthayanan |
| 2026-09-30 | Gamze Esen-Erdemir | M2 documentation: three-class task decision based on the CantoNLU paper; `DATA_CARD.md` additions (EDA verification, sample inputs/outputs, privacy check, risks and mitigations); `LICENSE` and license/attribution fixes in `README.md` and `data/README.md`; updated proposal; approval-memo skeleton; task plan for the rest of M2 | `DATA_CARD.md`, `LICENSE`, `README.md`, `data/README.md`, branch `m2-ld-eda`; updated proposal and memo (shared as files) | Kharintirasakar Uthayanan |
| 2026-09-30 | Gamze Esen-Erdemir | Language-detection EDA: class balance, `source_id` group structure, duplicates and conflicting labels, overlap between splits, sentence length, missing values; summary of findings and open questions for the 2-class vs. 3-class decision | `notebooks/language_detection_eda.ipynb`, `DATA_CARD.md` branch `m2-ld-eda` | David Zajac |
| 2026-09-30 | Gamze Esen-Erdemir | M2 documentation: three-class task decision based on the CantoNLU paper; `DATA_CARD.md` additions (EDA verification, sample inputs/outputs, privacy check, risks and mitigations); `LICENSE` and license/attribution fixes in `README.md` and `data/README.md`; updated proposal; approval-memo skeleton; task plan for the rest of M2 | `DATA_CARD.md`, `LICENSE`, `README.md`, `data/README.md`, branch `m2-ld-eda`; updated proposal and memo (shared as files) | Kharintirasakar Uthayanan |
| 2026-10-01 | Diana Riyahi | Section 6 (data splits and evaluation protocol) official splits, group split by `source_id`, validation-only tuning, single test run, seeds and learning curve| Approval memo, Section 6 (shared as a file) | David Zajac |
| 2026-10-01 | Shawn Xiao | Benchmarked TF-IDF with Naive Bayes and logistic regression, measured LaBSE CPU embedding time, estimated full-dataset runtime and cache size, completed approval memo Section 8, and pinned package versions. | Branch `m2-compute-plan`; approval memo Section 8 | Diana Riyahi |
| 2026-10-01 | Kharintirasakar Uthayanan and Diana Riyahi | Worked on section 7 (primary metric): Macro-F1, the unwieghted average of the per-class F1 scroes, writing a justification as to why it was the best metric as well as secondary measures such as report accuracy and more.| Approval memo, section 7| Gamze Esen-Erdemir | 
| 2026-10-01 | Kharintirasakar |  Approved pull request reviewing the branch, reading through and editing the approval memo. I also revised the proposal adding changes since milestone along with Gamze. Also created a folder reports to hold the proposal and approval memo. |Aproval memo, Proposal | Shawn Xiao |

## M3 — Baseline reproduction (due Oct 19, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| 2026-10-09 | Gamze Esen-Erdemir | Data pipeline (`src/data.py`: loading and checking the official splits); naive benchmarks and TF-IDF + Naive Bayes / logistic regression baselines with a shared evaluation function and the `--final` test-split flag (`src/train_baselines.py`); validation results table; Naive Bayes diagnostic; README run and results sections; draft of the technical memo | `src/data.py`, `src/train_baselines.py`, `results/baseline_valid.csv`, `scripts/nb_ld_check.py`, `README.md`, branch `m3-baseline-pipeline` | Diana Riyahi |
| 2026-10-11 | Diana Riyahi | Reproduction check and code review of the M3 baseline pipeline on Windows 11, Python 3.11.9, scikit-learn 1.7.2: data shapes and `nb_ld_check.py` output match exactly; `results/baseline_valid.csv` differs only in LD TF-IDF + LR weighted-F1 (0.722 vs 0.721, unrounded 0.72161, a rounding-boundary difference); reported it in a review comment and approved the PR | PR #20 review and comment on `results/baseline_valid.csv` | Gamze Esen-Erdemir |
| 2026-10-11 | David Zajac | Fresh-clone reproducibility check and documentation review for the M3 baseline pipeline on Windows 10, Python 3.12.6, scikit-learn 1.7.2. Verified data loading, baseline execution, and Naive Bayes diagnostic; identified a 0.001 LD logistic-regression macro-F1/weighted-F1 difference from the committed Python 3.13.14 results; reviewed `src/data.py`; clarified `README.md` and `data/README.md` setup instructions for clone location and shell-specific data copying. | PR #20 review/comment; commits to `README.md` and `data/README.md`; `src/data.py`; `scripts/nb_ld_check.py` | Gamze Esen-Erdemir |
| 2026-10-11 | Shawn Xiao | Reproduction check and code review of the M3 baseline pipeline on Windows, Python 3.10, scikit-learn 1.7.2 (fresh virtual environment): data shapes and `nb_ld_check.py` output match exactly; `results/baseline_valid.csv` differs only in LD TF-IDF + LR (macro-F1 0.600 vs 0.599, weighted-F1 0.722 vs 0.721), consistent with the Python-version note in `README.md`; suggested writing the CSV with `float_format="%.3f"` so rounded values keep three decimals; approved the PR | PR #20 review and inline comment on `src/train_baselines.py` | *pending* |
| 2026-10-16 | Kharintirasakar Uthayanan | Created the v0.3-m3-baseline release; completed the M3 technical memo (repository link, date, final edits) and prepared the submission. Also tested the code to ensure there were less errors. I also prepared the release for the milestone. | Release v0.3-m3-baseline; M3 technical memo | Shawn Xiao |

## M4 — Progress presentation (due Oct 30, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| | | | | |

## M5 — Final experimentation, analysis and fine tuning (due Nov 20, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| | | | | |

## M6 — Final report, repository and demonstration (due Dec 4, 2026)

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| | | | | |

## M7 — Individual reflection and oral defence (due Dec 7, 2026)

Each member writes their own reflection separately; this section records only that it was
submitted.

| Date | Member | Contribution | Evidence | Reviewed by |
|---|---|---|---|---|
| | | | | |
