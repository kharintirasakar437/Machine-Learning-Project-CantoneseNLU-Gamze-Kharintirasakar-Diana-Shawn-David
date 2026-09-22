# Data Card

> Status: Draft for Milestone 2. Dataset sources, licensing details, and preprocessing decisions are still being verified.

# 1. Sentiment Analysis Dataset

## Task

Classify Cantonese restaurant-review text into one of three sentiment labels.

## Source

CantoNLU repository:

https://github.com/Aatlantise/canto-nlu

Relevant directory:

    data/sentiment/

The dataset is based on OpenRice restaurant reviews.

Exact upstream attribution and licensing information will be confirmed during Milestone 2.

## Unit of Analysis

One restaurant review per example.

## File Format

JSON Lines (`.jsonl`).

Fields include:

- `id`
- `sentence`
- `label`
- `source`

## Splits

| Split | Rows |
|---|---:|
| Train | 9,999 |
| Validation | 999 |
| Test | 999 |

## Labels

- `smile`
- `ok`
- `cry`

Each split is balanced across the three labels.

The project will preserve the original label names unless the upstream documentation provides explicit semantic definitions.

## Initial Data Quality Findings

Initial repository inspection found:

- no duplicate examples within the sentiment splits
- no overlap between train, validation, and test
- balanced class distributions

These findings will be independently verified during exploratory data analysis.

## Planned Preprocessing

The team plans to:

- check missing or invalid values
- check duplicate examples
- inspect text lengths
- inspect unusual or malformed examples
- use character-level TF-IDF features for classical models

Any fitted preprocessing tools will be trained using the training split only.

## Evaluation

Primary metric:

- Macro-F1

Secondary measures:

- Accuracy
- Per-class precision
- Per-class recall
- Confusion matrix

Validation data will be used for model selection and tuning.

The final test set will be reserved for final evaluation.

## Known Limitations

Possible limitations include:

- restaurant-review domain only
- informal Cantonese
- code-switching
- emojis and other non-standard text
- limited generalization to other Cantonese domains

---

# 2. Language Detection Dataset

## Task

Classify text according to language category.

The current dataset contains three classes:

- Cantonese
- Mandarin
- Code-mixed / corrupted text

The original project proposal described the task as Cantonese vs. Mandarin.

The team will decide during Milestone 2 whether to keep all three classes or restrict the task to Cantonese and Mandarin.

## Source

CantoNLU repository:

https://github.com/Aatlantise/canto-nlu

Relevant directory:

    data/ld/

Repository inspection indicates that the task was generated using Cantonese-Mandarin parallel sentence data.

Exact upstream attribution and licensing information will be confirmed during Milestone 2.

## Unit of Analysis

One sentence per example.

## File Format

JSON Lines (`.jsonl`).

Fields include:

- `id`
- `source_id`
- `sentence`
- `label`

## Splits

| Split | Rows |
|---|---:|
| Train | 42,753 |
| Validation | 2,425 |
| Test | 2,341 |

## Labels

- `0` = Cantonese
- `1` = Mandarin
- `2` = code-mixed / corrupted sentence

## Initial Data Quality Findings

Initial repository inspection identified several possible issues:

- identical sentences with different labels
- a small amount of overlap between dataset splits
- possible label noise in generated code-mixed examples
- possible inconsistencies between the written documentation and the dataset-generation code
- class imbalance if all three labels are retained

These findings will be independently verified before any cleaning decisions are made.

## Planned Treatment

Before model training, the team will:

1. reproduce the class counts
2. check duplicate examples
3. check overlap between splits
4. identify conflicting labels
5. decide between a 2-class and 3-class version of the task
6. document all cleaning decisions

The original dataset files will remain unchanged.

Any cleaned or filtered datasets will be generated through reproducible code.

## Evaluation

Primary metric:

- Macro-F1

Secondary measures:

- Accuracy
- Per-class precision
- Per-class recall
- Confusion matrix

Macro-F1 is particularly important if the 3-class task is retained because the class distribution is imbalanced.

## Known Limitations

Possible limitations include:

- class imbalance
- generated code-mixed examples
- label noise
- duplicate or overlapping examples
- preprocessing artefacts
- possible shortcut features that make Cantonese and Mandarin easier to distinguish

---

# 3. Train / Validation / Test Policy

For both tasks:

- training data will be used to fit models
- validation data will be used for model selection and hyperparameter tuning
- the final test set will be reserved for final evaluation
- fitted preprocessing tools will use training data only
- models will be evaluated under the same protocol wherever possible

---

# 4. Remaining Milestone 2 Work

The team still needs to confirm:

- exact upstream dataset sources
- exact licensing and attribution
- final language-detection task definition
- final language-detection cleaning policy
- exploratory data analysis results
- final train/validation/test strategy
- primary metric
- compute feasibility