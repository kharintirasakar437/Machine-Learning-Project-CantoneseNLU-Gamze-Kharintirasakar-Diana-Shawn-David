# Data Card

> Status: Draft for Milestone 2. Dataset sources and licensing have been identified. Exploratory analysis and final preprocessing decisions are still in progress.

## Shared Licensing and Attribution

The CantoNLU project lead confirmed by email that the project code and datasets are available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Our group has permission to use the data and code for this project provided that appropriate attribution is given to the CantoNLU authors.

The original CantoNLU repository and related paper will be cited in the final project documentation.

# 1. Sentiment Analysis Dataset

## Task

Classify Cantonese restaurant-review text into one of three sentiment labels.

## Source

CantoNLU repository:

https://github.com/Aatlantise/canto-nlu

Relevant directory:

    data/sentiment/

The sentiment dataset is based on OpenRice restaurant reviews and is distributed as part of the CantoNLU project.

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

The project will preserve the original smile, ok, and cry labels used by CantoNLU.

## Verified Data Quality Findings

Exploratory data analysis confirmed:

- no missing values in the training, validation, or test splits
- no duplicate rows within any split
- no duplicate review texts within any split
- no review-text overlap between train, validation, and test
- perfectly balanced class distributions across `smile`, `ok`, and `cry`
- similar average and median review lengths across all three splits
- a small number of long review outliers, including a maximum training length of 4,720 characters

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
- emojis and other non-standard text- substantial variation in review length
- limited generalization to other Cantonese domains

## Sample Inputs and Outputs

The model input is the review text (`sentence`); the output is one of the three labels. Examples from the training split (English translations are AI-assisted and have not been checked by a Cantonese speaker):

| Input (`sentence`) | English translation | Output (`label`) |
|---|---|---|
| 環境好,食物好,正!!! 每次到石澳, 都會到儿回味吃魷魚筒焗飯, 好正! 真係唔好錯過呀!! | Good atmosphere, good food, great!!! Every time I go to Shek O I come back here for the baked squid rice, so good! Really don't miss it!! | `smile` |
| 唔見得比其他酒家特別好食 一行7人到此酒家食晚飯, 7道海鮮 + 雞連炒飯 & 炒菜共10道菜 total: $2750 無特別好食, 水準只屬合格, 不值專程到流浮山食呢間嘢... | Not noticeably better than other restaurants. Seven of us had dinner here, 10 dishes in total for $2,750. Nothing special, only passable; not worth a special trip to Lau Fau Shan... | `ok` |
| 侍應唔得 侍應無溝通，多過一枱客問ｏ野，就開始亂．ｏ野食一般，好逼．人多唔好去，有位坐無位食． | The waiters are poor and don't communicate; once more than one table asks for something, it becomes chaotic. The food is average and it is very crowded. Don't go when it's busy. | `cry` |

Privacy check (training split, Sept 30, 2026): the reviews are publicly posted restaurant reviews, and the data has no reviewer name or ID field (`source` is always `openrice`). A small number of review texts contain URLs (14), 8-digit numbers that are mostly restaurant phone numbers (15), or `@` handles (45, mostly restaurant or hotel accounts, a few reviewers promoting their own social-media accounts). No e-mail addresses were found. We do not redistribute the data, and such reviews will not be used as examples in reports.

While selecting examples we also noticed that some labels do not obviously match the text (e.g. training review `id` 94 is labelled `ok` but reads as clearly positive). The labels appear to come from the reviewers' own ratings rather than from separate annotation; this is noted as a possible source of label noise.

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

Repository inspection indicates that the language-detection dataset was generated from Cantonese-Mandarin parallel sentence data and then processed into Cantonese, Mandarin, and code-mixed examples.

The EDA (`notebooks/language_detection_eda.ipynb`) is consistent with this: rows that share a `source_id` are versions of the same original sentence, and each source sentence produced between 1 and 8 rows in the training set.

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

Class distribution (from the EDA):

| Split | 0 (Cantonese) | 1 (Mandarin) | 2 (Code-mixed) |
|---|---:|---:|---:|
| Train | 6,924 (16.2%) | 6,971 (16.3%) | 28,858 (67.5%) |
| Validation | 398 (16.4%) | 399 (16.5%) | 1,628 (67.1%) |
| Test | 396 (16.9%) | 398 (17.0%) | 1,547 (66.1%) |

The class proportions are consistent across splits. No `source_id` appears in more than one split, so the splits were made by source sentence (a group split): all versions of the same original sentence stay in the same split.

## Labels

- `0` = Cantonese
- `1` = Mandarin
- `2` = code-mixed / corrupted sentence

Full definitions from `canto-nlu/data/ld/README.md`:

- `0`: original Cantonese sentence
- `1`: original Mandarin sentence (converted to traditional characters)
- `2`: corrupted, partially code-mixed sentence

These definitions come from the upstream README and have not yet been checked against the upstream generation code.

## Initial Data Quality Findings

Initial repository inspection identified several possible issues:

- identical sentences with different labels
- a small amount of overlap between dataset splits
- possible label noise in generated code-mixed examples
- possible inconsistencies between the written documentation and the dataset-generation code
- class imbalance if all three labels are retained

These findings will be independently verified before any cleaning decisions are made.

## EDA Verification of the Findings

Verified in `notebooks/language_detection_eda.ipynb` (Sept 30, 2026). Test sentences were only counted, not inspected.

| Initial finding | Result |
|---|---|
| Identical sentences with different labels | **Confirmed.** 161 sentences in train (322 rows, about 0.75%), 18 in validation and 12 in test appear with two different labels. Every conflict involves label 2: 148 are labelled both 1 and 2, and 13 both 0 and 2. No sentence is labelled both Cantonese and Mandarin. |
| Overlap between splits | **Confirmed, small.** No `source_id` is shared between splits, but 7 identical sentences appear in both train and validation and 8 in both train and test (0 between validation and test). The source corpus repeats some sentences, mostly Wikipedia template sentences, under different `source_id`s. The overlapping sentences have the same label in both splits. |
| Label noise in generated code-mixed examples | **Partly supported.** The label conflicts above suggest that the corruption step sometimes left a sentence unchanged while still labelling it as code-mixed. Why Mandarin sentences are affected much more often than Cantonese ones is still open. |
| Documentation vs. generation code | **Not yet verified.** The generation code has not been read. |
| Class imbalance with three labels | **Confirmed.** Label 2 is about two thirds of every split. |

Additional findings from the EDA:

- Versions of the same source sentence are very similar at the character level (Cantonese vs. Mandarin similarity 0.88 in the inspected group), so the task depends on small character-level differences.
- 268 extra copies of sentences exist within train (23 in validation, 14 in test); 161 of the train copies come from the label conflicts above.
- Some text contains likely errors from automatic simplified-to-traditional conversion (e.g. `賣瞭`, `代錶隊` instead of `賣了`, `代表隊`). This is a hypothesis to check against the upstream conversion code.
- Some rows are sentence fragments (e.g. starting with a comma), and 4 rows (about 714 characters, all from `source_id` 8038) look like a whole paragraph rather than a sentence.
- No missing values or empty sentences were found in any split.

## Planned Treatment

Before model training, the team will:

1. reproduce the class counts
2. check duplicate examples
3. check overlap between splits
4. identify conflicting labels
5. decide between a 2-class and 3-class version of the task
6. document all cleaning decisions

Status (Sept 30, 2026): steps 1–4 are done in `notebooks/language_detection_eda.ipynb`. Steps 5 and 6 are still open. Note for step 5: every label conflict involves label 2, so a 2-class version (labels 0 and 1) would be almost exactly balanced and free of these conflicts, but would drop about two thirds of the data.

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

Sentence length was checked as a possible shortcut: median length is 32 characters for Cantonese and Mandarin and 35 for code-mixed sentences, and the distributions overlap heavily, so length alone is a weak signal.

## Sample Inputs and Outputs

The model input is a single sentence (`sentence`); the output is one of the three labels. The examples below are versions of the same source sentence (training split, `source_id` 0), which shows how small the differences between the classes are. English translation (AI-assisted, not checked by a Cantonese speaker): "Kin Sang Estate is a housing estate under the Tenants Purchase Scheme; in the same building some flats have already been sold, and the rest are rental flats."

| Input (`sentence`) | Output (`label`) |
|---|---|
| 建生邨屬於租者置其屋計劃**嘅**屋邨，同一大廈內，有**啲**單位已經賣**咗**，其餘**嘅係**租住單位。 | `0` (Cantonese) |
| 建生邨屬於租者置其屋計劃**的**屋邨，同一大廈內，有**些**單位已經賣**瞭**，其餘**的是**租住單位。 | `1` (Mandarin) |
| 建生邨屬於租者置其屋計劃**的**屋邨，同一大廈內，有**些**單位已經賣**咗**，其餘**嘅係**租住單位。 | `2` (code-mixed) |

The bold characters are the only differences: Cantonese 嘅 / 啲 / 咗 / 係 versus Mandarin 的 / 些 / 了 / 是. The Mandarin sentence shows `賣瞭` instead of `賣了`, which is likely an error from the automatic conversion to traditional characters (the CantoNLU paper states that HanziConv was used).

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

- final language-detection task definition
- final language-detection cleaning policy
- exploratory data analysis results
- verification of train/validation/test handling
- confirmation of macro-F1 as the final primary metric
- compute feasibility

Update (Sept 30, 2026): the language-detection EDA is complete; see `notebooks/language_detection_eda.ipynb` and the "EDA Verification of the Findings" section above.
