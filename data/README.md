# Data

This folder contains the local datasets used for the CSCI 3052U Machine Learning I group project on Cantonese natural language understanding.

The raw dataset files are not committed to GitHub. Each team member should obtain the data from the original CantoNLU repository and place the files in the same local folder structure.

## Source Repository

CantoNLU repository:

https://github.com/Aatlantise/canto-nlu

The two tasks currently being used in this project are:

- Sentiment analysis
- Language detection

## Local Folder Structure

    data/
    ├── README.md
    ├── sentiment/
    │   ├── train.jsonl
    │   ├── valid.jsonl
    │   └── test.jsonl
    └── ld/
        ├── ld_train.jsonl
        ├── ld_val.jsonl
        └── ld_test.jsonl

The `.jsonl` files above are stored locally only and are ignored by Git.

---

## Sentiment Analysis

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

The dataset is balanced across the three labels.

Initial inspection found no duplicate examples within the sentiment splits and no overlap between the train, validation, and test sets.

The data comes from OpenRice restaurant reviews through the CantoNLU project.

The CantoNLU data is released under CC BY 4.0, as confirmed by the CantoNLU authors; see `DATA_CARD.md` and the "License and Attribution" section of the main README.

---

## Language Detection

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

Current labels:

- `0` = Cantonese
- `1` = Mandarin
- `2` = code-mixed / corrupted sentence

We keep all three classes, as in the CantoNLU paper, so that our results are comparable with the published ones.

Initial inspection found some data-quality issues in the language-detection dataset, including:

- identical sentences with different labels
- a small amount of overlap between splits
- possible noise in the generated code-mixed examples
- inconsistencies between some documentation and preprocessing code

These issues are documented in `DATA_CARD.md` and are retained in the released splits for comparability with CantoNLU.

## Data Handling

The original data files should not be manually edited.

Any cleaning, filtering, or preprocessing used in the project should be performed through reproducible code.

The final project should document:

- exact dataset sources
- licensing and attribution
- train/validation/test splits
- label definitions
- preprocessing decisions
- known limitations

## Copying the Data

Run the commands below from the root of this repository.

The upstream CantoNLU repository should be cloned into a folder named `canto-nlu` inside this project directory.
Choose the instructions for the shell you are using.

**Data version.** This project uses the CantoNLU repository at commit `2283874` (Sept 28, 2026), so every
command block below checks out that commit after cloning. On Oct 11–12, 2026, upstream CantoNLU replaced
the language-detection data with a new grammar-based task (`lid_train.jsonl` / `lid_test.jsonl`, no
validation file); with the latest upstream version, the `ld_*.jsonl` files below do not exist. The
sentiment data is the same in both versions. If you already cloned the latest version, run
`git -C canto-nlu checkout 2283874` and copy the files again.

### Git Bash, macOS, or Linux

```bash
git clone https://github.com/Aatlantise/canto-nlu.git canto-nlu
git -C canto-nlu checkout 2283874
mkdir -p data/sentiment data/ld
cp canto-nlu/data/sentiment/{train,valid,test}.jsonl data/sentiment/
cp canto-nlu/data/ld/ld_{train,val,test}.jsonl data/ld/
```

### Windows Command Prompt

```bash
git clone https://github.com/Aatlantise/canto-nlu.git canto-nlu
git -C canto-nlu checkout 2283874
mkdir data\sentiment
mkdir data\ld

copy canto-nlu\data\sentiment\train.jsonl data\sentiment\
copy canto-nlu\data\sentiment\valid.jsonl data\sentiment\
copy canto-nlu\data\sentiment\test.jsonl data\sentiment\

copy canto-nlu\data\ld\ld_train.jsonl data\ld\
copy canto-nlu\data\ld\ld_val.jsonl data\ld\
copy canto-nlu\data\ld\ld_test.jsonl data\ld\
```

### Powershell

```bash
git clone https://github.com/Aatlantise/canto-nlu.git canto-nlu
git -C canto-nlu checkout 2283874

New-Item -ItemType Directory -Force data/sentiment, data/ld

Copy-Item canto-nlu/data/sentiment/train.jsonl data/sentiment/
Copy-Item canto-nlu/data/sentiment/valid.jsonl data/sentiment/
Copy-Item canto-nlu/data/sentiment/test.jsonl data/sentiment/

Copy-Item canto-nlu/data/ld/ld_train.jsonl data/ld/
Copy-Item canto-nlu/data/ld/ld_val.jsonl data/ld/
Copy-Item canto-nlu/data/ld/ld_test.jsonl data/ld/
```

After copying the files, verify that the datasets load correctly:
```
python src/data.py
```

The expected output should show:
- sentiment train: 9,999 rows
- sentiment validation: 999 rows
- language detection train: 42,753 rows
- language detection validation: 2,425 rows
