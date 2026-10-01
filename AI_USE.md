# AI Use Log

Course: CSCI 3052U Machine Learning I — Group 3 (Team CantoneseNLU).

Policy (from the teamwork contract and course spec): AI assistance is logged here **at the time of
use**, not reconstructed later. Generative models studied as an *object of the experiment* (the
stretch pilot LLM evaluation) are logged in the separate section at the bottom, because that is
research data, not assistance.

## Section 1 — AI assistance used to produce our work

| Date | Member | Tool / Model | Use | Verification |
|---|---|---|---|---|
| 2026-09-18 | David Zajac | ChatGPT | Helped review and revise the Milestone 1 proposal and make sure it matched the course requirements. | Compared the suggestions against the project specification, proposal template, and group documents before using them. |
| 2026-09-21 | David Zajac | ChatGPT | Helped plan next steps for Milestone 2 and organize the repository and data workflow. | Checked the recommendations against the Milestone 2 requirements and the current project plan. |
| 2026-09-21 | David Zajac | Claude Code | Inspected the CantoNLU repository to locate relevant datasets, splits, labels, preprocessing code, and possible data-quality issues. | The repository was inspected in read-only mode and the findings were reviewed before being used for project planning. |
| 2026-09-22 | David Zajac | ChatGPT | Helped interpret the CantoNLU repository findings and draft documentation such as `README.md`, `data/README.md`, and `DATA_CARD.md`. | Content was reviewed against the repository findings and course requirements before being added. |
| 2026-09-23 | Gamze Esen-Erdemir | Claude Code | Resolved merge conflicts between `main` and `m2-data-documentation` and removed duplicated sections introduced by the PR #4 merge (`README.md`, `AI_USE.md`, `CONTRIBUTIONS.md`, `requirements.txt`). | Changes reviewed by Gamze before commit; PR reviewed by another member. |
| 2026-09-30 | Gamze Esen-Erdemir | Claude Code | Language-detection EDA (`notebooks/language_detection_eda.ipynb`), done as a step-by-step discussion. Gamze chose to work through the analysis step by step rather than use a ready-made notebook, located the label definitions in the upstream README, wrote the class-count and empty-sentence checks, and decided what belonged in the EDA (e.g. leaving the majority-baseline calculation to M3). Claude Code explained each step, suggested code for most checks, reorganized the notebook into a clean run order, and drafted most of the Markdown findings. | Gamze ran every cell and inspected the outputs, questioned claims that were not yet supported by the notebook (which led to added checks and corrected findings), and re-ran the notebook top to bottom without errors before commit. PR reviewed by another member. |
| 2026-09-30 | Gamze Esen-Erdemir | Claude Code | Milestone 2 documentation. Gamze decided the scope and the key choices: keep all three language-detection classes after checking the paper, use MIT for our code, add to `DATA_CARD.md`, base the updated proposal on the submitted M1 version, and split the remaining M2 tasks among the team. Claude Code looked up the language-detection definition and results in the CantoNLU paper, wrote the privacy-check code, drafted the `DATA_CARD.md` additions (EDA verification, sample inputs/outputs with English translations, risks), the `LICENSE` file and the license/attribution fixes in `README.md` and `data/README.md`, edited the updated proposal (`.docx`), drafted the approval-memo skeleton, and drafted the task instructions for teammates. | Gamze reviewed every change; all numbers were checked against the data and the EDA notebook, and sample rows were matched against the data files by code. Gamze checked the quoted passages and Table 3 values in the CantoNLU paper herself. The English translations were not checked by a Cantonese speaker and are marked as such. PR reviewed by another member. |
| 2026-10-01 | Diana Riyahi | Claude Code | Approval memo Sections 6 and 7. Diana wrote the draft notes; Claude Code turned them into sentences, edited the grammar and wording, shortened them | Diana reviewed the final text against her notes before adding it to the memo. |



## Section 2 — Generative models as an object of study

None yet. The pilot evaluation (one model, one task) is a stretch goal for M4; if it runs, record
the model name and version, the exact prompt, the decoding settings, the date, and the task and
split used.
