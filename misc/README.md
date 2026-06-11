# Pooling System Dataset — Unified Analytic Table

## Overview

The **Pooling System Dataset** (`data/processed/validations/df_pooling_system.parquet`) is a single wide-format table that consolidates all task-specific data collected across Calls 1, 2, and 4 of the MATHANX (Math Anxiety in LLMs) study, together with synthetic persona demographics. Each row represents one experimental run (`run_id`).

It was produced by `misc/dataset_aggregations_pooling_system.py` and serves as the primary input to the downstream ML feature-engineering and training pipeline (`notebooks/ml_dataset_creation.py` → `data/processed/ml/ml_dataset.csv`).

| Property | Value |
|----------|-------|
| Rows | 27,987 |
| Columns | 205 |
| Format | Parquet (snappy-compressed) |
| Size (disk) | ~120 MB |

---

## Study Context: The Four Calls

In each run an LLM, prompted with either a synthetic persona (`human` mode) or no persona (`llm` mode), completed four tasks:

| Call | Task | Description | Output Structure |
|------|------|-------------|------------------|
| **Call 1** | **TFMN** (Textual Forma Mentis) | 7 open-ended questions probing math relationship, anxiety, AI use, math explanations, and LLMs in education. Each free-text answer was analysed with the `emoatlas` library to extract 8 emotion z-scores. | 7 answers × (text + 8 emotion scores) |
| **Call 2** | **Psychometric Scales** | Three validated Likert-scale instruments: **MAES** (9 items), **AMAS** (9 items), and **MSEAQ** (28 items). Each item rated 1–5. | 46 item ratings |
| **Call 3** | **Forma Mentis Network** | 25 cue words prompted free associations to build a behavioural semantic network. *(Not included in this table — used for network analysis only.)* | — |
| **Call 4** | **MSESR Problem Solving** | 18 multiple-choice math problems. Each records the chosen option (A–E), free-text reasoning, and a confidence rating (1–5). | 18 × (option + reasoning + confidence) |

Only `human`-mode runs have associated persona demographics.

---

## Source Datasets

Four files were read and transformed by the pooling script:

| Source | File | Rows | Columns | Format |
|--------|------|------|---------|--------|
| Call 1 | `data/processed/validations/task-1/tfmn_dataset.csv` | 196,000 | 13 | CSV |
| Call 2 | `data/processed/validations/task-2/call2_dataset.csv` | 1,287,402 | 6 | CSV |
| Call 4 | `data/processed/validations/task-4_accuracy/call4.pkl` | 503,766 | 9 | Pickle |
| Demographics | `data/processed/demographics/persona_dataset.csv` | 27,543 | 28 | CSV |

### Call 1 — `tfmn_dataset.csv`

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | `str` | Model identifier |
| `run_id` | `str` | Unique run identifier |
| `mode` | `str` | `"human"` or `"llm"` |
| `question_number` | `str` | Full question text (see `MAPPING_CALL1_QUESTIONS` in `src/mathanx/constants.py`) |
| `answer_text` | `str` | Free-text answer |
| `z_scores_anger` | `float64` | Emotion z-score |
| `z_scores_trust` | `float64` | Emotion z-score |
| `z_scores_surprise` | `float64` | Emotion z-score |
| `z_scores_disgust` | `float64` | Emotion z-score |
| `z_scores_joy` | `float64` | Emotion z-score |
| `z_scores_sadness` | `float64` | Emotion z-score |
| `z_scores_fear` | `float64` | Emotion z-score |
| `z_scores_anticipation` | `float64` | Emotion z-score |

The 7 open-ended questions (numbered 1–7) are:

1. *What is your relationship with mathematics?*
2. *Do you ever get anxious when thinking about mathematics?*
3. *Did you ever use AI to support your math learning in the last year? If yes, how was your experience?*
4. *How would you explain, step by step, how to solve a second order algebraic equation?*
5. *How would you explain, step by step, how to find the stationary points of an equation y = f(x)?*
6. *Briefly, how do you perform a Principal Component Analysis? Should I get anxious about its mathematics? Please, teach me.*
7. *According to you, how can LLMs be used to innovate math learning in schools and universities?*

### Call 2 — `call2_dataset.csv`

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `mode` | `str` | `"human"` or `"llm"` |
| `Model` | `str` | Raw model folder name |
| `scale` | `str` | One of `"maes"`, `"amas"`, `"mseaq"` |
| `item number` | `object` | Item identifier (parsed to numeric in the pooling script) |
| `rating` | `int64` | Likert rating (1–5) |

**Scale composition:** MAES = 9 items, AMAS = 9 items, MSEAQ = 28 items (46 items total).

The raw `item number` field contains string values; the pooling script extracts the first numeric sequence and filters to items in range 1–28. Non-numeric entries are dropped.

### Call 4 — `call4.pkl`

| Column | Type | Description |
|--------|------|-------------|
| `persona` | `object` | Full persona dictionary (JSON) |
| `gender` | `str` | Extracted gender |
| `run_id` | `str` | Unique run identifier |
| `model` | `str` | Model identifier |
| `mode` | `str` | `"human"` or `"llm"` |
| `chosen_option` | `str` | Answer (A/B/C/D/E) |
| `reasoning` | `str` | Free-text reasoning |
| `confidence_score` | `int64` | Confidence rating (1–5) |
| `question_number` | `str` | Question number (1–18; spurious values filtered out) |

The pooling script filters to questions 1–18 only. The correct answers for accuracy computation are defined in `MSESR_CORRECT_ANSWERS` in `src/mathanx/constants.py`.

### Demographics — `persona_dataset.csv`

Contains the synthetic persona data for every `human`-mode run. 27,543 rows × 28 columns including: `age`, `gender`, `sexual_orientation`, `city_of_living`, `employment_status`, `education_level`, `marital_status`, `children`, `migration_status`, `religious_beliefs`, `parent_1_education`, `parent_2_education`, `hobbies`, `fav_subjects`, `hat_subjects`, and 10 OCEAN Big Five columns (score + level for each of openness, conscientiousness, extraversion, agreeableness, neuroticism).

---

## Construction Pipeline

The notebook `misc/dataset_aggregations_pooling_system.py` performs the following steps:

1. **Call 4 — Load and pivot.** Read `call4.pkl`, filter `question_number` to range `1`–`18`, then pivot on `run_id` with question number as columns. Three value columns (`chosen_option`, `reasoning`, `confidence_score`) produce 54 wide columns.

2. **Call 2 — Load, clean, and pivot.** Read `call2_dataset.csv`, extract the first numeric sequence from `item number`, filter to items `1`–`28`, then pivot on `(run_id, Model)` with `(scale, item number)` as the column index. The 46 scale items become 46 wide columns plus the `Model` column.

3. **Call 1 — Load, map, and pivot.** Read `tfmn_dataset.csv`, map `question_number` from full text to integer via `MAPPING_CALL1_QUESTIONS`, then pivot on `(run_id, mode)` with question number as columns. The text, answer, and 8 emotion-score columns produce 77 wide columns.

4. **Demographics — Load and rename.** Read `persona_dataset.csv`, rename `city_of_living → city` and `education_level → education`, drop `mode` and `model_name`.

5. **Merge.** Inner-join the three pivoted Call datasets on `run_id`, then left-join the demographics table on `run_id`. Demographics are only available for `human`-mode runs.

6. **Export.** Write the merged table to `data/processed/validations/df_pooling_system.parquet`.

---

## Schema

The 205 columns group as follows:

### Identifiers (2 columns)

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `mode` | `str` | `"human"` (persona-driven) or `"llm"` (raw LLM) |

### Call 1 — TFMN Open-Ended Questions (77 columns)

| Pattern | Count | Type | Description |
|---------|-------|------|-------------|
| `question_text_{1..7}` | 7 | `str` | Full question text |
| `answer_text_{1..7}` | 7 | `str` | Free-text answer |
| `z_scores_anger_{1..7}` | 7 | `float64` | Anger emotion z-score |
| `z_scores_trust_{1..7}` | 7 | `float64` | Trust emotion z-score |
| `z_scores_surprise_{1..7}` | 7 | `float64` | Surprise emotion z-score |
| `z_scores_disgust_{1..7}` | 7 | `float64` | Disgust emotion z-score |
| `z_scores_joy_{1..7}` | 7 | `float64` | Joy emotion z-score |
| `z_scores_sadness_{1..7}` | 7 | `float64` | Sadness emotion z-score |
| `z_scores_fear_{1..7}` | 7 | `float64` | Fear emotion z-score |
| `z_scores_anticipation_{1..7}` | 7 | `float64` | Anticipation emotion z-score |
| `question_number_{1..7}` | 7 | `int64` | Integer question ID (1–7) |

**Total:** 77 columns (9 patterns × 7 questions, but `question_text` and `answer_text` have no z-score prefix)

### Call 2 — Psychometric Scales (47 columns)

| Column | Type | Description |
|--------|------|-------------|
| `Model` | `str` | Model identifier (e.g., `MANX_LLM_MistralSmall4`) |
| `maes_{1..9}` | `float64` | MAES item rating (1–5) |
| `amas_{1..9}` | `float64` | AMAS item rating (1–5) |
| `mseaq_{1..28}` | `float64` | MSEAQ item rating (1–5) |

### Call 4 — MSESR Problem Solving (54 columns)

| Pattern | Count | Type | Description |
|---------|-------|------|-------------|
| `chosen_option_{1..18}` | 18 | `str` | Chosen answer (A/B/C/D/E) |
| `reasoning_{1..18}` | 18 | `str` | Free-text reasoning for each problem |
| `confidence_score_{1..18}` | 18 | `int64` / `float64` | Self-reported confidence (1–5) |

### Demographics (20 columns)

| Column | Type | Description |
|--------|------|-------------|
| `age` | `float64` | Age (years) |
| `gender` | `str` | Gender identity |
| `sexual_orientation` | `str` | Sexual orientation |
| `city` | `str` | City of living |
| `employment_status` | `str` | Employment status |
| `education` | `str` | Education level |
| `marital_status` | `str` | Marital status |
| `children` | `float64` | Number of children |
| `migration_status` | `str` | Migration status |
| `religious_beliefs` | `str` | Religious beliefs |
| `parent_1_education` | `str` | First parent's education level |
| `parent_2_education` | `str` | Second parent's education level |
| `hobbies` | `str` | Hobbies (semicolon-separated list) |
| `fav_subjects` | `str` | Favourite school subjects (semicolon-separated list) |
| `hat_subjects` | `str` | Least-liked school subjects (semicolon-separated list) |

### OCEAN Big Five (10 columns)

| Column | Type | Description |
|--------|------|-------------|
| `ocean_openness_score` | `float64` | Openness to experience score |
| `ocean_openness_level` | `str` | `low` / `moderate` / `high` |
| `ocean_conscientiousness_score` | `float64` | Conscientiousness score |
| `ocean_conscientiousness_level` | `str` | `low` / `moderate` / `high` |
| `ocean_extraversion_score` | `float64` | Extraversion score |
| `ocean_extraversion_level` | `str` | `low` / `moderate` / `high` |
| `ocean_agreeableness_score` | `float64` | Agreeableness score |
| `ocean_agreeableness_level` | `str` | `low` / `moderate` / `high` |
| `ocean_neuroticism_score` | `float64` | Neuroticism score |
| `ocean_neuroticism_level` | `str` | `low` / `moderate` / `high` |

---

## Data Quality Notes

- **6,983 rows** have `NaN` for all demographics columns (`age`, `gender`, `sexual_orientation`, etc.). These correspond to `llm`-mode runs, for which no synthetic persona was generated. The demographics table was left-joined on `run_id`, so `llm` runs receive no demographic record.
---

## Usage

```python
import pandas as pd

df = pd.read_parquet("data/processed/validations/df_pooling_system.parquet")
```
---

## File Format

The dataset is stored as Apache Parquet with snappy compression at:

```
data/processed/validations/df_pooling_system.parquet
```

---

# Edge List Wide Dataset

## Overview

The **Edge List Wide Dataset** (`data/processed/edge_list_wide.csv`) is a single wide-format CSV that pivots the Call 3 Forma Mentis semantic-network data from edge-list (long) form into a run_id-level table. Each row represents one experimental run.

It was produced by `misc/extract_edge_list_wide.py`.

| Property | Value |
|----------|-------|
| Rows | 27,987 |
| Columns | 352 |
| Format | CSV |
| Size (disk) | ~53 MB |

## Source Data

The raw edge lists reside in `data/processed/NEW_edge_list_individual/`. For each model there is a subdirectory containing one folder per `run_id`, each holding a single `edgelist.csv` with four columns:

| Column | Type | Description |
|--------|------|-------------|
| `cue_word` | `str` | One of 50 cue words |
| `association_word` | `str` | Free-association provided by the LLM |
| `cue_valence` | `int` | Valence of the cue word (−1, 0, +1) |
| `associated_valence` | `int` | Valence of the association word (−1, 0, +1) |

Each `run_id` contains 48–50 cues, each with 1–3 associations (most have all 3).

## Extraction

`misc/extract_edge_list_wide.py` iterates over every model folder and every `run_id` folder inside it, reads the `edgelist.csv`, and pivots from long to wide:

1. Group rows by `cue_word` (preserving row order within each cue).
2. For each cue, up to 3 associations and their corresponding associated valences are placed in numbered columns (`association_1` … `association_3`).
3. The shared `cue_valence` is stored in a single column.
4. Cues with fewer than 3 associations receive `NaN` in the missing slots.
5. All rows are concatenated and written to a single CSV.

## Schema — 352 columns

### Identifiers (2 columns)

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `model` | `str` | Model folder name (e.g. `MANX_LLM_anitamistral`) |

### Per-cue columns (7 columns × 50 cues = 350 columns)

For every cue word `{cue}` the following columns are created:

| Pattern | Count | Type | Description |
|---------|-------|------|-------------|
| `{cue}_association_{1..3}` | 3 | `str` | Association word (`NaN` if fewer than 3) |
| `{cue}_valence` | 1 | `int64` | Cue word valence (same for all associations of this cue) |
| `{cue}_associated_valence_{1..3}` | 3 | `float64` | Per-association valence (`NaN` if fewer than 3) |

### Cue words by category

The 50 cue words, grouped thematically:

| Category | Cues |
|----------|------|
| **Math domain knowledge** | `mathematic`, `equation`, `number`, `theorem`, `proof` |
| **Computational thinking** | `informatic`, `algorithm`, `computation`, `problem - solve`, `variable` |
| **Artificial intelligence** | `ai`, `llm`, `model`, `chatgpt`, `datum` |
| **Academic assessment** | `exam`, `grade`, `homework`, `failure`, `success` |
| **Academic context** | `class`, `lecture`, `study`, `classroom`, `blackboard` |
| **Work context** | `job`, `career`, `work`, `society`, `future` |
| **STEM fields** | `stem`, `science`, `physics`, `chemistry`, `biology` |
| **Non-STEM fields** | `art`, `music`, `literature`, `history`, `philosophy` |
| **Skills** | `creativity`, `experiment`, `logic`, `anxiety`, `teamwork` |
| **Actors** | `professor`, `teacher`, `student`, `knowledge`, `scientist` |

## Usage

```python
import pandas as pd

df = pd.read_csv("data/processed/edge_list_wide.csv")
```

---

## File Format

The dataset is stored as a plain CSV file at:

```
data/processed/edge_list_wide.csv
```
