# Underdog Bias in LLM-Based Dispute Resolution: Replication Materials

This repository reproduces every table, figure, and in-text estimate of the paper from
the frozen model decisions, and contains the code that builds the four samples,
constructs the exact prompts of every experimental condition, calls the four models, and
reads their decisions.

The paper compares large language models' decisions with the real outcomes of four
dispute settings: retail marketplace disputes (VerdictBench), housing complaints
(Housing Ombudsman Service), labor-service disputes (CPEE), and divorce-property
disputes (China Judgments Online). No case text is included; see
[DATA_LICENSES.md](DATA_LICENSES.md).

## Three levels of replication

| Level | What it reproduces | Needs |
|---|---|---|
| 1. Tables and figures | All tables (main text and appendices), Figure 1, and the numbers reported in the text, from the frozen decisions in this repository | Python only (about 10 seconds) |
| 2. Samples and prompts | The four constructed samples from the public source data, and the exact prompt of every unit and condition | The source data (`data/download/`) |
| 3. Model calls | Re-sending the prompts to the four models and reading their decisions | API keys for the four providers |

## Level 1: reproduce the tables and figures

```bash
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` writes every table to `output/tables/` (CSV and LaTeX), Figure 1 to
`output/figures/`, and the in-text numbers to `output/inline_values.json`. It then
compares every printed estimate, standard error, and significance marker with the paper
(`expected/paper_tables.json`, `expected/inline_values.json`), writes the comparison to
`output/check_report.txt`, and exits with an error if any value differs.

| Paper | Output file |
|---|---|
| Table 1 (samples) | `output/tables/data_overview.csv` |
| Table 2 | `output/tables/reversals.*` |
| Figure 1 | `output/figures/xianyu_vote_split.*` |
| Table 3 (heterogeneity) | `output/tables/boundary_conditions.*` |
| Table 4 (advisory opinions) | `output/tables/human_feedback_anchor.*` |
| Table 5 (identity masking) | `output/tables/identity_masking.*` |
| Table 6 (vulnerability cues) | `output/tables/social_identity_tags.*` |
| Table 7 (pretrained vs. post-trained) | `output/tables/base_post.*` |
| Table 8 (debiasing instruction) | `output/tables/debias_instruction.*` |
| Table 9 (fine-tuning) | `output/tables/targeted_sft.*` |
| Table A.1 (baseline estimates) | `output/tables/full_overturns.*` |
| Tables A.2–A.4 (heterogeneity) | `heterogeneity_consensus.*`, `heterogeneity_labor.*`, `heterogeneity_length_other.*` |
| Table A.5 (advisory opinions) | `opinion_details.*` |
| Tables A.6–A.7 (identity masking) | `identity_masking_full.*`, `mask_details.*` |
| Table A.8 (debiasing instruction) | `debias_instruction_full.*` |
| Table B.1 (vulnerability cues) | `social_cue_regressions.*` |
| Table B.2 (divorce-property masking) | `appendix_divorce_mask_robustness.*` |
| Tables B.3–B.4 (post-training) | `base_post_full.*`, `targeted_sft_full.*` |

The estimators are in `src/analysis/estimators.py`: weighted least squares with standard
errors clustered by dispute (retail marketplace), report (housing), or source case
(labor, divorce property), using the G/(G−1) correction and two-sided t(G−1) tests;
unit fixed effects are absorbed by the within transformation; equal-asymmetry tests are
joint Wald tests. `src/analysis/tables.py` builds each table.

### Data files

`data/derived/units_<setting>.csv`: one row per adjudication unit of the full
constructed sample.

| Column | Meaning |
|---|---|
| `unit_id` | Dispute (marketplace), complaint head (housing), plea (labor), or claim (divorce property) identifier |
| `cluster_id` | Inference cluster: dispute, report, or source case |
| `weight` | 1, except housing: 1 / number of complaint heads in the report |
| `real_underdog` | 1 if the real outcome favors the underdog (buyer, resident, employee, female spouse) |
| other columns | Numeric features used in the analyses (juror votes, record length in non-whitespace characters, litigation role, and so on) |

`decisions/decisions_<setting>.csv`: one row per unit, model, and condition.

| Column | Meaning |
|---|---|
| `model` | `Qwen`, `DeepSeek`, `Luna`, `Haiku 5.5` |
| `condition` | `C0` (baseline); `OPINION_U` / `OPINION_S` (advisory opinion supporting the underdog / the stronger party); `MASK`; `DEBIAS`; `CUE_CARE_U`, `CUE_CARE_S`, `CUE_DIS_U`, `CUE_DIS_S` (caregiving or disability cue assigned to the underdog or the stronger party) |
| `underdog_support` | 1 if the decision favors the underdog |
| `fully_formatted` | 1 if every auxiliary output field followed the requested format; the decision is read from the unique explicit decision in either case |

`posttraining/`: the pretrained and post-trained Qwen3.5-9B decisions from conditional
label scoring (`base_post_<setting>.csv`, with the label log-probability difference `D`
and exact ties), the fine-tuning splits and held-out test decisions
(`sft_<setting>.csv`), and the fine-tuning configuration (`sft_config.json`).

## Level 2: rebuild the samples and prompts

Obtain the source data as described in `data/download/<setting>.md` and keep it outside
version control (for example under `data/raw/`, which is git-ignored).

- **Retail marketplace** (`src/build/marketplace.py`): `build_units(raw_dir)` applies the
  screening chain of Appendix C.1 and returns the sample; `load_records(raw_dir,
  unit_ids, image_dir=...)` returns each dispute's pre-adjudication record and copies its
  images.
- **Labor** (`src/build/labor.py`): `python -m src.build.labor <raw_dir> [units.csv]`
  prints the screening counts of Appendix C.3 and rebuilds the sample.
- **Divorce property** (`src/build/divorce.py`): `python -m src.build.divorce <raw_dir>
  --out units.csv` rebuilds the claim sample of Appendix C.4 from the annual archives.
- **Housing**: the sample is identified by the report and complaint-head IDs in
  `data/derived/units_housing.csv`; the decisions are public on the Housing Ombudsman
  Service website.

`src/prompts/<setting>.py` holds the exact prompt text sent to the models (Chinese for
the retail marketplace, labor, and divorce property; English for housing).
`build_request(record, condition)` returns the system message and the ordered user-message
parts for any unit and condition. Appendix D of the paper shows the same prompts in
English translation.

## Level 3: re-run the model calls

```bash
pip install -r requirements.txt -r requirements-run.txt
export DASHSCOPE_API_KEY=...  DEEPSEEK_API_KEY=...  OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...
python -m src.run.run --setting labor --model qwen --condition C0 --raw data/raw/labor --dry-run
python -m src.run.run --setting labor --model qwen --condition C0 --raw data/raw/labor
```

Each unit is sent as one independent request. Raw replies are stored in `runs/`
(git-ignored) and the decisions are written in the format of `decisions/`.

| Model | Model ID | Configuration |
|---|---|---|
| Qwen | `qwen3.5-flash` | temperature 0; thinking disabled |
| DeepSeek | `deepseek-flash` | temperature 0; thinking disabled |
| Luna | `gpt-5.6-luna` | temperature 0; `reasoning_effort=none` |
| Haiku 5.5 | `claude-haiku-5-5` | no temperature, `top_p`, or `top_k`; thinking disabled; medium effort |

Maximum output length is 3,072 tokens for Qwen, DeepSeek, and Luna and 4,096 for Haiku 5.5
(housing: 768 and 1,024). Retail marketplace requests contain text and images; the other
settings are text only.

## Repository layout

```
reproduce.py            Level 1 entry point
expected/               Values printed in the paper, used by reproduce.py
data/derived/           Samples: identifiers, real outcomes, numeric features
data/download/          How to obtain each source dataset
decisions/              Frozen model decisions
posttraining/           Pretrained/post-trained and fine-tuning results
src/analysis/           Estimators, tables, Figure 1
src/build/              Sample construction from the source data
src/prompts/            Prompt templates and request construction
src/parse/              Reading decisions from model responses
src/run/                Model clients and the run command
```
