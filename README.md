# Credit Risk Portfolio Monitor

A credit-risk monitoring dashboard built from 5 synthetic bank tables (14,685 loan accounts, 10,000 customers), with two
embedded AI features: **Ask the Portfolio** (LLM Q&A grounded in the data) and **anomaly detection** with an analyst
review queue.

- Hosted app: https://credit-risk-monitor-el.streamlit.app/
- Demo video: https://drive.google.com/file/d/1pJHhp0iSjp1QQ_O_xGxhlYcu0t7juxHg/view?usp=sharing

**Python version:** 3.13 locally (the hosted app on Streamlit Community Cloud runs 3.14); every package in `requirements.txt` is pinned.

## Run it (three commands)

```bash
pip install -r requirements.txt
python -m src.build
streamlit run app/app.py
```

`src.build` cleans the raw data, trains both models and writes everything the app reads to `data/processed/`
(about 10 seconds). The processed files are committed too, so the hosted app starts without a build; if they are
missing, the app runs the build on first load. `pytest -q` runs the tests.

## Setting the LLM API key

The app works with any one of three providers and picks the one whose key is set:

| Variable | Provider | Default model |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini (has a free tier) | `gemini-3.5-flash-lite`, falls back to `gemini-3.6-flash` if rate-limited |
| `ANTHROPIC_API_KEY` | Anthropic Claude | `claude-haiku-5-5` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4.1-mini` |

Optional: `LLM_MODEL` overrides the model, `LLM_PROVIDER` forces a provider if more than one key is set.

- **Locally:** copy `.env.example` to `.env` and fill in one key, or copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml`. Both files are git-ignored.
- **Streamlit Community Cloud:** App settings → Secrets, paste for example `GEMINI_API_KEY = "..."`.

**If no key is set** the app still runs. Ask the Portfolio shows a banner, parses questions with a keyword matcher
and answers with a fixed template built from the same computed table, so numbers stay exact; it just understands
fewer phrasings. If the LLM call fails (bad key, quota), the app falls back the same way and says so under the answer.

## Where each item in the brief lives

| Brief item | Where |
|---|---|
| 1. Data ingestion, joins, data-quality decisions | `src/ingest.py` (cleaning + DQ log), `src/features.py` (joins and aggregation). Log in `data/processed/dq_log.csv` and the **Data & method** view |
| 2. Portfolio analytics, 4-5 insights with "so what", cross-cut | `src/analytics.py`; shown at the bottom of the **Portfolio** view; cross-cut heatmap in the same view |
| 3. Default model: LR baseline + ML model, AUC/KS/Gini, SHAP, monitoring | `src/model.py`, `src/monitoring.py`; **Model** and **Monitoring** views; `data/processed/model_results.json` |
| 4. Dashboard: portfolio view, account view, filters, drill-down | `app/app.py` (**Portfolio**, **Account** views; sidebar filters; drill-down table) |
| AI feature 1: Ask the Portfolio | `src/ask.py`, `src/llm.py`; **Ask the Portfolio** view |
| AI feature 2: Anomaly detection + review | `src/anomaly.py`; **Anomaly review** view |
| Trigger / false positive / false negative / human-in-the-loop | Expander at the top of each AI view; `docs/solution_overview.pdf` |
| Architecture diagram | `docs/architecture.pdf` (source: `docs/make_docs.py`) |
| Solution overview | `docs/solution_overview.pdf` |
| Tests | `tests/test_pipeline.py` (ingestion, metrics, grounding guard, provider parsing) |

## Repository layout

```
README.md
requirements.txt          # pinned
.env.example              # key names only
app/app.py                # dashboard entry point
src/
  config.py               # paths, bands, snapshot date
  ingest.py               # load, validate, clean, DQ log
  features.py             # account-level dataset (joins + aggregations)
  analytics.py            # profiles and insights
  model.py                # LR + LightGBM, metrics, SHAP
  monitoring.py           # PSI, cohort AUC/KS, calibration, RAG health check
  anomaly.py              # three anomaly detectors + reasons
  llm.py                  # provider-agnostic LLM client (plain HTTPS)
  ask.py                  # Ask the Portfolio: plan -> validate -> execute -> narrate -> verify
  build.py                # runs the whole pipeline
data/raw/                 # the 5 CSVs as provided + data dictionary
data/processed/           # build outputs read by the app
docs/architecture.pdf
docs/solution_overview.pdf
tests/
```

## Design notes

**Join logic.** One row per loan account. `customers` joins many-to-one on `customer_id`. Repayment (12 months per
account) and bureau lines (many per customer) are aggregated first and then left-joined, with a row-count assertion.
Collections events are summarised for the dashboard only and kept out of the model because they happen after
delinquency.

**Data-quality decisions** (full list in the DQ log): gender M/F mapped to Male/Female; missing years employed,
dependents, residence years and interest rates filled with group medians plus a missing flag; 726 cases of years
employed longer than age minus 16 capped; bureau DPD missing on closed lines set to 0; 3,149 bureau close dates after
the snapshot, 7,293 loans dated before `customer_since` and 3,875 collections events dated before origination are
flagged as data errors and kept, not dropped.

**Model.** Split grouped by customer so the same person is never in train and test. Holdout, 5-fold grouped CV and an
out-of-time test on 2023 originations. Champion chosen on CV AUC (currently the logistic regression, after dropping seven weak or fairness-sensitive fields; see the Model view). Dashboard scores are out-of-fold. Expected loss = PD x LGD x exposure, with LGD assumed per product (`src/config.py`). Bureau features
use the latest snapshot (a monitoring score); a point-in-time variant using only lines opened before origination is
reported as a leakage check.

**Grounding in Ask the Portfolio.** The LLM gets the schema (column names and allowed values) and the question,
never row-level data. It returns a JSON query that is validated against a whitelist and executed with pandas. A second
call phrases the answer from the aggregated result (at most 30 rows). Any number in the answer that is not in the
result table causes the answer to be replaced by a deterministic template. Each answer shows its scope, the result
table and the query spec.

**Anomaly decisions** are stored in `data/processed/anomaly_decisions.csv` and can be downloaded from the app. On
Streamlit Community Cloud the file resets when the app restarts; production would use a database table.
