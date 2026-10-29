"""'Ask the Portfolio': grounded natural-language Q&A.

The LLM never sees row-level data and never produces numbers on its own.

  1. PLAN    question + schema (column names, allowed values, metric
             definitions)  ->  LLM  ->  JSON query spec, or a refusal.
  2. VALIDATE the spec against a whitelist (metrics, dimensions, filter
             columns, operators, values). Anything else is declined.
  3. EXECUTE the spec with pandas on the clean dataset -> exact result table.
  4. NARRATE question + result table (aggregates only, max 30 rows) -> LLM
             -> 2-3 sentence answer.
  5. VERIFY  every number in the answer must appear in the result table;
             otherwise the answer is replaced by a deterministic template.

Without an API key, step 1 falls back to a keyword parser and step 4 to the
template, so the feature degrades instead of breaking.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .llm import LLM, LLMError, parse_json

SOURCE = "data/processed/portfolio.parquet (clean account-level dataset built by src/build.py)"

METRICS = {
    "default_rate": ("Default rate", "share of accounts with target_default = 1", "pct"),
    "defaults": ("Defaulted accounts", "count of accounts with target_default = 1", "int"),
    "accounts": ("Accounts", "number of loan accounts", "int"),
    "customers": ("Customers", "number of distinct customers", "int"),
    "exposure": ("Exposure", "sum of outstanding balance after month 12, USD", "usd"),
    "defaulted_exposure": ("Defaulted exposure", "outstanding balance of defaulted accounts, USD", "usd"),
    "exposure_share": ("Share of exposure", "exposure of the group (after filters) / exposure of the whole portfolio", "pct"),
    "total_loan_amount": ("Total loan amount", "sum of original loan amount, USD", "usd"),
    "avg_loan_amount": ("Average loan amount", "mean original loan amount, USD", "usd"),
    "avg_interest_rate": ("Average interest rate", "mean annual interest rate, %", "rate"),
    "avg_income": ("Average income", "mean annual income of the borrower, USD", "usd"),
    "avg_pd": ("Average model PD", "mean out-of-fold probability of default from the champion model", "pct"),
    "anomaly_count": ("Flagged anomalies", "accounts flagged by anomaly detection", "int"),
    "missed_payment_rate": ("Accounts with a missed EMI", "share of accounts with 1+ missed payment in 12 months", "pct"),
    "avg_dpd": ("Average DPD", "mean days past due over the 12 months of history", "num"),
}

DIMENSIONS = {
    "loan_type": "Loan type", "loan_purpose": "Loan purpose", "employment_type": "Employment type",
    "income_band": "Income band", "age_band": "Age band", "region": "Region", "city_tier": "City tier",
    "housing_type": "Housing type", "education_level": "Education", "marital_status": "Marital status",
    "gender": "Gender", "disbursal_channel": "Disbursal channel", "risk_band": "Model risk band",
    "origination_year": "Origination year", "occupation_sector": "Occupation sector",
    "thin_file": "No bureau history (1 = yes)", "anomaly_flag": "Anomaly flagged",
    "collateral_flag": "Secured (1 = yes)",
}
NUMERIC_FILTERS = {
    "annual_income": "USD", "age": "years", "loan_amount": "USD", "pd": "0-1",
    "interest_rate_pct": "%", "exposure": "USD", "bur_n_written_off": "count", "rep_n_missed": "count",
}
OPS = {"eq", "ne", "in", "gt", "gte", "lt", "lte"}


@dataclass
class Answer:
    status: str                      # "answered" | "declined" | "error"
    text: str
    table: pd.DataFrame | None = None
    spec: dict | None = None
    scope: str = ""
    planner: str = ""
    narrator: str = ""
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------ schema
def schema_text(df: pd.DataFrame) -> str:
    lines = ["METRICS (use these names exactly):"]
    lines += [f"- {k}: {v[1]}" for k, v in METRICS.items()]
    lines.append("\nDIMENSIONS for group_by or equality filters, with allowed values:")
    for k, v in DIMENSIONS.items():
        vals = sorted(map(str, df[k].dropna().unique()))
        lines.append(f"- {k} ({v}): {vals}")
    lines.append("\nNUMERIC FILTER COLUMNS (ops gt/gte/lt/lte/eq): " + ", ".join(f"{k} [{u}]" for k, u in NUMERIC_FILTERS.items()))
    lines.append(f"\nThe dataset has {len(df)} loan accounts, one row per account, synthetic data, snapshot 2025-12-31. "
                 "Repayment history covers months 1-12 after origination. Account ids look like ACC200001.")
    return "\n".join(lines)


PLANNER_SYSTEM = """You translate a risk analyst's question into a JSON query over a loan-portfolio table.
You never answer the question yourself and never invent numbers.

Return ONLY a JSON object with these keys:
{"answerable": true|false,
 "reason": "<if not answerable: one sentence on why, and what the data can answer instead>",
 "intent": "aggregate" | "account_lookup",
 "metrics": ["<1-3 metric names>"],
 "group_by": ["<0-2 dimension names>"],
 "filters": [{"column": "<dimension or numeric column>", "op": "eq|ne|in|gt|gte|lt|lte", "value": <value or list>}],
 "sort_by": "<one of the metrics or null>", "sort": "desc"|"asc", "top_n": <int or null>,
 "account_id": "<ACC... for account_lookup, else null>"}

Rules:
- Use only metric, dimension and column names and category values from the schema. Match values exactly.
- "Which X has the highest Y" -> group_by [X], sort_by Y, sort desc (do not set top_n so the full ranking is shown).
- Set answerable=false for anything the table cannot answer: forecasts or future values, causes ("why"),
  external facts (interest-rate outlook, competitors, news), personal identity data (names, phone, address),
  advice or opinions, columns that do not exist, or anything unrelated to this portfolio.
- Questions about a single account id -> intent account_lookup.
- Exact thresholds on age, income, loan amount, PD, interest rate or exposure ("under 30", "over $100k") use the
  NUMERIC FILTER COLUMNS with gt/gte/lt/lte. The *_band dimensions are only for grouping.
- "Share of exposure" for a subset -> metric exposure_share with the subset as filters (share of the whole portfolio).

SCHEMA
"""

NARRATOR_SYSTEM = """You write a short answer (2-3 plain sentences) for a credit-risk analyst.
Use ONLY the numbers in the RESULT table, copied exactly as formatted there. Do not compute new numbers,
do not round differently, do not add percentages or differences that are not in the table.
Name the top item(s) and give the exact figures. If a group has very few accounts (under 30), say the figure is
based on few accounts. Do not mention the query, JSON or the table itself. No preamble."""


# ------------------------------------------------------------------ validation
class SpecError(ValueError):
    pass


def validate(spec: dict, df: pd.DataFrame) -> dict:
    if not isinstance(spec, dict):
        raise SpecError("spec is not an object")
    if not spec.get("answerable", False):
        return {"answerable": False, "reason": str(spec.get("reason") or "The data cannot answer this.")}
    intent = spec.get("intent", "aggregate")
    if intent == "account_lookup":
        acc = str(spec.get("account_id") or "").upper().strip()
        if not re.fullmatch(r"ACC\d+", acc):
            raise SpecError("account_lookup needs a valid account_id")
        return {"answerable": True, "intent": "account_lookup", "account_id": acc}
    metrics = [m for m in (spec.get("metrics") or []) if m in METRICS][:3]
    if not metrics:
        raise SpecError(f"no valid metric in {spec.get('metrics')}")
    group_by = [g for g in (spec.get("group_by") or []) if g in DIMENSIONS][:2]
    if len(group_by) != len(spec.get("group_by") or []):
        raise SpecError(f"unknown group_by {spec.get('group_by')}")
    filters = []
    for f in spec.get("filters") or []:
        col, op, val = f.get("column"), f.get("op", "eq"), f.get("value")
        if op not in OPS:
            raise SpecError(f"bad operator {op}")
        if col in DIMENSIONS:
            allowed = set(map(str, df[col].dropna().unique()))
            vals = val if isinstance(val, list) else [val]
            vals = [str(int(v)) if isinstance(v, (float, bool)) and col in ("thin_file", "collateral_flag", "origination_year") else str(v) for v in vals]
            if col == "anomaly_flag":
                vals = ["True" if v.lower() in ("true", "1", "yes") else "False" for v in vals]
            bad = [v for v in vals if v not in allowed]
            if bad:
                raise SpecError(f"value {bad} not found in {col}")
            if op in ("gt", "gte", "lt", "lte") and col != "origination_year":
                raise SpecError(f"operator {op} not valid for {col}")
            filters.append({"column": col, "op": op, "value": vals if op in ("in",) or len(vals) > 1 else vals[0]})
            if len(vals) > 1:
                filters[-1]["op"] = "in" if op in ("eq", "in") else "not_in"
        elif col in NUMERIC_FILTERS:
            try:
                v = float(val)
            except (TypeError, ValueError):
                raise SpecError(f"numeric value needed for {col}")
            if op == "in":
                raise SpecError("'in' not valid for numeric columns")
            filters.append({"column": col, "op": op, "value": v})
        else:
            raise SpecError(f"unknown filter column {col}")
    sort_by = spec.get("sort_by") if spec.get("sort_by") in metrics else (metrics[0] if group_by else None)
    sort = "asc" if spec.get("sort") == "asc" else "desc"
    top_n = spec.get("top_n")
    top_n = int(top_n) if isinstance(top_n, (int, float)) and top_n and top_n > 0 else None
    return {"answerable": True, "intent": "aggregate", "metrics": metrics, "group_by": group_by,
            "filters": filters, "sort_by": sort_by, "sort": sort, "top_n": top_n}


# ------------------------------------------------------------------ execution
def _apply_filters(df, filters):
    m = pd.Series(True, index=df.index)
    for f in filters:
        col, op, v = f["column"], f["op"], f["value"]
        s = df[col]
        if col in DIMENSIONS:
            s = s.astype(str)
            if col == "origination_year" and op in ("gt", "gte", "lt", "lte"):
                s, v = df[col].astype(float), float(v)
        if op == "eq":
            m &= s == v
        elif op == "ne":
            m &= s != v
        elif op == "in":
            m &= s.isin(v)
        elif op == "not_in":
            m &= ~s.isin(v)
        elif op == "gt":
            m &= s > v
        elif op == "gte":
            m &= s >= v
        elif op == "lt":
            m &= s < v
        elif op == "lte":
            m &= s <= v
    return df[m]


def _metric(g: pd.DataFrame, name: str, total_exposure: float):
    if name == "default_rate":
        return g.target_default.mean()
    if name == "defaults":
        return g.target_default.sum()
    if name == "accounts":
        return len(g)
    if name == "customers":
        return g.customer_id.nunique()
    if name == "exposure":
        return g.exposure.sum()
    if name == "defaulted_exposure":
        return g.loc[g.target_default == 1, "exposure"].sum()
    if name == "exposure_share":
        return g.exposure.sum() / total_exposure if total_exposure else np.nan
    if name == "total_loan_amount":
        return g.loan_amount.sum()
    if name == "avg_loan_amount":
        return g.loan_amount.mean()
    if name == "avg_interest_rate":
        return g.interest_rate_pct.mean()
    if name == "avg_income":
        return g.annual_income.mean()
    if name == "avg_pd":
        return g.pd.mean()
    if name == "anomaly_count":
        return g.anomaly_flag.sum()
    if name == "missed_payment_rate":
        return (g.rep_n_missed > 0).mean()
    if name == "avg_dpd":
        return g.rep_mean_dpd.mean()
    raise SpecError(name)


def fmt(v, kind):
    if pd.isna(v):
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "usd":
        return f"${v / 1e6:,.1f}M" if abs(v) >= 1e6 else f"${v:,.0f}"
    if kind == "rate":
        return f"{v:.2f}%"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:.1f}"


def execute(spec: dict, df: pd.DataFrame):
    """Returns (raw table, formatted table, scope description)."""
    sub = _apply_filters(df, spec["filters"])
    if sub.empty:
        return None, None, "no accounts match the filters"
    tot = df.exposure.sum()  # share is of the whole portfolio in scope, before the question's filters
    metrics = list(spec["metrics"])
    cols = metrics + ([] if "accounts" in metrics else ["accounts"])
    if spec["group_by"]:
        rows = []
        for key, g in sub.groupby(spec["group_by"], observed=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(spec["group_by"], map(str, key))), **{m: _metric(g, m, tot) for m in cols}})
        raw = pd.DataFrame(rows)
        raw = raw.sort_values(spec["sort_by"], ascending=spec["sort"] == "asc", kind="stable").reset_index(drop=True)
        if spec["top_n"]:
            raw = raw.head(spec["top_n"])
    else:
        raw = pd.DataFrame([{m: _metric(sub, m, tot) for m in cols}])
    show = raw.copy()
    for m in cols:
        show[m] = raw[m].map(lambda v, k=METRICS[m][2]: fmt(v, k))
    show = show.rename(columns={**{m: METRICS[m][0] for m in cols}, **DIMENSIONS})
    return raw, show, describe_scope(spec, len(sub))


def describe_scope(spec, n):
    if not spec["filters"]:
        f = "all accounts"
    else:
        parts = []
        sym = {"eq": "=", "ne": "!=", "in": "in", "not_in": "not in", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        for x in spec["filters"]:
            parts.append(f"{x['column']} {sym[x['op']]} {x['value']}")
        f = "; ".join(parts)
    return f"{n:,} accounts ({f})"


def account_lookup(acc: str, df: pd.DataFrame, drivers: pd.DataFrame | None):
    row = df[df.account_id == acc]
    if row.empty:
        return None, None
    r = row.iloc[0]
    rec = {
        "Account": acc, "Customer": r.customer_id, "Loan type": r.loan_type,
        "Loan amount": fmt(r.loan_amount, "usd"), "Exposure": fmt(r.exposure, "usd"),
        "Model PD": fmt(r.pd, "pct"), "Risk band": r.risk_band,
        "Defaulted (target)": "Yes" if r.target_default == 1 else "No",
        "Missed EMIs (12m)": int(r.rep_n_missed), "Anomaly flagged": "Yes" if r.anomaly_flag else "No",
    }
    if drivers is not None:
        from .model import top_drivers
        drv = top_drivers(drivers[drivers.account_id == acc].iloc[0], r, k=3)
        rec["Top drivers"] = "; ".join(f"{d['feature']} ({d['direction']})" for d in drv)
    t = pd.DataFrame([rec]).T.reset_index()
    t.columns = ["Field", "Value"]
    return t, rec


# ------------------------------------------------------------------ keyword fallback planner
_METRIC_WORDS = [
    (r"defaulted exposure|exposure .*default", "defaulted_exposure"),
    (r"share of exposure|exposure share|concentrat", "exposure_share"),
    (r"default rate|default|bad rate", "default_rate"),
    (r"exposure|outstanding|balance", "exposure"),
    (r"interest|rate of interest", "avg_interest_rate"),
    (r"\bpd\b|probability|risk score|predicted", "avg_pd"),
    (r"anomal|flagged", "anomaly_count"),
    (r"average loan|loan size|avg loan|mean loan", "avg_loan_amount"),
    (r"total loan|disbursed", "total_loan_amount"),
    (r"average income|avg income|mean income", "avg_income"),
    (r"missed", "missed_payment_rate"),
    (r"\bdpd\b|days past due", "avg_dpd"),
    (r"how many customers|number of customers|customers", "customers"),
    (r"how many|number of|count", "accounts"),
]
_DIM_WORDS = [
    (r"loan type|product|loan types", "loan_type"), (r"purpose", "loan_purpose"),
    (r"employment", "employment_type"), (r"income", "income_band"), (r"\bage\b|age band|age group", "age_band"),
    (r"region", "region"), (r"city tier|tier", "city_tier"), (r"housing", "housing_type"),
    (r"education", "education_level"), (r"marital", "marital_status"), (r"gender", "gender"),
    (r"channel", "disbursal_channel"), (r"risk band|risk segment|segment", "risk_band"),
    (r"year|vintage|cohort", "origination_year"), (r"sector|industry", "occupation_sector"),
]
_DECLINE = r"\b(why|predict|forecast|next (year|month|quarter)|will|should|recommend|name|phone|address|email|competitor|news)\b"


def keyword_plan(q: str, df: pd.DataFrame) -> dict:
    ql = q.lower()
    acc = re.search(r"acc\d{4,}", ql)
    if acc:
        return {"answerable": True, "intent": "account_lookup", "account_id": acc.group(0).upper()}
    if re.search(_DECLINE, ql):
        return {"answerable": False, "reason": "That needs a forecast, a cause or outside information; "
                "this dataset can only describe the current portfolio."}
    metrics = []
    for pat, m in _METRIC_WORDS:
        if re.search(pat, ql) and m not in metrics:
            metrics.append(m)
            ql_ = re.sub(pat, " ", ql)
            ql = ql_
    if not metrics:
        return {"answerable": False, "reason": "I could not match this to a portfolio metric (default rate, "
                "exposure, accounts, PD, interest rate, anomalies)."}
    qd = q.lower()
    filters, used_cols = [], set()
    for col in ["loan_type", "employment_type", "region", "city_tier", "housing_type", "loan_purpose",
                "disbursal_channel", "education_level", "marital_status", "occupation_sector"]:
        for v in df[col].dropna().unique():
            v = str(v)
            stem = v.lower().replace(" loan", "").rstrip("s")
            if re.search(r"\b" + re.escape(stem) + r"s?\b", qd) and len(stem) > 3:
                filters.append({"column": col, "op": "eq", "value": v})
                used_cols.add(col)
    group_by = []
    # several values of one column ("unemployed vs salaried") -> compare them side by side
    for col in list(used_cols):
        vals = [f["value"] for f in filters if f["column"] == col]
        if len(vals) > 1:
            filters = [f for f in filters if f["column"] != col] + [{"column": col, "op": "in", "value": vals}]
            group_by.append(col)
    # numeric thresholds: "under 30", "over 100k", "aged 25-35"
    num = r"\$?(\d+(?:\.\d+)?)\s*(k)?"
    for word, op in [(r"under|below|less than|younger than", "lt"), (r"over|above|more than|older than", "gt")]:
        for m in re.finditer(rf"(?:{word})\s+{num}", qd):
            v = float(m.group(1)) * (1000 if m.group(2) else 1)
            ctx = qd[max(0, m.start() - 40): m.end() + 15]
            if v >= 1000 or re.search(r"income|earn|salary", ctx):
                col = "annual_income" if not re.search(r"loan amount|loan size", ctx) else "loan_amount"
            elif re.search(r"\bage|aged|old|young|borrowers under|years", ctx) or 18 <= v <= 90:
                col = "age"
            else:
                continue
            filters.append({"column": col, "op": op, "value": v})
            if col == "annual_income" and "income_band" not in used_cols:
                used_cols.add("income_band")
            if col == "age":
                used_cols.add("age_band")
    for pat, d in _DIM_WORDS:
        if re.search(pat, qd) and d not in group_by and d not in used_cols:
            group_by.append(d)
    group_by = group_by[:2]
    sort = "asc" if re.search(r"lowest|least|smallest|minimum|best", qd) else "desc"
    return {"answerable": True, "intent": "aggregate", "metrics": metrics[:2], "group_by": group_by,
            "filters": filters, "sort_by": metrics[0], "sort": sort, "top_n": None}


# ------------------------------------------------------------------ narration + verification
def template_answer(spec, raw, show, scope):
    m = spec["metrics"][0]
    label, _, kind = METRICS[m]
    if not spec["group_by"]:
        parts = [f"{METRICS[x][0]}: {fmt(raw.iloc[0][x], METRICS[x][2])}" for x in spec["metrics"]]
        return f"For {scope}: " + "; ".join(parts) + "."
    dims = spec["group_by"]
    name = lambda r: " / ".join(str(r[d]) for d in dims)
    first = raw.iloc[0]
    word = "lowest" if spec["sort"] == "asc" else "highest"
    lab = label.lower().replace(" pd", " PD").replace("dpd", "DPD").replace("emi", "EMI")
    txt = (f"{name(first)} has the {word} {lab} at {fmt(first[m], kind)} "
           f"({int(first['accounts']):,} accounts).")
    if len(raw) > 1:
        last = raw.iloc[-1]
        other = "highest" if word == "lowest" else "lowest"
        txt += f" The {other} is {name(last)} at {fmt(last[m], kind)} ({int(last['accounts']):,} accounts)."
    small = raw[raw.accounts < 30]
    if len(small):
        txt += f" {len(small)} group(s) have fewer than 30 accounts, so treat those figures with care."
    return txt


_NUM = re.compile(r"\$?\d[\d,]*\.?\d*\s?[%MK]?")


def _norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "").rstrip(".")


def allowed_numbers(show: pd.DataFrame, extra=()):
    vals = set()
    for v in show.astype(str).values.ravel().tolist() + [str(e) for e in extra]:
        for t in _NUM.findall(v):
            n = _norm(t)
            vals.add(n)
            vals.add(n.lstrip("$"))
            vals.add(n.rstrip("%"))
    return vals


def verify(text: str, show: pd.DataFrame, question: str) -> list[str]:
    """Return the numbers in `text` that are not in the result table or question."""
    allowed = allowed_numbers(show, extra=re.findall(r"\d[\d,.]*", question) + ["30", "12", "1", "2", "3"])
    bad = []
    for t in _NUM.findall(text):
        n = _norm(t)
        if not n or n in allowed or n.lstrip("$") in allowed or n.rstrip("%") in allowed:
            continue
        bad.append(t.strip())
    return bad


# ------------------------------------------------------------------ main entry
def ask(question: str, df: pd.DataFrame, drivers: pd.DataFrame | None = None, llm: LLM | None = None) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer("declined", "Please type a question.")
    llm = llm or LLM()
    notes = []

    # 1. plan
    planner = "keyword parser (no LLM key)"
    if llm.available:
        try:
            raw_spec = parse_json(llm.complete(PLANNER_SYSTEM + schema_text(df), f"QUESTION: {question}", json_mode=True))
            planner = f"LLM ({llm.provider} / {getattr(llm, 'last_model', llm.model)})"
        except (LLMError, json.JSONDecodeError) as e:
            notes.append(f"LLM planner failed ({e}); used keyword parser instead.")
            raw_spec = keyword_plan(question, df)
            planner = "keyword parser (LLM unavailable)"
    else:
        raw_spec = keyword_plan(question, df)

    # 2. validate
    try:
        spec = validate(raw_spec, df)
    except SpecError as e:
        return Answer("declined", f"I can't answer that from the portfolio data ({e}). Try rephrasing with a metric "
                      "such as default rate, exposure or accounts, and a dimension such as loan type or region.",
                      spec=raw_spec, planner=planner, notes=notes)
    if not spec["answerable"]:
        return Answer("declined", f"I can't answer that from this dataset. {spec['reason']}", spec=raw_spec,
                      planner=planner, notes=notes)

    # 3. execute
    if spec["intent"] == "account_lookup":
        show, rec = account_lookup(spec["account_id"], df, drivers)
        if show is None:
            return Answer("declined", f"Account {spec['account_id']} is not in the portfolio.", spec=spec, planner=planner)
        scope = f"1 account ({spec['account_id']})"
        default_text = (f"{rec['Account']} is a {rec['Loan type']} with exposure {rec['Exposure']}, model PD "
                        f"{rec['Model PD']} (band {rec['Risk band']}), {rec['Missed EMIs (12m)']} missed EMIs in 12 months. "
                        f"Anomaly flagged: {rec['Anomaly flagged']}.")
    else:
        raw, show, scope = execute(spec, df)
        if raw is None:
            return Answer("declined", "No accounts match those filters, so there is nothing to report.", spec=spec,
                          planner=planner)
        default_text = template_answer(spec, raw, show, scope)

    # 4. narrate
    text, narrator = default_text, "template"
    if llm.available:
        try:
            table_txt = show.head(30).to_csv(index=False, sep="|", quoting=3)  # no quote marks around numbers
            cand = llm.complete(NARRATOR_SYSTEM, f"QUESTION: {question}\nSCOPE: {scope}\nRESULT:\n{table_txt}",
                                max_tokens=250).strip()
            bad = verify(cand, show.head(30), question)
            if bad:
                notes.append(f"LLM answer contained numbers not in the result ({', '.join(bad)}); "
                             "replaced with the template answer.")
            elif cand:
                text, narrator = cand, (f"LLM ({llm.provider} / {getattr(llm, 'last_model', llm.model)}), "
                                        "numbers verified against the result")
        except LLMError as e:
            notes.append(f"LLM narration failed ({e}); used template answer.")
    return Answer("answered", text, table=show, spec=spec, scope=scope, planner=planner, narrator=narrator, notes=notes)
