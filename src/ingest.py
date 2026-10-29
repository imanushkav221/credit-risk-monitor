"""Load the 5 raw tables, fix data-quality issues and log every decision.

Join logic (documented in README and docs):
  customers 1 --< loan_accounts 1 --< repayment_history   (account_id)
  customers 1 --< bureau_credit                            (customer_id)
  loan_accounts 1 --< collections_events                   (account_id)

The analysis grain is one row per loan account. One-to-many tables are
aggregated to account level first (see features.py) and then left-joined onto
loan_accounts, so the row count never changes from 14,685 accounts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class DQLog:
    """Collects one row per data-quality check or decision."""

    def __init__(self):
        self.rows = []

    def add(self, table, check, rows_affected, decision):
        self.rows.append(
            {"table": table, "check": check, "rows_affected": int(rows_affected), "decision": decision}
        )

    def frame(self):
        return pd.DataFrame(self.rows)


def load_raw(raw_dir=C.RAW):
    return {
        "customers": pd.read_csv(raw_dir / "customers.csv", parse_dates=["customer_since"]),
        "loans": pd.read_csv(raw_dir / "loan_accounts.csv", parse_dates=["origination_date"]),
        "bureau": pd.read_csv(raw_dir / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"]),
        "repay": pd.read_csv(raw_dir / "repayment_history.csv"),
        "collections": pd.read_csv(raw_dir / "collections_events.csv", parse_dates=["event_date"]),
    }


def _key_checks(t, log):
    """Primary-key uniqueness and referential integrity."""
    checks = [
        ("customers", "customer_id"),
        ("loans", "account_id"),
        ("bureau", "bureau_id"),
        ("collections", "collection_event_id"),
    ]
    for name, key in checks:
        d = t[name][key].duplicated().sum()
        log.add(name, f"duplicate {key}", d, "none found" if d == 0 else "kept first occurrence")
        if d:
            t[name] = t[name].drop_duplicates(key)
    d = t["repay"].duplicated(["account_id", "payment_month"]).sum()
    log.add("repayment_history", "duplicate account_id + payment_month", d, "none found" if d == 0 else "kept first")
    t["repay"] = t["repay"].drop_duplicates(["account_id", "payment_month"])

    cust = set(t["customers"].customer_id)
    acc = set(t["loans"].account_id)
    for name, key, ref in [
        ("loans", "customer_id", cust),
        ("bureau", "customer_id", cust),
        ("repay", "account_id", acc),
        ("collections", "account_id", acc),
    ]:
        orphans = (~t[name][key].isin(ref)).sum()
        log.add(name, f"orphan {key} (no parent row)", orphans, "none found" if orphans == 0 else "dropped")
        t[name] = t[name][t[name][key].isin(ref)]

    months = t["repay"].groupby("account_id").size()
    log.add(
        "repayment_history",
        "accounts without exactly 12 months of history",
        (months != 12).sum() + len(acc - set(months.index)),
        "none found; every account has months 1-12, so features use a fixed 12-month window",
    )
    no_bureau = len(cust - set(t["bureau"].customer_id))
    log.add(
        "bureau_credit",
        "customers with no bureau record (thin file)",
        no_bureau,
        "kept; bureau counts set to 0 and thin_file=1 so the model can treat them separately",
    )


def clean_customers(c, log):
    c = c.copy()
    n = c.gender.isin(["M", "F"]).sum()
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})
    log.add("customers", "gender coded as M/F instead of Male/Female", n, "mapped M->Male, F->Female")

    miss = c.years_employed.isna()
    c["years_employed_missing"] = miss.astype(int)
    unemp = miss & (c.employment_type == "Unemployed")
    c.loc[unemp, "years_employed"] = 0.0
    med = c.groupby("employment_type").years_employed.transform("median")
    c.loc[miss & ~unemp, "years_employed"] = med[miss & ~unemp]
    log.add(
        "customers",
        "years_employed missing",
        miss.sum(),
        f"{unemp.sum()} unemployed set to 0 (all other unemployed rows are 0); rest filled with "
        "employment-type median; missing flag kept",
    )

    too_long = c.years_employed > (c.age - 16)
    c["years_employed_capped"] = too_long.astype(int)
    c.loc[too_long, "years_employed"] = (c.age - 16).clip(lower=0)[too_long]
    log.add(
        "customers",
        "years_employed longer than age minus 16 (implausible)",
        too_long.sum(),
        "capped at age - 16 and flagged",
    )

    for col in ["num_dependents", "years_at_residence"]:
        miss = c[col].isna()
        c[f"{col}_missing"] = miss.astype(int)
        c[col] = c[col].fillna(c[col].median())
        log.add("customers", f"{col} missing", miss.sum(), "filled with median; missing flag kept")

    c["income_band"] = pd.cut(c.annual_income, C.INCOME_BINS, labels=C.INCOME_LABELS, right=False).astype(str)
    c["age_band"] = pd.cut(c.age, C.AGE_BINS, labels=C.AGE_LABELS, right=True).astype(str)
    return c


def clean_loans(l, c, log):
    l = l.copy()
    miss = l.interest_rate_pct.isna()
    l["interest_rate_missing"] = miss.astype(int)
    med = l.groupby(["loan_type", "tenure_months"]).interest_rate_pct.transform("median")
    med = med.fillna(l.groupby("loan_type").interest_rate_pct.transform("median"))
    l.loc[miss, "interest_rate_pct"] = med[miss]
    log.add(
        "loan_accounts",
        "interest_rate_pct missing",
        miss.sum(),
        "filled with median for the same loan_type and tenure; missing flag kept",
    )

    bad_coll = ((l.loan_type.isin(["Mortgage", "Auto Loan"])) != (l.collateral_flag == 1)).sum()
    log.add("loan_accounts", "collateral_flag inconsistent with loan_type", bad_coll, "none found")

    since = l.merge(c[["customer_id", "customer_since"]], on="customer_id", how="left").customer_since
    before = (l.origination_date < since.values).values
    l["orig_before_customer_since"] = before.astype(int)
    log.add(
        "loan_accounts",
        "loan originated before customer_since date",
        before.sum(),
        "kept and flagged; customer_since is treated as unreliable and not used as a model feature",
    )
    l["origination_year"] = l.origination_date.dt.year
    return l


def clean_bureau(b, log):
    b = b.copy()
    miss = b.days_past_due.isna()
    b["dpd_missing"] = miss.astype(int)
    closed = miss & (b.account_status == "Closed")
    b.loc[closed, "days_past_due"] = 0
    med = b.groupby("account_status").days_past_due.transform("median")
    b.loc[miss & ~closed, "days_past_due"] = med[miss & ~closed]
    log.add(
        "bureau_credit",
        "days_past_due missing",
        miss.sum(),
        f"{closed.sum()} closed lines set to 0 (every reported closed line has DPD 0); "
        "active/written-off filled with status median; missing flag kept",
    )

    asof = pd.Timestamp(C.AS_OF_DATE)
    fut = b.date_closed > asof
    b["future_close_date"] = fut.astype(int)
    log.add(
        "bureau_credit",
        f"date_closed after the {C.AS_OF_DATE} snapshot",
        fut.sum(),
        "status kept as Closed, date flagged as unreliable; dates not used in features",
    )
    neg = (b.date_closed < b.date_opened).sum()
    log.add("bureau_credit", "date_closed before date_opened", neg, "none found")
    over = (b.outstanding_amount > b.sanctioned_amount).sum()
    log.add("bureau_credit", "outstanding above sanctioned amount", over, "none found")
    return b


def clean_repay(r, log):
    r = r.sort_values(["account_id", "payment_month"]).copy()
    bad = (
        ((r.payment_status == "On-Time") & (r.days_past_due != 0))
        | ((r.payment_status == "Late") & ~r.days_past_due.between(1, 29))
        | ((r.payment_status == "Missed") & (r.days_past_due < 30))
    ).sum()
    log.add("repayment_history", "payment_status inconsistent with days_past_due", bad, "none found")
    over = (r.emi_paid > r.emi_due + 0.01).sum()
    log.add("repayment_history", "emi_paid above emi_due", over, "none found")
    r["pay_ratio"] = np.where(r.emi_due > 0, r.emi_paid / r.emi_due, 1.0)
    return r


def clean_collections(e, l, log):
    e = e.merge(l[["account_id", "origination_date"]], on="account_id", how="left")
    bad = e.event_date < e.origination_date
    e["event_before_origination"] = bad.astype(int)
    log.add(
        "collections_events",
        "collections event dated before the loan was originated",
        bad.sum(),
        "flagged as invalid; excluded from collections metrics (kept in raw for audit)",
    )
    log.add(
        "collections_events",
        "use in default model",
        len(e),
        "not used as model features: collections happen after delinquency, so they would leak the outcome",
    )
    return e.drop(columns="origination_date")


def ingest(raw_dir=C.RAW):
    t = load_raw(raw_dir)
    log = DQLog()
    for name, df in t.items():
        log.add(name, "rows loaded", len(df), "-")
    _key_checks(t, log)
    t["customers"] = clean_customers(t["customers"], log)
    t["loans"] = clean_loans(t["loans"], t["customers"], log)
    t["bureau"] = clean_bureau(t["bureau"], log)
    t["repay"] = clean_repay(t["repay"], log)
    t["collections"] = clean_collections(t["collections"], t["loans"], log)
    return t, log
