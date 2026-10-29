"""Aggregate the one-to-many tables to account level and build the clean,
analysis-ready dataset (one row per loan account)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def repayment_features(r: pd.DataFrame) -> pd.DataFrame:
    """12-month behaviour per account. Months 1-9 are the account's own
    baseline, months 10-12 are 'recent' (used by anomaly detection too)."""
    r = r.copy()
    r["missed"] = (r.payment_status == "Missed").astype(int)
    r["late"] = (r.payment_status == "Late").astype(int)
    r["dpd30"] = (r.days_past_due >= 30).astype(int)
    r["recent"] = r.payment_month >= 10
    g = r.groupby("account_id")
    out = pd.DataFrame(
        {
            "rep_n_missed": g.missed.sum(),
            "rep_n_late": g.late.sum(),
            "rep_max_dpd": g.days_past_due.max(),
            "rep_mean_dpd": g.days_past_due.mean(),
            "rep_pay_ratio": g.emi_paid.sum() / g.emi_due.sum(),
            "emi_due": g.emi_due.first(),
            "outstanding_balance": g.outstanding_balance.last(),
        }
    )
    rec = r[r.recent].groupby("account_id")
    base = r[~r.recent].groupby("account_id")
    out["rep_missed_last3"] = rec.missed.sum()
    out["rep_mean_dpd_last3"] = rec.days_past_due.mean()
    out["rep_pay_ratio_last3"] = rec.emi_paid.sum() / rec.emi_due.sum()
    out["rep_mean_dpd_first9"] = base.days_past_due.mean()
    out["rep_std_dpd_first9"] = base.days_past_due.std()
    out["rep_pay_ratio_first9"] = base.emi_paid.sum() / base.emi_due.sum()
    out["rep_dpd_last"] = g.days_past_due.last()

    # longest run of consecutive missed payments
    def longest_run(s):
        best = cur = 0
        for v in s:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    out["rep_max_consec_missed"] = g.missed.apply(longest_run)
    # slope of DPD over the 12 months (positive = getting worse)
    m = r.payment_month - 6.5
    r["_xy"] = m * r.days_past_due
    out["rep_dpd_trend"] = r.groupby("account_id")._xy.sum() / (m**2).groupby(r.account_id).sum()
    return out.reset_index()


def bureau_features(b: pd.DataFrame, loans: pd.DataFrame, log=None, point_in_time=False) -> pd.DataFrame:
    """Bureau view per loan.

    Default (point_in_time=False): the full bureau snapshot as at AS_OF_DATE.
    This is what a monitoring / behavioural score would see today.
    point_in_time=True keeps only lines opened on or before the loan's
    origination date; used as a leakage sensitivity check in model.py."""
    x = loans[["account_id", "customer_id", "origination_date"]].merge(b, on="customer_id", how="left")
    has_line = x.bureau_id.notna()
    after = has_line & (x.date_opened > x.origination_date)
    if log is not None:
        log.add(
            "bureau_credit",
            "loan-to-bureau pairs where the line was opened after the loan's origination",
            after.sum(),
            "kept for the monitoring score (bureau is a snapshot at the as-of date); a point-in-time "
            "model without these lines is reported as a leakage sensitivity check",
        )
    x = x[has_line & ~after].copy() if point_in_time else x[has_line].copy()
    x["active"] = (x.account_status == "Active").astype(int)
    x["wo"] = (x.account_status == "Written-Off").astype(int)
    x["closed"] = (x.account_status == "Closed").astype(int)
    x["dpd30"] = (x.days_past_due >= 30).astype(int)
    x["nonbank"] = x.lender_category.isin(["NBFC", "Fintech"]).astype(int)
    ref = x.origination_date if point_in_time else pd.Timestamp(C.AS_OF_DATE)
    x["recent12"] = ((ref - x.date_opened).dt.days.between(0, 365)).astype(int)
    x["age_m"] = (ref - x.date_opened).dt.days / 30.44
    x["act_out"] = x.outstanding_amount * x.active
    x["act_sanc"] = x.sanctioned_amount * x.active
    g = x.groupby("account_id")
    out = pd.DataFrame(
        {
            "bur_n_lines": g.size(),
            "bur_n_active": g.active.sum(),
            "bur_n_written_off": g.wo.sum(),
            "bur_n_closed": g.closed.sum(),
            "bur_total_sanctioned": g.sanctioned_amount.sum(),
            "bur_total_outstanding": g.outstanding_amount.sum(),
            "bur_max_dpd": g.days_past_due.max(),
            "bur_mean_dpd": g.days_past_due.mean(),
            "bur_n_dpd30": g.dpd30.sum(),
            "bur_n_nonbank": g.nonbank.sum(),
            "bur_n_opened_12m": g.recent12.sum(),
            "bur_oldest_line_months": g.age_m.max(),
        }
    )
    out["bur_utilization"] = (g.act_out.sum() / g.act_sanc.sum().replace(0, np.nan)).fillna(0)
    out = out.reset_index()
    out = loans[["account_id"]].merge(out, on="account_id", how="left")
    out["thin_file"] = out.bur_n_lines.isna().astype(int)
    return out.fillna(0)


def collections_summary(e: pd.DataFrame) -> pd.DataFrame:
    """Dashboard-only (never used by the model)."""
    v = e[e.event_before_origination == 0]
    g = v.groupby("account_id")
    return pd.DataFrame(
        {
            "coll_n_events": g.size(),
            "coll_amount_recovered": g.amount_recovered.sum(),
            "coll_last_action": g.apply(lambda d: d.sort_values("event_date").action_type.iloc[-1], include_groups=False),
        }
    ).reset_index()


def build_portfolio(t: dict, log=None) -> pd.DataFrame:
    loans, cust = t["loans"], t["customers"]
    df = loans.merge(cust, on="customer_id", how="left", validate="many_to_one")
    df = df.merge(repayment_features(t["repay"]), on="account_id", how="left", validate="one_to_one")
    df = df.merge(bureau_features(t["bureau"], loans, log), on="account_id", how="left", validate="one_to_one")
    df = df.merge(collections_summary(t["collections"]), on="account_id", how="left", validate="one_to_one")
    df["coll_n_events"] = df.coll_n_events.fillna(0).astype(int)
    df["coll_amount_recovered"] = df.coll_amount_recovered.fillna(0.0)
    df["coll_last_action"] = df.coll_last_action.fillna("None")

    # affordability and exposure ratios
    df["loan_to_income"] = df.loan_amount / df.annual_income
    df["emi_to_income"] = df.emi_due * 12 / df.annual_income
    df["bur_debt_to_income"] = df.bur_total_outstanding / df.annual_income
    df["n_loans_customer"] = df.groupby("customer_id").account_id.transform("size")
    df["exposure"] = df.outstanding_balance  # balance after month 12
    df["origination_quarter"] = df.origination_date.dt.to_period("Q").astype(str)
    assert len(df) == len(loans), "join changed the account count"
    return df
