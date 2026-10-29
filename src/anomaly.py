"""Anomaly detection on repayment and bureau behaviour.

Three detectors, none of which uses target_default:

1. Own-history break (repayment): the last 3 months look very different
   from the account's own first 9 months, e.g. a clean payer who starts
   missing, or a sharp DPD jump vs the account's own variation.
2. Peer-segment outlier (bureau + affordability): robust z-score (median and
   MAD) inside the peer segment loan_type x income_band. Flags values far
   from what similar borrowers look like.
3. Multivariate outlier: Isolation Forest over the combined behaviour
   vector; catches unusual combinations that no single rule sees.

An account is flagged if any rule fires or the Isolation Forest puts it in
the top 1.5 %. Each flag carries plain-English reasons for the reviewer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from . import config as C
from .model import LABELS

EXTRA_LABELS = {
    "rep_mean_dpd_first9": "avg DPD months 1-9", "rep_mean_dpd_last3": "avg DPD months 10-12",
    "rep_pay_ratio_first9": "share paid months 1-9", "rep_pay_ratio_last3": "share paid months 10-12",
}

PEER_Z = 4.5           # robust z threshold for peer outliers
IF_CONTAMINATION = 0.015

PEER_FEATURES = {
    "emi_to_income": "annual EMI / income",
    "bur_utilization": "bureau utilisation",
    "bur_n_opened_12m": "bureau lines opened in last 12 months",
    "bur_debt_to_income": "bureau debt / income",
    "bur_n_written_off": "bureau write-offs",
}

IF_FEATURES = [
    "rep_mean_dpd_first9", "rep_mean_dpd_last3", "rep_pay_ratio_first9", "rep_pay_ratio_last3",
    "rep_max_consec_missed", "rep_dpd_trend", "emi_to_income", "loan_to_income",
    "bur_utilization", "bur_n_opened_12m", "bur_debt_to_income", "bur_mean_dpd", "bur_n_written_off",
]


def robust_z(s: pd.Series, groups: pd.Series) -> pd.Series:
    med = s.groupby(groups).transform("median")
    mad = (s - med).abs().groupby(groups).transform("median")
    # fall back to the standard deviation when MAD is 0 (count features)
    sd = s.groupby(groups).transform("std")
    scale = (1.4826 * mad).where(mad > 0, sd).replace(0, np.nan)
    return ((s - med) / scale).fillna(0)


def detect(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    reasons = {a: [] for a in d.account_id}
    kinds = {a: set() for a in d.account_id}

    # 1. own-history break --------------------------------------------------
    clean_base = (d.rep_mean_dpd_first9 <= 3) & (d.rep_pay_ratio_first9 >= 0.98)
    new_missing = clean_base & (d.rep_missed_last3 >= 2)
    sd = d.rep_std_dpd_first9.clip(lower=3)
    dpd_jump_z = (d.rep_mean_dpd_last3 - d.rep_mean_dpd_first9) / sd
    jump = (dpd_jump_z >= 4) & (d.rep_mean_dpd_last3 >= 30)
    pay_drop = (d.rep_pay_ratio_first9 - d.rep_pay_ratio_last3) >= 0.5
    for i in d.index[new_missing]:
        a = d.at[i, "account_id"]
        reasons[a].append(
            f"Own history: paid cleanly for months 1-9 (avg DPD {d.at[i,'rep_mean_dpd_first9']:.1f}) "
            f"then missed {int(d.at[i,'rep_missed_last3'])} of the last 3 EMIs"
        )
        kinds[a].add("Own history")
    for i in d.index[jump & ~new_missing]:
        a = d.at[i, "account_id"]
        reasons[a].append(
            f"Own history: avg DPD rose from {d.at[i,'rep_mean_dpd_first9']:.1f} (months 1-9) "
            f"to {d.at[i,'rep_mean_dpd_last3']:.1f} (months 10-12), {dpd_jump_z[i]:.1f} sd above its own norm"
        )
        kinds[a].add("Own history")
    for i in d.index[pay_drop & ~new_missing & ~jump]:
        a = d.at[i, "account_id"]
        reasons[a].append(
            f"Own history: share of EMI paid fell from {d.at[i,'rep_pay_ratio_first9']:.0%} "
            f"to {d.at[i,'rep_pay_ratio_last3']:.0%} in the last 3 months"
        )
        kinds[a].add("Own history")

    # 2. peer-segment outliers ---------------------------------------------
    d["peer_segment"] = d.loan_type + " | income " + d.income_band
    for f, label in PEER_FEATURES.items():
        z = robust_z(d[f].astype(float), d.peer_segment)
        d[f"z_{f}"] = z
        med = d[f].groupby(d.peer_segment).transform("median")
        hit = z >= PEER_Z
        if f == "bur_n_written_off":
            hit = hit & (d[f] >= 3)
        for i in d.index[hit]:
            a = d.at[i, "account_id"]
            reasons[a].append(
                f"Peer segment: {label} = {d.at[i,f]:.2f} vs segment median {med[i]:.2f} "
                f"({d.at[i,'peer_segment']}), z = {z[i]:.1f}"
            )
            kinds[a].add("Peer segment")

    # 3. multivariate -------------------------------------------------------
    X = RobustScaler().fit_transform(d[IF_FEATURES].fillna(0))
    iso = IsolationForest(n_estimators=300, contamination=IF_CONTAMINATION, random_state=C.RANDOM_STATE)
    iso.fit(X)
    score = -iso.score_samples(X)  # higher = more unusual
    d["if_score"] = score
    d["if_pct"] = pd.Series(score).rank(pct=True).values
    if_flag = iso.predict(X) == -1
    zX = pd.DataFrame(X, columns=IF_FEATURES, index=d.index)
    for i in d.index[if_flag]:
        a = d.at[i, "account_id"]
        top = zX.loc[i].abs().sort_values(ascending=False).index[:3]
        reasons[a].append(
            "Unusual combination (Isolation Forest, top 1.5%): most extreme on "
            + ", ".join(f"{EXTRA_LABELS.get(t, LABELS.get(t, t))} = {d.at[i, t]:.2f}" for t in top)
        )
        kinds[a].add("Multivariate")

    d["anomaly_reasons"] = d.account_id.map(lambda a: reasons[a])
    d["anomaly_types"] = d.account_id.map(lambda a: ", ".join(sorted(kinds[a])))
    d["anomaly_flag"] = d.anomaly_reasons.map(len) > 0
    d["anomaly_n_reasons"] = d.anomaly_reasons.map(len)
    # priority: number of independent detectors, then isolation score
    d["anomaly_priority"] = d.account_id.map(lambda a: len(kinds[a])) + d.if_pct
    cols = ["account_id", "anomaly_flag", "anomaly_types", "anomaly_n_reasons", "anomaly_reasons",
            "anomaly_priority", "if_score", "if_pct", "peer_segment"] + [f"z_{f}" for f in PEER_FEATURES]
    return d[cols]
