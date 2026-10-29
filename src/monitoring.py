"""Post-deployment monitoring metrics, computed on the scored portfolio.

Origination cohorts stand in for monthly scoring batches: 2018-2020 is the
reference (development) population and each later year is compared with it,
the same way a monthly batch would be compared in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .model import ks_stat, psi

PSI_FEATURES = {
    "pd": "Model PD (score)", "annual_income": "Annual income", "rep_mean_dpd": "Average DPD (12m)",
    "emi_to_income": "Annual EMI / income", "bur_n_written_off": "Bureau write-offs", "loan_amount": "Loan amount",
}
REFERENCE_YEARS = (2018, 2019, 2020)


def status(value, amber, red, higher_is_worse=True):
    if not higher_is_worse:
        value, amber, red = -value, -amber, -red
    return "Red" if value >= red else "Amber" if value >= amber else "Green"


def monitoring_report(df: pd.DataFrame) -> dict:
    ref = df[df.origination_year.isin(REFERENCE_YEARS)]
    out = {"reference": f"{REFERENCE_YEARS[0]}-{REFERENCE_YEARS[-1]} originations ({len(ref):,} accounts)"}

    # 1. stability (PSI) per cohort vs reference
    rows = []
    for yr in sorted(df.origination_year.unique()):
        if yr in REFERENCE_YEARS:
            continue
        cur = df[df.origination_year == yr]
        for f, label in PSI_FEATURES.items():
            rows.append({"cohort": int(yr), "feature": label, "psi": round(psi(ref[f].values, cur[f].values), 4)})
    out["psi"] = rows

    # 2. discrimination per cohort (out-of-fold scores, so honest)
    perf = []
    for yr, g in df.groupby("origination_year"):
        auc = roc_auc_score(g.target_default, g.pd)
        perf.append({"cohort": int(yr), "accounts": int(len(g)), "auc": round(float(auc), 4),
                     "ks": round(ks_stat(g.target_default, g.pd), 4),
                     "actual_default_rate": round(float(g.target_default.mean()), 4),
                     "avg_pd": round(float(g.pd.mean()), 4)})
    out["performance"] = perf

    # 3. calibration by risk band
    cal = []
    for band in C.PD_LABELS:
        g = df[df.risk_band == band]
        if len(g):
            cal.append({"band": band, "accounts": int(len(g)), "avg_pd": round(float(g.pd.mean()), 4),
                        "actual_default_rate": round(float(g.target_default.mean()), 4)})
    out["calibration"] = cal

    # 4. scorecard: worst value per check across cohorts
    psi_df = pd.DataFrame(rows)
    score_psi = psi_df[psi_df.feature == PSI_FEATURES["pd"]].psi.max()
    feat_psi = psi_df[psi_df.feature != PSI_FEATURES["pd"]].psi.max()
    ref_auc = roc_auc_score(ref.target_default, ref.pd)
    min_auc = min(p["auc"] for p in perf)
    cal_gap = max(abs(c["avg_pd"] - c["actual_default_rate"]) / c["actual_default_rate"] for c in cal)
    out["scorecard"] = [
        {"check": "Score stability (max PSI of PD vs reference)", "value": round(float(score_psi), 3),
         "threshold": "amber 0.10, red 0.25", "status": status(score_psi, 0.10, 0.25)},
        {"check": "Input drift (max feature PSI)", "value": round(float(feat_psi), 3),
         "threshold": "amber 0.10, red 0.25", "status": status(feat_psi, 0.10, 0.25)},
        {"check": "Discrimination (lowest cohort AUC)", "value": round(float(min_auc), 3),
         "threshold": f"amber < {ref_auc - 0.03:.3f}, red < {ref_auc - 0.05:.3f} (reference {ref_auc:.3f})",
         "status": status(min_auc, ref_auc - 0.03, ref_auc - 0.05, higher_is_worse=False)},
        {"check": "Calibration (worst band, |PD - actual| / actual)", "value": round(float(cal_gap), 3),
         "threshold": "amber 0.15, red 0.25", "status": status(cal_gap, 0.15, 0.25)},
    ]
    return out
