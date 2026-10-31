"""Default model: logistic-regression baseline and a LightGBM challenger.

Validation design
  * Split is grouped by customer_id so the same customer never sits in both
    train and test (customers hold up to 3 loans).
  * 75/25 holdout for the headline metrics, plus 5-fold grouped CV for
    stability, plus an out-of-time check (train on 2018-2022 originations,
    test on 2023).
  * Dashboard scores are out-of-fold, so no account is scored by a model that
    saw its own label.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

NUMERIC = [
    "age", "annual_income", "years_employed", "num_dependents", "years_at_residence",
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag",
    "loan_to_income", "emi_to_income", "n_loans_customer",
    "years_employed_missing", "interest_rate_missing",
    # repayment history (12 months)
    "rep_n_missed", "rep_n_late", "rep_max_dpd", "rep_mean_dpd", "rep_pay_ratio",
    "rep_missed_last3", "rep_mean_dpd_last3", "rep_max_consec_missed", "rep_dpd_trend",
    # bureau
    "bur_n_lines", "bur_n_active", "bur_n_written_off", "bur_total_outstanding",
    "bur_max_dpd", "bur_mean_dpd", "bur_n_dpd30", "bur_n_nonbank", "bur_n_opened_12m",
    "bur_utilization", "bur_debt_to_income", "thin_file",
]
CATEGORICAL = ["loan_type", "employment_type"]

# Feature selection (grouped 5-fold CV, logistic regression): with all 9
# categoricals AUC was 0.676; dropping the 7 below raises it to 0.681. They add
# noise rather than signal, and marital status and housing are attributes a
# lender should avoid (fair-lending risk). Gender was never used.
DROPPED_FEATURES = {
    "marital_status": "no signal; protected attribute in many fair-lending rules",
    "education_level": "no stable signal; fairness risk",
    "housing_type": "no signal; proxy risk",
    "loan_purpose": "no signal",
    "disbursal_channel": "no signal",
    "region": "default rate varies only 15.0-16.9% by region",
    "city_tier": "no signal",
}
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target_default"

LABELS = {
    "age": "Age", "annual_income": "Annual income", "years_employed": "Years employed",
    "num_dependents": "Dependents", "years_at_residence": "Years at residence",
    "loan_amount": "Loan amount", "interest_rate_pct": "Interest rate", "tenure_months": "Tenure",
    "collateral_flag": "Secured loan", "loan_to_income": "Loan / income", "emi_to_income": "Annual EMI / income",
    "n_loans_customer": "Loans held with us", "years_employed_missing": "Years employed missing",
    "interest_rate_missing": "Rate missing", "rep_n_missed": "Missed payments (12m)",
    "rep_n_late": "Late payments (12m)", "rep_max_dpd": "Worst DPD (12m)", "rep_mean_dpd": "Average DPD (12m)",
    "rep_pay_ratio": "Share of EMI paid (12m)", "rep_missed_last3": "Missed payments (last 3m)",
    "rep_mean_dpd_last3": "Average DPD (last 3m)", "rep_max_consec_missed": "Longest missed streak",
    "rep_dpd_trend": "DPD trend", "bur_n_lines": "Bureau lines", "bur_n_active": "Active bureau lines",
    "bur_n_written_off": "Bureau write-offs", "bur_total_outstanding": "Bureau outstanding",
    "bur_max_dpd": "Worst bureau DPD", "bur_mean_dpd": "Average bureau DPD", "bur_n_dpd30": "Bureau lines 30+ DPD",
    "bur_n_nonbank": "NBFC / fintech lines", "bur_n_opened_12m": "Lines opened last 12m",
    "bur_utilization": "Bureau utilisation", "bur_debt_to_income": "Bureau debt / income",
    "thin_file": "No bureau history", "loan_type": "Loan type", "loan_purpose": "Loan purpose",
    "disbursal_channel": "Channel", "employment_type": "Employment type", "region": "Region",
    "city_tier": "City tier", "housing_type": "Housing", "education_level": "Education",
    "marital_status": "Marital status",
}

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, n_estimators=500, num_leaves=4, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=C.RANDOM_STATE,
    verbose=-1,
)


# ---------------------------------------------------------------- metrics
def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"AUC": round(float(auc), 4), "Gini": round(float(2 * auc - 1), 4), "KS": round(ks_stat(y, p), 4)}


def psi(expected, actual, bins=10):
    """Population stability index using decile edges of `expected`."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def decile_table(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(10, 0, -1)).astype(int)
    t = d.groupby("decile").agg(accounts=("y", "size"), defaults=("y", "sum"), avg_pd=("p", "mean"))
    t["default_rate"] = t.defaults / t.accounts
    t = t.sort_index()
    t["cum_capture"] = t.defaults.cumsum() / t.defaults.sum()
    return t.reset_index()


# ---------------------------------------------------------------- models
def make_lr():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.3, max_iter=2000))])


def _lgb_frame(X):
    X = X[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c])
    return X


def fit_lgb(X, y):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(_lgb_frame(X), y)
    return m


def _cat_levels(df):
    return {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}


def _lgb_frame_fixed(X, levels):
    X = X[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    return X


def train_and_evaluate(df: pd.DataFrame, df_pit: pd.DataFrame | None = None):
    """Returns (results dict, scored dataframe, shap dataframe)."""
    levels = _cat_levels(df)
    y = df[TARGET].values
    groups = df.customer_id.values
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=C.RANDOM_STATE).split(df, y, groups))
    Xtr, Xte = df.iloc[tr], df.iloc[te]

    lr = make_lr().fit(Xtr[FEATURES], y[tr])
    p_lr = lr.predict_proba(Xte[FEATURES])[:, 1]

    gbm = lgb.LGBMClassifier(**LGB_PARAMS).fit(_lgb_frame_fixed(Xtr, levels), y[tr])
    p_gbm = gbm.predict_proba(_lgb_frame_fixed(Xte, levels))[:, 1]

    res = {
        "holdout": {"Logistic regression": metrics(y[te], p_lr), "LightGBM": metrics(y[te], p_gbm)},
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_default_rate": round(float(y[tr].mean()), 4), "test_default_rate": round(float(y[te].mean()), 4),
    }

    # ROC curves for the dashboard
    roc = {}
    for name, p in [("Logistic regression", p_lr), ("LightGBM", p_gbm)]:
        fpr, tpr, _ = roc_curve(y[te], p)
        idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
        roc[name] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
    res["roc"] = roc
    res["_p_test"] = {"LightGBM": p_gbm, "Logistic regression": p_lr}

    # LR coefficients (standardised), top 15 by magnitude
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["lr"].coef_[0], index=names)
    coefs.index = [i.replace("num__", "").replace("cat__", "") for i in coefs.index]
    res["lr_coefficients"] = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:15].round(4).to_dict()

    # 5-fold grouped CV and out-of-fold scores (champion and LR)
    gkf = GroupKFold(n_splits=5)
    oof_gbm, oof_lr = np.zeros(len(df)), np.zeros(len(df))
    shap = np.zeros((len(df), len(FEATURES)))
    lr_contrib = np.zeros((len(df), len(FEATURES)))
    cv = {"Logistic regression": [], "LightGBM": []}
    for ftr, fte in gkf.split(df, y, groups):
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(_lgb_frame_fixed(df.iloc[ftr], levels), y[ftr])
        Xf = _lgb_frame_fixed(df.iloc[fte], levels)
        oof_gbm[fte] = m.predict_proba(Xf)[:, 1]
        shap[fte] = m.predict(Xf, pred_contrib=True)[:, :-1]
        l2 = make_lr().fit(df.iloc[ftr][FEATURES], y[ftr])
        oof_lr[fte] = l2.predict_proba(df.iloc[fte][FEATURES])[:, 1]
        lr_contrib[fte] = lr_contributions(l2, df.iloc[fte])
        cv["LightGBM"].append(roc_auc_score(y[fte], oof_gbm[fte]))
        cv["Logistic regression"].append(roc_auc_score(y[fte], oof_lr[fte]))
    res["cv_auc"] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in cv.items()}
    res["dropped_features"] = DROPPED_FEATURES
    res["oof_all"] = {"LightGBM": metrics(y, oof_gbm), "Logistic regression": metrics(y, oof_lr)}
    # Champion = higher mean grouped-CV AUC (more stable than one holdout).
    champion = max(res["cv_auc"], key=lambda k: res["cv_auc"][k]["mean"])
    res["champion"] = champion
    res["decile_table"] = decile_table(y[te], res.pop("_p_test")[champion]).round(4).to_dict(orient="list")

    # out-of-time: train <=2022, test 2023
    oot_tr, oot_te = df.origination_year <= 2022, df.origination_year == 2023
    m = lgb.LGBMClassifier(**LGB_PARAMS).fit(_lgb_frame_fixed(df[oot_tr], levels), y[oot_tr])
    p_oot = m.predict_proba(_lgb_frame_fixed(df[oot_te], levels))[:, 1]
    p_oot_train = m.predict_proba(_lgb_frame_fixed(df[oot_tr], levels))[:, 1]
    l3 = make_lr().fit(df[oot_tr][FEATURES], y[oot_tr])
    res["out_of_time_2023"] = {
        "LightGBM": metrics(y[oot_te], p_oot),
        "Logistic regression": metrics(y[oot_te], l3.predict_proba(df[oot_te][FEATURES])[:, 1]),
        "score_psi_2018_22_vs_2023": round(psi(p_oot_train, p_oot), 4),
    }

    # leakage sensitivity: bureau restricted to lines opened before origination
    if df_pit is not None:
        ypit = df_pit[TARGET].values
        Xa, Xb = df_pit.iloc[tr], df_pit.iloc[te]
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(_lgb_frame_fixed(Xa, levels), ypit[tr])
        res["point_in_time_bureau_holdout"] = metrics(ypit[te], m.predict_proba(_lgb_frame_fixed(Xb, levels))[:, 1])

    # feature importance: mean |SHAP| from the OOF LightGBM, and mean |contribution| for LR
    imp = pd.Series(np.abs(shap).mean(0), index=FEATURES).sort_values(ascending=False)
    res["shap_importance"] = imp.round(5).to_dict()
    imp_lr = pd.Series(np.abs(lr_contrib).mean(0), index=FEATURES).sort_values(ascending=False)
    res["lr_importance"] = imp_lr.round(5).to_dict()

    # PSI of key inputs, train vs test (monitoring template)
    res["feature_psi_train_vs_test"] = {
        f: round(psi(Xtr[f].values, Xte[f].values), 4)
        for f in ["annual_income", "rep_mean_dpd", "bur_n_written_off", "emi_to_income"]
    }

    scored = df[["account_id"]].copy()
    oof = oof_gbm if champion == "LightGBM" else oof_lr
    scored["pd"] = oof
    scored["pd_lr"] = oof_lr
    scored["risk_band"] = pd.cut(scored.pd, C.PD_BINS, labels=C.PD_LABELS, right=False).astype(str)
    shap_df = pd.DataFrame(shap if champion == "LightGBM" else lr_contrib, columns=FEATURES)
    shap_df.insert(0, "account_id", df.account_id.values)

    return res, scored, shap_df


def lr_contributions(pipe, X):
    """Per-feature contribution to log-odds for the LR pipeline, relative to
    the average numeric value and the dropped category level. Same units as
    TreeSHAP from LightGBM, so the dashboard treats both the same way."""
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["lr"]
    Z = pre.transform(X[FEATURES])
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib = Z * lr.coef_[0]
    names = pre.get_feature_names_out()
    out = np.zeros((len(X), len(FEATURES)))
    for j, n in enumerate(names):
        base = n.split("__", 1)[1]
        f = base if base in NUMERIC else next(c for c in CATEGORICAL if base.startswith(c + "_"))
        out[:, FEATURES.index(f)] += contrib[:, j]
    return out


def top_drivers(shap_row: pd.Series, feat_row: pd.Series, k=5):
    """Readable top-k drivers for one account (positive = raises PD)."""
    s = shap_row[FEATURES].astype(float)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:k]
    out = []
    for f, v in s.items():
        val = feat_row[f]
        if isinstance(val, (float, np.floating)):
            val = round(float(val), 2)
        out.append({"feature": LABELS.get(f, f), "value": val, "impact": round(float(v), 3),
                    "direction": "raises risk" if v > 0 else "lowers risk"})
    return out


def save_results(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)
