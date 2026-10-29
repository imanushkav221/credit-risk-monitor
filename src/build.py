"""Build everything the dashboard needs:  python -m src.build

raw CSVs -> ingest (clean + DQ log) -> account-level features -> model
(LR + LightGBM, metrics, out-of-fold scores, drivers) -> anomaly flags ->
insights -> data/processed/*
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import config as C
from .analytics import insights
from .anomaly import detect
from .features import build_portfolio, bureau_features
from .ingest import ingest
from .model import save_results, train_and_evaluate


def main():
    t0 = time.time()
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    t, log = ingest()
    df = build_portfolio(t, log)
    print(f"[build] clean dataset: {df.shape[0]} accounts x {df.shape[1]} columns")

    # point-in-time bureau variant for the leakage sensitivity check
    pit = df.drop(columns=[c for c in df.columns if c.startswith("bur_")] + ["thin_file"])
    pit = pit.merge(bureau_features(t["bureau"], t["loans"], point_in_time=True), on="account_id")
    pit["bur_debt_to_income"] = pit.bur_total_outstanding / pit.annual_income

    res, scored, contrib = train_and_evaluate(df, pit)
    print(f"[build] champion={res['champion']} holdout={res['holdout']}")
    df = df.merge(scored, on="account_id", how="left")

    anom = detect(df)
    df = df.merge(anom, on="account_id", how="left")
    print(f"[build] anomalies flagged: {int(df.anomaly_flag.sum())}")
    res["anomaly_validation"] = {
        "flagged": int(df.anomaly_flag.sum()),
        "default_rate_flagged": round(float(df[df.anomaly_flag].target_default.mean()), 4),
        "default_rate_not_flagged": round(float(df[~df.anomaly_flag].target_default.mean()), 4),
    }

    ins = insights(df)

    # save
    df.to_parquet(C.PROCESSED / "portfolio.parquet", index=False)
    contrib.to_parquet(C.PROCESSED / "drivers.parquet", index=False)
    t["repay"].to_parquet(C.PROCESSED / "repayment.parquet", index=False)
    t["bureau"].to_parquet(C.PROCESSED / "bureau.parquet", index=False)
    t["collections"].to_parquet(C.PROCESSED / "collections.parquet", index=False)
    log.frame().to_csv(C.PROCESSED / "dq_log.csv", index=False)
    save_results(res, C.PROCESSED / "model_results.json")
    with open(C.PROCESSED / "insights.json", "w") as f:
        json.dump(ins, f, indent=2)
    print(f"[build] done in {time.time() - t0:.0f}s -> {C.PROCESSED}")


if __name__ == "__main__":
    main()
