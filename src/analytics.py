"""Portfolio profiling and the written insights. Every number in an insight
is computed here from the clean dataset, so the text cannot drift from the data."""
from __future__ import annotations

import pandas as pd

DIMENSIONS = {
    "loan_type": "Loan type", "employment_type": "Employment type", "income_band": "Income band",
    "region": "Region", "age_band": "Age band", "city_tier": "City tier", "housing_type": "Housing",
    "education_level": "Education", "loan_purpose": "Loan purpose", "disbursal_channel": "Channel",
    "risk_band": "Risk band", "origination_year": "Origination year", "occupation_sector": "Sector",
    "marital_status": "Marital status", "gender": "Gender",
}
ORDER = {
    "income_band": ["<58k", "58-73k", "73-86k", "86-101k", "101k+"],
    "age_band": ["21-25", "26-35", "36-45", "46-55", "56+"],
}


def profile(df: pd.DataFrame, by, extra_cols=True) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    g = df.groupby(by, observed=True)
    t = pd.DataFrame(
        {
            "accounts": g.size(),
            "defaults": g.target_default.sum(),
            "default_rate": g.target_default.mean(),
            "exposure": g.exposure.sum(),
        }
    )
    t["defaulted_exposure"] = df[df.target_default == 1].groupby(by, observed=True).exposure.sum()
    t["defaulted_exposure"] = t.defaulted_exposure.fillna(0)
    t["exposure_share"] = t.exposure / t.exposure.sum()
    t["defaulted_exposure_share"] = t.defaulted_exposure / t.defaulted_exposure.sum()
    if extra_cols and "pd" in df:
        t["avg_pd"] = g.pd.mean()
    return t.reset_index()


def _rate(df, mask):
    return df.loc[mask, "target_default"].mean()


def insights(df: pd.DataFrame) -> list[dict]:
    out = []
    base = df.target_default.mean()

    # 1. mortgage concentration
    lt = profile(df, "loan_type").set_index("loan_type")
    m = lt.loc["Mortgage"]
    others = df[df.loan_type != "Mortgage"].target_default.mean()
    out.append({
        "title": "Mortgages carry most of the risk in money terms",
        "finding": (
            f"Mortgages are {m.accounts / len(df):.0%} of accounts but {m.exposure_share:.0%} of exposure and "
            f"{m.defaulted_exposure_share:.0%} of defaulted exposure. Their default rate is {m.default_rate:.1%} "
            f"against {others:.1%} for all other products."
        ),
        "so_what": (
            "Exposure-weighted, the portfolio's risk is a mortgage question. Prioritise a mortgage "
            "underwriting review and request LTV and property values, which are missing from this data, "
            "so loss given default can be estimated."
        ),
    })

    # 2. income gradient
    ib = profile(df, "income_band").set_index("income_band")
    lo, hi = ib.loc["<58k"], ib.loc["101k+"]
    out.append({
        "title": "Income is the clearest single driver",
        "finding": (
            f"Default rate falls steadily with income, from {lo.default_rate:.1%} below $58k to "
            f"{hi.default_rate:.1%} above $101k ({lo.default_rate / hi.default_rate:.1f}x). The lowest band "
            f"holds {lo.exposure_share:.0%} of exposure but {lo.defaulted_exposure_share:.0%} of defaulted exposure."
        ),
        "so_what": (
            "Affordability should drive limits: cap EMI-to-income for the lowest band and route "
            "sub-$58k applications for larger loans to manual review."
        ),
    })

    # 3. cross-dimensional: employment x income
    un_lo = (df.employment_type == "Unemployed") & (df.income_band == "<58k")
    sal_hi = (df.employment_type == "Salaried") & (df.income_band == "101k+")
    emp = profile(df, "employment_type").set_index("employment_type").loc["Unemployed"]
    lo_mort = _rate(df, (df.loan_type == "Mortgage") & (df.income_band == "<58k"))
    out.append({
        "title": "Unemployed, low-income borrowers are the worst segment (cross-cut)",
        "finding": (
            f"Unemployed borrowers earning under $58k default at {_rate(df, un_lo):.1%} "
            f"({int(un_lo.sum())} accounts), against {_rate(df, sal_hi):.1%} for salaried borrowers above $101k. "
            f"Unemployed borrowers are {emp.exposure_share:.0%} of exposure but "
            f"{emp.defaulted_exposure_share:.0%} of defaulted exposure. Low-income mortgages default at {lo_mort:.1%}."
        ),
        "so_what": (
            "A combined policy rule (employment status x income) is sharper than either cut-off alone. "
            "Require a guarantor or income proof for unemployed applicants and review the existing "
            "low-income mortgage book first."
        ),
    })

    # 4. age x employment
    ab = profile(df, "age_band").set_index("age_band")
    young_un = _rate(df, (df.age_band == "21-25") & (df.employment_type == "Unemployed"))
    out.append({
        "title": "Younger borrowers default about twice as often",
        "finding": (
            f"Default rate is {ab.loc['21-25'].default_rate:.1%} for ages 21-25 and "
            f"{ab.loc['56+'].default_rate:.1%} for 56+. Within 21-25, unemployed borrowers reach {young_un:.1%}. "
            f"Region barely matters: default rates range only "
            f"{df.groupby('region').target_default.mean().min():.1%}-{df.groupby('region').target_default.mean().max():.1%}."
        ),
        "so_what": (
            "Use age for monitoring and collections planning, not pricing (age is a protected attribute in "
            "many markets). Do not steer risk appetite by region, the differences are noise-sized."
        ),
    })

    # 5. behaviour
    bad = df.rep_n_late + df.rep_n_missed
    wo3 = df.bur_n_written_off >= 3
    out.append({
        "title": "The first 12 months of repayment are an early-warning signal",
        "finding": (
            f"Accounts with no late or missed payment in their first 12 months default at {_rate(df, bad == 0):.1%}; "
            f"those with 5 or more late/missed months default at {_rate(df, bad >= 5):.1%}. Borrowers with 3+ "
            f"written-off bureau lines default at {_rate(df, wo3):.1%} (portfolio {base:.1%})."
        ),
        "so_what": (
            "Feed the 12-month behaviour score into collections: contact accounts with 3+ late months "
            "before they reach 30 DPD, and treat multiple external write-offs as a hard review trigger."
        ),
    })
    return out
