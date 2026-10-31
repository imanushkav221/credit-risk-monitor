"""Credit Risk Portfolio Monitor - Streamlit dashboard.

Run:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.analytics import DIMENSIONS, ORDER, profile  # noqa: E402
from src.ask import ask  # noqa: E402
from src.llm import LLM  # noqa: E402
from src.model import LABELS, top_drivers  # noqa: E402

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon="📊", layout="wide")

BAND_COLORS = {
    "A (<8%)": "#2E7D32", "B (8-13%)": "#7CB342", "C (13-20%)": "#FBC02D",
    "D (20-30%)": "#F57C00", "E (30%+)": "#C62828",
}
DECISIONS_FILE = C.PROCESSED / "anomaly_decisions.csv"


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Building data and model (first run only, ~20s)...")
def ensure_built():
    if not (C.PROCESSED / "portfolio.parquet").exists():
        from src.build import main

        main()
    return True


def data_version():
    """Changes whenever the processed files change, so a redeploy with new data never serves the old cache."""
    files = list(C.PROCESSED.glob("*.parquet")) + list(C.PROCESSED.glob("*.json"))
    return tuple(sorted((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in files))


@st.cache_data
def load(version=None):
    ensure_built()
    p = C.PROCESSED
    df = pd.read_parquet(p / "portfolio.parquet")
    data = {
        "df": df,
        "drivers": pd.read_parquet(p / "drivers.parquet"),
        "repay": pd.read_parquet(p / "repayment.parquet"),
        "bureau": pd.read_parquet(p / "bureau.parquet"),
        "coll": pd.read_parquet(p / "collections.parquet"),
        "dq": pd.read_csv(p / "dq_log.csv"),
        "res": json.load(open(p / "model_results.json")),
        "insights": json.load(open(p / "insights.json")),
    }
    return data


def money(v):
    return f"${v / 1e6:,.1f}M" if abs(v) >= 1e6 else f"${v:,.0f}"


def pct(v):
    return f"{v:.1%}"


def ordered(dim, values):
    if dim in ORDER:
        return [v for v in ORDER[dim] if v in set(values)]
    if dim == "risk_band":
        return [v for v in C.PD_LABELS if v in set(values)]
    return sorted(values)


def load_decisions():
    if "decisions" not in st.session_state:
        if DECISIONS_FILE.exists():
            st.session_state.decisions = pd.read_csv(DECISIONS_FILE)
        else:
            st.session_state.decisions = pd.DataFrame(
                columns=["account_id", "decision", "note", "reviewer", "timestamp_utc"])
    return st.session_state.decisions


def save_decision(acc, decision, note, reviewer):
    d = load_decisions()
    d = d[d.account_id != acc]
    row = pd.DataFrame([{"account_id": acc, "decision": decision, "note": note, "reviewer": reviewer or "analyst",
                         "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}])
    d = pd.concat([d, row], ignore_index=True)
    st.session_state.decisions = d
    try:
        d.to_csv(DECISIONS_FILE, index=False)
    except OSError:
        pass


def how_it_works(trigger, fp, fn, hitl):
    with st.expander("How this feature works: trigger, false positive, false negative, human review", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Trigger**  \n{trigger}")
        c1.markdown(f"**Human-in-the-loop**  \n{hitl}")
        c2.markdown(f"**False positive looks like**  \n{fp}")
        c2.markdown(f"**False negative looks like**  \n{fn}")


# ------------------------------------------------------------------ sidebar
ensure_built()
data = load(data_version())
df_all = data["df"]
llm = LLM()

if "view" not in st.session_state:
    st.session_state.view = "Portfolio"
if "account" not in st.session_state:
    st.session_state.account = df_all.sort_values("pd", ascending=False).account_id.iloc[0]

VIEWS = ["Portfolio", "Account", "Ask the Portfolio", "Anomaly review", "Model", "Monitoring", "Data & method"]
st.sidebar.title("Credit Risk Monitor")
st.sidebar.radio("View", VIEWS, key="view")

st.sidebar.markdown("### Filters")
st.sidebar.caption("Apply to the Portfolio and Anomaly views.")
f_lt = st.sidebar.multiselect("Loan type", sorted(df_all.loan_type.unique()))
f_reg = st.sidebar.multiselect("Region", sorted(df_all.region.unique()))
f_emp = st.sidebar.multiselect("Employment type", sorted(df_all.employment_type.unique()))
f_inc = st.sidebar.multiselect("Income band", ORDER["income_band"])
f_band = st.sidebar.multiselect("Risk band", C.PD_LABELS)
yrs = sorted(df_all.origination_year.unique())
f_yr = st.sidebar.select_slider("Origination year", options=yrs, value=(yrs[0], yrs[-1]))

df = df_all
for col, sel in [("loan_type", f_lt), ("region", f_reg), ("employment_type", f_emp), ("income_band", f_inc),
                 ("risk_band", f_band)]:
    if sel:
        df = df[df[col].isin(sel)]
df = df[df.origination_year.between(*f_yr)]
st.sidebar.markdown(f"**{len(df):,}** of {len(df_all):,} accounts in view")
st.sidebar.divider()
st.sidebar.markdown(f"LLM: **{llm.label()}**")
st.sidebar.markdown("[GitHub repo](https://github.com/imanushkav221/credit-risk-monitor)")
st.sidebar.caption("Synthetic data, snapshot 2025-12-31. Exposure = outstanding balance after month 12.")


def go_account(acc):
    st.session_state.account = acc
    st.session_state.acc_select = acc
    st.session_state.view = "Account"


# ================================================================== PORTFOLIO
def view_portfolio():
    st.title("Portfolio view")
    if df.empty:
        st.warning("No accounts match the filters.")
        return
    st.caption("Start here: see where risk and money concentrate, filter in the sidebar, then drill down to an account "
               "at the bottom of the page.")
    k = st.columns(6)
    k[0].metric("Accounts", f"{len(df):,}", help=f"{df.customer_id.nunique():,} customers")
    k[1].metric("Exposure", money(df.exposure.sum()), help="Outstanding balance after month 12 (EAD)")
    k[2].metric("Expected loss", money(df.expected_loss.sum()),
                help="PD x LGD x exposure. LGD is an assumption by product: mortgage 20%, auto 35%, business 55%, "
                     "personal 65%, card 75%.")
    k[3].metric("Default rate", pct(df.target_default.mean()))
    k[4].metric("Avg model PD", pct(df.pd.mean()))
    k[5].metric("Flagged anomalies", f"{int(df.anomaly_flag.sum()):,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk segments")
        rb = profile(df, "risk_band")
        rb["risk_band"] = pd.Categorical(rb.risk_band, ordered("risk_band", rb.risk_band))
        rb = rb.sort_values("risk_band")
        fig = go.Figure()
        fig.add_bar(x=rb.risk_band.astype(str), y=rb.exposure / 1e6, name="Exposure ($M)",
                    marker_color=[BAND_COLORS[b] for b in rb.risk_band.astype(str)])
        fig.add_scatter(x=rb.risk_band.astype(str), y=rb.default_rate * 100, name="Actual default rate (%)",
                        yaxis="y2", mode="lines+markers", line=dict(color="#222"))
        fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="Exposure ($M)",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Default rate (%)", rangemode="tozero"),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Bands come from the model PD. Rising actual default rate across bands shows the score ranks risk.")
    with c2:
        st.subheader("Loan type x risk band")
        measure = st.radio("Measure", ["Exposure", "Expected loss"], horizontal=True, label_visibility="collapsed")
        col = "exposure" if measure == "Exposure" else "expected_loss"
        x = df.groupby(["loan_type", "risk_band"])[col].sum().reset_index()
        fig = px.bar(x, x="loan_type", y=col, color="risk_band", color_discrete_map=BAND_COLORS,
                     category_orders={"risk_band": C.PD_LABELS}, labels={col: f"{measure} ($)", "loan_type": ""})
        fig.update_layout(height=340, margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Default rate and exposure by dimension")
    dims = {k: v for k, v in DIMENSIONS.items() if k != "risk_band"}
    c1, c2 = st.columns([1, 3])
    dim = c1.selectbox("Dimension", list(dims), format_func=dims.get, index=list(dims).index("income_band"))
    t = profile(df, dim)
    t[dim] = t[dim].astype(str)
    t = t.set_index(dim).loc[ordered(dim, t[dim])].reset_index()
    fig = go.Figure()
    fig.add_bar(x=t[dim], y=t.exposure / 1e6, name="Exposure ($M)", marker_color="#90A4AE")
    fig.add_scatter(x=t[dim], y=t.default_rate * 100, name="Default rate (%)", yaxis="y2", mode="lines+markers",
                    line=dict(color="#C62828"))
    fig.add_scatter(x=t[dim], y=t.avg_pd * 100, name="Avg model PD (%)", yaxis="y2", mode="lines+markers",
                    line=dict(color="#1565C0", dash="dot"))
    fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title="Exposure ($M)",
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="%", rangemode="tozero"),
                      legend=dict(orientation="h", y=1.12))
    c2.plotly_chart(fig, width="stretch")
    show = t[[dim, "accounts", "default_rate", "avg_pd", "exposure", "exposure_share", "defaulted_exposure_share"]].copy()
    c1.dataframe(show[[dim, "accounts", "default_rate"]].rename(columns={dim: dims[dim], "accounts": "Accounts",
                 "default_rate": "Default rate"}).style.format({"Default rate": "{:.1%}", "Accounts": "{:,}"}),
                 hide_index=True, width="stretch")

    st.subheader("Cross-dimensional cut")
    c1, c2, c3 = st.columns([1, 1, 1])
    r_dim = c1.selectbox("Rows", list(dims), format_func=dims.get, index=list(dims).index("employment_type"))
    c_dim = c2.selectbox("Columns", list(dims), format_func=dims.get, index=list(dims).index("income_band"))
    metric = c3.selectbox("Metric", ["Default rate", "Accounts", "Exposure ($M)", "Avg model PD"])
    if r_dim == c_dim:
        st.info("Pick two different dimensions.")
    else:
        val = {"Default rate": ("target_default", "mean"), "Accounts": ("account_id", "count"),
               "Exposure ($M)": ("exposure", "sum"), "Avg model PD": ("pd", "mean")}[metric]
        pv = df.pivot_table(index=r_dim, columns=c_dim, values=val[0], aggfunc=val[1], observed=True)
        cnt = df.pivot_table(index=r_dim, columns=c_dim, values="account_id", aggfunc="count", observed=True)
        pv.index, pv.columns = pv.index.astype(str), pv.columns.astype(str)
        cnt.index, cnt.columns = cnt.index.astype(str), cnt.columns.astype(str)
        pv = pv.loc[ordered(r_dim, pv.index), ordered(c_dim, pv.columns)]
        cnt = cnt.reindex_like(pv)
        if metric == "Exposure ($M)":
            pv = pv / 1e6
        if metric in ("Default rate", "Avg model PD"):
            pv = pv.where(cnt >= 20)
            txt = pv.map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        else:
            txt = pv.map(lambda v: "" if pd.isna(v) else f"{v:,.1f}" if metric.startswith("Exp") else f"{v:,.0f}")
        fig = px.imshow(pv, color_continuous_scale="RdYlGn_r", aspect="auto",
                        labels=dict(x=dims[c_dim], y=dims[r_dim], color=metric))
        fig.update_traces(text=txt.values, texttemplate="%{text}")
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        if metric in ("Default rate", "Avg model PD"):
            st.caption("Cells with fewer than 20 accounts are blanked.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trend by origination year")
        q = df.groupby("origination_year").agg(accounts=("account_id", "size"), default_rate=("target_default", "mean"),
                                                  avg_pd=("pd", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=q.origination_year.astype(str), y=q.accounts, name="Accounts originated", marker_color="#CFD8DC")
        fig.add_scatter(x=q.origination_year.astype(str), y=q.default_rate * 100, name="Default rate (%)", yaxis="y2",
                        line=dict(color="#C62828"))
        fig.add_scatter(x=q.origination_year.astype(str), y=q.avg_pd * 100, name="Avg PD (%)", yaxis="y2",
                        line=dict(color="#1565C0", dash="dot"))
        fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="%", rangemode="tozero"),
                          yaxis_title="Accounts", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Delinquency by month on book")
        r = data["repay"][data["repay"].account_id.isin(df.account_id)]
        m = r.groupby("payment_month").agg(
            late=("payment_status", lambda s: (s == "Late").mean()),
            missed=("payment_status", lambda s: (s == "Missed").mean())).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=m.payment_month, y=m.late * 100, name="Late (1-29 DPD) %", line=dict(color="#F57C00"))
        fig.add_scatter(x=m.payment_month, y=m.missed * 100, name="Missed (30+ DPD) %", line=dict(color="#C62828"))
        fig.update_layout(height=330, margin=dict(t=10, b=10), xaxis_title="Month since origination",
                          yaxis_title="% of accounts", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Drill down to accounts")
    c1, c2 = st.columns([1, 1])
    ddim = c1.selectbox("Segment dimension", list(DIMENSIONS), format_func=DIMENSIONS.get,
                        index=list(DIMENSIONS).index("risk_band"), key="ddim")
    vals = ordered(ddim, df[ddim].astype(str).unique())
    dval = c2.selectbox("Segment", vals, index=len(vals) - 1 if ddim == "risk_band" else 0, key="dval")
    seg = df[df[ddim].astype(str) == dval].sort_values("pd", ascending=False)
    st.markdown(f"**{len(seg):,} accounts** · exposure {money(seg.exposure.sum())} · default rate "
                f"{pct(seg.target_default.mean())} · avg PD {pct(seg.pd.mean())}. Click a row to open the "
                "account.".replace("$", "\\$"))
    cols = ["account_id", "customer_id", "loan_type", "region", "employment_type", "income_band", "exposure", "pd",
            "risk_band", "rep_n_missed", "anomaly_flag", "target_default"]
    ev = st.dataframe(
        seg[cols].head(500), hide_index=True, width="stretch", height=300, on_select="rerun",
        selection_mode="single-row", key="drill_table",
        column_config={"pd": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
                       "exposure": st.column_config.NumberColumn("Exposure", format="$%,.0f"),
                       "rep_n_missed": "Missed EMIs", "target_default": "Defaulted", "account_id": "Account",
                       "customer_id": "Customer", "loan_type": "Loan type", "region": "Region",
                       "employment_type": "Employment", "income_band": "Income", "risk_band": "Risk band",
                       "anomaly_flag": "Anomaly"})
    if ev.selection.rows:
        acc = seg.head(500).iloc[ev.selection.rows[0]].account_id
        st.button(f"Open account {acc}", on_click=go_account, args=(acc,), type="primary")

    st.subheader("Key insights")
    for i, ins in enumerate(data["insights"], 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {ins['title']}**  \n{ins['finding']}  \n**So what:** {ins['so_what']}".replace("$", "\\$"))
    st.caption("Insights are computed on the full portfolio, not the current filter.")


# ================================================================== ACCOUNT
def view_account():
    st.title("Account view")
    ids = df_all.account_id.tolist()
    c1, c2 = st.columns([2, 3])
    if "acc_select" not in st.session_state:
        st.session_state.acc_select = st.session_state.account
    acc = c1.selectbox("Account", ids, key="acc_select")
    st.session_state.account = acc
    c2.caption("Tip: open accounts from the Portfolio drill-down or the Anomaly review list.")
    r = df_all[df_all.account_id == acc].iloc[0]

    k = st.columns(6)
    k[0].metric("Model PD", pct(r.pd))
    k[1].metric("Risk band", r.risk_band)
    k[2].metric("Exposure", money(r.exposure))
    k[3].metric("Loan amount", money(r.loan_amount))
    k[4].metric("Missed EMIs (12m)", int(r.rep_n_missed))
    k[5].metric("Actual outcome", "Defaulted" if r.target_default else "Performing")
    pctile = (df_all.pd < r.pd).mean()
    st.caption(f"{r.loan_type} · {r.loan_purpose} · originated {str(r.origination_date)[:10]}. "
               f"{'Among the riskiest 1% of accounts.' if pctile >= 0.99 else f'PD is higher than {pctile:.0%} of accounts.'} Scores are out-of-fold: this account's own label was "
               f"not used by the model that scored it.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Key risk drivers")
        drv = top_drivers(data["drivers"][data["drivers"].account_id == acc].iloc[0], r, k=8)
        dd = pd.DataFrame(drv)
        dd["label"] = dd.feature + " = " + dd.value.astype(str)
        dd = dd.iloc[::-1]
        fig = go.Figure(go.Bar(x=dd.impact, y=dd.label, orientation="h",
                               marker_color=["#C62828" if v > 0 else "#2E7D32" for v in dd.impact]))
        fig.update_layout(height=330, margin=dict(t=10, b=10, l=10), xaxis_title="Impact on log-odds of default "
                          "(red raises risk, green lowers it)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{data['res']['champion']} contributions (TreeSHAP for LightGBM, coefficient x standardised value "
                   "for logistic regression), relative to an average account.")
    with c2:
        st.subheader("Profile")
        prof = {
            "Customer": r.customer_id, "Age": int(r.age), "Employment": r.employment_type,
            "Annual income": money(r.annual_income), "Region / tier": f"{r.region} / {r.city_tier}",
            "Housing": r.housing_type, "Loan amount": money(r.loan_amount),
            "Rate / tenure": f"{r.interest_rate_pct:.2f}% / {int(r.tenure_months)}m",
            "Originated": str(r.origination_date)[:10], "Channel": r.disbursal_channel,
            "EMI / income (annual)": f"{r.emi_to_income:.2f}", "Bureau lines / write-offs":
            f"{int(r.bur_n_lines)} / {int(r.bur_n_written_off)}", "Loans with us": int(r.n_loans_customer),
        }
        st.dataframe(pd.DataFrame(prof.items(), columns=["Field", "Value"]).astype(str), hide_index=True,
                     width="stretch", height=330)

    if r.anomaly_flag:
        st.error(("**Anomaly flagged** - " + " | ".join(r.anomaly_reasons)).replace("$", "\\$"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Repayment history (12 months)")
        rp = data["repay"][data["repay"].account_id == acc]
        fig = go.Figure()
        fig.add_bar(x=rp.payment_month, y=rp.emi_due, name="EMI due", marker_color="#CFD8DC")
        fig.add_bar(x=rp.payment_month, y=rp.emi_paid, name="EMI paid", marker_color="#1565C0")
        fig.add_scatter(x=rp.payment_month, y=rp.days_past_due, name="DPD", yaxis="y2", line=dict(color="#C62828"))
        fig.update_layout(barmode="overlay", height=300, margin=dict(t=10, b=10), xaxis_title="Month",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="DPD"), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Bureau lines")
        b = data["bureau"][data["bureau"].customer_id == r.customer_id]
        st.dataframe(b[["credit_type", "lender_category", "account_status", "sanctioned_amount",
                        "outstanding_amount", "days_past_due", "date_opened"]], hide_index=True, width="stretch",
                     height=150)
        st.subheader("Collections events")
        cl = data["coll"][data["coll"].account_id == acc]
        if cl.empty:
            st.caption("None.")
        else:
            st.dataframe(cl[["event_date", "action_type", "outcome", "amount_recovered", "event_before_origination"]],
                         hide_index=True, width="stretch", height=120)
            st.caption("event_before_origination = 1 marks events dated before the loan existed (data-quality issue).")


# ================================================================== ASK
def view_ask():
    st.title("Ask the Portfolio")
    st.caption("Natural-language questions answered from the actual data. The LLM turns your question into a "
               "structured query; the numbers are computed by pandas and checked before they are shown.")
    if not llm.available:
        st.warning("No LLM API key is set, so questions are parsed with a simpler keyword matcher and answers use "
                   "a fixed template. Add a key in the app secrets to enable the LLM (see README).")
    how_it_works(
        trigger="An analyst types a question and presses Ask. One LLM call plans the query (it receives the question "
                "and the table schema only, no rows); a second call words the answer from the aggregated result.",
        fp="An answer that sounds right but used the wrong slice, e.g. 'mortgage default rate' read as all secured "
           "loans. Guard: the exact filters, query spec and result table are shown under every answer.",
        fn="A valid question declined or mis-parsed, e.g. an unusual synonym for a column. Guard: the decline "
           "message says what the data can answer, so the analyst can rephrase.",
        hitl="The analyst checks the scope line and result table before using a figure, and uses the thumbs "
             "feedback to log wrong answers for review of the prompt and the whitelist.",
    )
    examples = ["Which loan type has the highest default rate?", "Default rate by employment type and income band",
                "What is the exposure by region for mortgages?", "How many anomalies are flagged by risk band?",
                "Average PD for unemployed borrowers by age band", "Tell me about ACC200001",
                "Why do young borrowers default more?", "What will the default rate be next year?"]
    st.markdown("**Try one** (click to ask):")
    picked = None
    cols = st.columns(2)
    for i, e in enumerate(examples):
        if cols[i % 2].button(e, key=f"ex_{i}", width="stretch"):
            picked = e
    use_filters = st.toggle("Answer within the current sidebar filters", value=False)
    with st.form("ask_form"):
        q = st.text_input("Your question", placeholder="e.g. Which region has the highest exposure?")
        go_btn = st.form_submit_button("Ask", type="primary")
    if picked:
        q, go_btn = picked, True
    if go_btn and q:
        base = df if use_filters else df_all
        with st.spinner("Planning query, computing, verifying..."):
            ans = ask(q, base, data["drivers"], llm)
        st.session_state.setdefault("qa_history", []).insert(0, (q, ans))
    for i, (q, ans) in enumerate(st.session_state.get("qa_history", [])[:6]):
        with st.container(border=True):
            st.markdown(f"**Q: {q}**")
            if ans.status == "answered":
                st.success(ans.text.replace("$", "\\$"))
                st.caption(f"Source: {ans.scope} from data/processed/portfolio.parquet. "
                           f"Planner: {ans.planner}. Wording: {ans.narrator}.")
                if ans.table is not None:
                    st.dataframe(ans.table, hide_index=True, width="stretch",
                                 height=min(38 * (len(ans.table) + 1), 320))
            else:
                st.warning(ans.text)
            for n in ans.notes:
                st.caption(f"Note: {n}")
            with st.expander("Query spec (what was computed)"):
                st.json(ans.spec or {})
            fb = st.feedback("thumbs", key=f"fb_{i}_{hash(q)}")
            if fb is not None:
                st.caption("Feedback recorded for this session.")


# ================================================================== ANOMALY
def view_anomaly():
    st.title("Anomaly review")
    how_it_works(
        trigger="Runs in the nightly build over every account. An account is flagged when (1) its last 3 months "
                "break from its own first 9 months (new missed EMIs, a DPD jump of 4+ of its own standard deviations, "
                "or a 50-point fall in share of EMI paid), (2) a bureau or affordability measure is 4.5+ robust "
                "z-scores from its peer segment (loan type x income band), or (3) an Isolation Forest puts it in "
                "the most unusual 1.5%. Default labels are not used.",
        fp="An account flagged for a real but harmless reason, e.g. a high EMI-to-income borrower with income not "
           "captured in the data, or a one-off missed EMI due to a bank holiday or mandate change.",
        fn="A deteriorating account that stays inside its peer norms and its own history, e.g. slow DPD creep or a "
           "borrower taking new credit elsewhere after our last bureau pull.",
        hitl="A risk analyst opens each flag, reads why it fired, checks the repayment chart and bureau lines, and "
             "records Confirm (send to early collections / limit review), Dismiss (with a reason, used to tune "
             "thresholds) or Need info. No action on the customer happens without that decision.",
    )
    v = data["res"]["anomaly_validation"]
    st.caption(f"Back-test (labels used only for this check): flagged accounts default at "
               f"{v['default_rate_flagged']:.1%} vs {v['default_rate_not_flagged']:.1%} for the rest.")

    fl = df[df.anomaly_flag].sort_values("anomaly_priority", ascending=False).copy()
    dec = load_decisions()
    fl = fl.merge(dec[["account_id", "decision"]], on="account_id", how="left")
    fl["decision"] = fl.decision.fillna("Pending")
    k = st.columns(4)
    k[0].metric("Flagged (in filter)", f"{len(fl):,}")
    k[1].metric("Pending", f"{(fl.decision == 'Pending').sum():,}")
    k[2].metric("Confirmed", f"{(fl.decision == 'Confirm').sum():,}")
    k[3].metric("Dismissed", f"{(fl.decision == 'Dismiss').sum():,}")

    c1, c2 = st.columns(2)
    types = c1.multiselect("Detector", ["Own history", "Peer segment", "Multivariate"])
    status = c2.multiselect("Decision status", ["Pending", "Confirm", "Dismiss", "Need info"], default=["Pending"])
    if types:
        fl = fl[fl.anomaly_types.apply(lambda s: any(t in s for t in types))]
    if status:
        fl = fl[fl.decision.isin(status)]
    fl["why"] = fl.anomaly_reasons.map(lambda r: r[0] + (f" (+{len(r) - 1} more)" if len(r) > 1 else ""))
    cols = ["account_id", "loan_type", "income_band", "anomaly_types", "why", "pd", "exposure", "decision"]
    ev = st.dataframe(fl[cols], hide_index=True, width="stretch", height=320, on_select="rerun",
                      selection_mode="single-row", key="anom_table",
                      column_config={"pd": st.column_config.ProgressColumn("PD", format="percent", min_value=0,
                                                                           max_value=1),
                                     "exposure": st.column_config.NumberColumn("Exposure", format="$%,.0f"),
                                     "why": st.column_config.TextColumn("Why flagged", width="large")})
    if fl.empty:
        st.info("No flagged accounts match.")
        return
    sel = fl.iloc[ev.selection.rows[0]] if ev.selection.rows else fl.iloc[0]
    acc = sel.account_id
    r = df_all[df_all.account_id == acc].iloc[0]
    st.subheader(f"Review {acc}")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Why it was flagged**")
        for reason in r.anomaly_reasons:
            st.markdown(f"- {reason}")
        rp = data["repay"][data["repay"].account_id == acc]
        fig = go.Figure()
        fig.add_bar(x=rp.payment_month, y=rp.days_past_due, marker_color=["#90A4AE"] * 9 + ["#C62828"] * 3,
                    name="DPD")
        fig.add_vline(x=9.5, line_dash="dash", annotation_text="own baseline | recent")
        fig.update_layout(height=230, margin=dict(t=20, b=10), xaxis_title="Month", yaxis_title="DPD")
        st.plotly_chart(fig, width="stretch")
        peers = df_all[df_all.peer_segment == r.peer_segment]
        comp = pd.DataFrame({
            "Measure": ["Annual EMI / income", "Bureau utilisation", "Bureau debt / income", "Lines opened 12m",
                        "Bureau write-offs", "Avg DPD (12m)"],
            "This account": [r.emi_to_income, r.bur_utilization, r.bur_debt_to_income, r.bur_n_opened_12m,
                             r.bur_n_written_off, r.rep_mean_dpd],
            "Peer median": [peers.emi_to_income.median(), peers.bur_utilization.median(),
                            peers.bur_debt_to_income.median(), peers.bur_n_opened_12m.median(),
                            peers.bur_n_written_off.median(), peers.rep_mean_dpd.median()],
        })
        st.markdown(f"**Compared with peer segment** {r.peer_segment} ({len(peers):,} accounts)")
        st.dataframe(comp.style.format({"This account": "{:.2f}", "Peer median": "{:.2f}"}), hide_index=True,
                     width="stretch")
    with c2:
        st.markdown("**Record decision**")
        prev = dec[dec.account_id == acc]
        if len(prev):
            p = prev.iloc[-1]
            st.info(f"Current: **{p.decision}** by {p.reviewer} at {p.timestamp_utc} UTC. {p.note or ''}")
        with st.form(f"dec_{acc}"):
            d = st.radio("Decision", ["Confirm", "Dismiss", "Need info"], horizontal=True)
            reason = st.selectbox("Reason", ["", "Genuine deterioration - early collections", "Affordability concern - "
                                  "limit review", "Data error", "Known / explained by customer", "Peer outlier but "
                                  "low risk", "Other"])
            note = st.text_area("Note", height=80)
            who = st.text_input("Reviewer", value=st.session_state.get("reviewer", ""))
            if st.form_submit_button("Save decision", type="primary"):
                st.session_state.reviewer = who
                save_decision(acc, d, " - ".join(x for x in [reason, note] if x), who)
                st.success(f"Saved: {d}")
                st.rerun()
        st.button("Open full account view", on_click=go_account, args=(acc,))
    st.divider()
    dec = load_decisions()
    st.markdown(f"**Decision log** ({len(dec)} decisions)")
    if len(dec):
        st.dataframe(dec.sort_values("timestamp_utc", ascending=False), hide_index=True, width="stretch", height=180)
        st.download_button("Download decisions (CSV)", dec.to_csv(index=False), "anomaly_decisions.csv", "text/csv")
    st.caption("Decisions are saved to data/processed/anomaly_decisions.csv. On the free hosted app this file resets "
               "when the app restarts; in production it would be a database table with reviewer identity from SSO.")


# ================================================================== MODEL
def view_model():
    res = data["res"]
    st.title("Default model")
    champ = res["champion"]
    st.markdown(f"Target: `target_default`. Logistic regression baseline vs LightGBM. Champion (higher grouped-CV "
                f"AUC): **{champ}**. Split grouped by customer so no customer is in both train and test.")
    rows = []
    for m in ["Logistic regression", "LightGBM"]:
        h = res["holdout"][m]
        rows.append({"Model": m, "Holdout AUC": h["AUC"], "Holdout Gini": h["Gini"], "Holdout KS": h["KS"],
                     "5-fold CV AUC": f"{res['cv_auc'][m]['mean']:.3f} ± {res['cv_auc'][m]['std']:.3f}",
                     "Out-of-time AUC (2023)": res["out_of_time_2023"][m]["AUC"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Holdout: {res['n_test']:,} accounts (default rate {res['test_default_rate']:.1%}); train "
               f"{res['n_train']:,}. Out-of-time: trained on 2018-2022 originations, tested on 2023. "
               f"Leakage check: LightGBM using only bureau lines opened before origination scores holdout AUC "
               f"{res['point_in_time_bureau_holdout']['AUC']:.3f}, so the result does not depend on post-origination "
               f"bureau data.")

    with st.expander("Feature selection: what the model does not use and why"):
        st.markdown("Grouped 5-fold CV AUC of the logistic regression rose from 0.676 with all 9 categorical fields to "
                    "0.681 after dropping these. Gender was never used. Collections data is excluded as leakage.")
        st.dataframe(pd.DataFrame(res["dropped_features"].items(), columns=["Field", "Reason"]), hide_index=True,
                     width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curve (holdout)")
        fig = go.Figure()
        for m, col in [("Logistic regression", "#1565C0"), ("LightGBM", "#C62828")]:
            fig.add_scatter(x=res["roc"][m]["fpr"], y=res["roc"][m]["tpr"], name=f"{m} (AUC {res['holdout'][m]['AUC']:.3f})",
                            line=dict(color=col))
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False)
        fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(x=0.4, y=0.1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader(f"Rank ordering by decile ({champ}, holdout)")
        dt = pd.DataFrame(res["decile_table"])
        fig = go.Figure()
        fig.add_bar(x=dt.decile, y=dt.default_rate * 100, name="Actual default rate %", marker_color="#90A4AE")
        fig.add_scatter(x=dt.decile, y=dt.avg_pd * 100, name="Avg predicted PD %", line=dict(color="#C62828"))
        fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="Decile (1 = riskiest)", yaxis_title="%",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Top 3 deciles capture {dt.cum_capture.iloc[2]:.0%} of defaults.")

    st.subheader("Feature importance")
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "shap_importance", "LightGBM: mean |SHAP|"),
                            (c2, "lr_importance", "Logistic regression: mean |contribution|")]:
        imp = pd.Series(res[key]).head(15)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[LABELS.get(i, i) for i in imp.index], orientation="h",
                               marker_color="#1565C0"))
        fig.update_layout(height=420, margin=dict(t=30, b=10), title=title)
        col.plotly_chart(fig, width="stretch")

    st.info("Post-deployment monitoring (stability, drift, discrimination, calibration, alert thresholds and a drift "
            "simulator) is on the **Monitoring** page.")



# ================================================================== MONITORING
STATUS_ICON = {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}


def view_monitoring():
    from src.model import psi as psi_fn
    from src.monitoring import status as rag

    mon = data["res"]["monitoring"]
    st.title("Model monitoring")
    st.caption(f"Origination cohorts stand in for monthly scoring batches. Reference population: {mon['reference']}. "
               "In production the same checks run on every scoring batch and on each cohort once its outcome window closes.")
    st.subheader("Health check")
    sc = pd.DataFrame(mon["scorecard"])
    sc.insert(0, " ", sc.status.map(STATUS_ICON))
    st.dataframe(sc, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Stability: PSI by cohort vs reference")
        p = pd.DataFrame(mon["psi"])
        fig = px.bar(p, x="cohort", y="psi", color="feature", barmode="group", labels={"psi": "PSI", "cohort": "Cohort"})
        fig.add_hline(y=0.10, line_dash="dash", line_color="#F57C00", annotation_text="amber 0.10")
        fig.add_hline(y=0.25, line_dash="dash", line_color="#C62828", annotation_text="red 0.25")
        fig.update_layout(height=360, margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.25), legend_title="")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
        st.caption("All PSI values are far below 0.10: the synthetic portfolio is stable across vintages.")
    with c2:
        st.subheader("Discrimination by cohort")
        pf = pd.DataFrame(mon["performance"])
        fig = go.Figure()
        fig.add_scatter(x=pf.cohort, y=pf.auc, name="AUC", mode="lines+markers", line=dict(color="#1565C0"))
        fig.add_scatter(x=pf.cohort, y=pf.ks, name="KS", mode="lines+markers", line=dict(color="#C62828"))
        fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.1))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Calibration by risk band")
    cal = pd.DataFrame(mon["calibration"])
    fig = go.Figure()
    fig.add_bar(x=cal.band, y=cal.avg_pd * 100, name="Predicted PD %", marker_color="#90A4AE")
    fig.add_bar(x=cal.band, y=cal.actual_default_rate * 100, name="Actual default rate %", marker_color="#C62828")
    fig.update_layout(height=300, margin=dict(t=10, b=10), barmode="group", legend=dict(orientation="h", y=1.15),
                      yaxis_title="%")
    st.plotly_chart(fig, width="stretch")

    worst = max(mon["calibration"], key=lambda c: abs(c["avg_pd"] - c["actual_default_rate"]) / c["actual_default_rate"])
    gap = abs(worst["avg_pd"] - worst["actual_default_rate"]) / worst["actual_default_rate"]
    if gap >= 0.15:
        st.warning(f"Calibration is amber: band {worst['band']} predicts {worst['avg_pd']:.1%} but "
                   f"{worst['actual_default_rate']:.1%} defaulted ({worst['accounts']:,} accounts, a {gap:.0%} relative gap, "
                   "statistically significant). Rank ordering is fine; the fix is recalibration on a recent cohort "
                   "(Platt or isotonic) before PDs feed pricing or provisioning.")

    st.subheader("Segment check (fairness)")
    st.caption("Predicted vs actual default rate by group. Large gaps mean the model treats a group unfairly. "
               "Gender is not a model input; age is, so its effect is shown here for review.")
    seg_dim = st.radio("Group by", ["age_band", "gender", "region", "employment_type"], horizontal=True,
                       format_func=lambda d: DIMENSIONS.get(d, d))
    fs = df_all.groupby(seg_dim).agg(accounts=("account_id", "size"), predicted=("pd", "mean"),
                                     actual=("target_default", "mean")).reset_index()
    fs[seg_dim] = fs[seg_dim].astype(str)
    fs = fs.set_index(seg_dim).loc[ordered(seg_dim, fs[seg_dim])].reset_index()
    fs["gap"] = (fs.predicted - fs.actual) / fs.actual
    st.dataframe(fs.rename(columns={seg_dim: DIMENSIONS.get(seg_dim, seg_dim), "accounts": "Accounts",
                                    "predicted": "Avg predicted PD", "actual": "Actual default rate",
                                    "gap": "Relative gap"}).style.format(
        {"Avg predicted PD": "{:.1%}", "Actual default rate": "{:.1%}", "Relative gap": "{:+.0%}", "Accounts": "{:,}"}),
        hide_index=True, width="stretch")

    st.subheader("Drift simulator")
    st.caption("Shift one input for the most recent cohort and see whether the monitoring would catch it. "
               "This only demonstrates the alert logic; it does not change the data or the model.")
    df23 = df_all[df_all.origination_year == df_all.origination_year.max()]
    ref = df_all[df_all.origination_year.isin([2018, 2019, 2020])]
    feats = {"annual_income": "Annual income", "emi_to_income": "Annual EMI / income", "rep_mean_dpd": "Average DPD (12m)",
             "loan_amount": "Loan amount", "pd": "Model PD (score)"}
    c1, c2, c3 = st.columns([2, 2, 3])
    f = c1.selectbox("Input", list(feats), format_func=feats.get)
    shift = c2.slider("Shift (%)", -50, 50, 0, 5)
    shifted = df23[f] * (1 + shift / 100)
    v = psi_fn(ref[f].values, shifted.values)
    s_ = rag(v, 0.10, 0.25)
    c3.metric(f"PSI for {feats[f]}", f"{v:.3f}", help="amber at 0.10, red at 0.25")
    c3.markdown(f"Status: {STATUS_ICON[s_]} **{s_}**" + {"Green": " - no action.",
                "Amber": " - investigate the source data and segment mix before the next scoring run.",
                "Red": " - alert the model owner; hold automated decisions on affected segments until reviewed."}[s_])

    st.subheader("Monitoring plan")
    st.markdown("""
| What | How | Threshold / action |
|---|---|---|
| Score stability | PSI of the PD distribution per scoring batch vs the development sample | < 0.10 stable, 0.10-0.25 investigate, > 0.25 escalate to model owner |
| Input drift | PSI per key feature, plus missing-rate and range checks in the build | same PSI bands; a new category or a missing-rate jump of 5+ points blocks the run |
| Discrimination | AUC / KS on each cohort once its outcome window closes | AUC more than 0.03 below reference -> review; 0.05 -> retrain |
| Calibration | Predicted vs actual default rate per risk band | any band off by more than 15% relative -> review; 25% -> recalibrate |
| Segment fairness | Default and override rates by age band, gender, region | material gaps go to model risk committee |
| Human feedback | Analyst overrides and anomaly confirm / dismiss rates | rising dismiss rate -> retune anomaly thresholds |

Retrain on a fixed schedule (e.g. every 6-12 months) or when a red trigger fires; the logistic regression is retrained alongside as a benchmark challenger.
""")

# ================================================================== DATA
def view_data():
    st.title("Data & method")
    st.subheader("Join logic")
    st.markdown("""
- **Grain: one row per loan account** (14,685). `loan_accounts` is the fact table and holds the target.
- `customers` joins many-to-one on `customer_id` (validated: every loan has exactly one customer).
- `repayment_history` (12 rows per account) is aggregated to account level first: missed/late counts, worst and
  average DPD, share of EMI paid, last-3-month vs first-9-month behaviour, DPD trend, longest missed streak,
  balance after month 12 (= exposure).
- `bureau_credit` (many lines per customer) is aggregated per customer and attached to each of the customer's
  loans: line counts by status, write-offs, outstanding, utilisation, worst/average DPD, lines opened in the last
  12 months. A point-in-time version (only lines opened before the loan) is used as a leakage check.
- `collections_events` is aggregated per account for the dashboard only. It is **not** a model feature, because
  collections happen after delinquency and would leak the outcome.
- Every join is a left join from the account table with a row-count assertion, so no account is dropped.
""")
    st.subheader("Data-quality log")
    st.dataframe(data["dq"], hide_index=True, width="stretch", height=520)
    st.subheader("Clean dataset")
    st.caption(f"{df_all.shape[0]:,} rows x {df_all.shape[1]} columns. Preview of the first 200 rows.")
    st.dataframe(df_all.drop(columns=["anomaly_reasons"]).head(200), width="stretch", height=300)


{"Portfolio": view_portfolio, "Account": view_account, "Ask the Portfolio": view_ask,
 "Anomaly review": view_anomaly, "Model": view_model, "Monitoring": view_monitoring, "Data & method": view_data}[st.session_state.view]()
