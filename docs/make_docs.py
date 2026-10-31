"""Generates docs/architecture.pdf and docs/solution_overview.pdf (one page each).
Run after `python -m src.build`:  python docs/make_docs.py"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Table, TableStyle, SimpleDocTemplate, Spacer
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "data/processed/model_results.json"))
INS = json.load(open(ROOT / "data/processed/insights.json"))

DARK, RED, BLUE, GREY, GREEN = (colors.HexColor(c) for c in ("#263238", "#C62828", "#1565C0", "#ECEFF1", "#2E7D32"))


# ------------------------------------------------------------------ architecture
def box(c, x, y, w, h, title, lines, fill=GREY, stroke=DARK, tcol=DARK):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(tcol)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 5, y + h - 12, title)
    c.setFont("Helvetica", 6.8)
    c.setFillColor(DARK)
    yy = y + h - 22
    for ln in lines:
        c.drawString(x + 5, yy, ln)
        yy -= 8.6


def arrow(c, x1, y1, x2, y2, col=DARK, label=None):
    import math

    c.setStrokeColor(col)
    c.setFillColor(col)
    c.setLineWidth(1.1)
    c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 6 * math.cos(a - 0.4), y2 - 6 * math.sin(a - 0.4))
    p.lineTo(x2 - 6 * math.cos(a + 0.4), y2 - 6 * math.sin(a + 0.4))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    if label:
        c.setFont("Helvetica-Oblique", 6.3)
        c.drawString((x1 + x2) / 2 + 3, (y1 + y2) / 2 + 2, label)


def architecture(path):
    W, H = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setTitle("Architecture - Credit Risk Portfolio Monitor")
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(DARK)
    c.drawString(20, H - 30, "Architecture: Credit Risk Portfolio Monitor")
    c.setFont("Helvetica", 8)
    c.drawString(20, H - 43, "Batch build (python -m src.build) produces everything the app reads; the Streamlit app "
                             "is read-only except for anomaly decisions. Tool or library shown at each stage.")

    bw, bh, top = 150, 118, H - 185
    xs = [20, 185, 350, 515, 680]
    box(c, xs[0], top, bw, bh, "1. Raw data  (data/raw/)", [
        "5 CSVs as provided (pandas.read_csv)", "customers  10,000", "loan_accounts  14,685 (target)",
        "bureau_credit  22,979", "repayment_history  176,220", "collections_events  8,635"])
    box(c, xs[1], top, bw, bh, "2. Ingestion  src/ingest.py", [
        "pandas: typed load, date parsing", "PK uniqueness + orphan-key checks", "Fix: gender M/F, missing values",
        "  (median + missing flags), implausible", "  tenure, bad collections dates", "Output: DQ log (every decision)"])
    box(c, xs[2], top, bw, bh, "3. Features  src/features.py", [
        "Grain: 1 row per loan account", "Repayment 12m -> DPD, missed, pay", "  ratio, last-3 vs first-9 months",
        "Bureau -> write-offs, DPD, utilisation", "Left joins + row-count assertion", "Clean dataset: 14,685 x 73"])
    box(c, xs[3], top + 62, bw, 56, "4a. Analytics  src/analytics.py", [
        "pandas group-bys by income, employment,", "region, age, loan type; 5 insights", "computed from data (no hand numbers)"])
    box(c, xs[3], top - 4, bw, 64, "4b. Model  src/model.py", [
        "scikit-learn LogisticRegression (baseline)", "LightGBM (challenger), GroupKFold by", "customer, AUC/KS/Gini, TreeSHAP,",
        "OOF PD -> bands A-E; monitoring.py: PSI,", "cohort AUC/KS, calibration, RAG status"])
    box(c, xs[3], top - 70, bw, 60, "4c. Anomaly  src/anomaly.py", [
        "Own history: last 3m vs first 9m", "Peer segment: robust z (loan type x", "income band)",
        "scikit-learn IsolationForest (top 1.5%)"], fill=colors.HexColor("#FFF3E0"), tcol=RED)
    box(c, xs[4], top, bw + 8, bh, "5. Processed store  data/processed/", [
        "portfolio.parquet (pyarrow)", "drivers.parquet (per-account SHAP)", "repayment / bureau / collections",
        "model_results.json, insights.json", "dq_log.csv", "anomaly_decisions.csv (written by app)"])
    for i in range(2):
        arrow(c, xs[i] + bw, top + bh / 2, xs[i + 1], top + bh / 2)
    arrow(c, xs[2] + bw, top + bh / 2, xs[3], top + 90)
    arrow(c, xs[2] + bw, top + bh / 2, xs[3], top + 26)
    arrow(c, xs[2] + bw, top + bh / 2, xs[3], top - 40)
    for yy in (top + 90, top + 26, top - 40):
        arrow(c, xs[3] + bw, yy, xs[4], top + bh / 2)

    # app layer
    ay, ah = 40, 215
    c.setStrokeColor(BLUE)
    c.setDash(3, 2)
    c.roundRect(15, ay - 8, W - 30, ah + 22, 6, stroke=1, fill=0)
    c.setDash()
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(22, ay + ah, "6. Dashboard: Streamlit (app/app.py) + Plotly, hosted on Streamlit Community Cloud; "
                              "API key in the Cloud secrets store, never in Git")
    box(c, 22, ay + 112, 150, 82, "Portfolio view", ["KPIs, risk bands x exposure,", "default rate by any dimension,",
                                                     "cross-cut heatmap, vintage trend,", "sidebar filters, drill-down",
                                                     "table -> account"])
    box(c, 22, ay + 10, 150, 92, "Account / Model / Monitoring", ["PD, band, top SHAP drivers,", "12m repayment chart,",
                                                                  "bureau + collections; ROC, deciles,",
                                                                  "importance; PSI, calibration, RAG", "health check, drift simulator"])
    # ask flow
    fx, fy, fw, fh = 185, ay + 118, 96, 76
    steps = [
        ("Question", ["Analyst types in", "Ask the Portfolio"], GREY, DARK),
        ("LLM call 1: plan", ["SENT: question +", "schema (columns,", "allowed values).", "NO rows."],
         colors.HexColor("#E3F2FD"), BLUE),
        ("Validate", ["Whitelist metrics,", "dimensions, values,", "operators; else", "decline"], GREY, DARK),
        ("Execute (pandas)", ["Exact numbers from", "portfolio.parquet;", "result table"], GREY, DARK),
        ("LLM call 2: word", ["SENT: question +", "aggregated result", "(<=30 rows)"], colors.HexColor("#E3F2FD"), BLUE),
        ("Verify numbers", ["Every number must be", "in the result, else", "template answer;", "show source + table"],
         GREY, DARK),
    ]
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(fx, fy + fh + 6, "AI feature 1: Ask the Portfolio (src/ask.py, src/llm.py: HTTPS to Gemini / Claude / "
                                  "OpenAI; no key -> keyword parser + template)")
    for i, (t, ls, f, tc) in enumerate(steps):
        x = fx + i * (fw + 10)
        box(c, x, fy, fw, fh, t, ls, fill=f, tcol=tc)
        if i:
            arrow(c, x - 10, fy + fh / 2, x, fy + fh / 2, col=BLUE)
    # anomaly flow
    gy = ay + 10
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(fx, gy + 84, "AI feature 2: Anomaly review (no LLM; no target labels used)")
    a_steps = [
        ("Flags from build", ["3 detectors, reasons", "in plain English,", "priority score"]),
        ("Review queue", ["Filter by detector /", "status; select row"]),
        ("Evidence", ["Why flagged, DPD", "baseline vs recent,", "account vs peer median"]),
        ("Analyst decision", ["Confirm / Dismiss /", "Need info + reason,", "reviewer, timestamp"]),
        ("Decision log", ["anomaly_decisions.csv", "+ CSV download; feeds", "threshold tuning"]),
    ]
    for i, (t, ls) in enumerate(a_steps):
        x = fx + i * (fw + 10)
        box(c, x, gy, fw, 74, t, ls, fill=colors.HexColor("#FFF3E0"), tcol=RED)
        if i:
            arrow(c, x - 10, gy + 37, x, gy + 37, col=RED)
    c.setFont("Helvetica", 6.8)
    c.setFillColor(DARK)
    c.drawString(fx + 5 * (fw + 10), gy + 50, "Human in the loop:")
    c.drawString(fx + 5 * (fw + 10), gy + 41, "no customer action")
    c.drawString(fx + 5 * (fw + 10), gy + 32, "without a decision")
    arrow(c, xs[4] + 70, top, xs[4] + 70, ay + ah + 14, col=BLUE, label="app reads (st.cache_data)")
    c.save()


# ------------------------------------------------------------------ overview
def overview(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=9 * mm,
                            bottomMargin=8 * mm, title="Solution overview - Credit Risk Portfolio Monitor")
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=DARK)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.4, leading=11.5, textColor=RED, spaceBefore=5,
                        spaceAfter=1)
    p = ParagraphStyle("p", fontName="Helvetica", fontSize=8, leading=9.9, textColor=DARK)
    sm = ParagraphStyle("sm", parent=p, fontSize=7.6, leading=9.2)
    ho, cv, oot = RES["holdout"], RES["cv_auc"], RES["out_of_time_2023"]
    av = RES["anomaly_validation"]
    champ = RES["champion"]
    S = []
    S.append(Paragraph("Solution overview: Credit Risk Portfolio Monitor", h1))
    S.append(Paragraph("Streamlit dashboard over 5 synthetic bank tables (14,685 loan accounts, 10,000 customers). "
                       "Repo layout and run commands are in README.md.", sm))

    S.append(Paragraph("What the dashboard shows and how an analyst uses it", h2))
    views = [
        ["Portfolio", "KPIs; exposure and actual default rate by model risk band (A-E); exposure by loan type x band; default "
                      "rate, PD and exposure by any dimension; cross-cut heatmap (any two dimensions); vintage and month-on-book "
                      "trends; drill-down from a segment to its accounts. Sidebar filters (loan type, region, employment, income, "
                      "band, year) apply throughout. Use: find where risk and money concentrate, then drill in."],
        ["Account", "PD, band, percentile, top risk drivers for the account, profile, 12-month EMI/DPD chart, bureau lines, "
                    "collections, anomaly reasons. Use: explain a score before a limit, collections or override decision."],
        ["Ask the Portfolio", "Plain-English questions with exact figures, the scope used, the result table and the query spec."],
        ["Anomaly review", "Prioritised queue of flagged accounts, why each was flagged, peer comparison, decision + log."],
        ["Model / Monitoring", "Metrics, ROC, deciles, importance; monitoring health check (PSI, cohort AUC/KS, calibration by band, RAG thresholds) and a drift simulator."], ["Data & method", "Join logic and the full data-quality log."],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", sm), Paragraph(b, sm)] for a, b in views], colWidths=[26 * mm, 160 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.lightgrey)]))
    S.append(t)

    S.append(Paragraph("The two AI features", h2))
    rows = [
        ["", "Ask the Portfolio (LLM)", "Anomaly detection"],
        ["How it works",
         "LLM call 1 turns the question into a JSON query using only the schema (no rows). Code validates it against a "
         "whitelist and computes the answer with pandas. LLM call 2 words a 2-3 sentence answer from the aggregated result; "
         "every number is checked against the result, otherwise a template answer is used. Out-of-scope questions "
         "(forecasts, 'why', outside facts, personal data) are declined. Works without a key via a keyword parser.",
         "Three detectors, no default labels: (1) own history: months 10-12 vs the account's months 1-9 (new missed EMIs, "
         "DPD jump of 4+ of its own SDs, 50-pt fall in share paid); (2) peer segment: robust z >= 4.5 vs loan type x "
         "income band on EMI/income, bureau utilisation, debt/income, new lines, write-offs; (3) Isolation Forest, top 1.5%. "
         f"{av['flagged']} accounts flagged; back-test default rate {av['default_rate_flagged']:.1%} vs "
         f"{av['default_rate_not_flagged']:.1%} unflagged."],
        ["Trigger", "Analyst submits a question.", "Every build (nightly in production) over all accounts."],
        ["False positive", "A confident answer on the wrong slice (e.g. a synonym mapped to the wrong column).",
         "A harmless outlier, e.g. high EMI/income with income not captured, or a one-off missed EMI."],
        ["False negative", "A valid question declined or mis-parsed.",
         "Slow deterioration that stays inside own and peer norms; new borrowing after the last bureau pull."],
        ["Human review", "Scope line, result table and query spec shown under each answer; thumbs feedback.",
         "Analyst records Confirm / Dismiss / Need info with reason; no customer action without it; dismiss "
         "reasons retune thresholds."],
    ]
    t = Table([[Paragraph(f"<b>{c}</b>" if i == 0 or j == 0 else c, sm) for j, c in enumerate(r)] for i, r in enumerate(rows)],
              colWidths=[21 * mm, 82.5 * mm, 82.5 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (-1, 0), GREY), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    S.append(t)

    S.append(Paragraph("Model results and approach", h2))
    mt = [["Model", "Holdout AUC", "KS", "Gini", "5-fold CV AUC", "Out-of-time AUC (2023)"]]
    for m in ["Logistic regression", "LightGBM"]:
        mt.append([m + (" (champion)" if m == champ else ""), f"{ho[m]['AUC']:.3f}", f"{ho[m]['KS']:.3f}",
                   f"{ho[m]['Gini']:.3f}", f"{cv[m]['mean']:.3f} ± {cv[m]['std']:.3f}", f"{oot[m]['AUC']:.3f}"])
    t = Table(mt, colWidths=[42 * mm, 22 * mm, 16 * mm, 16 * mm, 30 * mm, 36 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), GREY),
                           ("TOPPADDING", (0, 0), (-1, -1), 0.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8)]))
    S.append(t)
    S.append(Paragraph(
        "35 numeric features (12-month repayment behaviour, bureau aggregates, affordability ratios, age, income, missing "
        "flags) + loan type and employment type. Seven weak categoricals (marital status, education, housing, purpose, "
        "channel, region, city tier) were dropped: CV AUC rose 0.676 -> 0.681 and fair-lending risk fell; gender never used. "
        "Collections excluded (leakage). Split grouped by customer; champion by grouped-CV AUC: the logistic regression, "
        "an explainable scorecard; LightGBM kept as challenger. Top drivers: income, late payments, loan amount, bureau "
        f"lines and DPD, years employed, age. Leakage check (bureau lines opened before origination only): AUC "
        f"{RES['point_in_time_bureau_holdout']['AUC']:.3f}. Monitoring page: PSI, cohort AUC/KS, calibration by band "
        "(band A currently amber, under-predicts), fairness by segment, drift simulator. Expected loss = PD x assumed "
        "LGD x exposure. "
        "", p))

    S.append(Paragraph("Top 3 portfolio insights", h2))
    for i, ins in enumerate(INS[:3], 1):
        S.append(Paragraph(f"<b>{i}. {ins['title']}.</b> {ins['finding']} <i>So what:</i> {ins['so_what']}", p))

    S.append(Paragraph("Key assumptions", h2))
    S.append(Paragraph(
        "Data is a snapshot at 2025-12-31; exposure = outstanding balance after month 12; target_default is final and its "
        "observation window is unknown; bureau is the latest snapshot (a behavioural, monitoring-style score), checked "
        "against a point-in-time variant; 7,293 loans predating customer_since, 3,875 collections events before "
        "origination and 3,149 future close dates are treated as data errors and flagged, not dropped.", p))
    S.append(Paragraph("What I would need from the bank's IT and data teams for production", h2))
    S.append(Paragraph(
        "(1) Governed feeds from core banking, the bureau and the collections system into the warehouse, with point-in-time "
        "bureau archives and a defined default definition and outcome window. (2) A scheduled pipeline (Airflow/dbt) with "
        "data-quality tests that block bad loads. (3) Model registry, validation sign-off and a scoring service. (4) An "
        "approved LLM endpoint inside the bank's cloud tenancy with no data retention, plus logging of prompts and answers. "
        "(5) SSO and role-based access for the dashboard, and a decisions table in the database instead of a CSV. (6) Model "
        "risk policy for monitoring thresholds, fairness review and retraining.", p))
    doc.build(S)


if __name__ == "__main__":
    architecture(ROOT / "docs/architecture.pdf")
    overview(ROOT / "docs/solution_overview.pdf")
    print("written docs/architecture.pdf, docs/solution_overview.pdf")
