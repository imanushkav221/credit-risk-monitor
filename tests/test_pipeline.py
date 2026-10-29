"""Run with:  pytest -q   (requires data/processed from `python -m src.build`)."""
import json

import pandas as pd
import pytest

from src import config as C
from src.ask import ask, validate, verify, SpecError
from src.ingest import ingest
from src.features import build_portfolio


@pytest.fixture(scope="session")
def df():
    return pd.read_parquet(C.PROCESSED / "portfolio.parquet")


def test_ingest_keeps_every_account():
    t, log = ingest()
    out = build_portfolio(t, log)
    assert len(out) == 14685 and out.account_id.is_unique
    assert set(out.gender.unique()) == {"Male", "Female"}
    assert out[["years_employed", "num_dependents", "interest_rate_pct", "years_at_residence"]].notna().all().all()
    assert len(log.frame()) > 20


def test_model_metrics_reasonable():
    r = json.load(open(C.PROCESSED / "model_results.json"))
    for m in ("Logistic regression", "LightGBM"):
        assert 0.6 < r["holdout"][m]["AUC"] < 0.9
        assert abs(r["holdout"][m]["Gini"] - (2 * r["holdout"][m]["AUC"] - 1)) < 1e-3


def test_ask_answer_matches_data(df):
    a = ask("Which loan type has the highest default rate?", df, llm=_NoLLM())
    expected = df.groupby("loan_type").target_default.mean().idxmax()
    assert a.status == "answered" and a.text.startswith(expected)


def test_ask_declines_forecast(df):
    assert ask("What will the default rate be next year?", df, llm=_NoLLM()).status == "declined"


def test_validate_rejects_unknown_column(df):
    with pytest.raises(SpecError):
        validate({"answerable": True, "metrics": ["default_rate"], "group_by": ["credit_score"]}, df)


def test_verify_catches_invented_number(df):
    show = pd.DataFrame({"Loan type": ["Mortgage"], "Default rate": ["24.7%"], "Accounts": ["1,432"]})
    assert verify("Mortgage defaults at 24.7% across 1,432 accounts.", show, "q") == []
    assert verify("Mortgage defaults at 31.2%.", show, "q") == ["31.2%"]


class _NoLLM:
    available = False

    def label(self):
        return "off"


class _FakeLLM:
    """Plans correctly but invents a number when narrating."""
    available = True
    provider, model = "fake", "fake-model"

    def __init__(self, narration):
        self.narration = narration

    def label(self):
        return "fake"

    def complete(self, system, user, json_mode=False, **kw):
        if json_mode:
            return json.dumps({"answerable": True, "intent": "aggregate", "metrics": ["default_rate"],
                               "group_by": ["loan_type"], "filters": [], "sort_by": "default_rate", "sort": "desc",
                               "top_n": None})
        return self.narration


def test_llm_hallucination_is_replaced(df):
    a = ask("Which loan type defaults most?", df, llm=_FakeLLM("Mortgages default at 42.0%."))
    assert "42.0%" not in a.text and a.narrator == "template" and a.notes


def test_llm_grounded_answer_is_kept(df):
    rate = f"{df[df.loan_type == 'Mortgage'].target_default.mean() * 100:.1f}%"
    a = ask("Which loan type defaults most?", df, llm=_FakeLLM(f"Mortgage has the highest default rate, {rate}."))
    assert rate in a.text and a.narrator.startswith("LLM")


@pytest.mark.parametrize("prov,key,resp", [
    ("anthropic", "ANTHROPIC_API_KEY", {"content": [{"type": "text", "text": "hi"}]}),
    ("openai", "OPENAI_API_KEY", {"choices": [{"message": {"content": "hi"}}]}),
    ("gemini", "GEMINI_API_KEY", {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}),
])
def test_llm_providers_parse(monkeypatch, prov, key, resp):
    import src.llm as L

    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(key, "test")
    calls = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return resp

    def fake_post(url, **kw):
        calls["url"], calls["json"] = url, kw["json"]
        return R()

    monkeypatch.setattr(L.requests, "post", fake_post)
    llm = L.LLM()
    assert llm.provider == prov and llm.complete("sys", "user") == "hi"
    if prov == "anthropic":
        assert "temperature" not in calls["json"]


def test_llm_falls_back_to_next_model_on_rate_limit(monkeypatch):
    import src.llm as L

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    seen = []

    class R:
        def __init__(self, code):
            self.status_code, self.text = code, "busy"

        def raise_for_status(self):
            if self.status_code != 200:
                raise L.requests.HTTPError(response=self)

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    def fake_post(url, **kw):
        seen.append(url)
        return R(429 if len(seen) == 1 else 200)

    monkeypatch.setattr(L.requests, "post", fake_post)
    llm = L.LLM()
    assert llm.complete("s", "u") == "ok" and llm.last_model != llm.model and len(seen) == 2
