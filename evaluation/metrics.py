# ------------------------------------------
# PART 5 - EVALUATION METRICS
# ------------------------------------------
#   field_accuracy()   extracted profile vs the persona's known answers, field by field
#   invented_numbers() numbers in a reply that do not appear in any code / tool output available for that
#                      turn (or in the client's own messages)
#   holdings_match()   final whole units vs the Part 3 output file (part3_units_2026-11-18.csv)
from __future__ import annotations

import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- 1. field accuracy ------------------------------------------------------------------------------
SCORED_FIELDS = ["age", "employment", "num_dependants", "income", "invest_amount", "other_assets",
                 "cash_needed_now", "objective_str", "target_yn", "target", "target_years", "exp",
                 "rr_familiarity", "desc", "principal_protection", "sectors"]


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.replace("–", "-").replace("—", "-")).strip().lower()
    return v


def field_accuracy(answers: dict, truth: dict) -> dict:
    """{field: True/False}. `target` is scored only when the persona has a target.
    sectors: the exclusion screen the answer produces (tickers) must equal the persona's.
    other_assets (free text): scored on the key fact (the emergency fund) being captured."""
    import market_data
    import portfolio_agent
    meta = market_data.load()["meta"]
    out = {}
    for f in SCORED_FIELDS:
        if f == "target" and truth.get("target_yn") != "Yes":
            continue
        got, want = answers.get(f), truth.get(f)
        if f == "sectors":
            out[f] = got is not None and sorted(portfolio_agent.parse_exclusions(got, meta)[0]) == sorted(want)
        elif f == "other_assets":
            out[f] = isinstance(got, str) and want in got.lower()
        else:
            out[f] = got is not None and _norm(got) == _norm(want)
    return out


# ---- 2. invented numbers ----------------------------------------------------------------------------
def numbers_in(text: str) -> list[str]:
    """Same normalisation as the compliance checker: '$19,925.47' -> '19925.47', '-2.4%' -> '2.4'."""
    out = []
    for m in re.findall(r"\d[\d,]*\.?\d*", text):
        n = m.replace(",", "").rstrip(".")
        if n:
            out.append(str(float(n)))
    return out


def _exempt(n: str) -> bool:
    x = float(n)
    return (x.is_integer() and x <= 10) or (x.is_integer() and 1900 <= x <= 2100)   # small counts, calendar years


def invented_numbers(reply: str, sources: list) -> list[str]:
    """Numbers in `reply` that appear in none of `sources` (code/tool outputs and the client's messages)."""
    allowed = set(numbers_in(json.dumps(sources, default=str, ensure_ascii=False)))
    return [n for n in numbers_in(reply) if not _exempt(n) and n not in allowed]


# ---- 3. holdings vs Part 3 output ------------------------------------------------------------------
def part3_units(key: str) -> dict:
    df = pd.read_csv(os.path.join(ROOT, "data", "part3", "part3_units_2026-11-18.csv"))
    df.columns = ["key", "ticker"] + list(df.columns[2:])
    rows = df[(df["key"] == key) & (df["Units"] > 0)]
    return {r.ticker: int(round(r.Units)) for r in rows.itertuples()}


def holdings_match(portfolio: dict | None, key: str) -> dict:
    """Exact match of (ticker, units) between the chatbot's final orders and the Part 3 file."""
    want = part3_units(key)
    if portfolio is None:
        return {"shown": False, "exact": False, "matching_positions": 0, "expected_positions": len(want)}
    got = {h["ticker"]: h["units"] for h in portfolio["holdings"] if h["units"] > 0}
    same = sum(1 for t, u in want.items() if got.get(t) == u)
    return {"shown": True, "strategy": portfolio["strategy"], "exact": got == want, "matching_positions": same,
            "expected_positions": len(want), "got": got, "expected": want}
