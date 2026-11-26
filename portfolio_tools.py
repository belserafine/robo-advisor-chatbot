# ------------------------------------------
# PORTFOLIO AGENT TOOLS (deterministic, no LLM)
# ------------------------------------------
# Extra tools the chatbot calls during a conversation. Every number they return comes from
# the Part 2 data and the Part 3 optimizer - the LLM only explains them.
#   fractional_portfolio(answers, strategy)   Part 3 Section 8.1: no whole-unit restriction
#   esg_cost(answers)                         Part 3 Section 9: same model with vs without exclusions
#   stress_test_2022(answers)                 2022 rate-hiking sell-off, in-sample weekly prices
#   asset_stats(tickers=None)                 Part 2 summary statistics per asset
#   small_budget_diagnostics(answers)         why a budget is too small for whole units
#   fx_rate(currency)                         currency -> CAD rate (offline snapshot by default)
from __future__ import annotations

import json
import os
from datetime import date

import numpy as np
import pandas as pd

import market_data
import optimizer as opt
import portfolio_agent as pa

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STRESS_WINDOW = ("2022-01-03", "2022-10-17")   # Part 4: weekly returns labelled 3 Jan - 17 Oct 2022 (42 weeks)


def _setup(answers: dict):
    d = market_data.load()
    T, meta, mu, Q = d["tickers"], d["meta"], d["mu"].values, d["Q"].values
    profile, info = pa.answers_to_profile(answers, meta)
    inv = pa._investing_profile(profile, info["cash_reserve"])
    pi, _ = opt.implied_returns(Q, T)
    return d, T, meta, Q, {"S1": mu, "S2": pi}, profile, info, inv


def _weights_table(w, T, meta, min_w=1e-4):
    return [{"ticker": t, "name": meta.loc[t, "Name"], "weight": round(float(x), 4)}
            for t, x in sorted(zip(T, w), key=lambda z: -z[1]) if x > min_w]


# ---------------------------------------------------------------------------------------------
def fractional_portfolio(answers: dict, strategy: str = "S2") -> dict:
    """Continuous optimum without the whole-unit investability filter (needs a broker that sells
    fractional units), compared with the whole-unit recommendation."""
    d, T, meta, Q, means, profile, info, inv = _setup(answers)
    m = means[strategy]
    w = opt.markowitz_max_return(m, inv, Q, T, meta)
    prices = d["px_out"].iloc[-1].values
    budget = inv["budget"]
    units = w * budget / (prices * (1 + opt.TC))
    value = units * prices
    try:
        whole = pa.get_portfolio(answers, strategy)
        whole_w = np.zeros(len(T))
        for h in whole["holdings"]:
            whole_w[T.index(h["ticker"])] = h["target_weight"]
        whole_units = {"n_holdings": sum(1 for h in whole["holdings"] if h["units"] > 0),
                       "invested": whole["invested"], "cash_residual": whole["cash_residual"],
                       "expected_return": round(float(m @ whole_w), 4),
                       "dropped_not_buyable": whole["comparison"][strategy]["dropped_not_buyable"]}
    except ValueError as e:
        whole_units = {"available": False, "reason": str(e)}
    return {
        "strategy": pa.STRATEGY_NAMES[strategy],
        "fractional": {
            "holdings": [{"ticker": t, "name": meta.loc[t, "Name"], "units": round(float(u), 4),
                          "value_cad": round(float(v), 2), "weight": round(float(x), 4)}
                         for t, u, v, x in sorted(zip(T, units, value, w), key=lambda z: -z[3]) if x > 1e-4],
            "n_holdings": int((w > 1e-4).sum()),
            "invested": round(float(value.sum()), 2),
            "fees": round(float(value.sum() * opt.TC), 2),
            "expected_return": round(float(m @ w), 4),
        },
        "whole_units": whole_units,
        "note": "The fractional version is a different, less constrained optimum (assets too expensive for one "
                "whole unit are kept), not the same weights with decimals. Fractional trading depends on the broker.",
    }


# ---------------------------------------------------------------------------------------------
def esg_cost(answers: dict) -> dict:
    """Expected return of the same model with and without the client's exclusions (continuous weights,
    as in Part 3 Section 9), and which excluded assets the optimizer would hold if allowed."""
    d, T, meta, Q, means, profile, info, inv = _setup(answers)
    if not inv["exclude"]:
        return {"has_exclusions": False, "note": "The client has no exclusions, so there is no ESG cost."}
    free = dict(inv, exclude=[])
    out = {"has_exclusions": True, "excluded_tickers": list(inv["exclude"]),
           "excluded_because": info["exclusion_reasons"], "strategies": {}}
    for s in ("S1", "S2"):
        m = means[s]
        w_esg = opt.markowitz_max_return(m, inv, Q, T, meta)
        w_free = opt.markowitz_max_return(m, free, Q, T, meta)
        out["strategies"][pa.STRATEGY_NAMES[s]] = {
            "expected_return_with_exclusions": round(float(m @ w_esg), 4),
            "expected_return_without_exclusions": round(float(m @ w_free), 4),
            "cost_percentage_points": round(float(m @ w_free - m @ w_esg) * 100, 2),
            "volatility_cap": inv["vol_cap"],
            "excluded_assets_held_when_allowed": [
                {"ticker": t, "name": meta.loc[t, "Name"], "weight": round(float(x), 4)}
                for t, x in zip(T, w_free) if t in inv["exclude"] and x > 0.005],
        }
    out["note"] = ("Same volatility cap and constraints in both runs; the only change is the exclusion list. "
                   "Whether the trade-off is worth it is a personal values decision.")
    return out


# ---------------------------------------------------------------------------------------------
def _part4_weights(strategy, inv, mu_in, Q_in, T, meta, prices):
    """Part 4 build_weights(): same constraints and investability filter; if the volatility cap is infeasible
    under these inputs, the minimum-variance portfolio is used."""
    m = mu_in if strategy == "S1" else opt.implied_returns(Q_in, T)[0]
    status = {"fallback": False}

    def solve(q):
        try:
            return opt.markowitz_max_return(m, q, Q_in, T, meta)
        except (ValueError, AssertionError):
            status["fallback"] = True
            return opt.min_variance(q, Q_in, T, meta)
    w, _, _ = opt.investable(solve, inv, prices, T)
    return w, status["fallback"]


def stress_test_2022(answers: dict | None = None) -> dict:
    """Part 4 Section 4: 2022 interest-rate sell-off. Weekly log returns labelled 3 Jan - 17 Oct 2022 (US equity peak
    to trough), buy-and-hold at the portfolio weights built from 1 Apr 2026 prices. Also re-estimates mu, Q and pi
    WITHOUT the stress weeks (hindsight check), because 2022 is inside the estimation period. COVID (2020) is not
    covered: the data starts in April 2021."""
    d = market_data.load(); T, meta, Q, mu = d["tickers"], d["meta"], d["Q"].values, d["mu"].values
    ret = d["ret_ins"]
    win = ret.loc[STRESS_WINDOW[0]:STRESS_WINDOW[1]]
    after = ret.loc[STRESS_WINDOW[0]:]
    ret_ex = ret.drop(index=win.index)
    mu_ex, Q_ex = ret_ex.mean().values * 52, ret_ex.cov().values * 52

    def path(w, r):
        return np.r_[1.0, np.exp(r.cumsum().values) @ w]

    def metrics(w, budget):
        V = path(w, win)
        weekly = V[1:] / V[:-1] - 1
        t_i = int(np.argmin(V))
        rec = np.where(path(w, after)[t_i:] >= 1.0)[0]
        out = {"total_return": round(float(V[-1] - 1), 6),
               "max_drawdown": round(float((V / np.maximum.accumulate(V) - 1).min()), 6),
               "worst_week": round(float(weekly.min()), 6), "weeks_to_trough": t_i,
               "weeks_to_recover": int(rec[0]) if len(rec) else "not recovered in the data"}
        if budget:
            out["gain_or_loss_on_budget"] = round(float(budget * (V[-1] - 1)), 2)
        return out

    res, notes = {}, []
    p_bt = d["px_out"].iloc[0].values
    budget = None
    if answers and answers.get("invest_amount"):   # client's own portfolios (else benchmarks only)
        _, _, _, _, means, profile, info, inv = _setup(answers)
        budget = inv["budget"]
        for s in ("S1", "S2"):
            w, _ = _part4_weights(s, inv, mu, Q, T, meta, p_bt)
            res[f"{pa.STRATEGY_NAMES[s]} (as recommended)"] = metrics(w, budget)
            w_ex, fb = _part4_weights(s, inv, mu_ex, Q_ex, T, meta, p_bt)
            res[f"{pa.STRATEGY_NAMES[s]} (re-estimated without 2022)"] = metrics(w_ex, budget)
            if fb:
                notes.append(f"{s} without 2022: volatility cap infeasible, minimum-variance portfolio used")
        res["Benchmark: Equal weight"] = metrics(opt.equal_weight(inv, T), budget)
    else:
        res["Benchmark: Equal weight (all 24 assets)"] = metrics(np.full(len(T), 1 / len(T)), None)
    ref = np.array([opt.W_REF.get(t, 0.0) for t in T])
    res["60/40 reference (XIU.TO / VTI / ZAG.TO / AGG)"] = metrics(ref, budget)
    weeks = len(win)
    g_end = np.exp(win.cumsum().iloc[-1])
    return {"window": "3 Jan 2022 to 17 Oct 2022 (weekly returns)", "weeks": weeks, "portfolios": res,
            "cash_at_risk_free_rate": round(float(opt.RF * weeks / 52), 4),
            "bond_etf_returns": {t: round(float(g_end[t] - 1), 4) for t in T if meta.loc[t, "AssetClass"] == "Fixed Income"},
            "notes": notes,
            "data_note": "Uses in-sample (estimation-period) data, so 'as recommended' results benefit from hindsight; "
                         "the 're-estimated without 2022' rows remove the stress weeks from the estimates. COVID-19 "
                         "(2020) is not in the data, which starts in April 2021."}


# ---------------------------------------------------------------------------------------------
def asset_stats(tickers: list[str] | None = None) -> dict:
    """Part 2 in-sample statistics (annualised, CAD, risk-free 2.3%)."""
    s = pd.read_csv(os.path.join(market_data.DATA_DIR, "summary_statistics.csv"), index_col=0)
    if tickers:
        s = s.loc[[t for t in tickers if t in s.index]]
    return {"period": "in-sample weekly data, 2021-04-12 to 2026-03-30", "risk_free_rate": opt.RF,
            "assets": {t: {"name": r["Name"], "annual_return": round(float(r["Ann. Return"]), 4),
                           "annual_volatility": round(float(r["Ann. Volatility"]), 4),
                           "sharpe": round(float(r["Sharpe Ratio"]), 2),
                           "max_drawdown": round(float(r["Max Drawdown"]), 4)} for t, r in s.iterrows()}}


def universe() -> list[dict]:
    meta = market_data.load()["meta"]
    return [{"ticker": t, "name": r["Name"], "sector": r["Sector"], "type": r["Type"]} for t, r in meta.iterrows()]


# ---------------------------------------------------------------------------------------------
def small_budget_diagnostics(answers: dict) -> dict:
    """Why whole units fail for this budget: assets priced above the budget, and the smallest budget
    (on a $250 grid) that works for the same answers."""
    d = market_data.load(); T, meta = d["tickers"], d["meta"]
    prices = d["px_out"].iloc[-1]
    budget = float(answers.get("invest_amount") or 0)
    too_expensive = [{"ticker": t, "name": meta.loc[t, "Name"], "price_cad": round(float(prices[t]), 2)}
                     for t in prices.sort_values(ascending=False).index if prices[t] > budget]
    min_ok = None
    for b in range(int(max(250, budget // 250 * 250 + 250)), 10001, 250):
        try:
            pa.get_portfolio(dict(answers, invest_amount=b))
            min_ok = b; break
        except ValueError:
            continue
    return {"budget": budget, "assets_priced_above_budget": too_expensive,
            "smallest_working_budget_on_250_grid": min_ok,
            "alternatives": ["use a broker that allows fractional units", "invest a larger amount"]}


# ---------------------------------------------------------------------------------------------
FX_SNAPSHOT = os.path.join(DATA_DIR, "fx_snapshot.json")


def fx_rate(currency: str, live: bool | None = None) -> dict:
    """CAD per 1 unit of `currency`. Offline by default (saved snapshot, reproducible for the mystery
    client); set LIVE_FX=1 to fetch today's rate from Yahoo Finance."""
    currency = currency.upper()
    if currency == "CAD":
        return {"currency": "CAD", "cad_per_unit": 1.0, "source": "identity", "date": None}
    live = os.environ.get("LIVE_FX") == "1" if live is None else live
    if live:
        try:
            import yfinance as yf
            px = yf.download(f"{currency}CAD=X", period="5d", progress=False, auto_adjust=True)["Close"].dropna()
            rate = float(np.asarray(px.iloc[-1]).ravel()[0])
            return {"currency": currency, "cad_per_unit": round(rate, 4), "source": "Yahoo Finance (live)",
                    "date": str(px.index[-1].date())}
        except Exception as e:  # fall back to the snapshot
            print(f"[portfolio_tools] live FX failed ({type(e).__name__}); using snapshot")
    snap = json.load(open(FX_SNAPSHOT))
    if currency not in snap["rates"]:
        raise ValueError(f"no exchange rate available for {currency}")
    return {"currency": currency, "cad_per_unit": snap["rates"][currency], "source": snap["source"],
            "date": snap["date"]}


def save_fx_snapshot(currencies=("USD", "GBP", "EUR", "JPY", "AUD", "CHF", "CNY", "INR", "KRW", "MXN", "HKD")):
    """Run once (network needed) to refresh data/fx_snapshot.json."""
    rates, day = {}, None
    for c in currencies:
        r = fx_rate(c, live=True)
        if r["source"].endswith("(live)"):
            rates[c] = r["cad_per_unit"]; day = r["date"]
    json.dump({"source": "Yahoo Finance snapshot", "date": day or str(date.today()), "rates": rates},
              open(FX_SNAPSHOT, "w"), indent=2)
    return rates


if __name__ == "__main__":
    print(save_fx_snapshot())


# ---------------------------------------------------------------------------------------------
_PCT_KEYS = ("return", "drawdown", "weight", "volatility", "worst_week", "turnover", "cash_at_risk_free_rate", "volatility_cap", "risk_free_rate")
_MONEY_KEYS = ("invested", "fees", "value_cad", "cash_residual", "price_cad", "budget")


def for_llm(obj, decimals: int = 1, key: str = ""):
    """Format tool output for the LLM: fractions -> 'x.x%', amounts -> '$x,xxx.xx', so the model can quote
    them verbatim instead of converting numbers itself."""
    if isinstance(obj, dict):   # a child key that says nothing (e.g. a ticker) inherits the parent's meaning
        known = _PCT_KEYS + _MONEY_KEYS + ("cost_percentage_points",)
        return {k: for_llm(v, decimals, k if any(p in str(k).lower() for p in known) else key) for k, v in obj.items()}
    if isinstance(obj, list):
        return [for_llm(v, decimals, key) for v in obj]
    if isinstance(obj, float) and not isinstance(obj, bool):
        k = key.lower()
        if k == "cost_percentage_points":
            return f"{obj:.2f} percentage points"
        if any(p in k for p in _PCT_KEYS) and abs(obj) <= 5:
            return f"{obj * 100:.{decimals}f}%"
        if any(p in k for p in _MONEY_KEYS):
            return f"${obj:,.2f}"
    return obj
