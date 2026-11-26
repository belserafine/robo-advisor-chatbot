# ------------------------------------------
# PORTFOLIO AGENT
# ------------------------------------------
# Public API (used by app.py):
#   get_portfolio(answers: dict) -> dict
#       Deterministic: maps questionnaire answers -> Part 3 profile, runs the Part 3 optimizer
#       (Strategy 1, Strategy 2, equal-weight benchmark), converts weights to whole units,
#       backtests on the Part 2 out-of-sample period. EVERY number the chatbot reports comes from this dict.
#   draft_response(answers, portfolio, feedback=None) -> str
#       LLM writes the plain-language explanation using ONLY the numbers in portfolio["facts"].
#       `feedback` = issues returned by the checking agent (self-correction loop).
#   answer_followup(question, answers, portfolio, history) -> str
#       Follow-up Q&A after the recommendation, grounded in the same facts.
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import llm
import market_data
import optimizer as opt

# ------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------
MODEL = llm.MODEL  # Claude Haiku 5.5 (configured in llm.py)
# Strategy shown to the client as the recommendation. Part 3 favours Strategy 2 (diversified,
# ETF-based, realistic expected returns); revisit once Part 4 backtests are done.
RECOMMENDED = "S2"
STRATEGY_NAMES = {
    "S2": "Strategy 2: Reverse optimization + Markowitz",
    "S1": "Strategy 1: Markowitz with sample means",
    "EW": "Benchmark: Equal weight",
}

# ------------------------------------------
# 1. QUESTIONNAIRE ANSWER -> PART 3 PROFILE
# ------------------------------------------
# Part 3 mapping (Section 3):
# | Answer                          | Model input                                   |
# |---------------------------------|-----------------------------------------------|
# | invest_amount                   | budget B                                      |
# | desc (risk statement)           | volatility cap: low 6.5% / moderate 10% / high 14% |
# | target_years / horizon_years    | fixed-income floor: <=2y 50% / <=6y 20% / else 0% |
# | goal & capacity for loss        | max weight per asset: 40% (low risk) / 30%    |
# | capital-preservation goal       | commodity cap: 10% (low risk) / = asset cap   |
# | sectors (ESG preferences)       | excluded assets (w_i = 0)                     |
# Generalizations for arbitrary clients (Part 5; reproduce Personas A/B/C exactly):
#   * "Very low risk" -> "low" (Persona A precedent; 6.5% is just above the ~5.9% lowest achievable vol)
#   * horizon buckets -> years: 0-1 -> 1, 1-3 -> 2, 4-10 -> 5, 10+ -> 10; the shorter of the two answers is used
#   * suitability caps: horizon 0-1y or an emergency-fund goal -> at most "low";
#                       horizon 1-3y -> at most "moderate"
_RISK = {"Very low risk": "low", "Low risk": "low", "Moderate risk": "moderate", "High risk": "high"}
_RISK_ORDER = ["low", "moderate", "high"]
_HORIZON_YEARS = {"0-1 years": 1, "1-3 years": 2, "4-10 years": 5, "10+ years": 10}

# free-text exclusion keywords
_FLAG_KEYWORDS = {
    "Fossil": [r"fossil", r"\boil\b", r"\bgas\b", r"coal", r"pipeline", r"petroleum"],
    "AI": [r"\bai\b", r"artificial intelligence"],
}
_SECTOR_KEYWORDS = {
    "Technology": [r"\btech", r"software"],
    "Financials": [r"\bbank", r"financ"],
    "Energy": [r"\benergy (?:stocks|companies|sector)"],
    "Real Estate": [r"real estate", r"\breits?\b"],
    "Commodities": [r"\bgold\b", r"commodit"],
    "Healthcare": [r"health", r"pharma"],
    "Utilities": [r"utilit"],
    "Industrials": [r"industrial", r"rail"],
    "Consumer Staples": [r"consumer staples"],
}
_COMPANY_ALIASES = {
    "royal bank": "RY.TO", "rbc": "RY.TO", "td bank": "TD.TO", "toronto-dominion": "TD.TO",
    "cn rail": "CNR.TO", "canadian national": "CNR.TO", "enbridge": "ENB.TO", "apple": "AAPL",
    "microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL", "jpmorgan": "JPM", "jp morgan": "JPM",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ", "procter": "PG", "berkshire": "BRK-B",
    "exxon": "XOM", "nextera": "NEE",
}


def _norm(s):
    return s.replace("–", "-").replace("—", "-") if isinstance(s, str) else s


def parse_exclusions(text: str | None, meta: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (excluded tickers, human-readable reasons) from the free-text `sectors` answer."""
    if not text or text.strip().lower() in {"no", "none", "n/a", "nothing", "no exclusions"}:
        return [], []
    t = text.lower()
    tickers, reasons = set(), []
    for flag, pats in _FLAG_KEYWORDS.items():
        if any(re.search(p, t) for p in pats):
            tickers |= set(meta.index[meta[flag] == 1]); reasons.append("fossil fuels" if flag == "Fossil" else "AI companies")
    for sector, pats in _SECTOR_KEYWORDS.items():
        if any(re.search(p, t) for p in pats):
            tickers |= set(meta.index[meta["Sector"] == sector]); reasons.append(sector)
    for alias, ticker in _COMPANY_ALIASES.items():
        if alias in t:
            tickers.add(ticker); reasons.append(meta.loc[ticker, "Name"])
    for ticker in meta.index:  # ticker typed explicitly, e.g. "no XOM"
        if re.search(rf"(?<![\w.-]){re.escape(ticker.lower())}(?![\w-])", t):
            tickers.add(ticker)
    return sorted(tickers), sorted(set(reasons))


def answers_to_profile(answers: dict, meta: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (Part 3 profile dict, explanation of how it was derived)."""
    a = {k: _norm(v) for k, v in answers.items()}
    budget = a.get("invest_amount")
    if not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("the amount to invest must be a positive number")
    reserve = a.get("cash_needed_now") or 0
    if not isinstance(reserve, (int, float)) or reserve < 0:
        reserve = 0
    if reserve >= budget:
        raise ValueError("you need all of this money as cash right now, so there is nothing left to invest")

    years = [_HORIZON_YEARS[h] for h in (a.get("target_years"), a.get("horizon_years")) if h in _HORIZON_YEARS]
    horizon = min(years) if years else 5

    stated = _RISK.get(a.get("desc"), "moderate")
    risk, overrides = stated, []
    cap = "high"
    if horizon <= 1 or a.get("objective_str") == "Build an emergency fund":
        cap = "low"
    elif horizon <= 2:
        cap = "moderate"
    if _RISK_ORDER.index(risk) > _RISK_ORDER.index(cap):
        overrides.append(f"risk level lowered from '{risk}' to '{cap}' because of the short horizon / goal")
        risk = cap

    exclude, reasons = parse_exclusions(a.get("sectors"), meta)
    low = risk == "low"
    profile = opt.make_profile(
        name="client", budget=float(budget), risk=risk, horizon=horizon,
        max_weight=0.40 if low else 0.30, exclude=exclude, commodity_max=0.10 if low else None,
    )
    horizon_range = next((h for h in (a.get("target_years"), a.get("horizon_years"))
                          if _HORIZON_YEARS.get(h) == horizon), None)
    target = a.get("target") if a.get("target_yn") == "Yes" and isinstance(a.get("target"), (int, float)) else None
    info = {"stated_risk": a.get("desc"), "risk_level": risk, "horizon_years": horizon,
            "horizon_range": horizon_range, "goal": a.get("objective_str"), "target": target,
            "overrides": overrides, "exclusion_reasons": reasons, "cash_reserve": float(reserve),
            "principal_protection": a.get("principal_protection")}
    return profile, info


def _investing_profile(profile: dict, reserve: float) -> dict:
    """Constraints for the money actually invested when `reserve` is kept as cash.
    Final weights are value / TOTAL budget (compliance checker's definition), so with
    s = invested share: s*w <= cap -> w <= cap/s, s*FI >= floor -> FI >= floor/s, s*vol <= cap -> vol <= cap/s."""
    if reserve <= 0:
        return profile
    s = (profile["budget"] - reserve) / profile["budget"]
    q = dict(profile, budget=profile["budget"] - reserve,
             vol_cap=profile["vol_cap"] / s, max_weight=min(1.0, profile["max_weight"] / s),
             commodity_max=min(1.0, profile["commodity_max"] / s), sector_cap=min(1.0, profile["sector_cap"] / s),
             fi_min=profile["fi_min"] / s)
    if q["fi_min"] > 1:
        raise ValueError("the cash you need now is too large to keep the minimum bond share for your time horizon")
    return q


# ------------------------------------------
# 2. get_portfolio  (deterministic tool - no LLM)
# ------------------------------------------
def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def get_portfolio(answers: dict, strategy: str | None = None) -> dict:
    """strategy: "S2" (default, recommended) or "S1" - the strategy whose whole-unit orders are produced."""
    strategy = strategy or RECOMMENDED
    if strategy not in ("S1", "S2"):
        raise ValueError(f"unknown strategy {strategy}")
    d = market_data.load()
    T, meta, mu, Q, px = d["tickers"], d["meta"], d["mu"].values, d["Q"].values, d["px_out"]
    profile, info = answers_to_profile(answers, meta)
    inv = _investing_profile(profile, info["cash_reserve"])   # == profile when no cash is needed now
    pi, delta = opt.implied_returns(Q, T)
    means = {"S1": mu, "S2": pi}

    date_now, date_bt = px.index[-1], px.index[0]   # 2026-11-18 = "today", 2026-04-01 = backtest start
    p_now, p_bt = px.loc[date_now].values, px.loc[date_bt].values

    weights, dropped, weights_bt = {}, {}, {}
    try:
        for s in ("S1", "S2"):
            solve = (lambda m: (lambda q: opt.markowitz_max_return(m, q, Q, T, meta)))(means[s])
            weights[s], _, dropped[s] = opt.investable(solve, inv, p_now, T)
            weights_bt[s], _, _ = opt.investable(solve, inv, p_bt, T)  # uses only 1 Apr 2026 prices
    except (ValueError, RuntimeError) as e:
        print(f"[portfolio_agent] optimization failed: {e}")
        raise ValueError("no portfolio satisfies your constraints. The budget may be too small to "
                         "buy whole units of a diversified portfolio, or too many assets are excluded") from e
    weights["EW"] = weights_bt["EW"] = opt.equal_weight(inv, T)
    dropped["EW"] = []

    # selected portfolio -> whole units at today's prices
    w = weights[strategy]
    # whole units that still satisfy the persona constraints on the FINAL holdings (compliance checks these)
    rows, _ = opt.linear_constraints(inv, T, meta)
    try:
        units, cash = opt.to_units(w, p_now, inv["budget"], max_weight=inv["max_weight"], rows=rows,
                                   Q=Q, vol_cap=inv["vol_cap"],
                                   protect=(meta["AssetClass"] == "Fixed Income").values)
        cash["cash"] += info["cash_reserve"]       # requested cash is kept aside, not invested
    except ValueError as e:
        print(f"[portfolio_agent] unit conversion failed: {e}")
        raise ValueError("the budget is too small to buy whole units that respect your risk limits") from e
    values = units * p_now
    holdings = []
    for i in np.argsort(-w):
        if w[i] <= 1e-4 and units[i] == 0:
            continue
        t = T[i]
        holdings.append({
            "ticker": t, "name": meta.loc[t, "Name"], "type": meta.loc[t, "Type"],
            "asset_class": meta.loc[t, "AssetClass"], "sector": meta.loc[t, "Sector"],
            "price_cad": round(float(p_now[i]), 2), "units": int(units[i]),
            "value_cad": round(float(values[i]), 2),
            "target_weight": round(float(w[i]), 4),
            "weight": round(float(values[i] / profile["budget"]), 4),
        })

    # Part 4 backtest, exactly as in the Part 4 notebook: whole units at 1 Apr 2026 (Part 3 MILP), monthly
    # rebalancing on the first trading day, 5 bp per trade; benchmark in exact units. (Today's orders above use
    # the constrained conversion that the compliance checker verifies.)
    whole_units = lambda w_, p_, b_: opt.to_units(w_, p_, b_)
    comparison, backtest_series = {}, {}
    for s in ("S2", "S1", "EW"):
        ws = weights[s]
        own = means.get(s, mu)
        vol = float(np.sqrt(ws @ Q @ ws))
        unit_fn = opt.to_units_fractional if s == "EW" else whole_units
        bt = opt.backtest(weights_bt[s], px, inv["budget"], unit_fn, vol_cap=None if s == "EW" else inv["vol_cap"])
        # Figures are reported for the client's WHOLE budget. With cash kept aside (share 1-k, no return, no
        # risk) the portfolio's return and volatility are k x those of the invested part, and the backtest value
        # includes the cash. k = 1 (no cash reserve) leaves every number exactly as in Part 3/4.
        k_inv, ret_inv = inv["budget"] / profile["budget"], float(own @ ws)
        metrics = bt["metrics"]
        if info["cash_reserve"]:
            tr = bt["trades"].copy()
            tr["Turnover"] = tr["Turnover"] * tr["Value before"] / (tr["Value before"] + info["cash_reserve"])
            metrics = opt.perf_metrics(bt["series"] + info["cash_reserve"], profile["budget"], tr,
                                       None if s == "EW" else profile["vol_cap"])
        comparison[s] = {
            "name": STRATEGY_NAMES[s],
            "in_sample": {"expected_return": _r(k_inv * ret_inv), "volatility": _r(k_inv * vol),
                          "sharpe": _r((k_inv * ret_inv - opt.RF) / (k_inv * vol)),
                          "expected_return_on_invested": _r(ret_inv)},
            "out_of_sample": {k: (_r(v) if isinstance(v, float) else v) for k, v in metrics.items()},
            "constraint_issues": opt.check_portfolio(ws, inv, Q, T, meta, is_benchmark=(s == "EW")),
            "dropped_not_buyable": dropped[s],
            "num_assets": int((ws > 1e-4).sum()),
        }
        backtest_series[STRATEGY_NAMES[s]] = {k.strftime("%Y-%m-%d"): round(float(v) + info["cash_reserve"], 2)
                                              for k, v in bt["series"].items()}

    actual_w = pd.Series(values / profile["budget"], index=T)
    asset_class_mix = {k: round(float(v), 4) for k, v in actual_w.groupby(meta["AssetClass"]).sum().items() if v > 1e-4}
    asset_class_mix["Cash & fees"] = round(1 - sum(asset_class_mix.values()), 4)

    result = {
        "profile": {k: (list(v) if isinstance(v, list) else v) for k, v in profile.items()},
        "profile_info": info,
        "strategy": strategy,
        "strategy_name": STRATEGY_NAMES[strategy],
        "recommended_strategy": RECOMMENDED,
        "price_date": date_now.strftime("%Y-%m-%d"),
        "estimation_window": [d["ret_ins"].index[0].strftime("%Y-%m-%d"), d["ret_ins"].index[-1].strftime("%Y-%m-%d")],
        "test_window": [date_bt.strftime("%Y-%m-%d"), date_now.strftime("%Y-%m-%d")],
        "risk_free_rate": opt.RF,
        "transaction_cost": opt.TC,
        "budget": profile["budget"],
        "invested": round(cash["invested"], 2),
        "fees": round(cash["fees"], 2),
        "cash_residual": round(cash["cash"], 2),
        "holdings": holdings,
        "asset_class_mix": asset_class_mix,
        "comparison": comparison,
        "backtest_series": backtest_series,
    }
    result["facts"] = build_facts(result)
    return result


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _pct_vs(x, cap):
    """Percent with enough decimals that a value above/below `cap` never prints equal to it (10.04% vs 10.0%)."""
    one = _pct(x)
    return f"{100 * x:.2f}%" if x is not None and one == _pct(cap) and abs(x - cap) > 1e-9 else one


def _money(x):
    return f"${x:,.2f}"


STRATEGY_BASIS = {
    "S2": "model estimate from reverse optimization (returns implied by the covariance matrix and a 60/40 "
          "reference portfolio), not a historical average and not a promise",
    "S1": "model estimate from historical average returns over the estimation window, not a promise",
    "EW": "model estimate from historical average returns, not a promise",
}


def _years(n) -> str:
    return f"{n} year" if n == 1 else f"{n} years"


DEFAULT_STRATEGY_REASON = (
    "Strategy 2 is this tool's default for every client, chosen in advance by design (not by looking at the "
    "backtest): its expected returns are implied by market risk (reverse optimization) instead of historical "
    "averages, which makes it less sensitive to noise in past returns")


def _strategy_reasons(p: dict, f: dict) -> list[str]:
    """Why the shown strategy is shown, written by code so the reviewer can trace every claim to the tool results."""
    cap = f["volatility_cap"]
    reasons = ([f"you asked for {p['strategy_name']}; the tool's default is {STRATEGY_NAMES[p['recommended_strategy']]}"]
               if f["selected_by_client"] else [DEFAULT_STRATEGY_REASON])
    for s, c in p["comparison"].items():
        name, st, n = c["name"], f["strategies"][c["name"]], c.get("num_assets")
        facts = [f"holds {n} assets"] if n else []
        if s != "EW" and c["out_of_sample"].get("within_vol_cap") is False:
            facts.append(f"its realized volatility in the test period was "
                         f"{_pct_vs(c['out_of_sample']['annualized_volatility'], p['profile']['vol_cap'])}, above your "
                         f"{cap} cap")
        if s == "EW" and c["in_sample"]["volatility"] > p["profile"]["vol_cap"] + 1e-9:
            facts.append(f"its estimated volatility is {st['expected_annual_volatility']}, above your {cap} cap, "
                         "so it is shown only as a benchmark")
        elif s == "EW":
            facts.append("it is shown only as a benchmark")
        if s != p["strategy"] and c["out_of_sample"]["total_return"] > p["comparison"][p["strategy"]]["out_of_sample"]["total_return"]:
            facts.append(f"it did better in this test period ({st['backtest_total_return']} total), but one short "
                         "period does not say which will do better in the future")
        if facts:
            reasons.append(f"{name}: " + "; ".join(facts))
    return reasons


def _exclusion_note(p: dict, f: dict) -> str:
    """Fixed wording about the exclusion screen (appended by code, not written by the LLM)."""
    etfs = [h["ticker"] for h in p["holdings"] if h["units"] > 0 and
            ((h.get("type") == "ETF" and h.get("asset_class") != "Commodity") or h.get("sector") == "Diversified")]
    note = (f"**About your exclusions ({', '.join(f['excluded_because_of_preferences'])}):** I removed the individual "
            f"tickers flagged in our data ({', '.join(f['excluded_tickers'])}). This screen works at the ticker level "
            "only - the funds' underlying holdings were not checked")
    if etfs:
        note += (f", so the funds and diversified holdings in your portfolio ({', '.join(etfs)}) very likely still hold some of "
                 "these companies - broad market funds own most large companies in their market. "
                 "If that is not acceptable, tell me which ETFs to exclude as well and I will rebuild the portfolio")
    return note + "."


def build_facts(p: dict) -> dict:
    """Pre-formatted numbers the LLM is allowed to quote (verbatim). The checking agent
    compares every number in a draft against this dict."""
    prof, info = p["profile"], p["profile_info"]
    f = {
        "risk_level": info["risk_level"],
        "horizon_years": info["horizon_years"],
        "volatility_cap": _pct(prof["vol_cap"]),
        "min_fixed_income": _pct(prof["fi_min"]),
        "max_weight_per_asset": _pct(prof["max_weight"]),
        "budget": _money(p["budget"]),
        "invested": _money(p["invested"]),
        "fees": _money(p["fees"]),
        "cash_residual": _money(p["cash_residual"]),
        "price_date": p["price_date"],
        "test_window": f'{p["test_window"][0]} to {p["test_window"][1]}',
        "test_period_months": f'{(pd.Timestamp(p["test_window"][1]) - pd.Timestamp(p["test_window"][0])).days / 30.44:.1f}',
        "risk_free_rate": _pct(p["risk_free_rate"]),
        "excluded_because_of_preferences": info["exclusion_reasons"],
        "excluded_tickers": prof["exclude"],
        "profile_adjustments": info["overrides"],
        "asset_class_mix": {k: _pct(v) for k, v in p["asset_class_mix"].items()},
        "holdings": [{"ticker": h["ticker"], "name": h["name"], "units": h["units"],
                      "price": _money(h["price_cad"]), "value": _money(h["value_cad"]),
                      "weight": _pct(h["weight"])} for h in p["holdings"] if h["units"] > 0],
        "strategies": {},
    }
    for s, c in p["comparison"].items():
        f["strategies"][c["name"]] = {
            "expected_annual_return": _pct(c["in_sample"]["expected_return"]),
            "expected_annual_volatility": _pct(c["in_sample"]["volatility"]),
            "backtest_annualized_return": _pct(c["out_of_sample"]["annualized_return"]),
            "backtest_total_return": _pct(c["out_of_sample"]["total_return"]),
            "backtest_volatility": _pct(c["out_of_sample"]["annualized_volatility"]),
            "backtest_max_drawdown": _pct(c["out_of_sample"]["max_drawdown"]),
            "backtest_sharpe": "n/a" if c["out_of_sample"]["sharpe"] is None else f'{c["out_of_sample"]["sharpe"]:.2f}',
            "backtest_avg_monthly_turnover": _pct(c["out_of_sample"]["avg_monthly_turnover"]),
            "backtest_trading_costs": _money(c["out_of_sample"]["trading_costs"]),
        }
    f["recommended_strategy"] = p["strategy_name"]
    f["selected_by_client"] = p["strategy"] != p.get("recommended_strategy", p["strategy"])
    if f["selected_by_client"]:
        f["default_recommendation"] = STRATEGY_NAMES[p["recommended_strategy"]]
    f["recommended"] = f["strategies"][p["strategy_name"]]
    # Disclosures the compliance reviewer expects; all computed or stated by code.
    f["disclosures"] = {
        "expected_return_basis": STRATEGY_BASIS[p["strategy"]],
        "volatility_cap_meaning": "the optimizer keeps the portfolio's estimated volatility at or below the cap; "
                                  "this limits risk but does not guarantee the principal - the value can fall",
        "horizon_assumption": (f"the {info['horizon_range']} range was modelled as {_years(info['horizon_years'])} - "
                               "the fixed value the model uses for this answer range (set in the questionnaire "
                               "design), not the midpoint of the range or a figure chosen for this client"
                               if info.get("horizon_range") else f"{_years(info['horizon_years'])} assumed"),
        "backtest_return_meaning": "total return over the whole test period, not an annual rate",
    }
    rec_c = p["comparison"][p["strategy"]]
    if (rec_c["out_of_sample"]["annualized_volatility"] or 0) > prof["vol_cap"]:
        f["disclosures"]["realized_volatility_note"] = (
            f"the {f['volatility_cap']} cap applies to volatility estimated from historical data; in the held-out "
            f"test period this portfolio's realized volatility was "
            f"{_pct_vs(rec_c['out_of_sample']['annualized_volatility'], prof['vol_cap'])}, "
            "above the cap - actual volatility can be higher than the estimate")
    if info.get("cash_reserve"):
        f["cash_reserve"] = _money(info["cash_reserve"])
        f["disclosures"]["cash_reserve"] = (f"{_money(info['cash_reserve'])} you said you need now is kept as cash "
                                            "and not invested; it is included in the uninvested cash")
    if info.get("stated_risk") == "Very low risk":
        eq = f["asset_class_mix"].get("Equity")
        f["disclosures"]["risk_mapping"] = ("you chose 'Very low risk'; the model's most conservative setting is "
                                            f"'low' (volatility cap {f['volatility_cap']}), which is what we used - it is "
                                            "the closest available setting, not an exact match"
                                            + (f", and the portfolio still holds {eq} in stocks (its current share, "
                                               "not a limit)" if eq else ""))
    if info["exclusion_reasons"]:
        f["disclosures"]["exclusion_scope"] = ("screen applies to the individual tickers flagged in our data; broad "
                                               "ETFs are not looked through and very likely still hold some excluded companies, because broad market "
                                               "funds own most large companies in their market")
    f["why_this_strategy"] = _strategy_reasons(p, f)
    if info["exclusion_reasons"]:
        f["exclusion_note"] = _exclusion_note(p, f)
    if info.get("target"):
        years, r = info["horizon_years"], p["comparison"][p["strategy"]]["in_sample"]["expected_return_on_invested"]
        projected = p["invested"] * (1 + r) ** years + p["cash_residual"]
        f["goal_projection"] = {
            "goal": info.get("goal"), "target": _money(info["target"]), "years": years,
            "illustrative_value_at_expected_return": _money(projected),
            "reaches_target_at_expected_return": bool(projected >= info["target"]),
            "basis": f"the {_money(p['invested'])} actually invested grows at the model's expected annual return "
                     f"of {_pct(r)} for {_years(years)}; the "
                     f"{_money(p['cash_residual'])} of uninvested cash is assumed to stay flat and fees are already paid",
            "note": "illustration using the model's expected return; actual results will differ and are not guaranteed",
        }
        # caveat (no extrapolation of the short test period): compare the estimate with what the strategy earned
        if p["comparison"][p["strategy"]]["out_of_sample"]["annualized_return"] is not None:
            f["goal_projection"]["test_period_comparison"] = (
                f"in the {f['test_period_months']}-month held-out test this strategy's annualized return was "
                f"{f['recommended']['backtest_annualized_return']} (model estimate {f['recommended']['expected_annual_return']}); "
                "returns vary, so reaching the target is not assured and the value can end below it")
    return f


# ------------------------------------------
# 3. LLM: plain-language explanation grounded in FACTS
# ------------------------------------------
DRAFT_SYSTEM_PROMPT = (
    "You are the Portfolio Agent of an educational robo-advisor. Write a short, friendly, "
    "plain-language explanation (max ~250 words, markdown) of the recommended portfolio for this client.\n"
    "STRICT RULES:\n"
    "1. Every number you write MUST be copied verbatim from FACTS. Never calculate, round, sum, "
    "or estimate a number yourself. If a number is not in FACTS, describe it in words instead.\n"
    "2. Explain how the client's answers set the risk limits (volatility cap, minimum fixed income), why this "
    "strategy is shown and how the alternatives compare, and how each did in the held-out backtest "
    "(be honest if the recommendation did worse). For WHY, use only FACTS.why_this_strategy - do not invent "
    "other reasons or say that one fact 'caused' the choice.\n"
    "3. Mention any profile adjustments, and that past performance does not guarantee future results. Do NOT "
    "write about exclusions at all: the system appends FACTS.exclusion_note word for word.\n"
    "3b. Use FACTS.disclosures: call the expected return a model estimate (say how it is produced), say the "
    "volatility cap limits risk but does not guarantee the principal, state the horizon assumption exactly as FACTS.disclosures.horizon_assumption words it (do not explain how the value was chosen). If FACTS.goal_projection exists, state the "
    "illustrative projected value versus the target, its basis (FACTS.goal_projection.basis), "
    "FACTS.goal_projection.test_period_comparison, and whether it "
    "reaches the target, as an illustration only. "
    "Backtest returns are totals for the test period (say so, with its length), not annual rates. "
    "If FACTS.disclosures.realized_volatility_note exists, state it. If FACTS.selected_by_client is true, say the client asked for this strategy and name the default "
    "recommendation instead of claiming it is recommended. If the strategy holds only a few assets, note the "
    "concentration risk. If FACTS.disclosures.risk_mapping exists, explain it. Asset-class shares must be quoted from "
    "FACTS.asset_class_mix exactly.\n"
    "4. Do not give individual stock tips or tax/legal advice. Do not add a disclaimer or the price date "
    "(the system appends them).\n"
    "5. Do not repeat the full holdings table (the app shows it); mention at most the top 3 holdings."
)


def _generate(prompt: str) -> str:
    return llm.generate_text(prompt, system=DRAFT_SYSTEM_PROMPT, effort=llm.EFFORT_WRITE)


def _client_summary(answers: dict) -> dict:
    keys = ["age", "objective_str", "target_years", "horizon_years", "desc", "exp", "sectors"]
    return {k: answers.get(k) for k in keys}


def with_footer(text: str, portfolio: dict) -> str:
    """Data date + the compliance checker's standard disclaimer, appended by code (never by the LLM)."""
    from compliance_agent import DISCLAIMER
    note = portfolio["facts"].get("exclusion_note")
    body = text.rstrip() + (f"\n\n{note}" if note and note not in text else "")
    return f"{body}\n\n_Prices as of {portfolio['price_date']}. {DISCLAIMER}_"


def draft_response(answers: dict, portfolio: dict, feedback: list[str] | None = None) -> str:
    prompt = (f"CLIENT PROFILE:\n{json.dumps(_client_summary(answers), indent=2)}\n\n"
              f"FACTS:\n{json.dumps(portfolio['facts'], indent=2)}\n")
    if feedback:
        prompt += ("\nYOUR PREVIOUS DRAFT WAS REJECTED BY THE COMPLIANCE CHECK. Fix these issues:\n- "
                   + "\n- ".join(feedback) + "\n")
    try:
        return with_footer(_generate(prompt), portfolio)
    except Exception as e:  # quota errors etc. -> deterministic template, still grounded in facts
        print(f"[portfolio_agent] LLM draft failed ({type(e).__name__}); using template.")
        return template_response(portfolio)


def template_response(portfolio: dict) -> str:
    f = portfolio["facts"]
    rec = f["recommended"]
    top = ", ".join(f'{h["name"]} ({h["weight"]})' for h in f["holdings"][:3])
    mix = ", ".join(f"{k} {v}" for k, v in f["asset_class_mix"].items())
    lines = [
        (f"Based on your answers, your risk level is **{f['risk_level']}**"
         + (f" ({f['disclosures']['risk_mapping']})" if "risk_mapping" in f["disclosures"] else "")
         + f": the portfolio's volatility is capped at {f['volatility_cap']} and at least {f['min_fixed_income']} "
           "goes to bonds."),
        (f"As you asked, here is **{f['recommended_strategy']}** (our default is {f['default_recommendation']})"
         if f["selected_by_client"] else f"I recommend **{f['recommended_strategy']}**")
        + f": {f['invested']} invested out of {f['budget']} "
        f"(fees {f['fees']}, cash left over {f['cash_residual']}).",
        f"Asset mix: {mix}. Largest positions: {top}.",
        *([f"As you asked, {f['cash_reserve']} is kept as cash and not invested."] if "cash_reserve" in f else []),
        f"Expected annual return {rec['expected_annual_return']} with estimated volatility "
        f"{rec['expected_annual_volatility']}. The expected return is a {f['disclosures']['expected_return_basis']}. "
        f"Keeping volatility at or below {f['volatility_cap']} limits risk but does not guarantee your principal: "
        "the value can fall.",
        f"Time horizon: {f['disclosures']['horizon_assumption']}.",
        *([f"Note: {f['disclosures']['realized_volatility_note']}."] if "realized_volatility_note" in f["disclosures"] else []),
        f"Over the held-out test period {f['test_window']} (about {f['test_period_months']} months, data the model "
        f"never saw), its total return was {rec['backtest_total_return']} - a total for the period, not an annual "
        f"rate - with a maximum drawdown of {rec['backtest_max_drawdown']}. Past performance does not "
        "guarantee future results.",
    ]
    lines.append("Why this strategy: " + " ".join(r[0].upper() + r[1:] + "." for r in f["why_this_strategy"]))
    if "goal_projection" in f:
        g = f["goal_projection"]
        verdict = "would reach" if g["reaches_target_at_expected_return"] else "would fall short of"
        lines.append(f"Your goal: {g['target']} in about {_years(g['years'])}. Illustration: if {g['basis']}, you "
                     f"would have about {g['illustrative_value_at_expected_return']}, which {verdict} your target. "
                     + (f"For comparison, {g['test_period_comparison']}. " if "test_period_comparison" in g else "")
                     + "These are illustrations only; actual results will differ and are not guaranteed.")
    if f["profile_adjustments"]:
        lines.append("Note: " + "; ".join(f["profile_adjustments"]) + ".")
    return with_footer("\n\n".join(lines), portfolio)


ANSWER_SYSTEM_PROMPT = (
    "You are an educational robo-advisor chatbot answering a client's question. Be friendly, plain-language and "
    "brief (under ~180 words, markdown).\n"
    "NUMBERS: every number you write must be copied verbatim from DATA (code output). Never calculate, round, "
    "convert or estimate. If a number is not in DATA, say you don't have that figure.\n"
    "SCOPE RULES:\n"
    "- Single stocks: never say whether to buy/sell an individual stock and never predict prices or returns; explain "
    "you build diversified portfolios, suggest a registered advisor for single-stock decisions, and offer a portfolio.\n"
    "- Tax/legal: explain account types only at a general, conceptual level (e.g. TFSA vs RRSP); no personalised "
    "recommendation and no dollar amounts; recommend a CPA, tax professional or financial planner.\n"
    "- Guarantees: no investment can guarantee high returns without risk; higher return requires higher risk. Flag ads "
    "promising guaranteed high returns as likely misleading and suggest checking with a CIRO-registered advisor.\n"
    "- Fear / panic selling: acknowledge the feeling, do not encourage selling, ask whether their situation or goal has "
    "changed, explain that selling after a drop locks in losses; use stress-test figures from DATA if present and "
    "offer a stress test.\n"
    "- Choosing between strategies: give an objective comparison from DATA (return, risk, diversification) and ask "
    "the client which matters more to them; do not pick for them unless they state a preference.\n"
    "- Assets: only the 24 assets in DATA.universe can be held; say clearly if a company they mention is not in it.\n"
    "- Historical data: data runs from April 2021; COVID-19 (2020) is not covered - say so and offer the 2022 "
    "stress test. Never claim bonds protected portfolios in 2022 if DATA shows bond ETFs fell.\n"
    "- Fractional units: depend on the broker; the fractional optimum is a different, less constrained portfolio.\n"
    "- Disclaimer / 'is this a good investment?': restate DATA.disclaimer, explain outputs are educational and based "
    "on historical data and optimisation, never say it is a good investment or guaranteed.\n"
    "- Profile changes: if the client wants different settings, tell them to say what changed (e.g. 'change my risk "
    "to high') and the portfolio will be rebuilt.\n"
    "Do not add a disclaimer line yourself (the system appends it)."
)


def answer_with_data(question: str, data: dict, history: list | None = None, during_profiling: bool = False) -> str:
    """LLM answer grounded in `data` (code/tool output). If a reply has numbers that are not in `data`, it is
    regenerated once with those numbers named; if that still fails, only the lines containing them are removed."""
    from compliance_agent import check_numbers_grounded
    convo = "\n".join(f"{r}: {c}" for r, c in (history or [])[-6:])
    note = ("\nThe questionnaire is still in progress: answer the question, then stop - the system will re-ask the "
            "pending questionnaire question. Do not build or describe a portfolio yet." if during_profiling else "")
    prompt = (f"DATA:\n{json.dumps(data, indent=1, default=str)}\n\nRECENT CONVERSATION:\n{convo}\n\n"
              f"The client now says: {question}{note}")
    try:
        from compliance_agent import DISCLAIMER
        allowed = {"data": data, "client_message": question}   # numbers the client wrote may be repeated
        reply = llm.generate_text(prompt, system=ANSWER_SYSTEM_PROMPT, effort=llm.EFFORT_WRITE)
        bad = _ungrounded(reply, allowed)
        if bad:   # say exactly which numbers are not code output, then retry once
            reply = llm.generate_text(
                prompt + f"\n\n(Your previous answer used numbers that are not in DATA: {', '.join(bad)}. Do not use "
                "them or derive new numbers; quote only numbers that appear verbatim in DATA.)",
                system=ANSWER_SYSTEM_PROMPT, effort=llm.EFFORT_WRITE)
            if _ungrounded(reply, allowed):   # last resort: keep the answer, drop only the lines with such numbers
                kept = [ln for ln in reply.split("\n") if not check_numbers_grounded(ln, allowed)]
                reply = "\n".join(kept).strip()
                if len(reply) < 80:
                    reply = ("I can only quote figures produced by our portfolio model, and I don't have a verified "
                             "figure for that. Please see the tables and charts above.")
        # the system appends the disclaimer once; drop any copy the model wrote itself
        reply = re.sub(r"\n*_?\s*" + re.escape(DISCLAIMER) + r"\s*_?", "", reply).strip()
        return reply
    except Exception as e:
        return f"Sorry, I couldn't reach the language model right now ({type(e).__name__}). Please try again."


def _ungrounded(text: str, allowed: dict) -> list[str]:
    from compliance_agent import check_numbers_grounded
    return [re.search(r"number (\S+) ", m).group(1) for m in check_numbers_grounded(text, allowed)]


def answer_followup(question: str, answers: dict, portfolio: dict, history: list | None = None) -> str:
    """Backwards-compatible wrapper: answer about the shown portfolio."""
    return answer_with_data(question, {"portfolio_facts": portfolio["facts"]}, history)


# ------------------------------------------
# 4. Part 1 personas (answers as the profiling agent would extract them)
# ------------------------------------------
PERSONA_ANSWERS = {
    "A": {"age": "18-24", "employment": "student", "num_dependants": 0, "income": "Under $20,000",
          "invest_amount": 4000, "other_assets": "$2,000 emergency savings", "objective_str": "General savings",
          "target_yn": "Yes", "target": 4000, "target_years": "0-1 years", "exp": "none",
          "rr_familiarity": "Yes", "desc": "Very low risk", "horizon_years": "0–1 years", "sectors": "No exclusions",
          # Part 1: "capital preservation, cannot risk losing money" -> Yes (team to confirm; see README note)
          "cash_needed_now": 0, "principal_protection": "Yes"},
    "B": {"age": "25-34", "employment": "employed", "num_dependants": 0, "income": "$80,000 – $99,999",
          "invest_amount": 35000, "other_assets": "$20,000 emergency savings, $15,000 other investments",
          "objective_str": "Buy a house", "target_yn": "Yes", "target": 50000, "target_years": "4-10 years",
          "exp": "some", "rr_familiarity": "Yes", "desc": "Moderate risk", "horizon_years": "4–10 years",
          "sectors": "Exclude fossil-fuel companies and AI-focused companies",
          "cash_needed_now": 0, "principal_protection": "No"},
    "C": {"age": "45-54", "employment": "employed", "num_dependants": 0, "income": "$150,000+",
          "invest_amount": 80000, "other_assets": "paid-off home, pension, $60,000 emergency savings",
          "objective_str": "Save for retirement", "target_yn": "Yes", "target": 140000, "target_years": "10+ years",
          "exp": "some", "rr_familiarity": "Yes", "desc": "High risk", "horizon_years": "10+ years",
          "sectors": "No exclusions",
          "cash_needed_now": 0, "principal_protection": "No"},
}

if __name__ == "__main__":
    for k, ans in PERSONA_ANSWERS.items():
        p = get_portfolio(ans)
        print(f"\n=== Persona {k} ===")
        print(template_response(p))
