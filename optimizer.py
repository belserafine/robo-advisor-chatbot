# ------------------------------------------
# PART 3 OPTIMIZER (ported from Part3_Portfolio_Optimization.ipynb)
# ------------------------------------------
# Same formulations, constraints, solver (SciPy SLSQP) and MILP unit conversion as the Part 3
# notebook. The only change: notebook globals (TICKERS, Q, META, P_NOW) are passed as arguments.
#
# Persona feasible set W_p (Strategies 1 and 2):
#   sum w = 1;  0 <= w_i <= u_p;  w_i = 0 for excluded i
#   sum_{i in sector k} w_i <= sector_cap        (single-industry sectors only)
#   sum_{i in duplicate group} w_i <= u_p        (SPY+VTI, XIU.TO+XESG.TO)
#   sum_{FI} w_i >= fi_min;  sum_{commodity} w_i <= commodity_max
# Strategy 1: max mu_hat' w  s.t. w'Qw <= vol_cap^2, w in W_p   (sample means)
# Strategy 2: max pi' w      s.t. w'Qw <= vol_cap^2, w in W_p   (pi = rf + delta Q w_ref, reverse optimization)
# Benchmark : equal weight over allowed assets
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

RF = 0.023            # 3-month GoC T-bill (Part 2 / Part 3)
TC = 0.0005           # 5 bps per trade, one way (Part 2 handoff / Part 3)
SR_REF = 0.40         # assumed Sharpe ratio of the reference portfolio (Part 3, Strategy 2)
W_REF = {"XIU.TO": 0.25, "VTI": 0.35, "ZAG.TO": 0.25, "AGG": 0.15}   # Canadian 60/40 reference
DUPLICATE_GROUPS = [["SPY", "VTI"], ["XIU.TO", "XESG.TO"]]
SECTOR_CAP_EXEMPT = {"Broad Market", "Fixed Income"}
RISK_TO_VOLCAP = {"low": 0.065, "moderate": 0.10, "high": 0.14}


def horizon_to_fi_min(years: float) -> float:
    return 0.50 if years <= 2 else (0.20 if years <= 6 else 0.0)


def make_profile(name, budget, risk, horizon, max_weight, exclude=(), commodity_max=None, sector_cap=0.40):
    return dict(name=name, budget=budget, risk=risk, horizon=horizon,
                vol_cap=RISK_TO_VOLCAP[risk], fi_min=horizon_to_fi_min(horizon),
                max_weight=max_weight, commodity_max=commodity_max if commodity_max is not None else max_weight,
                sector_cap=sector_cap, exclude=list(exclude))


# ------------------------------------------
# Optimization engine
# ------------------------------------------
def linear_constraints(p, tickers, meta):
    """List of (a, lo, hi, name) meaning lo <= a @ w <= hi, plus per-asset bounds, for profile p."""
    n = len(tickers)
    sector, aclass = meta["Sector"].values, meta["AssetClass"].values
    rows = [(np.ones(n), 1.0, 1.0, "budget")]
    for grp in DUPLICATE_GROUPS:
        rows.append((np.isin(tickers, grp).astype(float), 0.0, p["max_weight"], "dup:" + "+".join(grp)))
    for sec in sorted(set(sector) - SECTOR_CAP_EXEMPT):
        a = (sector == sec).astype(float)
        if a.sum() > 1:
            rows.append((a, 0.0, p["sector_cap"], "sector:" + sec))
    rows.append(((aclass == "Fixed Income").astype(float), p["fi_min"], 1.0, "fixed income floor"))
    rows.append(((aclass == "Commodity").astype(float), 0.0, p["commodity_max"], "commodity cap"))
    ub = np.where(np.isin(tickers, p["exclude"]), 0.0, p["max_weight"])
    return rows, list(zip(np.zeros(n), ub))


def _scipy_cons(rows):
    cons = []
    for a, lo, hi, _ in rows:
        if lo == hi:
            cons.append({"type": "eq", "fun": (lambda w, a=a, lo=lo: a @ w - lo), "jac": (lambda w, a=a: a)})
        else:
            cons.append({"type": "ineq", "fun": (lambda w, a=a, lo=lo: a @ w - lo), "jac": (lambda w, a=a: a)})
            cons.append({"type": "ineq", "fun": (lambda w, a=a, hi=hi: hi - a @ w), "jac": (lambda w, a=a: -a)})
    return cons


def _clean(w):
    w = np.where(np.abs(w) < 1e-7, 0.0, w)
    return np.clip(w, 0, None)


def min_variance(p, Q, tickers, meta):
    rows, bounds = linear_constraints(p, tickers, meta)
    x0 = np.array([b[1] for b in bounds]); x0 = x0 / x0.sum()
    res = minimize(lambda w: w @ Q @ w, x0, jac=lambda w: 2 * Q @ w, bounds=bounds,
                   constraints=_scipy_cons(rows), method="SLSQP", options={"maxiter": 2000, "ftol": 1e-14})
    if not res.success:
        raise ValueError(f"min-variance solve failed: {res.message}")
    return _clean(res.x)


def markowitz_max_return(m, p, Q, tickers, meta, vol_cap=None):
    """max m'w  s.t.  w'Qw <= vol_cap^2,  w in W_p."""
    vol_cap = p["vol_cap"] if vol_cap is None else vol_cap
    w_mv = min_variance(p, Q, tickers, meta)
    vol_floor = np.sqrt(w_mv @ Q @ w_mv)
    if vol_floor > vol_cap + 1e-9:
        raise ValueError(f"lowest achievable volatility under these constraints is {vol_floor:.2%}, "
                         f"above the {vol_cap:.2%} cap")
    rows, bounds = linear_constraints(p, tickers, meta)
    cons = _scipy_cons(rows) + [{"type": "ineq", "fun": lambda w: vol_cap**2 - w @ Q @ w,
                                 "jac": lambda w: -2 * Q @ w}]
    ub = np.array([b[1] for b in bounds])
    best = None
    for x0 in (w_mv, ub / ub.sum()):
        res = minimize(lambda w: -(m @ w), x0, jac=lambda w: -m, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 3000, "ftol": 1e-14})
        w = _clean(res.x)
        feasible = (w @ Q @ w <= vol_cap**2 * (1 + 1e-6)) and abs(w.sum() - 1) < 1e-6
        if feasible and (best is None or m @ w > m @ best):
            best = w
    if best is None:
        raise ValueError("solver failed to find a feasible solution")
    return best


def equal_weight(p, tickers):
    allowed = ~np.isin(tickers, p["exclude"])
    return allowed / allowed.sum()


def implied_returns(Q, tickers, w_ref=W_REF, sr_ref=SR_REF, rf=RF):
    w = pd.Series(0.0, index=tickers); w[list(w_ref)] = list(w_ref.values())
    w = w.values
    sigma_ref = np.sqrt(w @ Q @ w)
    delta = sr_ref / sigma_ref
    return rf + delta * Q @ w, delta


def investable(solve_fn, p, prices, tickers, max_iter=10):
    """Re-solve after dropping assets whose target amount is below one unit's price."""
    q = dict(p, exclude=list(p["exclude"])); dropped = []
    for _ in range(max_iter):
        w = solve_fn(q)
        bad = [t for t, wi, pr in zip(tickers, w, prices) if wi > 1e-4 and wi * p["budget"] < pr]
        if not bad:
            return w, q, dropped
        dropped += bad; q["exclude"] += bad
    raise RuntimeError("investability filter did not converge")


def to_units(w, prices, budget, tc=TC, max_weight=None, rows=None, Q=None, vol_cap=None, protect=None):
    """Whole units N_i tracking target amounts w_i*B within budget incl. fees (MILP, HiGHS).

    Part 3 notebook version + three Part 5 additions so the FINAL whole-unit portfolio (weights =
    value / initial budget, the compliance checker's definition) still meets the persona constraints:
      max_weight: N_i p_i <= max_weight * B  (Part 3 version let Persona C hold VTI at 30.2% > 30%)
      rows:       linear constraints (a, lo, hi, name) from linear_constraints(); each becomes
                  lo*B <= sum_i a_i N_i p_i <= hi*B (fixed-income floor, commodity/sector/duplicate caps)
      Q, vol_cap: after rounding, remove one unit at a time from the asset with the largest marginal
                  risk until sqrt(w'Qw) <= vol_cap (cash carries no risk); `protect` assets are never cut.
    Returns (units array, dict(invested, fees, cash))."""
    idx = np.where(w > 1e-4)[0]
    k = len(idx); pr = prices[idx]; tgt = w[idx] * budget
    n_max = np.floor(budget / pr)
    if max_weight is not None:
        n_max = np.minimum(n_max, np.floor(max_weight * budget / pr + 1e-9))
    c = np.r_[np.zeros(k), np.ones(k)]
    I = np.eye(k)
    A = [np.hstack([np.diag(pr), -I]), np.hstack([-np.diag(pr), -I]), np.r_[pr * (1 + tc), np.zeros(k)][None, :]]
    ub = [tgt, -tgt, [budget]]
    for a, lo, hi, name in rows or []:
        if name == "budget":
            continue
        row = np.r_[a[idx] * pr, np.zeros(k)]
        A.append(row[None, :]); ub.append([hi * budget])
        if lo > 0:
            A.append(-row[None, :]); ub.append([-lo * budget])
    A, ub = np.vstack(A), np.concatenate(ub)
    integrality = np.r_[np.ones(k), np.zeros(k)]
    res = milp(c, constraints=LinearConstraint(A, -np.inf, ub), integrality=integrality,
               bounds=Bounds(np.zeros(2 * k), np.r_[n_max, np.full(k, np.inf)]))
    if not res.success and res.status == 4:
        # HiGHS numerical "solve error" (seen for some budgets, e.g. $37,360): solve the identical problem with
        # dollar rows expressed as shares of the budget (d_i scaled the same way). Only used as a fallback, so
        # every case that already solved (incl. the Part 3/4 reproductions) is unchanged.
        S = np.r_[np.ones(k), np.full(k, budget)]               # x = (N, d) with d = budget * d'
        res = milp(c * S / budget, constraints=LinearConstraint(A * S / budget, -np.inf, ub / budget),
                   integrality=integrality, bounds=Bounds(np.zeros(2 * k), np.r_[n_max, np.full(k, np.inf)]))
    if not res.success:
        raise ValueError(f"integer-unit conversion failed: {res.message}")
    N = np.zeros(len(w)); N[idx] = np.round(res.x[:k])

    if Q is not None and vol_cap is not None:
        protect = np.zeros(len(w), bool) if protect is None else np.asarray(protect, bool)
        while True:
            wf = N * prices / budget
            if np.sqrt(wf @ Q @ wf) <= vol_cap:
                break
            marginal = np.where((N > 0) & ~protect, Q @ wf, -np.inf)
            i = int(np.argmax(marginal))
            if not np.isfinite(marginal[i]):
                raise ValueError("cannot bring the whole-unit portfolio under the volatility cap")
            N[i] -= 1

    value = N * prices; fees = value.sum() * tc
    return N.astype(int), {"invested": float(value.sum()), "fees": float(fees),
                           "cash": float(budget - value.sum() - fees)}


def check_portfolio(w, p, Q, tickers, meta, is_benchmark=False, tol=1e-6) -> list[str]:
    """Part 3 Section 10 constraint checks, returning a list of violations instead of asserting."""
    issues = []
    if abs(w.sum() - 1) >= tol: issues.append(f"weights sum to {w.sum():.6f}")
    if (w < -tol).any(): issues.append("short position")
    if np.any(w[np.isin(tickers, p["exclude"])] > tol): issues.append("excluded asset held")
    if is_benchmark:
        return issues
    if (w > p["max_weight"] + tol).any(): issues.append("asset cap breached")
    rows, _ = linear_constraints(p, tickers, meta)
    for a, lo, hi, name in rows:
        if not (lo - tol <= a @ w <= hi + tol): issues.append(f"{name} violated ({a @ w:.4f})")
    if np.sqrt(w @ Q @ w) > p["vol_cap"] + tol: issues.append("volatility cap breached")
    return issues


# ------------------------------------------
# PART 4 (ported from Part3_Portfolio_Optimization_Part4_Validation.ipynb)
# Section 2: out-of-sample backtest, Section 3: performance metrics.
# Test period 1 Apr - 18 Nov 2026 (daily). The portfolio starts as whole units bought at 1 Apr prices and is
# rebalanced back to its target weights on the first trading day of each month (May - Nov); every trade pays
# TC (5 bp one way) out of cash, and the budget is reduced until cash after costs is non-negative.
# The equal-weight benchmark uses exact (fractional) units, as in Part 3/4.
# `unit_fn(w, prices, budget) -> (units, {"invested","fees","cash"})` converts weights to units; with
# `to_units(w, p, B)` (no extra arguments) this is exactly the Part 4 notebook calculation.
# ------------------------------------------
def to_units_fractional(w, prices, budget, tc=TC):
    """Part 3 to_units(fractional=True): exact units w_i*B / (p_i (1+tc))."""
    N = w * budget / (prices * (1 + tc)); value = N * prices; fees = value.sum() * tc
    return N, {"invested": float(value.sum()), "fees": float(fees), "cash": max(0.0, float(budget - value.sum() - fees))}


def rebalance_dates(px: pd.DataFrame) -> list:
    return list(px.groupby(px.index.to_period("M")).apply(lambda x: x.index[0]))[1:]


def _rebalance_units(w, prices, units_old, value, unit_fn, tc=TC):
    budget = value
    for _ in range(20):
        units_new, _ = unit_fn(w, prices, budget)
        trade_value = np.abs(units_new - units_old) * prices
        cost = tc * trade_value.sum()
        cash_new = value - units_new @ prices - cost
        if cash_new >= -1e-9:
            return np.asarray(units_new, float), max(cash_new, 0.0), cost, trade_value
        budget -= (-cash_new) + 0.01
    raise RuntimeError("could not rebalance within budget")


def run_backtest(w, px: pd.DataFrame, budget: float, unit_fn, tc=TC) -> tuple[pd.Series, pd.DataFrame]:
    """Part 4 Section 2: daily portfolio value and the trade log."""
    rebal = set(rebalance_dates(px))
    units, c0 = unit_fn(w, px.iloc[0].values, budget)
    units = np.asarray(units, float); cash = c0["cash"]
    trades = [{"Date": px.index[0], "Value before": budget, "Traded value": c0["invested"], "Cost": c0["fees"],
               "Turnover": np.nan, "Cash after": cash}]
    values = []
    for date, row in px.iterrows():
        prices = row.values
        if date in rebal:
            v_before = units @ prices + cash
            units, cash, cost, tv = _rebalance_units(w, prices, units, v_before, unit_fn, tc)
            trades.append({"Date": date, "Value before": v_before, "Traded value": tv.sum(), "Cost": cost,
                           "Turnover": 0.5 * tv.sum() / v_before, "Cash after": cash})
        values.append(units @ prices + cash)
    return pd.Series(values, index=px.index), pd.DataFrame(trades)


def perf_metrics(v: pd.Series, budget: float, trades: pd.DataFrame, vol_cap: float | None = None, rf=RF) -> dict:
    """Part 4 Section 3."""
    r = v.pct_change().dropna()
    n_days = len(r)
    ann_ret = (v.iloc[-1] / budget) ** (252 / n_days) - 1
    ann_vol = r.std() * np.sqrt(252)
    to = trades["Turnover"].dropna()
    out = {"final_value": float(v.iloc[-1]), "total_return": float(v.iloc[-1] / budget - 1),
           "annualized_return": float(ann_ret), "annualized_volatility": float(ann_vol),
           "sharpe": float((ann_ret - rf) / ann_vol) if ann_vol > 0 else None,
           "max_drawdown": float((v / v.cummax() - 1).min()),
           "avg_monthly_turnover": float(to.mean()) if len(to) else 0.0,
           "annualized_turnover": float(to.mean() * 12) if len(to) else 0.0,
           "trading_costs": float(trades["Cost"].sum())}
    if vol_cap is not None:
        out["within_vol_cap"] = bool(ann_vol <= vol_cap)
    return out


def backtest(w, px: pd.DataFrame, budget: float, unit_fn, vol_cap: float | None = None) -> dict:
    """Part 4 backtest + metrics for one portfolio. Returns {"series", "trades", "metrics"}."""
    series, trades = run_backtest(w, px, budget, unit_fn)
    return {"series": series, "trades": trades, "metrics": perf_metrics(series, budget, trades, vol_cap)}
