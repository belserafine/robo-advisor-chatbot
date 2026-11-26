# ------------------------------------------
# PART 4 CORRECTNESS TESTS (ported from the Part 4 notebook, Section 1) - run on the chatbot's own code
# ------------------------------------------
#   python validation.py
# 1 Budget      : invested + fees <= budget, no negative units, fees = TC x invested, cash within the rounding bound
# 2 Constraints : weights sum to 1, long only, exclusions, asset/sector/duplicate caps, fixed-income floor,
#                 commodity cap, volatility cap (target weights) - plus the same checks on the FINAL whole units
# 3 Independent : closed-form minimum-variance portfolio on 4 assets vs our optimizer
# Personas A/B/C x Strategy 1/2 x both execution dates, using exactly the functions the chatbot calls.
from __future__ import annotations

import numpy as np
import pandas as pd

import compliance_agent
import market_data
import optimizer as opt
import portfolio_agent as pa

TOL = 1e-6


def _run(log, group, case, fn):
    try:
        out = fn(); log.append({"Test": group, "Case": case, "Status": "PASS", "Detail": out or ""})
    except AssertionError as e:
        log.append({"Test": group, "Case": case, "Status": "FAIL", "Detail": str(e)})


def run_all(verbose: bool = True) -> pd.DataFrame:
    d = market_data.load(); T, meta, Q, mu, px = d["tickers"], d["meta"], d["Q"].values, d["mu"].values, d["px_out"]
    pi, _ = opt.implied_returns(Q, T)
    log = []
    for k, ans in pa.PERSONA_ANSWERS.items():
        ans = dict(ans, principal_protection="No")
        prof, _ = pa.answers_to_profile(ans, meta)
        for date in (px.index[0], px.index[-1]):
            prices = px.loc[date].values
            for s, m in (("S1", mu), ("S2", pi)):
                w, _, _ = opt.investable(lambda q, m=m: opt.markowitz_max_return(m, q, Q, T, meta), prof, prices, T)
                rows, _ = opt.linear_constraints(prof, T, meta)
                N, cash = opt.to_units(w, prices, prof["budget"], max_weight=prof["max_weight"], rows=rows, Q=Q,
                                       vol_cap=prof["vol_cap"], protect=(meta["AssetClass"] == "Fixed Income").values)
                tag = f"{k}-{s} @ {date.date()}"

                def budget_test():
                    B = prof["budget"]
                    assert cash["invested"] + cash["fees"] <= B + 1e-6, "over budget"
                    assert (N >= 0).all(), "negative units"
                    assert abs(cash["fees"] - opt.TC * cash["invested"]) < 1e-6, "fees != TC x invested"
                    held = w > 1e-4
                    bound = max(0.5 * prices[held].sum(), prices[held].min())
                    assert cash["cash"] <= bound + 1e-6, f"cash {cash['cash']:.2f} > rounding bound {bound:.2f}"
                    return f"invested ${cash['invested']:,.2f} + fees ${cash['fees']:.2f}; cash ${cash['cash']:,.2f}"

                def constraint_test():
                    issues = opt.check_portfolio(w, prof, Q, T, meta)
                    assert not issues, "; ".join(issues)
                    return f"target weights: all constraints hold (vol {np.sqrt(w @ Q @ w):.2%}, cap {prof['vol_cap']:.1%})"

                def final_units_test():
                    port = {"profile": prof, "budget": prof["budget"], "price_date": str(date.date()),
                            "holdings": [{"ticker": t, "units": int(n)} for t, n in zip(T, N) if n > 0]}
                    ok, bad, vol = compliance_agent.final_holdings_check(port)
                    assert ok, "; ".join(bad)
                    return f"final whole units: all caps/floors hold (vol {vol:.2%})"

                _run(log, "1 Budget", tag, budget_test)
                _run(log, "2 Constraints (target)", tag, constraint_test)
                _run(log, "2 Constraints (final units)", tag, final_units_test)

    def closed_form():
        keep = ["ZAG.TO", "AGG", "XIU.TO", "GLD"]
        ix = [T.index(t) for t in keep]
        x = np.linalg.solve(Q[np.ix_(ix, ix)], np.ones(len(ix))); w_cf = x / x.sum()
        p = dict(name="test", budget=1, vol_cap=1.0, fi_min=0.0, max_weight=1.0, commodity_max=1.0, sector_cap=1.0,
                 exclude=[t for t in T if t not in keep])
        err = np.abs(opt.min_variance(p, Q, T, meta)[ix] - w_cf).max()
        assert err < 1e-5, f"max weight difference {err:.2e}"
        return f"max weight difference {err:.1e} (tolerance 1e-5)"

    _run(log, "3 Independent solution", "Closed-form min-variance, 4 assets", closed_form)
    df = pd.DataFrame(log)
    if verbose:
        print(df.groupby("Test")["Status"].value_counts().unstack(fill_value=0))
        fails = df[df.Status == "FAIL"]
        if len(fails):
            print(fails.to_string())
    return df


if __name__ == "__main__":
    run_all()
