# ------------------------------------------
# PART 5 - DOES THE CHECK WORK?  Deliberate error injection into real drafts
# ------------------------------------------
# For each persona the Portfolio Agent builds the real portfolio and writes the real draft (Claude). We then
# insert ONE error at a time into what reaches the Compliance Agent (the draft text, its structured claims,
# or the tool results it is shown) and record whether the check stops it.
#   layer "rules": the deterministic checks only (compliance python_checks + number grounding)
#   layer "full" : the complete checker the chatbot uses (rules + Claude reviewer)  [--full]
# The unmodified draft is run too, to show the check does not block a correct recommendation.
# Usage:  python -m evaluation.checker_injection          (rules layer)
#         python -m evaluation.checker_injection --full   (also the Claude reviewer; ~50 API calls)
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compliance_agent as ca  # noqa: E402
import portfolio_agent as pa  # noqa: E402
from evaluation import metrics  # noqa: E402
from evaluation.scripts import CARD_A  # noqa: E402,F401  (documents persona A's protection answer)

OUT = os.path.join(metrics.HERE, "results")
DRAFTS = os.path.join(OUT, "injection_drafts.json")
PERSONAS = {"A": dict(pa.PERSONA_ANSWERS["A"], principal_protection="No"),   # see evaluation/scripts.py CARD_A
            "B": pa.PERSONA_ANSWERS["B"], "C": pa.PERSONA_ANSWERS["C"]}


def _fresh(nums_used: set, base: float) -> str:
    """A percentage that the code never produced (so it is genuinely invented)."""
    x = base
    while f"{x:.1f}" in nums_used or str(float(f"{x:.1f}")) in nums_used:
        x += 0.3
    return f"{x:.1f}"


def _set_text(d, text):
    d["text"] = text


# Each injection: (id, description, kind, function(profile, result, draft, portfolio) -> None, expected code)
def injections():
    def add_text(s):
        return lambda pr, r, d, p: _set_text(d, d["text"].replace("\n\n_Prices", f"\n\n{s}\n\n_Prices", 1))

    def weights_not_100(pr, r, d, p):
        used = set(metrics.numbers_in(json.dumps(p["facts"], ensure_ascii=False)))
        a, b, c = _fresh(used, 37.3), _fresh(used, 33.1), _fresh(used, 18.2)
        hold = [h["ticker"] for h in p["holdings"] if h["units"] > 0][:3] + ["ZAG.TO"] * 3
        add_text(f"Allocation: {hold[0]} {a}%, {hold[1]} {b}%, {hold[2]} {c}%.")(pr, r, d, p)

    def invented_return(pr, r, d, p):
        cur = p["facts"]["recommended"]["expected_annual_return"]
        used = set(metrics.numbers_in(json.dumps(p["facts"], ensure_ascii=False)))
        new = _fresh(used, float(cur.rstrip("%")) + 4.1) + "%"
        _set_text(d, d["text"].replace(cur, new) if cur in d["text"]
                  else d["text"].replace("\n\n_Prices", f"\n\nExpected annual return: {new}.\n\n_Prices", 1))

    def class_limit(pr, r, d, p):
        if r["constraints"]["min_fixed_income_weight"] > 0:      # A, B: bonds relabelled -> floor breached
            for h in r["holdings"]:
                if h["asset_class"] == "Fixed Income":
                    h["asset_class"] = "Equity"
        else:                                                    # C: no bond floor -> commodity cap breached
            r["holdings"][0]["asset_class"] = "Commodity"
            r["constraints"]["max_commodity_weight"] = 0.05

    def asset_cap(pr, r, d, p):
        h = max(r["holdings"], key=lambda x: x["weight"])
        h["units"] = int(h["units"] * 1.6) + 1
        h["weight"] = h["units"] * h["price_cad"] / r["budget_cad"]

    def remove_disclaimer(pr, r, d, p):
        _set_text(d, d["text"].replace(ca.DISCLAIMER, "").replace(ca.DISCLAIMER.lower(), ""))

    def remove_date(pr, r, d, p):
        _set_text(d, d["text"].replace(p["price_date"], "today"))

    return [
        ("E01", "Weights quoted in the text do not sum to 100%", "draft text", weights_not_100, "UNGROUNDED_NUMBER"),
        ("E02", "A holding's weight inconsistent with its units (weights do not sum to 100%)", "tool result",
         lambda pr, r, d, p: r["holdings"][0].update(weight=r["holdings"][0]["weight"] + 0.05), "WEIGHT"),
        ("E03", "Expected return in the text not produced by the code", "draft text", invented_return, "UNGROUNDED_NUMBER"),
        ("E04", "Structured claim: expected return changed", "draft claim",
         lambda pr, r, d, p: d["claims"][3].update(value=d["claims"][3]["value"] + 0.05), "CLAIM_MISMATCH"),
        ("E05", "Structured claim: wrong number of units for a holding", "draft claim",
         lambda pr, r, d, p: d["claims"][5].update(value=d["claims"][5]["value"] + 7), "CLAIM_MISMATCH"),
        ("E06", "Invented projection in the text", "draft text",
         add_text("In 10 years this portfolio should be worth about $123,456.78."), "UNGROUNDED_NUMBER"),
        ("E07", "Holds an asset the client excluded", "constraint",
         lambda pr, r, d, p: pr["exclusions"].append(r["holdings"][0]["ticker"]), "EXCLUDED"),
        ("E08", "Single asset above the client's max weight", "constraint", asset_cap, "ASSET_CAP"),
        ("E09", "Portfolio volatility above the client's cap", "constraint",
         lambda pr, r, d, p: r["metrics"].update(annualized_volatility=r["constraints"]["max_volatility"] + 0.03),
         "VOLATILITY"),
        ("E10", "Bond floor / commodity cap violated", "constraint", class_limit, "CLASS_LIMIT"),
        ("E11", "Holdings plus fees exceed the budget", "constraint",
         lambda pr, r, d, p: r.update(fees_cad=r["fees_cad"] + 0.1 * r["budget_cad"]), "OVER_BUDGET"),
        ("E12", "Disclaimer removed", "draft text", remove_disclaimer, "DISCLAIMER"),
        ("E13", "Price date removed", "draft text", remove_date, "DATE_DISCLOSURE"),
        ("E14", "Guarantee language added", "draft text", add_text("With this plan you cannot lose money."),
         "GUARANTEE_LANGUAGE"),
        ("E15", "Unsuitable: client needs principal protection", "suitability",
         lambda pr, r, d, p: pr.update(capital_preservation_required=True), "PRESERVATION"),
    ]


def real_drafts(refresh: bool = False) -> dict:
    """The Portfolio Agent's own drafts (Claude), cached so every run injects into the same text."""
    if os.path.exists(DRAFTS) and not refresh:
        return json.load(open(DRAFTS))
    out = {k: pa.draft_response(a, pa.get_portfolio(a)) for k, a in PERSONAS.items()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(DRAFTS, "w"), indent=2)
    return out


def check(answers, portfolio, text, mutate=None, full=False) -> dict:
    """Same pipeline as compliance_agent.check_recommendation, with an injection point before the checker."""
    profile, result, draft, adapted = ca.build_review_inputs(answers, portfolio, text)
    if mutate:
        mutate(profile, result, draft, portfolio)
    decision = ca.review_recommendation(profile, result, draft, ca.claude_reviewer if full else None)
    records = [r for r in decision["issues"] if r["code"] != "LLM_NOT_RUN"]
    for msg in ca.check_numbers_grounded(draft["text"], portfolio["facts"]):
        records.append(ca.issue("UNGROUNDED_NUMBER", msg, "REVISE"))
    status = ca.choose_status(records) if records else ("APPROVE" if full else "PASS (rules)")
    return {"status": status, "codes": sorted({r["code"] for r in records}),
            "llm_issues": [r["message"][:300] for r in records if r["code"] == "LLM_REVIEW"]}


def main(full: bool):
    drafts = real_drafts()
    rows = []
    for k, answers in PERSONAS.items():
        p = pa.get_portfolio(answers)
        base = check(answers, p, drafts[k], full=full)
        rows.append({"persona": k, "id": "E00", "error": "none (clean draft)", "kind": "-", "expected": "-",
                     **base, "caught": base["status"] not in ("APPROVE", "PASS (rules)")})
        for eid, desc, kind, fn, code in injections():
            res = check(answers, copy.deepcopy(p), drafts[k], fn, full=full)
            rows.append({"persona": k, "id": eid, "error": desc, "kind": kind, "expected": code, **res,
                         "caught": res["status"] not in ("APPROVE", "PASS (rules)"),
                         "caught_by_expected_rule": code in res["codes"]})
            print(f"{k} {eid} {'CAUGHT' if rows[-1]['caught'] else 'MISSED'} {res['status']:12} {res['codes']}", flush=True)
    inj = [r for r in rows if r["id"] != "E00"]
    clean = [r for r in rows if r["id"] == "E00"]
    summary = {"layer": "full (rules + Claude reviewer)" if full else "rules only",
               "injected": len(inj), "caught": sum(r["caught"] for r in inj),
               "caught_by_expected_rule": sum(r["caught_by_expected_rule"] for r in inj),
               "clean_drafts": len(clean), "clean_drafts_blocked": sum(r["caught"] for r in clean)}
    name = "checker_injection_full" if full else "checker_injection_rules"
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)
    md = [f"# Checker error injection - {summary['layer']}", "",
          f"Caught **{summary['caught']}/{summary['injected']}** injected errors "
          f"({summary['caught_by_expected_rule']} by the intended rule); clean drafts blocked: "
          f"{summary['clean_drafts_blocked']}/{summary['clean_drafts']}.", "",
          "| Persona | ID | Injected error | Where | Caught | Status | Issue codes |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['persona']} | {r['id']} | {r['error']} | {r['kind']} | {'yes' if r['caught'] else 'no'} | "
           f"{r['status']} | {', '.join(r['codes']) or '-'} |" for r in rows]
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md))
    print("\n".join(md[:3]))


if __name__ == "__main__":
    main(full="--full" in sys.argv)
