# ------------------------------------------
# MULTI-AGENT ORCHESTRATION (LangGraph)
# ------------------------------------------
# One graph run = one client turn. The graph state is stored by the caller (app.py keeps it
# in st.session_state) and passed back in on the next turn.
#
#   START -> router (classifies the client message)
#     questionnaire phase:  answer -> profiling --(complete)--> portfolio -> compliance -> respond / withhold
#                           foreign currency -> currency (rate tool, client confirms the CAD amount)
#                           question / out of scope / emotional -> assist (scoped answer, questionnaire continues)
#     after a recommendation: "show Strategy 1/2" -> switch_strategy -> portfolio -> compliance -> ...
#                           profile change       -> update_profile  -> portfolio -> compliance -> ...
#                           anything else        -> followup (Portfolio Agent tools + grounded answer)
#     clarify phase:        client's answer -> clarify_answer -> compliance
#
#   portfolio <-> compliance: REVISE -> rewrite (max MAX_RETRIES) -> fallback template; only APPROVE is shown.
from __future__ import annotations

import copy
import operator
import re
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import compliance_agent
import llm
import market_data
import portfolio_agent
import portfolio_tools
import profiling_agent
import router

MAX_RETRIES = 2        # self-correction rounds between Portfolio and Checking agents
MAX_CLARIFY_ROUNDS = 2 # CLARIFY questions asked to the client before the recommendation is withheld
DONE_MSG = "Thanks, I have everything I need!"  # returned by profiling_agent when complete
_YES = re.compile(r"\b(yes|yeah|yep|sure|ok|okay|fine|correct|go ahead|sounds good|that works|please do|use it)\b", re.I)


class RoboState(TypedDict):
    phase: str                         # "profiling" | "clarify" | "done" | "failed"
    user_message: Optional[str]        # this turn's client message (None on the very first turn)
    prof_state: dict                   # profiling_agent state: {"messages": [...], "answers": {...}}
    portfolio: Optional[dict]          # get_portfolio() output (the only source of numbers)
    strategy: str                      # strategy whose orders are shown ("S2" default, "S1" on request)
    released: bool                     # True once a recommendation passed compliance and was shown
    build_error: Optional[dict]        # why no portfolio could be built (small budget etc.)
    draft: Optional[str]               # current draft written by the Portfolio Agent
    check: Optional[dict]              # last Checking Agent result
    attempts: int                      # number of drafts checked so far
    intent: Optional[dict]             # router output for this turn
    fx_pending: Optional[dict]         # currency conversion in progress {"currency","amount","rate"?,"cad"?}
    replies: list                      # assistant messages produced in this turn: {"content", "portfolio"?}
    history: list                      # full conversation [(role, content)], maintained by run_turn()
    clarifications: list               # [{"question", "client_answer"}] answers to CLARIFY questions
    pending_clarify: Optional[str]     # the CLARIFY question waiting for the client's answer
    prev: Optional[dict]               # last released recommendation, restored if a switch / update fails
    pipeline_log: Annotated[list, operator.add]   # audit trail across turns (appended by nodes)


DEFAULTS = {"strategy": portfolio_agent.RECOMMENDED, "released": False, "build_error": None, "intent": None,
            "fx_pending": None, "clarifications": [], "pending_clarify": None, "prev": None}


def _snapshot(state) -> Optional[dict]:
    if not state["released"]:
        return None
    return {"portfolio": state["portfolio"], "strategy": state["strategy"], "check": state["check"],
            "prof_state": state["prof_state"]}


def _restore(state, reasons: list[str], what: str) -> dict:
    """A switch / profile change could not be released: keep the previous recommendation."""
    p = state["prev"]
    text = (f"I couldn't release the {what}. {_plain(reasons[:4], ask=False)}"
            + f"\n\nYour previous recommendation (**{portfolio_agent.STRATEGY_NAMES[p['strategy']]}**) still stands.")
    return {"portfolio": p["portfolio"], "strategy": p["strategy"], "check": p["check"], "prof_state": p["prof_state"],
            "phase": "done", "released": True, "prev": None, "replies": state["replies"] + [{"content": text}]}


def new_state() -> RoboState:
    return {
        "phase": "profiling", "user_message": None,
        "prof_state": {"messages": [], "answers": profiling_agent.questionnaire_answers().model_dump()},
        "portfolio": None, "draft": None, "check": None, "attempts": 0,
        "replies": [], "history": [], "pipeline_log": [], **copy.deepcopy(DEFAULTS),
    }


def _pending_question(prof: dict) -> str | None:
    """The questionnaire question currently waiting for an answer."""
    msgs = prof.get("messages") or []
    if msgs and msgs[-1][0] == "bot" and msgs[-1][1] in profiling_agent.QUESTION_TEXT.values():
        return msgs[-1][1]
    return None


def _base_data() -> dict:
    u = portfolio_tools.universe()
    return {"disclaimer": compliance_agent.DISCLAIMER, "universe_size": len(u),
            "data_coverage": "estimation: weekly prices April 2021 - March 2026; test: daily prices April - November "
                             "2026; COVID-19 (2020) is not covered",
            "universe": [x["ticker"] + " " + x["name"] for x in u]}


def _not_in_universe(names: list[str]) -> list[str]:
    meta = market_data.load()["meta"]
    known = {t.lower() for t in meta.index} | {t.split(".")[0].lower() for t in meta.index}
    known_names = " ".join(meta["Name"].str.lower())
    aliases = set(portfolio_agent._COMPANY_ALIASES)
    out = []
    for n in names:
        k = n.lower().strip()
        if k and k not in known and k not in aliases and k not in known_names:
            out.append(n)
    return out


# ------------------------------------------
# Router
# ------------------------------------------
def router_node(state: RoboState) -> dict:
    msg, phase = state["user_message"], state["phase"]
    if msg is None:
        intent = {"intent": "answer", "strategy": "none", "not_in_universe": []}
    elif phase == "profiling" and (state["fx_pending"] or router.detect_currency(msg)[0]):
        intent = {"intent": "currency", "strategy": "none", "not_in_universe": []}
    else:
        last_bot = state["history"][-1][1] if state["history"] else ""
        r = router.classify(msg, phase, last_bot)
        intent = {"intent": r.intent, "strategy": r.strategy,
                  "not_in_universe": _not_in_universe(r.mentioned_companies)}
    return {"intent": intent, "pipeline_log": [{"node": "router", **intent}]}


def route_after_router(state: RoboState) -> str:
    i = state["intent"]["intent"]
    if state["phase"] == "profiling":
        if i == "currency":
            return "currency"
        return "profiling" if i in ("answer", "change_profile") else "assist"
    if i == "show_strategy" and state["phase"] == "done":
        return "switch_strategy"
    if i == "change_profile":
        return "update_profile"
    return "followup"


# ------------------------------------------
# Questionnaire phase
# ------------------------------------------
_NO_EXCLUSIONS = re.compile(r"^\W*(no|none|nothing|n/?a|nope)\b|\bno (sector |ethical |esg )?(restrictions?|exclusions?|"
                            r"preferences?|limits?)\b|\bnot? (exclude|restrict)|\bdon'?t (want|need) to exclude", re.I)


def _vague_exclusion(text) -> str | None:
    """The exclusion answer if it names nothing the screen can act on (and is not a 'no exclusions' answer)."""
    if not isinstance(text, str) or not text.strip() or _NO_EXCLUSIONS.search(text):
        return None
    tickers, _ = portfolio_agent.parse_exclusions(text, market_data.load()["meta"])
    return None if tickers else text.strip()


def profiling_node(state: RoboState) -> dict:
    """Profiling Agent: extract answers from the client message, check contradictions, ask the next question."""
    prof = copy.deepcopy(state["prof_state"])  # run_profiling_turn mutates in place
    bot_msg, prof = profiling_agent.run_profiling_turn(prof, state["user_message"])
    vague = _vague_exclusion(prof["answers"].get("sectors"))
    if vague:   # e.g. "ESG exclusions": nothing we can screen -> ask again instead of guessing
        prof["answers"]["sectors"] = None
        q = profiling_agent.QUESTION_TEXT["sectors"]
        prof["messages"].append(("bot", q))
        bot_msg = (f"I can't tell which companies \"{vague}\" should exclude. I can screen out fossil fuels, AI "
                   f"companies, a sector (e.g. energy, technology) or specific companies in our universe.\n\n{q}")
    complete = bot_msg == DONE_MSG and profiling_agent.all_answered(prof["answers"])
    reply = bot_msg
    missing = state["intent"]["not_in_universe"] if state["intent"] else []
    if missing:
        reply = (f"Note: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in our investable universe "
                 f"of 24 assets, so I can't include {'it' if len(missing) == 1 else 'them'}.\n\n{bot_msg}")
    return {"prof_state": prof, "replies": [{"content": reply}],
            "pipeline_log": [{"node": "profiling", "complete": complete}]}


def assist_node(state: RoboState) -> dict:
    """A question or out-of-scope message during the questionnaire: answer it within scope, then
    re-ask the pending questionnaire question."""
    intent = state["intent"]
    wanted = {intent["intent"]} | router.keyword_tools(state["user_message"])
    data = _base_data()
    if wanted & {"metric_or_asset_info", "other_question"}:
        data["asset_statistics"] = portfolio_tools.for_llm(portfolio_tools.asset_stats(), 2)
    if wanted & {"emotional", "stress_test"}:
        data["stress_test_2022_benchmarks"] = portfolio_tools.for_llm(portfolio_tools.stress_test_2022(None), 1)
    if intent["not_in_universe"]:
        data["not_in_universe"] = intent["not_in_universe"]
    reply = portfolio_agent.answer_with_data(state["user_message"], data, state["history"], during_profiling=True)
    pending = _pending_question(state["prof_state"]) or profiling_agent.QUESTION_TEXT["age"]
    if not state["prof_state"]["messages"]:   # questionnaire not started yet
        prof = copy.deepcopy(state["prof_state"]); prof["messages"].append(("bot", pending))
    else:
        prof = state["prof_state"]
    return {"prof_state": prof, "replies": [{"content": f"{reply}\n\nTo continue: {pending}", "tool_data": data}],
            "pipeline_log": [{"node": "assist", "intent": intent["intent"]}]}


def currency_node(state: RoboState) -> dict:
    """Foreign-currency amount: never treat it as CAD. Ask -> fetch rate (tool) -> client confirms the CAD amount."""
    msg, fx = state["user_message"], copy.deepcopy(state["fx_pending"])
    prof = copy.deepcopy(state["prof_state"])
    log = {"node": "currency"}
    if fx is None:
        code, amount = router.detect_currency(msg)
        fx = {"currency": code, "amount": amount}
        amt = f"{amount:,.0f} {code}" if amount else f"an amount in {code}"
        text = (f"All of our portfolios are in Canadian dollars (CAD), so I won't treat {amt} as CAD. Would you "
                f"like me to convert it at the current {code}/CAD exchange rate, or would you rather tell me the "
                "amount in CAD?")
        # the same message may answer other questions ("I'm 61 ... US$50,000"): keep those answers, but never
        # store the foreign amount - the investment amount is set only after the client confirms the CAD value
        prof["messages"].append(("client", msg))
        amount_before = prof["answers"].get("invest_amount")
        prof["answers"] = profiling_agent.listen_node(prof)["answers"]
        prof["answers"]["invest_amount"] = amount_before
        return {"fx_pending": fx, "prof_state": prof, "replies": [{"content": text}],
                "pipeline_log": [dict(log, step="ask")]}
    if "rate" not in fx:
        cad_given = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:cad|canadian)", msg, re.I)
        if cad_given:   # client gave the CAD amount directly
            fx = {"currency": fx["currency"], "amount": fx["amount"], "rate": None,
                  "cad": float(cad_given.group(1).replace(",", ""))}
        else:
            rate = portfolio_tools.fx_rate(fx["currency"])
            amount = fx["amount"] or 0
            fx.update(rate=rate["cad_per_unit"], cad=round(amount * rate["cad_per_unit"], 2), source=rate["source"],
                      date=rate["date"])
        if fx["rate"]:
            text = (f"Using {fx['rate']} CAD per 1 {fx['currency']} ({fx['source']}, {fx['date']}), "
                    f"{fx['amount']:,.0f} {fx['currency']} is ${fx['cad']:,.2f} CAD. Shall I use ${fx['cad']:,.2f} "
                    "as your investment amount?")
        else:
            text = f"Thanks - shall I use ${fx['cad']:,.2f} CAD as your investment amount?"
        return {"fx_pending": fx, "replies": [{"content": text}], "pipeline_log": [dict(log, step="rate", **fx)]}
    if _YES.search(msg):
        prof["answers"]["invest_amount"] = int(round(fx["cad"]))
        # record the confirmed conversion in the questionnaire transcript, so the contradiction check sees why the
        # profile holds a CAD amount that differs from the foreign amount the client typed
        src = (f"{fx['amount']:,.0f} {fx['currency']} at {fx['rate']} CAD per {fx['currency']}" if fx.get("rate")
               else f"the client's {fx['currency']} amount")
        prof["messages"].append(("bot", f"(Currency conversion confirmed by the client: {src} = ${fx['cad']:,.2f} CAD. "
                                        "All profile amounts are in CAD, so this is the same amount, not a contradiction.)"))
        nxt, prof = profiling_agent.run_profiling_turn(prof, None)
        complete = nxt == DONE_MSG and profiling_agent.all_answered(prof["answers"])
        text = f"Great - I'll use ${fx['cad']:,.2f} CAD.\n\n{nxt}"
        return {"fx_pending": None, "prof_state": prof, "replies": [{"content": text}],
                "pipeline_log": [dict(log, step="confirmed", complete=complete)]}
    text = "No problem - please tell me the amount you'd like to invest in Canadian dollars."
    return {"fx_pending": None, "replies": [{"content": text}], "pipeline_log": [dict(log, step="declined")]}


def route_after_profiling(state: RoboState) -> str:
    return "portfolio" if state["pipeline_log"][-1].get("complete") else END


# ------------------------------------------
# Recommendation
# ------------------------------------------
def portfolio_node(state: RoboState) -> dict:
    """Portfolio Agent: build the portfolio (once per profile/strategy) and write / rewrite the explanation."""
    answers = state["prof_state"]["answers"]
    portfolio = state["portfolio"]
    if portfolio is None:
        try:
            portfolio = portfolio_agent.get_portfolio(answers, state["strategy"])
        except ValueError as e:
            if state["prev"]:
                return {**_restore(state, [str(e)], "updated portfolio"),
                        "pipeline_log": [{"node": "portfolio", "error": str(e), "restored": True}]}
            text = f"I couldn't build a whole-unit portfolio from these answers: {e}."
            err = {"reason": str(e)}
            try:
                dg = portfolio_tools.small_budget_diagnostics(answers)
                err["diagnostics"] = dg
                pricey = ", ".join(f"{a['ticker']} (${a['price_cad']:,.2f})" for a in dg["assets_priced_above_budget"][:6])
                if pricey:
                    text += f"\n\nWith ${dg['budget']:,.0f}, some assets cost more than your whole budget for one unit: {pricey}."
                if dg["smallest_working_budget_on_250_grid"]:
                    text += (f"\n\nWith the same answers, a whole-unit portfolio works from about "
                             f"${dg['smallest_working_budget_on_250_grid']:,} (checked in $250 steps).")
                text += ("\n\nOptions: invest a larger amount, or use a broker that allows fractional units - ask me "
                         "*\"show me a fractional portfolio\"* to see it. You can also click **Start over**.")
            except Exception:
                text += " Click **Start over** to try again."
            return {"phase": "failed", "build_error": err, "released": False,
                    "replies": state["replies"] + [{"content": text}],
                    "pipeline_log": [{"node": "portfolio", "error": str(e)}]}
    feedback = state["check"]["issues"] if state["check"] and not state["check"]["passed"] else None
    draft = portfolio_agent.draft_response(answers, portfolio, feedback=feedback)
    return {"portfolio": portfolio, "draft": draft, "build_error": None,
            "pipeline_log": [{"node": "portfolio", "strategy": portfolio["strategy"], "attempt": state["attempts"],
                              "used_feedback": bool(feedback)}]}


def compliance_node(state: RoboState) -> dict:
    """Checking Agent: verify constraints, budget and that every number in the draft came from the code."""
    check = compliance_agent.check_recommendation(state["prof_state"]["answers"], state["portfolio"], state["draft"],
                                                  clarifications=state["clarifications"])
    return {"check": check, "attempts": state["attempts"] + 1,
            "pipeline_log": [{"node": "compliance", "attempt": state["attempts"], **check}]}


def fallback_node(state: RoboState) -> dict:
    """Retries exhausted: use the deterministic template (built only from code output) and re-check it."""
    draft = portfolio_agent.template_response(state["portfolio"])
    check = compliance_agent.check_recommendation(state["prof_state"]["answers"], state["portfolio"], draft,
                                                  clarifications=state["clarifications"])
    return {"draft": draft, "check": check, "pipeline_log": [{"node": "fallback", **check}]}


def respond_node(state: RoboState) -> dict:
    """Only reached with status APPROVE."""
    return {"phase": "done", "released": True, "prev": None,
            "replies": state["replies"] + [
                {"content": state["draft"], "portfolio": state["portfolio"]},
                {"content": "Feel free to ask questions about this portfolio (for example *\"show me Strategy 1\"*, "
                            "*\"how did it do in 2022?\"*), tell me if something changed, or click **Start over**."},
            ],
            "pipeline_log": [{"node": "respond", "status": state["check"]["status"]}]}


# ------------------------------------------
# Client-facing wording for compliance outcomes (the reviewer's notes are written for the system, not the client)
# ------------------------------------------
_PLAIN_SYSTEM = (
    "You rewrite internal compliance-review notes for a robo-advisor client. Write plain, friendly English, "
    "addressed to the client as 'you'. Never mention internal names (tool, draft, facts, fields, profile_id, "
    "compliance notes, reviewer) and never give instructions to the system. Use only numbers that appear in the "
    "notes; do not add or calculate any. Markdown allowed, no headings.")


def _known_points(state, reasons: list[str]) -> tuple[list[str], list[str]]:
    """Common CLARIFY situations, stated by code. Returns (points, reviewer items no rule covered) so every
    item the reviewer raised reaches the client - either as a code sentence or rewritten by _plain()."""
    p = state.get("portfolio")
    items = [x.strip() for r in reasons for x in r.split("; ") if x.strip()]
    if not p:
        return [], items
    info, f = p["profile_info"], p["facts"]
    rules = []    # (predicate on an item's lower-case text, sentence)
    if any("risk level lowered" in o for o in info.get("overrides", [])):
        why = ("this money is for an emergency fund" if info.get("goal") == "Build an emergency fund"
               else f"you need this money within {info['horizon_range']}" if info.get("horizon_range")
               else "of your time horizon")
        rules.append((lambda t: "risk" in t and any(w in t for w in ("lower", "high", "moderate", "chang", "stated")),
                      f"You chose **{info['stated_risk']}**, but because {why}, the highest risk level I can use is "
                      f"**{info['risk_level'].capitalize()}** (volatility cap {f['volatility_cap']})."))
    if "risk_mapping" in f["disclosures"]:
        eq = f["asset_class_mix"].get("Equity")
        rules.append((lambda t: "very low" in t,
                      f"You chose **Very low risk**; the most conservative setting this model has is **Low** (volatility "
                      f"cap {f['volatility_cap']}), so that is what I used" + (f" - the portfolio still holds {eq} in stocks."
                                                                                 if eq else ".")))
    if f["selected_by_client"]:
        dflt = f["strategies"][f["default_recommendation"]]
        rules.append((lambda t: any(w in t for w in ("strategy 1", "strategy 2", "selected", "default", "chosen strategy")),
                      f"You asked for **{f['recommended_strategy']}**. In the test period it returned "
                      f"{f['recommended']['backtest_total_return']} in total, versus {dflt['backtest_total_return']} for the "
                      f"default **{f['default_recommendation']}**"
                      + (f", and its volatility was above your {f['volatility_cap']} cap." if "realized_volatility_note"
                         in f["disclosures"] else ".")))
    g = f.get("goal_projection")
    if g and not g["reaches_target_at_expected_return"]:
        rules.append((lambda t: any(w in t for w in ("target", "goal")),
                      f"At the model's expected return the illustration is {g['illustrative_value_at_expected_return']}, "
                      f"below your {g['target']} target, so reaching it is not assured."))
    if f.get("exclusion_note"):
        rules.append((lambda t: any(w in t for w in ("look through", "looked through", "etf", "underlying", "indirect",
                                                     "funds", "exclusion")),
                      "Your exclusions remove the individual companies we flag, but broad funds in your portfolio very "
                      "likely still hold some of them (see the note under the recommendation)."))
    points, uncovered = [], []
    for it in items:
        hit = [sent for pred, sent in rules if pred(it.lower())]
        for sent in hit:
            if sent not in points:
                points.append(sent)
        if not hit:
            uncovered.append(it)
    return points, uncovered


def _plain(reasons: list[str], ask: bool, statement: bool = False) -> str:
    """LLM rewrite of reviewer notes for the client; falls back to a fixed sentence if it adds numbers."""
    if statement:
        goal = ("State each point plainly for the client in 1-3 short sentences, as facts they should be aware of. "
                "Do not ask a question.")
    else:
        goal = ("Turn them into 1-3 short sentences that end with ONE yes/no question covering everything, so the "
            "client can confirm it all by answering 'Yes, that's fine' (no either/or choices)."
            if ask else "Explain in 1-3 sentences why the recommendation can't be shown yet.")
    fallback = ("Please also note the points above about your risk level, time horizon and goal." if statement
                else None)
    notes = "\n".join(f"- {r}" for r in reasons)
    try:
        out = llm.generate_text(f"NOTES:\n{notes}\n\n{goal}", system=_PLAIN_SYSTEM, effort=llm.EFFORT_EXTRACT).strip()
        if out and not compliance_agent.check_numbers_grounded(out, {"notes": notes}):
            return out
    except Exception:
        pass
    return fallback or ("Could you confirm that your answers above (risk level, time horizon and any exclusions) are "
                        "what you want?" if ask else "Our suitability check could not confirm this recommendation for "
                        "your answers.")


def withhold_node(state: RoboState) -> dict:
    """CLARIFY -> ask the client (up to MAX_CLARIFY_ROUNDS); BLOCK / unresolved -> withhold the recommendation."""
    check = state["check"]; status = check["status"]
    reasons = [r["message"] for r in check["issue_records"] if r["status"] == status][:4]
    if state["prev"] and not (status == "CLARIFY" and len(state["clarifications"]) < MAX_CLARIFY_ROUNDS):
        return {**_restore(state, reasons, "new recommendation"),
                "pipeline_log": [{"node": "withhold", "status": status, "restored": True}]}
    if status == "CLARIFY" and any(r["code"] == "PRESERVATION" for r in check["issue_records"]):
        # The client asked for full principal protection; a market portfolio cannot promise that, and no
        # answer in this chat changes it - so explain and offer a restart instead of asking again.
        text = ("You told me your original amount must be fully protected. A portfolio of stocks, ETFs and bonds "
                "can lose value, so I can't recommend one that guarantees your principal. Insured savings products "
                "(for example a high-interest savings account or a GIC) are designed for that, but they are outside "
                "what this tool covers.\n\nIf you are comfortable with the value going up and down, tell me "
                "*\"I don't need my principal fully protected\"* or click **Start over**.")
        return {"phase": "failed", "released": False, "replies": state["replies"] + [{"content": text}],
                "pipeline_log": [{"node": "withhold", "status": status, "reason": "principal_protection"}]}
    if status == "CLARIFY" and len(state["clarifications"]) < MAX_CLARIFY_ROUNDS:
        action = (check.get("llm_review") or {}).get("required_action") or ""
        known, rest = _known_points(state, reasons)
        if rest:   # reviewer items no code sentence covers: plain-language statement of each
            known.append(_plain(rest, ask=False, statement=True))
        body = ("\n\n".join(f"- {k}" for k in known) + "\n\nAre you OK to go ahead on this basis?") if known else \
            _plain(reasons + ([action] if action else []), ask=True)
        question = ("Before I show your recommendation, please confirm:\n\n" + body
                    + "\n\nReply *Yes, that's fine* to continue, tell me what to change, or click **Start over**.")
        # the reviewer sees the client's answer next to its own notes on the next check
        record = question + "\n\n[Compliance notes behind this question: " + " | ".join(reasons) + "]"
        return {"phase": "clarify", "pending_clarify": record,
                "replies": state["replies"] + [{"content": question}],
                "pipeline_log": [{"node": "withhold", "status": status, "asked_client": True}]}
    head = ("I couldn't confirm the information needed for this recommendation." if status == "CLARIFY"
            else "I couldn't release this recommendation because it did not pass our suitability review.")
    text = f"{head} {_plain(reasons, ask=False)}\n\nPlease click **Start over** to try again."
    return {"phase": "failed", "released": False, "replies": state["replies"] + [{"content": text}],
            "pipeline_log": [{"node": "withhold", "status": status}]}


def clarify_node(state: RoboState) -> dict:
    """Record the client's answer to the CLARIFY question; the same recommendation is then re-checked."""
    entry = {"question": state["pending_clarify"], "client_answer": state["user_message"]}
    return {"clarifications": state["clarifications"] + [entry], "pending_clarify": None, "phase": "profiling",
            "replies": [], "pipeline_log": [{"node": "clarify_answer", "answer": state["user_message"]}]}


# ------------------------------------------
# After a recommendation
# ------------------------------------------
def switch_strategy_node(state: RoboState) -> dict:
    """Client asks for Strategy 1 / Strategy 2 orders: rebuild with that strategy (and re-check it)."""
    s = state["intent"]["strategy"]
    if s not in ("S1", "S2"):
        s = "S1" if state["strategy"] == "S2" else "S2"
    return {"strategy": s, "portfolio": None, "check": None, "attempts": 0, "clarifications": [],
            "prev": _snapshot(state),
            "replies": [{"content": f"Sure - here is **{portfolio_agent.STRATEGY_NAMES[s]}** for your profile."}],
            "pipeline_log": [{"node": "switch_strategy", "strategy": s}]}


def update_profile_node(state: RoboState) -> dict:
    """Client changes an earlier answer: update the profile and rebuild the portfolio from scratch."""
    prof = copy.deepcopy(state["prof_state"])
    new = profiling_agent.extract_update(f"client: {state['user_message']}")
    changes = {k: v for k, v in (new.model_dump().items() if new else []) if v is not None and prof["answers"].get(k) != v}
    for k in ("target_years", "horizon_years"):      # one timing question feeds both fields
        if k in changes:
            other = "horizon_years" if k == "target_years" else "target_years"
            changes[other] = profiling_agent._coerce(other, changes[k])
    if not changes:
        return {"intent": dict(state["intent"], intent="other_question"),
                "pipeline_log": [{"node": "update_profile", "changes": {}}]}
    prof["answers"].update(changes)
    labels = ", ".join(f"{k.replace('_', ' ')} -> {v}" for k, v in changes.items())
    return {"prof_state": prof, "portfolio": None, "check": None, "attempts": 0, "clarifications": [],
            "prev": _snapshot(state), "released": False, "phase": "profiling",
            "replies": [{"content": f"Got it - I've updated your profile ({labels}) and rebuilt your portfolio with a "
                                    "new optimizer run."}],
            "pipeline_log": [{"node": "update_profile", "changes": changes}]}


def route_after_update(state: RoboState) -> str:
    return "portfolio" if state["pipeline_log"][-1].get("changes") else "followup"


def _metric_cards(wanted: set, p: dict | None, answers: dict, strategy: str) -> dict | None:
    """Small metric cards shown under risk-related answers. Values come straight from code output
    (portfolio facts / stress-test tool), never from the LLM text."""
    if p is None or not wanted & {"stress_test", "emotional", "metric_or_asset_info"}:
        return None
    f, rec = p["facts"], p["facts"]["recommended"]
    items = []
    if wanted & {"stress_test", "emotional"}:
        try:
            st22 = portfolio_tools.stress_test_2022(answers)["portfolios"]
            name = next(k for k in st22 if k.startswith(p["strategy_name"]) and "as recommended" in k)
            items.append(("2022 sell-off: worst fall", f'{100 * st22[name]["max_drawdown"]:.1f}%'))
        except Exception:
            pass
    items += [("Maximum drawdown", rec["backtest_max_drawdown"]), ("Volatility", rec["backtest_volatility"]),
              ("Volatility limit", f["volatility_cap"])]
    return {"title": "Historical risk metrics", "items": items,
            "note": f"Backtest {f['test_window']}. Past performance does not guarantee future results."}


def followup_node(state: RoboState) -> dict:
    """Questions after a recommendation (or after a failed build): call the relevant Portfolio Agent tool,
    then answer grounded in its output."""
    answers, p, intent = state["prof_state"]["answers"], state["portfolio"], state["intent"]
    i = intent["intent"]
    wanted = {i} | router.keyword_tools(state["user_message"])
    data = _base_data()
    if p is not None and state["released"]:
        data["portfolio_facts"] = p["facts"]
    if state["build_error"]:
        data["portfolio_build_error"] = portfolio_tools.for_llm(state["build_error"])
    tools_used = []

    def add(key, fn, *args, decimals=1):
        try:
            data[key] = portfolio_tools.for_llm(fn(*args), decimals); tools_used.append(key)
        except Exception as e:
            data[key] = {"error": f"not available ({e})"}

    if wanted & {"fractional", "small_budget"}:
        add("fractional_portfolio", portfolio_tools.fractional_portfolio, answers, state["strategy"])
    if "small_budget" in wanted or state["build_error"]:
        add("budget_diagnostics", portfolio_tools.small_budget_diagnostics, answers)
    if "esg_cost" in wanted:
        add("esg_cost", portfolio_tools.esg_cost, answers, decimals=2)
    if wanted & {"stress_test", "emotional"}:
        add("stress_test_2022", portfolio_tools.stress_test_2022, answers)
    if "metric_or_asset_info" in wanted:
        add("asset_statistics", portfolio_tools.asset_stats, decimals=2)
    if "compare_strategies" in wanted:
        def strategy_comparison(a):
            out = {}
            for s in ("S1", "S2"):
                q = portfolio_agent.get_portfolio(a, s)
                out[q["strategy_name"]] = {"holdings": q["facts"]["holdings"], "invested": q["facts"]["invested"],
                                           "cash_residual": q["facts"]["cash_residual"], **q["facts"]["recommended"]}
            return out
        add("strategy_comparison", strategy_comparison, answers)
    if intent["not_in_universe"]:
        data["not_in_universe"] = intent["not_in_universe"]
    reply = portfolio_agent.answer_with_data(state["user_message"], data, state["history"])
    out = {"content": reply + f"\n\n_{compliance_agent.DISCLAIMER}_", "tool_data": data}   # tool_data: for evaluation
    cards = _metric_cards(wanted, p if (p is not None and state["released"]) else None, answers, state["strategy"])
    if cards:
        out["metrics"] = cards
    return {"replies": [out],
            "pipeline_log": [{"node": "followup", "intent": i, "tools": tools_used}]}


# ------------------------------------------
# Routing
# ------------------------------------------
def route_start(state: RoboState) -> str:
    return "clarify_answer" if state["phase"] == "clarify" else "router"


def route_after_currency(state: RoboState) -> str:
    return "portfolio" if state["pipeline_log"][-1].get("complete") else END


def route_after_portfolio(state: RoboState) -> str:
    return END if state["phase"] == "failed" else "compliance"


def route_after_compliance(state: RoboState) -> str:
    status = state["check"]["status"]
    if status == "APPROVE":
        return "respond"
    if status == "REVISE":   # wording only -> Portfolio Agent rewrites, then re-check
        return "portfolio" if state["attempts"] <= MAX_RETRIES else "fallback"
    return "withhold"        # CLARIFY / BLOCK: never auto-relax constraints


def route_after_fallback(state: RoboState) -> str:
    return "respond" if state["check"]["status"] == "APPROVE" else "withhold"


def build_graph():
    g = StateGraph(RoboState)
    for name, fn in [("router", router_node), ("profiling", profiling_node), ("assist", assist_node),
                     ("currency", currency_node), ("portfolio", portfolio_node), ("compliance", compliance_node),
                     ("fallback", fallback_node), ("respond", respond_node), ("withhold", withhold_node),
                     ("clarify_answer", clarify_node), ("switch_strategy", switch_strategy_node),
                     ("update_profile", update_profile_node), ("followup", followup_node)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route_start, ["router", "clarify_answer"])
    g.add_conditional_edges("router", route_after_router,
                            ["profiling", "assist", "currency", "switch_strategy", "update_profile", "followup"])
    g.add_conditional_edges("profiling", route_after_profiling, ["portfolio", END])
    g.add_conditional_edges("currency", route_after_currency, ["portfolio", END])
    g.add_edge("assist", END)
    g.add_edge("switch_strategy", "portfolio")
    g.add_conditional_edges("update_profile", route_after_update, ["portfolio", "followup"])
    g.add_edge("clarify_answer", "compliance")
    g.add_conditional_edges("portfolio", route_after_portfolio, ["compliance", END])
    g.add_conditional_edges("compliance", route_after_compliance, ["respond", "portfolio", "fallback", "withhold"])
    g.add_conditional_edges("fallback", route_after_fallback, ["respond", "withhold"])
    g.add_edge("withhold", END)
    g.add_edge("respond", END)
    g.add_edge("followup", END)
    return g.compile()


GRAPH = build_graph()


def run_turn(state: RoboState, user_message: Optional[str]) -> RoboState:
    """Run one client turn through the graph. Returns the new state; state["replies"] holds this
    turn's assistant messages. Raises on LLM errors (e.g. quota) so the caller can keep the old state."""
    turn_input = {**copy.deepcopy(DEFAULTS), **state, "user_message": user_message, "replies": [], "attempts": 0,
                  "check": None if state["phase"] == "profiling" else state["check"], "pipeline_log": []}
    out = GRAPH.invoke(turn_input)
    out["pipeline_log"] = state["pipeline_log"] + out["pipeline_log"]
    out["history"] = (state["history"] + ([("client", user_message)] if user_message is not None else [])
                      + [("bot", r["content"]) for r in out["replies"]])
    return out


if __name__ == "__main__":
    # print the graph (paste into https://mermaid.live for the report), then chat in the terminal
    print(GRAPH.get_graph().draw_mermaid())
    s = run_turn(new_state(), None)
    print("Bot:", s["replies"][-1]["content"])
    while True:
        s = run_turn(s, input("You: "))
        for r in s["replies"]:
            print("Bot:", r["content"] if "portfolio" not in r else portfolio_agent.template_response(r["portfolio"]))
