# ------------------------------------------
# PART 5 - RUN THE 20 SCRIPTED CONVERSATIONS THROUGH THE CHATBOT
# ------------------------------------------
# Usage (from the repo root, ANTHROPIC_API_KEY set):
#   python -m evaluation.run_scripts                 # all 20 scripts, 4 in parallel
#   python -m evaluation.run_scripts 01 03 18        # selected scripts
# Output: evaluation/results/chatbot/<id>.json (full transcript, extracted profile, agent trace, per-reply
# number check) and evaluation/results/summary_chatbot.{json,md}.
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
import profiling_agent  # noqa: E402
import robo_graph  # noqa: E402
from evaluation import metrics  # noqa: E402
from evaluation.scripts import SCRIPTS  # noqa: E402

OUT = os.path.join(metrics.HERE, "results")
MAX_TURNS = 45
CONFIRM = "Yes, that's fine."


def _pending_field(state) -> str | None:
    msgs = state["prof_state"]["messages"]
    if not msgs or msgs[-1][0] != "bot":
        return None
    return next((k for k, q in profiling_agent.QUESTION_TEXT.items() if q == msgs[-1][1]), None)


def _sources(state, reply: dict, user_msgs: list[str]) -> list:
    """Everything code produced that this reply may quote, plus what the client wrote."""
    p = state.get("portfolio")
    port = {k: v for k, v in p.items() if k != "backtest_series"} if p else None
    return [port, reply.get("tool_data"), reply.get("metrics"), state.get("fx_pending"), state.get("build_error"),
            state["pipeline_log"][-12:], user_msgs, list(profiling_agent.QUESTION_TEXT.values()),
            robo_graph._base_data()]


def run_script(sc: dict) -> dict:
    t0 = time.time()
    state = robo_graph.run_turn(robo_graph.new_state(), None)
    turns, user_msgs, filled = [], [], []

    def record(user, st):
        replies = []
        for r in st["replies"]:
            inv = metrics.invented_numbers(r["content"], _sources(st, r, user_msgs))
            replies.append({"content": r["content"], "has_portfolio": "portfolio" in r,
                            "metrics_cards": r.get("metrics"), "numbers": len(metrics.numbers_in(r["content"])),
                            "invented": inv})
        turns.append({"user": user, "phase_after": st["phase"], "replies": replies})

    record(None, state)

    def send(msg):
        nonlocal state
        user_msgs.append(msg)
        state = robo_graph.run_turn(state, msg)
        record(msg, state)

    for m in sc["opening"]:
        send(m)
    n = 0
    while sc.get("complete", True) and state["phase"] in ("profiling", "clarify") and n < MAX_TURNS:
        n += 1
        if state["phase"] == "clarify" or state.get("fx_pending"):
            send(CONFIRM); continue
        f = _pending_field(state)
        if f and f in sc["persona"]:
            filled.append(f); send(sc["persona"][f])
        else:   # bot asked something else (contradiction check, clarification): confirm
            send(CONFIRM)
    for m in sc["after"]:
        if state["phase"] == "clarify":
            send(CONFIRM)
        send(m)
    if state["phase"] == "clarify":
        send(CONFIRM)

    replies = [r for t in turns for r in t["replies"]]
    out = {
        "id": sc["id"], "title": sc["title"], "version": "chatbot", "settings": llm.SETTINGS,
        "seconds": round(time.time() - t0, 1), "final_phase": state["phase"],
        "fields_filled_by_persona_card": filled, "answers": state["prof_state"]["answers"],
        "released_portfolio": state["released"], "strategy": state["strategy"],
        "turns": turns, "pipeline_log": state["pipeline_log"],
        "replies_total": len(replies),
        "replies_with_numbers": sum(1 for r in replies if r["numbers"]),
        "replies_with_invented": sum(1 for r in replies if r["invented"]),
        "numbers_total": sum(r["numbers"] for r in replies),
        "numbers_invented": sum(len(r["invented"]) for r in replies),
    }
    if sc.get("truth"):
        out["field_accuracy"] = metrics.field_accuracy(state["prof_state"]["answers"], sc["truth"])
    if sc.get("expect"):
        out["holdings"] = metrics.holdings_match(state["portfolio"] if state["released"] else None, sc["expect"])
    return out


def _safe(sc):
    try:
        res = run_script(sc)
    except Exception as e:
        res = {"id": sc["id"], "title": sc["title"], "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()}
    os.makedirs(os.path.join(OUT, "chatbot"), exist_ok=True)
    with open(os.path.join(OUT, "chatbot", f"{sc['id']}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"[{sc['id']}] {res.get('error') or res['final_phase']} ({res.get('seconds', '-')}s)", flush=True)
    return res


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    fa = [v for r in ok if "field_accuracy" in r for v in r["field_accuracy"].values()]
    per_field = {}
    for r in ok:
        for f, v in r.get("field_accuracy", {}).items():
            per_field.setdefault(f, []).append(v)
    hold = [r["holdings"] for r in ok if "holdings" in r]
    s = {
        "version": "chatbot", "model": llm.MODEL, "run_at": datetime.now().isoformat(timespec="seconds"),
        "scripts_run": len(results), "scripts_failed": [r["id"] for r in results if "error" in r],
        "field_accuracy": {"correct": sum(fa), "total": len(fa), "rate": round(sum(fa) / len(fa), 4) if fa else None,
                           "per_field": {f: f"{sum(v)}/{len(v)}" for f, v in per_field.items()}},
        "invented_numbers": {
            "replies_total": sum(r["replies_total"] for r in ok),
            "replies_with_numbers": sum(r["replies_with_numbers"] for r in ok),
            "replies_with_invented": sum(r["replies_with_invented"] for r in ok),
            "numbers_total": sum(r["numbers_total"] for r in ok),
            "numbers_invented": sum(r["numbers_invented"] for r in ok)},
        "holdings_vs_part3": {"scripts": len(hold), "shown": sum(h["shown"] for h in hold),
                              "exact": sum(h["exact"] for h in hold),
                              "positions_matching": f"{sum(h['matching_positions'] for h in hold)}/"
                                                    f"{sum(h['expected_positions'] for h in hold)}"},
        "per_script": [{k: r.get(k) for k in ("id", "title", "final_phase", "released_portfolio", "strategy",
                                              "replies_with_numbers", "replies_with_invented", "numbers_invented")}
                       | ({"field_accuracy": f"{sum(r['field_accuracy'].values())}/{len(r['field_accuracy'])}"}
                          if "field_accuracy" in r else {})
                       | ({"holdings_exact": r["holdings"]["exact"]} if "holdings" in r else {})
                       for r in ok],
    }
    inv = s["invented_numbers"]
    inv["reply_rate"] = round(inv["replies_with_invented"] / inv["replies_with_numbers"], 4) if inv["replies_with_numbers"] else None
    inv["number_rate"] = round(inv["numbers_invented"] / inv["numbers_total"], 4) if inv["numbers_total"] else None
    return s


def to_markdown(s: dict) -> str:
    fa, inv, h = s["field_accuracy"], s["invented_numbers"], s["holdings_vs_part3"]
    lines = [f"# Chatbot evaluation - {s['scripts_run']} scripted conversations ({s['model']}, {s['run_at']})", "",
             "| Metric | Result |", "|---|---|",
             f"| Profile fields captured correctly (scripts 01-09) | {fa['correct']}/{fa['total']} ({100 * (fa['rate'] or 0):.1f}%) |",
             f"| Replies containing a number not produced by code | {inv['replies_with_invented']}/{inv['replies_with_numbers']} replies with numbers ({100 * (inv['reply_rate'] or 0):.1f}%) |",
             f"| Numbers not produced by code | {inv['numbers_invented']}/{inv['numbers_total']} ({100 * (inv['number_rate'] or 0):.2f}%) |",
             f"| Final orders identical to Part 3 file | {h['exact']}/{h['scripts']} scripts (positions {h['positions_matching']}) |",
             "", "| Script | Title | End state | Field accuracy | Orders = Part 3 | Replies w/ invented numbers |",
             "|---|---|---|---|---|---|"]
    for r in s["per_script"]:
        lines.append(f"| {r['id']} | {r['title']} | {r['final_phase']}{' (shown)' if r['released_portfolio'] else ''} | "
                     f"{r.get('field_accuracy', '-')} | {r.get('holdings_exact', '-')} | {r['replies_with_invented']} |")
    lines += ["", "Per field: " + ", ".join(f"{f} {v}" for f, v in fa["per_field"].items())]
    return "\n".join(lines)


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    todo = [s for s in SCRIPTS if not wanted or s["id"] in wanted]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_safe, todo))
    if not wanted:
        summary = summarize(results)
        with open(os.path.join(OUT, "summary_chatbot.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        md = to_markdown(summary)
        with open(os.path.join(OUT, "summary_chatbot.md"), "w") as fh:
            fh.write(md)
        print(md)
