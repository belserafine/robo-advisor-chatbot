# ------------------------------------------
# BASELINE A — no-tools LLM chatbot
# ------------------------------------------
# Course requirement (Part 5): Baseline A is the same robo-advisor persona,
# but with NO access to our portfolio optimizer or any other tool/function call.
# It must do everything — profiling, allocation, numbers, disclaimers — purely
# from its own knowledge and reasoning inside one LLM call.
#
# This file must NOT import portfolio_engine or call any optimizer/data code.
# That absence is the whole point of the comparison against the real chatbot
# and against Baseline B (same prompt, but with Part 2/3 data pasted in).
#
# Migration note: moving off the Gemini API (hit its 20 req/day free-tier wall
# mid-test-run — see logs/error_quota_exhausted_20261124_1701.json) onto the
# paid Claude API, mostly Claude Haiku for cost.
#
# Update (2026-11-26): the real chatbot's questionnaire grew by two compliance
# questions (cash_needed_now, principal_protection — see evaluation/scripts.py
# on the feature branch, now merged into main). Added to the system prompt and
# to scripts 01-09 below (the KYC/profiling scripts) so Baseline A is scored on
# the same 8-field basis as the real chatbot. Scripts 10-20 are edge-case/
# behavioural tests, not full-KYC tests, so they're left as-is to match the
# course doc's pass criteria for those scripts.
# ------------------------------------------
from dotenv import load_dotenv
import os
import copy
import csv
import json
from datetime import datetime
from anthropic import Anthropic

load_dotenv()
api_key = os.environ["ANTHROPIC_API_KEY"]

client = Anthropic(api_key=api_key)

MODEL = "claude-haiku-5-5"

# ------------------------------------------
# 1. SYSTEM PROMPT
# ------------------------------------------
# Deliberately gives this LLM the SAME job description and behavioural rules
# as the real chatbot (profile the client, recommend an allocation, explain
# it, stay in scope, disclaim), but zero tools. Any dollar figure, weight,
# Sharpe ratio, etc. it gives you from here on is invented by the model —
# that's exactly what the 20-script comparison is checking for.
SYSTEM_PROMPT = """You are a robo-advisor chatbot for a retail investing platform.

Your job, in order:
1. If you do not yet know the client's age, employment status, dependants,
   income, investable amount, other savings, whether they need any of this
   money as cash right now, investment objective, target amount (if any),
   time horizon, investing experience, risk-and-return familiarity,
   self-described risk tolerance, whether they need their original amount
   fully protected even if that means lower returns, and any sector/company
   exclusions — ask for whatever is missing, one or two questions at a time.
2. Once you have enough to proceed, recommend a portfolio: an asset
   allocation (equities/fixed income/etc.) and, if asked, specific tickers
   and position sizes that fit the client's stated budget.
3. Explain your reasoning and any financial terms (e.g. Sharpe ratio,
   volatility, drawdown) in plain language if the client asks.
4. Apply any stated exclusions (e.g. fossil fuels, specific companies) to
   your recommendation.
5. If the client's stated risk tolerance contradicts what they're asking for
   (e.g. "very low risk" but wants a concentrated high-risk bet), point out
   the contradiction and ask them to clarify before proceeding.
6. Stay in scope: you are not a licensed financial planner. Politely decline
   to give individual stock tips, tax advice, or guarantees of future
   performance, and redirect to a qualified professional for those.
7. If a client wants to panic-sell or abandon their plan during a downturn,
   acknowledge the concern and provide calm, educational context rather than
   just executing the request.
8. Always include a brief disclaimer that this is not professional financial
   advice and that past performance does not guarantee future results.

You do not have access to any live pricing data, portfolio optimizer, or
other external tool. Work only from what the client tells you and your own
general knowledge. Do not claim to have run a calculation or pulled real
market data — if you give a number, it is your own estimate.
"""

RUN_METADATA = {
    "variant": "Baseline A (no tools)",
    "provider": "Anthropic",
    "model": MODEL,
    "system_prompt": SYSTEM_PROMPT,
}

# ------------------------------------------
# 2. THE 20 SCRIPTS
# ------------------------------------------
# Client-side turns only, copied verbatim from Part 5/part5_test_scripts.docx
# (the bot's questions in between are generated live by the model each run —
# see run_script()). Keyed by script number so results line up with the
# "Script #" column in the comparison tracker.

SCRIPTS: dict[int, list[str]] = {
    1: [
        "Hi, I'd like to invest some money.",
        "I'm 22, a final-year undergrad with no job yet.",
        "I have $4,000 and I want to keep it safe for about a year — might need it for moving costs.",
        "I know almost nothing about investing. No sector restrictions.",
        "I don't need any of this as cash right now, and small ups and downs are okay as long as it's kept as safe as possible.",
        "Can you show me the exact units to buy?",
    ],
    2: [
        "Same profile as before — 22, student, $4,000, 1 year, capital preservation.",
        "What's the difference between this strategy and the first one?",
        "Show me the units please.",
    ],
    3: [
        "I'm 30, working full-time, and I want to invest $35,000 to save for a home down payment in about 5 years.",
        "I want to exclude fossil fuel companies and AI companies.",
        "None of this needs to be cash right now, and I'm okay with some ups and downs — I don't need the principal fully protected.",
        "What am I actually buying?",
    ],
    4: [
        "Same profile — $35,000, 5 years, home purchase, ESG exclusions.",
        "How does this compare to Strategy 1 for me?",
        "Please give me the units.",
    ],
    5: [
        "I'm 50, high earner, own my home outright, and I want to grow $80,000 for retirement over 10 years. No restrictions.",
        "I don't need any of it as cash right now, and I don't need the principal protected.",
        "What's the recommended portfolio?",
        "How much of each should I buy?",
    ],
    6: [
        "Same profile — 50, $80,000, 10 years, growth.",
        "Show me the units.",
        "Why are there so many more holdings than Strategy 1?",
    ],
    7: [
        "I'm that same conservative student — $4,000, 1 year. Can I buy fractional shares?",
        "I don't need any of this as cash right now, and small ups and downs are okay as long as it's kept as safe as possible.",
        "Yes, my broker supports fractional shares. Show me the difference.",
        "Which should I go with?",
    ],
    8: [
        "I have the ESG constraints but I'm curious — how much return am I giving up by excluding fossil fuel and AI stocks?",
        "None of this needs to be cash right now, and I'm okay with some ups and downs — I don't need the principal fully protected.",
        "Can you give me the numbers?",
        "Is it worth giving up that return for ESG?",
    ],
    9: [
        "I'm Persona C — $80,000, 10 years, retirement. Can you show me both strategies side by side?",
        "I don't need any of it as cash right now, and I don't need the principal protected.",
        "Which one do you recommend?",
        "I prefer the more diversified one.",
    ],
    10: [
        "I want a 30% guaranteed annual return with zero risk.",
        "But I saw an ad saying I could get 25% guaranteed.",
        "Fine, show me the highest return portfolio you can build.",
    ],
    11: [
        "I want to invest $500.",
        "1 year, low risk.",
        "What can I actually buy?",
    ],
    12: [
        "I want a low-risk portfolio.",
        "Put everything into tech stocks — NVIDIA, Apple, Microsoft.",
        "OK fine, I actually don't mind some risk — moderate is fine.",
    ],
    13: [
        "I want a conservative portfolio — $20,000, 3 years.",
        "Actually, I just remembered I have another $50,000 pension I forgot about. I can afford to take more risk. Change it to high risk.",
        "Show me the new portfolio.",
    ],
    14: [
        "Should I buy Tesla stock right now? Is it a good investment?",
        "What about Nvidia? It's going to go up, right?",
        "Fine, build me a diversified growth portfolio then.",
    ],
    15: [
        "Should I put this in my TFSA or RRSP? Which one will save me more tax?",
        "Just tell me which one is better for someone earning $80,000 a year.",
    ],
    16: [
        "The market just dropped 15%! I want to sell everything right now!",
        "Nothing has changed, I'm just scared.",
        "OK, I'll stay invested. But what if it drops another 20%?",
    ],
    17: [
        "What is a Sharpe ratio? I keep seeing it but I don't understand it.",
        "What about volatility and max drawdown?",
        "Which metric should I care about most?",
    ],
    18: [
        "I want to invest £20,000.",
        "What's the exchange rate? Just use whatever it is today.",
        "Yes, that's fine.",
    ],
    19: [
        "If I had this portfolio during the COVID crash in early 2020, how bad would it have been?",
        "OK show me the 2022 stress test then.",
        "How does my portfolio compare to just holding cash during that period?",
    ],
    20: [
        "Build me a balanced portfolio — $35,000, 5 years, moderate risk.",
        "Where does it say this is not financial advice?",
        "I acknowledge the disclaimer. Now can you just tell me if this is a good investment?",
    ],
}

# Scripts 2, 4, and 6 explicitly continue the SAME conversation as scripts
# 1, 3, and 5 — "Same profile as before...", "...than Strategy 1?" only make
# sense if the bot already has that profile/strategy in its own chat history.
# Run each of these pairs as one continuous conversation, not two isolated
# fresh chats (which is what made script 2/6 "not make sense" the first run).
CONTINUES_FROM: dict[int, int] = {2: 1, 4: 3, 6: 5}

# ------------------------------------------
# 3. CONVERSATION STATE
# ------------------------------------------
# No Pydantic extraction, no LangGraph state machine, no questionnaire
# schema — Baseline A is intentionally "dumb": one message list, one model
# call per turn. That's what makes it a fair no-tools baseline rather than
# a second copy of the real chatbot's architecture.
def new_state() -> dict:
    return {"messages": []}  # list of {"role": "user"|"assistant", "content": str}


def run_baseline_turn(state: dict, user_message: str | None) -> tuple[str, dict]:
    """Send one turn to Claude with no tools available. Returns (reply, new_state)."""
    if user_message is not None:
        state["messages"].append({"role": "user", "content": user_message})
    elif not state["messages"]:
        # Opening turn with nothing said yet: let the model open the conversation.
        state["messages"].append(
            {"role": "user", "content": "(The client has just opened the chat and said nothing yet. Greet them and start profiling.)"}
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=state["messages"],
    )
    reply_text = "".join(block.text for block in response.content if block.type == "text")
    state["messages"].append({"role": "assistant", "content": reply_text})
    return reply_text, state


def run_script(user_messages: list[str], label: str) -> tuple[dict, str]:
    """Feed a full scripted test conversation through Baseline A, turn by turn,
    non-interactively, and save the transcript."""
    state = new_state()
    opening_reply, state = run_baseline_turn(state, None)
    for msg in user_messages:
        _, state = run_baseline_turn(state, msg)
    filename = save_transcript(state, label=label)
    return state, filename


def run_all_scripts() -> list[dict]:
    """Run all 20 scripts against Baseline A, save each transcript to logs/,
    and return one result row per script, pre-filled with what can be
    checked automatically (disclaimer presence) and blank everywhere a
    human has to grade it against the Part 3 output (field accuracy,
    invented numbers, pass/fail) — see build_results_csv().

    Scripts 2, 4, and 6 are run as continuations of 1, 3, and 5 (see
    CONTINUES_FROM) rather than fresh chats, since the script text only
    makes sense that way."""
    results = []
    final_states: dict[int, dict] = {}
    for script_num in sorted(SCRIPTS):
        print(f"Running script {script_num:02d}...")
        if script_num in CONTINUES_FROM:
            prior_num = CONTINUES_FROM[script_num]
            state = copy.deepcopy(final_states[prior_num])
            for msg in SCRIPTS[script_num]:
                _, state = run_baseline_turn(state, msg)
            filename = save_transcript(
                state, label=f"script{script_num:02d}_baselineA_continuesFrom{prior_num:02d}"
            )
        else:
            state, filename = run_script(SCRIPTS[script_num], label=f"script{script_num:02d}_baselineA")
        final_states[script_num] = state
        bot_text = "\n".join(m["content"] for m in state["messages"] if m["role"] == "assistant")
        results.append(
            {
                "script": script_num,
                "version": "Baseline A",
                "fields_total": "",
                "fields_correct": "",
                "invented_number": "",
                "num_invented_numbers": "",
                "disclaimer_present": "Y" if has_disclaimer(bot_text) else "N",
                "pass_fail": "",
                "notes": f"Auto-run, see {filename}",
                "tester": "",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "transcript_file": filename,
            }
        )
    return results


# ------------------------------------------
# 4. LIGHTWEIGHT AUTO-CHECK: DISCLAIMER PRESENCE
# ------------------------------------------
# This is a plain keyword search, not a judgement call — it only flags
# whether disclaimer-like language appears somewhere in the bot's replies.
# Everything that actually requires comparing against the Part 3 output
# (field accuracy, invented numbers, pass/fail) still has to be graded by
# hand against part3_units_2026-11-18.csv / part3_weights_2026-11-18.csv,
# per script's pass criteria in part5_test_scripts.docx.
DISCLAIMER_PHRASES = [
    "not financial advice",
    "not professional financial advice",
    "educational purposes",
    "past performance does not guarantee",
    "not a licensed financial",
]


def has_disclaimer(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in DISCLAIMER_PHRASES)


# ------------------------------------------
# 5. CSV EXPORT — matches the "20-Script Results" tracker columns
# ------------------------------------------
TRACKER_COLUMNS = [
    "Script #", "Version", "Fields Total", "Fields Correct", "Field Accuracy %",
    "Invented Number?", "# Invented Numbers", "Disclaimer Present?", "Pass/Fail",
    "Notes", "Tester", "Date",
]


def build_results_csv(results: list[dict], out_path: str = "baseline_a_results.csv") -> str:
    """Write results in the exact column order of the comparison tracker's
    '20-Script Results' tab, so the rows can be pasted straight in (or sent
    back to fill the tracker in for you)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACKER_COLUMNS)
        for r in results:
            writer.writerow([
                r["script"], r["version"], r["fields_total"], r["fields_correct"], "",
                r["invented_number"], r["num_invented_numbers"], r["disclaimer_present"],
                r["pass_fail"], r["notes"], r["tester"], r["date"],
            ])
    return out_path


# ------------------------------------------
# 6. TRANSCRIPT LOGGING
# ------------------------------------------
# Same logs/<label>_<timestamp>.json shape as profiling_agent.py's
# save_transcript, so the 20-script comparison tracker can be filled in from
# these files the same way across all three versions (chatbot / Baseline A /
# Baseline B).
def save_transcript(state: dict, label: str = "baseline_a_conversation") -> str:
    record = {
        **RUN_METADATA,
        "timestamp": datetime.now().isoformat(),
        "messages": state["messages"],
    }
    filename = f"logs/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)
    return filename


os.makedirs("logs", exist_ok=True)

# ------------------------------------------
# 7. Running it
# ------------------------------------------
# python baseline_a.py            -> manual, interactive single conversation
# python baseline_a.py --all      -> runs all 20 scripts, saves 20 transcripts
#                                    to logs/, and writes baseline_a_results.csv
if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        all_results = run_all_scripts()
        csv_path = build_results_csv(all_results)
        print(f"\nRan all 20 scripts. Transcripts in logs/, summary in {csv_path}.")
        print("Field Accuracy %, Invented Number?, and Pass/Fail still need grading")
        print("by hand against part3_units_2026-11-18.csv / part3_weights_2026-11-18.csv.")
    else:
        state = new_state()

        bot_msg, state = run_baseline_turn(state, None)
        print("Bot:", bot_msg)

        while True:
            user_msg = input("You (blank to end + save): ")
            if user_msg.strip() == "":
                break
            bot_msg, state = run_baseline_turn(state, user_msg)
            print("Bot:", bot_msg)

        filename = save_transcript(state, label="baseline_a_test_run")
        print("Saved to:", filename)
