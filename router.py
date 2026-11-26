# ------------------------------------------
# CONVERSATION ROUTER
# ------------------------------------------
# Classifies each client message so the graph can send it to the right agent / tool:
# questionnaire answers -> Profiling Agent; strategy switches, profile changes and data questions
# -> Portfolio Agent tools; out-of-scope / emotional messages -> scoped answer (and the questionnaire
# continues). Currency mentions are detected deterministically so a foreign amount is never
# silently treated as CAD.
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

import llm

INTENTS = Literal[
    "answer",                    # answers / details for the questionnaire (incl. greetings like "I'd like to invest")
    "change_profile",            # client changes an earlier answer (risk, amount, horizon, exclusions, ...)
    "show_strategy",             # wants to see / switch to Strategy 1 or Strategy 2 (units, portfolio)
    "compare_strategies",        # side-by-side comparison, which strategy to choose
    "fractional",                # fractional shares / partial units
    "esg_cost",                  # cost of exclusions / ESG screen
    "stress_test",               # crisis / crash / 2022 / COVID / "how bad could it get" / compare with cash
    "small_budget",              # what can I buy with a small amount / why no portfolio
    "metric_or_asset_info",      # explain Sharpe, volatility, drawdown, an asset, the universe
    "stock_tip",                 # buy/sell a single stock, price predictions
    "tax_or_legal",              # TFSA/RRSP/tax/legal advice
    "emotional",                 # panic, fear, wants to sell everything because of the market
    "unrealistic",               # guaranteed / risk-free high returns
    "disclaimer",                # asks about the disclaimer / whether it is financial advice / is it a good investment
    "other_question",            # any other question about the portfolio or investing
]


class TurnIntent(BaseModel):
    intent: INTENTS
    strategy: Literal["S1", "S2", "none"]     # for show_strategy: S1 = Markowitz sample means, S2 = reverse optimisation
    mentioned_companies: list[str]            # company names or tickers the client wants to buy / asks about


ROUTER_PROMPT = """Classify the client's latest message to a robo-advisor chatbot.
Context: the chatbot first runs a short questionnaire, then recommends a portfolio built from 24 assets.
Strategy 1 = Markowitz with sample (historical) mean returns - higher expected return, concentrated in few assets;
Strategy 2 = reverse optimisation (implied returns) - the default recommendation, more diversified, more conservative.
Choose ONE intent. Use "answer" when the message mainly provides personal/financial details or answers the bot's
question (even if phrased casually). Use "change_profile" only after a portfolio was shown and the client changes
earlier details (risk level, amount, horizon, exclusions, new assets elsewhere). "unrealistic" = asks for guaranteed or
risk-free high returns. "emotional" = fear/panic, wants to sell because of market moves.
strategy: "S1"/"S2" only if the client names or clearly describes one, else "none".
mentioned_companies: companies/tickers the client mentions wanting to buy or asks about (empty list if none).
Examples: "how much return am I giving up by excluding fossil fuels" -> esg_cost; "show me both strategies side by
side" / "which one do you recommend" -> compare_strategies; "show me Strategy 1" / "I prefer the more diversified
one" -> show_strategy; "how bad would it have been in the COVID crash" / "compare to holding cash" -> stress_test;
"what can I actually buy" (small budget) -> small_budget; "what is a Sharpe ratio" -> metric_or_asset_info;
"change it to high risk" / "I have another $50,000 pension" -> change_profile (after a portfolio was shown)."""
KEYWORD_TOOLS = [  # deterministic safety net: these words always attach the matching tool output
    ("esg_cost", r"\besg\b|exclu|screen|giving up|fossil|ethical"),
    ("fractional", r"fraction|partial (unit|share)"),
    ("stress_test", r"\b2022\b|covid|crash|stress|sell-?off|drop|downturn|recession|holding cash|just cash"),
    ("metric_or_asset_info", r"sharpe|volatil|drawdown|\bmetric"),
    ("compare_strategies", r"strateg(y|ies) (1|2|one|two)|both strategies|side by side|compare.*strateg|which (one|strategy)"),
    ("small_budget", r"what can i (actually )?buy|too small|minimum (amount|budget)"),
]


def keyword_tools(text: str) -> set[str]:
    t = text.lower()
    return {name for name, pat in KEYWORD_TOOLS if re.search(pat, t)}

_CURRENCY = [("GBP", r"£|\bgbp\b|\bpounds?\b|\bsterling\b"), ("EUR", r"€|\beur\b|\beuros?\b"),
             ("USD", r"\busd\b|\bus\$|\bus dollars?\b|\bamerican dollars?\b"), ("JPY", r"¥|\bjpy\b|\byen\b"),
             ("INR", r"₹|\binr\b|\brupees?\b"), ("KRW", r"₩|\bkrw\b|\bwon\b"), ("AUD", r"\baud\b|\baustralian dollars?\b"),
             ("CHF", r"\bchf\b|\bswiss francs?\b"), ("CNY", r"\bcny\b|\byuan\b|\brmb\b"), ("MXN", r"\bmxn\b|\bpesos?\b"),
             ("HKD", r"\bhkd\b")]


def detect_currency(text: str) -> tuple[str | None, float | None]:
    """Non-CAD currency mentioned with an amount, e.g. '£20,000' -> ('GBP', 20000.0)."""
    t = text.lower()
    for code, pat in _CURRENCY:
        cur = re.search(pat, t)
        if cur:
            # the amount is the number written next to the currency marker ("I'm 61 ... US$50,000" -> 50,000),
            # not simply the first number in the message
            nums = [(m, float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1))
                    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand)?", t)]
            if not nums:
                return code, None
            def gap(m):
                return max(cur.start() - m.end(), m.start() - cur.end(), 0)
            best = min(nums, key=lambda x: (gap(x[0]), -x[1]))
            return code, best[1]
    return None, None


def classify(message: str, phase: str, last_bot: str = "") -> TurnIntent:
    out = llm.generate_structured(
        f"Chat phase: {phase} ({'questionnaire in progress' if phase == 'profiling' else 'portfolio already shown'})\n"
        f"Bot's last message: {last_bot[:400]}\nClient's message: {message}",
        TurnIntent, system=ROUTER_PROMPT)
    return out or TurnIntent(intent="answer" if phase == "profiling" else "other_question", strategy="none",
                             mentioned_companies=[])
