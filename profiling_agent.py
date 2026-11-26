# ------------------------------------------
# 0. IMPORTS & CONFIGURATION
# ------------------------------------------
import os
from pydantic import BaseModel
from typing import Literal, Optional, TypedDict, Annotated, get_args, get_origin
from langgraph.graph.message import add_messages
import json
from datetime import datetime

import llm  # Claude API wrapper (llm.py) - provider/model are configured there

MODEL = llm.MODEL

# ------------------------------------------
# 1. QUESTIONNAIRE ANSWERS
# ------------------------------------------
class questionnaire_answers(BaseModel):
    ### SECTION 1: PERSONAL CIRCUMSTANCES
    # What is your age?
    age: Optional[Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]] = None

    # What is your current employment status?
    employment: Optional[Literal["student", "employed", "self-employed", "retired"]] = None

    # Do you have any dependants? How many?
    num_dependants: Optional[int] = None

    ### SECTION 2: FINANCIAL CIRCUMSTANCES
    # What is your approximate annual income?
    income: Optional[Literal["Under $20,000",
                    "$20,000 – $39,999",
                    "$40,000 – $59,999",
                    "$60,000 – $79,999",
                    "$80,000 – $99,999",
                    "$100,000 – $149,999",
                    "$150,000+"]] = None

    # What is the amount you’re looking to invest today?
    invest_amount: Optional[int] = None

    # Do you have other savings or assets outside this investment (e.g., emergency fund)?
    other_assets: Optional[str] = None

    # Will you need any of this money as cash right now? If so, how much? (Part 1 liquidity question)
    cash_needed_now: Optional[int] = None

    ### SECTION 3: INVESTMENT NEEDS & OBJECTIVES
    # What are you investing for?
    objective_str: Optional[Literal["Save for retirement",
                            "General savings",
                            "Buy a house",
                            "Save for education",
                            "Build an emergency fund"]] = None

    # Do you have a specific target amount this goal? If yes, how much?
    target_yn: Optional[Literal["Yes", "No"]] = None
    target: Optional[int] = None

    # When will you need this money?
    target_years: Optional[Literal["10+ years",
                        "4-10 years",
                        "1-3 years",
                        "0-1 years"]] = None

    ### SECTION 4: INVESTMENT KNOWLEDGE
    # How would you describe your investing experience?
    exp: Optional[Literal["none", "some", "experienced"]] = None

    # Are you familiar with how risk and return are related in investing?
    rr_familiarity: Optional[Literal["Yes", "No", "Unsure"]] = None

    ### SECTION 5: RISK PROFILE
    # Which statement best describes you?
    desc: Optional[Literal["Very low risk", "Low risk", "Moderate risk", "High risk"]] = None

    # Do you need your original amount fully protected, even if that means lower returns?
    principal_protection: Optional[Literal["Yes", "No"]] = None

    ### SECTION 6: INVESTMENT TIME HORIZON
    horizon_years: Optional[Literal["10+ years", "4–10 years", "1–3 years", "0–1 years"]] = None

    ### SECTION 7: PREFERENCES
    # Are there any sectors, industries, or companies you’d like to exclude? (e.g., fossil fuels, specific themes)
    sectors: Optional[str] = None

# --- Extraction schema sent to the LLM -------------------------------------------------------
# Claude's structured outputs reject a schema with 15 nullable enum fields ("Schema is too complex").
# So the model returns only the fields it found, as a flat list of (field, value) pairs - no optional
# or nullable fields - and the code below validates each value against questionnaire_answers.
FIELD_NAMES = list(questionnaire_answers.model_fields)
INT_FIELDS = ("num_dependants", "invest_amount", "target", "cash_needed_now")


class ExtractedField(BaseModel):
    field: Literal[tuple(FIELD_NAMES)]  # type: ignore[valid-type]
    value: str


class Extraction(BaseModel):
    fields: list[ExtractedField]


def _allowed_values(field: str) -> list[str] | None:
    """Allowed strings for a Literal field (None for free-text / integer fields)."""
    for arg in get_args(questionnaire_answers.model_fields[field].annotation):
        if get_origin(arg) is Literal:
            return list(get_args(arg))
    return None


def _field_guide() -> str:
    lines = []
    for f in FIELD_NAMES:
        allowed = _allowed_values(f)
        if allowed:
            lines.append(f"- {f}: one of {allowed}")
        elif f in INT_FIELDS:
            lines.append(f"- {f}: a whole number, digits only (e.g. 35000)")
        else:
            lines.append(f"- {f}: short free text")
    return "\n".join(lines)


_FIELD_GUIDE = _field_guide()
_DASHES = str.maketrans({"–": "-", "—": "-"})


def _coerce(field: str, raw: str):
    """Map the model's string onto the questionnaire_answers type; None if it doesn't fit."""
    raw = raw.strip()
    allowed = _allowed_values(field)
    if allowed:  # match ignoring case and dash style (some options use '–', others '-')
        key = raw.translate(_DASHES).lower()
        return next((a for a in allowed if a.translate(_DASHES).lower() == key), None)
    if field in INT_FIELDS:
        digits = raw.replace(",", "").replace("$", "").split(".")[0].strip()
        return int(digits) if digits.isdigit() else None
    return raw or None


def extract_update(conversation_text: str, asked_field: str | None = None) -> questionnaire_answers | None:
    # Some questions share the same wording (target_years / horizon_years both ask "When will you need
    # this money?"), so tell the model which field the bot's question was for.
    hint = (f"The bot's last question asks for the field `{asked_field}`. A direct answer to that question "
            f"belongs in `{asked_field}`, not in another field with similar wording.\n"
            if asked_field else "")
    result = llm.generate_structured(
        (
            "Extract any investor-profile fields the client has stated so far. "
            "Return only fields that were actually mentioned - never guess or infer a value. "
            "Exception: for fields that are a count or amount (num_dependants, cash_needed_now), "
            "if the client answers 'no' or 'none', record 0.\n"
            "Record the client's CURRENT situation, not a future plan: e.g. 'retiring next year' or 'about to "
            "retire' means they still work (employed, or self-employed if they said so), not retired.\n"
            "`sectors` is ONLY for things the client wants to EXCLUDE or avoid. A wish to invest in something (e.g. "
            "'put everything into tech stocks') is not an exclusion - do not record it in `sectors`.\n"
            "Use exactly one of the allowed values where a field lists them.\n"
            f"{hint}\n"
            f"Fields:\n{_FIELD_GUIDE}\n\n"
            f"Conversation:\n{conversation_text}"
        ),
        Extraction,
    )
    if result is None:
        return None
    values = {}
    for item in result.fields:
        v = _coerce(item.field, item.value)
        if v is not None:
            values[item.field] = v
    return questionnaire_answers(**values)

def merge_answers(current: dict, new: questionnaire_answers | None) -> dict:
    if new is None:
        return current # extraction failed validation, keep what we already had
    for field, value in new.model_dump().items(): # model_dump() turns pydantic object into plain dict
        if value is not None:
            current[field] = value
    return sync_time_fields(current)


def sync_time_fields(answers: dict) -> dict:
    """target_years (Section 3, when the goal money is needed) and horizon_years (Section 6, investment
    time horizon) are the same question, so it is asked once and the answer fills both fields.
    The two Literals differ only in dash style ('4-10 years' vs '4–10 years')."""
    for src, dst in (("target_years", "horizon_years"), ("horizon_years", "target_years")):
        if answers.get(src) is not None and answers.get(dst) is None:
            answers[dst] = _coerce(dst, answers[src])
    return answers

# ------------------------------------------
# 3. LANGGRAPH CONVERSATION LOOP
# ------------------------------------------
class ChatState(TypedDict):
    # worker appends notes here; operator.add concatenates the lists instead of overwriting
    messages: Annotated[list, add_messages]
    answers: dict

QUESTION_TEXT = {
    "age": "What is your age? (18–24, 25–34, 35–44, 45–54, 55–64, 65+)",
    "employment": "What is your current employment status? (student / employed / self-employed / retired)",
    "num_dependants": "Do you have any dependants? How many?",
    "income": "What is your approximate annual income? (Under $20,000, $20,000–$39,999, $40,000–$59,999, $60,000–$79,999, $80,000–$99,999, $100,000–$149,999, $150,000+)",
    "invest_amount": "What is the amount you're looking to invest today?",
    "other_assets": "Do you have other savings or assets outside this investment (e.g., emergency fund)?",
    "cash_needed_now": "Will you need any of this money as cash right now? If so, how much?",
    "objective_str": "What are you investing for? (Save for retirement / General savings / Buy a house / Save for education / Build an emergency fund)",
    "target_yn": "Do you have a specific target amount for this goal? (Yes/No)",
    "target": "If yes, how much is your target amount?",
    "target_years": "When will you need this money? (10+ years, 4–10 years, 1–3 years, 0–1 years)",
    "exp": "How would you describe your investing experience? (none / some / experienced)",
    "rr_familiarity": "Are you familiar with how risk and return are related in investing? (Yes/No/Unsure)",
    "desc": "Which statement best describes you? (Very low risk / Low risk / Moderate risk / High risk)",
    "principal_protection": "Do you need your original amount fully protected, even if that means lower returns? (Yes/No)",
    "horizon_years": "When will you need this money? (10+ years, 4–10 years, 1–3 years, 0–1 years)",
    "sectors": "Are there any sectors, industries, or companies you'd like to exclude? (e.g., fossil fuels, specific themes)",
}

def effective_answers(answers: dict) -> dict:
    a = answers.copy()
    if a.get("target_yn") == "No":
        a["target"] = "n/a"  # treat as answered, not a real value to save
    return a

def ask_node(state: ChatState) -> dict:
    checked = effective_answers(state["answers"])
    first_none_key = next((k for k, v in checked.items() if v is None), None)
    question = QUESTION_TEXT.get(first_none_key, "Thanks, I have everything I need!")
    return {"messages": [("bot", question)]}

def listen_node(state: ChatState) -> dict:
    recent = state["messages"][-2:]  # the bot's question + the user's answer
    convo_text = "\n".join(f"{role}: {content}" for role, content in recent)

    # which questionnaire field was the bot asking for? (None after a clarifying question)
    pending = next((k for k, v in effective_answers(state["answers"]).items() if v is None), None)
    asked = pending if recent and recent[0] == ("bot", QUESTION_TEXT.get(pending)) else None

    new_answers = extract_update(convo_text, asked_field=asked)
    merged_answers = merge_answers(state["answers"], new_answers)
    return {"answers": merged_answers}

def all_answered(answers: dict) -> bool:
    checked = effective_answers(answers)
    return all(value is not None for value in checked.values())

# ------------------------------------------
# 4. CONTRADICTION CHECK
# ------------------------------------------
class ContradictionCheck(BaseModel):
    has_contradiction: bool
    clarifying_question: Optional[str] = None

def check_contradiction(state: ChatState) -> ContradictionCheck:
    convo_text = "\n".join(f"{role}: {content}" for role, content in state["messages"])
    result = llm.generate_structured(
        (
            "Review this investor conversation for contradictions or unrealistic requests — "
            "e.g. a client who states a low risk tolerance but then asks for a highly concentrated "
            "or very risky allocation, or asks for a guaranteed high return. "
            "If you find one, set has_contradiction=true and write a clarifying question to ask the client."
            "Otherwise set has_contradiction=false and leave clarifying_question null. "
            "Do not calculate or quote any number the client did not state (no required return rates or "
            "projections). If the client already answered a clarifying question about the same issue, treat it "
            "as resolved. These are NOT contradictions: a current status together with a future plan (e.g. "
            "employed and retiring next year), and an amount the bot converted to CAD with the client's "
            "confirmation (all profile amounts are in CAD).\n\n"
            f"Conversation:\n{convo_text}\n\nCurrent profile so far:\n{state['answers']}"
        ),
        ContradictionCheck,
    )
    # model declined / no output -> don't block the conversation
    return result if result is not None else ContradictionCheck(has_contradiction=False)

# ------------------------------------------
# 4. Wrap into a callable function
# ------------------------------------------
MAX_CONTRADICTION_QUESTIONS = 2


def run_profiling_turn(state: ChatState, user_message: str | None) -> tuple[str, ChatState]:
    if user_message is not None:
        state["messages"].append(("client", user_message))
        listen_result = listen_node(state)
        state["answers"] = listen_result["answers"]

        # Ask about contradictions at most MAX_CONTRADICTION_QUESTIONS times, and never re-check the answer to
        # a contradiction question itself (otherwise the same question repeats forever).
        prev_bot = state["messages"][-2][1] if len(state["messages"]) >= 2 and state["messages"][-2][0] == "bot" else None
        answering_contradiction = prev_bot is not None and prev_bot not in QUESTION_TEXT.values()
        asked = state.get("contradictions_asked", 0)
        if asked < MAX_CONTRADICTION_QUESTIONS and not answering_contradiction:
            contradiction = check_contradiction(state)
            if contradiction.has_contradiction and contradiction.clarifying_question:
                state["contradictions_asked"] = asked + 1
                state["messages"].append(("bot", contradiction.clarifying_question))
                return contradiction.clarifying_question, state

    if all_answered(state["answers"]):
        return "Thanks, I have everything I need!", state

    ask_result = ask_node(state)
    question_text = ask_result["messages"][0][1]
    state["messages"].append(("bot", question_text))
    return question_text, state

RUN_METADATA = {
    **llm.SETTINGS,
    "system_prompt": "Extract any investor-profile fields the client has stated so far...",
}

def save_transcript(state: ChatState, label: str = "conversation"):
    record = {
        **RUN_METADATA,
        "timestamp": datetime.now().isoformat(),
        "messages": [{"role": role, "content": content} for role, content in state["messages"]],
        "final_answers": state["answers"],
    }
    filename = f"logs/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)
    return filename

# ------------------------------------------
# 5. Testing
# ------------------------------------------
os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    state = {"messages": [], "answers": questionnaire_answers().model_dump()}

    bot_msg, state = run_profiling_turn(state, None)
    print("Bot:", bot_msg)

    while not all_answered(state["answers"]):
        user_msg = input("You: ")
        bot_msg, state = run_profiling_turn(state, user_msg)
        print("Bot:", bot_msg)

    print("Final profile:", state["answers"])
    filename = save_transcript(state, label="test_run")
    print("Saved to:", filename)