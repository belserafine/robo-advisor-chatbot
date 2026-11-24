# ------------------------------------------
# 0. IMPORTS & CONFIGURATION
# ------------------------------------------
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Literal, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
import json
from datetime import datetime

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]

llm = genai.Client(api_key=api_key)

MODEL = "gemini-3.5-flash"

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

    ### SECTION 6: INVESTMENT TIME HORIZON
    horizon_years: Optional[Literal["10+ years", "4–10 years", "1–3 years", "0–1 years"]] = None

    ### SECTION 7: PREFERENCES
    # Are there any sectors, industries, or companies you’d like to exclude? (e.g., fossil fuels, specific themes)
    sectors: Optional[str] = None

def extract_update(conversation_text: str) -> questionnaire_answers:
    response = llm.models.generate_content(
        model=MODEL,
        contents=(
            "Extract any investor-profile fields the client has stated so far. "
            "Leave a field null if it was not mentioned - never guess or infer a value. "
            "Exception: for fields that are a count (like num_dependants), "
            "if the client answers 'no' or 'none', record 0, not null.\n\n"
            f"Conversation:\n{conversation_text}"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", # tells Gemini to respond in JSON not free-flowing prose like "Sure, here's the classification:: ..."
            response_schema = questionnaire_answers,
    ),
    )
    return response.parsed

def merge_answers(current: dict, new: questionnaire_answers | None) -> dict:
    if new is None:
        return current # extraction failed validation, keep what we already had
    for field, value in new.model_dump().items(): # model_dump() turns pydantic object into plain dict
        if value is not None:
            current[field] = value
    return current

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
    "objective_str": "What are you investing for? (Save for retirement / General savings / Buy a house / Save for education / Build an emergency fund)",
    "target_yn": "Do you have a specific target amount for this goal? (Yes/No)",
    "target": "If yes, how much is your target amount?",
    "target_years": "When will you need this money? (10+ years, 4–10 years, 1–3 years, 0–1 years)",
    "exp": "How would you describe your investing experience? (none / some / experienced)",
    "rr_familiarity": "Are you familiar with how risk and return are related in investing? (Yes/No/Unsure)",
    "desc": "Which statement best describes you? (Very low risk / Low risk / Moderate risk / High risk)",
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

    new_answers = extract_update(convo_text)
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
    response = llm.models.generate_content(
        model=MODEL,
        contents=(
            "Review this investor conversation for contradictions or unrealistic requests — "
            "e.g. a client who states a low risk tolerance but then asks for a highly concentrated "
            "or very risky allocation, or asks for a guaranteed high return. "
            "If you find one, set has_contradiction=true and write a clarifying question to ask the client."
            "Otherwise set has_contradiction=false and leave clarifying_question null.\n\n"
            f"Conversation:\n{convo_text}\n\nCurrent profile so far:\n{state['answers']}"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ContradictionCheck,
            temperature=0.2,
        ),
    )
    return response.parsed

# ------------------------------------------
# 4. Wrap into a callable function
# ------------------------------------------
def run_profiling_turn(state: ChatState, user_message: str | None) -> tuple[str, ChatState]:
    if user_message is not None:
        state["messages"].append(("client", user_message))
        listen_result = listen_node(state)
        state["answers"] = listen_result["answers"]

        contradiction = check_contradiction(state)
        if contradiction.has_contradiction:
            state["messages"].append(("bot", contradiction.clarifying_question))
            return contradiction.clarifying_question, state

    if all_answered(state["answers"]):
        return "Thanks, I have everything I need!", state

    ask_result = ask_node(state)
    question_text = ask_result["messages"][0][1]
    state["messages"].append(("bot", question_text))
    return question_text, state

RUN_METADATA = {
    "provider": "Google Gemini",
    "model": MODEL,
    "temperature": 0.2,
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