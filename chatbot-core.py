from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]

llm = genai.Client(api_key=api_key)


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
        model="gemini-3.5-flash-lite", # TODO: can change to any model
        contents=(
            "Extract any investor-profile fields the client has stated so far."
            "Leave a field null if it was not mentioned - never guess or infer a value\n\n"
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