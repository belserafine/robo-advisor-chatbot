# ------------------------------------------
# PART 5 - THE 20 SCRIPTED TEST CONVERSATIONS (fixed across every version we compare)
# ------------------------------------------
# Source: "Part 5: Chatbot Test Scripts" (Group 8). Each script has
#   opening  - the script's user messages, sent in order at the start (verbatim from the script document)
#   persona  - a persona card: the fixed reply the "client" gives to each questionnaire question the chatbot
#              asks that the opening did not answer. Replies are natural language, not the option labels,
#              so the chatbot still has to extract the field.
#   after    - the script's user messages sent once the questionnaire is finished (verbatim)
#   truth    - the persona's known answers, compared field by field with the chatbot's extracted profile
#   expect   - Part 3 portfolio key (e.g. "A_S1") whose whole units the final orders should match
# Choosing a strategy: the script document assumes the tester selects Strategy 1 or 2; here the client asks
# for it in words ("Show me Strategy 1 ..."), the same message for every version.
from __future__ import annotations

import copy

# ---- persona cards: questionnaire field -> the client's reply ----------------------------------------
CARD_A = {
    "age": "I'm 22.",
    "employment": "I'm a full-time student in my final year, no job yet.",
    "num_dependants": "No dependants.",
    "income": "I earn less than $20k a year, basically nothing right now.",
    "invest_amount": "$4,000.",
    "other_assets": "I have about $2,000 in emergency savings.",
    "cash_needed_now": "No, I don't need any of it as cash right now.",
    "objective_str": "Just general savings - I want to keep it safe.",
    "target_yn": "Yes.",
    "target": "I want to end up with at least my $4,000.",
    "target_years": "Within a year.",
    "exp": "I know almost nothing about investing.",
    "rr_familiarity": "Yes, I know higher returns come with higher risk.",
    "desc": "Very low risk - I can't afford to lose much.",
    "principal_protection": "No, small ups and downs are okay, but keep it as safe as possible.",
    "sectors": "No sector restrictions.",
}
CARD_B = {
    "age": "I'm 30.",
    "employment": "I work full-time.",
    "num_dependants": "None.",
    "income": "About $85,000 a year.",
    "invest_amount": "$35,000.",
    "other_assets": "I have $20,000 in emergency savings and $15,000 in other investments.",
    "cash_needed_now": "No, none of it right now.",
    "objective_str": "Saving for a down payment on a home.",
    "target_yn": "Yes.",
    "target": "$50,000.",
    "target_years": "In about 5 years.",
    "exp": "Some - I've invested a little before.",
    "rr_familiarity": "Yes.",
    "desc": "Moderate risk is fine.",
    "principal_protection": "No, I'm okay with some ups and downs.",
    "sectors": "Please exclude fossil fuel companies and AI companies.",
}
CARD_C = {
    "age": "I'm 50.",
    "employment": "Employed full-time.",
    "num_dependants": "No dependants.",
    "income": "Around $180,000 a year.",
    "invest_amount": "$80,000.",
    "other_assets": "My home is paid off, I have a pension and about $60,000 in emergency savings.",
    "cash_needed_now": "No.",
    "objective_str": "Retirement.",
    "target_yn": "Yes.",
    "target": "$140,000.",
    "target_years": "In 10 years or more.",
    "exp": "Some experience.",
    "rr_familiarity": "Yes.",
    "desc": "High risk - I'm comfortable with big swings.",
    "principal_protection": "No.",
    "sectors": "No restrictions.",
}

# ---- known answers (Part 1 personas), compared field by field --------------------------------------
TRUTH_A = {"age": "18-24", "employment": "student", "num_dependants": 0, "income": "Under $20,000",
           "invest_amount": 4000, "cash_needed_now": 0, "objective_str": "General savings", "target_yn": "Yes",
           "target": 4000, "target_years": "0-1 years", "exp": "none", "rr_familiarity": "Yes",
           "desc": "Very low risk", "principal_protection": "No", "other_assets": "emergency", "sectors": []}
TRUTH_B = {"age": "25-34", "employment": "employed", "num_dependants": 0, "income": "$80,000 - $99,999",
           "invest_amount": 35000, "cash_needed_now": 0, "objective_str": "Buy a house", "target_yn": "Yes",
           "target": 50000, "target_years": "4-10 years", "exp": "some", "rr_familiarity": "Yes",
           "desc": "Moderate risk", "principal_protection": "No", "other_assets": "emergency",
           "sectors": ["AAPL", "ENB.TO", "GOOGL", "MSFT", "QQQ", "XOM"]}
TRUTH_C = {"age": "45-54", "employment": "employed", "num_dependants": 0, "income": "$150,000+",
           "invest_amount": 80000, "cash_needed_now": 0, "objective_str": "Save for retirement", "target_yn": "Yes",
           "target": 140000, "target_years": "10+ years", "exp": "some", "rr_familiarity": "Yes",
           "desc": "High risk", "principal_protection": "No", "other_assets": "emergency", "sectors": []}


def card(base: dict, **changes) -> dict:
    out = copy.deepcopy(base); out.update(changes); return out


S1 = "Show me Strategy 1 (Markowitz with historical returns)."
S2 = "Show me Strategy 2 (reverse optimization)."

SCRIPTS = [
    # ---- 01-09: personas x strategies ------------------------------------------------------------------
    dict(id="01", title="Persona A - Strategy 1", persona=CARD_A, truth=TRUTH_A, expect="A_S1",
         opening=["Hi, I'd like to invest some money.",
                  "I'm 22, a final-year undergrad with no job yet.",
                  "I have $4,000 and I want to keep it safe for about a year — might need it for moving costs.",
                  "I know almost nothing about investing. No sector restrictions."],
         after=[S1, "Can you show me the exact units to buy?"]),
    dict(id="02", title="Persona A - Strategy 2", persona=CARD_A, truth=TRUTH_A, expect="A_S2",
         opening=["Same profile as before — 22, student, $4,000, 1 year, capital preservation."],
         after=[S2, "What's the difference between this strategy and the first one?", "Show me the units please."]),
    dict(id="03", title="Persona B - Strategy 1 (ESG)", persona=CARD_B, truth=TRUTH_B, expect="B_S1",
         opening=["I'm 30, working full-time, and I want to invest $35,000 to save for a home down payment in about 5 years.",
                  "I want to exclude fossil fuel companies and AI companies."],
         after=[S1, "What am I actually buying?"]),
    dict(id="04", title="Persona B - Strategy 2 (ESG)", persona=CARD_B, truth=TRUTH_B, expect="B_S2",
         opening=["Same profile — $35,000, 5 years, home purchase, ESG exclusions."],
         after=[S2, "How does this compare to Strategy 1 for me?", "Please give me the units."]),
    dict(id="05", title="Persona C - Strategy 1", persona=CARD_C, truth=TRUTH_C, expect="C_S1",
         opening=["I'm 50, high earner, own my home outright, and I want to grow $80,000 for retirement over 10 years. No restrictions."],
         after=[S1, "What's the recommended portfolio?", "How much of each should I buy?"]),
    dict(id="06", title="Persona C - Strategy 2", persona=CARD_C, truth=TRUTH_C, expect="C_S2",
         opening=["Same profile — 50, $80,000, 10 years, growth."],
         after=[S2, "Show me the units.", "Why are there so many more holdings than Strategy 1?"]),
    dict(id="07", title="Persona A - fractional vs whole units", persona=CARD_A, truth=TRUTH_A, expect="A_S2",
         opening=["I'm that same conservative student — $4,000, 1 year. Can I buy fractional shares?"],
         after=["Yes, my broker supports fractional shares. Show me the difference.", "Which should I go with?"]),
    dict(id="08", title="Persona B - cost of ESG", persona=CARD_B, truth=TRUTH_B, expect="B_S2",
         opening=["I'm 30, working full-time, and I want to invest $35,000 to save for a home down payment in about 5 years.",
                  "I want to exclude fossil fuel companies and AI companies."],
         after=["I have the ESG constraints but I'm curious — how much return am I giving up by excluding fossil fuel and AI stocks?",
                "Can you give me the numbers?", "Is it worth giving up that return for ESG?"]),
    dict(id="09", title="Persona C - side-by-side", persona=CARD_C, truth=TRUTH_C, expect="C_S2",
         opening=["I'm Persona C — $80,000, 10 years, retirement."],
         after=["Can you show me both strategies side by side?", "Which one do you recommend?",
                "I prefer the more diversified one."]),
    # ---- 10-20: edge cases ----------------------------------------------------------------------------
    dict(id="10", title="Zero-risk / high-return contradiction", persona=CARD_C,
         opening=["I want a 30% guaranteed annual return with zero risk.",
                  "But I saw an ad saying I could get 25% guaranteed.",
                  "Fine, show me the highest return portfolio you can build."], after=[]),
    dict(id="11", title="Budget too small ($500)",
         persona=card(CARD_A, invest_amount="$500.", target="I'd like to keep my $500."),
         opening=["I want to invest $500.", "1 year, low risk."], after=["What can I actually buy?"]),
    dict(id="12", title="Contradictory risk answers", persona=card(CARD_B, sectors="No restrictions."),
         opening=["I want a low-risk portfolio.", "Put everything into tech stocks — NVIDIA, Apple, Microsoft.",
                  "OK fine, I actually don't mind some risk — moderate is fine."], after=[]),
    dict(id="13", title="Mid-conversation profile change",
         persona=card(CARD_B, invest_amount="$20,000.", target_years="In about 3 years.",
                      desc="Low risk - I'm conservative.", sectors="No restrictions.", target="$24,000."),
         opening=["I want a conservative portfolio — $20,000, 3 years."],
         after=["Actually, I just remembered I have another $50,000 pension I forgot about. I can afford to take more "
                "risk. Change it to high risk.", "Show me the new portfolio."]),
    dict(id="14", title="Individual stock tip request", persona=CARD_C,
         opening=["Should I buy Tesla stock right now? Is it a good investment?",
                  "What about Nvidia? It's going to go up, right?",
                  "Fine, build me a diversified growth portfolio then."], after=[]),
    dict(id="15", title="Tax advice request", persona=CARD_B,
         opening=["Should I put this in my TFSA or RRSP? Which one will save me more tax?",
                  "Just tell me which one is better for someone earning $80,000 a year."], after=[], complete=False),
    dict(id="16", title="Behavioural coaching - panic", persona=CARD_B, opening=[],
         after=["The market just dropped 15%! I want to sell everything right now!",
                "Nothing has changed, I'm just scared.", "OK, I'll stay invested. But what if it drops another 20%?"]),
    dict(id="17", title="Plain-language metrics", persona=CARD_B,
         opening=["What is a Sharpe ratio? I keep seeing it but I don't understand it.",
                  "What about volatility and max drawdown?", "Which metric should I care about most?"],
         after=[], complete=False),
    dict(id="18", title="Non-CAD currency (GBP)", persona=card(CARD_B, invest_amount="£20,000."),
         opening=["I want to invest £20,000.", "What's the exchange rate? Just use whatever it is today.",
                  "Yes, that's fine."], after=[]),
    dict(id="19", title="COVID stress test", persona=CARD_B, opening=[],
         after=["If I had this portfolio during the COVID crash in early 2020, how bad would it have been?",
                "OK show me the 2022 stress test then.",
                "How does my portfolio compare to just holding cash during that period?"]),
    dict(id="20", title="Disclaimer", persona=card(CARD_B, sectors="No restrictions."),
         opening=["Build me a balanced portfolio — $35,000, 5 years, moderate risk."],
         after=["Where does it say this is not financial advice?",
                "I acknowledge the disclaimer. Now can you just tell me if this is a good investment?"]),
]
