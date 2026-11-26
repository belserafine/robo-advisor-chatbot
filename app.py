# ------------------------------------------
# ROBO-ADVISOR CHATBOT - Streamlit UI
# ------------------------------------------
# Run:  streamlit run app.py
#
# The multi-agent flow is a LangGraph StateGraph (robo_graph.py); this file only holds the UI and
# the per-session graph state. Each client message = one graph run:
#   Profiling agent -> (complete?) -> Portfolio agent <-> Checking agent (self-correction) -> Response
# Every number shown here comes from portfolio_agent.get_portfolio() (code), never from the LLM.
from __future__ import annotations

import json
import os
import re
from datetime import datetime

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

import llm
import portfolio_agent
import profiling_agent
import robo_graph

BOT_AVATAR = ":material/smart_toy:"  # styled as a dark tile with a gold icon (CSS)
USER_AVATAR = ":material/person:"
DISCLAIMER = (
    "For educational purposes only - not financial, investment, tax or legal advice. "
    "Past performance does not guarantee future results. Talk to a registered advisor before investing."
)
GREETING = (
    "Hi there 👋 I'll ask you about 16 short questions about your finances, goals and comfort with risk, "
    "then build a portfolio for you: what to buy, how many units, and how it would have performed.\n\n"
    "Tip: you can answer several questions in one message."
)

# ---- palette (design spec) ----
FOREST, EMERALD, GOLD = "#142D28", "#26735C", "#C5A46D"
IVORY, WHITE, TEXT, TEXT2 = "#F7F8F5", "#FFFFFF", "#172E27", "#68766F"
BORDER, BUBBLE, NEUTRAL = "#DEE5DF", "#EAF2ED", "#B9C4BE"

# Chart colours: fixed per entity (never by rank)
_SN = portfolio_agent.STRATEGY_NAMES
STRATEGY_COLORS = {_SN["S2"]: EMERALD, _SN["S1"]: GOLD, _SN["EW"]: TEXT2}
CLASS_COLORS = {"Equity": EMERALD, "Fixed Income": GOLD, "Commodity": "#7FA595", "Cash & fees": NEUTRAL}

# One-tap answers for multiple-choice questions (sent as a normal chat message)
QUICK_REPLIES = {
    "age": ["18–24", "25–34", "35–44", "45–54", "55–64", "65+"],
    "employment": ["Student", "Employed", "Self-employed", "Retired"],
    "num_dependants": ["None", "1", "2", "3 or more"],
    "income": ["Under $20,000", "$20,000 – $39,999", "$40,000 – $59,999", "$60,000 – $79,999",
               "$80,000 – $99,999", "$100,000 – $149,999", "$150,000+"],
    "other_assets": ["Yes, I have an emergency fund", "No other savings"],
    "cash_needed_now": ["No, none right now"],
    "objective_str": ["Save for retirement", "General savings", "Buy a house", "Save for education",
                      "Build an emergency fund"],
    "target_yn": ["Yes", "No"],
    "target_years": ["0–1 years", "1–3 years", "4–10 years", "10+ years"],
    "exp": ["None", "Some", "Experienced"],
    "rr_familiarity": ["Yes", "No", "Unsure"],
    "desc": ["Very low risk", "Low risk", "Moderate risk", "High risk"],
    "principal_protection": ["Yes", "No"],
    "horizon_years": ["0–1 years", "1–3 years", "4–10 years", "10+ years"],
    "sectors": ["No exclusions", "Fossil fuels", "AI companies", "Fossil fuels and AI companies"],
}
FIELD_LABELS = {
    "age": "Age", "employment": "Employment", "num_dependants": "Dependants", "income": "Income",
    "invest_amount": "Amount to invest", "other_assets": "Other savings", "cash_needed_now": "Cash needed now",
    "objective_str": "Goal",
    "target_yn": "Target amount?", "target": "Target", "target_years": "Time horizon",
    "exp": "Experience", "rr_familiarity": "Knows risk/return", "desc": "Risk comfort",
    "principal_protection": "Principal protection",
    "sectors": "Exclusions",
}

st.set_page_config(page_title="Robo-Advisor", page_icon=":material/home:", layout="wide")

st.html(f"""
<style>
  .block-container {{padding-top: 2.2rem; padding-bottom: 7rem; max-width: 1180px;}}
  h1, h2, h3, h4 {{color: {TEXT};}}

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] {{min-width: 320px;}}
  [data-testid="stSidebarUserContent"] {{padding-top: 1.4rem;}}
  .brand {{display: flex; align-items: center; gap: .85rem; margin-bottom: 1.6rem;}}
  .brand .logo {{width: 54px; height: 54px; border-radius: 12px; background: {GOLD}; display: grid; place-items: center;}}
  .brand .logo {{background: {GOLD} url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMTQyRDI4IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PHBhdGggZD0iTTMgMTEuNSAxMiA0bDkgNy41Ii8+PHBhdGggZD0iTTUuNSAxMHY5LjVoMTNWMTAiLz48cGF0aCBkPSJNMTAgMTkuNXYtNWg0djUiLz48L3N2Zz4=") center / 30px no-repeat;}}
  .brand .name {{font-size: 1.45rem; font-weight: 700; color: {WHITE}; line-height: 1.15;}}
  .brand .tag {{font-size: .68rem; letter-spacing: .2em; color: {GOLD}; font-weight: 600;}}
  .status {{display: flex; align-items: center; gap: .55rem; color: {WHITE}; font-size: .95rem; margin: 0 0 .5rem;}}
  .status .dot {{width: 11px; height: 11px; border-radius: 50%; background: #8FD1AE;}}
  .status .dot.pending {{background: {GOLD};}}
  .pbar {{height: 4px; background: #2E4A43; border-radius: 4px; overflow: hidden; margin-bottom: 1.4rem;}}
  .pbar div {{height: 100%; background: {GOLD};}}
  .sb-section {{font-size: .74rem; letter-spacing: .18em; color: {GOLD}; font-weight: 700; margin: 1.4rem 0 .55rem;}}
  .sb-row {{display: flex; justify-content: space-between; gap: .8rem; font-size: .92rem; padding: .38rem 0;}}
  .sb-row span:first-child {{color: #B9C4BE;}}
  .sb-row span:last-child {{color: {WHITE}; text-align: right; font-weight: 500;}}
  .sb-rule {{border-top: 1px solid #2E4A43; margin: 1.1rem 0;}}
  .sb-title {{color: {WHITE}; font-weight: 700; font-size: .98rem; margin-bottom: .55rem;}}
  .pill {{display: inline-block; padding: .38rem .9rem; border-radius: 8px; font-size: .9rem; font-weight: 500;}}
  .pill.ok {{background: #2B5A4C; color: #CDEBDC;}}
  .pill.warn {{background: #4A3F27; color: #F0D9AE;}}
  .pill.bad {{background: #5A2B2B; color: #F3C9C9;}}
  .sb-issue {{color: #B9C4BE; font-size: .8rem; margin-top: .4rem; line-height: 1.4;}}
  [data-testid="stSidebar"] .stButton button {{background: transparent; border: 1px solid #8FA39B; color: {WHITE};
      border-radius: 10px; padding: .55rem 0;}}
  [data-testid="stSidebar"] .stButton button:hover {{border-color: {GOLD}; color: {GOLD};}}

  /* ---------- page headers ---------- */
  .eyebrow {{font-size: .82rem; letter-spacing: .22em; color: #A8874F; font-weight: 600; margin-bottom: .25rem;}}
  .page-title {{font-size: 2.7rem; font-weight: 800; color: {TEXT}; line-height: 1.1; margin: 0 0 .35rem;}}
  .page-sub {{color: {TEXT2}; font-size: 1.05rem;}}
  .asof {{color: {TEXT2}; font-size: .95rem; text-align: right; padding-top: .3rem;}}
  .head-row {{display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 1.4rem;}}
  .disclaimer-bar {{background: {WHITE}; border: 1px solid {BORDER}; border-left: 3px solid {GOLD}; color: {TEXT2};
      border-radius: 10px; padding: .55rem .9rem; font-size: .82rem; margin: 0 0 1.2rem;}}
  .steps {{display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.2rem;}}
  .step {{border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.1rem; background: {WHITE};}}
  .step b {{display: block; font-size: .95rem; margin-bottom: .2rem; color: {TEXT};}}
  .step span {{color: {TEXT2}; font-size: .85rem; line-height: 1.45;}}
  .step .n {{display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px;
      background: {BUBBLE}; color: {EMERALD}; font-size: .8rem; font-weight: 700; margin-bottom: .5rem;}}

  /* ---------- metric cards ---------- */
  .mcards {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .9rem; margin: .3rem 0 1.2rem;}}
  .mcard {{background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.2rem;}}
  .mcard .l {{color: {TEXT2}; font-size: .92rem; margin-bottom: .45rem;}}
  .mcard .v {{color: {TEXT}; font-size: 1.9rem; font-weight: 700; line-height: 1.1;}}
  .mcards.small .mcard .v {{font-size: 1.55rem;}}
  .mcards-title {{font-weight: 700; font-size: 1.15rem; color: {TEXT}; margin-top: .2rem;}}
  .mcards-note {{color: {TEXT2}; font-size: .9rem;}}

  /* ---------- portfolio panel + tabs ---------- */
  [class*="st-key-panel_"] {{margin-top: 1.2rem; width: 100%;}}
  .st-key-overview {{background: {WHITE}; border: 1px solid {BORDER}; border-radius: 16px; padding: .6rem 1.6rem 1.2rem;}}
  .stTabs [data-baseweb="tab-list"] {{gap: 2.2rem; border-bottom: 1px solid {BORDER};}}
  .stTabs [data-baseweb="tab"] {{color: {TEXT2}; padding: .8rem .1rem; font-size: 1rem;}}
  .stTabs [data-baseweb="tab"] p {{font-size: 1.02rem;}}
  .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {{color: {EMERALD}; font-weight: 600;}}
  .stTabs [data-baseweb="tab-highlight"] {{background: {EMERALD}; height: 3px;}}
  .chart-title {{font-weight: 700; font-size: 1.1rem; color: {TEXT}; margin: .4rem 0 0;}}

  /* ---------- chat ---------- */
  [data-testid="stChatMessage"] {{background: transparent; padding: .35rem 0; gap: .9rem;}}
  [data-testid="stChatMessageAvatarCustom"] {{background: {FOREST}; color: {GOLD}; border-radius: 12px;
      width: 2.7rem; height: 2.7rem;}}
  [data-testid="stChatMessageAvatarCustom"] span {{color: {GOLD} !important;}}
  [class*="st-key-umsg_"], [class*="st-key-bmsg_"] {{flex-shrink: 0; height: auto !important; min-height: max-content;}}
  [class*="st-key-umsg_"] [data-testid="stMarkdownContainer"], [class*="st-key-umsg_"] .stMarkdown > div,
  [class*="st-key-bmsg_"] [data-testid="stMarkdownContainer"], [class*="st-key-bmsg_"] .stMarkdown > div {{margin-bottom: 0 !important;}}
  [data-testid="stChatInput"] > div {{background: {WHITE} !important; border-color: {BORDER} !important;}}
  [data-testid="stChatInputTextArea"] {{background: {WHITE} !important; color: {TEXT};}}
  button[data-baseweb="tab"] p {{font-size: 1.05rem !important;}}
  [data-testid="stChatMessage"]:has([class*="st-key-umsg_"]) [data-testid="stChatMessageAvatarCustom"] {{display: none;}}
  [class*="st-key-bmsg_"] {{background: {BUBBLE}; border-radius: 4px 18px 18px 18px; padding: 1rem 1.4rem;
      width: fit-content; max-width: 90%; gap: .55rem;}}
  [class*="st-key-bmsg_"]::before {{content: "Robo-Advisor"; font-weight: 700; color: {TEXT}; font-size: 1rem;}}
  [class*="st-key-bmsg_"] p, [class*="st-key-bmsg_"] li {{color: {TEXT}; line-height: 1.6;}}
  [class*="st-key-umsg_"] {{background: {EMERALD}; border-radius: 18px 18px 4px 18px; padding: .7rem 1.3rem;
      width: fit-content; max-width: 75%; margin-left: auto;}}
  [class*="st-key-umsg_"] p {{color: {WHITE}; margin: 0;}}
  [data-testid="stChatMessageContent"] p:last-child {{margin-bottom: 0;}}
  .msg-foot {{border-top: 1px solid #CFDCD4; padding-top: .7rem; margin-top: .3rem; color: {TEXT2};
      font-style: italic; font-size: .9rem;}}
  .st-key-quick_replies {{padding-left: 3.6rem; margin-top: -.1rem;}}
  .st-key-quick_replies button {{border-radius: 999px; border: 1px solid #9CC3B2; background: {WHITE};
      color: {EMERALD}; min-height: 0; padding: .3rem 1rem;}}
  .st-key-quick_replies button:hover {{background: {BUBBLE}; border-color: {EMERALD}; color: {EMERALD};}}
  [data-testid="stChatInput"] {{background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px;}}
  [data-testid="stChatInputSubmitButton"] {{background: {EMERALD}; border-radius: 8px; color: {WHITE};}}
  [data-testid="stChatInputSubmitButton"] svg {{fill: {WHITE}; color: {WHITE};}}
  [data-testid="stChatInputSubmitButton"]:disabled {{background: {EMERALD}; opacity: .85;}}
</style>
""")


# ------------------------------------------
# 1. SESSION STATE
# ------------------------------------------
def init_state():
    st.session_state.chat = [{"role": "assistant", "content": GREETING}]
    st.session_state.graph_state = robo_graph.run_turn(robo_graph.new_state(), None)  # first question
    for r in st.session_state.graph_state["replies"]:
        st.session_state.chat.append({"role": "assistant", **r})
    st.session_state.pending = None


if "graph_state" not in st.session_state:
    init_state()


def queue_message(text: str):
    st.session_state.pending = text


def current_question_key(gs) -> str | None:
    """Questionnaire field the bot is asking right now (None for clarifying questions / after profiling)."""
    if gs["phase"] != "profiling" or not gs["prof_state"]["messages"]:
        return None
    last_role, last_text = gs["prof_state"]["messages"][-1]
    checked = profiling_agent.effective_answers(gs["prof_state"]["answers"])
    key = next((k for k, v in checked.items() if v is None), None)
    if key and last_role == "bot" and last_text == profiling_agent.QUESTION_TEXT.get(key):
        return key
    return None


def save_log():
    os.makedirs("logs", exist_ok=True)
    gs = st.session_state.graph_state
    p = gs["portfolio"]
    record = {
        **llm.SETTINGS,
        "portfolio_system_prompt": portfolio_agent.DRAFT_SYSTEM_PROMPT,
        "timestamp": datetime.now().isoformat(),
        "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat],
        "final_answers": gs["prof_state"]["answers"],
        "portfolio": {k: v for k, v in p.items() if k != "backtest_series"} if p else None,
        "pipeline_log": gs["pipeline_log"],
    }
    path = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


# ------------------------------------------
# 2. PORTFOLIO PANEL
# ------------------------------------------
PLOTLY_CONFIG = {"displayModeBar": False}

def _style(fig: go.Figure, height: int) -> go.Figure:
    """Shared look: transparent background, recessive grid/axes, legend below the plot."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family='"Source Sans Pro", "Source Sans 3", sans-serif', size=13, color=TEXT2),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, title=None,
                                  font=dict(color=TEXT2)),
                      hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font=dict(size=12, color=TEXT)))
    fig.update_xaxes(gridcolor="#EDF1EE", linecolor=BORDER, zeroline=False, automargin=True, title_standoff=10)
    fig.update_yaxes(gridcolor="#EDF1EE", linecolor=BORDER, zeroline=False, automargin=True, title_standoff=10)
    return fig


def _money0(x: float) -> str:
    return f"${x:,.0f}"


def chart_allocation(p: dict):
    classes = [c for c in CLASS_COLORS if c in p["asset_class_mix"]]
    donut = go.Figure(go.Pie(
        labels=classes, values=[p["asset_class_mix"][c] for c in classes], hole=0.66, sort=False,
        direction="clockwise", marker=dict(colors=[CLASS_COLORS[c] for c in classes], line=dict(color=WHITE, width=2)),
        textinfo="none", hovertemplate="%{label}: %{value:.1%}<extra></extra>"))
    _style(donut, 340).update_layout(showlegend=True, legend=dict(y=-0.02, x=0.5, xanchor="center"),
                                     margin=dict(l=8, r=8, t=8, b=8))
    donut.add_annotation(text=f"<b>{_money0(p['budget'])}</b>", x=0.5, y=0.54, showarrow=False,
                         font=dict(size=26, color=TEXT))
    donut.add_annotation(text="Total budget", x=0.5, y=0.42, showarrow=False, font=dict(size=13, color=TEXT2))
    h = sorted([x for x in p["holdings"] if x["units"] > 0], key=lambda x: x["weight"], reverse=True)
    bars = go.Figure(go.Bar(
        x=[x["weight"] for x in h], y=[x["ticker"] for x in h], orientation="h",
        marker=dict(color=[CLASS_COLORS.get(x["asset_class"], EMERALD) for x in h], cornerradius=3), width=0.6,
        text=[f"{x['weight']:.1%}" for x in h], textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT, size=13),
        customdata=[[x["name"], x["value_cad"], x["units"]] for x in h],
        hovertemplate="%{customdata[0]}<br>%{customdata[2]} units · $%{customdata[1]:,.2f}<extra></extra>"))
    _style(bars, max(240, 34 * len(h) + 30))
    bars.update_xaxes(visible=False, range=[0, max(x["weight"] for x in h) * 1.22])
    bars.update_yaxes(autorange="reversed", showgrid=False, showline=False, tickfont=dict(color=TEXT2, size=13))
    return donut, bars


def chart_backtest(p: dict, budget: float):
    s = pd.DataFrame(p["backtest_series"])
    s.index = pd.to_datetime(s.index)
    fig = go.Figure()
    for name, color in STRATEGY_COLORS.items():
        if name in s:
            main = name == p["strategy_name"]
            fig.add_trace(go.Scatter(x=s.index, y=s[name], name=name, mode="lines",
                                     line=dict(color=color, width=3 if main else 2, dash=None if main else "solid"),
                                     hovertemplate="$%{y:,.0f}"))
    fig.add_hline(y=budget, line=dict(color=NEUTRAL, width=1, dash="dash"),
                  annotation_text="Starting amount", annotation_position="bottom right",
                  annotation_font=dict(color=TEXT2, size=11))
    _style(fig, 380).update_layout(hovermode="x unified", legend=dict(y=-0.1))
    fig.update_xaxes(tickformat="%b %Y", dtick="M1", hoverformat="%b %d, %Y", showgrid=False)
    fig.update_yaxes(title_text="Value of your money (CAD)", tickformat="$,.0f")
    return fig


def metric_cards(items, small=False) -> str:
    cards = "".join(f'<div class="mcard"><div class="l">{l}</div><div class="v">{str(v).replace("-", "−", 1) if str(v).startswith("-") else v}</div></div>'
                    for l, v in items)
    return f'<div class="mcards{" small" if small else ""}">{cards}</div>'


def render_recommendation(p: dict, key: str, header: bool = True):
    f = p["facts"]
    rec = f["recommended"]
    if header:
        asof = datetime.strptime(f["price_date"], "%Y-%m-%d").strftime("%b %d, %Y")
        st.html(f"""
<div class="head-row"><div><div class="eyebrow">PORTFOLIO OVERVIEW</div>
  <div class="page-title">Your portfolio</div>
  <div class="page-sub">{f['recommended_strategy'].replace(': ', ' · ')}</div></div>
  <div class="asof">As of {asof}</div></div>""")
    st.html(metric_cards([("Risk level", f["risk_level"].capitalize()), ("Invested", f["invested"]),
                          ("Expected return / yr", rec["expected_annual_return"]),
                          ("Expected volatility", rec["expected_annual_volatility"])]))
    with st.container(key=f"overview{'' if header else '_' + key}"):
        t_orders, t_alloc, t_perf, t_cmp, t_why = st.tabs(
            ["Orders", "Allocation", "Past performance", "Compare", "Why this?"])

        with t_orders:
            h = pd.DataFrame([x for x in p["holdings"] if x["units"] > 0])
            table = pd.DataFrame({
                "Ticker": h["ticker"], "Asset": h["name"], "Type": h["type"], "Units to buy": h["units"],
                "Price (CAD)": h["price_cad"], "Amount (CAD)": h["value_cad"], "Weight": h["weight"] * 100,
            })
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Asset": st.column_config.TextColumn(width="medium"),
                "Price (CAD)": st.column_config.NumberColumn(format="dollar"),
                "Amount (CAD)": st.column_config.NumberColumn(format="dollar"),
                "Weight": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100, width="small"),
            })
            st.caption(f"Whole units only · fees {f['fees']} · uninvested cash {f['cash_residual']} · "
                       f"budget {f['budget']}")
            st.download_button("Download orders (CSV)", table.to_csv(index=False), "orders.csv", "text/csv",
                               icon=":material/download:", key=f"dl_{key}")

        with t_alloc:
            donut, bars = chart_allocation(p)
            a, b = st.columns([5, 6], gap="large")
            with a:
                st.html('<div class="chart-title">Asset allocation</div>')
                st.plotly_chart(donut, config=PLOTLY_CONFIG, theme=None, key=f"donut_{key}")
            with b:
                st.html('<div class="chart-title">Portfolio holdings</div>')
                st.plotly_chart(bars, config=PLOTLY_CONFIG, theme=None, key=f"bars_{key}")

        with t_perf:
            st.markdown(f"How {f['budget']} invested on {p['test_window'][0]} would have grown until "
                        f"{p['test_window'][1]}: data the model never saw.")
            st.plotly_chart(chart_backtest(p, p["budget"]), config=PLOTLY_CONFIG, theme=None, key=f"bt_{key}")
            st.caption(f"Bought as whole units on {p['test_window'][0]}, rebalanced on the first trading day of each month, "
                       f"{p['transaction_cost']:.2%} cost per trade (Part 4 backtest). Past performance does not guarantee "
                       "future results.")

        with t_cmp:
            rows = []
            for name, s in f["strategies"].items():
                rows.append({"Strategy": ("⭐ " if name == f["recommended_strategy"] else "") + name.split(": ", 1)[-1],
                             "Exp. return": s["expected_annual_return"], "Exp. vol.": s["expected_annual_volatility"],
                             "Backtest": s["backtest_total_return"], "Ann. return": s["backtest_annualized_return"],
                             "Ann. vol.": s["backtest_volatility"], "Max DD": s["backtest_max_drawdown"],
                             "Sharpe": s["backtest_sharpe"], "Turnover / mo": s["backtest_avg_monthly_turnover"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Strategy": st.column_config.TextColumn(width="medium"),
                "Exp. return": st.column_config.TextColumn(help="Expected annual return (model estimate)"),
                "Exp. vol.": st.column_config.TextColumn(help="Expected annual volatility (model estimate)"),
                "Backtest": st.column_config.TextColumn(help="Total return over the held-out test period"),
                "Ann. return": st.column_config.TextColumn(help="Backtest return converted to an annual rate (Part 4)"),
                "Ann. vol.": st.column_config.TextColumn(help="Annualized volatility of daily returns in the backtest (Part 4)"),
                "Max DD": st.column_config.TextColumn(help="Maximum drawdown: worst peak-to-trough fall"),
                "Sharpe": st.column_config.TextColumn(help="Return per unit of risk in the backtest"),
                "Turnover / mo": st.column_config.TextColumn(help="Average share of the portfolio traded at each monthly rebalance"),
            })
            st.caption(f"Expected = model estimate from historical data. Backtest = {f['test_window']}, "
                       f"risk-free rate {f['risk_free_rate']}.")

        with t_why:
            info = p["profile_info"]
            dropped = p["comparison"][p["strategy"]]["dropped_not_buyable"]
            excl = ", ".join(info["exclusion_reasons"]) or "none"
            st.markdown(
                "**Why this strategy**\n\n" + "".join(f"- {r[0].upper() + r[1:]}.\n" for r in f.get("why_this_strategy", []))
                + "\n**How your answers shaped it**\n\n"
                f"- **Your risk comfort** ({info['stated_risk']}) sets a volatility cap of **{f['volatility_cap']}**.\n"
                f"- **Your time horizon** (~{info['horizon_years']} years) requires at least **{f['min_fixed_income']}** in bonds.\n"
                f"- No single asset above **{f['max_weight_per_asset']}**, no sector above 40%.\n"
                f"- **Excluded** for your values: {excl}.\n"
                + "".join(f"- **Adjusted:** {o}.\n" for o in info["overrides"])
                + (f"- Not buyable in whole units with your budget: {', '.join(dropped)}.\n" if dropped else "")
                + f"- Estimated from weekly data {p['estimation_window'][0]} to {p['estimation_window'][1]}."
            )


# questions whose "(option, option, ...)" list is shown as quick-reply chips instead
_CHIP_QUESTIONS = {profiling_agent.QUESTION_TEXT[k] for k in QUICK_REPLIES if k != "sectors"}


def display_text(text: str) -> str:
    if text in _CHIP_QUESTIONS:
        return re.sub(r"\s*\([^()]*\)\s*$", "", text)
    return text


def split_footer(text: str) -> tuple[str, str | None]:
    """Separates the code-appended italic footer (price date + disclaimer) so it renders under a divider."""
    body, sep, last = text.rstrip().rpartition("\n\n")
    if sep and last.startswith("_") and last.endswith("_") and len(last) > 2:
        return body, last[1:-1]
    return text, None


def render_message(m: dict, idx: int, latest_portfolio_idx: int | None = None):
    user = m["role"] == "user"
    with st.chat_message(m["role"], avatar=USER_AVATAR if user else BOT_AVATAR):
        if user:
            with st.container(key=f"umsg_{idx}"):
                st.markdown(m["content"])
            return
        with st.container(key=f"bmsg_{idx}"):
            body, foot = split_footer(m["content"])
            st.markdown(display_text(body))
            if m.get("metrics"):
                mt = m["metrics"]
                st.html(f'<div class="mcards-title">{mt["title"]}</div>' + metric_cards(mt["items"], small=True)
                        + f'<div class="mcards-note">{mt["note"]}</div>')
            if m.get("portfolio") and idx != latest_portfolio_idx:
                with st.expander("Show this earlier portfolio"):
                    render_recommendation(m["portfolio"], key=str(idx), header=False)
            if foot:
                st.html(f'<div class="msg-foot">{foot}</div>')
        if m.get("portfolio") and idx == latest_portfolio_idx:   # full panel right under the recommendation
            with st.container(key=f"panel_{idx}"):
                render_recommendation(m["portfolio"], key=str(idx))


# ------------------------------------------
# 3. LAYOUT
# ------------------------------------------
gs = st.session_state.graph_state
answers = gs["prof_state"]["answers"]
checked = profiling_agent.effective_answers(answers)
n_done = sum(v is not None for v in checked.values())
chat = st.session_state.chat
latest_idx = next((i for i in range(len(chat) - 1, -1, -1) if chat[i].get("portfolio")), None)


def _short(k, v):
    if v in (None, ""):
        return None
    if k in ("invest_amount", "target", "cash_needed_now") and isinstance(v, (int, float)):
        return _money0(v)
    if k == "income":
        v = re.sub(r"\$(\d+),\d{3}", r"$\1k", str(v))
        return re.sub(r"\s*[–-]\s*\$", "–", v)
    if k == "desc":
        return str(v).replace(" risk", "")
    if k == "sectors" and gs.get("portfolio"):
        v = ", ".join(gs["portfolio"]["profile_info"]["exclusion_reasons"]) or "None"
    if isinstance(v, str) and v[:1].islower():
        return v[0].upper() + v[1:]
    return v


SB_SECTIONS = [
    ("INVESTOR PROFILE", [("age", "Age"), ("employment", "Employment"), ("income", "Income"),
                          ("num_dependants", "Dependants"), ("exp", "Experience")]),
    ("INVESTMENT GOALS", [("invest_amount", "Budget"), ("objective_str", "Goal"), ("target", "Target"),
                          ("target_years", "Horizon"), ("desc", "Risk")]),
    ("PREFERENCES", [("cash_needed_now", "Cash needed now"), ("principal_protection", "Principal protection"),
                     ("other_assets", "Other savings"), ("sectors", "Exclusions")]),
]

with st.sidebar:
    html = (f'<div class="brand"><div class="logo"></div><div><div class="name">Robo-Advisor</div>'
            '<div class="tag">INTELLIGENT INVESTING</div></div></div>')
    if gs["phase"] == "profiling":
        pct = 100 * n_done / len(checked)
        html += (f'<div class="status"><span class="dot pending"></span>Profile {n_done} of {len(checked)} answered</div>'
                 f'<div class="pbar"><div style="width:{pct:.0f}%"></div></div>')
    else:
        html += '<div class="status"><span class="dot"></span>Profile complete</div>'
    for title, fields in SB_SECTIONS:
        rows = [(lab, _short(k, answers.get(k))) for k, lab in fields]
        rows = [(lab, v) for lab, v in rows if v is not None and not (lab == "Target" and v == "n/a")]
        if rows:
            html += f'<div class="sb-section">{title}</div>' + "".join(
                f'<div class="sb-row"><span>{lab}</span><span>{v}</span></div>' for lab, v in rows)
    if gs["check"]:
        status = gs["check"]["status"]
        label, cls = {"APPROVE": ("✓ Approved", "ok"), "REVISE": ("Needs revision", "warn"),
                      "CLARIFY": ("Needs clarification", "warn"), "BLOCK": ("Blocked", "bad")}[status]
        html += (f'<div class="sb-rule"></div><div class="sb-title">Compliance review</div>'
                 f'<span class="pill {cls}">{label}</span>'
                 + "".join(f'<div class="sb-issue">{i}</div>' for i in gs["check"]["issues"][:3]))
    html += '<div class="sb-rule"></div>'
    st.html(html)
    if st.button("Start over", icon=":material/restart_alt:", width="stretch"):
        init_state()
        st.rerun()
    with st.expander("Agent trace (LangGraph)"):
        st.json(gs["pipeline_log"][-12:], expanded=False)

if latest_idx is not None:
    st.html(f"""
<div class="head-row"><div><div class="eyebrow">YOUR ROBO-ADVISOR</div>
  <div class="page-title">Your investment plan</div>
  <div class="page-sub">Your recommended portfolio is in the conversation below; ask follow-up questions at the bottom.</div>
</div></div>
<div class="disclaimer-bar"><b>Disclaimer:</b> {DISCLAIMER}</div>""")
else:
    st.html(f"""
<div class="head-row"><div><div class="eyebrow">GET STARTED</div>
  <div class="page-title">Build your investment portfolio</div>
  <div class="page-sub">Answer a few questions and get a personalized portfolio of Canadian and US stocks, ETFs and bonds.</div>
</div></div>
<div class="disclaimer-bar"><b>Disclaimer:</b> {DISCLAIMER}</div>""")
    if len(chat) <= 2:
        st.html("""
<div class="steps">
  <div class="step"><div class="n">1</div><b>Tell us about you</b><span>Your situation, goal, timeline and comfort with risk.</span></div>
  <div class="step"><div class="n">2</div><b>We optimize</b><span>A portfolio built and checked against your limits.</span></div>
  <div class="step"><div class="n">3</div><b>Get your plan</b><span>Exact units to buy, charts and past performance.</span></div>
</div>""")

for i, m in enumerate(chat):
    render_message(m, i, latest_idx)

# quick replies for the current multiple-choice question
key = current_question_key(gs)
if key in QUICK_REPLIES:
    with st.container(horizontal=True, gap="small", key="quick_replies"):
        for opt in QUICK_REPLIES[key]:
            st.button(opt, key=f"qr_{key}_{opt}_{len(st.session_state.chat)}", on_click=queue_message, args=(opt,))

placeholder = {"done": "Ask about your portfolio...",
               "failed": "Ask a question, or click Start over...",
               "clarify": "Reply to confirm, or click Start over..."}.get(gs["phase"], "Type your answer...")
user_msg = st.chat_input(placeholder)
if st.session_state.pending:
    user_msg, st.session_state.pending = st.session_state.pending, None

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    render_message(st.session_state.chat[-1], len(st.session_state.chat) - 1, latest_idx)
    building = gs["phase"] == "profiling" and n_done >= len(checked) - 1
    try:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Building and checking your portfolio…" if building else "Thinking…"):
                new_gs = robo_graph.run_turn(gs, user_msg)
    except Exception as e:  # e.g. Gemini 429 quota -> keep the previous graph state so the user can retry
        st.session_state.chat.pop()
        st.error(f"The language model is unavailable right now ({type(e).__name__}). Please resend your message.\n\n{e}",
                 icon=":material/cloud_off:")
        st.stop()
    st.session_state.graph_state = new_gs
    for r in new_gs["replies"]:
        st.session_state.chat.append({"role": "assistant", **r})
    if new_gs["phase"] != "profiling":
        save_log()
    st.rerun()
