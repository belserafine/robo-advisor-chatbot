# Robo-Advisor Chatbot (MIE1624 Course Project, Group 8: Part 5)

A Streamlit chatbot that runs the Part 1 questionnaire as a conversation, calls the Part 3 optimizer and the
Part 4 backtest **during the conversation**, and shows the recommended portfolio with charts and a
plain-language explanation. Every number shown to the client comes from Python code; a compliance agent
reviews each recommendation before the client sees it.

> For educational purposes only; not financial advice. Past performance does not guarantee future results.

---

## 1. Quick start

Requires Python 3.12.

```bash
git clone https://github.com/belserafine/robo-advisor-chatbot.git
cd robo-advisor-chatbot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env              # your own key; .env is git-ignored
streamlit run app.py                                    # opens http://localhost:8501
```

On Google Colab, store the key in **Colab Secrets** and load it with
`os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")`.
Never hard-code a key.

Terminal-only chat (no UI), which also prints the LangGraph diagram in Mermaid format:

```bash
python robo_graph.py
```

---

## 2. How it works

Each client message is one run of a **LangGraph** state graph ([robo_graph.py](robo_graph.py)):

```
client message ──► router ──┬─► profiling ──(questionnaire complete)──► portfolio ◄──┐
                            ├─► currency  (foreign amount → CAD, client confirms)    │ REVISE: rewrite
                            ├─► assist    (question during the questionnaire)        │ (max 2), then
                            ├─► switch_strategy / update_profile ──► portfolio       │ code template
                            └─► followup  (tools + grounded answer after a recommendation)
                                                                    portfolio ──► compliance
                                                 compliance ──APPROVE──► respond   (only path that shows a portfolio)
                                                 compliance ──CLARIFY──► withhold  (asks the client, then re-checks)
                                                 compliance ──BLOCK────► withhold  (recommendation not shown)
```

| Agent / component | File | What it does |
|---|---|---|
| Router | [router.py](router.py) | Classifies each message: questionnaire answer, strategy switch, profile change, data question, out-of-scope, emotional or unrealistic. Detects foreign currencies with code. |
| **Profiling agent** | [profiling_agent.py](profiling_agent.py) | Asks the 16 questionnaire questions, extracts answers from free text, checks for contradictions. |
| **Portfolio agent** | [portfolio_agent.py](portfolio_agent.py) | Maps the answers to model inputs, runs the optimizer and backtest, and writes the explanation using only code-produced numbers (`facts`). |
| **Compliance agent** | [compliance_agent.py](compliance_agent.py) | Runs deterministic checks (budget, weights, units, caps, exclusions, volatility, disclaimer, claims vs. tool output, number grounding), then a Claude reviewer. Returns APPROVE / REVISE / CLARIFY / BLOCK. |
| Optimizer + backtest | [optimizer.py](optimizer.py) | Port of the Part 3/4 notebook: Strategy 1 (Markowitz, sample means), Strategy 2 (reverse optimization + Markowitz), equal-weight benchmark, whole-unit MILP, monthly-rebalanced backtest. |
| Tools | [portfolio_tools.py](portfolio_tools.py) | 2022 stress test, ESG-exclusion cost, fractional units, small-budget diagnostics, asset statistics, FX rates. |
| Data | [market_data.py](market_data.py) | Loads `data/part2/` (CAD prices, returns, μ, Q) with integrity checks. |
| UI | [app.py](app.py) | Streamlit chat, sidebar profile, portfolio panel (Orders / Allocation / Past performance / Compare / Why this?), Plotly charts. |
| LLM access | [llm.py](llm.py) | The only module that calls the model provider. |

**Self-correction loop.** A REVISE result sends the reviewer's notes back to the Portfolio agent, which
rewrites the draft (up to 2 times) before falling back to a code-written template. A CLARIFY result asks
the client a yes/no question and re-checks. Follow-up answers whose numbers are not in the tool output
are regenerated once with the offending numbers named, and any remaining lines with such numbers are
removed.

---

## 3. LLM and reproducibility settings

| Setting | Value |
|---|---|
| Provider / model | Anthropic, `claude-haiku-5-5` (override with the `LLM_MODEL` environment variable) |
| Thinking | adaptive (model default) |
| Effort | `low` for extraction, classification and checks; `medium` for explanations |
| Temperature / top-p | not configurable on this model (provider defaults) |
| Random seed | optimizer and data are deterministic; Part 3 uses `SEED = 1624` |
| System prompts | `DRAFT_SYSTEM_PROMPT` and `ANSWER_SYSTEM_PROMPT` in [portfolio_agent.py](portfolio_agent.py); `REVIEW_PROMPT` in [compliance_agent.py](compliance_agent.py); `ROUTER_PROMPT` in [router.py](router.py); extraction and contradiction prompts in [profiling_agent.py](profiling_agent.py) |

**Deterministic vs. not.** Portfolios, units, weights and every metric come from code and are identical on
every run. The wording of explanations comes from the LLM and can differ between runs.

**Offline mode (default).** Prices come from `data/part2/` (latest price 2026-11-18). Currency conversion
uses the saved snapshot `data/fx_snapshot.json` (2026-11-26). Set `LIVE_FX=1` to fetch live FX rates from
Yahoo Finance instead. Use the default offline mode for the mystery client, so results can be reproduced.

---

## 4. Tests and evaluation

```bash
python validation.py                           # Part 4 correctness tests (budget, constraints, closed-form check)
python compliance_agent.py                     # compliance unit tests, no API call (11 injected errors)
python -m evaluation.run_scripts               # the 20 scripted conversations (about 15-20 min, uses the API)
python -m evaluation.run_scripts 01 18         # selected scripts only (no summary table)
python -m evaluation.checker_injection         # error injection, deterministic rules only
python -m evaluation.checker_injection --full  # error injection, full checker (rules + Claude reviewer)
```

| File | Contents |
|---|---|
| [evaluation/scripts.py](evaluation/scripts.py) | The 20 scripts (user messages verbatim from the test-script document), persona cards and known answers. **Use the same file for the baselines** so the scripts stay fixed across versions. |
| [evaluation/metrics.py](evaluation/metrics.py) | Field-by-field profile accuracy, numbers not produced by code, and orders compared with the Part 3 file. |
| `evaluation/results/summary_chatbot.md` | Summary table of the 20 scripts |
| `evaluation/results/chatbot/NN.json` | Full log of each test conversation: transcript, extracted profile, agent trace, numbers check |
| `evaluation/results/checker_injection_full.md` | Error-injection results (15 error types × 3 personas) |
| `evaluation/results/injection_drafts.json` | The real drafts that the errors were inserted into |

**Final results** (run 2026-11-26 with the submitted code):

| Metric | Result |
|---|---|
| Profile fields captured correctly (scripts 01-09) | 140/144 (97.2%) |
| Numbers in replies not produced by code | 0/2,131 (0%) |
| Injected errors caught by the checker | 45/45 (100%) |

---

## 5. Repository layout

```
app.py  robo_graph.py  router.py  profiling_agent.py  portfolio_agent.py  compliance_agent.py
optimizer.py  portfolio_tools.py  market_data.py  llm.py  validation.py
baseline-a/        Baseline A (same LLM, no tools or data) - notebook, script, results, logs
data/part2/        Part 2 CAD prices, returns, mu, Q (in-sample 2021-04 to 2026-03, test 2026-04-01 to 2026-11-18)
data/part3/        Part 3 weights / units / config used to verify the port
data/fx_snapshot.json   saved FX rates (offline mode)
evaluation/        scripted-conversation runner, metrics, error injection, results
.streamlit/config.toml  UI theme
```
