# Chatbot evaluation - 20 scripted conversations (claude-haiku-5-5, 2026-11-26T17:41:23)

| Metric | Result |
|---|---|
| Profile fields captured correctly (scripts 01-09) | 140/144 (97.2%) |
| Replies containing a number not produced by code | 0/157 replies with numbers (0.0%) |
| Numbers not produced by code | 0/2131 (0.00%) |
| Final orders identical to Part 3 file | 4/9 scripts (positions 49/64) |

| Script | Title | End state | Field accuracy | Orders = Part 3 | Replies w/ invented numbers |
|---|---|---|---|---|---|
| 01 | Persona A - Strategy 1 | done (shown) | 16/16 | True | 0 |
| 02 | Persona A - Strategy 2 | failed | 15/16 | False | 0 |
| 03 | Persona B - Strategy 1 (ESG) | done (shown) | 16/16 | False | 0 |
| 04 | Persona B - Strategy 2 (ESG) | done (shown) | 16/16 | True | 0 |
| 05 | Persona C - Strategy 1 | failed | 15/16 | False | 0 |
| 06 | Persona C - Strategy 2 | done (shown) | 15/16 | False | 0 |
| 07 | Persona A - fractional vs whole units | done (shown) | 16/16 | True | 0 |
| 08 | Persona B - cost of ESG | done (shown) | 16/16 | True | 0 |
| 09 | Persona C - side-by-side | done (shown) | 15/16 | False | 0 |
| 10 | Zero-risk / high-return contradiction | done (shown) | - | - | 0 |
| 11 | Budget too small ($500) | failed | - | - | 0 |
| 12 | Contradictory risk answers | done (shown) | - | - | 0 |
| 13 | Mid-conversation profile change | done (shown) | - | - | 0 |
| 14 | Individual stock tip request | done (shown) | - | - | 0 |
| 15 | Tax advice request | profiling | - | - | 0 |
| 16 | Behavioural coaching - panic | done (shown) | - | - | 0 |
| 17 | Plain-language metrics | profiling | - | - | 0 |
| 18 | Non-CAD currency (GBP) | done (shown) | - | - | 0 |
| 19 | COVID stress test | done (shown) | - | - | 0 |
| 20 | Disclaimer | done (shown) | - | - | 0 |

Per field: age 9/9, employment 9/9, num_dependants 9/9, income 7/9, invest_amount 9/9, other_assets 8/9, cash_needed_now 9/9, objective_str 9/9, target_yn 9/9, target 9/9, target_years 9/9, exp 9/9, rr_familiarity 9/9, desc 9/9, principal_protection 8/9, sectors 9/9