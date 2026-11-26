# Checker error injection - full (rules + Claude reviewer)

Caught **45/45** injected errors (45 by the intended rule); clean drafts blocked: 1/3.

| Persona | ID | Injected error | Where | Caught | Status | Issue codes |
|---|---|---|---|---|---|---|
| A | E00 | none (clean draft) | - | yes | REVISE | LLM_REVIEW |
| A | E01 | Weights quoted in the text do not sum to 100% | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| A | E02 | A holding's weight inconsistent with its units (weights do not sum to 100%) | tool result | yes | BLOCK | LLM_REVIEW, WEIGHT |
| A | E03 | Expected return in the text not produced by the code | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| A | E04 | Structured claim: expected return changed | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| A | E05 | Structured claim: wrong number of units for a holding | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| A | E06 | Invented projection in the text | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| A | E07 | Holds an asset the client excluded | constraint | yes | BLOCK | EXCLUDED, LLM_REVIEW |
| A | E08 | Single asset above the client's max weight | constraint | yes | BLOCK | ASSET_CAP, CLAIM_MISMATCH, LLM_REVIEW, OVER_BUDGET, RECONCILIATION |
| A | E09 | Portfolio volatility above the client's cap | constraint | yes | BLOCK | VOLATILITY |
| A | E10 | Bond floor / commodity cap violated | constraint | yes | BLOCK | CLASS_LIMIT |
| A | E11 | Holdings plus fees exceed the budget | constraint | yes | BLOCK | CLAIM_MISMATCH, OVER_BUDGET, RECONCILIATION |
| A | E12 | Disclaimer removed | draft text | yes | REVISE | DISCLAIMER |
| A | E13 | Price date removed | draft text | yes | CLARIFY | DATE_DISCLOSURE, LLM_REVIEW |
| A | E14 | Guarantee language added | draft text | yes | REVISE | GUARANTEE_LANGUAGE, LLM_REVIEW |
| A | E15 | Unsuitable: client needs principal protection | suitability | yes | CLARIFY | LLM_REVIEW, PRESERVATION |
| B | E00 | none (clean draft) | - | no | APPROVE | - |
| B | E01 | Weights quoted in the text do not sum to 100% | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| B | E02 | A holding's weight inconsistent with its units (weights do not sum to 100%) | tool result | yes | BLOCK | ASSET_CAP, REVIEW_FAILED, WEIGHT |
| B | E03 | Expected return in the text not produced by the code | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| B | E04 | Structured claim: expected return changed | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| B | E05 | Structured claim: wrong number of units for a holding | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| B | E06 | Invented projection in the text | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| B | E07 | Holds an asset the client excluded | constraint | yes | BLOCK | EXCLUDED, LLM_REVIEW |
| B | E08 | Single asset above the client's max weight | constraint | yes | BLOCK | ASSET_CAP, CLAIM_MISMATCH, LLM_REVIEW, OVER_BUDGET, RECONCILIATION |
| B | E09 | Portfolio volatility above the client's cap | constraint | yes | BLOCK | LLM_REVIEW, VOLATILITY |
| B | E10 | Bond floor / commodity cap violated | constraint | yes | BLOCK | CLASS_LIMIT, LLM_REVIEW |
| B | E11 | Holdings plus fees exceed the budget | constraint | yes | BLOCK | CLAIM_MISMATCH, LLM_REVIEW, OVER_BUDGET, RECONCILIATION |
| B | E12 | Disclaimer removed | draft text | yes | CLARIFY | DISCLAIMER, LLM_REVIEW |
| B | E13 | Price date removed | draft text | yes | REVISE | DATE_DISCLOSURE, LLM_REVIEW |
| B | E14 | Guarantee language added | draft text | yes | REVISE | GUARANTEE_LANGUAGE, LLM_REVIEW |
| B | E15 | Unsuitable: client needs principal protection | suitability | yes | CLARIFY | LLM_REVIEW, PRESERVATION |
| C | E00 | none (clean draft) | - | no | APPROVE | - |
| C | E01 | Weights quoted in the text do not sum to 100% | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| C | E02 | A holding's weight inconsistent with its units (weights do not sum to 100%) | tool result | yes | BLOCK | ASSET_CAP, LLM_REVIEW, WEIGHT |
| C | E03 | Expected return in the text not produced by the code | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| C | E04 | Structured claim: expected return changed | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| C | E05 | Structured claim: wrong number of units for a holding | draft claim | yes | REVISE | CLAIM_MISMATCH, LLM_REVIEW |
| C | E06 | Invented projection in the text | draft text | yes | REVISE | LLM_REVIEW, UNGROUNDED_NUMBER |
| C | E07 | Holds an asset the client excluded | constraint | yes | BLOCK | EXCLUDED, LLM_REVIEW |
| C | E08 | Single asset above the client's max weight | constraint | yes | BLOCK | ASSET_CAP, CLAIM_MISMATCH, LLM_REVIEW, OVER_BUDGET, RECONCILIATION |
| C | E09 | Portfolio volatility above the client's cap | constraint | yes | BLOCK | LLM_REVIEW, VOLATILITY |
| C | E10 | Bond floor / commodity cap violated | constraint | yes | BLOCK | CLASS_LIMIT, LLM_REVIEW |
| C | E11 | Holdings plus fees exceed the budget | constraint | yes | BLOCK | CLAIM_MISMATCH, LLM_REVIEW, OVER_BUDGET, RECONCILIATION |
| C | E12 | Disclaimer removed | draft text | yes | REVISE | DISCLAIMER, LLM_REVIEW |
| C | E13 | Price date removed | draft text | yes | REVISE | DATE_DISCLOSURE, LLM_REVIEW |
| C | E14 | Guarantee language added | draft text | yes | REVISE | GUARANTEE_LANGUAGE, LLM_REVIEW |
| C | E15 | Unsuitable: client needs principal protection | suitability | yes | CLARIFY | LLM_REVIEW, PRESERVATION |