# Quick check that the Claude API key in .env works (one short, cheap request).
#   python test-llm.py
import llm

print(llm.SETTINGS)
print(llm.generate_text("Reply with 'API test successful' and one short sentence about Toronto.", effort="low"))
