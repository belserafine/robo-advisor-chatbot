from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Reply with 'API test successful' and one short sentence about Toronto.",
)

print(response.text)

''' # to check list of available models
for m in client.models.list():
    print(m.name, "-> supports:", m.supported_actions)
'''