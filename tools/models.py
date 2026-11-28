from google import genai
from src.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    print(model.name)


for model in client.models.list():
    print(f"Name: {model.name}")
    print(f"Supported actions: {model.supported_actions}")
    print("-" * 50)