import os
import asyncio
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY not found in .env")
    exit(1)

print(f"🔑 Found API Key: {OPENROUTER_API_KEY[:4]}...{OPENROUTER_API_KEY[-4:]}")

client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

async def test_perplexity():
    print("Testing Perplexity Sonar via OpenRouter...")
    try:
        response = await client.chat.completions.create(
            model="perplexity/sonar",
            messages=[{
                "role": "user",
                "content": "Hello, are you working?"
            }],
        )
        print("✅ Success!")
        print(response.choices[0].message.content)
    except openai.APIConnectionError as e:
        print(f"❌ Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"❌ Rate Limit / Credit Error: {e}")
    except openai.APIStatusError as e:
        print(f"❌ API Status Error: {e.status_code}")
        print(f"Response: {e.response}")
        print(f"Message: {e.message}")
    except Exception as e:
        print(f"❌ Unknown Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_perplexity())
