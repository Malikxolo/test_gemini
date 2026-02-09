"""
Quick test for Groq LLM Client
Run: python new_system/test_groq.py
"""

import asyncio
import sys
import pathlib

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment from root .env
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from new_system.llm_client import LLMClient


async def test_groq():
    print("=" * 50)
    print("Testing Groq LLM Client with Llama 3.3 70B")
    print("=" * 50)
    
    llm = LLMClient()
    print(f"\nModel: {llm.model}")
    print(f"API Key set: {'Yes' if llm.api_key else 'No'}")
    
    # Test 1: Warmup connection
    print("\n1. Warming up connection...")
    await llm.warmup()
    
    # Test 2: Test silence behavior (should output [SILENT])
    print("\n2. Testing silence behavior (general greeting)...")
    msgs = [{"role": "user", "content": "Hello everyone, let's start the meeting"}]
    full_response = ""
    async for chunk in llm.generate_response(msgs, use_tools=False):
        if chunk["type"] == "text":
            full_response += chunk["content"]
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "done":
            print()
    
    print(f"\nFull response: {repr(full_response)}")
    if "[SILENT]" in full_response:
        print("✅ PASS: Model correctly stayed silent!")
    else:
        print("⚠️  Model responded when it should have been silent")
    
    # Test 3: Test wake word response
    print("\n3. Testing wake word response...")
    msgs = [{"role": "user", "content": "Gemini, what is 2 plus 2?"}]
    full_response = ""
    async for chunk in llm.generate_response(msgs, use_tools=False):
        if chunk["type"] == "text":
            full_response += chunk["content"]
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "done":
            print()
    
    print(f"\nFull response: {repr(full_response)}")
    if "[SILENT]" not in full_response and len(full_response) > 5:
        print("✅ PASS: Model responded to wake word!")
    else:
        print("⚠️  Model stayed silent when it should have responded")
    
    await llm.close()
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_groq())
