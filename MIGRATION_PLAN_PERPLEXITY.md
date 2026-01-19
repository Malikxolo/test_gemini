# Migration Plan: Google Web Search → Perplexity Sonar via OpenRouter

## Executive Summary

This document outlines the plan to replace the expensive Google web search tool in the Gemini Live Audio Streaming application with a more cost-effective Perplexity Sonar web search via OpenRouter.

---

## 1. Current System Architecture Analysis

### 1.1 How It Currently Works

Your application is a **real-time bidirectional voice AI assistant** with the following architecture:

```
┌─────────────────┐     WebSocket (Binary PCM)     ┌──────────────────┐
│    Frontend     │ ◄──────────────────────────────► │     Backend      │
│   (client.js)   │                                 │   (server.py)    │
│                 │                                 │                  │
│  - Mic capture  │                                 │  - FastAPI       │
│  - 16kHz input  │                                 │  - Google GenAI  │
│  - 24kHz output │                                 │    SDK           │
│  - PCM audio    │                                 │  - Gemini Live   │
└─────────────────┘                                 └────────┬─────────┘
                                                              │
                                                              │ SDK (NOT WebSocket)
                                                              ▼
                                                    ┌──────────────────┐
                                                    │  Gemini Live API │
                                                    │                  │
                                                    │  - Native Audio  │
                                                    │  - VAD (Voice    │
                                                    │    Activity Det) │
                                                    │  - google_search │
                                                    │    tool          │
                                                    └──────────────────┘
```

### 1.2 Key Components

| Component      | Technology                                      | Purpose                                        |
| -------------- | ----------------------------------------------- | ---------------------------------------------- |
| **Frontend**   | Vanilla JS + AudioWorklet                       | Captures 16kHz PCM audio, plays 24kHz response |
| **Backend**    | FastAPI + google-genai SDK                      | WebSocket server, bridges client to Gemini     |
| **AI Model**   | `gemini-2.5-flash-native-audio-preview-12-2025` | Native audio processing model                  |
| **Connection** | `client.aio.live.connect()`                     | SDK-based connection (NOT raw WebSocket)       |
| **Web Search** | `{"google_search": {}}`                         | Built-in Google Grounding tool                 |

### 1.3 Current Google Search Configuration

```python
config = {
    "response_modalities": ["AUDIO"],
    "tools": [{"google_search": {}}],  # ← THIS IS THE EXPENSIVE PART
    # ... other config
}
```

**How it works:**

1. When Gemini detects a query needing real-time info, it calls `google_search` internally
2. This is a **function calling** mechanism built into Gemini
3. Results are automatically incorporated into the response
4. You're billed per Google search query (expensive)

---

## 2. The Problem

### 2.1 Google Search Costs

Google's Grounding/Search tool pricing is high:

- Charged per search query
- No control over when searches are triggered
- Costs scale with usage

### 2.2 Why Switch to Perplexity Sonar?

| Feature           | Google Search      | Perplexity Sonar (via OpenRouter) |
| ----------------- | ------------------ | --------------------------------- |
| **Input Tokens**  | N/A (bundled)      | $1/M tokens                       |
| **Output Tokens** | N/A (bundled)      | $1/M tokens                       |
| **Request Fee**   | High               | $5/1K requests (~$0.005/request)  |
| **Control**       | No control         | Full control over when to search  |
| **Latency**       | ~1-2s              | ~1.5s avg                         |
| **Model**         | `perplexity/sonar` | Lightweight, fast, affordable     |

**Cost Comparison Example (1000 searches):**

- Google Search: $$$$ (variable, expensive)
- Perplexity Sonar: ~$5 (request fee) + ~$0.002-0.01 (tokens) ≈ **$5-10 total**

---

## 3. Available Perplexity Models via OpenRouter

| Model                            | Best For              | Pricing                           |
| -------------------------------- | --------------------- | --------------------------------- |
| **`perplexity/sonar`** ⭐        | Fast, lightweight Q&A | $1/M in, $1/M out, $5/K requests  |
| `perplexity/sonar-pro`           | Complex queries       | $3/M in, $15/M out, $5/K searches |
| `perplexity/sonar-reasoning`     | Chain-of-thought      | Higher cost                       |
| `perplexity/sonar-deep-research` | Exhaustive research   | $2/M in, $8/M out + search costs  |

**Recommendation: `perplexity/sonar`** - It's the fastest, cheapest, and designed for quick Q&A which matches your voice assistant use case.

---

## 4. Integration Challenges & Solutions

### 4.1 The Core Challenge

**Problem:** Gemini Live API uses a **native audio-to-audio pipeline** with built-in function calling. You can't simply swap Google Search for Perplexity because:

1. Gemini's `google_search` is a **native tool** that executes automatically
2. Gemini handles audio input → text understanding → tool call → audio output seamlessly
3. Perplexity is a **text-based API** that returns text, not audio

### 4.2 Proposed Solution: Hybrid Architecture

```
┌─────────────────┐                              ┌──────────────────┐
│    Frontend     │ ◄─────────────────────────► │     Backend      │
└─────────────────┘                              └────────┬─────────┘
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    │                                     │                                     │
                    ▼                                     ▼                                     ▼
          ┌──────────────────┐              ┌──────────────────────┐              ┌──────────────────┐
          │  Gemini Live API │              │  Custom Web Search   │              │  Perplexity API  │
          │                  │              │  Function Handler    │              │  (via OpenRouter)│
          │  - Audio I/O     │◄────────────►│                      │─────────────►│                  │
          │  - NO google_    │  Tool Call   │  - Intercepts tool   │   HTTP/REST  │  - Text search   │
          │    search tool   │  Response    │    calls             │              │  - Citations     │
          │  - Custom func   │              │  - Calls Perplexity  │              │                  │
          └──────────────────┘              └──────────────────────┘              └──────────────────┘
```

### 4.3 How It Will Work

1. **Remove** `{"google_search": {}}` from Gemini config
2. **Add** a custom function definition for `web_search` that Gemini can call
3. **Intercept** Gemini's tool calls in the response stream
4. **Call Perplexity** via OpenRouter when web search is requested
5. **Return** search results back to Gemini as tool response
6. **Gemini** incorporates results and generates audio response

---

## 5. Technical Implementation Plan

### 5.1 OpenRouter Configuration

**OpenRouter API Key:** `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`

**Base URL:** `https://openrouter.ai/api/v1`

**Model:** `perplexity/sonar`

### 5.2 Code Changes Required

#### 5.2.1 Update `.env`

```env
GOOGLE_API_KEY=your_existing_google_key
OPENROUTER_API_KEY=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
```

#### 5.2.2 Update `requirements.txt`

```txt
fastapi
uvicorn
python-dotenv
websockets
google-genai
openai          # ← ADD: For OpenRouter (OpenAI-compatible SDK)
httpx           # ← ADD: For async HTTP requests (alternative)
```

#### 5.2.3 Backend Changes (server.py)

**Key modifications:**

1. **Replace Google Search tool with custom function:**

```python
# BEFORE (current)
config = {
    "tools": [{"google_search": {}}],
    # ...
}

# AFTER (new)
config = {
    "tools": [{
        "function_declarations": [{
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need real-time data, news, or information that may have changed since your training.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }]
    }],
    # ...
}
```

2. **Add Perplexity search function:**

```python
import openai

async def search_with_perplexity(query: str) -> str:
    """Call Perplexity Sonar via OpenRouter for web search"""
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    response = await client.chat.completions.create(
        model="perplexity/sonar",
        messages=[{
            "role": "user",
            "content": query
        }],
        max_tokens=500
    )

    return response.choices[0].message.content
```

3. **Handle tool calls in Gemini response stream:**

```python
async for response in turn:
    # Check for tool calls
    if response.server_content and response.server_content.model_turn:
        for part in response.server_content.model_turn.parts:
            if part.function_call:
                if part.function_call.name == "web_search":
                    query = part.function_call.args.get("query")
                    # Call Perplexity
                    search_result = await search_with_perplexity(query)
                    # Send result back to Gemini
                    await session.send_tool_response(
                        function_responses=[{
                            "name": "web_search",
                            "response": {"result": search_result}
                        }]
                    )
```

### 5.3 Flow Diagram (After Migration)

```
User speaks → Gemini (Audio) → Needs web info?
                                      │
                         ┌────────────┼────────────┐
                         │ NO                      │ YES
                         ▼                         ▼
                   Direct audio            Gemini calls web_search()
                   response                        │
                                                   ▼
                                          Backend intercepts
                                                   │
                                                   ▼
                                          Call Perplexity Sonar
                                          via OpenRouter
                                                   │
                                                   ▼
                                          Return text results
                                                   │
                                                   ▼
                                          Send to Gemini as
                                          tool_response
                                                   │
                                                   ▼
                                          Gemini generates
                                          audio with results
                                                   │
                                                   ▼
                                          Audio sent to user
```

---

## 6. Important SDK Considerations

### 6.1 Confirmed: You're Using SDK, NOT WebSocket

Your code uses:

```python
async with client.aio.live.connect(model=MODEL, config=config) as session:
```

This is the **Google GenAI SDK** which is more stable than raw WebSocket. ✅

### 6.2 Tool Response in SDK

The SDK provides methods for tool responses. Need to verify exact method name:

- Likely: `session.send_tool_response()`
- Or: `session.send_realtime_input()` with tool response format
- **Action needed:** Check google-genai SDK docs for exact API

### 6.3 Gemini Native Audio Model Compatibility

The model `gemini-2.5-flash-native-audio-preview-12-2025` supports:

- ✅ Custom function definitions
- ✅ Function calling
- ✅ Tool responses
- ✅ Audio output after tool use

---

## 7. Potential Issues & Mitigations

### 7.1 Latency Increase

| Concern                        | Mitigation                                            |
| ------------------------------ | ----------------------------------------------------- |
| Extra round-trip to Perplexity | Use `perplexity/sonar` (fastest model, ~1.5s latency) |
| Sequential calls               | Consider parallel search + "Let me check" audio       |

### 7.2 Audio Continuity

**Current behavior:** "Main check kar rahi hu" fills silence during Google search

**New behavior:** Need to replicate this:

1. When Gemini calls `web_search`, it can still generate filler audio
2. Or we inject a pre-recorded filler while waiting

### 7.3 Error Handling

- OpenRouter API failures → Fallback to generic response
- Rate limiting → Implement exponential backoff
- Network issues → Timeout with user-friendly message

---

## 8. Testing Plan

### Phase 1: Unit Testing

- [ ] Test Perplexity API call independently
- [ ] Verify response format and content quality
- [ ] Measure latency

### Phase 2: Integration Testing

- [ ] Test custom function definition with Gemini
- [ ] Test tool response flow
- [ ] Verify audio generation after tool use

### Phase 3: End-to-End Testing

- [ ] Full voice conversation with web search
- [ ] Compare quality vs Google Search
- [ ] Measure total response time

---

## 9. Rollback Plan

If issues arise:

1. Revert to `{"google_search": {}}` in config
2. Remove Perplexity integration code
3. Remove OpenRouter env variable

The modular design allows easy rollback.

---

## 10. Cost Estimation

### Current (Google Search)

- Unknown/high per search

### After Migration (Perplexity Sonar)

- **Per 1000 searches:**
  - Request fee: $5
  - Input tokens (~100 tokens avg): $0.0001
  - Output tokens (~300 tokens avg): $0.0003
  - **Total: ~$5.01 per 1000 searches**

### Estimated Monthly Savings

If you make 10,000 searches/month:

- Perplexity: ~$50
- Google: Likely $100-500+
- **Savings: 50-90%**

---

## 11. Files to Modify

| File                       | Changes                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `backend/.env`             | Add `OPENROUTER_API_KEY`                                           |
| `backend/requirements.txt` | Add `openai` package                                               |
| `backend/server.py`        | Replace google_search with custom function, add Perplexity handler |
| `frontend/*`               | No changes needed                                                  |

---

## 12. Action Items

1. [ ] **Review this plan** - Confirm approach
2. [ ] **Verify SDK API** - Check `google-genai` docs for tool response methods
3. [ ] **Test OpenRouter key** - Verify the API key works
4. [ ] **Implement changes** - After approval
5. [ ] **Test thoroughly** - All scenarios
6. [ ] **Deploy** - Staged rollout

---

## 13. Questions for Review

1. **Latency tolerance:** Is ~1.5s additional latency for web search acceptable?
2. **Filler audio:** Should we keep "Main check kar rahi hu" behavior?
3. **Fallback:** If Perplexity fails, should we fallback to no search or a cached response?
4. **Search context:** Any specific domains or sources to prioritize/exclude?

---

## 14. Next Steps

Waiting for your review. Once approved, I will:

1. Update the `.env` file with OpenRouter API key
2. Add `openai` to requirements.txt
3. Modify `server.py` with the new tool definition and Perplexity handler
4. Test the integration

**Please confirm:**

- ✅ Overall approach looks good
- ✅ Cost savings are acceptable
- ✅ Any concerns about latency
- ✅ Ready to proceed with implementation

---

_Plan created: January 19, 2026_
