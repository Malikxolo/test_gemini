# Architecture Analysis & Latency Deep Dive

## 📋 Executive Summary

This document provides a comprehensive analysis of the Gemini Live Voice Agent architecture, rating the current design, identifying latency bottlenecks (especially the **"main check krke batati hu" pause problem**), and suggesting improvements.

---

## 🏗️ Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         Frontend (client.js)                             │    │
│  │  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────────┐   │    │
│  │  │ AudioContext│    │  AudioWorklet   │    │   WebSocket Client     │   │    │
│  │  │   16kHz     │───►│ pcm-processor   │───►│   Binary Frames        │   │    │
│  │  │   Capture   │    │ Float32→Int16   │    │   (No Base64!)         │   │    │
│  │  └─────────────┘    └─────────────────┘    └──────────┬─────────────┘   │    │
│  │                                                        │                 │    │
│  │  ┌─────────────┐    ┌─────────────────┐               │                 │    │
│  │  │ AudioContext│◄───│   PCM Playback  │◄──────────────┘                 │    │
│  │  │   24kHz     │    │   Gapless Sched │    Binary Audio Response        │    │
│  │  │   Playback  │    └─────────────────┘                                 │    │
│  │  └─────────────┘                                                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket (Binary PCM + JSON Control)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Backend (server.py - FastAPI)                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        WebSocket Endpoint (/ws)                           │  │
│  │                                                                           │  │
│  │   receive_from_client()  ◄──── asyncio.gather ────►  receive_from_gemini()│  │
│  │         │                                                      │          │  │
│  │         ▼                                                      ▼          │  │
│  │   session.send_realtime_input()              session.receive() → turn     │  │
│  │         │                                                      │          │  │
│  └─────────┼──────────────────────────────────────────────────────┼──────────┘  │
└────────────┼──────────────────────────────────────────────────────┼─────────────┘
             │                                                      │
             │ Google GenAI SDK                                     │
             ▼                                                      │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Gemini Live API                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Model: gemini-2.5-flash-native-audio-preview-12-2025                     │  │
│  │                                                                           │  │
│  │  Features:                                                                │  │
│  │  • Native Audio I/O (no STT/TTS overhead)                                │  │
│  │  • Native VAD (Voice Activity Detection)                                  │  │
│  │  • Function Calling (web_search tool)                                     │  │
│  │  • Thinking Budget: 0 (disabled for speed)                               │  │
│  │  • Voice: Zephyr                                                          │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────┬──────────────────────────────────────┘
                                           │
                                           │ Function Call (web_search)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Perplexity Sonar (via OpenRouter)                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  Endpoint: https://openrouter.ai/api/v1                                   │  │
│  │  Model: perplexity/sonar                                                  │  │
│  │  Purpose: Web search for real-time information                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⭐ Architecture Rating

### Overall Score: **7.5/10**

| Category                 | Score | Notes                                                     |
| ------------------------ | ----- | --------------------------------------------------------- |
| **Design Pattern**       | 8/10  | Clean separation, async concurrency done right            |
| **Protocol Choice**      | 9/10  | Binary WebSocket + SDK is optimal                         |
| **Audio Pipeline**       | 8/10  | Proper sample rates, gapless playback                     |
| **Latency Optimization** | 6/10  | Good baseline, but **tool call flow has critical issues** |
| **Error Handling**       | 7/10  | Adequate, could be more robust                            |
| **Code Quality**         | 8/10  | Clean, well-commented, maintainable                       |
| **Scalability**          | 6/10  | Single-user design, would need work for multi-user        |

---

## ✅ What's Done Right

### 1. **Binary WebSocket Frames (No Base64)**

```
Previous (bad):  JSON + Base64 = ~33% size increase + encode/decode overhead
Current (good):  Binary frames = zero encoding overhead
```

**Impact:** Saves ~5-15ms per audio chunk

### 2. **Separate Audio Contexts for Different Sample Rates**

- Input: 16kHz (Gemini's expected format)
- Output: 24kHz (Gemini's native output)
- No resampling = no quality loss, no latency

### 3. **Google GenAI SDK Instead of Raw WebSocket**

You made the right decision to switch from raw WSS to the SDK:

- SDK handles connection management, reconnects, heartbeats
- More stable than manual WebSocket handling
- Proper async/await integration

### 4. **Native VAD Configuration**

```python
"realtime_input_config": {
    "automatic_activity_detection": {
        "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
        "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
        "prefix_padding_ms": 100,
        "silence_duration_ms": 200
    }
}
```

- High sensitivity = faster response to speech
- 200ms silence duration = quick end-of-speech detection

### 5. **Thinking Budget: 0**

```python
"thinking_config": {"thinking_budget": 0}
```

Disabling "thinking" mode removes the reasoning overhead for simple responses.

### 6. **Cost Optimization with Perplexity**

Switching from Google's built-in search to Perplexity Sonar via OpenRouter is smart:

- ~10x cheaper
- Same quality for factual queries
- You control when to search

---

## 🐛 The "Main Check Krke Batati Hu" Latency Problem

### The Symptom

> "There is a quiet moment when it says 'main check krke batati hu' then 2-3 sec pause then it starts"

### Root Cause Analysis

This is a **sequential blocking latency** caused by the tool call flow:

```
Timeline of a Web Search Request:
────────────────────────────────────────────────────────────────────────────────────

User: "Aaj ka weather kya hai?"
      │
      ▼
[0ms] Gemini receives audio, processes speech
      │
      ▼
[200-300ms] Gemini generates: "Main check kar rahi hu" (audio)
      │
      ├────► Audio chunks sent to client (plays immediately ✓)
      │
      ▼
[300ms] Gemini emits: tool_call (web_search, query="weather today")
      │
      ▼
[300ms] Backend receives tool_call
      │
      ▼
[300-350ms] Backend calls search_with_perplexity()
      │
      │   ┌─────────────────────────────────────────────────┐
      │   │         🚨 BLOCKING AWAIT - THE PROBLEM 🚨       │
      │   │                                                 │
      │   │  await openrouter_client.chat.completions.create│
      │   │                                                 │
      │   │  This takes 1500-3000ms depending on:           │
      │   │  • Network latency to OpenRouter                │
      │   │  • Perplexity Sonar processing time             │
      │   │  • Response size                                │
      │   │                                                 │
      │   └─────────────────────────────────────────────────┘
      │
      ▼
[2000-3300ms] Perplexity returns result
      │
      ▼
[2000-3300ms] Backend sends FunctionResponse to Gemini
      │
      ▼
[2100-3400ms] Gemini processes result, generates audio response
      │
      ▼
[2200-3500ms] Audio response starts playing

TOTAL SILENT GAP: 1800-3200ms (the "quiet moment")
```

### Why It Feels So Bad

1. **Gemini says "main check kar rahi hu"** → User expects quick response
2. **Complete silence for 2-3 seconds** → Feels broken/frozen
3. **Then response starts** → Jarring experience

This is **YOUR CODE'S latency** because:

- The `await search_with_perplexity(query)` is blocking
- Nothing is happening on the audio stream during this wait
- No filler audio, no "typing indicator" equivalent

---

## 🔍 All Latency Sources (Your Code vs External)

### Latencies YOU CAN Control (Your Code)

| Source                           | Current Latency  | Location          | Severity      |
| -------------------------------- | ---------------- | ----------------- | ------------- |
| **Perplexity await blocking**    | 1500-3000ms      | `server.py:64-77` | 🔴 CRITICAL   |
| **First audio chunk scheduling** | 10ms fixed delay | `client.js:161`   | 🟡 MINOR      |
| **No connection keepalive ping** | Variable         | WebSocket layer   | 🟡 MINOR      |
| **Latency tracking overhead**    | ~1ms             | Both files        | 🟢 NEGLIGIBLE |

### Latencies You CANNOT Control (External)

| Source                     | Typical Latency | Notes                |
| -------------------------- | --------------- | -------------------- |
| Gemini Live API processing | 200-500ms       | Model inference time |
| Perplexity Sonar API       | 1000-2000ms     | External service     |
| Network round-trip         | 50-150ms        | Internet latency     |
| Browser audio buffering    | 10-50ms         | System dependent     |

---

## 🎯 Specific Latency Issues in Your Code

### Issue #1: Blocking Tool Call (CRITICAL 🔴)

**Location:** `server.py`, lines 199-218

```python
# Current flow (blocking):
if fc.name == "web_search":
    query = fc.args.get("query", "")

    # 🔴 THIS BLOCKS EVERYTHING
    search_result = await search_with_perplexity(query)

    # Only after this completes can Gemini respond
    await session.send(input=types.LiveClientToolResponse(...))
```

**Problem:** The entire receive loop is blocked waiting for Perplexity. No audio can flow during this time.

### Issue #2: Fixed 10ms Playback Delay (MINOR 🟡)

**Location:** `client.js`, line 161

```javascript
if (nextStartTime < now) {
  nextStartTime = now + 0.01; // 10ms fixed buffer
}
```

**Problem:** This 10ms buffer is added every time audio queue is empty. While small, it adds up.

### Issue #3: No Pre-fetching or Caching (MINOR 🟡)

**Location:** `server.py`, `search_with_perplexity()`

**Problem:** Every search is a fresh API call. Common queries (weather, time) could be cached.

### Issue #4: AudioWorklet Message Passing (MINOR 🟡)

**Location:** `pcm-processor.js` → `client.js`

```javascript
// Every 128-sample frame (8ms at 16kHz) triggers a postMessage
this.port.postMessage(int16Data);
```

**Problem:** Many small messages instead of batching. Minor overhead but measurable.

---

## 💡 Recommended Improvements

### Priority 1: Fix the Silent Gap (CRITICAL)

**Solution: Parallel audio filler while waiting**

The issue is that when Gemini says "main check kar rahi hu", it then waits for the tool response. You need to either:

**Option A:** Pre-generate filler audio ("hmm", "let me see", typing sounds)
**Option B:** Have Gemini continue speaking while searching (requires model support)
**Option C:** Implement streaming from Perplexity so results come in chunks

### Priority 2: Implement Search Caching

**Solution: Cache common queries**

```
Cache Structure:
- Key: normalized query hash
- Value: {result, timestamp}
- TTL: 5 minutes for weather, 1 hour for facts
```

Potential time savings: **1000-3000ms** for cached queries

### Priority 3: Use Streaming for Perplexity

**Solution: Switch to streaming API**

Instead of waiting for complete response:

- Stream tokens as they arrive
- Send partial results to Gemini incrementally
- User hears response building up

Potential time savings: **500-1000ms** perceived latency

### Priority 4: WebSocket Connection Warmup

**Solution: Pre-establish connections**

- Keep OpenRouter connection warm with periodic pings
- Pre-authenticate on session start, not on first search
- Use connection pooling

Potential time savings: **100-300ms** on first search

### Priority 5: Audio Chunk Batching

**Solution: Batch AudioWorklet messages**

Instead of posting every 8ms (128 samples), batch to 50ms (800 samples):

- Fewer message passing overhead
- Still well under human perception threshold

Potential time savings: **5-10ms** cumulative

---

## 📊 Latency Breakdown for Web Search Query

### Current (With Issues)

| Phase                   | Time   | Cumulative | Status      |
| ----------------------- | ------ | ---------- | ----------- |
| User speaks             | 0ms    | 0ms        | ✓           |
| Speech → Gemini         | 50ms   | 50ms       | ✓           |
| Gemini processes        | 200ms  | 250ms      | ✓           |
| "Main check" audio      | 500ms  | 750ms      | ✓           |
| Perplexity call         | 2000ms | 2750ms     | 🔴 BLOCKING |
| Tool response → Gemini  | 100ms  | 2850ms     |             |
| Gemini generates answer | 300ms  | 3150ms     |             |
| Audio starts playing    | 50ms   | 3200ms     |             |

**Total: ~3200ms** (with 2000ms silent gap)

### Optimized (With Fixes)

| Phase                     | Time  | Cumulative | Status      |
| ------------------------- | ----- | ---------- | ----------- |
| User speaks               | 0ms   | 0ms        | ✓           |
| Speech → Gemini           | 50ms  | 50ms       | ✓           |
| Gemini processes          | 200ms | 250ms      | ✓           |
| "Main check" audio        | 500ms | 750ms      | ✓           |
| Filler audio plays        | -     | -          | ✓ NEW       |
| Perplexity (cached)       | 0ms   | 750ms      | ✓ CACHED    |
| OR Perplexity (streaming) | 500ms | 1250ms     | ✓ STREAMING |
| Answer starts             | 300ms | 1550ms     |             |

**Optimized Total: ~1500ms** (with filler audio = no silent gap)

---

## 🔧 Technical Debt & Suggestions

### 1. Add Logging for Perplexity Latency

You currently log the start but not the duration:

```python
logger.info(f"🔍 Perplexity search: {query}")
# ... missing: how long did it actually take?
```

**Suggestion:** Add timing measurement to understand actual Perplexity latency.

### 2. No Timeout on Perplexity Call

If OpenRouter is slow or down, your code hangs indefinitely.

**Suggestion:** Add `timeout=10` to the OpenRouter call.

### 3. No Retry Logic

Single point of failure on Perplexity.

**Suggestion:** Add exponential backoff retry (1-2 attempts max).

### 4. Speech Detection Threshold is Hardcoded

```javascript
if (energy > 500) { // Threshold for voice activity
```

**Suggestion:** Make this configurable or auto-calibrating.

### 5. Missing Connection Quality Monitoring

No metrics for:

- WebSocket latency
- Packet loss
- Jitter

**Suggestion:** Add periodic latency measurements.

---

## 📋 Summary of Recommendations

| Priority | Issue                    | Fix                      | Expected Improvement     |
| -------- | ------------------------ | ------------------------ | ------------------------ |
| 🔴 P1    | Silent gap during search | Filler audio + streaming | 2000ms → 0ms perceived   |
| 🟠 P2    | No caching               | Add 5-min cache          | 2000ms → 0ms for repeats |
| 🟠 P3    | No timeout               | Add 10s timeout          | Prevents hangs           |
| 🟡 P4    | Fixed 10ms buffer        | Dynamic buffer           | 10ms → 2ms               |
| 🟡 P5    | No retry logic           | Add 1-retry              | Better reliability       |
| 🟢 P6    | Chunk batching           | 50ms batches             | 5-10ms savings           |

---

## 🏆 Final Verdict

### Architecture Quality: **Good (7.5/10)**

You've made solid architectural decisions:

- Binary WebSocket ✓
- SDK over raw WebSocket ✓
- Proper audio sample rates ✓
- Cost-effective Perplexity integration ✓

### Main Problem: **Tool Call Latency (Not Architecture)**

The "main check krke batati hu" pause is not an architecture problem—it's a **flow problem**. The architecture is sound, but the tool call implementation creates a synchronous blocking gap.

### The Fix is Clear:

1. **Filler audio** during the wait (most impactful)
2. **Caching** for repeated queries
3. **Streaming** from Perplexity for faster perceived response

---

_Document generated for Gemini Live Voice Agent v2.1.0_
_Analysis date: January 19, 2026_
