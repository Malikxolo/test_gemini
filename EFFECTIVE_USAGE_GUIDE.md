# AI Voice Assistant - Effective Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Wake Words & Activation](#wake-words--activation)
3. [Conversation Patterns](#conversation-patterns)
4. [Stop Words & Deactivation](#stop-words--deactivation)
5. [Feature Examples](#feature-examples)
6. [Limitations & Workarounds](#limitations--workarounds)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites
- Server running on `http://localhost:8000`
- Bot joined to Google Meet
- Bot admitted in meeting
- Microphone access enabled

### First Time Setup
```bash
# 1. Start the server
cd test_gemini/backend
source ../venv/bin/activate
python server_enhanced.py

# 2. Join a meeting
curl -X POST http://localhost:8000/api/bot/join \
  -H "Content-Type: application/json" \
  -d '{"meeting_url": "YOUR_MEET_LINK", "bot_name": "AI Assistant"}'

# 3. Admit bot in Google Meet
# 4. Start talking!
```

---

## Wake Words & Activation

### Available Wake Words

#### English Variants (12)
- "Assistant"
- "Hey Assistant"
- "Hi Assistant"
- "Okay Assistant"
- "AI"
- "Hey AI"
- "Hi AI"
- "Mochan"
- "Hey Mochan"
- "Hi Mochan"
- "Okay Mochan"
- "Bot"

#### Hindi/Hinglish Variants (10)
- "मोचन"
- "हे मोचन"
- "मोचन जी"
- "Assistant जी"
- "AI जी"
- "बोट"
- "सहायक"

#### STT Error Variations (7)
- "Asistant"
- "Assistnt"
- "Jiminy" (catches misheard words)
- "the AI"
- "this AI"

**Total: 29 wake word variations**

### Examples of Effective Activation

#### ✅ Good Examples
```
"Hey Assistant, what's the weather today?"
"Mochan, can you help me with something?"
"हे मोचन, क्या हाल है?"
"AI, tell me about the meeting agenda"
```

#### ❌ Bad Examples
```
"What's the weather?" (missing wake word)
"Can you help me?" (missing wake word)
"Hey, what's up?" (ambiguous)
"Um, Assistant..." (hesitation breaks detection)
```

### Limitations of Wake Words

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Must be at start of sentence | "By the way, Assistant..." won't work | Always start with wake word |
| Case sensitive matching | "ASSISTANT" may not work | Use normal case |
| Background noise interference | TV/radio may trigger false positives | Use in quiet environment |
| Accent variations | Strong accents may not be recognized | Speak clearly, use multiple wake words |
| No partial matching | "Assist" won't trigger "Assistant" | Use full wake word |

---

## Conversation Patterns

### Pattern 1: Single Question
**Use Case:** Quick information retrieval

```
You: "Hey Assistant, what time is it?"
Bot: "It's 3:45 PM."

You: "Thanks"
Bot: "You're welcome." [Goes silent]
```

**Best For:**
- Quick facts
- Time/date queries
- Simple calculations
- Weather checks

**Limitations:**
- Bot goes silent after response if no follow-up
- Must say wake word again for next question
- No context maintained between separate conversations

### Pattern 2: Multi-Turn Conversation
**Use Case:** Complex queries with follow-ups

```
You: "Hey Assistant, what's the weather in Delhi?"
Bot: "It's 35°C and sunny in Delhi."

You: "What about Mumbai?" (no wake word needed!)
Bot: "Mumbai is 30°C with light rain."

You: "And Bangalore?"
Bot: "Bangalore is 28°C and pleasant."

You: "Thanks, that's all I needed"
Bot: "You're welcome." [Goes silent]
```

**Best For:**
- Comparative queries
- Exploratory questions
- Building on previous answers
- Natural conversation flow

**Limitations:**
- Only works within 45-second timeout
- Bot may misinterpret if you pause too long (>2 seconds)
- Context limited to current conversation session
- Doesn't remember across different activation periods

### Pattern 3: Interruption Handling
**Use Case:** When you need to stop the bot mid-sentence

```
Bot: "The weather today is quite interesting because there are multiple factors including humidity, wind speed, and atmospheric pressure that contribute to..."

You: "[start speaking] Actually, just tell me the temperature"
Bot: [stops immediately]
Bot: "It's 72°F."
```

**Best For:**
- Stopping long responses
- Correcting misunderstandings
- Taking control of conversation

**Limitations:**
- 1-second grace period before bot stops
- Bot may finish current word
- Audio fade-out takes 100ms
- May not work if bot is in middle of word

### Pattern 4: Side Conversations
**Use Case:** Talking to others while bot is active

```
You: "Hey Assistant, what's the capital of France?"
Bot: "The capital of France is Paris."

You: [to John] "John, do you agree?"
Bot: [stays silent - detects other name]

John: "Yes, that's correct"

You: "Assistant, what about Germany?" [wake word to reactivate]
Bot: "The capital of Germany is Berlin."
```

**Best For:**
- Multi-person meetings
- Collaborative discussions
- Avoiding bot interruptions

**Limitations:**
- Bot detects 187 common names
- May false-trigger on similar-sounding words
- Requires explicit wake word to reactivate
- No way to temporarily mute without deactivating

---

## Stop Words & Deactivation

### Available Stop Words

#### English (11)
- "thank you"
- "thanks"
- "thankyou"
- "ty"
- "that's all"
- "that's it"
- "done"
- "finished"
- "goodbye"
- "bye"
- "see you"

#### Hindi (11)
- "धन्यवाद"
- "शुक्रिया"
- "थैंक यू"
- "थैंक्स"
- "बस"
- "हो गया"
- "खत्म"
- "अलविदा"
- "फिर मिलेंगे"
- "बाद में बात करते हैं"

**Total: 22 stop word variations**

### Examples of Effective Deactivation

#### ✅ Good Examples
```
"Thanks, Assistant"
"Thank you, that's all"
"धन्यवाद, बस"
"Thanks, I'm done"
"That's it, thank you"
```

#### ❌ Bad Examples
```
"Okay" (not a stop word)
"Got it" (not a stop word)
"Hmm" (not a stop word)
"Alright" (not a stop word)
```

### Important: Only User Says Stop Words

**Critical Rule:** Bot saying "You're welcome" does NOT deactivate it. Only when YOU say thanks.

```
You: "Hey Assistant, what's the time?"
Bot: "It's 3:45 PM."

You: "Thanks"
Bot: "You're welcome." ← Bot says this, but YOU said thanks, so it deactivates

[Bot goes silent]
```

### Limitations of Stop Words

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Must be explicit | "OK" or "alright" won't work | Use exact phrases: "thanks", "thank you" |
| Case variations | "THANKS" may not work | Use normal speaking tone |
| Context matters | "No thanks" still deactivates | Say "not now" instead if you want to continue |
| Only user triggers | Bot saying "thank you" doesn't count | Only your speech triggers deactivation |
| 45-second timeout | Auto-deactivates after timeout | Say thanks before timeout to control |

---

## Feature Examples

### Feature 1: Web Search
**What it does:** Searches internet for current information

#### ✅ Good Examples
```
"Hey Assistant, what's the latest news?"
"Mochan, who won the match yesterday?"
"AI, what's the stock price of Apple?"
"Assistant, weather in London"
"Hey Mochan, latest technology news"
```

#### ❌ Bad Examples
```
"Search for..." (bot doesn't need explicit "search" command)
"Google this..." (just ask the question directly)
"Find me..." (ask directly)
```

**Limitations:**
- Requires OPENROUTER_API_KEY to be set
- 10-second timeout on searches
- May fail if search service is down
- Results limited to 2-3 sentences
- Cannot browse websites or click links
- No access to paywalled content

**Workarounds:**
- Ask specific questions for better results
- If search fails, bot will say so
- Try rephrasing if no results
- For complex queries, ask step by step

### Feature 2: Weather Queries
**What it does:** Provides current weather information

#### ✅ Good Examples
```
"Hey Assistant, what's the weather?"
"Mochan, weather in Delhi"
"AI, temperature in Mumbai"
"Assistant, is it raining in Bangalore?"
```

**Limitations:**
- Requires web search to be working
- May not have real-time data
- Limited to major cities
- No hourly forecasts
- No weather alerts

### Feature 3: Time/Date Queries
**What it does:** Tells current time and date

#### ✅ Good Examples
```
"Hey Assistant, what time is it?"
"Mochan, what's today's date?"
"AI, what day is it?"
```

**Limitations:**
- Uses server timezone (may differ from user)
- No timezone conversion
- No alarm/timer functionality

### Feature 4: General Knowledge
**What it does:** Answers factual questions

#### ✅ Good Examples
```
"Hey Assistant, what's the capital of Japan?"
"Mochan, who wrote Romeo and Juliet?"
"AI, what is photosynthesis?"
"Assistant, explain quantum computing simply"
```

**Limitations:**
- Knowledge cutoff (may not know very recent events)
- Cannot access personal/company data (unless RAG configured)
- May hallucinate incorrect information
- No source citations
- Cannot show images or videos

### Feature 5: Hindi Language Support
**What it does:** Understands and responds in Hindi/Hinglish

#### ✅ Good Examples
```
"हे मोचन, समाचार बताओ"
"मोचन जी, मौसम कैसा है?"
"Assistant जी, समय क्या हुआ?"
"हे AI, भारत की राजधानी क्या है?"
```

**Limitations:**
- Mixed Hindi-English (Hinglish) works best
- Pure Hindi may have accuracy issues
- Regional dialects not supported
- Script input (Devanagari) may have issues
- Best with common Hindi words

**Workarounds:**
- Use Hinglish (Hindi words in English script)
- Speak clearly and slowly
- Use common Hindi phrases
- Switch to English if Hindi doesn't work

---

## Limitations & Workarounds

### 1. Audio Quality Issues

**Problem:** Choppy, robotic, or delayed voice

**Causes:**
- Poor internet connection
- High CPU usage
- Background noise
- Browser audio issues

**Workarounds:**
- Use stable WiFi (not mobile data)
- Close unnecessary applications
- Reduce browser tabs
- Use headphones
- Speak closer to microphone
- Refresh browser if audio breaks

### 2. Wake Word Not Detected

**Problem:** Bot doesn't respond to wake words

**Causes:**
- Speaking too fast
- Background noise
- Wake word not at start of sentence
- Accent/pronunciation issues

**Workarounds:**
- Speak clearly and slowly
- Start with wake word: "Assistant, [question]"
- Try different wake words (Mochan, AI)
- Reduce background noise
- Speak louder
- Wait 2-3 seconds between attempts

### 3. Bot Responds to Others

**Problem:** Bot answers when someone else speaks

**Causes:**
- Other person mentions wake word
- Similar-sounding words trigger activation
- No speaker identification

**Workarounds:**
- Use specific wake word combinations
- Mention other person's name to deactivate
- Use grace period override: "ignore that"
- Wait for bot to finish before others speak

### 4. Context Lost

**Problem:** Bot forgets previous conversation

**Causes:**
- 45-second timeout
- Said "thanks" and deactivated
- Connection dropped
- New meeting session

**Workarounds:**
- Keep conversation within 45 seconds
- Don't say thanks until you're done
- Re-ask with full context: "Assistant, about that weather question..."
- Use web search for persistent info

### 5. Interruption Not Working

**Problem:** Bot keeps talking when you try to interrupt

**Causes:**
- Grace period (1 second) hasn't passed
- Audio energy threshold not met
- Bot in middle of word

**Workarounds:**
- Speak louder when interrupting
- Wait 1 second after bot starts
- Say "stop" or "wait" clearly
- Use grace period override phrases

### 6. Ngrok Tunnel Expires

**Problem:** "ERR_NGROK_3200" error

**Causes:**
- Free tier ngrok expires after time
- Connection lost
- URL changed

**Workarounds:**
- Restart ngrok: `ngrok http 8000`
- Update .env with new URL
- Restart server
- Consider paid ngrok for stability

### 7. Bot Not Joining Meeting

**Problem:** Bot doesn't appear in meeting

**Causes:**
- Meeting link invalid
- Meeting ended
- Recall.ai API error
- Bot stuck in lobby

**Workarounds:**
- Verify meeting link is active
- Check server logs for errors
- Try joining again
- Ensure meeting allows external participants
- Admit bot from lobby if waiting

### 8. Web Search Not Working

**Problem:** Bot says "Search failed" or no results

**Causes:**
- OPENROUTER_API_KEY not set
- Perplexity API down
- Query too complex
- Rate limit exceeded

**Workarounds:**
- Check .env has OPENROUTER_API_KEY
- Try simpler queries
- Ask factual questions instead of opinions
- Wait and retry
- Use general knowledge instead

---

## Best Practices

### 1. Starting Conversations
- ✅ Start with wake word: "Hey Assistant..."
- ✅ Speak clearly and at normal pace
- ✅ Wait 1-2 seconds after finishing
- ❌ Don't mumble or speak too fast
- ❌ Don't start mid-sentence

### 2. During Conversations
- ✅ Ask follow-ups without repeating wake word
- ✅ Keep questions concise
- ✅ Pause briefly between questions
- ❌ Don't talk over the bot
- ❌ Don't pause too long (>2 seconds)

### 3. Ending Conversations
- ✅ Say "thanks" or "thank you" to end
- ✅ Wait for "You're welcome" response
- ✅ Use "that's all" or "done" if thanks feels awkward
- ❌ Don't just stop talking (timeout is 45 seconds)
- ❌ Don't say "ok" or "alright" (not stop words)

### 4. Multi-Person Meetings
- ✅ Mention other person's name to deactivate bot
- ✅ Use "ignore that" if bot responds incorrectly
- ✅ Wait for bot to finish before others speak
- ❌ Don't all speak at once
- ❌ Don't assume bot knows who is speaking

### 5. Complex Queries
- ✅ Break into simple steps
- ✅ Ask one thing at a time
- ✅ Clarify if bot misunderstands
- ❌ Don't ask multiple questions in one sentence
- ❌ Don't use ambiguous pronouns ("it", "that")

---

## Troubleshooting Quick Reference

| Problem | Check | Solution |
|---------|-------|----------|
| Bot not responding | Server running? | `curl http://localhost:8000` |
| Bot not responding | Bot in meeting? | Check participants list |
| Bot not responding | Wake word used? | Start with "Assistant" or "Mochan" |
| Bot not responding | Microphone working? | Test in other apps |
| Ngrok error | Tunnel expired? | Restart ngrok, update .env |
| No audio | Bot admitted? | Click "Admit" in Google Meet |
| No audio | Volume up? | Check system and browser volume |
| Choppy audio | Internet stable? | Use WiFi, close other apps |
| Bot interrupts | Speaking over it? | Wait for bot to finish |
| Context lost | Said thanks? | Use wake word again |
| Context lost | Timeout? | Keep conversation active |
| Search fails | API key set? | Check .env file |
| Hindi not working | Using Hinglish? | Try Hindi words in English script |

---

## Command Reference

### Server Commands
```bash
# Start server
cd test_gemini/backend && source ../venv/bin/activate && python server_enhanced.py

# Check status
curl http://localhost:8000

# Join meeting
curl -X POST http://localhost:8000/api/bot/join -H "Content-Type: application/json" -d '{"meeting_url": "LINK", "bot_name": "AI Assistant"}'

# Leave meeting
curl -X POST http://localhost:8000/api/bot/{BOT_ID}/leave

# View logs
tail -100 test_gemini/backend/server_enhanced.log
```

### Ngrok Commands
```bash
# Start ngrok
ngrok http 8000

# Get current URL
curl http://localhost:4040/api/tunnels | grep public_url
```

---

## Summary

### What Works Great ✅
- Simple Q&A with wake word activation
- Multi-turn conversations (within 45 seconds)
- Web search for current information
- Hindi/English bilingual support
- Graceful interruption handling
- Side conversation detection

### What Has Limitations ⚠️
- Context only within single activation
- 45-second timeout
- Requires wake word for each new session
- No persistent memory
- Audio quality depends on connection
- Limited to Google Meet via Recall.ai

### What's Not Supported ❌
- Multiple simultaneous users
- Persistent cross-session memory
- Visual content (images, videos)
- File uploads/downloads
- Advanced reasoning tasks
- Real-time collaborative editing

---

**Remember:** The bot is designed for simple, natural voice conversations. Keep questions straightforward, use wake words clearly, and say thanks when done!

**Happy Meeting! 🤖✨**
