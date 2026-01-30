# AI Voice Assistant - User Manual

## Overview
Your AI Voice Assistant is a smart meeting companion that joins Google Meet calls and responds to voice commands naturally. It can answer questions, search the web, and maintain contextual conversations.

---

## Quick Start Guide

### 1. Starting the Assistant

**Prerequisites:**
- Python 3.8+ installed
- All dependencies installed (`pip install -r requirements.txt`)
- Environment variables configured in `.env` file

**Start the Server:**
```bash
cd test_gemini/backend
source ../venv/bin/activate
python server_enhanced.py
```

**Server will start on:** `http://localhost:8000`

### 2. Joining a Meeting

**Method 1: API Call**
```bash
curl -X POST http://localhost:8000/api/bot/join \
  -H "Content-Type: application/json" \
  -d '{"meeting_url": "https://meet.google.com/xxx-xxxx-xxx", "bot_name": "AI Assistant"}'
```

**Method 2: Python Script**
```python
import httpx

resp = httpx.post(
    'http://localhost:8000/api/bot/join',
    json={
        'meeting_url': 'https://meet.google.com/xxx-xxxx-xxx',
        'bot_name': 'AI Assistant'
    }
)
print(resp.json())
```

**Method 3: Auto-join on Startup**
```bash
python server_enhanced.py "https://meet.google.com/xxx-xxxx-xxx"
```

### 3. Admitting the Bot

1. In Google Meet, you'll see "AI Assistant" requesting to join
2. Click "Admit" to let the bot enter the meeting
3. The bot will show as "Active" with a camera feed

---

## How to Use

### Wake Words (To Activate)

Say any of these to get the assistant's attention:

**English:**
- "Assistant"
- "Hey Assistant"
- "Hi Assistant"
- "AI"
- "Hey AI"
- "Mochan"
- "Hey Mochan"

**Hindi/Hinglish:**
- "मोचन"
- "हे मोचन"
- "मोचन जी"
- "Assistant जी"

**Examples:**
- "Hey Assistant, what's the weather?"
- "Mochan, can you help me?"
- "हे मोचन, क्या हाल है?"

### Stop Words (To End Conversation)

Say any of these to end the conversation:

**English:**
- "thank you"
- "thanks"
- "that's all"
- "done"

**Hindi:**
- "धन्यवाद"
- "शुक्रिया"
- "थैंक यू"
- "बस"
- "हो गया"

**Example Flow:**
```
You: "Hey Assistant, what's the weather?"
Bot: "It's 72°F and sunny."

You: "What about tomorrow?"
Bot: "Tomorrow will be 75°F."

You: "Thanks"
Bot: "You're welcome." [Bot goes silent]

[Bot waits for next wake word...]
```

---

## Features

### 1. Natural Conversation
- Responds to follow-up questions without repeating wake words
- Maintains context during conversation
- Automatically detects when conversation ends

### 2. Web Search
Ask about current events, news, weather:
- "What's the latest news?"
- "Weather in Delhi"
- "Who won the match yesterday?"

### 3. Knowledge Base (RAG)
If configured, can search internal documents:
- "What's our leave policy?"
- "Find the project guidelines"

### 4. Bilingual Support
Understands and responds in:
- English
- Hindi/Hinglish

### 5. Smart Interruption
- Gracefully handles interruptions
- Doesn't talk over you
- Waits for you to finish speaking

---

## Conversation Examples

### Example 1: Weather Query
```
You: "Hey Assistant, what's the weather?"
Bot: "It's 72°F and sunny in Delhi."

You: "What about tomorrow?"
Bot: "Tomorrow will be 75°F with light clouds."

You: "And Mumbai?"
Bot: "Mumbai is 30°C and humid."

You: "Thanks"
Bot: "You're welcome."
```

### Example 2: Web Search
```
You: "Mochan, who is the current president?"
Bot: "The current president is [latest information]."

You: "What about the prime minister?"
Bot: "The prime minister is [latest information]."

You: "Thank you"
Bot: "You're welcome."
```

### Example 3: Hindi Conversation
```
You: "हे मोचन, समाचार बताओ"
Bot: "आज की मुख्य खबरें..." [Latest news in Hindi]

You: "धन्यवाद"
Bot: "आपका स्वागत है।"
```

---

## Tips for Best Experience

### 1. Speak Clearly
- Use wake words at the beginning of your sentence
- Speak at normal pace
- The assistant works best in quiet environments

### 2. Natural Pauses
- The assistant detects when you stop speaking
- Wait 1-2 seconds after finishing your question
- Don't interrupt the assistant while it's speaking

### 3. Follow-up Questions
- After initial activation, you don't need to repeat wake words
- Just ask follow-up questions naturally
- Say "thanks" when done to end the conversation

### 4. Multiple People
- The assistant tracks the speaker who activated it
- If someone else says a wake word, it will respond to them
- Mentioning another person's name (e.g., "John, what do you think?") ends the conversation

---

## Troubleshooting

### Bot Not Responding

**Check 1: Is the server running?**
```bash
curl http://localhost:8000
```
Should return: `{"status": "running", ...}`

**Check 2: Is the bot in the meeting?**
- Look for "AI Assistant" in the participants list
- Check if it shows "Active"

**Check 3: Wake word detection**
- Speak clearly and use wake words: "Assistant", "Mochan", "AI"
- Try different wake words if one doesn't work

**Check 4: Server logs**
```bash
tail -50 test_gemini/backend/server_enhanced.log
```

### Bot Joins But No Audio

**Problem**: Ngrok tunnel expired
**Solution**: Restart ngrok and update .env
```bash
pkill -f ngrok
ngrok http 8000
# Update WEBHOOK_BASE_URL in .env
# Restart server
```

### Bot Responds Once Then Stops

**Problem**: Connection issue
**Solution**: 
1. Check logs for errors
2. Restart the server
3. Rejoin the meeting

### Audio Quality Issues

**Problem**: Choppy or robotic voice
**Solutions**:
- Check internet connection
- Close unnecessary applications
- Reduce background noise
- Speak closer to microphone

---

## API Reference

### Endpoints

#### Join Meeting
```
POST /api/bot/join
Content-Type: application/json

{
  "meeting_url": "https://meet.google.com/xxx-xxxx-xxx",
  "bot_name": "AI Assistant"
}
```

**Response:**
```json
{
  "bot_id": "uuid-string",
  "status": "joining"
}
```

#### Leave Meeting
```
POST /api/bot/{bot_id}/leave
```

#### Check Status
```
GET /
```

**Response:**
```json
{
  "status": "running",
  "bot_id": "uuid-string",
  "activation": {
    "is_activated": false,
    "time_remaining": 0,
    "total_activations": 0
  },
  "speakers": 0
}
```

#### Manual Activation (Testing)
```
POST /api/activate
POST /api/deactivate
GET /api/activation
```

---

## Configuration

### Environment Variables (.env)

```bash
# Required
GOOGLE_API_KEY=your_google_api_key
RECALLAI_API_KEY=your_recall_api_key
WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok-free.app

# Optional
OPENROUTER_API_KEY=your_openrouter_key
RAG_API_URL=http://your-rag-server.com
BOT_API_KEY=your_rag_api_key
RECALLAI_REGION=us-west-2
```

### Customizing Wake Words

Edit `server_enhanced.py`:
```python
BOT_ALIASES = [
    "YourCustomName",
    "Hey YourCustomName",
    # Add more...
]
```

### Adjusting Response Timeout

Edit `server_enhanced.py`:
```python
ACTIVATION_TIMEOUT_SECONDS = 45  # Change to desired seconds
```

---

## Best Practices

### 1. Security
- Never commit `.env` file to git
- Rotate API keys regularly
- Use strong, unique API keys

### 2. Performance
- Run on stable internet connection
- Use wired connection if possible
- Close unnecessary applications

### 3. Privacy
- Inform meeting participants about the AI assistant
- Don't record sensitive conversations
- Follow your organization's AI usage policies

### 4. Maintenance
- Monitor server logs regularly
- Restart server daily for best performance
- Update dependencies monthly

---

## Advanced Features

### Custom System Instructions

Edit the system instruction in `get_gemini_config()`:
```python
system_instruction = """You are [Name], a [role].

Your personality: [traits]
Your job: [responsibilities]

Rules:
1. [Custom rule 1]
2. [Custom rule 2]
"""
```

### Adding New Tools

1. Define the tool:
```python
my_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="my_function",
            description="What it does",
            parameters=types.Schema(...)
        )
    ]
)
```

2. Add to config:
```python
tools=[web_search_tool, rag_search_tool, my_tool]
```

3. Implement handler:
```python
async def execute_tool(fc, tool_name: str, args: dict):
    if tool_name == "my_function":
        result = await my_function_impl(args)
        # Send result back...
```

---

## Support & Troubleshooting

### Common Error Messages

**"ERR_NGROK_3200"**
- Ngrok tunnel expired
- Solution: Restart ngrok and update .env

**"Deadline expired"**
- Gemini API timeout
- Solution: Check internet connection, restart server

**"Bot not responding"**
- Check server status, logs, and meeting admission

**"Audio suppressed"**
- Bot not activated
- Solution: Use wake words

### Getting Help

1. Check server logs: `tail -100 server_enhanced.log`
2. Verify all environment variables are set
3. Test API endpoints manually
4. Restart server and rejoin meeting

---

## Version History

### Current Version: 4.0
- Two-state conversation logic
- Bilingual support (English/Hindi)
- Web search integration
- RAG support
- Improved interruption handling
- Audio smoothing

### Previous Versions
- v3.0: Complex activation logic
- v2.0: Basic wake word support
- v1.0: Initial implementation

---

## License & Credits

Built with:
- Google Gemini Live API
- Recall.ai
- FastAPI
- OpenRouter (Perplexity)

---

**Last Updated:** January 2026  
**Document Version:** 1.0  
**Assistant Version:** 4.0

**Happy Meeting! 🤖✨**
