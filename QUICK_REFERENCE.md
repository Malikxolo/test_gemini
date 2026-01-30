# AI Voice Assistant - Quick Reference Card

## 🎯 WAKE WORDS (Say to Activate)
**English:** Assistant, Hey Assistant, AI, Hey AI, Mochan, Hey Mochan  
**Hindi:** मोचन, हे मोचन, मोचन जी, Assistant जी

## 🛑 STOP WORDS (Say to End)
**English:** thank you, thanks, that's all, done  
**Hindi:** धन्यवाद, शुक्रिया, थैंक यू, बस, हो गया

## 💬 HOW IT WORKS

```
[LISTENING] → Wake Word → [ACTIVE] → Stop Word → [LISTENING]
   Silent       "Hey        Responds     "Thanks"      Silent
              Assistant"    to all
```

## 📝 EXAMPLE CONVERSATION

```
You: "Hey Assistant, what's the weather?"
Bot: "It's 72°F and sunny."

You: "What about tomorrow?"  ← No wake word needed!
Bot: "Tomorrow will be 75°F."

You: "Thanks"  ← Ends conversation
Bot: "You're welcome." [Goes silent]

You: "Assistant, what about Delhi?"  ← Wake word to restart
Bot: "Delhi is 35°C."
```

## 🚀 QUICK COMMANDS

### Start Server
```bash
cd test_gemini/backend
source ../venv/bin/activate
python server_enhanced.py
```

### Join Meeting
```bash
curl -X POST http://localhost:8000/api/bot/join \
  -H "Content-Type: application/json" \
  -d '{"meeting_url": "YOUR_MEET_LINK", "bot_name": "AI Assistant"}'
```

### Check Status
```bash
curl http://localhost:8000
```

### View Logs
```bash
tail -50 test_gemini/backend/server_enhanced.log
```

## 🔧 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Bot not responding | Check server: `curl http://localhost:8000` |
| Ngrok error | Restart ngrok, update .env, restart server |
| No audio | Check meeting admission, verify bot is "Active" |
| Choppy audio | Check internet, close other apps |

## 📞 SUPPORT

1. Check logs: `tail -100 server_enhanced.log`
2. Verify .env variables
3. Restart server
4. Rejoin meeting

## ⚡ PRO TIPS

✅ Speak wake words clearly at the start  
✅ Wait 1-2 seconds after finishing speaking  
✅ Don't interrupt the bot while it's speaking  
✅ Use "thanks" to end conversation cleanly  
✅ Bot remembers context during conversation  

## 🌐 FEATURES

- ✅ Natural conversation (no repeat wake words)
- ✅ Web search (news, weather, current events)
- ✅ Hindi + English support
- ✅ Smart interruption handling
- ✅ Context awareness

---

**Bot Name:** Assistant / Mochan / मोचन  
**Wake Words:** 29 aliases  
**Timeout:** 45 seconds  
**Server:** http://localhost:8000

**Keep this card handy during meetings! 📋**
