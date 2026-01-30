# Voice Agent Development Log
## Complete Change History from Initial Setup to Final Implementation

---

## 1. INITIAL SETUP (Pull and Run)

### 1.1 Repository Setup
- **Date**: Session start
- **Action**: Cloned repository from `https://github.com/Malikxolo/test_gemini/tree/aakash`
- **Location**: `/Users/_iayushsharma_/Documents/final voice agent/test_gemini/`

### 1.2 Environment Configuration
- **File**: `.env`
- **Variables Set**:
  ```
  GOOGLE_API_KEY=AIzaSyA3_h3oEzq-vqDFybiFarCla7JhYtc0uCM
  OPENROUTER_API_KEY=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
  RAG_API_URL=http://46.62.157.117:8001
  BOT_API_KEY=rag_nKTewd_pJdzeKAjYPjSaujGWuw96_4JUcSX37bQDxiQ
  RECALLAI_API_KEY=c2872f3f3b04c681f9367e7fb3e45dd49006dfad
  RECALLAI_REGION=us-west-2
  WEBHOOK_BASE_URL=[dynamic ngrok URL]
  ```

### 1.3 Server Setup
- Installed dependencies from `requirements.txt`
- Started backend server on port 8000
- Configured ngrok tunnel for webhook URLs
- Multiple ngrok URL updates due to tunnel expiration

---

## 2. ENHANCED IMPLEMENTATION (server_enhanced.py)

### 2.1 Core Architecture Changes

#### 2.1.1 Bot Identity Update
**Original**: Bot name "Gemini" with limited aliases
**Changed to**: Bot name "Assistant" with expanded wake word vocabulary

**Wake Words Added**:
- English: "Assistant", "Hey Assistant", "Hi Assistant", "AI", "Hey AI", "Mochan", "Hey Mochan"
- Hindi/Hinglish: "मोचन", "हे मोचन", "Mochan ji", "Assistant ji"
- Variations: "Asistant", "Assistnt" (for STT errors)

**File**: `server_enhanced.py` (lines 68-103)

#### 2.1.2 Two-State Logic Implementation
**Original**: Binary activation with complex conditions
**Changed to**: Simple two-state system

**States**:
1. **LISTENING** (default): Only responds to wake words
2. **ACTIVE** (after wake word): Responds to everything until "thank you"

**Stop Words** (conversation end triggers):
- English: "thank you", "thanks", "that's all", "done", "goodbye"
- Hindi: "धन्यवाद", "शुक्रिया", "थैंक यू", "बस", "हो गया"

**File**: `server_enhanced.py` (Speaker class, lines 380-398)

### 2.2 Audio Processing Improvements

#### 2.2.1 Interruption Handling
**Added**: `InterruptionManager` class
- Grace period: 1.0 second to finish current word
- Audio energy threshold: >500 for human speech detection
- Prevents false triggers from background noise

**File**: `server_enhanced.py` (lines 560-590)

#### 2.2.2 Audio Smoothing
**Added**: Client-side audio buffering
- Buffer size: 3 chunks (60ms) for smoother playback
- Fade-in: 50ms at start of speech
- Fade-out: 100ms when interrupted

**File**: `server_enhanced.py` (CONTROLLER_HTML, lines 920-970)

#### 2.2.3 Latency Optimization
**Changes**:
- Audio chunk size: 320 bytes (20ms)
- Max queue size: 50 (reduced from 100)
- Throttling: ~50 chunks/sec

**File**: `server_enhanced.py` (lines 369-372)

### 2.3 Conversation Management

#### 2.3.1 Speaker Tracking
**Added**: `SpeakerTracker` class
- Tracks individual speakers
- Maintains activation state per speaker
- Simplified two-state logic (active/listening)

**File**: `server_enhanced.py` (lines 401-526)

#### 2.3.2 Common Names Detection
**Added**: List of 187 common names to detect when user addresses others
- Names: John, Jane, Mike, Sarah, David, Emma, etc.
- Triggers conversation end when another person is mentioned

**File**: `server_enhanced.py` (lines 186-353)

### 2.4 System Instruction Evolution

#### 2.4.1 Initial Complex Instructions
**Original**: 13 sections with strict wake word requirements
- Too restrictive
- Conflicting instructions
- Made Gemini too cautious

#### 2.4.2 Simplified Instructions (Final)
**Current**: Minimal, clear instructions
```
You are Assistant, a helpful AI assistant in a meeting.

CRITICAL: RESPOND TO EVERYTHING

When you hear someone speak:
- ALWAYS respond with a brief answer (1-3 sentences)
- Never stay silent
- Keep responding to every question
- If they say "thank you", say "You're welcome" then you can pause
- Otherwise, keep the conversation going

Be helpful, concise, and responsive.
```

**File**: `server_enhanced.py` (lines 1356-1370)

### 2.5 Gemini Configuration Changes

#### 2.5.1 Audio Detection Settings
**Changes**:
- Enabled automatic activity detection
- Start sensitivity: LOW
- End sensitivity: LOW (less likely to interrupt)
- Prefix padding: 300ms
- Silence duration: 2000ms

**File**: `server_enhanced.py` (lines 1510-1520)

#### 2.5.2 Response Gating
**Evolution**:
1. Initially: Strict activation required (`REQUIRE_ACTIVATION_FOR_AUDIO = True`)
2. Testing phase: Disabled gating (`REQUIRE_ACTIVATION_FOR_AUDIO = False`)
3. Final: Always play audio (removed gating logic)

**File**: `server_enhanced.py` (lines 1553-1570)

### 2.6 Tools Integration

#### 2.6.1 Web Search
**Added**: `web_search` tool using Perplexity API via OpenRouter
- Model: perplexity/sonar
- Timeout: 10 seconds
- Concise responses (2-3 sentences)

**File**: `server_enhanced.py` (lines 648-667)

#### 2.6.2 RAG Search
**Added**: `rag_search` tool for internal knowledge base
- Endpoint: Configurable RAG_API_URL
- API key: BOT_API_KEY
- Top-k results: 3

**File**: `server_enhanced.py` (lines 623-662)

### 2.7 Logging and Debugging

#### 2.7.1 Enhanced Logging
**Added**: Comprehensive logging throughout
- Activation/deactivation events
- Audio chunk processing
- Interruption detection
- Transcription logging
- WebSocket connection status

**File**: `server_enhanced.py` (multiple locations)

#### 2.7.2 Debug Endpoints
**Added**: API endpoints for debugging
- `GET /api/activation` - Check activation state
- `POST /api/activate` - Manual activation
- `POST /api/deactivate` - Manual deactivation

**File**: `server_enhanced.py` (lines 1024-1047)

---

## 3. ISSUES ENCOUNTERED AND RESOLVED

### 3.1 Ngrok Tunnel Issues
**Problem**: Free tier ngrok tunnels expire frequently
**Solution**: Multiple tunnel restarts with URL updates
**Impact**: Required frequent .env updates and server restarts

### 3.2 Interruption Detection
**Problem**: Bot interrupting constantly
**Solution**: 
- Added grace period (1.0s)
- Audio energy threshold (>500)
- Only interrupt after bot has spoken 5+ chunks

### 3.3 Gemini Not Responding
**Problem**: Bot responds once then stops
**Root Cause**: Complex system instructions confusing Gemini
**Solution**: Simplified to minimal instructions

### 3.4 Wake Word Reliability
**Problem**: Wake words not always detected
**Solution**: 
- Expanded wake word vocabulary
- Added Hindi/Hinglish variants
- Added STT error variations

### 3.5 Audio Jitter
**Problem**: Voice cutting/choppy audio
**Solution**:
- Increased chunk buffer (3 chunks)
- Added fade-in/fade-out
- Client-side audio smoothing

---

## 4. FINAL ARCHITECTURE

### 4.1 State Machine
```
LISTENING STATE (is_active = False)
    ↓ [Wake word detected: Assistant, AI, Mochan, मोचन]
ACTIVE STATE (is_active = True)
    ↓ [User says: thank you, thanks, धन्यवाद, शुक्रिया]
LISTENING STATE (is_active = False)
```

### 4.2 Data Flow
1. **Audio Input**: Recall.ai → WebSocket → Audio Processing
2. **Transcription**: Gemini generates text from audio
3. **State Check**: SpeakerTracker checks wake/stop words
4. **Response**: Gemini generates audio response
5. **Output**: Audio queue → WebSocket → Browser → Google Meet

### 4.3 Key Components
- **SpeakerTracker**: Manages speaker states
- **ActivationManager**: Controls activation timing
- **InterruptionManager**: Handles interruptions
- **AudioResampler**: 16kHz ↔ 24kHz conversion
- **RAGSystem**: Knowledge base queries
- **Web Search**: Perplexity API integration

---

## 5. CONFIGURATION SUMMARY

### 5.1 Timing Parameters
- Activation timeout: 45 seconds
- Grace period: 1.0 second
- Audio chunk size: 20ms
- Silence duration: 2000ms
- Prefix padding: 300ms

### 5.2 Thresholds
- Audio energy threshold: 500
- Interruption chunk threshold: 5 chunks
- Max queue size: 50
- Throttle rate: 50 chunks/sec

### 5.3 Wake Words (29 total)
- Assistant (English variants)
- AI (English variants)
- Mochan (English/Hindi variants)
- Common STT variations

### 5.4 Stop Words (22 total)
- English: thank you, thanks, etc.
- Hindi: धन्यवाद, शुक्रिया, etc.

---

## 6. FILES MODIFIED

### 6.1 New Files Created
- `test_gemini/backend/server_enhanced.py` - Main enhanced server

### 6.2 Modified Files
- `test_gemini/.env` - Environment variables (multiple updates)
- `test_gemini/backend/server_enhanced.log` - Log files

### 6.3 Original Files (Unchanged)
- `test_gemini/backend/server.py` - Original server (backup)
- `test_gemini/backend/server_ayush.py` - Alternative version

---

## 7. TESTING HISTORY

### 7.1 Meeting Links Used
1. `https://meet.google.com/qef-zedh-pav` (initial testing)
2. `https://meet.google.com/atr-fcsp-cgi` (extended testing)
3. `https://meet.google.com/aro-jpmf-zzp` (final testing)

### 7.2 Bot IDs Generated
Multiple bot instances created throughout testing:
- 8a1483e0-c53e-49ba-84ad-3e56b7175979
- 00c1bc02-d031-4fad-bb8a-df1aad3a0f4a
- 66a3d6a7-04cb-418a-b925-e965ffde005e
- And many more...

---

## 8. LESSONS LEARNED

### 8.1 What Worked
- Simple two-state logic is more reliable than complex conditions
- Minimal system instructions work better than detailed rules
- Audio buffering significantly improves voice quality
- Energy-based interruption detection reduces false triggers

### 8.2 What Didn't Work
- Complex activation logic with multiple flags
- Strict wake word requirements in system prompt
- Disabling automatic activity detection
- Over-engineering the conversation flow

### 8.3 Best Practices Discovered
1. Keep system instructions minimal and clear
2. Use application-level control instead of trying to control Gemini
3. Buffer audio for smooth playback
4. Test with real users early and often
5. Simple state machines are more reliable than complex logic

---

## 9. CURRENT STATUS (Final Implementation)

### 9.1 Working Features
✅ Wake word detection (Assistant, AI, Mochan, मोचन)
✅ Two-state conversation management
✅ Stop word detection (thank you, धन्यवाद)
✅ Audio smoothing and buffering
✅ Interruption handling with grace period
✅ Web search integration
✅ RAG search integration
✅ Hindi/English bilingual support
✅ Speaker tracking

### 9.2 Known Limitations
- Requires ngrok tunnel (free tier expires)
- Single speaker tracking (simplified)
- No voice fingerprinting
- Limited to Google Meet via Recall.ai

### 9.3 Server Status
- **Status**: Running
- **Port**: 8000
- **Bot Name**: Assistant
- **Current Bot ID**: [Dynamic]
- **ngrok URL**: [Dynamic]

---

## 10. NEXT STEPS (If Development Continues)

### 10.1 Potential Improvements
1. Implement proper speaker identification (voice fingerprinting)
2. Add persistent conversation memory across sessions
3. Implement proactive assistance (detect when help might be needed)
4. Add support for multiple languages beyond Hindi/English
5. Implement better error recovery and reconnection logic
6. Add metrics and analytics dashboard
7. Implement A/B testing for response styles

### 10.2 Production Considerations
1. Upgrade to paid ngrok or use custom domain
2. Implement proper authentication and security
3. Add rate limiting and abuse prevention
4. Set up monitoring and alerting
5. Create deployment pipeline
6. Add comprehensive testing suite

---

## DOCUMENT END

**Total Development Time**: Extended session
**Files Created**: 1 (server_enhanced.py)
**Lines of Code**: ~1700 lines
**Major Iterations**: 15+ significant changes
**Bot Instances**: 30+ created during testing

**Final Note**: The implementation evolved from a complex, restrictive system to a simple, reliable two-state voice assistant that responds naturally to user speech.
