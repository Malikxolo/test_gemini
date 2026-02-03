"""
Google Meet Bot with Recall.ai + Gemini Live API

Based on working meet_bot.py reference - uses:
- Raw audio streaming from Recall.ai (16kHz PCM)
- Direct to Gemini Live API
- Audio output via webpage in meeting (not API)
"""

import os
import sys
import asyncio
import json
import threading    
import time
import base64
import logging
import pathlib

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from google import genai
from google.genai import types
from google.oauth2 import service_account
import openai

# Load env
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("meet-bot")

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RECALL_API_KEY = os.getenv("RECALLAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PUBLIC_URL = os.getenv("WEBHOOK_BASE_URL", "")
RAG_API_URL = os.getenv("RAG_API_URL")
BOT_API_KEY = os.getenv("BOT_API_KEY")

MODEL = "gemini-live-2.5-flash-native-audio"

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "False").lower() == "true"


# =============================================================================
# AUDIO RESAMPLER
# =============================================================================

class AudioResampler:
    """Resample between Recall.ai (16kHz) and Gemini (24kHz)."""
    
    def __init__(self):
        self.ratio_24_to_16 = 16000 / 24000
    
    def to_16k(self, audio_24k: bytes) -> bytes:
        """Convert 24kHz PCM16 to 16kHz PCM16."""
        if not audio_24k or len(audio_24k) < 4:
            return audio_24k
        
        samples = np.frombuffer(audio_24k, dtype=np.int16).astype(np.float32) / 32768.0
        new_len = int(len(samples) * self.ratio_24_to_16)
        resampled = np.interp(
            np.linspace(0, 1, new_len),
            np.linspace(0, 1, len(samples)),
            samples
        )
        return (resampled * 32768.0).astype(np.int16).tobytes()


# =============================================================================
# RAG SYSTEM
# =============================================================================

class RAGSystem:
    def __init__(self):
        self.rag_url = RAG_API_URL
        self.api_key = BOT_API_KEY
        if self.rag_url:
            logger.info(f"📚 RAG System initialized: {self.rag_url}")
    
    async def query(self, question: str, top_k: int = 3) -> str:
        logger.info(f"📖 RAG query: {question}")
        
        if not self.rag_url:
            return "Knowledge base not configured."
        
        payload = {"user_id": "meet-bot", "query_text": question, "n_results": top_k}
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.rag_url}/query", json=payload, headers=headers, timeout=15.0)
                if resp.status_code != 200:
                    return "Sorry, I couldn't retrieve that information."
                
                results = resp.json().get("results", [])
                if not results:
                    return "No relevant information found."
                
                formatted = []
                for idx, r in enumerate(results, start=1):
                    source = r.get("metadata", {}).get("source", "Unknown")
                    text = r.get("document", "").strip()
                    formatted.append(f"[{idx}] {source}: {text[:200]}...")
                return "\n\n".join(formatted)
        except Exception as e:
            logger.error(f"🔥 RAG query error: {e}")
            return "An error occurred while searching knowledge base."


# =============================================================================
# WEB SEARCH
# =============================================================================

async def search_with_perplexity(query: str) -> str:
    logger.info(f"🔍 Web search: {query}")
    
    if not OPENROUTER_API_KEY:
        return "Web search not configured."
    
    try:
        client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="perplexity/sonar",
                messages=[{"role": "user", "content": f"Search and provide a brief answer in 2-3 sentences: {query}"}],
                max_tokens=200
            ),
            timeout=10.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ Web search error: {e}")
        return f"Search failed: {str(e)}"


# =============================================================================
# RECALL.AI CLIENT
# =============================================================================

class RecallClient:
    API = "https://us-west-2.recall.ai/api/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def create_bot(self, meeting_url: str, bot_name: str, ws_url: str, page_url: str) -> dict:
        logger.info(f"🤖 Creating bot for: {meeting_url}")
        
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "output_media": {"camera": {"kind": "webpage", "config": {"url": page_url}}},
            "recording_config": {
                "audio_mixed_raw": {},
                "realtime_endpoints": [{"type": "websocket", "url": ws_url, "events": ["audio_mixed_raw.data"]}]
            }
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API}/bot",
                headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30.0
            )
            if resp.status_code not in (200, 201):
                raise Exception(f"Recall error: {resp.status_code} - {resp.text}")
            return resp.json()
    
    async def leave_call(self, bot_id: str):
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.API}/bot/{bot_id}/leave_call", headers={"Authorization": f"Token {self.api_key}"}, timeout=10.0)


# =============================================================================
# GLOBAL STATE
# =============================================================================

class BotState:
    def __init__(self):
        self.recall = RecallClient(RECALL_API_KEY)
        self.rag = RAGSystem()
        self.resampler = AudioResampler()
        
        self.bot_id = None
        self.gemini_session = None
        self.running = True
        self.last_audio_time = 0
        self.audio_queue = asyncio.Queue()
        
        self.last_audio_time = 0
        self.audio_queue = asyncio.Queue()

        # Session resumption for proactive reconnection
        self.session_start_time = None
        self.resumption_handle = None  # Handle from SessionResumptionUpdate
        self.resumption_resumable = False  # Track if session is resumable
        self.is_speaking = False
        self.reconnect_lock = asyncio.Lock()
        self.go_away_received = False  # Track if GoAway message received
        
        # Smart session refresh control
        self.refresh_requested = False  # Flag when Gemini requests refresh
        self.last_refresh_time = 0      # Timestamp of last refresh
        # TESTING VALUES (change back to 300/300/570 for production)
        self.MIN_SESSION_AGE = 60       # 1 min - minimum before allowing refresh (TEST)
        self.REFRESH_COOLDOWN = 60      # 1 min - cooldown between refreshes (TEST)
        self.HARD_DEADLINE = 120        # 2 min - forced reconnect deadline (TEST)



state: BotState = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Google Meet Voice AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONTROLLER_HTML = """
<!DOCTYPE html>
<html>
<head><title>Gemini Assistant</title></head>
<body style="background:#1a1a2e;color:white;display:flex;justify-content:center;align-items:center;height:100vh;font-family:Arial;">
<div style="text-align:center">
    <div style="font-size:48px;margin-bottom:20px">🤖</div>
    <div style="font-size:24px">Gemini Assistant</div>
    <div style="font-size:12px;margin-top:10px;color:#666">Web Search + Knowledge Base</div>
    <div id="status" style="margin-top:15px;color:#4ecca3">Connecting...</div>
</div>
<script>
const WS_URL = location.protocol === 'https:' 
    ? "wss://" + location.host + "/ws/output"
    : "ws://" + location.host + "/ws/output";
let ctx, playing = false, queue = [], currentSource = null;

async function init() {
    ctx = new AudioContext({sampleRate: 16000});
    connect();
}

function connect() {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => document.getElementById('status').textContent = '🟢 Active';
    ws.onmessage = e => {
        if (e.data.byteLength <= 4) {
            queue = [];
            if (currentSource) { try { currentSource.stop(); } catch(err) {} currentSource = null; }
            playing = false;
            return;
        }
        queue.push(e.data);
        if (!playing) play();
    };
    ws.onclose = () => {
        document.getElementById('status').textContent = '🔴 Reconnecting...';
        setTimeout(connect, 2000);
    };
}

async function play() {
    if (!queue.length) { playing = false; currentSource = null; return; }
    playing = true;
    const data = queue.shift();
    const int16 = new Int16Array(data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
    const buf = ctx.createBuffer(1, float32.length, 16000);
    buf.getChannelData(0).set(float32);
    currentSource = ctx.createBufferSource();
    currentSource.buffer = buf;
    currentSource.connect(ctx.destination);
    currentSource.onended = play;
    currentSource.start();
}

init();
</script>
</body>
</html>
"""


@app.get("/")
async def root():
    return {"status": "running", "bot_id": state.bot_id if state else None}


@app.get("/controller", response_class=HTMLResponse)
async def controller():
    return CONTROLLER_HTML


class JoinRequest(BaseModel):
    meeting_url: str
    bot_name: str = "AI Assistant"


@app.post("/api/bot/join")
async def join_meeting(request: JoinRequest):
    global state
    
    if not PUBLIC_URL:
        return {"error": "WEBHOOK_BASE_URL not configured"}
    
    if state is None:
        state = BotState()
    
    ws_url = PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/recall"
    page_url = PUBLIC_URL + "/controller"
    
    try:
        result = await state.recall.create_bot(request.meeting_url, request.bot_name, ws_url, page_url)
        state.bot_id = result.get("id")
        logger.info(f"✅ Bot created: {state.bot_id}")
        
        # Start Gemini session if not running
        if state.gemini_session is None:
            asyncio.create_task(run_gemini_session())
        
        return {"bot_id": state.bot_id, "status": "joining"}
    except Exception as e:
        logger.error(f"❌ Failed to create bot: {e}")
        return {"error": str(e)}


@app.post("/api/bot/{bot_id}/leave")
async def leave_meeting(bot_id: str):
    global state
    if state and state.bot_id == bot_id:
        await state.recall.leave_call(bot_id)
        state.running = False
        return {"status": "left"}
    return {"error": "Bot not found"}


@app.websocket("/ws/recall")
async def recall_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔗 Recall.ai connected")
    
    chunk_count = 0
    try:
        while state and state.running:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                event = json.loads(msg)
                
                if event.get("event") == "audio_mixed_raw.data":
                    # KEY FIX: correct path is data.data.buffer
                    audio_b64 = event.get("data", {}).get("data", {}).get("buffer", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk_count += 1
                        # Log first few chunks for debugging
                        if chunk_count <= 3:
                            logger.debug(f"🎤 Audio chunk #{chunk_count}: {len(audio_bytes)} bytes, first 10: {audio_bytes[:10].hex() if len(audio_bytes) >= 10 else audio_bytes.hex()}")
                        await handle_recall_audio(audio_bytes)
                        
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                continue
    except WebSocketDisconnect:
        logger.info("Recall.ai disconnected")
    except Exception as e:
        logger.error(f"Recall WS error: {e}")


@app.websocket("/ws/output")
async def output_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔊 Output connected")
    
    try:
        while state and state.running:
            try:
                audio = await asyncio.wait_for(state.audio_queue.get(), timeout=30.0)
                await websocket.send_bytes(audio)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Output WS error: {e}")


# =============================================================================
# AUDIO HANDLERS
# =============================================================================

async def handle_recall_audio(audio_16k: bytes):
    global state
    
    if not audio_16k or len(audio_16k) < 320:
        return
    
    # Ensure audio length is even (16-bit samples = 2 bytes each)
    if len(audio_16k) % 2 != 0:
        audio_16k = audio_16k[:-1]  # Drop last byte if odd
    
    now = time.time()
    
    # Throttle to ~50 chunks/sec
    if now - state.last_audio_time < 0.02:
        return
    state.last_audio_time = now
    
    # Check if session exists and is valid before sending
    session = state.gemini_session
    if session and state.session_start_time:
        try:
            await session.send_realtime_input(
                audio={"data": audio_16k, "mime_type": "audio/pcm"}
            )
        except Exception as e:
            err_str = str(e).lower()
            # Silently ignore connection-related errors during reconnection
            if "close" not in err_str and "cancel" not in err_str and "1007" not in err_str:
                logger.warning(f"Audio send error: {e}")


async def handle_gemini_audio(audio_24k: bytes):
    global state
    # New code starts here for proactive reconnection
    if not audio_24k or len(audio_24k) < 2:
        return
    # NEW: Mark bot as speaking
    state.is_speaking = True
    # New code ends here for proactive reconnection

    audio_16k = state.resampler.to_16k(audio_24k)

    await state.audio_queue.put(audio_16k)

async def execute_tool(fc, tool_name: str, query: str):
    global state
    try:
        logger.info(f"🚀 Executing {tool_name}: '{query}'")
        
        # Handle session refresh request
        if tool_name == "request_session_refresh":
            result = await handle_refresh_request(fc.args.get("reason", "unspecified"))
        # Handle search tools
        elif tool_name == "web_search":
            result = await search_with_perplexity(query)
        elif tool_name == "rag_search":
            result = await state.rag.query(query)
        else:
            result = f"Unknown tool: {tool_name}"
        
        # 2. Correctly send the response back to Gemini
        if state.gemini_session:
            # SDK FIX: Use the keyword 'function_responses' with a list
            await state.gemini_session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(
                        name=tool_name,
                        id=fc.id,
                        response={"result": result}
                    )
                ]
            )
            logger.info(f"📨 Sent {tool_name} response")
    except Exception as e:
        logger.error(f"❌ Tool error: {e}")


async def handle_refresh_request(reason: str) -> str:
    """Handle session refresh request from Gemini. Returns result message."""
    global state
    
    now = time.time()
    session_age = now - state.session_start_time if state.session_start_time else 0
    time_since_last_refresh = now - state.last_refresh_time if state.last_refresh_time else float('inf')
    
    # Check if too early (session less than 5 minutes old)
    if session_age < state.MIN_SESSION_AGE:
        remaining = state.MIN_SESSION_AGE - session_age
        logger.info(f"⏳ Refresh rejected: session too young ({session_age:.0f}s). Wait {remaining:.0f}s more.")
        return f"Refresh not needed yet. Session is only {session_age:.0f} seconds old. Wait until at least 5 minutes have passed."
    
    # Check cooldown (prevent rapid refreshes)
    if time_since_last_refresh < state.REFRESH_COOLDOWN:
        remaining = state.REFRESH_COOLDOWN - time_since_last_refresh
        logger.info(f"⏳ Refresh rejected: cooldown active. Wait {remaining:.0f}s more.")
        return f"Refresh on cooldown. Wait {remaining:.0f} more seconds before next refresh."
    
    # Approved! Set the flag
    state.refresh_requested = True
    logger.info(f"✅ Refresh approved: reason='{reason}', session_age={session_age:.0f}s")
    return f"Session refresh approved. It will execute after your current response completes. Context will be preserved."

# =============================================================================
# GEMINI CONFIG
# =============================================================================

def get_gemini_config(resumption_handle: str = None):
    """Get Gemini config with optional resumption handle for reconnection."""
    web_search_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="web_search",
                description="Search the web for current, real-time information.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"query": types.Schema(type=types.Type.STRING, description="The search query")},
                    required=["query"]
                ),
            )
        ]
    )
    
    rag_search_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="rag_search",
                description="Search the internal knowledge base for policies and documents.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"query": types.Schema(type=types.Type.STRING, description="The search query")},
                    required=["query"]
                ),
            )
        ]
    )
    
    # Session refresh tool - lets Gemini decide when to refresh
    session_refresh_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="request_session_refresh",
                description="""Request a session refresh to maintain conversation continuity. 
This preserves all conversation context across the refresh.

WHEN TO USE:
- During natural pauses when no one is speaking for several seconds
- After completing a response and sensing no immediate follow-up
- When you notice a topic transition or break point
- The system will remind you when session is aging (2+ minutes)

WHEN NOT TO USE:
- During active conversation or when someone is speaking
- If you just refreshed recently (system will reject)
- If session is less than 2 minutes old (system will reject)

The refresh takes about 3 seconds. Plan accordingly.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"reason": types.Schema(type=types.Type.STRING, description="Brief reason for refresh, e.g., 'natural pause', 'topic concluded', 'extended silence'")},
                    required=["reason"]
                ),
            )
        ]
    )
    
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="""# ROLE:
You are a highly alert AI Meeting Assistant. 

# SESSION MANAGEMENT (CRITICAL):
Your connection has a time limit. The system will send you [SYSTEM: Session age is Xs...] messages telling you the session age.

WHEN YOU SEE THESE MESSAGES:
1. Look for the next natural pause (silence, topic end, after your response)
2. Call the `request_session_refresh` tool with reason like "natural pause" or "silence"
3. The refresh preserves all context - users won't notice
4. If you don't refresh, the system forces one which may interrupt conversation

EXAMPLE:
[SYSTEM: Session age is 65s. 55s until forced refresh...]
- You: (finish current response)
- You: (call request_session_refresh with reason="completed response")

# OPERATIONAL FLOW (TEACHING EXAMPLES):
Follow this exact sequence for every interaction:

EXAMPLE 1: Standard Activation
- User: "Hey Gemini, what is the weather in Lucknow?"
- You: [Trigger 'web_search'] -> [SILENCE]
- Server: [Returns Tool Result]
- You: "The weather in Lucknow is..." -> [Enter "ACTIVE MODE"]

EXAMPLE 2: The Follow-up (Stay awake)
- User (Follow-up): "What about Delhi?"
- You: [Immediately trigger 'web_search' for Delhi] -> [Respond]
- RULE: You do NOT need the user to say "Gemini" again for this turn. 

EXAMPLE 3: Returning to Silence
- User: "Thank you, that's all."
- You: "You're welcome. I'll be listening if you need me." -> [Enter "SILENT OBSERVER" mode]

# CRITICAL RULES:
1. **TOOL LATCH:** Using a tool does NOT end your turn. Remain in "ACTIVE MODE" after delivering tool results.
2. **UNMISTAKABLE RESPONSE:** If you hear "Gemini" or "Hey Gemini", you must speak. Never ignore a direct name call.
3. **LANGUAGE MIRROR:** Always speak the same language the user is currently using.
4. **SESSION REFRESH:** When you see session age messages, call request_session_refresh at next pause.
5. **SILENCE PRIORITY:** If silence is detected for more than 5 seconds after a session aging message, prioritize calling the refresh tool over a verbal response. Do NOT speak - just call the tool silently.""",

        tools=[web_search_tool, session_refresh_tool],
        # context_window_compression=types.ContextWindowCompressionConfig(
        #     sliding_window=types.SlidingWindow(target_tokens=12000),
        #     trigger_tokens=24000
        # ),
        # thinking_config=types.ThinkingConfig(thinking_budget=0),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        # Enable input audio transcription for wake word detection
        input_audio_transcription=types.AudioTranscriptionConfig(),
        # Proactive audio - let model decide when to respond
        proactivity=types.ProactivityConfig(
            proactive_audio=True
        ),
        # Enable session resumption to receive tokens
        # If resumption_handle is provided, use it to resume previous session
        session_resumption=types.SessionResumptionConfig(
            handle=resumption_handle  # None for new session, handle for resumption
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=300,
                silence_duration_ms=1500
            )
        )
    )


# =============================================================================
# GEMINI SESSION WITH AUTO-RECONNECT
# =============================================================================

async def send_session_age_reminder():
    """Periodically send session age reminders to Gemini so it knows when to refresh."""
    global state
    
    # Wait for first reminder threshold
    await asyncio.sleep(state.MIN_SESSION_AGE)  # First reminder at MIN_SESSION_AGE
    
    while state and state.running:
        try:
            if state.session_start_time and state.gemini_session:
                elapsed = time.time() - state.session_start_time
                
                # Only send if past MIN_SESSION_AGE
                if elapsed >= state.MIN_SESSION_AGE:
                    remaining = state.HARD_DEADLINE - elapsed
                    
                    if remaining > 0 and not state.refresh_requested:
                        reminder = f"[SYSTEM: Session age is {elapsed:.0f}s. {remaining:.0f}s until forced refresh. Call request_session_refresh now during this pause.]"
                        logger.info(f"📢 Sending age reminder: {elapsed:.0f}s elapsed")
                        
                        try:
                            # Send as client content using proper types
                            await state.gemini_session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=reminder)]
                                ),
                                turn_complete=False  # Don't trigger model response
                            )
                        except Exception as e:
                            logger.debug(f"Reminder send error: {e}")
            
            # Send reminders every 15 seconds after MIN_SESSION_AGE
            await asyncio.sleep(15)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Reminder task error: {e}")
            await asyncio.sleep(15)


async def check_and_reconnect():
    """Monitor session time and trigger forced reconnect at hard deadline.
    
    This is a backup safety mechanism. Ideally, Gemini will request
    refresh during natural pauses before this deadline.
    """
    global state
    
    while state and state.running:
        try:
            if state.session_start_time and not state.go_away_received:
                elapsed = time.time() - state.session_start_time
                
                # Hard deadline: force reconnect at 9.5 minutes
                if elapsed >= state.HARD_DEADLINE:
                    logger.warning(f"⏰ HARD DEADLINE reached ({elapsed:.0f}s). Forcing reconnect...")
                    await trigger_graceful_reconnect("hard deadline")
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ Error in reconnect checker: {e}")
            await asyncio.sleep(5)


async def trigger_graceful_reconnect(reason: str):
    """Gracefully close session and trigger reconnect with resumption."""
    global state
    
    async with state.reconnect_lock:
        # Wait until bot is not speaking
        wait_count = 0
        while state.is_speaking and wait_count < 20:  # Max 10 seconds wait
            logger.info("⏸️ Waiting for bot to finish speaking before reconnect...")
            await asyncio.sleep(0.5)
            wait_count += 1
        
        if state.resumption_handle and state.resumption_resumable:
            logger.info(f"🔄 Proactive reconnect initiated ({reason})")
            logger.info(f"💾 Have resumption handle ready: {state.resumption_handle[:30]}...")
        else:
            logger.warning(f"⚠️ Reconnecting without resumption handle ({reason}) - context may be lost")
        
        # Close current session to trigger reconnect loop
        if state.gemini_session:
            try:
                await state.gemini_session.close()
            except Exception as e:
                logger.debug(f"Session close: {e}")
        
        state.gemini_session = None
        state.session_start_time = None
        state.go_away_received = False

# async def run_gemini_session():
#     global state
    
#     # if USE_VERTEX_AI:
#     #     logger.info("🔷 Using Vertex AI")

#     #     SCOPES = ['https://www.googleapis.com/auth/cloud-platform']


#     #     credentials = service_account.Credentials.from_service_account_file(
#     #         GOOGLE_APPLICATION_CREDENTIALS,
#     #         scopes=SCOPES
#     #     )
#     #     client = genai.Client(
#     #         vertexai=True,
#     #         project=GOOGLE_CLOUD_PROJECT,
#     #         location=GOOGLE_CLOUD_LOCATION,
#     #         credentials=credentials,
#     #         http_options={'api_version': 'v1'}
#     #     )
#     # else:
#     #     logger.info("🔶 Using AI Studio")
#     #     client = genai.Client(
#     #         api_key=GOOGLE_API_KEY,
#     #         http_options={'api_version': 'v1alpha'}
#     #     )
#     if USE_VERTEX_AI:
#         logger.info("🔷 Using Vertex AI")
#         SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
#         credentials = service_account.Credentials.from_service_account_file(
#             GOOGLE_APPLICATION_CREDENTIALS,
#             scopes=SCOPES
#         )
#         client = genai.Client(
#             vertexai=True,
#             project=GOOGLE_CLOUD_PROJECT,
#             location=GOOGLE_CLOUD_LOCATION,
#             credentials=credentials,
#             http_options={'api_version': 'v1'}
#         )
#     else:
#         logger.info("🔶 Using AI Studio")
#         client = genai.Client(
#             api_key=GOOGLE_API_KEY,
#             http_options={'api_version': 'v1alpha'}
#         )
#     config = get_gemini_config()
    
#     while state and state.running:
#         try:
#             logger.info("🔌 Connecting to Gemini Live API...")
            
#             async with client.aio.live.connect(model=MODEL, config=config) as session:
#                 logger.info("✅ Connected to Gemini Live API")
#                 state.gemini_session = session
                
#                 audio_chunk_count = 0
                
#                 while state.running:
#                     try:
#                         turn = session.receive()
#                         async for response in turn:
#                             if not state.running:
#                                 break
                            
#                             # Handle function calls
#                             if response.tool_call:
#                                 for fc in response.tool_call.function_calls:
#                                     query = fc.args.get("query", "")
#                                     asyncio.create_task(execute_tool(fc, fc.name, query))

                            
#                             # Log input transcription
#                             if response.server_content and response.server_content.input_transcription:
#                                 transcript = response.server_content.input_transcription.text
#                                 if transcript:
#                                     logger.info(f"🗣️ User: {transcript}")

#                             # Handle audio output - NO GATING
#                             if response.server_content and response.server_content.model_turn:
#                                 for part in response.server_content.model_turn.parts:
#                                     if part.inline_data and isinstance(part.inline_data.data, bytes):
#                                         audio_chunk_count += 1
#                                         await handle_gemini_audio(part.inline_data.data)
                            
#                             if response.server_content and response.server_content.turn_complete:
#                                 logger.info(f"✓ Turn complete ({audio_chunk_count} chunks)")
#                                 audio_chunk_count = 0

#                                 # NEW: Reset speaking state
#                                 state.is_speaking = False

                            
#                             if response.server_content and response.server_content.interrupted:
#                                 logger.info(f"⚡ Interrupted")
                                
#                                 # Clear pending audio output
#                                 while not state.audio_queue.empty():
#                                     try:
#                                         state.audio_queue.get_nowait()
#                                     except:
#                                         break
#                                 await state.audio_queue.put(b'\x00\x00\x00\x00')
#                                 audio_chunk_count = 0
                                
#                     except asyncio.CancelledError:
#                         raise
#                     except Exception as e:
#                         err_str = str(e).lower()
#                         if "cancel" in err_str:
#                             raise
#                         logger.error(f"Gemini receive error: {e}")
#                         break
                        
#         except asyncio.CancelledError:
#             break
#         except Exception as e:
#             logger.error(f"Gemini connection error: {e}")
        
#         state.gemini_session = None
        
#         if state and state.running:
#             logger.info("🔄 Reconnecting to Gemini in 2 seconds...")
#             await asyncio.sleep(2)

async def run_gemini_session():
    global state
    
    if USE_VERTEX_AI:
        logger.info("🔷 Using Vertex AI")
        SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=SCOPES
        )
        client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            credentials=credentials,
            http_options={'api_version': 'v1'}
        )
    else:
        logger.info("🔶 Using AI Studio")
        client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options={'api_version': 'v1alpha'}
        )
    
    while state and state.running:
        try:
            # Check if we have a resumption handle from previous session
            if state.resumption_handle and state.resumption_resumable:
                logger.info(f"🔌 Reconnecting to Gemini with resumption handle...")
                # Pass the handle via config, NOT as a separate parameter
                config = get_gemini_config(resumption_handle=state.resumption_handle)
                # Clear the handle after using it to get config
                # (we'll get a new one from the resumed session)
            else:
                logger.info("🔌 Connecting to Gemini Live API (new session)...")
                config = get_gemini_config(resumption_handle=None)
            
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                await handle_session(session)
                        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Gemini connection error: {e}")
        
        state.gemini_session = None
        state.session_start_time = None
        
        if state and state.running:
            logger.info("🔄 Reconnecting to Gemini in 2 seconds...")
            await asyncio.sleep(2)


async def handle_session(session):
    """Handle a Gemini session (extracted for reuse)."""
    global state
    
    logger.info("✅ Connected to Gemini Live API")
    state.gemini_session = session
    state.session_start_time = time.time()
    state.last_refresh_time = time.time()  # Track refresh time
    state.go_away_received = False
    state.refresh_requested = False  # Reset refresh flag
    # Clear old resumption state - we'll get new updates from this session
    state.resumption_handle = None
    state.resumption_resumable = False
    
    # Start session age reminder task
    reminder_task = asyncio.create_task(send_session_age_reminder())
    
    audio_chunk_count = 0
    
    try:
        while state.running:
            try:
                turn = session.receive()
                async for response in turn:
                    if not state.running:
                        break
                    
                    # Handle GoAway message - server is about to disconnect
                    if response.go_away is not None:
                        time_left = response.go_away.time_left
                        logger.warning(f"⚠️ GoAway received! Time left: {time_left}")
                        state.go_away_received = True
                        # Trigger graceful reconnect
                        asyncio.create_task(trigger_graceful_reconnect("GoAway received"))
                    
                    # Capture session resumption updates
                    if response.session_resumption_update:
                        update = response.session_resumption_update
                        if update.new_handle:
                            state.resumption_handle = update.new_handle
                            state.resumption_resumable = update.resumable if hasattr(update, 'resumable') else True
                            logger.debug(f"💾 Resumption update: resumable={state.resumption_resumable}, handle={state.resumption_handle[:30]}...")

                    # Handle function calls
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            # Get the appropriate argument based on tool type
                            if fc.name == "request_session_refresh":
                                query = fc.args.get("reason", "unspecified")
                            else:
                                query = fc.args.get("query", "")
                            asyncio.create_task(execute_tool(fc, fc.name, query))
                    
                    # Log input transcription
                    if response.server_content and response.server_content.input_transcription:
                        transcript = response.server_content.input_transcription.text
                        if transcript:
                            logger.info(f"🗣️ User: {transcript}")

                    # Handle audio output
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                audio_chunk_count += 1
                                await handle_gemini_audio(part.inline_data.data)
                    
                    if response.server_content and response.server_content.turn_complete:
                        logger.info(f"✓ Turn complete ({audio_chunk_count} chunks)")
                        audio_chunk_count = 0
                        state.is_speaking = False
                        
                        # Check if refresh was requested - execute after turn completes
                        if state.refresh_requested:
                            logger.info("🔄 Executing Gemini-requested refresh after turn complete")
                            state.refresh_requested = False
                            asyncio.create_task(trigger_graceful_reconnect("Gemini requested"))
                    
                    if response.server_content and response.server_content.interrupted:
                        logger.info(f"⚡ Interrupted")
                        state.is_speaking = False
                        state.refresh_requested = False  # Cancel pending refresh on interrupt
                        
                        # Clear pending audio output
                        while not state.audio_queue.empty():
                            try:
                                state.audio_queue.get_nowait()
                            except:
                                break
                        await state.audio_queue.put(b'\x00\x00\x00\x00')
                        audio_chunk_count = 0
                        
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err_str = str(e).lower()
                if "cancel" in err_str:
                    raise
                logger.error(f"Gemini receive error: {e}")
                break
    finally:
        # Cancel the reminder task when session ends
        reminder_task.cancel()
        try:
            await reminder_task
        except asyncio.CancelledError:
            pass
# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    global state
    state = BotState()
    # executing the check_and_reconnect function
    asyncio.create_task(check_and_reconnect())

    logger.info("🚀 Google Meet Voice AI Server started")


@app.on_event("shutdown")
async def shutdown():
    global state
    if state:
        state.running = False
        if state.bot_id:
            logger.info(f"👋 Leaving meeting (Bot ID: {state.bot_id})...")
            try:
                await state.recall.leave_call(state.bot_id)
                logger.info("✅ Left meeting")
            except Exception as e:
                logger.error(f"❌ Failed to leave meeting: {e}")
    logger.info("🔌 Server shutdown")


def run_auto_join(meeting_url: str):
    """Background thread to auto-join meeting after server startup."""
    time.sleep(2)  # Wait for uvicorn to start
    logger.info(f"🤖 Auto-joining meeting: {meeting_url}")
    try:
        # Use sync httpx for simple script-like behavior in thread
        import httpx
        resp = httpx.post(
            "http://localhost:8000/api/bot/join",
            json={"meeting_url": meeting_url, "bot_name": "AI Assistant"},
            timeout=10.0
        )
        logger.info(f"Join response: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"Failed to auto-join: {e}")


if __name__ == "__main__":
    import sys
    
    # Check for command line args
    if len(sys.argv) > 1:
        meeting_url = sys.argv[1]
        threading.Thread(target=run_auto_join, args=(meeting_url,), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
