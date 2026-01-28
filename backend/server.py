"""
Google Meet Bot with Recall.ai + Gemini Live API

INTELLIGENT ACTIVATION SYSTEM:
- Gemini listens to all audio continuously
- Gemini decides WHETHER it's being addressed using its understanding
- Gemini calls `activate_response` function when it wants to respond
- Audio output is ONLY allowed after activation
- Auto-deactivates after response or timeout
"""

import os
import sys
import asyncio
import json
import time
import threading
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
import openai

# Load env
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("meet-bot")


# =============================================================================
# ⚙️ GLOBAL CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# =============================================================================

# Bot identity - Gemini uses this to know when it's being addressed
BOT_NAME = "Gemini"  # Primary name the bot responds to
BOT_ALIASES = ["Gemini", "Hey Gemini", "AI", "Assistant", "Bot"]  # Alternative names (for prompt context)

# Activation timing
ACTIVATION_TIMEOUT_SECONDS = 10  # Short window - just enough for the response to complete
DEACTIVATE_AFTER_TURN = True  # If True, deactivate after each response (require fresh activation per utterance)

# Logging verbosity
LOG_ACTIVATION_EVENTS = True  # Log activation/deactivation events
LOG_SUPPRESSED_AUDIO = False  # Log when audio is suppressed (verbose)
LOG_ALL_TRANSCRIPTS = True  # Log all input transcriptions

# Behavior settings
REQUIRE_ACTIVATION_FOR_AUDIO = True  # If False, all audio passes through (for testing)


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RECALL_API_KEY = os.getenv("RECALLAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PUBLIC_URL = os.getenv("WEBHOOK_BASE_URL", "")
RAG_API_URL = os.getenv("RAG_API_URL")
BOT_API_KEY = os.getenv("BOT_API_KEY")

MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"


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
# ACTIVATION MANAGER - Handles intelligent response gating
# =============================================================================

class ActivationManager:
    """
    Manages whether the bot should output audio based on Gemini's decisions.
    
    Flow:
    1. Gemini hears everything
    2. When Gemini decides it's being addressed, it calls `activate_response` function
    3. This manager tracks activation state
    4. Audio output is only allowed when activated
    5. Deactivates after timeout or turn completion
    """
    
    def __init__(self):
        self.is_activated: bool = False
        self.activation_time: float = 0
        self.last_audio_time: float = 0
        self.current_turn_activated: bool = False  # Track if current turn was activated
        self._lock = asyncio.Lock()
        
        # Stats for debugging
        self.total_activations: int = 0
        self.total_suppressions: int = 0
        
        logger.info(f"🎯 ActivationManager initialized")
        logger.info(f"   Bot name: {BOT_NAME}")
        logger.info(f"   Timeout: {ACTIVATION_TIMEOUT_SECONDS}s")
    
    async def activate(self, reason: str = ""):
        """Called when Gemini decides to respond."""
        async with self._lock:
            was_activated = self.is_activated
            self.is_activated = True
            self.activation_time = time.time()
            self.current_turn_activated = True
            self.total_activations += 1
            
            if LOG_ACTIVATION_EVENTS:
                if was_activated:
                    logger.info(f"🔔 RE-ACTIVATED (reason: {reason}) - timer reset")
                else:
                    logger.info(f"✨ ACTIVATED (reason: {reason}) - bot will respond")
    
    async def deactivate(self, reason: str = ""):
        """Called to deactivate (turn complete, timeout, etc.)."""
        async with self._lock:
            if self.is_activated and LOG_ACTIVATION_EVENTS:
                logger.info(f"😴 DEACTIVATED (reason: {reason})")
            self.is_activated = False
            self.current_turn_activated = False
    
    async def on_turn_complete(self):
        """Called when a response turn completes."""
        async with self._lock:
            if DEACTIVATE_AFTER_TURN:
                # Deactivate immediately - require fresh activation for next utterance
                if self.is_activated and LOG_ACTIVATION_EVENTS:
                    logger.info(f"😴 Turn complete - DEACTIVATING (fresh activation required for next question)")
                self.is_activated = False
                self.current_turn_activated = False
            else:
                # Legacy behavior - stay activated
                self.current_turn_activated = False
                if LOG_ACTIVATION_EVENTS:
                    remaining = self.get_time_remaining()
                    logger.info(f"✓ Turn complete - staying active for {remaining:.1f}s more")
    
    async def record_audio_output(self):
        """Record that audio was output."""
        async with self._lock:
            self.last_audio_time = time.time()
            # Don't extend activation - we want fresh evaluation per utterance
    
    async def should_allow_audio(self) -> bool:
        """Check if audio output should be allowed."""
        if not REQUIRE_ACTIVATION_FOR_AUDIO:
            return True
        
        async with self._lock:
            if not self.is_activated:
                self.total_suppressions += 1
                return False
            
            # Check timeout
            elapsed = time.time() - self.activation_time
            if elapsed >= ACTIVATION_TIMEOUT_SECONDS:
                self.is_activated = False
                self.total_suppressions += 1
                if LOG_ACTIVATION_EVENTS:
                    logger.info(f"⏰ Auto-deactivated (timeout after {elapsed:.1f}s)")
                return False
            
            return True
    
    def get_time_remaining(self) -> float:
        """Get seconds until auto-deactivation."""
        if not self.is_activated:
            return 0
        elapsed = time.time() - self.activation_time
        return max(0, ACTIVATION_TIMEOUT_SECONDS - elapsed)
    
    def get_state_info(self) -> dict:
        """Get current state for API/debugging."""
        return {
            "is_activated": self.is_activated,
            "time_remaining": self.get_time_remaining(),
            "total_activations": self.total_activations,
            "total_suppressions": self.total_suppressions,
            "timeout_setting": ACTIVATION_TIMEOUT_SECONDS,
            "deactivate_after_turn": DEACTIVATE_AFTER_TURN,
            "bot_name": BOT_NAME
        }


# =============================================================================
# GLOBAL STATE
# =============================================================================

class BotState:
    def __init__(self):
        self.recall = RecallClient(RECALL_API_KEY)
        self.rag = RAGSystem()
        self.resampler = AudioResampler()
        self.activation = ActivationManager()
        
        self.bot_id = None
        self.gemini_session = None
        self.running = True
        self.last_audio_time = 0
        self.audio_queue = asyncio.Queue()
        
        # Deduplication
        self.processed_call_ids = set()
        self.pending_queries = {}
        self.DEDUPE_WINDOW = 10


state: BotState = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Google Meet Voice AI", version="4.0.0")

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
    <div id="icon" style="font-size:48px;margin-bottom:20px">🤖</div>
    <div style="font-size:24px">Gemini Assistant</div>
    <div style="font-size:12px;margin-top:10px;color:#666">Intelligent Voice Activation</div>
    <div id="status" style="margin-top:15px;color:#4ecca3">Connecting...</div>
    <div id="activation" style="margin-top:10px;font-size:14px;color:#888">State: Unknown</div>
    <div id="stats" style="margin-top:5px;font-size:11px;color:#555"></div>
</div>
<script>
const WS_URL = location.protocol === 'https:' 
    ? "wss://" + location.host + "/ws/output"
    : "ws://" + location.host + "/ws/output";
let ctx, playing = false, queue = [], currentSource = null;

async function init() {
    ctx = new AudioContext({sampleRate: 16000});
    connect();
    setInterval(updateStatus, 1000);
}

async function updateStatus() {
    try {
        const resp = await fetch('/api/activation');
        const data = await resp.json();
        const el = document.getElementById('activation');
        const icon = document.getElementById('icon');
        const stats = document.getElementById('stats');
        
        if (data.is_activated) {
            el.textContent = `🟢 LISTENING (${Math.round(data.time_remaining)}s)`;
            el.style.color = '#4ecca3';
            icon.textContent = '🎤';
        } else {
            el.textContent = '😴 IDLE - say "' + data.bot_name + '" to activate';
            el.style.color = '#888';
            icon.textContent = '🤖';
        }
        stats.textContent = `Activations: ${data.total_activations} | Suppressed: ${data.total_suppressions}`;
    } catch (e) {}
}

function connect() {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => document.getElementById('status').textContent = '🟢 Connected';
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
    return {
        "status": "running", 
        "bot_id": state.bot_id if state else None,
        "activation": state.activation.get_state_info() if state else None
    }


@app.get("/controller", response_class=HTMLResponse)
async def controller():
    return CONTROLLER_HTML


@app.get("/api/activation")
async def get_activation():
    """Get current activation state."""
    if state and state.activation:
        return state.activation.get_state_info()
    return {"error": "Bot not initialized"}


@app.post("/api/activate")
async def force_activate():
    """Manually activate (for testing)."""
    if state and state.activation:
        await state.activation.activate(reason="manual API call")
        return state.activation.get_state_info()
    return {"error": "Bot not initialized"}


@app.post("/api/deactivate")
async def force_deactivate():
    """Manually deactivate (for testing)."""
    if state and state.activation:
        await state.activation.deactivate(reason="manual API call")
        return state.activation.get_state_info()
    return {"error": "Bot not initialized"}


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
    
    try:
        while state and state.running:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                event = json.loads(msg)
                
                if event.get("event") == "audio_mixed_raw.data":
                    audio_b64 = event.get("data", {}).get("data", {}).get("buffer", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
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
    """Forward audio from Recall.ai to Gemini."""
    global state
    
    if not audio_16k or len(audio_16k) < 320:
        return
    
    now = time.time()
    if now - state.last_audio_time < 0.02:
        return
    state.last_audio_time = now
    
    if state.gemini_session:
        try:
            await state.gemini_session.send_realtime_input(
                audio={"data": audio_16k, "mime_type": "audio/pcm"}
            )
        except Exception as e:
            err_str = str(e).lower()
            if "close" not in err_str and "cancel" not in err_str:
                logger.warning(f"Audio send error: {e}")


async def handle_gemini_audio(audio_24k: bytes):
    """
    Handle audio output from Gemini.
    GATED: Only allows audio through if activated.
    """
    global state
    
    # Check if we should allow audio
    should_allow = await state.activation.should_allow_audio()
    
    if not should_allow:
        if LOG_SUPPRESSED_AUDIO:
            logger.debug("🔇 Audio suppressed (not activated)")
        return
    
    # Activated - allow audio through
    audio_16k = state.resampler.to_16k(audio_24k)
    await state.audio_queue.put(audio_16k)
    await state.activation.record_audio_output()


async def handle_activation_function(reason: str):
    """Called when Gemini's activate_response function is triggered."""
    global state
    await state.activation.activate(reason=reason)


async def execute_tool(fc, tool_name: str, args: dict):
    """Execute tool calls from Gemini."""
    global state
    
    try:
        # Handle activation function
        if tool_name == "activate_response":
            reason = args.get("reason", "user addressed me")
            await handle_activation_function(reason)
            
            # Send acknowledgment back to Gemini
            if state.gemini_session:
                function_response = types.FunctionResponse(
                    name=tool_name,
                    id=fc.id,
                    response={"status": "activated", "message": "You may now respond to the user."}
                )
                await state.gemini_session.send(input=types.LiveClientToolResponse(
                    function_responses=[function_response]
                ))
            return
        
        # Handle other tools
        logger.info(f"🚀 Executing {tool_name}: {args}")
        
        if tool_name == "web_search":
            result = await search_with_perplexity(args.get("query", ""))
        elif tool_name == "rag_search":
            result = await state.rag.query(args.get("query", ""))
        else:
            result = f"Unknown tool: {tool_name}"
        
        if state.gemini_session:
            function_response = types.FunctionResponse(
                name=tool_name,
                id=fc.id,
                response={"result": result},
                scheduling="WHEN_IDLE"
            )
            await state.gemini_session.send(input=types.LiveClientToolResponse(
                function_responses=[function_response]
            ))
            logger.info(f"📨 Sent {tool_name} response")
        
    except Exception as e:
        logger.error(f"❌ Tool error: {e}")


# =============================================================================
# GEMINI CONFIG
# =============================================================================

def get_gemini_config():
    # ACTIVATION FUNCTION - Gemini calls this when it decides to respond
    activate_response_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="activate_response",
                description=f"""Call this function ONLY when someone EXPLICITLY addresses you by name ({BOT_NAME}).

REQUIRED CONDITIONS (ALL must be true):
1. The person said your name: "{BOT_NAME}", "Hey {BOT_NAME}", "Gemini", etc.
2. They are asking YOU a question or giving YOU a command
3. It's clear they want YOU to respond (not another person in the meeting)

DO NOT CALL when:
- Questions without your name (probably asking another human)
- People talking to each other
- General questions to the room
- Third-person mentions ("I asked {BOT_NAME}...")
- You're uncertain who they're addressing

DEFAULT TO NOT CALLING THIS FUNCTION unless you're confident they explicitly addressed you.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "reason": types.Schema(
                            type=types.Type.STRING, 
                            description="Why you're activating - must include how they addressed you by name"
                        )
                    },
                    required=["reason"]
                )
            )
        ]
    )
    
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
                behavior="NON_BLOCKING"
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
                behavior="NON_BLOCKING"
            )
        ]
    )
    
    # Build the system instruction with bot identity
    system_instruction = f"""You are {BOT_NAME}, an AI voice assistant participating in a Google Meet call with multiple people.

=== CRITICAL: YOU MUST EVALUATE EVERY SINGLE UTTERANCE INDEPENDENTLY ===

You can hear EVERYTHING in the meeting. Most of what you hear is NOT directed at you.

**For EACH new thing you hear, you must decide: "Is this person talking TO me?"**

**BEFORE you speak, you MUST call the `activate_response` function.** If you don't call it, your audio is blocked.

=== DECISION CRITERIA ===

✅ ACTIVATE only when ALL of these are true:
- The person EXPLICITLY addresses you by name ("{BOT_NAME}", "Hey {BOT_NAME}", "AI", etc.)
- AND they are asking YOU a question or making a request TO you

❌ DO NOT ACTIVATE when:
- People are talking to EACH OTHER (even if asking questions)
- Someone asks a question but doesn't say your name (they're probably asking another human)
- Background conversation, even if it sounds like a question
- Someone mentions you in third person ("I asked {BOT_NAME}...", "Maybe {BOT_NAME} knows...")
- General questions to the room that aren't specifically directed at you
- You're unsure → DEFAULT TO SILENCE

=== CRITICAL: NO ASSUMPTIONS ===

- Do NOT assume a question is for you just because you answered a previous question
- Do NOT assume follow-up questions are for you unless they say your name again
- Each utterance is independent - evaluate it fresh
- When in doubt, STAY SILENT

=== EXAMPLES ===

✅ ACTIVATE:
- "Hey {BOT_NAME}, what time is it?" → activate(reason="explicitly addressed me by name")
- "{BOT_NAME}, search for news about AI" → activate(reason="said my name and gave command")
- "Gemini, can you help?" → activate(reason="addressed me directly")

❌ STAY SILENT (do not activate):
- "What time is it?" (no name - probably asking another person)
- "Can someone tell me about the accounts?" (asking the room, not you specifically)
- "जरा बताना कि इस क्वार्टर के अकाउंट्स कितना है" (asking someone else - no mention of you)
- "How do you know that?" (talking to another human)
- "John, what do you think?" (clearly addressing John)
- "I wonder if the AI could help with that..." (talking ABOUT you, not TO you)

=== RESPONSE GUIDELINES (only after activation) ===

1. Call `activate_response` with clear reason
2. Keep responses concise (this is voice)
3. Match the speaker's language (Hindi → Hindi, English → English)

=== IF INTERRUPTED ===

Stop immediately. Wait for explicit mention of your name before responding again."""

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        tools=[activate_response_tool],  # Enable activation function
        # tools=[activate_response_tool, web_search_tool, rag_search_tool],  # Uncomment to enable all tools
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(target_tokens=12000),
            trigger_tokens=24000
        ),
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        proactivity=types.ProactivityConfig(
            proactive_audio=True
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=100,
                silence_duration_ms=1500
            )
        )
    )


# =============================================================================
# GEMINI SESSION
# =============================================================================

async def run_gemini_session():
    global state
    
    client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})
    config = get_gemini_config()
    
    while state and state.running:
        try:
            logger.info("🔌 Connecting to Gemini Live API...")
            
            async with client.aio.live.connect(model=MODEL, config=config) as session:
                logger.info("✅ Connected to Gemini Live API")
                state.gemini_session = session
                
                audio_chunk_count = 0
                
                while state.running:
                    try:
                        turn = session.receive()
                        async for response in turn:
                            if not state.running:
                                break
                            
                            # Handle function calls (including activation)
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    if fc.id in state.processed_call_ids:
                                        continue
                                    state.processed_call_ids.add(fc.id)
                                    
                                    tool_name = fc.name
                                    args = dict(fc.args) if fc.args else {}
                                    
                                    # Dedupe for non-activation tools
                                    if tool_name != "activate_response":
                                        query = args.get("query", "")
                                        query_key = f"{tool_name}:{query.lower().strip()}"
                                        current_time = time.time()
                                        if query_key in state.pending_queries:
                                            if current_time - state.pending_queries[query_key] < state.DEDUPE_WINDOW:
                                                continue
                                        state.pending_queries[query_key] = current_time
                                    
                                    asyncio.create_task(execute_tool(fc, tool_name, args))
                            
                            # Log transcriptions
                            if response.server_content and response.server_content.input_transcription:
                                transcript_text = response.server_content.input_transcription.text or ""
                                if transcript_text.strip() and LOG_ALL_TRANSCRIPTS:
                                    logger.info(f"📝 Heard: '{transcript_text}'")
                            
                            # Handle audio output (gated by activation)
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                                        audio_chunk_count += 1
                                        if audio_chunk_count == 1:
                                            is_active = state.activation.is_activated
                                            if is_active:
                                                logger.info("🔊 Responding (activated)")
                                            else:
                                                logger.info("🔇 Audio generated but NOT activated - suppressing")
                                        await handle_gemini_audio(part.inline_data.data)
                            
                            # Turn complete
                            if response.server_content and response.server_content.turn_complete:
                                if audio_chunk_count > 0 and state.activation.is_activated:
                                    logger.info(f"✓ Response complete ({audio_chunk_count} chunks)")
                                    await state.activation.on_turn_complete()
                                audio_chunk_count = 0
                            
                            # Interrupted
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ Interrupted - stopping")
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
                        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Gemini connection error: {e}")
        
        state.gemini_session = None
        
        if state and state.running:
            logger.info("🔄 Reconnecting in 2 seconds...")
            await asyncio.sleep(2)


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    global state
    state = BotState()
    logger.info("🚀 Google Meet Voice AI Server v4.1 - Per-Utterance Activation")
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   Bot name: {BOT_NAME}")
    logger.info(f"   Aliases: {BOT_ALIASES}")
    logger.info(f"   Activation timeout: {ACTIVATION_TIMEOUT_SECONDS}s")
    logger.info(f"   Deactivate after turn: {DEACTIVATE_AFTER_TURN}")
    logger.info(f"   Mode: {'Per-utterance evaluation' if DEACTIVATE_AFTER_TURN else 'Persistent activation'}")


@app.on_event("shutdown")
async def shutdown():
    global state
    if state:
        state.running = False
    logger.info("🔌 Server shutdown")


def run_auto_join(meeting_url: str):
    """Background thread to auto-join meeting after server startup."""
    time.sleep(2)
    logger.info(f"🤖 Auto-joining: {meeting_url}")
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:8000/api/bot/join",
            json={"meeting_url": meeting_url, "bot_name": "AI Assistant"},
            timeout=10.0
        )
        logger.info(f"Join response: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failed to auto-join: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        meeting_url = sys.argv[1]
        threading.Thread(target=run_auto_join, args=(meeting_url,), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
