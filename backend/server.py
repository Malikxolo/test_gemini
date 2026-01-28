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
        
        # Deduplication
        self.processed_call_ids = set()
        self.pending_queries = {}
        self.DEDUPE_WINDOW = 10


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
    
    now = time.time()
    
    # Throttle to ~50 chunks/sec
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
    global state
    audio_16k = state.resampler.to_16k(audio_24k)
    await state.audio_queue.put(audio_16k)


async def execute_tool(fc, tool_name: str, query: str):
    global state
    
    try:
        logger.info(f"🚀 Executing {tool_name}: '{query}'")
        
        if tool_name == "web_search":
            result = await search_with_perplexity(query)
        elif tool_name == "rag_search":
            result = await state.rag.query(query)
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
    
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="""You are Gemini, an AI assistant in a Google Meet call with multiple people talking.

=== WAKE WORD ACTIVATION ===

You are ONLY activated when someone says "Gemini" or "Hey Gemini".

If you hear "Gemini" + a question -> RESPOND with helpful answer.
If you DO NOT hear "Gemini" -> DO NOT RESPOND. End your turn immediately.

=== DO NOT GENERATE (IMPORTANT) ===

When you should not respond, you must:
- Generate ZERO audio output
- Do NOT say "shhh", "hmm", "okay", "um", or ANY sound
- Just end your turn with no speech at all

=== DO NOT RESPOND TO (examples) ===

"What's the weather today?" -> END TURN (no "Gemini")
"Tell me a joke" -> END TURN (no "Gemini")
"weather batao" -> END TURN (no "Gemini")
"How are you?" -> END TURN (no "Gemini")
People talking to each other -> END TURN
Background noise or laughter -> END TURN

=== DO RESPOND TO (examples) ===

"Gemini, what's the weather?" -> RESPOND
"Hey Gemini, tell me a joke" -> RESPOND  
"Gemini weather batao" -> RESPOND

=== AFTER INTERRUPTION ===

If interrupted: STOP immediately -> END TURN -> wait for "Gemini" again.

=== LANGUAGE ===

Match the speaker's language. Hindi question = Hindi answer.""",

        # TOOLS DISABLED - they trigger unwanted responses after interruption
        # tools=[web_search_tool, rag_search_tool],
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
        # Input transcription for logging/debugging
        input_audio_transcription=types.AudioTranscriptionConfig(),
        # Proactive audio - allows model to skip responding when appropriate
        proactivity=types.ProactivityConfig(
            proactive_audio=True
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=50,
                silence_duration_ms=1500
            )
        )
    )


# =============================================================================
# GEMINI SESSION WITH AUTO-RECONNECT
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
                            
                            # Handle function calls
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    if fc.id in state.processed_call_ids:
                                        continue
                                    state.processed_call_ids.add(fc.id)
                                    query = fc.args.get("query", "")
                                    
                                    query_key = f"{fc.name}:{query.lower().strip()}"
                                    current_time = time.time()
                                    if query_key in state.pending_queries:
                                        if current_time - state.pending_queries[query_key] < state.DEDUPE_WINDOW:
                                            continue
                                    state.pending_queries[query_key] = current_time
                                    
                                    asyncio.create_task(execute_tool(fc, fc.name, query))
                            
                            # Log input transcription for debugging
                            if response.server_content and response.server_content.input_transcription:
                                transcript_text = response.server_content.input_transcription.text or ""
                                if transcript_text.strip():
                                    logger.info(f"📝 Input: '{transcript_text}'")
                            
                            # Handle audio output - Gemini decides when to respond via proactive_audio
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                                        audio_chunk_count += 1
                                        if audio_chunk_count == 1:
                                            logger.info("🔊 Gemini responding (first audio chunk)")
                                        await handle_gemini_audio(part.inline_data.data)
                            
                            if response.server_content and response.server_content.turn_complete:
                                logger.info(f"✓ Turn complete ({audio_chunk_count} chunks)")
                                audio_chunk_count = 0
                            
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ User interrupted - stopping")
                                
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
                        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Gemini connection error: {e}")
        
        state.gemini_session = None
        
        if state and state.running:
            logger.info("🔄 Reconnecting to Gemini in 2 seconds...")
            await asyncio.sleep(2)


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    global state
    state = BotState()
    logger.info("🚀 Google Meet Voice AI Server started")


@app.on_event("shutdown")
async def shutdown():
    global state
    if state:
        state.running = False
    logger.info("🔌 Server shutdown")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)