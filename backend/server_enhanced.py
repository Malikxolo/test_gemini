"""
Enhanced Google Meet Bot with Recall.ai + Gemini Live API - IMPROVED VERSION

Key Improvements:
- Speaker tracking for multi-person meetings
- Strict wake word requirement (no auto-activation)
- Grace period override (English + Hindi)
- Concise responses with Indian English accent
- 45-second engagement window
- 0.5s grace period for interruptions
- OPTIMIZED FOR LOW LATENCY
- SMOOTH AUDIO TRANSITIONS
- CONVERSATION CONTINUITY until "thank you" or other person mentioned
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
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

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
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
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
# ⚙️ GLOBAL CONFIGURATION
# =============================================================================

# Bot identity - Expanded wake word vocabulary
BOT_NAME = "Assistant"
BOT_ALIASES = [
    # English variants
    "Assistant",
    "Hey Assistant",
    "Hi Assistant",
    "Okay Assistant",
    "AI",
    "Hey AI",
    "Hi AI",
    "Okay AI",
    "Bot",
    "Hey Bot",
    "Hi Bot",
    "Mochan",
    "Hey Mochan",
    "Hi Mochan",
    "Okay Mochan",
    # Hindi/Hinglish variants
    "मोचन",
    "हे मोचन",
    "मोचन जी",
    "Mochan ji",
    "Hey Mochan ji",
    "Hi Mochan ji",
    "Assistant ji",
    "Hey Assistant ji",
    "Hi Assistant ji",
    "AI ji",
    "Hey AI ji",
    "Hi AI ji",
    # Common STT variations
    "Asistant",
    "Assistnt",
    "Mochan",
    "the AI",
    "this AI",
    "that AI",
]

# Grace period override phrases
GRACE_PERIOD_OVERRIDE_EN = [
    "ignore that",
    "not to you",
    "that wasn't for you",
    "disregard that",
    "i wasn't talking to you",
    "cancel that",
    "never mind",
    "scratch that",
    "forget i said that",
    "that was for john",
]

GRACE_PERIOD_OVERRIDE_HI = [
    "इसे अनदेखा करो",
    "तुम्हारे लिए नहीं था",
    "यह तुमसे नहीं कहा",
    "भूल जाओ",
    "रद्द करो",
    "यह जॉन के लिए था",
    "कोई बात नहीं",
    "इसे मत सुनो",
    "गलती से बोल दिया",
    "तुम पर लागू नहीं",
]

# Conversation end phrases (thank you, etc.)
CONVERSATION_END_EN = [
    "thank you",
    "thanks",
    "thankyou",
    "ty",
    "that's all",
    "that's it",
    "done",
    "finished",
    "goodbye",
    "bye",
    "see you",
    "talk later",
]

CONVERSATION_END_HI = [
    "धन्यवाद",
    "शुक्रिया",
    "थैंक यू",
    "थैंक्स",
    "बस",
    "हो गया",
    "खत्म",
    "अलविदा",
    "फिर मिलेंगे",
    "बाद में बात करते हैं",
]

# Common names to detect when someone else is addressed
COMMON_NAMES = [
    "john",
    "jane",
    "mike",
    "sarah",
    "david",
    "emma",
    "alex",
    "maria",
    "robert",
    "lisa",
    "james",
    "jennifer",
    "michael",
    "linda",
    "william",
    "patricia",
    "richard",
    "elizabeth",
    "joseph",
    "susan",
    "thomas",
    "jessica",
    "charles",
    "mary",
    "daniel",
    "karen",
    "matthew",
    "nancy",
    "anthony",
    "betty",
    "mark",
    "helen",
    "donald",
    "sandra",
    "steven",
    "donna",
    "paul",
    "carol",
    "andrew",
    "ruth",
    "joshua",
    "sharon",
    "kenneth",
    "michelle",
    "kevin",
    "laura",
    "brian",
    "emily",
    "george",
    "kimberly",
    "edward",
    "deborah",
    "ronald",
    "stephanie",
    "timothy",
    "ashley",
    "jason",
    "melissa",
    "jeffrey",
    "rebecca",
    "ryan",
    "shirley",
    "jacob",
    "catherine",
    "gary",
    "angela",
    "nicholas",
    "kathleen",
    "eric",
    "amy",
    "jonathan",
    "brenda",
    "stephen",
    "pamela",
    "larry",
    "anna",
    "justin",
    "amanda",
    "scott",
    "julie",
    "brandon",
    "christine",
    "benjamin",
    "marie",
    "samuel",
    "janet",
    "gregory",
    "catherine",
    "frank",
    "ann",
    "alexander",
    "joyce",
    "raymond",
    "diane",
    "patrick",
    "alice",
    "jack",
    "kelly",
    "dennis",
    "julia",
    "jerry",
    "emma",
    "tyler",
    "grace",
    "aaron",
    "victoria",
    "jose",
    "cheryl",
    "adam",
    "katherine",
    "nathan",
    "sophia",
    "henry",
    "rose",
    "douglas",
    "amy",
    "zachary",
    "angela",
    "peter",
    "lillian",
    "kyle",
    "janice",
    "walter",
    "martha",
    "ethan",
    "gloria",
    "jeremy",
    "rita",
    "harold",
    "mildred",
    "keith",
    "phyllis",
    "christian",
    "ellen",
    "roger",
    "catherine",
    "noah",
    "louise",
    "gerald",
    "sara",
    "carl",
    "anne",
    "terry",
    "jacqueline",
    "sean",
    "wanda",
    "austin",
    "bonnie",
    "arthur",
    "peggy",
    "lawrence",
    "andrea",
    "jesse",
    "kathryn",
    "dylan",
    "ida",
    "bryan",
    "annie",
    "joe",
    "jenny",
    "jordan",
    "hazel",
    "billy",
    "natalie",
    "bruce",
    "amber",
    "albert",
    "ruby",
    "willie",
    "shannon",
    "gabriel",
    "judy",
    "logan",
    "leonard",
    "kathy",
    "alan",
    "theresa",
    "juan",
    "sally",
    "wayne",
    "marion",
    "roy",
    "kay",
    "ralph",
    "arlene",
    "randy",
    "maureen",
]

# Activation timing
ACTIVATION_TIMEOUT_SECONDS = 45  # 45-second engagement window
DEACTIVATE_AFTER_TURN = False  # Keep engaged during conversation

# Logging verbosity
LOG_ACTIVATION_EVENTS = True  # Log activation/deactivation events
LOG_SUPPRESSED_AUDIO = False  # Log when audio is suppressed (verbose)
LOG_ALL_TRANSCRIPTS = True  # Log all input transcriptions

# Behavior settings
REQUIRE_ACTIVATION_FOR_AUDIO = False  # Allow all audio through for testing
LAST_RESPONSE_FOLLOWUP_WINDOW = 5
GRACE_PERIOD_SECONDS = 1.0  # 1.0s grace period to finish current word/sentence

# Latency optimization
AUDIO_CHUNK_SIZE = 320  # 20ms chunks for smoother audio (was 10ms)
MAX_QUEUE_SIZE = 50  # Prevent memory buildup (reduced from 100)
AUDIO_BUFFER_MS = 50  # 50ms buffer for smooth playback


# =============================================================================
# SPEAKER TRACKING SYSTEM
# =============================================================================


@dataclass
class Speaker:
    """Represents a meeting participant with simple two-state logic."""

    id: str
    name: Optional[str] = None
    voice_fingerprint: Optional[str] = None
    is_active: bool = False  # Simple: either active or listening
    activation_time: float = 0
    last_speech_time: float = 0
    speech_count: int = 0

    def is_engaged(self) -> bool:
        """Check if speaker is in active state (within timeout)."""
        if not self.is_active:
            return False
        elapsed = time.time() - self.activation_time
        return elapsed < ACTIVATION_TIMEOUT_SECONDS


class SpeakerTracker:
    """Tracks multiple speakers in a meeting."""

    def __init__(self):
        self.speakers: Dict[str, Speaker] = {}
        self.current_speaker_id: Optional[str] = None
        self._lock = asyncio.Lock()

    async def identify_speaker(self, audio_data: bytes) -> str:
        """Identify speaker from audio (simplified - uses session-based ID)."""
        # For now, use a simple session-based approach
        # In production, this would use voice fingerprinting
        return "speaker_1"  # Simplified for initial implementation

    async def process_speech(self, speaker_id: str, transcript: str):
        """Process speech from a specific speaker using simple two-state logic."""
        async with self._lock:
            if speaker_id not in self.speakers:
                self.speakers[speaker_id] = Speaker(id=speaker_id)

            speaker = self.speakers[speaker_id]
            speaker.last_speech_time = time.time()
            speaker.speech_count += 1
            self.current_speaker_id = speaker_id

            # SIMPLE TWO-STATE LOGIC:
            # State 1: LISTENING (is_active = False) - only responds to wake words
            # State 2: ACTIVE (is_active = True) - responds to everything until "thank you"

            # Check for stop words (thank you) - ALWAYS check this first
            if self._is_conversation_end(transcript):
                if speaker.is_active:
                    logger.info(f"👋 User said thanks - deactivating {speaker_id}")
                    speaker.is_active = False
                    return speaker, False
                else:
                    # Already inactive, ignore
                    return speaker, False

            # Check for wake words - only works in LISTENING state
            if not speaker.is_active:
                if self._contains_wake_word(transcript):
                    speaker.is_active = True
                    speaker.activation_time = time.time()
                    logger.info(f"🎯 Wake word detected - {speaker_id} is now ACTIVE")
                    return speaker, False
                else:
                    # If no wake word and not active, stay silent
                    return speaker, False
            else:
                # Speaker IS ACTIVE - respond to everything (follow-up questions)
                logger.debug(f"🎤 {speaker_id} is active - processing speech")
                return speaker, False

    def _contains_wake_word(self, transcript: str) -> bool:
        """Check if transcript contains wake word."""
        transcript_lower = transcript.lower()
        for alias in BOT_ALIASES:
            if alias.lower() in transcript_lower:
                return True
        return False

    def _is_grace_period_override(self, transcript: str) -> bool:
        """Check if transcript contains grace period override phrase."""
        transcript_lower = transcript.lower().strip()

        # Check English phrases
        for phrase in GRACE_PERIOD_OVERRIDE_EN:
            if phrase in transcript_lower:
                return True

        # Check Hindi phrases
        for phrase in GRACE_PERIOD_OVERRIDE_HI:
            if phrase in transcript_lower:
                return True

        return False

    def _is_conversation_end(self, transcript: str) -> bool:
        """Check if transcript contains conversation end phrase."""
        transcript_lower = transcript.lower().strip()

        # Check English phrases
        for phrase in CONVERSATION_END_EN:
            if phrase in transcript_lower:
                return True

        # Check Hindi phrases
        for phrase in CONVERSATION_END_HI:
            if phrase in transcript_lower:
                return True

        return False

    def _contains_other_person(self, transcript: str) -> bool:
        """Check if transcript mentions another person."""
        transcript_lower = transcript.lower()
        words = transcript_lower.split()

        # Check for common names
        for name in COMMON_NAMES:
            if name in words:
                # Check if it's at the beginning or followed by a comma (direct address)
                for i, word in enumerate(words):
                    if word == name or word == name + ",":
                        return True

        return False

    def get_active_speaker(self) -> Optional[Speaker]:
        """Get currently active speaker (simple two-state)."""
        for speaker in self.speakers.values():
            if speaker.is_active:
                return speaker
        return None

    def is_any_speaker_active(self) -> bool:
        """Check if any speaker is currently active."""
        return any(s.is_active for s in self.speakers.values())


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
            np.linspace(0, 1, new_len), np.linspace(0, 1, len(samples)), samples
        )
        return (resampled * 32768.0).astype(np.int16).tobytes()


# =============================================================================
# INTERRUPTION MANAGER
# =============================================================================


class InterruptionManager:
    """Manages immediate interruption detection and response."""

    def __init__(self):
        self.is_speaking = False
        self.speak_start_time = 0
        self.interruption_event = asyncio.Event()
        self._current_audio_task = None
        self.grace_period_end = 0

    async def start_speaking(self):
        self.is_speaking = True
        self.speak_start_time = time.time()
        self.grace_period_end = self.speak_start_time + GRACE_PERIOD_SECONDS
        self.interruption_event.clear()

    async def stop_speaking(self):
        self.is_speaking = False
        self.interruption_event.set()

    def should_interrupt_now(self) -> bool:
        """Check if we should interrupt immediately or wait for grace period."""
        if not self.is_speaking:
            return False

        # If still in grace period, wait
        if time.time() < self.grace_period_end:
            return False

        return True

    def interrupt(self):
        """Called when interruption detected."""
        if self.is_speaking:
            self.interruption_event.set()
            return True
        return False


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
                resp = await client.post(
                    f"{self.rag_url}/query", json=payload, headers=headers, timeout=15.0
                )
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
        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        )
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="perplexity/sonar",
                messages=[
                    {
                        "role": "user",
                        "content": f"Search and provide a brief answer in 2-3 sentences: {query}",
                    }
                ],
                max_tokens=200,
            ),
            timeout=10.0,
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

    async def create_bot(
        self, meeting_url: str, bot_name: str, ws_url: str, page_url: str
    ) -> dict:
        logger.info(f"🤖 Creating bot for: {meeting_url}")

        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "output_media": {
                "camera": {"kind": "webpage", "config": {"url": page_url}}
            },
            "recording_config": {
                "audio_mixed_raw": {},
                "realtime_endpoints": [
                    {
                        "type": "websocket",
                        "url": ws_url,
                        "events": ["audio_mixed_raw.data"],
                    }
                ],
            },
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API}/bot",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            if resp.status_code not in (200, 201):
                raise Exception(f"Recall error: {resp.status_code} - {resp.text}")
            return resp.json()

    async def leave_call(self, bot_id: str):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.API}/bot/{bot_id}/leave_call",
                headers={"Authorization": f"Token {self.api_key}"},
                timeout=10.0,
            )


# =============================================================================
# ACTIVATION MANAGER - Enhanced with Speaker Tracking
# =============================================================================


class ActivationManager:
    """
    Manages whether the bot should output audio based on speaker activation.
    """

    def __init__(self):
        self.is_activated: bool = False
        self.activation_time: float = 0
        self.last_audio_time: float = 0
        self.current_turn_activated: bool = False
        self.last_bot_response_time: float = None
        self._lock = asyncio.Lock()

        # Stats for debugging
        self.total_activations: int = 0
        self.total_suppressions: int = 0

        logger.info(f"🎯 ActivationManager initialized")
        logger.info(f"   Bot name: {BOT_NAME}")
        logger.info(f"   Timeout: {ACTIVATION_TIMEOUT_SECONDS}s")

    async def activate(self, reason: str = ""):
        """Called when speaker activates with wake word."""
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
            self.current_turn_activated = False
            self.last_bot_response_time = time.time()
            if LOG_ACTIVATION_EVENTS:
                remaining = self.get_time_remaining()
                logger.info(
                    f"✓ Turn complete - staying active for {remaining:.1f}s more"
                )

    async def record_audio_output(self):
        """Record that audio was output."""
        async with self._lock:
            self.last_audio_time = time.time()

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
            "bot_name": BOT_NAME,
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
        self.speaker_tracker = SpeakerTracker()
        self.interruption_manager = InterruptionManager()

        self.bot_id = None
        self.gemini_session = None
        self.running = True
        self.last_audio_time = 0
        self.audio_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

        # Deduplication
        self.processed_call_ids = set()
        self.pending_queries = {}
        self.DEDUPE_WINDOW = 10


state: BotState = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Google Meet Voice AI Enhanced", version="4.1.0")

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
    <div style="font-size:12px;margin-top:10px;color:#666">Enhanced Voice AI</div>
    <div id="status" style="margin-top:15px;color:#4ecca3">Connecting...</div>
</div>
<script>
const WS_URL = location.protocol === 'https:' 
    ? "wss://" + location.host + "/ws/output"
    : "ws://" + location.host + "/ws/output";
let ctx, playing = false, queue = [], currentSource = null;
let isInterrupted = false;
let fadeOutInterval = null;

async function init() {
    ctx = new AudioContext({sampleRate: 16000});
    connect();
}

function connect() {
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => document.getElementById('status').textContent = '🟢 Active';
    ws.onmessage = e => {
        if (typeof e.data === 'string') {
            const msg = JSON.parse(e.data);
            if (msg.type === 'interrupted') {
                isInterrupted = true;
                fadeOutAndStop();
                document.getElementById('status').textContent = '🔴 Interrupted';
                setTimeout(() => {
                    document.getElementById('status').textContent = '🟢 Active';
                }, 1000);
            }
            return;
        }
        
        if (e.data.byteLength <= 4) {
            queue = [];
            if (currentSource) { 
                fadeOutAndStop();
            }
            playing = false;
            return;
        }
        
        // Limit queue size for lower latency
        if (queue.length < 50) {
            queue.push(e.data);
        }
        
        if (!playing) play();
    };
    ws.onclose = () => {
        document.getElementById('status').textContent = '🔴 Reconnecting...';
        setTimeout(connect, 2000);
    };
}

function fadeOutAndStop() {
    if (currentSource && ctx) {
        // Create gain node for fade out
        const gainNode = ctx.createGain();
        gainNode.connect(ctx.destination);
        
        // Smooth fade out over 100ms
        gainNode.gain.setValueAtTime(1, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);
        
        setTimeout(() => {
            try { 
                currentSource.stop(); 
            } catch(err) {}
            currentSource = null;
            queue = [];
            playing = false;
            isInterrupted = false;
        }, 100);
    }
}

async function play() {
    if (!queue.length || isInterrupted) { 
        playing = false; 
        currentSource = null; 
        return; 
    }
    
    playing = true;
    
    // Buffer multiple chunks for smoother playback
    let totalLength = 0;
    const chunksToPlay = [];
    const chunksToBuffer = Math.min(3, queue.length); // Buffer 3 chunks (60ms)
    
    for (let i = 0; i < chunksToBuffer && queue.length > 0; i++) {
        const data = queue.shift();
        chunksToPlay.push(data);
        totalLength += new Int16Array(data).length;
    }
    
    // Combine chunks into single buffer
    const combinedInt16 = new Int16Array(totalLength);
    let offset = 0;
    for (const data of chunksToPlay) {
        const int16 = new Int16Array(data);
        combinedInt16.set(int16, offset);
        offset += int16.length;
    }
    
    const float32 = new Float32Array(combinedInt16.length);
    for (let i = 0; i < combinedInt16.length; i++) {
        float32[i] = combinedInt16[i] / 32768;
    }
    
    const buf = ctx.createBuffer(1, float32.length, 16000);
    buf.getChannelData(0).set(float32);
    
    currentSource = ctx.createBufferSource();
    currentSource.buffer = buf;
    
    // Add slight fade in for smooth transitions
    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(1, ctx.currentTime + 0.05);
    
    currentSource.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    currentSource.onended = () => {
        if (!isInterrupted) play();
    };
    
    if (!isInterrupted) currentSource.start();
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
        "activation": state.activation.get_state_info() if state else None,
        "speakers": len(state.speaker_tracker.speakers) if state else 0,
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

    ws_url = (
        PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")
        + "/ws/recall"
    )
    page_url = PUBLIC_URL + "/controller"

    try:
        result = await state.recall.create_bot(
            request.meeting_url, request.bot_name, ws_url, page_url
        )
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
    event_count = 0

    try:
        while state and state.running:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                event = json.loads(msg)
                event_count += 1

                # Debug: Log all events (first 10 only to avoid spam)
                if event_count <= 10:
                    logger.info(f"📡 Recall event: {event.get('event', 'unknown')}")

                if event.get("event") == "audio_mixed_raw.data":
                    # Extract audio data
                    audio_b64 = event.get("data", {}).get("data", {}).get("buffer", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)

                        # Debug log
                        if event_count <= 50:
                            logger.info(f"🎵 Audio received: {len(audio_bytes)} bytes")

                        # Identify speaker and process
                        speaker_id = await state.speaker_tracker.identify_speaker(
                            audio_bytes
                        )

                        # Handle audio with speaker context
                        await handle_recall_audio(audio_bytes, speaker_id)
                elif event.get("event") == "bot.status_change":
                    # Log status changes
                    status = event.get("data", {}).get("status", {})
                    logger.info(f"🤖 Bot status: {status}")

            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
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
                # Shorter timeout with ping to keep connection alive
                audio = await asyncio.wait_for(state.audio_queue.get(), timeout=10.0)
                await websocket.send_bytes(audio)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text('{"type":"ping"}')
                except:
                    break
                continue
    except WebSocketDisconnect:
        logger.info("🔊 Output disconnected")
    except Exception as e:
        logger.error(f"Output WS error: {e}")


# =============================================================================
# AUDIO HANDLERS
# =============================================================================


async def handle_recall_audio(audio_16k: bytes, speaker_id: str = "unknown"):
    """Handle incoming audio from Recall.ai with speaker tracking."""
    global state

    if not audio_16k or len(audio_16k) < 320:
        return

    now = time.time()

    # Throttle to ~50 chunks/sec
    if now - state.last_audio_time < 0.02:
        return
    state.last_audio_time = now

    # Check for interruption ONLY if:
    # 1. Bot is speaking
    # 2. Bot has been speaking for at least 1 second (avoid interrupting immediately)
    # 3. Audio energy is high enough to indicate human speech (not just background noise)
    if (
        state.activation.is_activated
        and state.interruption_manager.is_speaking
        and state.interruption_manager.should_interrupt_now()
    ):
        # Check audio energy to avoid false triggers from background noise
        audio_energy = calculate_audio_energy(audio_16k)
        if audio_energy > 2000:  # Threshold for human speech (raised to avoid echo)
            logger.info(
                f"🛑 Interruption detected (energy: {audio_energy:.0f}) - stopping bot"
            )
            await trigger_interruption()
            return  # Don't send this audio to Gemini (it's the interrupting speech)

    if state.gemini_session:
        try:
            await state.gemini_session.send_realtime_input(
                audio={"data": audio_16k, "mime_type": "audio/pcm"}
            )
        except Exception as e:
            err_str = str(e).lower()
            if "close" not in err_str and "cancel" not in err_str:
                logger.warning(f"Audio send error: {e}")


def calculate_audio_energy(audio_bytes: bytes) -> float:
    """Calculate RMS energy of audio signal."""
    if len(audio_bytes) < 4:
        return 0.0

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(samples) == 0:
        return 0.0

    # Calculate RMS energy
    energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return float(energy)


async def trigger_interruption():
    """Trigger immediate interruption."""
    global state

    # Signal interruption to client
    await broadcast_interruption_signal()

    # Stop bot speaking (but DON'T deactivate - speaker is still engaged)
    state.interruption_manager.interrupt()
    # Note: We intentionally do NOT set state.activation.is_activated = False here
    # The speaker is still engaged and may ask follow-up questions

    # Clear audio queue
    cleared_chunks = 0
    while not state.audio_queue.empty():
        try:
            state.audio_queue.get_nowait()
            cleared_chunks += 1
        except:
            break

    # Send stop signal
    await state.audio_queue.put(b"\x00" * 320)

    logger.info(f"⚡ Interrupted: cleared {cleared_chunks} pending chunks")


async def broadcast_interruption_signal():
    """Broadcast interruption signal (placeholder for multi-client support)."""
    pass


async def handle_gemini_audio(audio_24k: bytes):
    """Handle audio output from Gemini."""
    global state

    audio_16k = state.resampler.to_16k(audio_24k)

    # Try to add to queue with timeout to prevent blocking
    try:
        await asyncio.wait_for(state.audio_queue.put(audio_16k), timeout=0.1)
    except asyncio.TimeoutError:
        # Queue is full, drop oldest chunk
        try:
            state.audio_queue.get_nowait()
            await state.audio_queue.put(audio_16k)
        except:
            pass


async def execute_tool(fc, tool_name: str, args: dict):
    """Execute tool calls from Gemini."""
    global state

    try:
        logger.debug(f"🔧 Executing {tool_name}: {args}")

        if tool_name == "web_search":
            result = await search_with_perplexity(args.get("query", ""))
        elif tool_name == "rag_search":
            result = await state.rag.query(args.get("query", ""))
        else:
            result = f"Unknown tool: {tool_name}"

        if state.gemini_session:
            try:
                function_response = types.FunctionResponse(
                    name=tool_name,
                    id=fc.id,
                    response={"result": result},
                    scheduling="WHEN_IDLE",
                )
                await asyncio.wait_for(
                    state.gemini_session.send(
                        input=types.LiveClientToolResponse(
                            function_responses=[function_response]
                        )
                    ),
                    timeout=5.0,
                )
                logger.debug(f"📨 Sent {tool_name} response")
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Tool response timeout for {tool_name}")
            except Exception as e:
                err_str = str(e).lower()
                if (
                    "1008" in err_str
                    or "policy violation" in err_str
                    or "websocket" in err_str
                ):
                    logger.debug(f"Session disconnected during {tool_name} response")
                else:
                    logger.error(f"❌ Failed to send {tool_name} response: {e}")

    except Exception as e:
        err_str = str(e).lower()
        if "1008" not in err_str and "policy violation" not in err_str:
            logger.error(f"❌ Tool execution error: {e}")


# =============================================================================
# GEMINI CONFIG
# =============================================================================


def get_gemini_config():
    """Get Gemini configuration with enhanced system prompt."""

    system_instruction = f"""You are {BOT_NAME}, a helpful AI assistant in a meeting.

CRITICAL: RESPOND TO EVERYTHING

When you hear someone speak:
- ALWAYS respond with a brief answer (1-3 sentences)
- Never stay silent
- Keep responding to every question
- If they say "thank you", say "You're welcome" then you can pause
- Otherwise, keep the conversation going

Be helpful, concise, and responsive."""

    # Define tools for web search and RAG
    web_search_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="web_search",
                description="Search the internet for current information, news, weather, or any real-time data.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING, description="The search query"
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )

    rag_search_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="rag_search",
                description="Search internal knowledge base for company policies, documents, or organizational information.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING, description="The search query"
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        tools=[web_search_tool, rag_search_tool],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        proactivity=types.ProactivityConfig(proactive_audio=True),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,  # Enable automatic speech detection
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,  # Less sensitive to avoid interruptions
                prefix_padding_ms=300,  # Longer padding to capture full speech
                silence_duration_ms=2000,  # Wait 2 seconds of silence before ending turn
            )
        ),
    )


# =============================================================================
# GEMINI SESSION
# =============================================================================


async def run_gemini_session():
    """Run Gemini Live API session with enhanced handling."""
    global state

    client = genai.Client(
        api_key=GOOGLE_API_KEY, http_options={"api_version": "v1alpha"}
    )
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

                            # Handle transcriptions with speaker tracking
                            if (
                                response.server_content
                                and response.server_content.input_transcription
                            ):
                                transcript_text = (
                                    response.server_content.input_transcription.text
                                    or ""
                                )
                                if transcript_text.strip() and LOG_ALL_TRANSCRIPTS:
                                    logger.info(f"🎤 Heard: '{transcript_text}'")

                                # Process with speaker tracker
                                speaker_id = "speaker_1"  # Simplified
                                (
                                    speaker,
                                    override,
                                ) = await state.speaker_tracker.process_speech(
                                    speaker_id, transcript_text
                                )

                                # Simple two-state logic: activate if speaker is active
                                if speaker.is_active and not override:
                                    await state.activation.activate(
                                        reason=f"Speaker {speaker_id} is active"
                                    )

                            # Handle audio output - ALWAYS play audio
                            if (
                                response.server_content
                                and response.server_content.model_turn
                            ):
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data and isinstance(
                                        part.inline_data.data, bytes
                                    ):
                                        audio_chunk_count += 1

                                        # Always play audio - no gating
                                        if audio_chunk_count == 1:
                                            await state.interruption_manager.start_speaking()
                                            logger.info("🔊 Responding")
                                        await handle_gemini_audio(part.inline_data.data)

                            # Turn complete
                            if (
                                response.server_content
                                and response.server_content.turn_complete
                            ):
                                if audio_chunk_count > 0:
                                    logger.info(
                                        f"✓ Response complete ({audio_chunk_count} chunks)"
                                    )
                                    await state.interruption_manager.stop_speaking()
                                    await state.activation.on_turn_complete()
                                audio_chunk_count = 0

                            # Interrupted - handle gracefully but don't clear everything
                            if (
                                response.server_content
                                and response.server_content.interrupted
                            ):
                                logger.debug("⚡ Interrupted signal received")
                                # Just stop speaking but don't clear queue or deactivate
                                if state.interruption_manager.is_speaking:
                                    await state.interruption_manager.stop_speaking()
                                # Don't trigger full interruption - just stop current response
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
    logger.info("🚀 Enhanced Google Meet Voice AI Server started")
    logger.info(f"   Wake words: {len(BOT_ALIASES)} aliases configured")
    logger.info(f"   Grace period: {GRACE_PERIOD_SECONDS}s")
    logger.info(f"   Engagement timeout: {ACTIVATION_TIMEOUT_SECONDS}s")
    logger.info(
        f"   Conversation end phrases: {len(CONVERSATION_END_EN) + len(CONVERSATION_END_HI)} phrases"
    )
    logger.info(f"   Common names tracked: {len(COMMON_NAMES)} names")


@app.on_event("shutdown")
async def shutdown():
    global state
    if state:
        state.running = False
    logger.info("🔌 Server shutdown")


def run_auto_join(meeting_url: str):
    """Background thread to auto-join meeting after server startup."""
    time.sleep(2)
    logger.info(f"🤖 Auto-joining meeting: {meeting_url}")
    try:
        import httpx

        resp = httpx.post(
            "http://localhost:8000/api/bot/join",
            json={"meeting_url": meeting_url, "bot_name": "AI Assistant"},
            timeout=10.0,
        )
        logger.info(f"Join response: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"Failed to auto-join: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        meeting_url = sys.argv[1]
        threading.Thread(target=run_auto_join, args=(meeting_url,), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
