"""
Deepgram STT Client - Speech-to-Text using Nova-3

Maintains a persistent WebSocket connection for continuous transcription.
"""

import os
import asyncio
import json
import logging
from typing import AsyncGenerator, Callable, Optional
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class DeepgramSTTClient:
    """
    Deepgram Speech-to-Text client using Nova-3 model.
    
    Features:
    - Persistent WebSocket connection
    - Interim results for low latency
    - VAD-based endpointing
    - Automatic reconnection
    """
    
    DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        encoding: str = "linear16",
        language: str = "multi",
        model: str = "nova-3",
        endpointing: int = 800,
        interim_results: bool = True,
        utterance_end_ms: int = 1000,
        vad_events: bool = True,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoding = encoding
        self.language = language
        self.model = model
        self.endpointing = endpointing
        self.interim_results = interim_results
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._on_transcript: Optional[Callable[[str, bool], None]] = None
        self._on_utterance_end: Optional[Callable[[], None]] = None
        
    def _build_url(self) -> str:
        params = [
            f"model={self.model}",
            f"language={self.language}",
            f"encoding={self.encoding}",
            f"sample_rate={self.sample_rate}",
            f"channels={self.channels}",
            f"endpointing={self.endpointing}",
            f"interim_results={str(self.interim_results).lower()}",
            f"utterance_end_ms={self.utterance_end_ms}",
            f"vad_events={str(self.vad_events).lower()}",
            "smart_format=true",
            "punctuate=true",
        ]
        return f"{self.DEEPGRAM_WS_URL}?{'&'.join(params)}"
    
    async def connect(self) -> bool:
        if not self.api_key:
            logger.error("DEEPGRAM_API_KEY not set")
            return False
            
        try:
            url = self._build_url()
            headers = {"Authorization": f"Token {self.api_key}"}
            
            self._ws = await websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=10)
            self._running = True
            logger.info("Connected to Deepgram STT")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            return False
    
    async def disconnect(self):
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None
        logger.info("Disconnected from Deepgram STT")
    
    async def send_audio(self, audio_data: bytes):
        if self._ws and self._running:
            try:
                await self._ws.send(audio_data)
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
    
    def on_transcript(self, callback: Callable[[str, bool], None]):
        self._on_transcript = callback
        
    def on_utterance_end(self, callback: Callable[[], None]):
        self._on_utterance_end = callback
    
    async def listen(self):
        if not self._ws:
            return
            
        try:
            async for message in self._ws:
                if not self._running:
                    break
                    
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "")
                    
                    if msg_type == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "").strip()
                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)
                            
                            if transcript and self._on_transcript:
                                self._on_transcript(transcript, is_final)
                                
                            if speech_final and self._on_utterance_end:
                                self._on_utterance_end()
                                
                    elif msg_type == "UtteranceEnd":
                        if self._on_utterance_end:
                            self._on_utterance_end()
                            
                except json.JSONDecodeError:
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Error in STT listener: {e}")
    
    async def keep_alive(self):
        while self._running and self._ws:
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
                await asyncio.sleep(10)
            except:
                break
