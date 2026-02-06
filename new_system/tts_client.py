"""
Deepgram TTS Client - Text-to-Speech using Aura

Streaming TTS with WebSocket for lowest latency.
Falls back to HTTP API if WebSocket unavailable.
"""

import os
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional
import httpx

logger = logging.getLogger(__name__)


class DeepgramTTSClient:
    """
    Deepgram Text-to-Speech client using Aura model.
    
    Features:
    - HTTP streaming for reliable operation
    - Multiple voice options
    - 16kHz output for direct playback
    """
    
    DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
    
    # Available Aura voices (English)
    VOICES = {
        "asteria": "aura-asteria-en",      # Female, American
        "luna": "aura-luna-en",            # Female, American  
        "stella": "aura-stella-en",        # Female, American
        "athena": "aura-athena-en",        # Female, British
        "hera": "aura-hera-en",            # Female, American
        "orion": "aura-orion-en",          # Male, American
        "arcas": "aura-arcas-en",          # Male, American
        "perseus": "aura-perseus-en",      # Male, American
        "angus": "aura-angus-en",          # Male, Irish
        "orpheus": "aura-orpheus-en",      # Male, American
        "helios": "aura-helios-en",        # Male, British
        "zeus": "aura-zeus-en",            # Male, American
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "asteria",
        sample_rate: int = 16000,
        encoding: str = "linear16",
    ):
        """
        Initialize TTS client.
        
        Args:
            api_key: Deepgram API key (or from DEEPGRAM_API_KEY env)
            voice: Voice name (see VOICES dict)
            sample_rate: Output sample rate (default 16kHz)
            encoding: Audio encoding (linear16 for PCM)
        """
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.voice = self.VOICES.get(voice, voice)
        self.sample_rate = sample_rate
        self.encoding = encoding
        
        self._client: Optional[httpx.AsyncClient] = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech with streaming output.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio chunks as bytes (PCM 16-bit)
        """
        if not text or not text.strip():
            return
            
        if not self.api_key:
            logger.error("DEEPGRAM_API_KEY not set")
            return
        
        client = await self._get_client()
        
        # Build URL with parameters
        url = f"{self.DEEPGRAM_TTS_URL}?model={self.voice}&encoding={self.encoding}&sample_rate={self.sample_rate}&container=none"
        
        try:
            async with client.stream(
                "POST",
                url,
                json={"text": text.strip()},
            ) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    logger.error(f"TTS error {response.status_code}: {error}")
                    return
                
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
    
    async def synthesize_full(self, text: str) -> Optional[bytes]:
        """
        Synthesize text and return complete audio.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete audio as bytes, or None on error
        """
        chunks = []
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        
        if chunks:
            return b"".join(chunks)
        return None
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("TTS client closed")


class SentenceBuffer:
    """
    Buffer for accumulating LLM tokens into sentences for TTS.
    
    Collects tokens until sentence boundary, then emits for synthesis.
    """
    
    SENTENCE_ENDINGS = {'.', '!', '?'}
    ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'etc.', 'e.g.', 'i.e.'}
    
    def __init__(self, min_chars: int = 10):
        """
        Args:
            min_chars: Minimum characters before considering sentence complete
        """
        self.buffer = ""
        self.min_chars = min_chars
        
    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer, return sentence if complete.
        
        Args:
            text: Text chunk to add
            
        Returns:
            Complete sentence if detected, else None
        """
        self.buffer += text
        
        # Check for sentence boundary
        for i, char in enumerate(self.buffer):
            if char in self.SENTENCE_ENDINGS:
                # Check if this is a real sentence end
                potential_sentence = self.buffer[:i+1].strip()
                
                # Skip if too short
                if len(potential_sentence) < self.min_chars:
                    continue
                
                # Skip common abbreviations
                lower = potential_sentence.lower()
                if any(lower.endswith(abbr) for abbr in self.ABBREVIATIONS):
                    continue
                
                # Found a sentence
                self.buffer = self.buffer[i+1:].lstrip()
                return potential_sentence
        
        return None
    
    def flush(self) -> Optional[str]:
        """
        Flush any remaining text in buffer.
        
        Returns:
            Remaining text if any
        """
        if self.buffer.strip():
            text = self.buffer.strip()
            self.buffer = ""
            return text
        return None
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = ""
