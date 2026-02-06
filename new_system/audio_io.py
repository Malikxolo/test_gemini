"""
Audio I/O - Microphone Input and Speaker Output

Uses sounddevice for cross-platform audio handling.
"""

import asyncio
import logging
import queue
import threading
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed - audio I/O disabled")


class AudioInput:
    """
    Microphone input handler.
    
    Captures audio from default microphone and provides
    it as PCM16 bytes for STT processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 20,
        device: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: Audio sample rate (16kHz for Deepgram)
            channels: Number of channels (1 = mono)
            chunk_ms: Chunk size in milliseconds
            device: Audio device index (None = default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_ms / 1000)  # Samples per chunk
        self.device = device
        
        self._stream: Optional[sd.InputStream] = None
        self._callback: Optional[Callable[[bytes], None]] = None
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if self._running:
            # Convert float32 to int16 PCM
            audio_int16 = (indata * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            self._audio_queue.put(audio_bytes)
            
            if self._callback:
                self._callback(audio_bytes)
    
    def start(self, callback: Optional[Callable[[bytes], None]] = None):
        """
        Start capturing audio from microphone.
        
        Args:
            callback: Function called with each audio chunk (bytes)
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available")
            return False
            
        self._callback = callback
        self._running = True
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(f"🎤 Microphone started ({self.sample_rate}Hz, {self.chunk_size} samples/chunk)")
            return True
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            self._running = False
            return False
    
    def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("🎤 Microphone stopped")
    
    def get_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Get next audio chunk from queue.
        
        Args:
            timeout: Max time to wait for chunk
            
        Returns:
            Audio bytes or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    async def get_chunk_async(self) -> Optional[bytes]:
        """Async version of get_chunk."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_chunk)


class AudioOutput:
    """
    Speaker output handler.
    
    Plays PCM16 audio through default speaker with
    queue-based buffering for smooth playback.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            channels: Number of channels
            device: Audio device index (None = default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._play_thread: Optional[threading.Thread] = None
        self._buffer: bytes = b""
        
    def _play_worker(self):
        """Background thread for audio playback using a persistent output stream."""
        def audio_callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Audio output status: {status}")
            
            needed = frames * 2  # 2 bytes per int16 sample
            
            # Pull chunks from queue into buffer
            while len(self._buffer) < needed:
                try:
                    chunk = self._audio_queue.get_nowait()
                    if chunk is None:  # Poison pill
                        break
                    self._buffer += chunk
                except queue.Empty:
                    break
            
            if len(self._buffer) >= needed:
                data = self._buffer[:needed]
                self._buffer = self._buffer[needed:]
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                outdata[:, 0] = audio_int16.astype(np.float32) / 32767.0
            else:
                # Underrun - use whatever we have, pad with silence
                available = len(self._buffer)
                if available > 0:
                    audio_int16 = np.frombuffer(self._buffer, dtype=np.int16)
                    outdata[:len(audio_int16), 0] = audio_int16.astype(np.float32) / 32767.0
                    outdata[len(audio_int16):, 0] = 0.0
                    self._buffer = b""
                else:
                    outdata.fill(0)
        
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self.device,
                callback=audio_callback,
                blocksize=1024,
            ) as stream:
                logger.debug("Audio output stream opened")
                while self._running:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Audio output stream error: {e}")
    
    def start(self):
        """Start the audio output system."""
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available")
            return False
            
        self._running = True
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()
        logger.info(f"🔊 Speaker started ({self.sample_rate}Hz)")
        return True
    
    def stop(self):
        """Stop audio output."""
        self._running = False
        self._audio_queue.put(None)  # Poison pill
        if self._play_thread:
            self._play_thread.join(timeout=2.0)
        logger.info("🔊 Speaker stopped")
    
    def play(self, audio_bytes: bytes):
        """
        Queue audio for playback.
        
        Args:
            audio_bytes: PCM16 audio data
        """
        if self._running and audio_bytes:
            self._audio_queue.put(audio_bytes)
    
    def clear(self):
        """Clear playback queue (for interrupts)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._buffer = b""  # Clear internal buffer too
        logger.info("🔊 Audio queue cleared (interrupt)")
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing or buffered for playback."""
        return not self._audio_queue.empty() or len(self._buffer) > 0


def list_audio_devices():
    """List available audio devices."""
    if not SOUNDDEVICE_AVAILABLE:
        print("sounddevice not installed")
        return
        
    print("\n=== Audio Devices ===")
    print(sd.query_devices())
    print(f"\nDefault Input: {sd.default.device[0]}")
    print(f"Default Output: {sd.default.device[1]}")
