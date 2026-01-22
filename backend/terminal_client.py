import asyncio
import websockets
import pyaudio
import json
import logging
import time
import sys

# Configuration
WS_URL = "ws://127.0.0.1:8000/ws"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_IN = 16000
RATE_OUT = 24000
CHUNK = 2048  # Use larger chunks like our optimized client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TerminalClient")

async def audio_stream():
    p = pyaudio.PyAudio()

    # Input Stream (16kHz)
    try:
        stream_in = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE_IN,
            input=True,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        logger.error(f"Failed to open input stream: {e}")
        return

    # Output Stream (24kHz)
    try:
        stream_out = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE_OUT,
            output=True,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        logger.error(f"Failed to open output stream: {e}")
        return

    logger.info("🎤 Microphone connected")
    logger.info("🔊 Speakers connected")

    try:
        async with websockets.connect(WS_URL) as websocket:
            logger.info(f"✅ Connected to {WS_URL}")
            
            # Helper to receive audio from server
            async def receive_audio():
                try:
                    while True:
                        message = await websocket.recv()
                        if isinstance(message, bytes):
                            # Play audio immediately (no browser buffering logic)
                            stream_out.write(message)
                        else:
                            data = json.loads(message)
                            logger.info(f"📩 Server: {data}")
                            if data.get("type") == "interrupted":
                                logger.info("⚡ Interrupted - Clearing buffer (simulation)")
                                # In terminal, we can't easily clear pyaudio buffer without stopping
                                # But we can verify if the server sent the signal
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server")
                except Exception as e:
                    logger.error(f"Receive error: {e}")

            # Helper to send mic audio to server
            async def send_mic():
                loop = asyncio.get_event_loop()
                try:
                    while True:
                        # logical non-blocking read
                        data = await loop.run_in_executor(None, lambda: stream_in.read(CHUNK, exception_on_overflow=False))
                        await websocket.send(data)
                        # logger.info(f"📤 Sent {len(data)} bytes")
                        await asyncio.sleep(0.001) # Yield
                except Exception as e:
                    logger.error(f"Send error: {e}")

            # Run both
            await asyncio.gather(receive_audio(), send_mic())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()
        logger.info("❌ Disconnected")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python terminal_client.py")
        sys.exit(0)
        
    try:
        asyncio.run(audio_stream())
    except KeyboardInterrupt:
        pass
