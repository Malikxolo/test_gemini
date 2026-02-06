"""
Voice AI - Main Orchestrator

Real-time voice assistant using:
- Deepgram Nova-3 for STT
- OpenRouter Maverick 4 for LLM (with function calling)
- Deepgram Aura for TTS

Run: python voice_ai.py
"""

import os
import sys
import asyncio
import logging
import pathlib
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment from root .env
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from new_system.stt_client import DeepgramSTTClient
from new_system.llm_client import LLMClient
from new_system.tts_client import DeepgramTTSClient, SentenceBuffer
from new_system.audio_io import AudioInput, AudioOutput, list_audio_devices
from new_system.context_manager import ConversationContext
from new_system.tools.web_search import web_search
from new_system.tools.rag_search import rag_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class VoiceAI:
    """
    Main Voice AI orchestrator.
    
    Coordinates STT, LLM, and TTS for real-time voice interaction.
    """
    
    SILENT_MARKER = "[SILENT]"
    
    def __init__(self):
        # Components
        self.stt = DeepgramSTTClient()
        self.llm = LLMClient()
        self.tts = DeepgramTTSClient(voice="asteria")  # Female voice
        self.audio_in = AudioInput()
        self.audio_out = AudioOutput()
        self.context = ConversationContext(max_entries=20)
        self.sentence_buffer = SentenceBuffer()
        
        # State
        self._running = False
        self._current_transcript = ""

        # Barge-in / interruption state
        self._generation_id: int = 0
        self._current_task: Optional[asyncio.Task] = None
        
        # Register tool handlers
        self.llm.register_tool("web_search", web_search)
        self.llm.register_tool("rag_search", rag_search)
        
    async def start(self):
        """Start the voice AI system."""
        logger.info("🚀 Starting Voice AI...")

        # Check API keys
        if not os.getenv("DEEPGRAM_API_KEY"):
            logger.error("❌ DEEPGRAM_API_KEY not set in .env")
            return False
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.error("❌ OPENROUTER_API_KEY not set in .env")
            return False

        # Start audio I/O
        if not self.audio_in.start():
            logger.error("❌ Failed to start microphone")
            return False
        if not self.audio_out.start():
            logger.error("❌ Failed to start speaker")
            return False

        # Connect to Deepgram STT
        if not await self.stt.connect():
            logger.error("❌ Failed to connect to Deepgram STT")
            return False

        # Warm up LLM connection (TCP + TLS + HTTP/2 negotiation)
        await self.llm.warmup()

        # Set up STT callbacks
        self.stt.on_transcript(self._on_transcript)
        self.stt.on_utterance_end(self._on_utterance_end)

        self._running = True
        logger.info("✅ Voice AI started - say 'Gemini' to activate!")
        logger.info("   Press Ctrl+C to stop")

        # Start processing loops
        await asyncio.gather(
            self._audio_capture_loop(),
            self.stt.listen(),
            self.stt.keep_alive(),
            self._llm_keep_alive_loop(),
        )

        return True
    
    async def stop(self):
        """Stop the voice AI system."""
        logger.info("🛑 Stopping Voice AI...")
        self._running = False

        # Cancel any active processing
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass

        # Stop components
        self.audio_in.stop()
        self.audio_out.stop()
        await self.stt.disconnect()
        await self.llm.close()
        await self.tts.close()

        logger.info("👋 Voice AI stopped")
    
    async def _audio_capture_loop(self):
        """Continuously capture and send audio to STT."""
        while self._running:
            try:
                audio_chunk = await self.audio_in.get_chunk_async()
                if audio_chunk:
                    await self.stt.send_audio(audio_chunk)
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(0.1)

    async def _llm_keep_alive_loop(self):
        """Periodically ping OpenRouter to keep the HTTP/2 connection alive."""
        while self._running:
            await asyncio.sleep(45)
            try:
                await self.llm.keep_alive()
            except Exception:
                pass

    def _is_bot_active(self) -> bool:
        """Check if the bot is currently processing or playing audio."""
        task_active = self._current_task is not None and not self._current_task.done()
        audio_active = self.audio_out.is_playing
        return task_active or audio_active

    def _do_interrupt(self):
        """
        Cancel current processing and clear all output.

        Fully synchronous - safe to call from STT callbacks.
        """
        self._generation_id += 1

        # Cancel the active processing task
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # Clear audio output immediately
        self.audio_out.clear()

        # Clear sentence accumulation buffer
        self.sentence_buffer.clear()

        logger.info(f"⚡ Interrupted -> generation {self._generation_id}")
    
    def _on_transcript(self, transcript: str, is_final: bool):
        """Handle STT transcript updates."""
        if is_final:
            self._current_transcript = transcript
            logger.info(f"🗣️ User: {transcript}")
        else:
            # Show interim results
            print(f"\r💭 {transcript}...", end="", flush=True)

            # Early interruption: if user starts speaking while bot is active,
            # stop output immediately. Don't wait for utterance_end.
            # Gives ~200-300ms faster interruption response.
            if transcript.strip() and self._is_bot_active():
                self._do_interrupt()

    def _on_utterance_end(self):
        """Handle end of user utterance - trigger LLM processing."""
        if not self._current_transcript:
            return

        transcript = self._current_transcript
        self._current_transcript = ""
        print()  # New line after interim display

        # If bot is still active, perform full interruption
        if self._is_bot_active():
            self._do_interrupt()

        # Launch new processing with current generation_id
        gen_id = self._generation_id
        self._current_task = asyncio.create_task(
            self._process_utterance(transcript, gen_id)
        )
    
    async def _process_utterance(self, transcript: str, gen_id: int):
        """Process user utterance through LLM and TTS with interruption support."""
        if gen_id != self._generation_id:
            return

        full_response = ""

        try:
            # Add to context
            self.context.add_user_message(transcript)
            messages = self.context.get_messages_for_llm()

            # --- Phase 1: LLM streaming ---
            tool_call_pending = None

            async for chunk in self.llm.generate_response(messages):
                if gen_id != self._generation_id:
                    logger.debug(f"Gen {gen_id} interrupted during LLM stream")
                    break

                if chunk["type"] == "text":
                    text = chunk["content"]
                    full_response += text

                    # Check for SILENT marker
                    if self.SILENT_MARKER in full_response:
                        logger.info("🤫 Bot: [staying silent]")
                        self.context.add_bot_response("", was_silent=True)
                        return

                    # Buffer for sentence-based TTS
                    sentence = self.sentence_buffer.add(text)
                    if sentence:
                        if gen_id != self._generation_id:
                            break
                        await self._speak(sentence, gen_id)

                elif chunk["type"] == "tool_call":
                    tool_call_pending = chunk
                    logger.info(f"🔧 Tool call: {chunk['name']}({chunk['arguments']})")

                elif chunk["type"] == "done":
                    remaining = self.sentence_buffer.flush()
                    if remaining and self.SILENT_MARKER not in remaining:
                        if gen_id == self._generation_id:
                            await self._speak(remaining, gen_id)

            # --- Phase 2: Tool execution ---
            if tool_call_pending and gen_id == self._generation_id:
                tool_name = tool_call_pending["name"]
                tool_args = tool_call_pending["arguments"]

                if gen_id != self._generation_id:
                    logger.debug(f"Gen {gen_id} interrupted before tool execution")
                    return

                # Execute tool
                if tool_name == "web_search":
                    tool_result = await web_search(tool_args.get("query", ""))
                elif tool_name == "rag_search":
                    tool_result = await rag_search(tool_args.get("query", ""))
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                # Check after tool execution - discard result if interrupted
                if gen_id != self._generation_id:
                    logger.debug(f"Gen {gen_id} interrupted after tool (result discarded)")
                    return

                logger.info(f"📋 Tool result: {tool_result[:100]}...")

                # --- Phase 3: LLM continuation after tool ---
                continuation_response = ""
                async for chunk in self.llm.execute_tool_and_continue(
                    messages, tool_call_pending, tool_result
                ):
                    if gen_id != self._generation_id:
                        logger.debug(f"Gen {gen_id} interrupted during continuation")
                        break

                    if chunk["type"] == "text":
                        text = chunk["content"]
                        continuation_response += text

                        sentence = self.sentence_buffer.add(text)
                        if sentence:
                            if gen_id != self._generation_id:
                                break
                            await self._speak(sentence, gen_id)

                    elif chunk["type"] == "done":
                        remaining = self.sentence_buffer.flush()
                        if remaining:
                            if gen_id == self._generation_id:
                                await self._speak(remaining, gen_id)

                full_response += continuation_response

            # --- Phase 4: Save to context ---
            if gen_id == self._generation_id:
                if full_response and self.SILENT_MARKER not in full_response:
                    self.context.add_bot_response(full_response)
                    logger.info(f"🤖 Bot: {full_response[:100]}...")

        except asyncio.CancelledError:
            logger.info(f"⚡ Processing cancelled (gen {gen_id})")
            if full_response.strip() and self.SILENT_MARKER not in full_response:
                self.context.add_bot_response(full_response.strip() + " [interrupted]")

        except Exception as e:
            logger.error(f"Processing error: {e}")

        finally:
            self.sentence_buffer.clear()
    
    async def _speak(self, text: str, gen_id: int):
        """Convert text to speech and queue for playback."""
        if not text.strip() or gen_id != self._generation_id:
            return

        logger.debug(f"🔊 Speaking: {text}")

        try:
            async for audio_chunk in self.tts.synthesize(text):
                if gen_id != self._generation_id:
                    return  # Interrupted during TTS streaming
                self.audio_out.play(audio_chunk)
        except asyncio.CancelledError:
            raise  # Propagate to _process_utterance handler
        except Exception as e:
            logger.error(f"TTS error: {e}")


async def main():
    """Main entry point."""
    print("\n" + "="*50)
    print("   VOICE AI - Real-Time Assistant")
    print("="*50)
    print("\nSay 'Gemini' followed by your question!")
    print("Examples:")
    print("  - 'Gemini, what's the weather in London?'")
    print("  - 'Gemini, search for latest AI news'")
    print("\n")
    
    # List audio devices
    list_audio_devices()
    print()
    
    voice_ai = VoiceAI()
    
    try:
        await voice_ai.start()
    except KeyboardInterrupt:
        print("\n")
        logger.info("Interrupted by user")
    finally:
        await voice_ai.stop()


if __name__ == "__main__":
    asyncio.run(main())
