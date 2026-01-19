"""
Gemini Live Audio Streaming Backend
FastAPI server using google-genai SDK with binary WebSocket for ultra-low latency

Features:
- Google GenAI SDK for stable connection management
- Binary WebSocket for audio (no JSON/base64 overhead)
- JSON messages for control signals only
- Native 24kHz output (no resampling)
- Perplexity Sonar web search via OpenRouter (cost-effective)
"""

import os
import asyncio
import logging
import time
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google import genai
from google.genai import types
import openai

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemini Live Audio",
    description="Real-time bidirectional audio streaming with Gemini Live API",
    version="2.1.0"
)

# Enable CORS for split hosting (Frontend on Vercel, Backend on Render/Railway)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your Vercel domain later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

# Initialize OpenRouter client for Perplexity
openrouter_client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Gemini Live API Configuration
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"


async def search_with_perplexity(query: str) -> str:
    """
    Call Perplexity Sonar via OpenRouter for web search.
    NOTE: Streaming disabled - Gemini Live API requires complete tool responses,
    so streaming tokens gives no benefit. Non-streaming is actually slightly faster
    due to less overhead.
    """
    logger.info(f"🔍 Perplexity search: {query}")
    start_time = time.time()
    
    try:
        # Non-streaming call - Gemini needs complete response anyway
        response = await asyncio.wait_for(
            openrouter_client.chat.completions.create(
                model="perplexity/sonar",
                messages=[{
                    "role": "user",
                    "content": f"Search the web and provide a brief, factual answer in 2-3 sentences: {query}"
                }],
                max_tokens=200  # Reduced for faster response
            ),
            timeout=10.0  # 10 second timeout
        )
        
        result = response.choices[0].message.content
        total_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Perplexity complete: {total_time:.0f}ms, {len(result)} chars")
        logger.info(f"📝 Result: {result[:150]}...")
        return result
        
    except asyncio.TimeoutError:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"❌ Perplexity timeout after {elapsed:.0f}ms")
        return "Search timed out. Please try again."
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"❌ Perplexity error after {elapsed:.0f}ms: {e}")
        return f"Search failed: {str(e)}"


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL,
        "version": "2.0.0"
    }


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional audio streaming.
    - Binary frames: PCM audio data
    - Text frames: JSON control messages
    """
    await client_ws.accept()
    logger.info("Client connected")
    
    # Connection state
    is_connected = True
    
    # Latency tracking
    user_speech_start_time = None
    first_response_time = None
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Define custom web search tool (uses Perplexity instead of Google)
        web_search_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="web_search",
                    description="Search the web for current, real-time information. Use this when you need up-to-date data, news, weather, sports scores, stock prices, or any information that may have changed recently.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="The search query to look up on the web"
                            )
                        },
                        required=["query"]
                    )
                )
            ]
        )
        
        # Configure Live API session (SDK format)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction="""You are a helpful and friendly AI assistant. Respond naturally and conversationally in 1-2 sentences. Be concise.

When you need current information (news, weather, sports, stocks, etc.), use the 'web_search' tool.

CRITICAL INSTRUCTION FOR WEB SEARCHES:
Before calling the web_search function, you MUST speak a natural filler phrase to keep the conversation flowing. The filler should be 2-3 sentences to cover the search time. Examples:
- "Accha, yeh toh interesting question hai! Main abhi check karti hoon aapke liye... ek second..."
- "Hmm, let me look that up for you right now. Just give me a moment to find the latest information..."
- "Oh that's a great question! I'm searching for the most up-to-date information on that... one moment please..."
- "Zaroor! Main internet se latest information dhundh rahi hoon aapke liye..."

Speak the filler phrase COMPLETELY first, then call web_search. This prevents awkward silence while searching.""",
            tools=[web_search_tool],  # Custom web search using Perplexity
            # Context window compression to prevent latency increase over time
            # When context exceeds trigger_tokens, older content is compressed
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(
                    target_tokens=16000  # Keep ~16k tokens, compress older content
                ),
                trigger_tokens=20000  # Start compressing when hitting 20k tokens
            ),
            generation_config=types.GenerationConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable reasoning for speed
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Zephyr"
                        )
                    )
                )
            ),
            # Native VAD Configuration
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=200
                )
            )
        )
        
        # Connect to Gemini Live API using SDK
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            logger.info("Connected to Gemini Live API (SDK mode)")
            
            # Notify client
            await client_ws.send_json({"type": "connected", "message": "Connected to Gemini Live"})
            
            async def receive_from_client():
                """Receive audio from client and forward to Gemini"""
                nonlocal is_connected, user_speech_start_time
                chunk_count = 0
                
                try:
                    while is_connected:
                        message = await client_ws.receive()
                        
                        if "bytes" in message:
                            # Binary PCM16 audio - direct to Gemini
                            pcm_data = message["bytes"]
                            chunk_count += 1
                            
                            # Track speech start (first audio chunk)
                            if user_speech_start_time is None:
                                user_speech_start_time = time.time()
                                logger.info(f"🎤 User started speaking (chunk #{chunk_count})")
                            
                            # Send to Gemini using SDK (audio/pcm format)
                            await session.send_realtime_input(
                                audio={"data": pcm_data, "mime_type": "audio/pcm"}
                            )
                            
                        elif "text" in message:
                            # JSON control message
                            data = json.loads(message["text"])
                            if data.get("type") == "end":
                                is_connected = False
                                break
                                
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                    is_connected = False
                except Exception as e:
                    logger.error(f"Error receiving from client: {e}")
                    is_connected = False
            
            async def receive_from_gemini():
                """Receive audio responses from Gemini and send to client"""
                nonlocal is_connected, first_response_time, user_speech_start_time
                audio_chunk_count = 0
                
                try:
                    while is_connected:
                        turn = session.receive()
                        async for response in turn:
                            if not is_connected:
                                break
                            
                            # Handle function calls (web_search)
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    if fc.name == "web_search":
                                        query = fc.args.get("query", "")
                                        logger.info(f"🔧 Function call: web_search('{query}')")
                                        
                                        # Call Perplexity via OpenRouter
                                        search_result = await search_with_perplexity(query)
                                        
                                        # Send tool response back to Gemini
                                        # Use session.send() with the tool_response message directly
                                        function_response = types.FunctionResponse(
                                            name="web_search",
                                            id=fc.id,
                                            response={"result": search_result}
                                        )
                                        await session.send(input=types.LiveClientToolResponse(
                                            function_responses=[function_response]
                                        ))
                                        logger.info("📨 Sent tool response to Gemini")
                            
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                                        audio_chunk_count += 1
                                        
                                        # Measure latency on first audio chunk
                                        if first_response_time is None and user_speech_start_time:
                                            first_response_time = time.time()
                                            latency_ms = (first_response_time - user_speech_start_time) * 1000
                                            logger.info(f"⚡ LATENCY: {latency_ms:.0f}ms (speech → first audio)")
                                        
                                        if audio_chunk_count == 1:
                                            logger.info("🔊 First audio chunk received")
                                        
                                        # Send binary PCM directly (24kHz from Gemini)
                                        audio_bytes = part.inline_data.data
                                        logger.info(f"📤 Sending audio chunk {audio_chunk_count}: {len(audio_bytes)} bytes")
                                        await client_ws.send_bytes(audio_bytes)
                            
                            # Handle turn complete
                            if response.server_content and response.server_content.turn_complete:
                                logger.info(f"✓ Turn complete ({audio_chunk_count} audio chunks)")
                                await client_ws.send_json({"type": "turnComplete"})
                                # Reset for next turn
                                first_response_time = None
                                user_speech_start_time = None
                                audio_chunk_count = 0
                            
                            # Log token usage to monitor context growth
                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                usage = response.usage_metadata
                                logger.info(f"📊 Tokens: {usage.total_token_count} total")
                            
                            # Handle interruption
                            if response.server_content and response.server_content.interrupted:
                                logger.info("⚡ User interrupted - VAD triggered")
                                await client_ws.send_json({"type": "interrupted"})
                                first_response_time = None
                                user_speech_start_time = None
                                audio_chunk_count = 0
                                
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")
                    is_connected = False
            
            # Run both tasks concurrently
            await asyncio.gather(
                receive_from_client(),
                receive_from_gemini(),
                return_exceptions=True
            )
    
    except asyncio.CancelledError:
        logger.info("Connection cancelled")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await client_ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await client_ws.close()
        except:
            pass
        logger.info("🔌 Connection closed")


# Serve frontend
if os.path.exists("../frontend"):
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
