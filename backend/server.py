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
import pathlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
import openai
from rag_tool import RAGTool

# Load env from root directory
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

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

# Initialize RAG Tool
rag_tool = RAGTool()

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
    
    # Session resumption for long meetings (30+ minutes)
    # Token is stored per-connection and can be used to resume if WebSocket drops
    session_resume_handle = None
    
    # Latency tracking
    user_speech_start_time = None
    first_response_time = None
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Define custom web search tool (uses Perplexity instead of Google)
        # NON_BLOCKING allows Gemini to continue speaking while search runs
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
                    ),
                    behavior="NON_BLOCKING"  # Async: Gemini continues while search runs
                )
            ]
        )
        
        # Define RAG search tool
        # NON_BLOCKING allows Gemini to continue speaking while RAG runs
        rag_search_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="rag_search",
                    description="Search the internal knowledge base for policies, guidelines, and specific documents. Use this for questions about 'grievance policies', 'company rules', or other internal matters.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="The search query to look up in the knowledge base"
                            )
                        },
                        required=["query"]
                    ),
                    behavior="NON_BLOCKING"  # Async: Gemini continues while RAG runs
                )
            ]
        )
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction="""You are a conversational AI voice assistant. You speak naturally and help users in real-time.

## CRITICAL RULE: STOP MEANS SILENCE

When the user says ANYTHING that means "stop talking" - including but not limited to:
- "stop", "ruko", "bas", "chup", "quiet", "shut up", "okay okay", "wait", "hold on"
- Or simply starts talking over you / interrupts you

**YOUR RESPONSE: ABSOLUTE SILENCE.**
- Do NOT say "okay", "theek hai", "main ruk gaya", "alright", "sure" or ANY acknowledgment
- Do NOT say anything at all
- Just stop and wait silently
- The next thing you say should ONLY be in response to their next actual question

This is non-negotiable. A stop command is NOT a conversation turn. It requires ZERO verbal response.

## LANGUAGE RULE

Match the user's language exactly. If they speak Hindi, respond in Hindi. English → English. Hinglish → Hinglish.

## TOOL USAGE

You have two tools:
- `web_search`: For weather, news, sports, stocks, current events, anything real-time
- `rag_search`: For internal policies, documents, company knowledge

### CRITICAL RULES:

1. **Call tool ONCE per question** - never call the same tool twice for one user query
2. **Say a natural filler while waiting** - when you call a tool, say something brief and natural like:
   - "hmm, ek second, main dekh ke batati hu..."
   - "achha, ruko zara check karti hu..."
   - "let me check that for you..."
   Then STOP speaking and wait for results.
3. **NEVER make up information** - while waiting for tool results, do NOT guess or speak any data. Only speak filler, nothing else.
4. **When results arrive, use them** - the tool response will interrupt your filler. Immediately speak the real data from the tool.

### What NOT to do:
- Do NOT say things like "the weather is around 30 degrees" before getting results - this is hallucination
- Do NOT continue speaking after the filler - just wait silently for results
- Do NOT call the tool again if you already called it

### When tool results arrive:
- Extract 2-3 key points the user needs
- Speak conversationally, not as a data dump
- Don't read JSON or raw data verbatim

## RESPONSE LENGTH

- Default: 1-2 sentences
- Only give longer responses if explicitly asked for details
- Voice conversation should be quick back-and-forth, not monologues
""",
            tools=[web_search_tool, rag_search_tool],  # Web search + RAG search
            # SESSION RESUMPTION for 30-minute meetings
            # - WebSocket resets every ~10 min, this allows seamless reconnection
            # - Tokens valid for 2 hours after session ends
            # - Server sends SessionResumptionUpdate with new handle periodically
            session_resumption=types.SessionResumptionConfig(
                handle=None  # None = new session, or pass previous handle to resume
            ),
            # Context window compression - optimized for 30-min meetings with 9-10 people
            # Native audio models have 128k token limit
            # With multiple speakers, context grows fast - compress aggressively
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(
                    target_tokens=8000   # After compression, keep ~8k tokens (enough for recent context)
                ),
                trigger_tokens=12000     # Trigger at 12k to avoid frequent compression spikes
            ),
            # IMPORTANT: thinking_config and speech_config are TOP-LEVEL in LiveConnectConfig
            # (generation_config is deprecated for Live API - fields must be set directly)
            thinking_config=types.ThinkingConfig(
                thinking_budget=0  # Disable thinking/reasoning for minimal latency
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Zephyr"
                    )
                )
            ),
            # Generation parameters for speed (lower = faster, more deterministic)
            temperature=1.0,  # Default for Gemini, balanced
            # top_p=0.95,       # Slightly focused for faster token selection
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
                
                # Track processed function calls to prevent duplicate execution
                # Gemini Live API with NON_BLOCKING can send multiple calls for the same query
                processed_function_call_ids = set()  # Track by ID
                pending_queries = {}  # Track by query string -> timestamp (to dedupe same queries within a window)
                QUERY_DEDUPE_WINDOW_SECONDS = 10  # Don't repeat same query within 10 seconds
                
                # Helper to run tool in background and send response when done
                async def execute_tool_in_background(fc, tool_name, tool_func, query):
                    """Execute tool and send response - runs as background task"""
                    try:
                        logger.info(f"🚀 Starting background {tool_name}: '{query}'")
                        result = await tool_func(query)
                        
                        # Send tool response back to Gemini
                        # IMPORTANT: scheduling is TOP-LEVEL in FunctionResponse, NOT inside response dict
                        # Per API docs: { id, name, response: {result}, scheduling: enum }
                        function_response = types.FunctionResponse(
                            name=tool_name,
                            id=fc.id,
                            response={"result": result},
                            scheduling="WHEN_IDLE"  # Wait for model to finish current speech, then use result
                        )
                        await session.send(input=types.LiveClientToolResponse(
                            function_responses=[function_response]
                        ))
                        logger.info(f"📨 Sent {tool_name} response to Gemini (background task complete)")
                    except Exception as e:
                        logger.error(f"❌ Background {tool_name} error: {e}")
                
                try:
                    while is_connected:
                        turn = session.receive()
                        async for response in turn:
                            if not is_connected:
                                break
                            
                            # Handle function calls - spawn background task, don't await!
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    # Skip if we've already processed this exact function call ID
                                    if fc.id in processed_function_call_ids:
                                        logger.debug(f"⏭️ Skipping duplicate function call ID: {fc.name} (id: {fc.id})")
                                        continue
                                    
                                    # Mark ID as processed
                                    processed_function_call_ids.add(fc.id)
                                    
                                    if fc.name == "web_search":
                                        query = fc.args.get("query", "")
                                        
                                        # Check if we've already searched this query recently (dedupe by content)
                                        query_key = f"web_search:{query.lower().strip()}"
                                        current_time = time.time()
                                        if query_key in pending_queries:
                                            elapsed = current_time - pending_queries[query_key]
                                            if elapsed < QUERY_DEDUPE_WINDOW_SECONDS:
                                                logger.info(f"⏭️ Skipping duplicate web_search query: '{query}' (already called {elapsed:.1f}s ago)")
                                                continue
                                        
                                        # Mark query as pending
                                        pending_queries[query_key] = current_time
                                        
                                        logger.info(f"🔧 Function call: web_search('{query}') - spawning background task")
                                        
                                        # Create background task - DON'T AWAIT, let it run in parallel
                                        asyncio.create_task(
                                            execute_tool_in_background(fc, "web_search", search_with_perplexity, query)
                                        )
                                        # Loop continues immediately - Gemini keeps streaming!

                                    elif fc.name == "rag_search":
                                        query = fc.args.get("query", "")
                                        
                                        # Check if we've already searched this query recently (dedupe by content)
                                        query_key = f"rag_search:{query.lower().strip()}"
                                        current_time = time.time()
                                        if query_key in pending_queries:
                                            elapsed = current_time - pending_queries[query_key]
                                            if elapsed < QUERY_DEDUPE_WINDOW_SECONDS:
                                                logger.info(f"⏭️ Skipping duplicate rag_search query: '{query}' (already called {elapsed:.1f}s ago)")
                                                continue
                                        
                                        # Mark query as pending
                                        pending_queries[query_key] = current_time
                                        
                                        logger.info(f"📚 Function call: rag_search('{query}') - spawning background task")
                                        
                                        # Create background task - DON'T AWAIT, let it run in parallel
                                        asyncio.create_task(
                                            execute_tool_in_background(fc, "rag_search", rag_tool.execute, query)
                                        )
                                        # Loop continues immediately - Gemini keeps streaming!
                            
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
                            
                            # Log token usage to monitor context growth (key for latency debugging)
                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                usage = response.usage_metadata
                                total = usage.total_token_count
                                # Warn if tokens approaching compression trigger (12k)
                                if total > 10000:
                                    logger.warning(f"⚠️ Tokens: {total} (HIGH - compression soon)")
                                elif total > 8000:
                                    logger.info(f"📊 Tokens: {total} (growing)")
                                else:
                                    logger.info(f"📊 Tokens: {total} total")
                            
                            # Handle SESSION RESUMPTION updates (for 30-min meetings)
                            # Server periodically sends new handles that can be used to resume
                            if hasattr(response, 'session_resumption_update') and response.session_resumption_update:
                                update = response.session_resumption_update
                                if update.resumable and update.new_handle:
                                    session_resume_handle = update.new_handle
                                    logger.info(f"� Session resumption handle updated (valid for 2hr)")
                            
                            # Handle GO_AWAY message (server warning before disconnect)
                            # This gives us time to prepare for reconnection
                            if hasattr(response, 'go_away') and response.go_away:
                                time_left = response.go_away.time_left
                                logger.warning(f"⏰ GO_AWAY received - connection ending in {time_left}")
                                # Notify client about impending reconnection
                                await client_ws.send_json({
                                    "type": "sessionWarning", 
                                    "message": f"Connection resetting soon, will auto-resume"
                                })
                            
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
