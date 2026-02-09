"""
Groq LLM Client - Llama 3.3 70B with Function Calling

Streaming responses with native tool/function calling support.
Uses persistent HTTP/2 connection for ultra-low latency on Groq's LPU.
"""

import os
import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable
import httpx

logger = logging.getLogger(__name__)


# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current, real-time information like weather, news, or recent events",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the internal knowledge base for policies, documents, and company information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]


SYSTEM_PROMPT = '''You are Gemini, a voice assistant in a Google Meet meeting. You listen to conversations but only respond when explicitly addressed.

## CORE RULE
Output ONLY one of these:
1. A spoken response (1-2 sentences max)
2. The exact text: [SILENT]

## WHEN TO RESPOND (say something)
✓ Wake word detected: "Gemini", "Hey Gemini", "Jemini"
✓ Direct follow-up to YOUR last response (same topic, "what about...", "and...")

## WHEN TO STAY SILENT (output only: [SILENT])
✗ No wake word AND not a follow-up to your response
✗ People talking to each other
✗ Background conversation
✗ Single words like "okay", "hmm", "yeah", "stop"
✗ Greetings to others: "Hello everyone", "Hi John"
✗ Acknowledgments after your response: "Thanks", "Got it"

## RESPONSE STYLE
- Maximum 1-2 sentences (users are listening, not reading)
- Match the user's language
- Use tools when needed: web_search for real-time info, rag_search for internal knowledge
- If interrupted (you see "[interrupted]"), respond to the NEW message only

## EXAMPLES
User: "Hello everyone, let's begin" → [SILENT]
User: "Gemini, what's the weather?" → "It's 28 degrees and partly cloudy."
User: "What about tomorrow?" → "Tomorrow will be around 25 degrees."  
User: "Thanks. Hey Rahul, can you share?" → [SILENT]
User: "So about the project budget..." → [SILENT]'''


class LLMClient:
    """
    Groq LLM client with streaming and function calling.
    
    Uses persistent HTTP/2 connection for ultra-low latency on Groq's LPU.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._client: Optional[httpx.AsyncClient] = None
        self._tool_handlers: Dict[str, Callable] = {}
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP/2 client for Groq."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                http2=True,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client
    
    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler function."""
        self._tool_handlers[name] = handler
        logger.info(f"Registered tool: {name}")

    async def warmup(self):
        """
        Warm up the HTTP/2 connection to Groq.

        Establishes TCP + TLS + HTTP/2 negotiation upfront so the first
        real LLM request doesn't pay connection overhead.
        """
        try:
            start_time = time.time()
            client = await self._get_client()
            response = await client.get("/models", params={"per_page": "1"})
            elapsed = (time.time() - start_time) * 1000
            if response.status_code == 200:
                logger.info(f"LLM connection warmed up in {elapsed:.0f}ms")
            else:
                logger.warning(f"LLM warmup returned status {response.status_code} in {elapsed:.0f}ms")
        except Exception as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")

    async def keep_alive(self):
        """Send a lightweight request to keep the HTTP/2 connection alive."""
        try:
            client = await self._get_client()
            await client.get("/models", params={"per_page": "1"})
            logger.debug("LLM keep-alive sent")
        except Exception:
            logger.debug("LLM keep-alive failed (will reconnect on next request)")
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        use_tools: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming response with optional tool calling.
        
        Yields dicts with:
        - {"type": "text", "content": "..."} for text chunks
        - {"type": "tool_call", "name": "...", "arguments": {...}} for tool calls
        - {"type": "done", "full_text": "..."} when complete
        - {"type": "error", "content": "..."} on error
        """
        if not self.api_key:
            yield {"type": "error", "content": "OPENROUTER_API_KEY not set"}
            return
            
        client = await self._get_client()
        
        # Build request with system prompt
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        
        request_body = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        
        if use_tools:
            request_body["tools"] = TOOLS
            request_body["tool_choice"] = "auto"
        
        start_time = time.time()
        accumulated_text = ""
        tool_calls = []
        
        try:
            async with client.stream("POST", "/chat/completions", json=request_body) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    logger.error(f"LLM error {response.status_code}: {error}")
                    yield {"type": "error", "content": f"LLM error: {response.status_code}"}
                    return
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        
                        # Handle text content
                        if "content" in delta and delta["content"]:
                            text = delta["content"]
                            accumulated_text += text
                            yield {"type": "text", "content": text}
                        
                        # Handle tool calls (streaming)
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                
                                # New tool call starting
                                if tc.get("id"):
                                    while len(tool_calls) <= idx:
                                        tool_calls.append({
                                            "id": "",
                                            "name": "",
                                            "arguments": ""
                                        })
                                    tool_calls[idx]["id"] = tc["id"]
                                
                                # Function name chunk
                                if tc.get("function", {}).get("name"):
                                    if idx < len(tool_calls):
                                        tool_calls[idx]["name"] = tc["function"]["name"]
                                
                                # Arguments chunk (streamed as string)
                                if tc.get("function", {}).get("arguments"):
                                    if idx < len(tool_calls):
                                        tool_calls[idx]["arguments"] += tc["function"]["arguments"]
                    
                    except json.JSONDecodeError:
                        continue
                
                # Emit completed tool calls
                for tc in tool_calls:
                    if tc["name"]:
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            yield {
                                "type": "tool_call",
                                "id": tc.get("id", "call_1"),
                                "name": tc["name"],
                                "arguments": args
                            }
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {tc['arguments']}")
                
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"LLM response completed in {elapsed:.0f}ms")
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            yield {"type": "error", "content": str(e)}
        
        yield {"type": "done", "full_text": accumulated_text}
    
    async def execute_tool_and_continue(
        self,
        messages: List[Dict[str, str]],
        tool_call: Dict[str, Any],
        tool_result: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continue generation after tool execution.
        
        Args:
            messages: Conversation history up to tool call
            tool_call: The tool call that was executed
            tool_result: Result from tool execution
            
        Yields:
            Same format as generate_response()
        """
        # Create a copy to avoid modifying original
        messages = messages.copy()
        
        # Add assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.get("id", "call_1"),
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["arguments"])
                }
            }]
        })
        
        # Add tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.get("id", "call_1"),
            "content": tool_result
        })
        
        # Continue generation without tools (to get final response)
        async for chunk in self.generate_response(messages, use_tools=False):
            yield chunk
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("LLM client closed")
