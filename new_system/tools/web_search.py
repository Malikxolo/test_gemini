"""
Web Search Tool using Perplexity via OpenRouter
"""

import os
import asyncio
import logging
import time
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            http2=True,
            timeout=httpx.Timeout(15.0, connect=5.0),
        )
    return _client


async def web_search(query: str) -> str:
    logger.info(f"Web search: '{query}'")
    start_time = time.time()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: Web search not configured (missing OPENROUTER_API_KEY)"
    
    try:
        client = _get_client()
        response = await client.post(
            "/chat/completions",
            json={
                "model": "perplexity/sonar",
                "messages": [{"role": "user", "content": f"Search and provide a concise answer (2-3 sentences): {query}"}],
                "max_tokens": 300,
                "temperature": 0.1,
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Web search error: {response.status_code}")
            return f"Search failed with error code {response.status_code}"
        
        data = response.json()
        result = data["choices"][0]["message"]["content"]
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Web search completed in {elapsed:.0f}ms")
        return result
        
    except asyncio.TimeoutError:
        return "Search timed out. Please try again."
    except Exception as e:
        logger.error(f"Web search exception: {e}")
        return f"Search error: {str(e)}"


async def close():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
