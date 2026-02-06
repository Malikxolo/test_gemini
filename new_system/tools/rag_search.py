"""
RAG Search Tool - Internal Knowledge Base
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
        _client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=2.0), http2=True)
    return _client


async def rag_search(query: str) -> str:
    logger.info(f"RAG search: '{query}'")
    start_time = time.time()
    
    api_url = os.getenv("RAG_API_URL", "http://46.62.157.117:8001")
    api_key = os.getenv("BOT_API_KEY")
    
    if not api_key:
        return "Error: RAG search not configured (missing BOT_API_KEY)"
    
    try:
        client = _get_client()
        response = await client.post(
            f"{api_url}/api/v1/query",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json={"query": query, "top_k": 4, "use_hybrid": True, "similarity_threshold": 0.7}
        )
        
        if response.status_code != 200:
            logger.error(f"RAG search error: {response.status_code}")
            return f"Knowledge base search failed with error code {response.status_code}"
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_docs = []
        for i, r in enumerate(results[:4], 1):
            content = r.get("content", "").strip()
            if content:
                formatted_docs.append(f"--- Document {i} ---\n{content}")
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"RAG search completed: {len(results)} results in {elapsed:.0f}ms")
        return "Retrieved from knowledge base:\n\n" + "\n\n".join(formatted_docs)
        
    except asyncio.TimeoutError:
        return "Knowledge base search timed out."
    except Exception as e:
        logger.error(f"RAG search exception: {e}")
        return f"Knowledge base error: {str(e)}"


async def close():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
