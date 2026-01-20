import os
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional
import pathlib
from dotenv import load_dotenv

# Load env from root directory
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTool:
    """
    RAG tool using Weaviate Bot API.
    Adapted for backend integration.
    """
    def __init__(self, api_url: str = None, api_key: str = None):
        """
        Initialize the RAG Tool.
        
        Args:
            api_url: The base URL of the Weaviate RAG API
            api_key: The Bot API Key (starts with "rag_")
        """
        self.name = "rag_search" # Renamed for clarity in tool definition
        self.description = "Retrieve information from uploaded knowledge base (policies, internal docs, etc.)"
        
        # Load from env if not provided
        self.api_url = api_url or os.getenv("RAG_API_URL", "http://46.62.157.117:8001")
        self.api_key = api_key or os.getenv("BOT_API_KEY")
        
        if not self.api_key:
            logger.warning("RAGTool initialized without BOT_API_KEY. usage will fail.")
        else:
            logger.info(f"RAGTool initialized with URL: {self.api_url}")
    
    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute RAG query using the Bot API and return a formatted string for the LLM.
        """
        logger.info(f"🔍 RAG query started: '{query}'")
        start_time = time.time()
        
        if not self.api_key:
            return "Error: RAG tool not configured (missing BOT_API_KEY)."
            
        try:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "top_k": 5,
                "use_hybrid": True,   # Enable hybrid search for better recall
                "similarity_threshold": 0.7  # Lower threshold
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/api/v1/query"
                # Add timeout to prevent infinite hangs
                async with asyncio.timeout(10.0):
                    async with session.post(url, json=payload, headers=headers) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"RAG API Error {response.status}: {error_text}")
                            return f"Error retrieving information: HTTP {response.status}"
                        
                        data = await response.json()
                        
                        # Parse results
                        results = data.get("results", [])
                        total_results = data.get("total_results", 0)
                        
                        if not results:
                            elapsed = (time.time() - start_time) * 1000
                            logger.info(f"RAG query returned no results ({elapsed:.0f}ms)")
                            return "No relevant information found in the knowledge base."

                        elapsed = (time.time() - start_time) * 1000
                        logger.info(f"✅ RAG query SUCCESS: {total_results} chunks in {elapsed:.0f}ms")
                        
                        # Format documents for the LLM
                        # We want to give the model the content clearly
                        formatted_docs = []
                        for i, r in enumerate(results, 1):
                            content = r.get("content", "").strip()
                            if content:
                                formatted_docs.append(f"--- Document {i} ---\n{content}")
                        
                        final_output = f"Ensure you answer the user's question based on the following retrieved information:\n\n" + "\n\n".join(formatted_docs)
                        
                        return final_output
                    
        except Exception as e:
            logger.error(f"❌ RAG query EXCEPTION: {str(e)}")
            return f"Error executing RAG search: {str(e)}"
