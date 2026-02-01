"""Tool registry with security checks."""
import os
import httpx
from typing import Dict, Any, List
from langchain.tools import Tool


class ToolRegistry:
    """Secure tool registry with input validation."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._load_builtin_tools()

    def _load_builtin_tools(self):
        """Load built-in tools with security checks."""

        # Web Search Tool
        def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
            """Search the web for information using Serper.dev API."""
            try:
                # Validate inputs
                if not query or len(query) > 500:
                    return {'results': [], 'error': 'Invalid query length'}

                num_results = max(1, min(num_results, 10))

                # Call Serper.dev API
                serper_api_key = os.environ.get('SERPER_API_KEY', '')
                if not serper_api_key:
                    return {'results': [], 'error': 'SERPER_API_KEY not configured'}

                response = httpx.post(
                    'https://google.serper.dev/search',
                    headers={
                        'X-API-KEY': serper_api_key,
                        'Content-Type': 'application/json'
                    },
                    json={'q': query, 'num': num_results},
                    timeout=10.0
                )

                if response.status_code != 200:
                    return {'results': [], 'error': f'Search API error: {response.status_code}'}

                data = response.json()

                return {
                    'results': [
                        {
                            'title': r.get('title'),
                            'link': r.get('link'),
                            'snippet': r.get('snippet'),
                        }
                        for r in data.get('organic', [])[:num_results]
                    ],
                    'query': query,
                }
            except Exception as e:
                return {'results': [], 'error': str(e)}

        # Calculator Tool
        def calculator(expression: str) -> float:
            """Evaluate a mathematical expression."""
            try:
                # Safe evaluation (only allows math operations)
                allowed_chars = set('0123456789+-*/(). ')
                if not all(c in allowed_chars for c in expression):
                    raise ValueError('Invalid characters in expression')
                return eval(expression)
            except Exception as e:
                return f'Error: {str(e)}'

        # Register tools
        self._tools['calculator'] = Tool(
            name='calculator',
            func=calculator,
            description='Useful for performing mathematical calculations. Input should be a valid mathematical expression.',
        )

        self._tools['web_search'] = Tool(
            name='web_search',
            func=web_search,
            description='Search the web for current information using Google. Input should be a search query string.',
        )

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f'Tool {name} not found')
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self._tools.keys())
