"""
LLM Router with cost optimization.
Routes to cheapest model that can handle the task.
"""
import os
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import anthropic


class LLMRouter:
    """Intelligent LLM routing with cost optimization."""

    MODELS = {
        'gpt-4o': {
            'provider': 'openai',
            'input_cost': 0.005,
            'output_cost': 0.015,
            'max_tokens': 128000,
        },
        'gpt-4o-mini': {
            'provider': 'openai',
            'input_cost': 0.00015,
            'output_cost': 0.0006,
            'max_tokens': 128000,
        },
        'claude-3-sonnet': {
            'provider': 'anthropic',
            'input_cost': 0.003,
            'output_cost': 0.015,
            'max_tokens': 200000,
        },
        'claude-3-haiku': {
            'provider': 'anthropic',
            'input_cost': 0.00025,
            'output_cost': 0.00125,
            'max_tokens': 200000,
        },
    }

    def __init__(self):
        self.openai = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.anthropic = anthropic.AsyncAnthropic(
            api_key=os.environ.get('ANTHROPIC_API_KEY')
        )

    async def classify_complexity(self, prompt: str) -> str:
        """Classify prompt complexity using cheap model."""
        response = await self.openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'system',
                    'content': 'Classify the complexity of this task as LOW, MEDIUM, or HIGH. '
                    'Consider: reasoning depth, domain knowledge, creativity required. '
                    'Respond with only: LOW, MEDIUM, or HIGH',
                },
                {'role': 'user', 'content': prompt[:1000]},
            ],
            max_tokens=10,
            temperature=0,
        )

        return response.choices[0].message.content.strip().lower()

    def select_model_for_complexity(self, complexity: str) -> str:
        """Select cheapest model for complexity level."""
        mapping = {
            'low': 'gpt-4o-mini',
            'medium': 'claude-3-haiku',
            'high': 'claude-3-sonnet',
        }
        return mapping.get(complexity, 'gpt-4o-mini')

    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for model usage."""
        if model not in self.MODELS:
            return 0.0

        config = self.MODELS[model]
        cost = (
            input_tokens / 1000 * config['input_cost']
            + output_tokens / 1000 * config['output_cost']
        )
        return cost
