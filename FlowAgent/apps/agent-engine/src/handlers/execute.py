import json
import os
import time
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..models.llm_router import LLMRouter
from ..tools.registry import ToolRegistry
from ..db import get_db_connection, update_execution_status


def handler(event, context):
    """
    Lambda handler for agent execution.
    Triggered by API or queue.
    """
    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        execution_id = body.get('executionId')
        agent_id = body.get('agentId')
        user_id = body.get('userId')
        input_data = body.get('input')
        agent_config = body.get('agentConfig', {})

        # Update status to running
        update_execution_status(execution_id, 'running', started_at=datetime.utcnow())

        # Initialize LLM router
        router = LLMRouter()

        # Get appropriate model
        model_name = agent_config.get('model', 'gpt-4o-mini')
        temperature = float(agent_config.get('temperature', 0.7))

        # Use user's BYOK keys if provided, otherwise use platform keys
        user_api_keys = body.get('userApiKeys', {})
        user_openai_key = user_api_keys.get('openaiApiKey')
        user_anthropic_key = user_api_keys.get('anthropicApiKey')

        if 'gpt' in model_name:
            api_key = user_openai_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError('OpenAI API key not configured')
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        elif 'claude' in model_name:
            api_key = user_anthropic_key or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError('Anthropic API key not configured')
            llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        else:
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

        # Initialize tools
        tool_registry = ToolRegistry()
        tool_names = agent_config.get('tools', [])
        tools = [tool_registry.get_tool(name) for name in tool_names]

        # Create prompt
        system_prompt = agent_config.get('systemPrompt', 'You are a helpful AI assistant.')
        prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ('human', '{input}'),
            MessagesPlaceholder(variable_name='agent_scratchpad'),
        ])

        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        # Execute
        start_time = time.time()

        result = agent_executor.invoke({
            'input': input_data.get('message', '')
        })

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract token usage (if available from LangChain callbacks)
        input_tokens = 0
        output_tokens = 0

        # LangChain stores usage metadata in result when available
        if hasattr(result, 'usage_metadata'):
            usage = result.usage_metadata
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        elif isinstance(result, dict) and 'usage_metadata' in result:
            usage = result['usage_metadata']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

        # Calculate cost
        estimated_cost = router.calculate_cost(model_name, input_tokens, output_tokens)

        # Update execution status
        update_execution_status(
            execution_id,
            'completed',
            output={'result': result.get('output', '') if isinstance(result, dict) else str(result)},
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'executionId': execution_id,
                'status': 'completed',
                'result': result.get('output', '') if isinstance(result, dict) else str(result),
                'durationMs': duration_ms,
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'estimatedCost': estimated_cost,
            })
        }

    except Exception as e:
        print(f'Error executing agent: {str(e)}')

        # Update execution status
        if 'execution_id' in locals():
            update_execution_status(
                execution_id,
                'failed',
                error=str(e),
                completed_at=datetime.utcnow()
            )

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': str(e),
                'executionId': execution_id if 'execution_id' in locals() else None,
            })
        }
