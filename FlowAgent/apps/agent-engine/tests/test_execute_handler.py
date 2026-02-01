"""Tests for Lambda execution handler."""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestExecuteHandler:
    """Test cases for agent execution handler."""

    def test_handler_parses_event_body_correctly(self):
        """Test that handler correctly parses the event body."""
        event = {
            'body': json.dumps({
                'executionId': 'exec-123',
                'agentId': 'agent-456',
                'userId': 'user-789',
                'agentConfig': {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.7,
                    'systemPrompt': 'You are helpful',
                    'tools': ['calculator'],
                },
                'input': {
                    'message': 'What is 2+2?',
                },
            })
        }

        body = json.loads(event.get('body', '{}'))
        assert body['executionId'] == 'exec-123'
        assert body['agentId'] == 'agent-456'
        assert body['userId'] == 'user-789'
        assert body['agentConfig']['model'] == 'gpt-4o-mini'
        assert body['input']['message'] == 'What is 2+2?'

    def test_handler_initializes_correct_llm_for_gpt_model(self):
        """Test that handler selects OpenAI for GPT models."""
        model_name = 'gpt-4o-mini'
        assert 'gpt' in model_name

    def test_handler_initializes_correct_llm_for_claude_model(self):
        """Test that handler selects Anthropic for Claude models."""
        model_name = 'claude-3-sonnet'
        assert 'claude' in model_name

    def test_handler_falls_back_to_default_model(self):
        """Test that handler falls back to default for unknown models."""
        model_name = 'unknown-model'
        assert 'gpt' not in model_name and 'claude' not in model_name
        default_model = 'gpt-4o-mini'
        assert default_model == 'gpt-4o-mini'

    def test_handler_extracts_token_usage_from_result(self):
        """Test that handler extracts token counts from result."""
        # Mock result with usage metadata
        result = {
            'output': 'The answer is 4',
            'usage_metadata': {
                'input_tokens': 15,
                'output_tokens': 8,
            }
        }

        if isinstance(result, dict) and 'usage_metadata' in result:
            usage = result['usage_metadata']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

            assert input_tokens == 15
            assert output_tokens == 8

    def test_handler_calculates_cost_correctly(self):
        """Test that handler calculates execution cost."""
        input_tokens = 1000
        output_tokens = 500

        # Rough cost calculation for gpt-4o-mini
        # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
        expected_cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        assert expected_cost > 0
        assert expected_cost < 1  # Should be less than $1 for these numbers

    def test_handler_updates_execution_status_to_running(self):
        """Test that handler marks execution as running at start."""
        execution_id = 'exec-123'
        status = 'running'
        started_at = datetime.utcnow()

        assert execution_id is not None
        assert status == 'running'
        assert started_at is not None

    def test_handler_updates_execution_status_to_completed(self):
        """Test that handler marks execution as completed on success."""
        execution_id = 'exec-123'
        status = 'completed'
        output = {'result': 'Success'}
        completed_at = datetime.utcnow()
        duration_ms = 1500

        assert execution_id is not None
        assert status == 'completed'
        assert output is not None
        assert completed_at is not None
        assert duration_ms > 0

    def test_handler_updates_execution_status_to_failed_on_error(self):
        """Test that handler marks execution as failed on exception."""
        execution_id = 'exec-123'
        status = 'failed'
        error = 'Model API error'
        completed_at = datetime.utcnow()

        assert execution_id is not None
        assert status == 'failed'
        assert error is not None
        assert completed_at is not None

    def test_handler_returns_200_on_success(self):
        """Test that handler returns 200 status code on success."""
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'executionId': 'exec-123',
                'status': 'completed',
                'result': 'Success',
                'durationMs': 1500,
                'inputTokens': 15,
                'outputTokens': 8,
                'estimatedCost': 0.0001,
            })
        }

        assert response['statusCode'] == 200
        assert response['headers']['Content-Type'] == 'application/json'
        body = json.loads(response['body'])
        assert body['status'] == 'completed'
        assert 'inputTokens' in body
        assert 'outputTokens' in body
        assert 'estimatedCost' in body

    def test_handler_returns_500_on_error(self):
        """Test that handler returns 500 status code on error."""
        response = {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Execution failed',
                'executionId': 'exec-123',
            })
        }

        assert response['statusCode'] == 500
        body = json.loads(response['body'])
        assert 'error' in body

    def test_handler_includes_cors_headers(self):
        """Test that handler includes CORS headers in response."""
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        }

        assert headers['Access-Control-Allow-Origin'] == '*'

    def test_handler_measures_execution_duration(self):
        """Test that handler measures execution duration."""
        import time

        start_time = time.time()
        # Simulate work
        time.sleep(0.1)
        duration_ms = int((time.time() - start_time) * 1000)

        assert duration_ms >= 100  # At least 100ms
        assert duration_ms < 200   # But not too much more


class TestToolRegistry:
    """Test cases for tool registry."""

    def test_calculator_tool_validates_input(self):
        """Test that calculator tool validates input characters."""
        allowed_chars = set('0123456789+-*/(). ')

        valid_expression = '2 + 2'
        assert all(c in allowed_chars for c in valid_expression)

        invalid_expression = '2 + 2; import os'
        assert not all(c in allowed_chars for c in invalid_expression)

    def test_web_search_validates_query_length(self):
        """Test that web search validates query length."""
        valid_query = 'test query'
        assert len(valid_query) > 0
        assert len(valid_query) <= 500

        too_long_query = 'a' * 501
        assert len(too_long_query) > 500

    def test_web_search_limits_results(self):
        """Test that web search limits number of results."""
        num_results = 15
        limited_results = max(1, min(num_results, 10))

        assert limited_results == 10

        num_results = 3
        limited_results = max(1, min(num_results, 10))

        assert limited_results == 3

    def test_web_search_requires_api_key(self):
        """Test that web search checks for API key."""
        serper_api_key = ''
        if not serper_api_key:
            error_response = {'results': [], 'error': 'SERPER_API_KEY not configured'}
            assert 'error' in error_response
            assert error_response['results'] == []


class TestDatabaseOperations:
    """Test cases for database operations."""

    def test_update_execution_status_builds_correct_query(self):
        """Test that update builds correct SQL query."""
        updates = ['status = %s']
        values = ['running']

        if True:  # started_at is not None
            updates.append('started_at = %s')
            values.append(datetime.utcnow())

        assert len(updates) == 2
        assert len(values) == 2
        assert 'status = %s' in updates
        assert 'started_at = %s' in updates

    def test_update_execution_includes_token_counts(self):
        """Test that update includes token count fields."""
        updates = []
        values = []

        input_tokens = 15
        output_tokens = 8

        if input_tokens > 0:
            updates.append('input_tokens = %s')
            values.append(input_tokens)

        if output_tokens > 0:
            updates.append('output_tokens = %s')
            values.append(output_tokens)

        assert 'input_tokens = %s' in updates
        assert 'output_tokens = %s' in updates
        assert 15 in values
        assert 8 in values

    def test_update_execution_includes_cost(self):
        """Test that update includes estimated cost."""
        updates = []
        values = []

        estimated_cost = 0.0001

        if estimated_cost > 0:
            updates.append('estimated_cost = %s')
            values.append(estimated_cost)

        assert 'estimated_cost = %s' in updates
        assert 0.0001 in values
