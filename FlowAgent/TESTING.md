# FlowAgent Testing Guide

This document outlines the testing strategy and how to run tests for the FlowAgent MVP.

## Test Structure

```
FlowAgent/
├── apps/
│   ├── api/
│   │   └── src/
│   │       ├── routes/__tests__/
│   │       │   ├── auth.test.ts
│   │       │   └── agents.test.ts
│   │       └── lib/__tests__/
│   │           ├── db.test.ts
│   │           └── queue.test.ts
│   ├── web/
│   │   └── src/
│   │       ├── lib/__tests__/
│   │       │   └── api.test.ts
│   │       └── app/__tests__/
│   │           └── auth.test.tsx
│   └── agent-engine/
│       └── tests/
│           └── test_execute_handler.py
└── tests/
    └── e2e/
        ├── auth-flow.test.ts
        └── agent-execution-flow.test.ts
```

## Test Categories

### 1. Unit Tests
Test individual functions and components in isolation.

**Location:**
- `apps/api/src/**/__tests__/*.test.ts`
- `apps/web/src/**/__tests__/*.test.ts`
- `apps/agent-engine/tests/test_*.py`

**What they test:**
- Input validation
- Business logic
- Data transformations
- Error handling

### 2. Integration Tests
Test how different parts of the system work together.

**Location:**
- `apps/api/src/lib/__tests__/*.test.ts`
- `apps/agent-engine/tests/test_*.py`

**What they test:**
- Database operations
- Queue integration
- API endpoint behavior
- Tool execution

### 3. End-to-End Tests
Test complete user flows from start to finish.

**Location:**
- `tests/e2e/*.test.ts`

**What they test:**
- Full authentication flow
- Agent creation and execution flow
- Error scenarios

## Running Tests

### API Tests (TypeScript)

```bash
# Install dependencies
pnpm install

# Run all tests
pnpm test

# Run tests with coverage
pnpm test:coverage

# Run tests in watch mode
pnpm test:watch

# Run specific test file
pnpm test apps/api/src/routes/__tests__/auth.test.ts
```

### Agent Engine Tests (Python)

```bash
# Navigate to agent engine
cd apps/agent-engine

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_execute_handler.py

# Run specific test class
pytest tests/test_execute_handler.py::TestExecuteHandler

# Run specific test
pytest tests/test_execute_handler.py::TestExecuteHandler::test_handler_parses_event_body_correctly
```

### E2E Tests

```bash
# Run from project root
pnpm test:e2e

# Or run specific E2E test
pnpm vitest tests/e2e/auth-flow.test.ts
```

## Test Coverage Goals

Target coverage: **80%+** for critical paths

**Critical Paths (Must be >90% covered):**
- Authentication (register, login, logout)
- Agent execution flow
- Input validation
- Database operations
- Queue integration

**Important Paths (Must be >80% covered):**
- Agent CRUD operations
- Tool execution
- Error handling

**Nice to Have (>70% covered):**
- Edge cases
- UI components
- Utility functions

## Writing New Tests

### API Route Tests (TypeScript)

```typescript
import { describe, it, expect } from 'vitest';

describe('MyRoute', () => {
  describe('POST /', () => {
    it('should handle valid input', () => {
      const validInput = {
        field: 'value',
      };

      expect(validInput.field).toBeTruthy();
    });

    it('should reject invalid input', () => {
      const invalidInput = {
        field: '',
      };

      expect(invalidInput.field.length).toBe(0);
    });
  });
});
```

### Lambda Tests (Python)

```python
import pytest
from unittest.mock import Mock, patch

class TestMyHandler:
    def test_handles_valid_input(self):
        """Test handler with valid input."""
        input_data = {
            'key': 'value'
        }

        assert input_data['key'] == 'value'

    def test_rejects_invalid_input(self):
        """Test handler rejects invalid input."""
        invalid_data = {}

        assert 'key' not in invalid_data
```

## CI/CD Integration

Tests run automatically on:
- Every pull request
- Every push to `main` branch
- Before deployment

**GitHub Actions Workflow:**

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test-api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: pnpm install
      - run: pnpm test

  test-agent-engine:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest
```

## Test Best Practices

### 1. **Test Naming**
- Use descriptive names: `should_reject_invalid_email`
- Follow pattern: `should_<expected behavior>_when_<condition>`

### 2. **Test Independence**
- Each test should run independently
- Don't rely on test execution order
- Clean up after tests (if using real DB)

### 3. **Arrange-Act-Assert Pattern**
```typescript
it('should do something', () => {
  // Arrange: Set up test data
  const input = { value: 123 };

  // Act: Perform the action
  const result = someFunction(input);

  // Assert: Verify the result
  expect(result).toBe(expected);
});
```

### 4. **Mock External Dependencies**
```typescript
// Mock database
vi.mock('../lib/db', () => ({
  createDbClient: vi.fn(() => mockDb),
}));

// Mock API calls
vi.mock('node-fetch', () => ({
  default: vi.fn(() => Promise.resolve(mockResponse)),
}));
```

### 5. **Test Edge Cases**
- Empty inputs
- Very large inputs
- Invalid types
- Null/undefined values
- Concurrent operations

## Common Test Scenarios

### Authentication Tests
- ✅ Valid registration
- ✅ Duplicate email
- ✅ Invalid email format
- ✅ Password too short
- ✅ Valid login
- ✅ Invalid credentials
- ✅ Session expiration

### Agent Tests
- ✅ Agent creation
- ✅ Agent limit enforcement
- ✅ Agent execution
- ✅ Execution limit enforcement
- ✅ Invalid input rejection
- ✅ Tool execution
- ✅ Error handling

### Security Tests
- ✅ SQL injection prevention
- ✅ XSS prevention
- ✅ CSRF protection
- ✅ Rate limiting
- ✅ Input sanitization

## Debugging Failed Tests

### 1. **Verbose Output**
```bash
# TypeScript
pnpm vitest --reporter=verbose

# Python
pytest -vv
```

### 2. **Run Single Test**
```bash
# TypeScript
pnpm vitest -t "should reject invalid email"

# Python
pytest -k "test_rejects_invalid_email"
```

### 3. **Debug Mode**
```bash
# TypeScript
pnpm vitest --inspect-brk

# Python
pytest --pdb
```

### 4. **Check Coverage Gaps**
```bash
# TypeScript
pnpm test:coverage
# Open coverage/index.html

# Python
pytest --cov=src --cov-report=html
# Open htmlcov/index.html
```

## Continuous Improvement

### Monthly Tasks
- [ ] Review test coverage reports
- [ ] Add tests for new features
- [ ] Update tests for changed behavior
- [ ] Remove obsolete tests
- [ ] Refactor flaky tests

### Before Each Release
- [ ] All tests passing
- [ ] Coverage > 80%
- [ ] No skipped tests
- [ ] E2E tests verified
- [ ] Performance tests run

## Getting Help

- Tests failing? Check TROUBLESHOOTING.md
- Need to add new tests? See examples in existing test files
- Questions? Open an issue on GitHub

---

**Last Updated**: January 31, 2026
**Maintained By**: FlowAgent Team
