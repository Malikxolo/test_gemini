---
description: Add new features, tools, or integrations with production-ready standards
---

# /implement - Feature Implementation Workflow

Use this workflow when adding new capabilities: tools, features, integrations, etc.

## Examples
- Adding a new RAG tool
- Integrating a new API (calendar, email, etc.)
- Adding a new voice command
- Implementing function calling
- Adding new audio processing features

---

## Step 1: Understand Requirements

Define:
1. What is the feature supposed to do?
2. What are the inputs and outputs?
3. How should it integrate with existing system?
4. What are the edge cases?

---

## Step 2: Research
// turbo
1. Search for best practices for this type of feature
2. Check documentation for APIs/libraries needed
3. Look for similar implementations in industry
4. Analyze how existing features are implemented in this codebase

---

## Step 3: Analyze Integration Points

For this voice bot system:
- Where in the audio pipeline does this feature fit?
- Does it need a new tool definition for Gemini?
- Does it need new WebSocket handlers?
- Does it interact with RAG or Perplexity?

---

## Step 4: Present Design Options

**Option A: [Name]**
- Architecture: [how it fits into the system]
- Implementation: [key components to add/modify]
- Pros: [benefits]
- Cons: [drawbacks]
- Effort: [low/medium/high]

**Option B: [Name]**
- Architecture: [how it fits into the system]
- Implementation: [key components to add/modify]
- Pros: [benefits]
- Cons: [drawbacks]
- Effort: [low/medium/high]

**Recommendation**: [Best for production and maintainability]

---

## Step 5: Implement with Production Standards

After approval, implement with:

### Error Handling
```python
try:
    result = await some_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Graceful fallback
```

### Logging
```python
logger.info(f"🔧 Feature: Starting operation with {params}")
logger.debug(f"Detailed state: {state}")
```

### Async Patterns
```python
# This system uses asyncio throughout
async def new_feature():
    result = await external_api_call()
    return result
```

### Tool Definition (if adding Gemini tool)
```python
{
    "name": "tool_name",
    "description": "Clear description of what the tool does",
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}
```

---

## Step 6: Test Integration
// turbo
1. Start the full system
2. Test the new feature in isolation
3. Test the feature with existing features
4. Verify no regressions in other functionality
5. Test edge cases and error conditions

---

## Step 7: Document

Add comments for:
- What the feature does
- How it integrates with the system
- Any configuration needed
- Usage examples

---

## Return Conditions
- Return when feature is implemented and tested
- Return when user needs to decide on design
- Return if implementation reveals new requirements
