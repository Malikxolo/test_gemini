---
description: Smart LLM prompting - avoid hardcoded patterns, use model intelligence
---

# /prompt - Prompting Workflow

Use this workflow when editing, merging, or creating LLM prompts. 

**Core Principle**: Use LLM intelligence, NOT hardcoded pattern matching.

## Step 1: Identify the Model
// turbo
Check which model is being prompted:
- Currently using: `Gemini 2.5 Flash native audio`
- Check documentation for model-specific capabilities
- Understand input/output formats

---

## Step 2: Research Model Capabilities

Search for:
1. Model-specific prompting best practices
2. Multi-turn conversation patterns
3. Tool/function calling syntax
4. Audio-specific considerations for voice models

---

## Step 3: Analyze Current Prompts

Review existing prompts for:
- **Anti-patterns to remove**:
  - Hardcoded response formats (e.g., "respond with YES or NO")
  - Regex-dependent outputs
  - Brittle parsing requirements
  
- **Patterns to keep**:
  - Clear instructions
  - Context and role definitions
  - Examples (few-shot) when helpful

---

## Step 4: Evaluate Strategy: Refine vs Re-engineer

**Design Philosophy**:
1. **Roburstness First**: Can we make the current prompt better by adding examples or clarifying instructions?
2. **Avoid Re-engineering**: Do not rewrite the entire prompt structure unless it is fundamentally broken.
3. **Justification**: If you choose to re-engineer, you must document *why* the current approach cannot be salvaged.

**Decision Decision Matrix**:
- **Minor Issues** (ambiguity, edge cases) -> **Refine** (Add constraints, examples)
- **Fundamental Flaws** (wrong model assumptions, hallucination loops) -> **Re-engineer** (Redesign interaction)

---

## Step 5: Design Prompt Strategy

**AVOID**:
```
❌ "If the user says 'search for X', extract X using format: SEARCH:query"
❌ "Respond with exactly 'TOOL:name:params'"
❌ "Parse the response by splitting on '|'"
```

**USE INSTEAD**:
```
✅ Use native function/tool calling (model handles routing)
✅ Let the model understand intent naturally
✅ Use structured outputs when available
✅ Trust model reasoning for parsing
```

---

## Step 6: Present Prompt Approaches

**Option A: [Name]**
- Prompting strategy: [description]
- How it leverages LLM intelligence: [explanation]
- Trade-offs: [...]

**Option B: [Name]**
- Prompting strategy: [description]
- How it leverages LLM intelligence: [explanation]
- Trade-offs: [...]

**Recommendation**: [Best for production voice bot]

---

## Step 7: Implement Prompt Changes

After approval:
1. Update the prompt in `get_gemini_config()` or relevant location
2. Ensure function/tool definitions are properly structured
3. Remove any hardcoded parsing logic

---

## Step 8: Test with Real Conversations
// turbo
1. Start the system
2. Test with various inputs:
   - Normal requests
   - Edge cases
   - Multi-language inputs (if applicable)
   - Voice/accent variations
3. Verify model routes to correct tools
4. Check response quality and consistency

---

## Voice-Specific Considerations

For this voice bot, also verify:
- Voice consistency across the session
- Language/accent matching
- Natural conversational flow
- Appropriate response length for audio

---

## Return Conditions
- Return when prompt changes are implemented and tested
- Return when user needs to decide between approaches
- Return if model behavior is unexpected
