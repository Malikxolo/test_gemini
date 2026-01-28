---
description: Deep research and planning before any major change
---

# /brainstorm - Research & Planning Workflow

Use this workflow before any major change to research thoroughly and plan the approach.

## Step 1: Understand the Request
- What exactly is being asked?
- What are the success criteria?
- What are the constraints?

---

## Step 2: Research Phase
// turbo
1. **Search internet** for:
   - Best practices for this type of change
   - Common pitfalls and solutions
   - Production patterns from industry

2. **Read documentation**:
   - Gemini API docs for audio/voice features
   - Library documentation (FastAPI, asyncio, etc.)
   - Any relevant external API docs

3. **Analyze codebase**:
   - Current architecture and patterns
   - Related components that might be affected
   - Existing similar implementations

4. **Check for similar solutions**:
   - How others have solved this
   - Open source examples
   - Community discussions

---

## Step 3: Present Findings

Create a findings report:

### Research Summary
- Key discoveries from internet search
- Relevant documentation insights
- Codebase analysis results

### Patterns Found
- Recommended patterns for this use case
- Anti-patterns to avoid

### Warnings & Caveats
- Potential risks or issues
- Compatibility concerns
- Performance considerations

### Affected Components
- Which files/modules will change
- Dependencies that might be impacted

---

## Step 4: Suggest Approaches

Always present 2-3 approaches:

**Option A: [Name]**
- Description: [what this approach does]
- Best for: [when to use this]
- Pros: [benefits]
- Cons: [drawbacks]
- Production impact: [deployment considerations]

**Option B: [Name]**
- Description: [what this approach does]
- Best for: [when to use this]
- Pros: [benefits]
- Cons: [drawbacks]
- Production impact: [deployment considerations]

**Recommendation**: [Which option is best for long-term production stability and why]

---

## Step 5: Wait for Approval

Do NOT proceed until user:
- Approves an approach
- Requests modifications
- Asks for more research

---

## Return Conditions
- Return with findings and approaches for user decision
- Return if user wants to pivot to a different direction
