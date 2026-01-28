---
description: Master workflow - intelligently uses all sub-workflows based on task needs
---

# /build - Master Development Workflow

This workflow automatically determines which sub-workflows to use based on your task.

## Usage
```
/build "your task description"
```

## Examples
- `/build "merge analysis and response prompts, add function calling"`
- `/build "fix voice consistency issues across sessions"`
- `/build "add new calendar tool integration"`

---

## Step 1: Analyze Task
Determine which sub-workflows apply to this task:

| Workflow | Applies When |
|----------|--------------|
| `/brainstorm` | Always - research phase |
| `/prompt` | Changing/merging LLM prompts |
| `/implement` | Adding new features/tools/integrations |
| `/fix` | Fixing bugs or issues |
| `/debug` | Debugging unexpected behavior |

---

## Step 2: Research Phase (Always)
// turbo
1. Search internet for best practices and patterns
2. Read relevant documentation (Gemini API, libraries used)
3. Analyze current codebase structure
4. Check for similar solutions in the industry

---

## Step 3: Present Findings
Create a findings report with:
- What was discovered during research
- Relevant patterns/solutions found
- Warnings or caveats to consider
- System components that will be affected

---

## Step 4: Suggest Approaches
Always present 2-3 approaches:

**Option A: [Name]**
- Best for: [scenario]
- Trade-offs: [...]
- Production impact: [...]

**Option B: [Name]**
- Best for: [scenario]
- Trade-offs: [...]
- Production impact: [...]

**Recommendation**: Which is best for long-term production stability

---

## Step 5: Get User Approval
Wait for user to:
- Pick an approach
- Ask clarifying questions
- Request modifications

**Do NOT proceed without explicit approval.**

---

## Step 6: Execute Relevant Sub-Workflows

Based on task analysis, execute the appropriate workflows:

### If changing prompts → Use /prompt workflow
- Research model capabilities
- Avoid hardcoded patterns
- Use LLM intelligence for parsing

### If adding features → Use /implement workflow
- Design for production from day one
- Proper error handling and logging
- Async/await patterns for this system

### If fixing bugs → Use /fix workflow
- Find root cause first
- Consider system-wide impact
- Make minimal, targeted fixes

### If debugging → Use /debug workflow
- Reproduce the issue
- Trace through system flow
- Identify root cause (not symptoms)

---

## Step 7: Auto-Test (Always)
// turbo
After implementation, run full system verification:

1. Start the server
2. Check for startup errors
3. Verify no regressions in existing features
4. Check logs for warnings

---

## Return Conditions
- Return when all applicable sub-workflows are complete
- Return when user explicitly stops the workflow
- Return when blocked on user decision
