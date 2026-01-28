---
description: Systematic debugging with root cause analysis and multiple fix approaches
---

# /debug - Debug Workflow

Use this workflow when something isn't working as expected and you need to find and fix the issue.

## Step 1: Research the Issue
// turbo
1. Search internet for known issues or patterns
2. Check if this is a documented problem
3. Look for similar issues in the codebase history

---

## Step 2: Reproduce the Issue

Before debugging:
1. Understand the exact steps to reproduce
2. Identify the expected vs actual behavior
3. Note any error messages or logs

---

## Step 3: Trace Through System

Follow the request flow:
1. **Entry point**: Where does the request come in?
2. **Processing**: What components handle it?
3. **External calls**: Any API calls (RAG, Perplexity, Gemini)?
4. **Output**: Where does it fail or produce wrong output?

For this voice bot system:
- Audio input → WebSocket → Gemini API → Tools → Response → Audio output

---

## Step 4: Identify Root Cause

**CRITICAL**: Find the ACTUAL cause, not symptoms.

Ask:
- Why is this happening?
- What changed that might have caused this?
- Is this a single issue or systemic problem?
- What other parts of the system might be affected?
- **Holistic View**: Are inputs/outputs consistent across the entire pipeline?

---

## Step 5: Present Findings

Report:
- **Root Cause**: What's actually causing the issue
- **System Impact**: What parts are affected
- **Related Issues**: Any connected problems discovered

---

## Step 6: Suggest Fix Approaches

Always present 2-3 approaches:

**Option A: Quick Fix**
- What: [minimal change to fix the symptom]
- Pros: Fast, low risk
- Cons: Might not address root cause, could recur
- Best for: Time-sensitive situations

**Option B: Proper Fix**
- What: [addresses root cause properly]
- Pros: Solves the actual problem
- Cons: More effort, needs testing
- Best for: Most situations

**Option C: Architectural Fix** (if applicable)
- What: [redesign to prevent this class of issues]
- Pros: Long-term stability
- Cons: Significant effort
- Best for: Recurring or systemic issues

**Recommendation**: [Best option for production stability]

---

## Step 7: Implement Chosen Fix

After user approval:
1. Make the minimal, targeted changes
2. Add logging if needed for future debugging
3. Document what was changed and why

---

## Step 8: Verify Fix
// turbo
1. Run the full system
2. Reproduce the original issue - should be fixed
3. Check for regressions in related features
4. Verify logs show healthy behavior

---

## Return Conditions
- Return when issue is fixed and verified
- Return when blocked on user decision
- Return if issue requires more investigation
