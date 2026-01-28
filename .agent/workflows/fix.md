---
description: Production-safe bug fixes with root cause analysis and multiple approaches
---

# /fix - Bug Fix Workflow

Use this workflow when you have a known bug to fix. Focus on root cause and production safety.

## Step 1: Understand the Bug

Before fixing:
1. What is the expected behavior?
2. What is the actual behavior?
3. What are the reproduction steps?
4. When did this start happening?

---

## Step 2: Research
// turbo
1. Search for known issues or patterns
2. Check if this is a library/API bug vs our code
3. Look for related issues in codebase

---

## Step 3: Find Root Cause

**CRITICAL**: Never fix symptoms. Find the actual cause.

Trace through:
1. Where does the bug manifest?
2. What code path leads there?
3. What's the actual source of the problem?
4. Is this affecting other parts of the system?

**Holistic Check**:
- Does this bug reveal a flaw in the overall architecture?
- Are we making assumptions that are not true for the whole system?
- Look beyond the immediate error line - understand the data flow.

---

## Step 4: Assess System Impact

Consider:
- What other features might be affected by this bug?
- What other features might be affected by the fix?
- Are there related bugs that should be fixed together?
- Will the fix require changes to multiple files?

---

## Step 5: Present Fix Approaches

Always present 2-3 approaches:

**Option A: Quick Fix**
- What: Minimal change to stop the bug
- Code impact: Small, localized
- Pros: Fast to implement, low risk
- Cons: May not address root cause
- When to use: Urgent production issue

**Option B: Proper Fix**
- What: Addresses root cause properly
- Code impact: Moderate, focused
- Pros: Actually solves the problem
- Cons: Needs thorough testing
- When to use: Most situations

**Option C: Comprehensive/Holistic Fix** (Preferred)
- What: Fix + improvements to prevent recurrence
- Code impact: Larger, may touch multiple files
- Pros: Long-term stability, prevents similar bugs, robust
- Cons: More effort and testing required
- When to use: Recurring issues, systemic problems, or whenever feasible

**Recommendation**: Preferred approach is to fix the system holistically, not just the symptom. Explain why a holistic fix is better for production.

---

## Step 6: Implement Fix

After approval:
1. Make minimal, targeted changes
2. Don't fix unrelated issues in the same change
3. Add comments explaining the fix if non-obvious
4. Add logging for future debugging if needed

---

## Step 7: Verify Fix
// turbo
1. Run the full system
2. Verify the bug is fixed
3. Test related features for regressions
4. Check logs for any new warnings

---

## Step 8: Document

Note:
- What the bug was
- What caused it
- How it was fixed
- Any follow-up items

---

## Return Conditions
- Return when bug is fixed and verified
- Return when user needs to choose approach
- Return if fix reveals deeper issues
