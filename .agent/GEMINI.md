# Production AI Development Rules

These rules apply to ALL interactions when working on this voice/chat bot codebase.

## Core Principles

### 1. Research First
- Always check internet, documentation, and codebase before implementing
- Search for best practices, known issues, and production patterns
- Understand the existing system architecture before making changes

### 2. Present Findings
- Show what was discovered during research
- Highlight relevant patterns, solutions, and warnings
- Never jump straight to implementation without presenting findings

### 3. Multiple Approaches
- Always suggest 2-3 different approaches for any change
- Compare trade-offs: quick fix vs proper refactor vs architectural change
- Recommend the best option for long-term production stability

### 4. No Hardcoded Pattern Matching
- NEVER use regex or hardcoded string matching for LLM responses
- Leverage LLM intelligence for parsing, understanding, and routing
- Check model capabilities (Gemini 2.5 Flash native audio) before prompting
- Test multi-language and voice consistency

### 5. Auto-Test After Every Change
// turbo-all
After ANY code change, run the full system to verify:
1. Start the server: `python backend/server.py` or `uvicorn backend.server:app --reload`
2. Check for startup errors
3. Verify WebSocket endpoints are accessible
4. Check logs for any warnings or errors

### 6. Root Cause Only
- Never fix symptoms - find the actual root cause
- Trace through the entire request flow
- Consider system-wide impact before any fix
- Document findings before implementing fixes

### 7. Production-Ready Always
- Every change must be deployable to production
- No "quick hacks" or temporary fixes
- Proper error handling, logging, and async patterns
- Consider edge cases and failure modes

## System Architecture Reference

This is a voice bot system with:
- **Audio Pipeline**: 16kHz (Recall.ai) ↔ 24kHz (Gemini)
- **LLM**: Gemini 2.5 Flash native audio
- **Tools**: RAG system, Perplexity web search
- **WebSocket**: Real-time audio streaming
- **Async**: Full asyncio-based architecture
