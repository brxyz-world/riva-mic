# Eddie (Riva Mic Orchestrator)

## Overview
Eddie is the ASR/TTS mic + orchestrator core of the larger **Etymo Holoson** system.  
This repo contains the mic client, orchestrator, and integration with NVIDIA Riva + Ollama.

## Scaffolding
Development is managed across **three chat lanes**:

1. **Eddie Debug Lane**  
   - Main development chat.  
   - All edits via Git Patch Protocol.  
   - Runtime logs/debugging posted here.

2. **Eddie Debug Kickoff (DDKB)**  
   - Template to reseed new debug chats when context drifts.  
   - Ensures Codex/ChatGPT always knows repo URL and protocol.

3. **Etymo Holoson Personality Plan (EHPP)**  
   - Exploratory chat.  
   - Holds roadmap for routing, personality, MCP integration, Unreal embodiment.  
   - Not used for debugging.

## Guardrails
- One active debug lane at a time.  
- Never delete chats; retire and label them.  
- Personality/vision kept strictly separate from technical debugging.  
- Kickoff briefs act as reproducible protocols for instantiating new chats.

## Future Repos
- Unreal/Holo Call embodiment will be its own repo (e.g. `Eddie-Unreal`).  
- MCP modules may live separately or as submodules.

---
