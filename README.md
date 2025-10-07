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
## Wake Word Setup
- Sign up at Picovoice Console and create a Porcupine AccessKey.
- Generate the "hello eddie" .ppn keyword and place it somewhere accessible (this repo keeps it under config/).
- Populate the new .env.example entries (PORCUPINE_ACCESS_KEY, PORCUPINE_KEYWORD_PATH, etc.), then copy them into your local .env or environment. See “Local Env (.env)” below.

## Wake Window Behaviour
- wakewatch.py runs Porcupine locally and POSTs to Eddie at http://127.0.0.1:6060/signal/wake when it hears the keyword.
- The orchestrator opens a configurable wake window (WAKE_WINDOW_MS, default 20s); transcripts outside the window are ignored.
- self.hush stops any playing TTS, optionally closes the window (HUSH_CLOSES_WINDOW) and can acknowledge silently via HUSH_SILENT_ACK.
- self.exit remains the only command that shuts the full stack down.

## Running Eddie
- start_eddie.ps1 now auto-launches config/wakewatch.py (hidden) when both PORCUPINE env vars are present and tears it down on exit.
- Adjust wake duration or binding by overriding WAKE_WINDOW_MS, WAKE_BIND, or WAKE_PORT in your environment. Note: WAKE_PORT defaults to 0 (disabled) in Phase 2.2; wake gating is handled via a local wake flag touched by wakewatch + honored by the mic.

## Local Env (.env)
- Keep secrets and machine-specific paths out of git by using a local `.env` file in the repo root. The start script loads it at runtime.
- Steps:
  - Copy `.env.example` to `.env` (do not commit `.env`).
  - Set your real values locally, e.g.:
    - `PORCUPINE_ACCESS_KEY=your-real-key-here`
    - `PORCUPINE_KEYWORD_PATH=config/Hello-Eddie.ppn`
    - Optionally override: `RIVA_SPEECH_API`, `OLLAMA_URL`, `OLLAMA_MODEL`, `VOICE_NAME`, etc.
- `.gitignore` excludes `.env` and `logs/` by default.

## Phase 2.2: Standby + Wake Window
- Wake word detection remains a separate helper (`config/wakewatch.py`).
- The mic (`riva_streaming_mic.py`) honors a time-based wake window via the local wake flag.
- `.env` support added to `start_eddie.ps1`; no secrets are hard-coded in the repo.
