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

## Docs
- `docs/ORCHESTRATOR.md` — hot path, long-form gates, recall injection  
- `docs/RAG_GUIDE.md` — local index over `/20_MEMORY/capsules`, top-2 snippet discipline  
- `docs/LOGGING.md` — JSONL schema (retrieval/model/perf fields)  
- `docs/TEST_HARNESS.md` — scenario runner and assertions  
- `docs/ROUTER_POLICY.md` — rule schema, examples, and mood weight overrides

## Logging (JSONL)
- Each turn appends a JSON object with (at minimum):
- turn, route, user_text, model_reply,
   retrieval_used, retrieval_ms, retrieval_caps,
   ollama_model, num_predict, temperature, stop_tokens_used,
   l_asr_ms, l_llm_ms, l_tts_ms, l_total_ms, mood, router_variant, long_form
- Long-form telemetry also reports:
  - `longform_system_append`
  - `longform_stop_sentinel_used`
  - `longform_retry_used`
  - `longform_suggest_continue`
  - `refusal_glitch`
  - `autoconfirm_used`
  - `autoconfirm_llm_ms`

## Memory Service (local RAG)
`Eddie/30_SYSTEMS/memory_service.py` exposes:
- `query(text, k_caps=2) -> [{id, score, text}]`  (≤ ~200 tokens total for injection)
- `upsert_capsule(id, text, meta)`                (writes into `/20_MEMORY/capsules`)

Embeddings + a small vector index live under `Eddie/20_MEMORY/indexes/` when enabled.

## Test Harness
Run `tests/run_scenarios.py` to exercise recall/latency/long-form without voice.  
Scenarios live in `tests/cases/*.json`.
- `tests/cases/long_story.json` asserts a >=500 character reply and checks for paragraph breaks when long-form is on.
- `tests/cases/long_story_retry.json` forces the retry branch by demanding ≥1200 characters; the harness may accept a shorter reply when the log reports `longform_retry_used=true`.

---
## Wake Word Setup
- Sign up at Picovoice Console and create a Porcupine AccessKey.
- Generate the "hello eddie" .ppn keyword and place it somewhere accessible (this repo keeps it under Eddie/config/).
- Populate the new .env.example entries (PORCUPINE_ACCESS_KEY, PORCUPINE_KEYWORD_PATH, etc.), then copy them into your local .env or environment. See “Local Env (.env)” below.

## Wake Window Behaviour
- wakewatch.py runs Porcupine locally and POSTs to Eddie at http://127.0.0.1:6060/signal/wake when it hears the keyword.
- The orchestrator opens a configurable wake window (WAKE_WINDOW_MS, default 20s); transcripts outside the window are ignored.
- self.hush stops any playing TTS, optionally closes the window (HUSH_CLOSES_WINDOW) and can acknowledge silently via HUSH_SILENT_ACK.
- self.exit remains the only command that shuts the full stack down.

## Running Eddie
- start_eddie.ps1 now auto-launches Eddie/30_SYSTEMS/audio/wakewatch.py (hidden) when both PORCUPINE env vars are present and tears it down on exit.
- Adjust wake duration or binding by overriding WAKE_WINDOW_MS, WAKE_BIND, or WAKE_PORT in your environment. Note: WAKE_PORT defaults to 0 (disabled) in Eddie 2.2; wake gating is handled via a local wake flag touched by wakewatch + honored by the mic.

## Local Env (.env)
- Keep secrets and machine-specific paths out of git by using a local `.env` file in the repo root. The start script loads it at runtime.
- Steps:
  - Copy `.env.example` to `.env` (do not commit `.env`).
  - Set your real values locally, e.g.:
    - `PORCUPINE_ACCESS_KEY=your-real-key-here`
    - `PORCUPINE_KEYWORD_PATH=Eddie/config/Hello-Eddie.ppn`
    - Optionally override: `RIVA_SPEECH_API`, `OLLAMA_URL`, `OLLAMA_MODEL`, `VOICE_NAME`, etc.
- `.gitignore` excludes `.env` and `logs/` by default.

Long-form knobs:
- `LONG_FORM` — set to `1` for multi-paragraph replies; leave unset/`0` for fast one-liners.
- `NUM_PREDICT` — completion token budget (default 160, clamped to >=512 whenever long-form is enabled).
- `LONGFORM_MIN_CHARS` — minimum reply length before the orchestrator considers a retry/suggest cycle (default 500).
- `LONGFORM_TEMPERATURE` — temperature applied in long-form mode (default 0.6). Short-form stays at 0.3.
- `LONGFORM_SYSTEM_APPEND` — optional override for the creative narrative appendix injected only when `LONG_FORM=1`.
- `LONGFORM_STOP_SENTINEL` — sentinel stop token used when `STOP_TOKENS` is unset in long-form (default `<END>`).
- `LLM_TIMEOUT_MS` — Ollama call timeout budget in milliseconds (default 20000). Set `6000` for interactive long-form to keep the first reply under ~6s.
- `STOP_TOKENS` — optional JSON array of custom stops. Leave unset to keep concise replies unbounded and let the sentinel be injected automatically in long-form.
- `TWF_DISABLE` — when `1`, disables time-wait/filler pacing for lowest latency.
- `OLLAMA_KEEP_ALIVE` — keep the Ollama model resident between turns (default `30m`).
- `OLLAMA_WARMUP_ON_START` — default `1`; fires a tiny warmup call at boot to dodge cold-start 499s.

Task 1R additions (interactive long-form safety):
- `LONGFORM_RETRY` — default `0` (interactive); when `1` (harness), allows a single auto-retry.
- `LONGFORM_SUGGEST_CONTINUE` — default `1`; when the first pass is short, Eddie asks to continue ("I can go long — want me to continue?").
- `LONGFORM_CONTINUE_TTS` — override the continue prompt text (short, <2s).
- `LONGFORM_REFUSAL_GUARD` — default `1`; neutralizes polite refusals by swapping in `LONGFORM_REFUSAL_SEED`.
- `LONGFORM_REFUSAL_SEED` — override for the guard’s neutral opener (default “Okay—starting the story.”).
- `LONGFORM_AUTOCONFIRM` — default `0`; when `1`, auto-consents after the prompt and fetches one continuation.
- `LONGFORM_AUTOCONFIRM_NUM_PREDICT` — completion budget for the auto-continuation (default `768`, clamped to ≥512 in long-form mode).
- `LONGFORM_AUTOCONFIRM_TIMEOUT_MS` — timeout for the auto-continuation call (default `20000`).
- `LONGFORM_AUTOCONFIRM_SUFFIX` — instruction block appended to the follow-up prompt (default “Continue the story … <END>.”).

## Eddie 2.2: Standby + Wake Window
- Wake word detection remains a separate helper (`Eddie/30_SYSTEMS/audio/wakewatch.py`).
- The mic (`riva_streaming_mic.py`) honors a time-based wake window via the local wake flag.
- `.env` support added to `start_eddie.ps1`; no secrets are hard-coded in the repo.
- **Long-Form Mode (env-gated):** set `LONG_FORM=1` to lift the clipper. Eddie bumps `NUM_PREDICT` to at least 512 (override via the env) and, when no custom `STOP_TOKENS` are supplied, injects the `<END>` sentinel to keep long replies bounded.
- **Unified JSONL telemetry:** per-turn logs now include retrieval flags/ids, model params, and timing so you can correlate behavior with performance.
- **Memory Service (stub):** a tiny in-process shim (`memory_service.py`) provides `query()`/`upsert()` and a path to a local vector index over `/20_MEMORY/capsules`.
- **Test harness (text-only):** quick scenarios run Eddie without voice to validate recall, latency, and long-form output.

## Eddie 2.2.3 (short-form restored; longform testable)
- start_eddie.ps1 restores the classic short-form path: venv → (optional) Porcupine wake watch → light Ollama warmup →            riva_streaming_mic.py.
- Longform can be exercised via tests/longform_smoketest.ps1.
- Personality “thinking beats” are available for future router/mood use.

# Backup 2.2.4 plan proposed by branch CETC1
- 2.2.4 plan (1R / 1S / 1A), for the next branch
- 1R (Runtime)
   Promote longform initiation into router/mood rules (no test harness bypass).
   Add ready-check gate (mic only): informal rephrase + “it might take me a minute…”; 3–6s timeout with “go”/“yep” intents.
   Stream “thinking beats” at low duty cycle while generating (tiny TTS interjections or text overlays).

- 1S (Systems)
   Harden autoconfirm (or remove for local): sequential backoff 15s → 30s; cancel on model load.
   Smarter TTS chunker (punctuation + length) with memory of last phoneme to avoid cut-offs.
   Optional: print-as-you-speak mirror in terminal.

- 1A (Artifacts/Docs)
   README sections for longform governance; router policy doc; a small “operator panel” table of ENV knobs.
   Update EHPDM with final longform governance and beat cadence controls.

# notes about knobs from CETC1.1
+ - `LONGFORM_TTS_ENABLE` — when `1`, long-form replies are spoken with sentence-sized chunks.
+ - `LONGFORM_TTS_CHARS`  — target characters per spoken chunk (default 280).
+ - `LONGFORM_TTS_MAX_S`  — max total playback seconds for long-form speech (default 120).
+ - `LONGFORM_PRINT`      — when `1`, prints the full long-form story to stdout (safe; JSON stays last).
+ - `LONGFORM_READY_CHECK`— reserved for mic sessions; keep `0` for the smoketest.
