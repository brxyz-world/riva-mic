# Eddie — Riva Mic Orchestrator

Eddie is the ASR/TTS mic + orchestrator core of the **Etymo Holoson** system. It integrates NVIDIA Riva and Ollama to run voice-first conversations with mood and router policy.

## Project Structure
- `Eddie/00_ADMIN/` — admin docs (release notes, versioning, task cards, report)
- `Eddie/10_CONTEXT/` — router/mood policy, lexicon, context pack & schemas
- `Eddie/20_MEMORY/` — capsules, indexes (future RAG)
- `Eddie/30_SYSTEMS/` — orchestrator + audio
- `Eddie/40_OPS/` — ops notes (highlights, metrics, segments)
- `Eddie/40_ARTIFACTS/` — specs/exports
- `Eddie/50_JOURNAL/` — changelog & notebook drops
- `config/` — wake keyword, persona config
- `tests/` — scenario JSON + harness
- `logs/` → roll to `archive/logs/`; state bundles in `archive/docs/`

## How Development Is Managed (Task Cards)
We track work with **CETC** Task Cards: each card splits into **1R (runtime), 1S (systems), 1A (artifacts/docs)** streams. This repo’s docs are the canon; debugging happens in the Eddie Debug Lane chat. :contentReference[oaicite:6]{index=6}

- Card registry & mapping live in `Eddie/00_ADMIN/TASK_CARDS.md`
- Release briefs live in `Eddie/00_ADMIN/RELEASE_<ver>.md`
- Operator knobs live in **`Eddie/30_SYSTEMS/orchestrator/OPERATOR_PANEL.md`** ← single source

## Router & Longform (2.2.4)
- Longform occurs **only** when routed as a “story” or when you explicitly continue on the same topic (“expand”, “go deeper”, “tell me more”). Primary story mood: `Content_Eve`. Utility keys never set longform. Wake opens a **time-boxed** window; it never sets longform by itself. Details: `Eddie/10_CONTEXT/ROUTER_POLICY_NOTES.md`. :contentReference[oaicite:7]{index=7}

## Operator Controls
- See **Operator Panel**: `Eddie/30_SYSTEMS/orchestrator/OPERATOR_PANEL.md` (README links only, no duplication).
- Troubleshooting wake/router overlap: `Eddie/40_OPS/WAKE_ROUTER_TROUBLESHOOTING.md`.

## Logging (JSONL)
Each turn appends: route, mood, user_text, model_reply, retrieval flags/ids, model params, and latency timings (l_asr_ms, l_llm_ms, l_tts_ms, l_total_ms). Longform telemetry adds ready_check and continuation flags. (See REPORT and LOGGING docs.) :contentReference[oaicite:8]{index=8}

## Running Eddie (quick)
- Generate “hello eddie” Porcupine `.ppn` and put it under `Eddie/config/`.
- Copy `.env.example` → `.env`, set credentials and paths.
- Run `start_eddie.ps1` (wakewatch → warmup → mic → orchestrator).
- Longform testing: `tests/longform_smoketest.ps1`.

## Links
- Release brief: `Eddie/00_ADMIN/RELEASE_2.2.4.md` :contentReference[oaicite:9]{index=9}
- Versioning & Cards: `Eddie/00_ADMIN/VERSIONING_GUIDE.md`, `Eddie/00_ADMIN/TASK_CARDS.md`
- Report: `Eddie/00_ADMIN/REPORT.md` (includes Divergence Protocol) :contentReference[oaicite:10]{index=10}
