# EHPP — Phase‑1 → Phase‑2 Handoff Packet (Canonical)
**Project:** Eddie (Etymo Holoson Personality Plan — Phase 2 Branched Chat)  
**Purpose:** Freeze Phase‑1 baselines, define Phase‑2 goals, outputs, file layout, and test rituals. This file is the *single source of truth* for the Phase‑1 → Phase‑2 migration.

---

## 1) Phase‑1 baselines (do not regress)
- **Router‑first flow with optional fillers**; speaking sentinel mutes ASR during TTS; TTS warmed; replies clipped to ≤240 chars.  
- **Wake/turn logic:** silence‑based finalization; wake‑word window; virtual PTT linger; clean subprocess call into orchestrator; dictation separated while asleep.

**Why it matters:** Phase‑2 must preserve snappiness and end‑to‑end stability while adding persona/micro‑memory.

---

## 2) Phase‑2 intent & definition of done (DoD)
**Intent:** Externalize persona; add short‑term recall; tune fillers/ACKs and prosody — without hurting Phase‑1 speed.

**DoD (all true):**
1. `/config/personality.xml` exists and loads at startup (ops knobs remain env‑driven).  
2. `/docs/stylebook.md` defines tone, tics, refusal/deflection voice.  
3. **Ephemeral memory (N=5)** yields a one‑sentence *topic tagline* and light recall in answers.  
4. **Latency‑shaped fillers/ACKs** read XML defaults; env overrides still work.  
5. **Per‑run logs** rotate to `logs/Eddie_Convo_<ts>.jsonl` and a tiny `session_meta.json`.  
6. Rituals pass: **routing gauntlet**, **mini‑monologue**, **three‑mood read** (KPIs below).

Non‑goals for Phase‑2: long‑term/RAG memory, new external tools integration (that’s Phase‑5/MCP).

---

## 3) Minimal repo layout changes
```
riva-mic/
├─ eddie_orchestrator.py
├─ riva_streaming_mic.py
├─ start_eddie.ps1
├─ config/
│  ├─ personality.xml            # NEW: persona (runtime loaded)
│  └─ guardrails.yml             # Optional: refusal categories/notes
├─ docs/
│  ├─ stylebook.md               # NEW: human style rules + examples
│  └─ EHPP_Phase2_Handoff.md     # THIS file (canonical packet)
├─ logs/
│  └─ Eddie_Convo_<ts>.jsonl     # NEW: per‑run log
├─ tools/                        # Move bench/test scripts here
└─ .gitignore  # add: /logs/, *.wav, /eddie_speaking.flag
```

---

## 4) Config & file types (rationale)
- **Persona in XML** (`/config/personality.xml`) for clean hierarchy + future schema.  
- **Ops in env** (unchanged).  
- **Stylebook in Markdown** for human examples (not parsed).  
- Optional **guardrails.yml** for hand‑tuned refusal categories.

### `/config/personality.xml` (v0.1 — starter content)
See the companion `personality.xml` in this directory (exact content included in this repo).

### `/docs/stylebook.md` (seed)
```
# Eddie Stylebook (v0.1)
Voice: concise, neutral, helpful. Humor: dry, gentle. No sarcasm, no emojis.
Ack palette (pick 1, not stacked): “Right.” “Okay.” “Got it.” “Sure.”
Refusal: brief + helpful alternative (“I can’t do that, but I can …”)
Examples:
- Greeting → “Hey, listening.”
- Error → “Something hiccuped; try again.”
- Recap → “We’re on OBS scenes; you asked about Talking.”
```

---

## 5) Implementation plan (surgical steps)

### 5.1 Log rotation & session meta (no hot‑path cost)
- **`start_eddie.ps1`**: set per‑run path and ensure folder (snippet in Appendix B).  
- Orchestrator: after init, write `logs/session_meta.json` with:
  ```json
  { "ts": "<timestamp>", "voice": "<VOICE_NAME>", "commit": "<git-head>" }
  ```

### 5.2 Load `personality.xml` at orchestrator init
- Parse once; map to in‑memory config: `fillers`, `prob`, `firstMs`, `postPauseMs`, `ack policy`, `ephemeralTurns`, `prosody presets`.  
- **Precedence:** env → XML → defaults. Keep env overrides intact.

### 5.3 Ephemeral memory (N=5) + topic tagline
- Maintain a ring buffer of the last N {user, reply}.  
- On each turn, synthesize a single *topic tagline* (e.g., “We’re on OBS scenes and hotkeys”). Either a tiny heuristic or a bounded one‑shot LLM (≤1 sentence).  
- Prepend the tagline + last 2 user turns to the LLM prompt; cap tokens. No RAG or external store in Phase‑2.

### 5.4 Latency‑shaped fillers/ACKs via XML
- Preserve current filler behavior (only if model drags; small probability; short post‑pause).  
- Move thresholds/weights to XML; still pre‑synth “Hm/Hmm/Uh”.

### 5.5 Prosody presets
- If TTS exposes pace/emphasis, map to presets; else simulate via textual rhythm (punctuation, clause length).  
- Keep reply ≤240 chars; 1–2 sentences.

### 5.6 Terminal cleanliness while asleep (optional, 1‑liner)
- Gate interim/final ASR prints when asleep; show during awake/PTT only (prevents dictation noise).

### 5.7 Housekeeping
- Move bench scripts to `/tools`; add `/logs/`, `*.wav`, `eddie_speaking.flag` to `.gitignore`.

---

## 6) Optional tech & “when to RAG”
- **RAG:** Defer to Phase‑5 (MCP era). For Phase‑2, a tiny keyword note file (`memory/personal_notes.json`) is sufficient if desired.  
- **Schemas:** Consider an XSD later for `personality.xml` once stable.  
- **Hot‑reload:** Optional debug command to reload XML without restart.

---

## 7) Rituals & KPIs (acceptance)
- **Routing Gauntlet (90s):** 10 mixed prompts; ACK median < **1.0 s**; route correctness ≥ **95%**.  
- **Mini‑Monologue (2m):** recap a short chat in Eddie’s tone; stylebook match ≥ **80%** by your rubric.  
- **Three‑Mood Read:** same sentence in calm/playful/solemn; human‑rated affect clarity ≥ **80%**.  
- **Session Hygiene:** per‑run files exist; `session_meta.json` present.

---

## 8) Git hygiene
- Commit Phase‑1 freeze; tag `phase-1-final`.  
- Commit this packet + `config/` + `docs/`; tag `phase-2-kickoff`.  
- Subsequent diffs: “Change + Rationale + Micro test plan” in each PR.

---

## Appendix A — Phase‑2 Codex Boot Prompt (copy/paste)
Save separately as `/docs/phase2_boot_prompt.txt` if preferred; content included here for convenience.

```
You are assisting Phase 2 (Personality Layer) of “Eddie,” a low-latency voice assistant.
Phase 1 shipped router-first flow with optional fillers and speaking sentinel (mute ASR during TTS), silence-based turns, wake-word window, virtual PTT, and per-run logging. Replies are concise (<=240 chars, 1–2 sentences). [Do not regress latency.]

Goals now:
1) Externalize persona to /config/personality.xml; keep ops knobs in env.
2) Create /docs/stylebook.md (tone, tics, refusal tone).
3) Add ephemeral memory (N=5) with a one-sentence “topic tagline” used in prompts.
4) Make fillers/ACKs latency-shaped from XML defaults; env overrides still apply.
5) Rotate per-run logs to /logs/Eddie_Convo_<ts>.jsonl and write session_meta.json.

Output as surgical diffs (file + exact changes) and a 3–5 step micro test plan per change.
```

---

## Appendix B — Quick snippets (drop‑in)

### B.1 PowerShell: per‑run log rotation (`start_eddie.ps1`)
```powershell
$ts = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$root = $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null
$env:EDDIE_LOG = "$root\logs\Eddie_Convo_$ts.jsonl"
```

### B.2 `.gitignore` additions
```
/logs/
*.wav
/eddie_speaking.flag
```

### B.3 Stylebook seed (paste to `/docs/stylebook.md`)
```
# Eddie Stylebook (v0.1)
Voice: concise, neutral, helpful. Humor: dry, gentle. No sarcasm, no emojis.
Ack palette (pick 1, not stacked): “Right.” “Okay.” “Got it.” “Sure.”
Refusal: brief + helpful alternative (“I can’t do that, but I can …”)
Examples:
- Greeting → “Hey, listening.”
- Error → “Something hiccuped; try again.”
- Recap → “We’re on OBS scenes; you asked about Talking.”
```

---

## Appendix C — Mapping notes (env ↔ XML)
- `VOICE_NAME` (env) ↔ `<Stylebook voice="...">` (XML). XML provides defaults; env can override.
- Filler knobs: `FILLER_FIRST_MS`, `FILLER_POST_PAUSE_MS`, `FILLER_PROB` ↔ `<Fillers firstMs="..." postPauseMs="..." prob="...">`.
- Reply length: `MAX_REPLY_CHARS` remains an env—Phase‑2 keeps ≤240. XML doesn’t need to duplicate.
- Memory span: `<Memory ephemeralTurns="5">` governs ring buffer length.
- Prosody presets: `<Prosody><Preset .../></Prosody>` provide names; map to TTS if available, else textual rhythm.
