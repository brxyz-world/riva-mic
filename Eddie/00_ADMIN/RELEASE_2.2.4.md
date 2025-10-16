# Eddie 2.2.4 — Router-Gated Longform & Operator Panel (Canon)

**Scope:** Advance CETC-1 to **1.1 (init)** + **1.2 (impl)**. Longform becomes an explicitly gated router outcome, with a mic-only ready-check (default ON) and smarter TTS chunking. README links to a single Operator Panel.

**Policy Triggers (non-testing)**
- **Explicit story ask:** if the user asks for a “story” or uses the word **story**, route `R_story.begin` → `long_form=true`.
- **Topic follow-ups:** if the user says “expand”, “go deeper”, “tell me more”, or similar on the **same topic as Eddie’s last reply**, route `R_story.continue` → `long_form=true`.
- **Allegory / reflective monologue (rare):** when conversation drifts slightly and mood permits, route `R_allegory` or `R_reflect` (max once per 5 turns).
- **Mood guardrail:** **Content_Eve** is the primary “story mood”; other moods may bias but must not force longform.

**Wake Window**
- Default **20s**; mood overrides allowed (see EHPDM `<WakeWindow>`).

**Links**
- Operator panel: `../../30_SYSTEMS/orchestrator/OPERATOR_PANEL.md`
- Troubleshooting (wake==router): `../../40_OPS/WAKE_ROUTER_TROUBLESHOOTING.md`
- State logs (2.2.3): `../../../archive/docs/`
- EHPDM XML: `../../../Etymo_Holoson_Personality_Dev_Map.xml`

## Streams in this Release
- **1R** Router/mood-gated longform; mic-only ready-check (default ON); optional thinking beats.
- **1S** Single console print per longform turn; smarter chunker avoids mid-word cuts.
- **1A** Operator panel file; wake==router note; README links → no duplication.

## Success Measures (canon) for 2.2.4—close CETC-1 when all pass

### 1R (Runtime)
- long_form=true on routed story turns; zero false positives on utility prompts.
- ready_check_ms p95 ≤ 1000; timeout path < 5% longform sessions.
- Thinking beats on longform gens > 8s; 0 beats in short-form.

### 1S (Systems)
- Exactly one console print per longform turn in smoketest.
- midword_cut_rate < 5%; spoken_s ≤ MAX_S + 1.0.

### 1A (Artifacts/Docs)
- README links to operator panel (no duplication); “wake==router” note present; links OK.

### Non-regression (short-form)
- first_audio_ms p95 regression ≤ +100ms vs 2.2.3.

### Longform Quality
- Story ≥ 600 chars, ≥ 4 paragraphs, no clipper; LLM latency p95 ≤ 15s at LONGFORM_NUM_PREDICT≈384 (warm).

## Deliverables
- `Eddie/10_CONTEXT/ROUTER_POLICY_NOTES.md`
- `Eddie/30_SYSTEMS/orchestrator/OPERATOR_PANEL.md`
- `Eddie/40_OPS/WAKE_ROUTER_TROUBLESHOOTING.md`
- EHPDM tail updated.
