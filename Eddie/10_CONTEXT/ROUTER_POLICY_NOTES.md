# Router Policy — Longform Gating via Keys/Moods (Canon)

## Keys (canonical ids)
- `R_hello` (greeting) → `long_form=false`
- `R_hush` / `R_exit` (utility) → `long_form=false`
- `R_story.begin` (explicit story ask) → `long_form=true`
- `R_story.continue` (topic follow-up: “expand”, “go deeper”, “tell me more”) → `long_form=true`
- `R_allegory` (rare allegorical riff on the active topic) → `long_form=true`
- `R_reflect` (rare reflective monologue ending with a question) → `long_form=true`

## Mood & Wake
- **Primary story mood:** `Content_Eve`. Other moods may bias story routes but must not override explicit utility routes.
- Wake phrase opens the time-boxed window; it never sets `long_form` by itself.
- Default wake window 20s with per-mood overrides.

## Invariants
1. Only keys above may set `long_form=true`.  
2. Utility keys must always set `long_form=false`.  
3. Allegory/reflect are **rate-limited** (≤1 per 5 turns).  
4. Log `{routed_key, mood_id}` for audit on every longform turn.
