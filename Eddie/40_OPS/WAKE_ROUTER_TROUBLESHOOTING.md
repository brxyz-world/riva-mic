# Wake == Router (Troubleshooting)

**Checklist**
- `WAKE_TEXT_MODE=exact` and exact phrases in `WAKE_TEXT_KEYS`.
- Wake window enforced (`WAKE_WINDOW_S`), with optional `WAKE_WINDOW_BY_MOOD` JSON.
- Router keys `R_hello`, `R_exit`, `R_hush` are minimal/deterministic.

**Symptoms → Fixes**
- Outside-window transcripts still route → verify wake gating precedes routing.
- Partial-match wake ups → use `exact` or tighten regex.
- Longform firing on simple utility turns → confirm only `R_story.*`, `R_allegory`, `R_reflect` set `long_form=true`.
