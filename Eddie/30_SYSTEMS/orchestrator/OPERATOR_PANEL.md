# Operator Panel — Longform & Session Controls (Canon, single source)

> README links here (no duplication).

| Env / Knob                    | Safe Range / Values | Default | Intent |
|------------------------------|---------------------|---------|--------|
| `LONG_FORM`                  | `0` / `1`           | 0       | Enable multi-paragraph mode; lift clipper; inject `<END>` if no hard stops. |
| `LONGFORM_NUM_PREDICT`       | 256–640             | 384     | Token budget when `LONG_FORM=1`. |
| `NUM_PREDICT`                | ≥160                | 160     | Baseline short-form budget. |
| `LONGFORM_MIN_CHARS`         | 300–1200            | 500     | Suggest/continue threshold for narrative length. |
| `LONGFORM_TEMPERATURE`       | 0.4–0.9             | 0.6     | Creativity for longform. |
| `LONGFORM_SYSTEM_APPEND`     | text ≤ ~2k chars    | preset  | Appendix injected only when `LONG_FORM=1`. |
| `LONGFORM_TTS_ENABLE`        | `0` / `1`           | 1       | Speak longform stories in sentence chunks. |
| `LONGFORM_TTS_CHARS`         | 120–400             | 280     | Target characters per TTS chunk. |
| `LONGFORM_TTS_MAX_S`         | 30–180              | 120     | Total speech seconds cap. |
| `LONGFORM_PRINT`             | `0` / `1`           | 1       | One full-story console emission per longform turn. |
| `LONGFORM_READY_CHECK`       | `0` / `1`           | **1**   | Mic-only “you ready?” gate (default ON). |
| `LONGFORM_RETRY`             | `0` / `1`           | 0       | Harness auto-retry once if below threshold. |
| `LONGFORM_SUGGEST_CONTINUE`  | `0` / `1`           | 1       | Offer to continue if short. |
| `LONGFORM_AUTOCONFIRM`       | `0` / `1`           | 0       | Auto-consent to fetch one continuation. |
| `LONGFORM_AUTOCONFIRM_*`     | policy fields       | varied  | Continuation budget/timeouts/phrasing. |
| `WAKE_WINDOW_S`              | 5–60                | 20      | Base wake window seconds. |
| `WAKE_WINDOW_BY_MOOD`        | JSON map            | {}      | Per-mood overrides, e.g. `{"Content_Eve": 30}`. |

**Notes**
- Router decides when `LONG_FORM` is valid (policy below). Keep `STOP_TOKENS` unset in longform unless hard stops are required; `<END>` is injected automatically.
