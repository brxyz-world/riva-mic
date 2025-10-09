# Mood Layer Policy

## Allowed Moods
- neutral — baseline tone; default when no overrides are active.
- upbeat — higher energy, warmer fillers, wider smile mapping.
- focused — clipped delivery, prioritise clarity over charm.
- mellow — softer pacing, lower volume, grounded presence.

## Setting & Decay
- Mood is set explicitly by operator command, tool output, or capsule tag.
- If no new signals arrive, decay to `neutral` after 7 minutes of inactivity.
- Consecutive mood requests of the same type refresh the decay timer.
- Hush/exit flows force a transition back to `neutral` once playback stops.

## Router Conditioning
- Each router rule may declare a `mood` hint; omit to inherit the current state.
- `replyVariants` support `replyWeights` per mood using `weightsByMood`.
- If a rule defines `mood: forced`, the orchestrator pins that mood for the response only.
- When no weights exist for the active mood, fall back to the `neutral` distribution.

## UE Tag Emission
- Router responses emit `moodTag` alongside `intentTag` for the body-language bridge.
- Mapping:
  - neutral -> `body_idle`
  - upbeat -> `body_expressive`
  - focused -> `body_point`
  - mellow -> `body_listen`
- Unreal bridge uses the tag to select preset blends; fallback is `body_idle`.

## Open Items
- TODO: Mood transitions should emit telemetry into `40_OPS/metrics/`.
- TODO: Align filler library in `personality.xml` with mood-specific variants.