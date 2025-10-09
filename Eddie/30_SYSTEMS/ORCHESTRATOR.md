# Orchestrator System

## Responsibilities
- Manage input from ASR and wake layer, assemble router requests.
- Apply router outputs to build prompts and trigger TTS playback.
- Emit telemetry into archive/logs/ and ops metrics (future).

## Mood Integration TODOs
- TODO: Load current mood state from Memory Service or cached capsule.
- TODO: Apply `weightsByMood` defined in ROUTER_POLICY when selecting reply variants.
- TODO: Publish `moodTag` alongside intent labels for the Unreal bridge.

## Open Issues
- Align persona filler selection with mood-specific distributions (see `MOOD_POLICY.md`).