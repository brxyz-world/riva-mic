# Versioning, Task Cards, and Streams (Canon)

## Terms
- **Version (Release):** Shippable increment (e.g., `2.2.4`) documented in `docs/Eddie/00_ADMIN/RELEASE_<ver>.md` and in the EHPDM XML.
- **CETC (Context Engineering Task Card):** Cross-release work item that runs from concept → done.
- **Streams:** Concern slices within a card:
  - **1R Runtime** (router gating, ready-check, behavior at mic/orchestrator)
  - **1S Systems** (chunker, logging/console shape, safety budgets)
  - **1A Artifacts/Docs** (operator docs, README links, notes)

## Mapping
- A Version may advance multiple CETC streams.
- A CETC may span multiple Versions and sub-milestones (e.g., **CETC-1.1** init, **CETC-1.2** runtime/systems).
- Canon lives here + EHPDM; README **links** to operator docs.

## Current Cards & Milestones (high level)
- **CETC-1 Longform Mode Switch**  
  - Streams: **1R/1S/1A**  
  - **CETC-1.0 → 2.2.3** (baseline smoketest)  
  - **CETC-1.1 → 2.2.4 (init)**: operator panel, gating canon, branch split  
  - **CETC-1.2 → 2.2.4 (impl)**: router gating, smarter chunker, ready-check default ON
- **CETC-2 JSONL Schema Unify** → 2.2.4
- **CETC-3 Memory Service (min)** → 2.2.5
- **CETC-4 Capsule Embed Index** → 2.2.5
- **CETC-5 Orchestrator RAG Inject** → 2.2.6
- **CETC-6 Test Harness CLI** → 2.3.0
- **CETC-7 Docs Refresh** → ongoing

## Branch / PR Guidance (for implementers)
Create one short-lived branch per stream:
- `rel/2.2.4-cetc1R`, `rel/2.2.4-cetc1S`, `rel/2.2.4-cetc1A`
Include a Measures checklist copied from the Release file.
