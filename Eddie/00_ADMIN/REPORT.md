# Eddie — Context Pack Bootstrap Report (Revised)

## Tree Highlights
- Canonical hierarchy: `00_ADMIN`, `10_CONTEXT`, `20_MEMORY`, `30_SYSTEMS`, `40_OPS`, `40_ARTIFACTS`, `50_JOURNAL`, `config`. 
- Orchestrator/audio centralized under `Eddie/30_SYSTEMS/`; persona & wake config under `Eddie/config/`.
- Live logs in `logs/`; session bundles in `archive/docs/`; raw transcript history in `archive/logs/`. :contentReference[oaicite:11]{index=11}

## Files/Docs (current)
- Policy: `Eddie/10_CONTEXT/ROUTER_POLICY_NOTES.md`
- Operator: `Eddie/30_SYSTEMS/orchestrator/OPERATOR_PANEL.md` *(single source; supersedes prior `LONGFORM_KNOBS.md`)* :contentReference[oaicite:12]{index=12}
- Release: `Eddie/00_ADMIN/RELEASE_2.2.4.md` (measures verbatim) :contentReference[oaicite:13]{index=13}
- Readme (short): top-level `README.md` linking to these docs. :contentReference[oaicite:14]{index=14}

## Open Threads
- **Longform p95 latency:** target ≤ 15s at `~384` tokens (warm path). 
- **JSONL schema unify (CETC-2):** retrieval fields, model params, perf timings as stable keys.
- **RAG path (CETC-3..5):** memory service → embed index → orchestrator inject.

---

## Appendix A — Divergence Protocol & Doc Loop
- **Reality > scaffolding.** If dev needs diverge, ship the working path, then return to align canon.  
- **On every push/PR:**  
  1) Update `RELEASE_<ver>.md` status/links,  
  2) Refresh EHPDM tail (versions/cards),  
  3) Re-run GoogleLM notebook brief and paste the export into `50_JOURNAL/` (or link).  
- **PR checklist (light):** README only links to Operator Panel; 2.2.4 measures still testable; router keys unchanged or updated in `ROUTER_POLICY_NOTES.md`.

## Appendix B — Consolidation Map
- **Operator knobs** → `OPERATOR_PANEL.md` (single source). README links.  
- **Router policy** → `10_CONTEXT/ROUTER_POLICY_NOTES.md` (worked examples; keep `personality.xml` minimal).  
- **Canon roadmap** → `Etymo_Holoson_Personality_Dev_Map.xml` (versions, task cards, measures). :contentReference[oaicite:15]{index=15}  
- **Logs** → `logs/` (hot); rollups to `archive/logs/`; state logs to `archive/docs/`. 
