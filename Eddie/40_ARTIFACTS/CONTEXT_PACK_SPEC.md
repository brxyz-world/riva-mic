Eddie/10_CONTEXT/CONTEXT_PACK_SPEC.md

(Drop this into the Eddie notebook. It’s written as canon, not a status log.)

title: "Eddie Context Pack — Practical Spec & Bootstrap (v0.2)"
status: "active"
tags: [#eddie, #context-pack, #router, #persona, #rag, #ops, #agentkit]
links:
related: ["../30_SYSTEMS/ORCHESTRATOR.md", "../30_SYSTEMS/VOICE_IO.md", "../40_ARTIFACTS/REPO_LOG.md"]
Purpose (one line)

Define the practical structure, files, and bootstrap steps for Eddie’s Context Pack and Memory layer so we can query in Google LM Notebook now and plug into Eddie’s local retrieval later without rework.

Naming correction (versions vs programs)

Eddie is the local system and versioned as Eddie 2.x (current: 2.2 → spine of memory; 2.3 → memory fusion + show-ops agents; 2.4 → body integration).

The Etymo Show is the public livestream program (3.x era begins when Eddie consistently hosts the show).

The Etymo Network (24/7 multi-show) and The Etymo (culture matrix) are systems above the show.
Action: Replace “Phase” language in operative files with Version (Eddie) and Program (Show/Network/The Etymo). Keep phase terms only in historical docs.

Directory hierarchy (canonical)
Eddie/
├─ 00_ADMIN/
│  └─ README.md
├─ 10_CONTEXT/
│  ├─ CONTEXT_PACK.md                # Human-readable canon: Facts/Rules/Tasks/Lexicon/Narrative
│  ├─ ROUTER_POLICY.md               # Intents, replyVariants, replyWeights, tool hooks (worked examples)
│  ├─ LEXICON.md                     # Names, aliases, normalization rules
│  ├─ ARTIFACTS_INDEX.json           # Machine index of artifacts & pointers to shards
│  ├─ context.manifest.json          # Entry point for Memory Service (shards, parsers, privacy)
│  └─ schemas/
│     ├─ ops.manifest.schema.json
│     ├─ capsule.schema.json
│     └─ graph.edge.schema.json
├─ 20_MEMORY/
│  ├─ capsules/                      # Event-sourced summaries (“capsules”), with metadata sidecars
│  │  ├─ 2025-10-08__dev-standup.md
│  │  └─ 2025-10-08__dev-standup.meta.json
│  ├─ graph/                         # Durable facts/relations (graph edges & nodes)
│  │  ├─ nodes.jsonl                 # {id,type,props}
│  │  └─ edges.jsonl                 # {src,dst,rel,weight,provenance}
│  └─ indexes/                       # Embedding & adjacency indexes (build artifacts)
├─ 30_SYSTEMS/
│  ├─ ORCHESTRATOR.md                # Recall discipline & prompt assembly
│  ├─ MEMORY_SERVICE.md              # API contract: POST /rag/query, /rag/upsert, /graph/query
│  └─ VOICE_IO.md
├─ 40_OPS/
│  ├─ segments/                      # Episode outlines (from Show-ops agent)
│  ├─ highlights/                    # Candidate timestamps (JSON)
│  └─ metrics/                       # Per-clip/episode feedback → planning
├─ 50_JOURNAL/
│  └─ CHANGELOG_2025-10.md
└─ config/
   └─ personality.xml                # Router IDs/policies, minimal intents (hello/exit/hush)

File types (why)

Markdown (.md): canonical human-edited content (reviewable, diffable).

JSON/JSONL: manifests, graph nodes/edges, programmatic indexes (schema’d, easily parsed).

Sidecars (.meta.json): provenance, sensitivity, tags, people, source—kept beside the md.

Context Pack contents (CONTEXT_PACK.md skeleton)
# Summary
Compact description of Eddie’s purpose and scope.

# Facts
- Local, low-latency; wake phrase “hello Eddie”.
- Minimal router: R_hello (gate), R_exit (hard), R_hush (soft).

# Rules
- Persona-as-policy: ACK, fillers, fallback concise.
- Router grammar: keys[] → replyVariants[] (+replyWeights?, +tool?).

# Tasks (Now / Next / Later)
- Now: freeze Router examples; seed capsules; stand up Memory Service stub.
- Next: hybrid retrieval; Operator QA Agent; body IO bridge spec.
- Later: Show-ops schedule agent v1; clip metrics loop.

# Lexicon
- “Context Pack”, “capsule”, “router-as-grammar”, “confirm-to-act”, etc.

# Narrative
Why context ≫ features (bridge to 3.x program).

Memory architecture (fusion model)

Event-sourced capsules (Model-2):
Small, timestamped summaries of sessions/segments stored in 20_MEMORY/capsules/*.md with *.meta.json.
Purpose: compact, semantically searchable narrative memory.

Context graph (Model-1):
Durable facts & relations as nodes.jsonl and edges.jsonl with provenance (capsule IDs, artifact paths).
Purpose: multi-hop reasoning, precise recall of identities, rules, and long-lived facts.

Retrieval pipeline (hot path):

Router selects need (e.g., identity, past-topic, policy).

/rag/query runs dual route:

Capsule search: embedding KNN → top-k (≤2) capsules.

Graph query: constrained traversal (≤2 hops) for durable facts.

Aggregator compresses to ≤ ~120–200 tokens (answer-first, then refs).

Orchestrator injects (a) short session facts, (b) one-liner summary, (c) compact retrieval bundle.

Maintenance (cold path):
Nightly AgentKit maintenance agent runs:

Promote recurring facts → graph edges.

Consolidate old capsules (week → digest), preserve provenance.

Memory Service (API)

POST /rag/query
Input: { query, need: ["identity","history","policy"], k_caps=2, k_edges=1 }
Output: { capsules: [...], facts: [...], tokens: n }

POST /rag/upsert (capsules)

POST /graph/upsert (nodes/edges)

POST /graph/query (patterns)

(Stub locally; Notebook can point to filesystem loader that emulates these endpoints.)

AgentKit’s role (clear boundaries)

Yes: Show-Ops (schedule → segments → promo), Operator QA, Maintenance agents (summarize, reindex), Approval flows (confirm-to-act), Tool registry governance.

No (in hot path): Direct chunking/embedding inside the live turn. AgentKit calls Memory Service; it doesn’t embed in-loop.

Unreal / body integration (Eddie 2.4 pre-3.0)

I/O bridge spec (outline):

Input: Riva ASR → Orchestrator (barge-in supported).

Output: TTS viseme stream + intent tags → UE MetaHuman (Live Link / ARKit blendshape map).

Gesture layer: rule-based mapper from intent tags → body language presets (idle, agree, emphasize, listen).

Latency guard: ≤ 150ms round-trip budget for viseme alignment.

Google LM Notebook “interim mode”

Mirror exact tree above inside the notebook sources.

Keep CONTEXT_PACK.md, ROUTER_POLICY.md, capsules/ and graph/ as separate sources.

Disable “rolling recall” in the Notebook hosts; rely on capsules + graph as explicit sources to avoid drift.

Bootstrap (one-liner & tasks)

Shell scaffold (safe to run anywhere):

# from repo root
mkdir -p Eddie/{00_ADMIN,10_CONTEXT/schemas,20_MEMORY/{capsules,graph,indexes},30_SYSTEMS,40_OPS/{segments,highlights,metrics},50_JOURNAL,config}
cat > Eddie/10_CONTEXT/CONTEXT_PACK.md <<'MD'
# Eddie Context Pack (seed)
[Fill sections: Summary/Facts/Rules/Tasks/Lexicon/Narrative]
MD
cat > Eddie/10_CONTEXT/ROUTER_POLICY.md <<'MD'
# Router Policy (seed)
- R_hello, R_exit, R_hush with 2–3 examples each.
MD
cat > Eddie/10_CONTEXT/LEXICON.md <<'MD'
# Lexicon (seed)
MD
cat > Eddie/10_CONTEXT/context.manifest.json <<'JSON'
{ "shards": ["capsules","graph"], "privacy": { "default":"internal" }, "paths": {
  "capsules":"Eddie/20_MEMORY/capsules", "graph":"Eddie/20_MEMORY/graph" } }
JSON
cat > Eddie/30_SYSTEMS/MEMORY_SERVICE.md <<'MD'
# Memory Service API (stub)
POST /rag/query | /rag/upsert | /graph/query | /graph/upsert
MD


Capsule sidecar schema (Eddie/10_CONTEXT/schemas/capsule.schema.json):

{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "title":"Capsule",
  "type":"object",
  "required":["id","date","tags","summary","tokens","provenance"],
  "properties":{
    "id":{"type":"string"},
    "date":{"type":"string","format":"date-time"},
    "tags":{"type":"array","items":{"type":"string"}},
    "summary":{"type":"string"},
    "tokens":{"type":"integer"},
    "provenance":{"type":"object","properties":{
      "source":{"type":"string"},
      "paths":{"type":"array","items":{"type":"string"}}
    }}
  }
}


Router reply schema snippet (ROUTER_POLICY.md):

- id: R_hello
  keys: ["hello eddie", "hey eddie", "yo eddie"]
  replyVariants:
    - "hey — i’m listening."
    - "yo. awake & here."
  replyWeights: [0.6, 0.4]
  tool: null


(If you want Codex CLI automation, wrap the shell scaffold and file templates as a “Scaffold: Eddie Context” task; Codex can render the files verbatim.)

Guardrails & privacy

Sidecars must include sensitivity: public|internal|private.

Public agents/hosts exclude private by default.

Confirm-to-act enforced via AgentKit/n8n for any side-effects (posting, scheduling).

Now / Next / Later

Now

Create folders & seed files (scaffold above).

Add 2–3 real capsules and 3–5 graph edges (e.g., “User:Alice —likes→ AI Art”).

Freeze Router examples for R_hello/R_exit/R_hush.

Next

Implement /rag/query stub that loads capsules+graph from disk and returns ≤2 snippets.

Stand up Operator QA Agent in AgentKit that calls /rag/query.

Write UE body I/O bridge outline (ports, formats, cues).

Later

Add Show-Ops schedule agent v1; output 40_OPS/segments/*.md.

Close the VODsmith feedback loop (40_OPS/metrics/*.json → planning).

Promote recurring facts from capsules → graph nightly.

Notes for your mind-map & language cleanup

Update your Google LM mind-map to: Eddie 2.2 → 2.3 → 2.4 → (Program) The Etymo Show 3.x → (System) The Etymo Network → (Matrix) The Etymo.

In operative files, replace ambiguous “phase” with Version (Eddie) or Program/System (Show/Network/The Etymo).

Add a short “Glossary: Version vs Program” block to your CONTEXT_PACK.md to kill ambiguity at the source.