# Lethe vNext — `todo.md`

**TL;DR:** Make Lethe faster and more accurate by adding Provence-style sentence pruning, a global token-budget knapsack with bookend packing, structure-aware chunking, dynamic fusion, and light graph retrieval—under strict reproducibility, verification, and statistical gates.

## Invariants (do not change)

* Local-first: no outbound network; only localhost (e.g., Ollama) allowed.
* Single datastore: SQLite (+ vector ext/HNSW WASM). All artifacts traceable via SHAs/hashes.
* Budget parity: equal token budget and ±5% params/FLOPs across A/B unless “systems budget” is explicitly declared.
* Oracles are source-of-truth; contracts & properties enforced at runtime.
* Hermetic spin-up required; boot transcript must verify.

## Assumptions & Scope

* **Assumption:** Repos: `{{ROOT}}/ctx-run` (tool), `{{ROOT}}/lethe-research` (bench/paper). CPU node 8C/16T, 32GB RAM; optional WebGPU.
* **Assumption:** Placeholders: `{{MODEL_NAME}}` (embeddings), `{{RERANK_NAME}}` (cross-encoder), `{{LLM_NAME}}` (local LLM), `{{DATASET}}` (LetheBench ≥600 queries/3 domains).
* **Assumption:** Thresholds if unspecified: **T\_mut=0.80**, **T\_prop=0.70**, **SAST\_high=0**, latency **p50≤3s**, **p95≤6s**, **RSS≤1.5GB** (CPU).
* **Scope:** Implement/measure: sentence pruning, knapsack+bookend, structure-aware chunking, dynamic fusion, entity graph retrieval, one-shot active retrieval, ANN/quant tuning. Paper rebuild included. LLM rerank optional behind flag.

## Objectives

1. **Primary quality:** ≥ **+10%** nDCG\@10 vs best strong baseline; **BCa 95% CI** lower bound > 0 (paired bootstrap), Full & Focused slices.
2. **Efficiency:** Reduce tokens-to-LLM by **30–50%** with **Answer-Span-Kept ≥ 98%** and **Contradiction-rate** not worse.
3. **Verification:** Achieve mutation ≥ **T\_mut** and property/metamorphic coverage ≥ **T\_prop**.
4. **Spin-up:** One-shot hermetic boot; golden smokes pass; **signed boot transcript** with env digest and hashes.
5. **Systems:** Maintain **p50≤3s**, **p95≤6s**, **RSS≤1.5GB** (LLM variants p50≤4s) at equal token budgets.

## Risks & Mitigations

* Over-pruning evidence → **Mitigation:** group-keep rules; Answer-Span-Kept guard; fallback to unpruned.
* Position bias harms mid-pack context → **Mitigation:** bookend packing with head/tail anchors.
* JSON/schema fragility → **Mitigation:** strict validators; balanced-brace parser; runtime guards; timeouts.
* Index bloat / ANN drift → **Mitigation:** cap expansions; quantization validation; ANN parameter sweeps with parity checks.
* Overfit of dynamic fusion/learned planning → **Mitigation:** nested CV; dev/test isolation; CI gates on effect sizes.

## Method Outline (idea → mechanism → trade-offs → go/no-go)

### Workstream A — Sentence Pruning (Provence-style)

* **Idea:** Query-conditioned sentence masking inside selected chunks.
* **Mechanism:** Score sentences with cross-encoder or ONNX sequence labeler; binary mask; preserve adjacent co-entailing sentences; never break fenced code; recompute chunk score as max/avg of kept sentences.
* **Trade-offs:** Extra scoring pass; calibration per domain.
* **Go/No-Go Gate:** Answer-Span-Kept ≥ 98%; ΔnDCG\@10 ≥ 0 with equal tokens; latency p50 Δ ≤ +150ms.

### Workstream B — Global Token-Budget Knapsack + Bookend Packing

* **Idea:** Optimize sentence selection globally under token cap; exploit start/end attention bias.
* **Mechanism:** 0/1 knapsack over kept sentences $(w_i,t_i)$; group constraints; then bookend: top evidence at head & tail, zig-zag remainder; prepend short index + claim-first preamble; tail beacon repeat.
* **Trade-offs:** Minor compute; ordering stability.
* **Go/No-Go Gate:** +nDCG\@10 ≥ 5% (CI>0); tokens −30%+; budgets met.

### Workstream C — Structure-Aware Chunking (AST/Logs)

* **Idea:** Align chunks with semantics for code/logs.
* **Mechanism:** tree-sitter fences for functions/classes; log grammar for tool outputs; tag `kind=tool_struct`.
* **Trade-offs:** Parser coverage; maintenance.
* **Go/No-Go Gate:** MRR\@10(Code) +5% and Recall\@50(Tool) +6% without latency regressions.

### Workstream D — Dynamic Fusion (Learned α,β + Metadata Boosts)

* **Idea:** Query-aware α/β and boosts improve retrieval mix.
* **Mechanism:** Ridge/LightGBM on query/state features to predict α,β (clamped); rule boosts for code/error/path tokens.
* **Trade-offs:** Overfit risk; minimal inference cost.
* **Go/No-Go Gate:** +nDCG\@10(Code) ≥ 3% (CI>0); p50 unchanged.

### Workstream E — Entity-Graph Retrieval (Light)

* **Idea:** Bias toward high-centrality neighbors.
* **Mechanism:** Per-session co-occurrence graph; personalized PageRank score mixed with hybrid.
* **Trade-offs:** Small memory; canonicalization.
* **Go/No-Go Gate:** Coverage\@N +10% on cross-referential queries; p50 Δ ≤ +50ms.

### Workstream F — Active Retrieval (One-Shot Tail Addendum)

* **Idea:** Insert one focused retrieval if preamble confidence low.
* **Mechanism:** Heuristic triggers; append best sentence at tail; hard cap to 1 probe.
* **Trade-offs:** p95 spikes; logging complexity.
* **Go/No-Go Gate:** Contradiction-rate −10% with p50 Δ ≤ +200ms; p95 ≤ 6s.

### Workstream G — ANN Tuning + Quantization

* **Idea:** Save memory & latency with negligible quality loss.
* **Mechanism:** int8 scalar quantization of embeddings; HNSW tuning (`ef_search` adaptive); warm shards.
* **Trade-offs:** Small cosine error.
* **Go/No-Go Gate:** RSS −40–60%, p50 −10–20%, ΔnDCG ≤ −0.5% (non-sig).

## Run Matrix

| ID | Method/Variant                                   | Budget              | Inputs                      | Expected Gain                   | Promote if…                                         |
| -- | ------------------------------------------------ | ------------------- | --------------------------- | ------------------------------- | --------------------------------------------------- |
| V0 | Strong Baseline (hybrid+rERANK+entity-diversify) | Parity              | Current Lethe               | Reference                       | Gates pass; acts as KNOWN-GOOD                      |
| V1 | + Sentence Pruning (Provence)                    | Parity              | V0 + sentence mask          | −30–50% tokens; = or ↑ quality  | Answer-Span-Kept ≥98%; ΔnDCG\@10 ≥0; p50 Δ ≤ +150ms |
| V2 | + Knapsack + Bookend                             | Parity              | V1 + optimizer + linearizer | +5–8% nDCG; position-bias gains | +≥5% nDCG\@10 (CI>0); budgets met                   |
| V3 | + Structure-Aware Chunking                       | Parity              | V2 + AST/log chunker        | Code/Tool slice lifts           | MRR\@10(Code) +5%; Recall\@50(Tool) +6%             |
| V4 | + Dynamic Fusion (learned α,β) + metadata boosts | Parity              | V2/V3 + regressor           | +3–5% Code nDCG                 | +≥3% nDCG\@10(Code) (CI>0); p50 unchanged           |
| V5 | + Entity-Graph Retrieval                         | Parity              | V4 + graph score            | Coverage +10–15% on hard refs   | Coverage\@N +≥10% (CI>0); p50 Δ ≤ +50ms             |
| V6 | + Active Retrieval (1-shot)                      | Systems budget      | V5 + one probe              | −10% contradictions             | Contradiction-rate −≥10%; p50 Δ ≤ +200ms; p95 ≤ 6s  |
| V7 | + ANN Quantization/Tuning                        | Systems budget      | V5/V6 + int8 + HNSW tune    | RSS −40–60%; p50 −10–20%        | ΔnDCG ≤ −0.5% (ns); memory/latency gates pass       |
| V8 | (Opt) LLM Rerank + Contradiction Penalty         | Declared LLM budget | V5 + LLM rerank/penalty     | +3% nDCG; −15% hallucinations   | +≥3% nDCG\@10; hallucination −≥15%; p50 ≤ 4s        |

## Implementation Notes

* **Attach points:** `packages/core/{retrieval.ts,rank_diversify.ts,chunker.ts,orchestrate.ts,summarize.ts}`, `packages/reranker/`, `packages/sqlite/`.
* **APIs/Contracts:**

  * `sentence_prune(query, chunk) -> {chunk_id, kept: [{sid, span:[s,e], tokens, score}], chunk_score}`; **invariants:** spans within original; sum(tokens)>0 unless fallback; JSON schema `verification/schemas/pruned.json`.
  * `knapsack_pack(sentences, budget, groups) -> ordered_ids` (respects group-keep and code fences); **invariant:** Σtokens ≤ budget; all cited spans ∈ kept.
  * `bookend_linearize(ordered, head_k=1, tail_k=1) -> sequence` preserves code blocks; logs head/tail indices.
  * `dynamic_fusion(features) -> {alpha,beta}` clamped \[0.3,1.5]; fallback to config on error.
  * `graph_score(query_entities) -> {chunk_id: score}` cached per session.
* **Metamorphic properties:**

  * Adding irrelevant sentences **must not** increase Claim-Support\@K at fixed budget.
  * Duplicating any kept sentence **must not** change scores/pack order.
  * Synonymized query (lemmatized) keeps nDCG within ε (report).
  * Removing a gold sentence **must** reduce Claim-Support\@K.
  * Shuffling non-selected items has no effect.
* **Quantization:** int8 with (scale, zero-point) per vector; store in BLOB; dequant on query only.
* **Telemetry:** Per-query JSONL: config hash, seeds, latency breakdown, RSS, tokens, nDCG/Recall, Coverage\@N, Answer-Span-Kept, Contradiction-rate.
* **Seeds/Hashes:** Record dataset hash, model versions/SHAs, ANN index hash, config hash.

## Acceptance Gates

* **Spin-up:** Clean checkout → container build → migrate/seed → readiness OK → smokes pass → **signed boot transcript**.
* **Static:** Types/linters clean; **0 high/critical SAST**; API surface diffs acknowledged; license policy OK.
* **Dynamic:** Mutation ≥ **T\_mut**; property/metamorphic ≥ **T\_prop**; fuzz ≥ 30m with 0 new medium+ crashes; concolic: 0 feasible high-risk paths.
* **Stats:** 10k paired bootstrap; **BCa 95% CI lower bound > 0** for promoted deltas; FDR within metric families.
* **Systems:** p50≤3s, p95≤6s, RSS≤1.5GB (LLM path p50≤4s); equal token budgets.
* **Safety:** Answer-Span-Kept ≥98%; Contradiction-rate not worse (unless variant specifically targets its reduction).
* **Paper/Artifact:** Tables/figures generated from CSV with hashes; reproducible build.

## “Make-sure-you” Checklist

* Pin toolchains/lockfiles; save env manifest and container digest.
* Validate JSON outputs; enforce schemas at runtime; fail fast.
* Save boot transcript + all artifact hashes; include in bundle.
* Record seeds; run flake detector 100×; fail if ≥1% flakiness.
* Quarantine network; stub external deps; localhost only allowed.
* Keep parity; log any budget deviations; include rationale.

## File/Layout Plan

```
{{ROOT}}/
  lethe-research/
    verification/
      schemas/                  # pack/pruned JSON schemas
      properties/               # property/metamorphic tests
      mutation/                 # operators + harness
      fuzz/                     # grammars + corpora
      concolic/                 # parser/JSON targets
      invariants/               # runtime monitors
    experiments/
      grids/                    # YAMLs for V0–V8
      run.py score.py plots.py select_best.py statistics.py
    datasets/
      builders/ manifests/
    paper/
      template/ figures/ tables/ build.sh
    scripts/
      spinup_smoke.sh bundle_artifact.sh record_env.py compute_risk.py
    artifacts/
      boot_transcript.json metrics/ logs/ diffs/ golden/
  ctx-run/
    packages/core/              # pruning/knapsack/linearizer hooks
    packages/reranker/          # LLM/cross-encoder rerank
    packages/sqlite/            # graph index hooks
```

## Workflows (required)

```xml
<workflows project="lethe_vnext" version="1.0">

  <workflow name="building">
    <env id="B0">
      <desc>Pin environment & container</desc>
      <commands>
        <cmd>python3 -m venv .venv && source .venv/bin/activate</cmd>
        <cmd>pip install -r lethe-research/infra/requirements.txt</cmd>
        <cmd>npm ci --prefix ctx-run</cmd>
        <cmd>docker build -t lethe:lock -f lethe-research/infra/Dockerfile .</cmd>
        <cmd>python lethe-research/scripts/record_env.py --out lethe-research/artifacts/env_manifest.json</cmd>
      </commands>
      <make_sure>
        <item>Lockfiles present & hashed</item>
        <item>Env manifest includes SHAs and container digest</item>
      </make_sure>
    </env>

    <assets id="B1">
      <desc>Datasets/models/indexes & hashes</desc>
      <commands>
        <cmd>python lethe-research/datasets/builders/build_all.py --out lethe-research/datasets --manifest lethe-research/datasets/manifests/manifest.json</cmd>
        <cmd>python lethe-research/scripts/hash_tree.py lethe-research/datasets &gt; lethe-research/datasets/manifests/datasets.sha</cmd>
        <cmd>node ctx-run/packages/cli/dist/index.js init .ctx || true</cmd>
      </commands>
      <make_sure>
        <item>≥600 queries across 3 domains; manifest lists licenses & SHAs</item>
      </make_sure>
    </assets>

    <contracts id="B2">
      <desc>Generate oracles & runtime guards</desc>
      <commands>
        <cmd>python lethe-research/verification/generate_schemas.py</cmd>
        <cmd>python lethe-research/verification/generate_properties.py</cmd>
        <cmd>python lethe-research/verification/inject_runtime_guards.py --attach ctx-run/packages/core</cmd>
      </commands>
      <make_sure>
        <item>Schemas validate sample outputs</item>
        <item>Guards enabled: citation integrity, span bounds, JSON shape</item>
      </make_sure>
    </contracts>

    <static id="B3">
      <desc>Static/semantic gates</desc>
      <commands>
        <cmd>npm run -w ctx-run typecheck &amp;&amp; npm run -w ctx-run lint</cmd>
        <cmd>python lethe-research/infra/run_sast.py --repo . --out lethe-research/artifacts/sast.json</cmd>
        <cmd>python lethe-research/infra/api_diff.py --repo ctx-run --out lethe-research/artifacts/api_diff.json</cmd>
      </commands>
      <make_sure>
        <item>0 high/critical SAST; API diffs acknowledged</item>
      </make_sure>
    </static>

    <spinup id="B4">
      <desc>Hermetic boot & smokes</desc>
      <commands>
        <cmd>bash lethe-research/scripts/spinup_smoke.sh</cmd>
        <cmd>python lethe-research/experiments/sanity.py --smoke --out lethe-research/artifacts/smoke.json</cmd>
        <cmd>python lethe-research/scripts/sign_transcript.py --env lethe-research/artifacts/env_manifest.json --smoke lethe-research/artifacts/smoke.json --out lethe-research/artifacts/boot_transcript.json</cmd>
      </commands>
      <make_sure>
        <item>Golden ingest→index→query passes and matches golden hashes</item>
        <item>Boot transcript signed with env digest</item>
      </make_sure>
    </spinup>
  </workflow>

  <workflow name="running">
    <baseline id="R0">
      <desc>Run strong baseline under parity</desc>
      <commands>
        <cmd>python lethe-research/experiments/run.py --exp V0 --grid lethe-research/experiments/grids/V0.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V0 --split dev --out lethe-research/artifacts/V0/dev</cmd>
      </commands>
      <make_sure>
        <item>Same chunker/embeddings across baselines</item>
        <item>Equal token budgets recorded</item>
      </make_sure>
    </baseline>

    <properties id="R1">
      <desc>Property & metamorphic suites</desc>
      <commands>
        <cmd>pytest lethe-research/verification/properties -q</cmd>
        <cmd>pytest lethe-research/verification/metamorphic -q</cmd>
      </commands>
      <make_sure>
        <item>Coverage ≥ 0.70</item>
      </make_sure>
    </properties>

    <fuzz_symbolic id="R2">
      <desc>Fuzzing + concolic on parsers/JSON</desc>
      <commands>
        <cmd>python lethe-research/verification/fuzz/run_fuzz.py --minutes 30 --out lethe-research/artifacts/fuzz</cmd>
        <cmd>python lethe-research/verification/concolic/run_concolic.py --targets pack_json_parser,json_sanitizer --out lethe-research/artifacts/concolic</cmd>
      </commands>
      <make_sure>
        <item>No new medium+ crashes; repros archived</item>
      </make_sure>
    </fuzz_symbolic>

    <mutation id="R3">
      <desc>Mutation adequacy</desc>
      <commands>
        <cmd>python lethe-research/verification/mutation/generate_mutants.py --repo ctx-run --out lethe-research/artifacts/mutants</cmd>
        <cmd>python lethe-research/verification/mutation/run_mutation.py --mutants lethe-research/artifacts/mutants --tests tests --out lethe-research/artifacts/mutation</cmd>
        <cmd>python lethe-research/verification/mutation/score.py --in lethe-research/artifacts/mutation --out lethe-research/artifacts/mutation_score.json</cmd>
      </commands>
      <make_sure>
        <item>Mutation ≥ 0.80</item>
      </make_sure>
    </mutation>

    <experiments id="R4">
      <desc>Variants V1–V8 on dev/test with locks</desc>
      <commands>
        <cmd>python lethe-research/experiments/run.py --exp V1 --grid lethe-research/experiments/grids/V1_prune.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V1 --split dev</cmd>
        <cmd>python lethe-research/experiments/select_best.py --from V1 --rule "AnswerSpanKept>=98 &amp;&amp; ΔnDCG10>=0 &amp;&amp; p50<=3150"</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V1 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V1 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V2 --grid lethe-research/experiments/grids/V2_knapsack_bookend.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V2 --split dev</cmd>
        <cmd>python lethe-research/experiments/select_best.py --from V2 --rule "ΔnDCG10>=5 &amp;&amp; tokens<=budget &amp;&amp; p50<=3000"</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V2 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V2 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V3 --grid lethe-research/experiments/grids/V3_struct_chunk.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V3 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V3 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V3 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V4 --grid lethe-research/experiments/grids/V4_dyn_fusion.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V4 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V4 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V4 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V5 --grid lethe-research/experiments/grids/V5_entity_graph.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V5 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V5 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V5 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V6 --grid lethe-research/experiments/grids/V6_active_retrieve.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V6 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V6 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V6 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V7 --grid lethe-research/experiments/grids/V7_quant_ann.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V7 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V7 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V7 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp V8 --grid lethe-research/experiments/grids/V8_llm_rerank.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V8 --split dev</cmd>
        <cmd>python lethe-research/experiments/run.py --exp V8 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp V8 --split test</cmd>
      </commands>
      <make_sure>
        <item>Per-query CSV + config hashes emitted</item>
        <item>Latency p50/p95 &amp; RSS tracked per variant</item>
      </make_sure>
    </experiments>

    <differential id="R5">
      <desc>Goldens & diffs vs KNOWN-GOOD</desc>
      <commands>
        <cmd>python lethe-research/verification/differential/capture.py --exp best --out lethe-research/artifacts/golden</cmd>
        <cmd>python lethe-research/verification/differential/diff.py --golden lethe-research/artifacts/golden --candidate lethe-research/artifacts/current --out lethe-research/artifacts/diffs</cmd>
      </commands>
      <make_sure>
        <item>No incompatible diffs; contracts green</item>
      </make_sure>
    </differential>

    <runtime id="R6">
      <desc>Shadow replay & invariants</desc>
      <commands>
        <cmd>python lethe-research/verification/invariants/replay.py --n 10000 --out lethe-research/artifacts/invariants.json</cmd>
      </commands>
      <make_sure>
        <item>0 invariant breaks over 10k replays</item>
      </make_sure>
    </runtime>
  </workflow>

  <workflow name="tracking">
    <harvest id="T1">
      <desc>Metrics, statistics, figures, tables</desc>
      <commands>
        <cmd>python lethe-research/experiments/collect.py --root lethe-research/artifacts --out lethe-research/artifacts/metrics/all.jsonl</cmd>
        <cmd>python lethe-research/experiments/statistics.py --in lethe-research/artifacts/metrics/all.jsonl --bootstrap 10000 --bca --fdr --out lethe-research/artifacts/stats</cmd>
        <cmd>python lethe-research/experiments/plots.py --in lethe-research/artifacts --out lethe-research/paper/figures</cmd>
        <cmd>python lethe-research/experiments/make_tables.py --in lethe-research/artifacts --out lethe-research/paper/tables</cmd>
      </commands>
      <make_sure>
        <item>All figs/tables cite source CSV + hash</item>
        <item>Stars only when CI lower bound &gt; 0</item>
      </make_sure>
    </harvest>

    <risk id="T2">
      <desc>Risk score & decision features</desc>
      <commands>
        <cmd>python lethe-research/scripts/compute_risk.py --artifacts lethe-research/artifacts --out lethe-research/artifacts/risk.json</cmd>
      </commands>
      <make_sure>
        <item>Features normalized to [0,1]; thresholds recorded</item>
      </make_sure>
    </risk>
  </workflow>

  <workflow name="evaluating">
    <gatekeeper id="E1">
      <desc>Apply acceptance gates & route</desc>
      <commands>
        <cmd>python lethe-research/scripts/apply_gates.py --gates lethe-research/acceptance.yaml --metrics lethe-research/artifacts/stats/summary.json --out lethe-research/artifacts/decision.json</cmd>
        <cmd>bash lethe-research/paper/build.sh</cmd>
        <cmd>bash lethe-research/scripts/bundle_artifact.sh lethe-research/artifacts lethe-research/paper/lethe_neurips2025.pdf</cmd>
      </commands>
      <make_sure>
        <item>No promotion without CI-backed wins & all gates met</item>
        <item>Paper builds cleanly; no dangling refs/numbers</item>
      </make_sure>
    </gatekeeper>
  </workflow>

  <workflow name="refinement">
    <agent_refine id="N1">
      <desc>Auto-iterate on failures (obligation prompts)</desc>
      <commands>
        <cmd>python lethe-research/scripts/make_actionable_prompt.py --from lethe-research/artifacts/decision.json --out lethe-research/artifacts/agent_prompt.txt</cmd>
        <cmd>bash lethe-research/scripts/agent_cycle.sh --prompt lethe-research/artifacts/agent_prompt.txt</cmd>
      </commands>
      <make_sure>
        <item>Prompt enumerates failing gates & quantitative obligations</item>
      </make_sure>
    </agent_refine>

    <manual_qa id="N2">
      <desc>Human exploration handoff</desc>
      <commands>
        <cmd>python lethe-research/scripts/open_dashboard.py --artifacts lethe-research/artifacts</cmd>
        <cmd>python lethe-research/scripts/prepare_handoff.py --include boot_transcript.json diffs/ repros/ contracts/ --out lethe-research/artifacts/handoff.zip</cmd>
      </commands>
      <make_sure>
        <item>Owner assigned; rollback/kill-switch documented</item>
      </make_sure>
    </manual_qa>
  </workflow>

</workflows>
```

## Minimal Pseudocode (optional)

```python
def sentence_prune(query, chunk):
    # returns masked sentences and a chunk score
    kept = []
    for s in chunk.sentences:
        score = cross_encoder(query, s)  # batched in practice
        if score >= tau or group_rule(s): kept.append((s, score))
    if not kept: return fallback_unpruned(chunk)
    chunk_score = max(sc for _, sc in kept)
    return {"chunk_id": chunk.id, "kept": kept, "chunk_score": chunk_score}

def knapsack_pack(sentences, B, groups):
    # 0/1 knapsack with group-keep constraints (greedy+swap)
    # sentences: [(sid, tokens, weight, group_id)]
    S = greedy_by_ratio(sentences, B, groups)
    S = local_swaps(S, sentences, B, groups)
    return order_for_bookend(S)

def bookend_linearize(S):
    # exploit position bias: head & tail anchors, then zig-zag
    S = sorted(S, key=lambda x: x.weight, reverse=True)
    head, tail = [], []
    for i, s in enumerate(S):
        (head if i % 2 == 0 else tail).append(s)
    return head + list(reversed(tail))
```

## Next Actions (strict order)

1. Implement `sentence_prune`, `knapsack_pack`, and `bookend_linearize` with schemas, guards, and tests; wire into `orchestrate`.
2. Add structure-aware chunker (AST/logs) and metadata boosts; expose flags; keep parity defaults.
3. Train/pickle dynamic fusion regressor; clamp; log features and predictions; add fallback.
4. Extend experiments/grids for V1–V5; run dev → lock → test; compute bootstrap CIs with FDR; enforce gates.
5. If gates pass, enable ANN quantization/tuning and (optionally) active retrieval; re-run gating; rebuild paper & artifact.
