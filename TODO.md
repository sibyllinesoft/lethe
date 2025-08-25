# Lethe Paper Reinforcement — `todo.md`

**TL;DR:** Expand LetheBench, deepen experiments (ablations/scaling/error analysis), enforce hermetic reproducibility + verification batteries, regenerate a statistically rigorous NeurIPS-grade paper and artifact.

## Invariants (do not change)

* Local-first only; no external network calls beyond localhost (e.g., Ollama).
* Budget parity: retrieval pipeline variants stay within ±5% params/FLOPs unless declared under “systems budget”.
* Primary metric: nDCG\@10 (paired, per-query), plus Recall\@50, Coverage\@N, latency p50/p95, RSS.
* Oracles are source-of-truth; contracts & properties enforced at runtime.
* Hermetic spin-up required; boot transcript must verify.

## Assumptions & Scope

* **Assumption:** Repo roots exist at `{{ROOT}}/ctx-run` (tool) and `{{ROOT}}/lethe-research` (bench/paper); adjust if different.
* **Assumption:** Local LLM available at `http://127.0.0.1:11434` (or disabled with feature flag).
* **Assumption:** Thresholds if unspecified: **T\_mut=0.80**, **T\_prop=0.70**, **SAST\_high=0**, latency budgets **p50≤3s**, **p95≤6s**, **RSS≤1.5GB** on CPU node (8C/16T, 32GB).
* **Scope:** Improve paper and artifact via dataset scaling, rigorous eval, correctness oracles, reproducibility; optional feature work must be gated behind flags.

## Objectives

1. **Dataset scale:** Expand LetheBench to **≥600 queries** across **3 domains** with full stats, split 60/20/20; emit manifest + hashes.
2. **Quality delta:** Demonstrate **≥+10%** nDCG\@10 vs best strong baseline with **BCa 95% CI lower bound > 0** (paired bootstrap); show on **Full** and **Focused** slices.
3. **Verification:** Achieve mutation ≥ **T\_mut** and property/metamorphic coverage ≥ **T\_prop**.
4. **Spin-up:** One-shot hermetic boot from clean checkout; golden smoke flows pass; produce signed boot transcript with env digest.
5. **Systems budget:** Keep **p50≤3s**, **p95≤6s**, **RSS≤1.5GB** on CPU; report breakdowns and uphold budget parity.

## Risks & Mitigations

* Thin dataset/generalization → **Mitigation:** Scale queries, add slice stats, inter-annotator checks, publish manifest with hashes.
* LLM JSON brittleness/timeouts → **Mitigation:** Schema validators, timeouts, fallbacks, placebo runs, invariant checks.
* Non-determinism → **Mitigation:** Pinned toolchain/locks, seeds, container image, recorded SHAs.
* Over-claiming significance → **Mitigation:** Paired bootstrap, FDR control, effect sizes, pre-registered promotion gates.
* OSS/license drift → **Mitigation:** License scan + manifest; block on non-compliant assets.

## Method Outline (idea → mechanism → trade-offs → go/no-go)

### Workstream A — Dataset Scale & Provenance

* **Idea:** Expand LetheBench with clean licenses and strong metadata.
* **Mechanism:** Deterministic builders; redactors; per-query gold; manifest of SHAs; IAA on a subset.
* **Trade-offs:** Build time; curation overhead.
* **Go/No-Go:** ≥600 queries, 3 domains; manifest+hashes; IAA ≥0.7 κ on audited subset.

### Workstream B — Experimental Battery

* **Idea:** Comprehensive ablations/scaling/error analysis.
* **Mechanism:** Grid/tuning on dev; lock best; test-only once; produce per-query CSV; CI-backed wins.
* **Trade-offs:** Compute time; result management.
* **Go/No-Go:** CI lower bound > 0 for primary delta; latency/RSS budgets met.

### Workstream C — Oracles & Invariants

* **Idea:** Formalize correctness: JSON schema, citation integrity, metamorphic properties.
* **Mechanism:** Contracts, property-based tests, metamorphic suites, runtime guards.
* **Trade-offs:** Strictness may surface refactors.
* **Go/No-Go:** Property coverage ≥ T\_prop; 0 invariant breaks on 10k shadow replays.

### Workstream D — Adversarial Verification

* **Idea:** Kill weak tests and explore edges.
* **Mechanism:** Mutation testing, grammar/coverage-guided fuzzing, concolic on parsers and JSON handlers.
* **Trade-offs:** CI wall clock.
* **Go/No-Go:** Mutation ≥ T\_mut; 0 high/critical SAST.

### Workstream E — Paper & Artifact Hardening

* **Idea:** Regenerate LaTeX from artifacts; no dangling numbers; figures/tables are programmatic.
* **Mechanism:** Table/figure generators; cross-ref checks; PDF build gate; artifact bundle with hashes.
* **Trade-offs:** Template rigidity.
* **Go/No-Go:** Paper builds cleanly; all numbers trace to CSV+hash; reproducibility script passes.

## Run Matrix

| ID | Method/Variant                                   | Budget                     | Inputs                                  | Expected Gain                  | Promote if…                                 |
| -- | ------------------------------------------------ | -------------------------- | --------------------------------------- | ------------------------------ | ------------------------------------------- |
| V0 | Strong Baselines (window/BM25/vector/hybrid etc) | ±5% parity                 | Locked configs, same chunker/embeddings | Reference floor/ceiling        | All contracts pass; budgets met             |
| V1 | Cheap Wins (metadata boosts + semantic divers.)  | Systems budget only        | Same as V0                              | +Coverage, +nDCG small         | +Coverage\@N ≥ +20% (dev), p50≤3s           |
| V2 | Query Rewrite+Decompose                          | Systems budget only        | Rewriter+decomposer enabled             | +Recall on complex/elliptical  | +Recall\@50 ≥ +10% (Prose/Tool), p50≤3.5s   |
| V3 | Dynamic Fusion + Learned Planning                | ±5% (feature-only compute) | Regressor+plan classifier               | +nDCG especially in Code       | +nDCG\@10 ≥ +5% (Code), contradictions −10% |
| V4 | LLM Rerank + Contradiction-aware                 | Declared LLM budget        | LLM reranker/penalty                    | +nDCG modest; −hallucinations  | +nDCG\@10 ≥ +3%, hallucination −15%, p50≤4s |
| V5 | Chunking Variants (AST/hier/propositional)       | ±5% parity                 | Alt chunkers                            | Precision ↑ on targeted slices | Best beats V0 with CI>0; budgets met        |

## Implementation Notes

* **Attach points:** `packages/core/{retrieval.ts,rank_diversify.ts,chunker.ts,orchestrate.ts}`, `packages/reranker/`, `packages/sqlite/`.
* **Config flags:** `retrieval.alpha/beta`, `diversify.method={'entity','semantic'}`, `plan.query_rewrite`, `plan.query_decompose`, `fusion.dynamic`, `plan.learned`, `rerank.use_llm`, `contradiction.enabled`.
* **Schemas:** Pack JSON schema; citation map must reference selected chunk IDs; spans within bounds; schema validator runs in CI and runtime guard.
* **Metamorphic properties:**

  * Adding irrelevant chunks must **not** increase precision at fixed K.
  * Duplicating a selected chunk **must not** change metrics.
  * Query synonym/lemmatization invariance within ε on nDCG.
  * Shuffling non-selected chunks **must not** change outputs.
  * Removing a supporting chunk **must** reduce claim support score.
* **Telemetry:** JSONL per-query with config hash, seeds, timings, RSS, metrics; dataset/index hashes.
* **Seeds/Hashes:** Record model SHA (or version), dataset SHA set, ANN index hash; embed in artifacts.

## Acceptance Gates

* **Spin-up:** Clean checkout → container build → migrate/seed → readiness OK → golden smokes pass → **boot transcript signed** with env digest.
* **Static:** Types/linters clean; **0 high/critical SAST**; license policy OK; API surface diff acknowledged.
* **Dynamic:** Mutation ≥ **T\_mut**; property/metamorphic coverage ≥ **T\_prop**; fuzzing ≥ 30 mins with **0** new medium+ crashes; concolic finds **0** feasible high-risk paths.
* **Stats:** 10k paired bootstrap; **BCa 95% CI lower bound > 0** for primary delta; FDR controlled within metric families.
* **Systems:** p50≤3s, p95≤6s, RSS≤1.5GB (CPU); LLM variants p50≤4s with explicit budget.
* **Dataset:** ≥600 queries, 3 domains, manifests+hashes present; IAA κ≥0.7 on audited subset.
* **Paper:** Builds cleanly; tables/figures generated from artifacts; cross-refs resolve; no dangling numbers.

## “Make-sure-you” Checklist

* Pin toolchains and lockfiles; store env manifest and container digest.
* Quarantine network; stub external deps; only localhost permitted.
* Validate all JSON outputs against schemas; fail fast on violations.
* Save boot transcript and all artifact hashes; include in bundle.
* Rerun flake detector 100×; fail if flakiness ≥1%.
* Keep budgets and parity; log any deviation with rationale.
* Promote only on CI-backed wins with gates met.

## File/Layout Plan

```
{{ROOT}}/
  lethe-research/
    datasets/
      builders/          # deterministic scripts, redactors
      manifests/         # csv+json with SHAs, licenses
    experiments/
      grids/
      run.py
      score.py
      plots.py
      sanity.py
    verification/
      schemas/           # JSON schemas for packs/results
      properties/        # property tests + metamorphic
      mutation/          # operators + harness
      fuzz/              # grammars + corpora
      concolic/          # targets for parsers/json
      invariants/        # runtime guards & monitors
    paper/
      figures/
      tables/
      template/          # neurips latex
      build.sh
    scripts/
      spinup_smoke.sh
      bundle_artifact.sh
      compute_risk.py
    artifacts/
      boot_transcript.json
      metrics/
      logs/
  ctx-run/
    packages/            # attach points per notes
```

## Workflows (required)

```xml
<workflows project="lethe_paper_reinforcement" version="1.0">

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
        <item>Lockfiles present (package-lock.json/pnpm-lock.yaml, requirements.txt hashes)</item>
        <item>Env manifest saved with tool versions and SHAs</item>
      </make_sure>
    </env>

    <assets id="B1">
      <desc>Prepare datasets/models; record hashes</desc>
      <commands>
        <cmd>python lethe-research/datasets/builders/build_all.py --out lethe-research/datasets --manifest lethe-research/datasets/manifests/manifest.json</cmd>
        <cmd>python lethe-research/scripts/hash_tree.py lethe-research/datasets &gt; lethe-research/datasets/manifests/datasets.sha</cmd>
        <cmd>node ctx-run/packages/cli/dist/index.js init .ctx &amp;&amp; echo ok</cmd>
      </commands>
      <make_sure>
        <item>≥600 queries across 3 domains in manifests</item>
        <item>License scan passes; non-compliant assets blocked</item>
      </make_sure>
    </assets>

    <contracts id="B2">
      <desc>Compile oracles (schemas, contracts, properties)</desc>
      <commands>
        <cmd>python lethe-research/verification/generate_schemas.py</cmd>
        <cmd>python lethe-research/verification/generate_properties.py</cmd>
        <cmd>python lethe-research/verification/inject_runtime_guards.py --attach ctx-run/packages/core</cmd>
      </commands>
      <make_sure>
        <item>Schema validators green on sample outputs</item>
        <item>Runtime guards wired (citation integrity, span bounds, JSON shape)</item>
      </make_sure>
    </contracts>

    <static id="B3">
      <desc>Static guardrails</desc>
      <commands>
        <cmd>npm run -w ctx-run lint &amp;&amp; npm run -w ctx-run typecheck</cmd>
        <cmd>python lethe-research/infra/run_sast.py --repo . --out lethe-research/artifacts/sast.json</cmd>
        <cmd>python lethe-research/infra/api_diff.py --repo ctx-run --out lethe-research/artifacts/api_diff.json</cmd>
      </commands>
      <make_sure>
        <item>0 high/critical SAST findings</item>
        <item>API surface diffs acknowledged</item>
      </make_sure>
    </static>

    <spinup id="B4">
      <desc>Hermetic boot + smokes</desc>
      <commands>
        <cmd>bash lethe-research/scripts/spinup_smoke.sh</cmd>
        <cmd>python lethe-research/experiments/sanity.py --smoke --out lethe-research/artifacts/smoke.json</cmd>
        <cmd>python lethe-research/scripts/sign_transcript.py --env lethe-research/artifacts/env_manifest.json --smoke lethe-research/artifacts/smoke.json --out lethe-research/artifacts/boot_transcript.json</cmd>
      </commands>
      <make_sure>
        <item>Golden smoke: ingest-index-query on sample session passes and matches golden hashes</item>
        <item>Boot transcript signed with env digest</item>
      </make_sure>
    </spinup>
  </workflow>

  <workflow name="running">
    <baseline id="R0">
      <desc>Run strong baselines under parity</desc>
      <commands>
        <cmd>python lethe-research/experiments/run.py --exp baselines --grid lethe-research/experiments/grids/baselines.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp baselines --split dev --out lethe-research/artifacts/baselines/dev</cmd>
      </commands>
      <make_sure>
        <item>Identical chunker/embeddings across baselines</item>
        <item>Budgets respected (±5%)</item>
      </make_sure>
    </baseline>

    <properties id="R1">
      <desc>Property & metamorphic suites</desc>
      <commands>
        <cmd>pytest lethe-research/verification/properties -q --maxfail=1 --disable-warnings</cmd>
        <cmd>pytest lethe-research/verification/metamorphic -q</cmd>
      </commands>
      <make_sure>
        <item>Property/metamorphic coverage ≥ {{T_prop}}</item>
      </make_sure>
    </properties>

    <fuzz_symbolic id="R2">
      <desc>Fuzzing + concolic on parsers/JSON handlers</desc>
      <commands>
        <cmd>python lethe-research/verification/fuzz/run_fuzz.py --minutes 30 --out lethe-research/artifacts/fuzz</cmd>
        <cmd>python lethe-research/verification/concolic/run_concolic.py --targets pack_json_parser,json_sanitizer --out lethe-research/artifacts/concolic</cmd>
      </commands>
      <make_sure>
        <item>No new medium+ crashes; repros archived if any</item>
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
        <item>Mutation score ≥ {{T_mut}}</item>
      </make_sure>
    </mutation>

    <experiments id="R4">
      <desc>Ablations, scaling, error analysis</desc>
      <commands>
        <cmd>python lethe-research/experiments/run.py --exp iter1 --grid lethe-research/experiments/grids/iter1.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter1 --split dev</cmd>
        <cmd>python lethe-research/experiments/select_best.py --from iter1 --rule "max:nDCG@10 && p50&lt;=3000"</cmd>
        <cmd>python lethe-research/experiments/run.py --exp iter1 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter1 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp iter2 --grid lethe-research/experiments/grids/iter2.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter2 --split dev</cmd>
        <cmd>python lethe-research/experiments/select_best.py --from iter2 --rule "max:Recall@50(Prose,Tool) && p50&lt;=3500"</cmd>
        <cmd>python lethe-research/experiments/run.py --exp iter2 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter2 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp iter3 --grid lethe-research/experiments/grids/iter3.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter3 --split dev</cmd>
        <cmd>python lethe-research/experiments/select_best.py --from iter3 --rule "max:nDCG@10(Code) && p50&lt;=3500"</cmd>
        <cmd>python lethe-research/experiments/run.py --exp iter3 --apply_best --split test</cmd>
        <cmd>python lethe-research/experiments/score.py --exp iter3 --split test</cmd>

        <cmd>python lethe-research/experiments/run.py --exp chunking --grid lethe-research/experiments/grids/chunking.yaml --split dev</cmd>
        <cmd>python lethe-research/experiments/score.py --exp chunking --split dev</cmd>
      </commands>
      <make_sure>
        <item>Per-query CSVs saved; config hashes recorded</item>
        <item>Latency p50/p95 &amp; RSS logged per variant</item>
      </make_sure>
    </experiments>

    <differential id="R5">
      <desc>Golden diffs vs last known-good</desc>
      <commands>
        <cmd>python lethe-research/verification/differential/capture.py --exp best --out lethe-research/artifacts/golden</cmd>
        <cmd>python lethe-research/verification/differential/diff.py --golden lethe-research/artifacts/golden --candidate lethe-research/artifacts/current --out lethe-research/artifacts/diffs</cmd>
      </commands>
      <make_sure>
        <item>No incompatible diffs; schema and contracts green</item>
      </make_sure>
    </differential>

    <runtime id="R6">
      <desc>Shadow traffic replay & invariants</desc>
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
      <desc>Metrics & stats</desc>
      <commands>
        <cmd>python lethe-research/experiments/collect.py --root lethe-research/artifacts --out lethe-research/artifacts/metrics/all.jsonl</cmd>
        <cmd>python lethe-research/experiments/statistics.py --in lethe-research/artifacts/metrics/all.jsonl --bca --bootstrap 10000 --fdr --out lethe-research/artifacts/stats</cmd>
        <cmd>python lethe-research/experiments/plots.py --in lethe-research/artifacts --out lethe-research/paper/figures</cmd>
        <cmd>python lethe-research/experiments/make_tables.py --in lethe-research/artifacts --out lethe-research/paper/tables</cmd>
      </commands>
      <make_sure>
        <item>All figures/tables tagged with source CSV + hash</item>
        <item>Stars shown only when CI lower bound &gt; 0</item>
      </make_sure>
    </harvest>

    <risk id="T2">
      <desc>Risk score & decision features</desc>
      <commands>
        <cmd>python lethe-research/scripts/compute_risk.py --artifacts lethe-research/artifacts --out lethe-research/artifacts/risk.json</cmd>
      </commands>
      <make_sure>
        <item>Features normalized; thresholds recorded</item>
      </make_sure>
    </risk>
  </workflow>

  <workflow name="evaluating">
    <gatekeeper id="E1">
      <desc>Apply gates; decide route</desc>
      <commands>
        <cmd>python lethe-research/scripts/apply_gates.py --gates lethe-research/acceptance.yaml --metrics lethe-research/artifacts/stats/summary.json --out lethe-research/artifacts/decision.json</cmd>
        <cmd>python lethe-research/paper/build.sh</cmd>
        <cmd>bash lethe-research/scripts/bundle_artifact.sh lethe-research/artifacts lethe-research/paper/lethe_neurips2025.pdf</cmd>
      </commands>
      <make_sure>
        <item>No promotion without all gates met and CI-backed deltas</item>
        <item>Paper builds cleanly; no dangling refs</item>
      </make_sure>
    </gatekeeper>
  </workflow>

  <workflow name="refinement">
    <agent_refine id="N1">
      <desc>Auto-iterate on failures</desc>
      <commands>
        <cmd>python lethe-research/scripts/make_actionable_prompt.py --from lethe-research/artifacts/decision.json --out lethe-research/artifacts/agent_prompt.txt</cmd>
        <cmd>bash lethe-research/scripts/agent_cycle.sh --prompt lethe-research/artifacts/agent_prompt.txt</cmd>
      </commands>
      <make_sure>
        <item>Prompt lists concrete obligations, thresholds, and failing gates</item>
      </make_sure>
    </agent_refine>

    <manual_qa id="N2">
      <desc>Human QA route</desc>
      <commands>
        <cmd>python lethe-research/scripts/open_dashboard.py --artifacts lethe-research/artifacts</cmd>
        <cmd>python lethe-research/scripts/prepare_handoff.py --include boot_transcript.json diffs/ repros/ contracts/ --out lethe-research/artifacts/handoff.zip</cmd>
      </commands>
      <make_sure>
        <item>Owner assigned; rollback plan documented</item>
      </make_sure>
    </manual_qa>
  </workflow>

</workflows>
```

## Minimal Pseudocode (optional)

```python
def gatekeeper(metrics, thresholds):
    if not metrics["spinup_pass"]: return "AGENT_REFINE: spin-up"
    if metrics["sast_high"] > 0:   return "AGENT_REFINE: SAST"
    if metrics["mutation"] < thresholds["T_mut"]: return "AGENT_REFINE: mutation"
    if metrics["prop_cov"] < thresholds["T_prop"]: return "AGENT_REFINE: properties"
    delta = metrics["nDCG10_delta"]
    ci_lo = metrics["nDCG10_delta_CI_lo"]
    if not (ci_lo > 0): return "MANUAL_QA"
    if metrics["p50_ms"] > thresholds["p50"] or metrics["p95_ms"] > thresholds["p95"] or metrics["rss_mb"] > thresholds["rss_mb"]:
        return "AGENT_REFINE: budgets"
    return "PROMOTE"
```

## Next Actions (strict order)

1. Build hermetic container, record env manifest, run golden smokes; sign boot transcript.
2. Expand LetheBench to ≥600 queries across 3 domains; emit manifests, hashes, and IAA report.
3. Wire schemas, properties, metamorphic tests, and runtime guards; enforce in CI.
4. Run baselines and Iterations V1–V3 on dev; lock best; run test; compute bootstrap CIs with FDR.
5. Regenerate figures/tables programmatically; rebuild paper; run Gatekeeper and bundle artifact.
