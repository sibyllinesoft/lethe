**TL;DR:** Treat Lethe as a research subject: freeze the code, build LetheBench (3 dialog genres), run a factorial grid (retrieval + planning knobs) against strong local baselines with strict CIs and preregistered hypotheses, then auto-generate a NeurIPS-ready LaTeX paper + artifact from the measured tables/figures—no “hand-curation,” all scripted.

---

### Assumptions (explicit)

* Claude Code has already produced a monorepo whose CLIs match the spec you sent to him; where commands differ, this plan treats Claude’s output as *mutable* to conform.
* No external services beyond localhost Ollama. All datasets must be license-clean and locally mirrored.
* Target venue: NeurIPS main track. Paper limit: 9 pages (excluding references/appendix). Reproducibility checklist required.

---

### What we’re trying to prove (hypotheses → ablations)

**H1 (quality):** Hybrid BM25+vector + rerank + diversification + HyDE beats strong local baselines on retrieval quality (nDCG\@10, Recall\@50) across dialog genres.
**H2 (efficiency):** Lethe attains <3 s median end-to-end latency and <1.5 GB RSS on 50k-token sessions with transformers.js embeddings on CPU.
**H3 (robustness):** Lethe’s submodular diversification measurably increases entity coverage and reduces contradiction-rate vs. top-K.
**H4 (adaptivity):** Plan heuristics (explore/verify/exploit) reduce hallucinated claims in summarization vs. static settings.

**Ablations:**
− reranker, − diversification, − HyDE, − vector, − BM25, embeddings alt, chunk size/overlap, vector backend (SQLite-vec vs WASM HNSW), plan fixed vs adaptive.

---

### Datasets (LetheBench; fully scripted build)

Three genres to stress different failure modes; all public, local, and license-safe.

1. **Code-centric dialogs (LetheBench-Code).**
   Source candidates: GitHub Issues/PR threads (code snippets + discussion), Stack Overflow multi-turn Q\&A, doc-comment threads.
   Construction: mine threads; split messages; identify “gold chunks” by weak labels: code symbols, function names, file paths present in the final accepted answer or merged commit. Human audit 200 items.

2. **Tool-result dialogs (LetheBench-Tool).**
   Source candidates: CLI tutorial transcripts, notebook execution narratives, benchmark logs.
   Construction: detect “tool outputs” (regex blocks, tables); gold = the minimal set of prior outputs needed to answer the final query (heuristics + auditor spot-check).

3. **General chat + long context (LetheBench-Prose).**
   Source candidates: meeting transcripts (e.g., QMSum-like structure), public hearing transcripts, Wikis + threaded comments.
   Construction: create sessions of 10–200 turns; gold = supporting spans for factual questions at later turns (alignment via entity/time overlap + auditor check).

**Splits:** 60/20/20 train/dev/test per genre. **Scale:** ≥1,000 sessions/genre; ≥10k chunks/genre.
**Licensing log:** store URLs + license in `datasets/manifest.csv`.
**Privacy pass:** filter emails/API keys with deterministic regex redaction; publish redaction script.

---

### Baselines (must be competitive, all local)

* **Window (recency) baseline:** last-N tokens naïve packer.
* **BM25-only:** tuned k1,b; DF/IDF per session.
* **Vector-only:** bge-small embeddings; ANN identical to Lethe’s backend.
* **BM25 + vector (no rerank/diversify):** hybrid αβ; no cross-encoder; top-K.
* **Cross-encoder rerank over BM25-only:** “lexical→rerank” classic pipeline.
* **FAISS IVF-Flat local RAG:** independent implementation to sanity-check the vector backend claim.
* **MMR diversification:** substitute for submodular pick (to validate the coverage mechanism).

*All baselines share the same chunker and embeddings where applicable to isolate effects.*

---

### Metrics (precise; preregistered)

**Retrieval** (computed against gold chunk IDs or spans):

* nDCG\@k (k∈{10,20}), Recall\@k (k∈{10,50}), MRR\@10.
* Coverage\@N: unique gold entities covered by selected pack chunks.
* Contradiction-rate: fraction of packs whose selected chunks contain flagged contradictions w\.r.t. the gold claim set (regex+rule detector).

**System** (end-to-end, per query):

* Latency breakdown (ms): embed, lex, ann, rerank, diversify, summarize, total.
* Peak RSS (MB), CPU %, GPU present?, model cold/warm status.
* Model load time (first-run) vs warm runs.

**Summarization fidelity** (optional if HyDE/summarize used):

* Claim precision (strict citation check): % claims whose cited chunk actually contains the asserted fact (string/entailment lite).
* Hallucination rate: claims with no supporting chunk above threshold similarity.

**Statistics:** 10k bootstrap for CI on per-session metrics; paired permutation tests for method comparisons; Holm-Bonferroni across families (per metric × genre). Report effect sizes (Cliff’s delta).

---

### Hardware & environment

* **CPU node:** 8C/16T x86\_64, 32GB RAM, no dGPU.
* **GPU node (optional):** single consumer GPU with WebGPU enabled.
* **OS:** Ubuntu 22.04 LTS; Node ≥20; SQLite 3.45+; Ollama local (model xgen-small:4b); transformers.js latest.
* Fix random seeds; pin npm lockfile; record extension status (sqlite-vec present?).

---

### Experimental design (factorial; bounded)

**Grid:**

* α ∈ {0.7, 1.0, 1.3}, β ∈ {0.5, 0.9, 1.2}.
* Chunk target ∈ {256, 320, 384}, overlap ∈ {32, 64}.
* Rerank topk\_in ∈ {50, 100}, topk\_out ∈ {20, 50}.
* Diversify pack\_chunks ∈ {12, 24, 36}.
* Plan ∈ {adaptive, fixed-explore, fixed-exploit}.
* HyDE ∈ {off, on (k per plan)}.
* Backend ∈ {sqlite-vec, wasm-hnsw}.

Run full grid on dev set for tuning; lock best per genre under latency ≤ 3 s median; apply *once* to test set. All ablations use the locked config except the toggled component.

---

### Automation & orchestration (what Claude must do, step-by-step)

1. **Freeze Lethe.**

   * Tag commit `research-freeze`. Emit `lethe_version.json` (commit, deps, extension status).
   * `ctx-run diagnose` must pass; if missing, implement a minimal health check command.

2. **Dataset builder.**

   * Implement `datasets/build.py` to: crawl sources, normalize to `{session_id, turn, role, text, ts, meta}`, redact secrets, derive gold annotations (weak labels + optional human CSV).
   * Write `datasets/README.md` with exact sources and license notes.
   * Produce `datasets/{code,tool,prose}/{train,dev,test}.jsonl`.

3. **Ingestion & indexing.**

   * For each split, run:

     ```
     npx ctx-run init workdir
     npx ctx-run ingest --session <sid> --from file <jsonl>
     npx ctx-run index --session <sid>
     ```
   * Cache model weights in `models/`; record cold vs warm run.

4. **Baseline runners.**

   * Implement `scripts/run_baseline.sh` wrappers that set config bits (e.g., disable reranker/diversify).
   * Ensure identical chunker across runs.

5. **Experiment controller.**

   * Implement `experiments/run.py` that:
     a) Reads grid from `experiments/config.yaml`.
     b) For each config, calls `ctx-run query` for each query in dev/test with `--debug`, streaming JSON lines.
     c) Captures timings + memory via `/usr/bin/time -v` or Node process metrics.
     d) Stores raw outputs under `artifacts/<exp_id>/raw/*.jsonl`.

6. **Scoring & stats.**

   * Implement `experiments/score.py` to compute metrics per session; bootstrap CIs; permutation tests; effect sizes.
   * Emit `metrics_summary.csv`, `per_session.csv`, and `stats.json` (with p-values, CIs).

7. **Figure & table generation.**

   * `experiments/plots.py` produces:

     * Quality–latency Pareto curves per genre.
     * Bar charts of ablations (ΔnDCG, ΔRecall, ΔCoverage).
     * Breakdown of latency by stage.
     * Entity coverage vs pack budget.
     * Contradiction-rate vs diversification on/off.
     * Scaling curves (latency vs session length; quality vs chunk target).
   * Save as vector graphics (`.pdf`/`.svg`) under `paper/figures/`.

8. **Sanity checks (fraud-proofing).**

   * Recompute metrics with randomized gold to ensure low scores (placebo).
   * Shuffle queries to ensure latency distribution unchanged (guard against caching artifacts).
   * Swap embeddings model with random vectors to ensure quality collapses (guard against silent leaks).
   * Emit a “fails-when-broken” report.

9. **Repro pack.**

   * Include `Makefile` or `justfile` targets: `make datasets`, `make dev-grid`, `make test-locked`, `make paper`, `make artifact`.
   * Write `ARTIFACT.md` with one-command repro (`./run_all.sh`) and total runtime estimate.

10. **Paper generation.**

* Use official NeurIPS LaTeX template.
* Auto-populate tables from `metrics_summary.csv`; include CIs and boldface bests when statistically significant.
* Insert figures from `paper/figures/`.
* Generate appendices: dataset construction details; prompts; full configs; extended ablations; failure cases with examples.

11. **Artifact evaluation bundle.**

* Zip: code (frozen), datasets (or scripts + hashes), configs, raw outputs, metrics, paper PDF, and a Dockerfile/conda env.

---

### Paper skeleton (Claude must fill programmatically from artifacts)

**Title:** Lethe: Local-First Conversational Context Packing with Hybrid Retrieval and Adaptive Planning
**Abstract:** Data-driven; 150–200 words; no claims without numbers.
**1. Introduction:** Motivation; contributions (bulleted; numbered; testable).
**2. Related Work:** RAG, conversational memory, HyDE, cross-encoders, diversification (MMR vs submodular), local-first privacy systems.
**3. Method:** Chunking; per-session DF/IDF; hybrid scoring (equations below); rerank; submodular pick (greedy coverage pseudocode); plan heuristics.
**4. LetheBench:** Dataset design, construction pipeline, stats, licensing, biases.
**5. Experimental Setup:** Hardware, software, baselines, metrics, preregistered grid, evaluation protocol.
**6. Results:** Main tables (per genre), Pareto curves, ablations, robustness, adaptivity; effect sizes + CIs.
**7. Analysis:** Error taxonomy; where diversification helps/hurts; when reranker fails; backend sensitivity (sqlite-vec vs WASM).
**8. Limitations & Ethics:** Data bias, privacy, compute budget, offline constraints.
**9. Conclusion:** What generalizes, what’s next.
**Reproducibility Checklist:** Filled from artifacts.
**Appendix:** Full configs, prompts, extra plots, dataset examples.

**Equations (annotated, plain-English):**

* IDF: $\text{idf}(t) = \log \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}$ (lower-bounded at 0).
* BM25 (k1=1.2, b=0.75); normalize by max over candidates to \[0,1].
* Cosine: normalized to \[0,1] as $(\cos +1)/2$.
* Hybrid: $s = \alpha \cdot \text{bm25}_{\text{norm}} + \beta \cdot \cos_{\text{norm}} + \gamma(\text{kind})$.
* Submodular pick objective: greedy maximize marginal entity coverage under chunk budget.

---

### Prompts Claude should use (deterministic, JSON-first, no prose drift)

* **Results to Table (system prompt):** “You will write LaTeX tables from CSV. Never invent numbers. Round to 3 decimals. Add boldface only if `stats.json` shows `p < 0.05` vs second best.”
* **Figure captions:** “One-sentence, specific, reference the metric, dataset, and sample size. No marketing.”
* **Error analysis extraction:** “From per\_session.csv, sample 10 worst deltas. For each, cite query, top-missed gold entity, and which ablation would have fixed it.”

---

### XML workflow for Claude (end-to-end, fully specified)

```xml
<workflow name="lethe_paper" version="1.0">
  <stage id="freeze">
    <task>Tag repo at research-freeze; export lethe_version.json (git SHA, deps, sqlite-vec present?).</task>
  </stage>

  <stage id="datasets">
    <task>Run datasets/build.py to generate LetheBench-{Code,Tool,Prose}/{train,dev,test}.jsonl with manifest.csv and LICENSE notes.</task>
    <task>Validate sizes, splits, and redaction coverage; emit datasets/QA_report.md.</task>
  </stage>

  <stage id="index">
    <task>For each split: ctx-run init; ingest; index; record timings (cold vs warm).</task>
    <task>Persist lock.json and environment snapshot per split.</task>
  </stage>

  <stage id="dev_grid">
    <task>Run experiments/run.py over grid on dev; store raw outputs in artifacts/dev-grid/*.</task>
    <task>Score with experiments/score.py; select best config per genre under latency constraint; write tuned_config.json.</task>
  </stage>

  <stage id="test_eval">
    <task>Apply tuned_config.json to test; run baselines and ablations; store artifacts/test/*.</task>
    <task>Compute metrics, bootstrap CIs, permutation tests; emit metrics_summary.csv, per_session.csv, stats.json.</task>
    <task>Run sanity checks (placebo gold, random vectors); emit fraudproof.md.</task>
  </stage>

  <stage id="viz">
    <task>Generate plots (Pareto, ablations, latency breakdown, coverage curves, scaling curves) into paper/figures/.</task>
  </stage>

  <stage id="paper">
    <task>Fill NeurIPS LaTeX template with auto-generated tables/figures/captions; compile to PDF.</task>
    <task>Generate appendix with full configs, prompts, extended results, and error cases.</task>
    <task>Fill Reproducibility Checklist from artifacts.</task>
  </stage>

  <stage id="artifact">
    <task>Bundle code (frozen), datasets or builders, configs, raw outputs, metrics, paper PDF, Dockerfile into artifact.zip.</task>
    <task>Write ARTIFACT.md with one-command repro and expected totals (time, disk).</task>
  </stage>

  <quality_gates>
    <gate>All tests show p<0.05 improvements for main method vs best baseline on at least 2/3 genres for nDCG@10.</gate>
    <gate>Median E2E latency <3s and RSS <1.5GB on CPU node (test set).</gate>
    <gate>Fraudproof checks fail where they should (quality collapses under random vectors).</gate>
    <gate>Paper builds cleanly; all numbers trace to CSV with matching hashes.</gate>
  </quality_gates>
</workflow>
```

---

### Trade-offs & pitfalls (so Claude doesn’t step in them)

* **Dataset leakage:** Ensure no query text appears in gold beyond historical turns; freeze sessions strictly at turn-t.
* **Timing noise:** Pin CPU governor; run three passes; report medians with MAD.
* **Ollama variance:** Hard timeouts; log “HyDE skipped” count; analyze separately.
* **Over-tuning:** All tuning on dev only; *never* revisit dev after seeing test.
* **Citations in packs:** Validate claim support with literal span match + semantic fallback; otherwise mark as hallucinated.

---

### What “best paper” looks like here

* Clear novelty: *local-first*, *per-session DF/IDF*, *adaptive planning*, and *diversification for dialog*—then *prove* each with an ablation that moves the metric.
* Clean system story: quality-latency Pareto beats baselines; method reasons about coverage and contradictions.
* Industrial-grade artifact: one-command repro; strong sanity checks; honest limitations.

