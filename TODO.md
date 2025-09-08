**TL;DR:** Now that Gap→Tune→Verify is live, run rolling “tuning campaigns” that (1) pick the highest-ROI failure slices, (2) sweep a constrained knob grid with counterfactual priors, (3) validate on paired matrices, and (4) auto-promote only if buyer-grade gates pass; the same runs feed the microsite fronts.

**Idea → prioritization.** Use a single priority score to choose what to tune next, built from the artifacts you’re already logging:

$$
\text{score} = \Big(\frac{\max(0,\Delta P@5)}{\text{CI\_width}}\Big)^2 \cdot S \cdot T\;-\;\rho\cdot R
$$

Where ΔP\@5 is (competitor−Lethe) on the *paired* slice; CI\_width is the paired bootstrap 95% width; $S$ is counterfactual sensitivity (e.g., $\partial P/\partial K2$, $\partial P/\partial \lambda$ estimated from IPS replays); $T$ is tenant/traffic weight; $R$ is risk (KV-prefix drop, ECE drift, p99/p95 inflation); $\rho$ is a fixed penalty weight. That moves attention to statistically real gaps where small knob moves have leverage and operational risk is low.

**Mechanism → campaigns.** Each “campaign” targets 1–2 slices for one budget tier (8/15/30) and runs 12–18 trials with BO+rules inside validator fences (ECE≤0.08, p95≥avg, p99/p95≤2.5, proxy-gap≤0.5%, KV-prefix drop≤3pp). Initialize from counterfactual frontiers, then replay M≈200 paired samples for gating before full-matrix promotion. Pseudocode:

```
for slice in TopKBy(score):
  Θ0 ← difficulty-gate(GBM)          # {256|768}, K2 cap, CE cap
  G  ← validator_fences ∩ safe_knobs  # λ, μ, K2, r, head_keep, W, s, τ
  Θ* ← BO_with_penalties(Θ0, G, objective=ΔP@5 per ms, KV_penalty)
  if paired_full_matrix(Θ*) passes all gates & union_non_degradation:
    promote(Θ*); microsite.annotate("Tuned-vX (Validated)")
```

**Trade-offs → guardrails.** Counterfactual uplift lies if calibration drifts; keep coverage-weighted CRPS checks per type and re-isotonic before trusting IPS deltas. Streaming tail changes (W,s,sinks) can make latency fronts look great while quietly eroding KV reuse; include KV-prefix Jaccard directly in the BO penalty. Raising DPP rank r helps near-dup storms but costs O(r²); gate r increases on measured curvature spikes only. Group-split τ helps deep symbol chains but can increase ILP incidence; cap τ moves to ±0.1 and alert if ILP\_used>10%.

**Next steps → two-week plan (concrete).**
Week 1 (fast wins, low risk):
• **Zh.QA @ 8%** (code-switch fragility). Grid: re-isotonic; CE early-exit cap +20%; $K2:+25\%$; $r=16$; $\lambda:+5\%$; hold head\_keep. Pass if ΔP\@5≥+1.5pp with CI>0, p95∆≤+1ms, KV drop≤1pp.
• **JSON/PassKey @ 15%** (fact needles). Grid: CE early-exit **off** for CE\@k≤50; $K2:+25\%$; $\lambda:+5\%$; head\_micro-summaries ON; $\gamma:+0.1$, $\delta:-0.05$. Same gates; additionally require ECE×FACT bin ≤0.06.

Week 2 (harder, higher ROI):
• **Code.Debug @ 15%** (long closures). Grid: stronger closures ON; head\_keep +2–3pp; $K2:+15\%$; $r=16$; τ=0.75; $\lambda:+5\%$. Gate on ILP\_used≤10% and zero closure breaks.
• **Retrieve.KV @ 30%** (KV stability under bigger budgets). Grid: maintain head anchor; shrink W or stride before touching head; sinks=64–96; $\mu:+5\%$. Require KV prefix-reuse ≥ baseline and p99/p95≤2.0.

Promotion rule (all campaigns): paired, budget-matched full matrix; Holm-corrected significance; **union non-degradation** across all datasets at 8/15/30; microsite auto-annotates fronts and links raw JSONL deltas. Assumptions: current datasets and competitor lineup unchanged; traffic weights $T$ set from your live mix. When these four campaigns land, flip the GBM difficulty gate from “init-only” to “per-turn routing,” then revisit grouped-DPP centroiding as the next algorithmic bet.
