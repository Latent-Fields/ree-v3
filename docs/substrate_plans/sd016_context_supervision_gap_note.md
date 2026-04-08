# SD-016 Context Supervision — Gap Note
**Gap ID:** SD-016-context
**Readiness:** architecture_unclear (mathematical analysis required)
**Blocking claims:** SD-016, ARC-041, ARC-042, MECH-153

---

## Current State

SD-016 claim: E3's selection circuits require supervised context-labeling training on
E1-ContextMemory representations to learn empirical ethics.

EXQ-239 FAIL (2026-04-05): `cosine_sim = 1.0` in BOTH supervised and ablated conditions.
This means context_memory representations are identical whether supervised terrain loss
is applied or not. The supervised head learns terrain classification but the latent
representations don't differentiate between harm-predictive and safe contexts.

Root cause diagnosis (DR-8, design_forward_analysis_2026-04-06.md):
"Context memory needs first-principles mathematical analysis before redesign. The
pipeline's expected mathematical functions are unknown -- we need to understand why
the latent collapses to cosine_sim=1.0 in both conditions before we can redesign."

---

## What Is Unclear

The EXQ-239 finding has multiple possible explanations that require different fixes:

**Explanation 1: Gradient vanishing in context memory writes**
The supervised terrain loss may not be backpropagating through the ContextMemory write
mechanism. If `context_memory.write()` uses a stop-gradient or detach(), the supervised
signal never reaches the E1 encoder weights. Fix: verify gradient flow with hooks.

**Explanation 2: ContextMemory slots collapse to the same representation**
If all 16 ContextMemory slots converge to nearly identical vectors (common in
attention-based memories), cosine_sim between any two query outputs approaches 1.0
regardless of context. Fix: add slot diversity regularization (e.g., orthogonality loss
on slot vectors).

**Explanation 3: Terrain classification is too easy for all contexts**
If the E1 prior already encodes enough terrain information that the supervised head
achieves high accuracy without differentiating latents, the loss signal is near zero.
The latents don't need to change because the classification is solved by memorization.
Fix: redesign the classification target to require genuinely distinct latent structure.

**Explanation 4: Cosine_sim is the wrong diagnostic metric**
If context_memory outputs are always normalized to unit sphere, cosine_sim=1.0 is
trivially satisfied without meaning representations are identical.
Fix: use Euclidean distance or information-theoretic metric instead.

---

## What Would Unblock This

1. **Gradient flow audit**: trace the computation graph from `supervised_terrain_loss`
   back through `context_memory.write()` to the E1 encoder parameters. Identify
   any stop-gradient / detach() calls that break the supervision chain.

2. **Slot vector diagnostics**: log `context_memory.memory.data` every 100 steps.
   Compute pairwise cosine_sim between slots. If all pairs ~1.0, explanation 2 is
   confirmed.

3. **Ablation confirmation**: run a 1-step diagnostic where the supervised loss weight
   is set very high (e.g., 10.0). If cosine_sim still stays at 1.0, the gradient is
   definitely not flowing.

4. **Metric correction**: switch diagnostic from cosine_sim to L2 distance between
   context vectors for harm-predictive vs. safe states.

This is an analysis task (1-2 days of debugging) before any code changes are warranted.
The implementation plan for SD-016 cannot be written until the mathematical failure
mode is understood.

---

## Suggested Next Step

Assign a single debugging session:
1. Enable gradient hooks on ContextMemory parameters.
2. Run EXQ-239 with `supervised_terrain_weight=5.0`.
3. Log: (a) gradient norm reaching E1 encoder, (b) slot pairwise distances,
   (c) terrain classification accuracy.
4. Based on findings, determine which explanation (1-4) applies and write the fix.

This is a diagnose-errors task, not a substrate implementation task.
