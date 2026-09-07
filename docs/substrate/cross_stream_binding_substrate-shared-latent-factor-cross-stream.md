## cross_stream_binding_substrate: Shared-latent-factor cross-stream binding (2026-07-08)
- cross_stream_binding_substrate: latent.cross_stream_binding -- IMPLEMENTED 2026-07-08.
  Module: ree_core/latent/cross_stream_binder.py (CrossStreamBinder); wired into
  ree_core/predictors/e2_fast.py (E2FastPredictor.__init__ + rollout_with_world).
  Config: E2Config.cross_stream_binding_enabled (default False; set True to enable),
  cross_stream_binding_dim (16), cross_stream_binding_strength (0.15),
  cross_stream_binding_theta_period (4).
  Data flow: joint (z_self_t, z_world_t) -> g_t = tanh(W_enc . [z_self;z_world]) ->
  b_t = W_out . g_t (shared perturbation, bind_out_dim = min(self_dim, world_dim)) ->
  the SAME theta-gated k_t*b_t added into the first bind_out_dim components of BOTH
  post-transition z_self and z_world -> trajectory.states / trajectory.world_states ->
  641a-style cross-stream coherence read. The two streams' step-deltas now share an
  explicit common component derived from their joint state (a genuine shared cause),
  which the pre-substrate rollout lacked (two independent forward models sharing only
  the action -> coherence reducible to E; failure_autopsy_V3-EXQ-641a_2026-06-06).
  Backward compatible: disabled by default; the CrossStreamBinder submodule is
  constructed ONLY when the master switch is enabled and LAST in E2.__init__, so with
  the flag OFF no parameters are created and no construction-time RNG is consumed --
  verified byte-identical (existing E2 weights unchanged; OFF rollout byte-identical to
  a binder-less baseline).
  Biological basis: MECH-089 theta-gamma nesting (per-step gamma-rate shared code b_t
  nested within the cosine theta window k_t), MECH-270 ephaptic coupling (a shared
  field both streams feel, imposed structurally). Fixed (untrained) projections by
  design -- the 641a retest runs eval() with no P0 curriculum, so a learned head would
  be untrained-random; a fixed joint-state-dependent shared field is the minimal
  genuine common cause. A learned binder is a V4 extension.
  Phased training required: NO (fixed projections; not trained).
  MECH-094: does NOT newly apply (no memory-write surface added; hypothesis_tag
  semantics unchanged; retest is waking simulation_mode=False).
  Validation experiment: V3-EXQ-720 queued (641a harness with binding ON at
  strength=0.5, injected ~34% of the base per-step delta; pre-registered gate
  >=4/6 seeds coherence_specific, margin >=0.05).
  See docs/architecture/sd_cross_stream_binding_substrate.md, INV-002, MECH-089,
  MECH-094, MECH-270, candidate entities/selection.coherence_nonreducibility.

- cross_stream_binding_substrate LEARNED (plastic) mode -- IMPLEMENTED 2026-07-09.
  The residual prerequisite named by failure_autopsy_V3-EXQ-720_2026-07-09: the
  FIXED field (720, strength 0.5) lifted coherence-specificity 1/6->3/6 but did NOT
  clear the 4/6 SPEC gate and n_rebind stayed 0 -- a random projection creates
  correlation but nothing SHAPES the coupling so real conjunctions beat a
  contrast-matched shuffle.
  Same module/file (ree_core/latent/cross_stream_binder.py, CrossStreamBinder), new
  branch on cross_stream_binding_learned; agent hook REEAgent.update_cross_stream_binder
  + REEAgent.cross_stream_binder property (ree_core/agent.py).
  Config: E2Config.cross_stream_binding_learned (default False = fixed field, byte-
  identical; True = learned binder), cross_stream_binding_lr (1e-3),
  cross_stream_binding_temperature (0.5), cross_stream_binding_buffer_size (512),
  cross_stream_binding_batch (64). All no-op default.
  Data flow: plastic phi_self(z_self), phi_world(z_world) -> g_t = tanh(h_self * h_world)
  MULTIPLICATIVE conjunction (coincidence/AND detector) -> b_t = to_common(g_t) -> the
  SAME theta-gated k_t*b_t into both streams (couple() unchanged, mode-agnostic).
  Training: contrastive co-encoding (InfoNCE) -- within-tick observed (z_self, z_world)
  pairs POSITIVE, in-batch shuffle NEGATIVE, symmetric CE over the pairwise
  binding_score = <phi_self(z_self), phi_world(z_world)> matrix. Binder owns its Adam
  optimizer + a detached observed-pair buffer; update_cross_stream_binder buffers a
  DETACHED pair and runs one learn_step (no gradient leaks into E1/E2 encoders).
  Substrate-level rebinding probe folded in: binding_score + rebinding_probe(z_self,
  z_world_candidates, anchor_perturbation) expose the binding intake's own falsifier
  (does a competing world-config OVERTAKE under an ANCHOR perturbation) AT the
  substrate. Anchor (z_self) perturbation is load-bearing: binding_score is bilinear,
  so a uniform candidate perturbation shifts every score by a candidate-independent
  constant and can never flip the argmax; an anchor perturbation gives a per-candidate
  shift that can. A fixed field cannot express any of this (binding_score == 0), which
  is why n_rebind stayed 0 across 641/641a/720.
  Backward compatible: learned=False preserves the fixed-field path byte-identical;
  enabled=False preserves pre-substrate byte-identity. Smoke: OFF binder=None; FIXED
  learned=False + update no-op; LEARNED trains (binding_score pos 2.32 vs neg 0.62 on a
  learned conjunction), rebinding_probe live (22/40 anchor-perturbation flips), fixed-
  mode probe returns rebound=False.
  Biological basis: binding-by-synchrony / communication-through-coherence (Fries;
  Singer/Gray; Buzsaki theta-gamma) is LEARNED and plastic (Hebbian) -- the load-
  bearing divergence the 720 autopsy named. MECH-089 theta gate + MECH-270 ephaptic
  analog carry over.
  Phased training required: YES. P0 = train the binder (per-step
  update_cross_stream_binder); P1 = FREEZE (eval, stop calling it) and run the 641a
  measurement. ML failure modes defended: contrastive collapse (temperature +
  multiplicative product + small bind_dim); encoder contamination / feature suppression
  (P0-only training on detached inputs, binder frozen in P1).
  MECH-094: does NOT newly apply (no memory-write surface; training uses observed waking
  pairs, not replayed/hypothesis content; hypothesis_tag semantics unchanged).
  Validation experiment: V3-EXQ-725 (641a harness with learned=True + P0 binder
  curriculum; n_rebind read through the substrate rebinding_probe; PASS = >=4/6
  SPEC AND n_rebind>0). Queued 2026-07-09 via /queue-experiment.
  See docs/architecture/sd_cross_stream_binding_substrate.md, INV-002, MECH-089,
  MECH-094, MECH-270, candidate entities/selection.coherence_nonreducibility.
- cross_stream_binding_substrate LEARNED-BINDER CONVERGENCE REPAIR -- 2026-07-09.
  failure_autopsy_V3-EXQ-725_2026-07-09: the first learned build did NOT converge --
  InfoNCE loss pinned at chance log(64)=4.16 (observed 3.75-3.96) across 487-1760
  steps, flat + non-monotone; SPEC regressed 3/6 (720 fixed field) -> 0/6, an
  untrained-substrate artifact (NOT a coherence verdict). Root cause (convergence
  probe, evidence/planning/binder_convergence_probe_2026-07-09.md): the observed
  latents are near-collinear buffer-wide (cos ~0.99), so the UN-normalized
  dot-product InfoNCE logit is dominated by near-constant projection MAGNITUDE and
  carries no per-pair contrast.
  Fix (ree_core/latent/cross_stream_binder.py): L2-NORMALIZE phi_self/phi_world
  before the dot in learn_step AND binding_score -- a COSINE InfoNCE (SimCLR-
  standard) that scores DIRECTION only, exposing the residual conjunction signal.
  Loss drops to 0.65-0.80 of chance across seeds (temperature 0.2 deepens the
  margin). Delta-binding / variety-filtering / bind_dim changes tested and NOT
  adopted (zero gain over plain cosine).
  New convergence stat: binder_converged = loss_ema (EMA decay 0.9) < conv_frac *
  log(effective_batch); conv_frac new E2Config param cross_stream_binding_conv_frac
  (default 0.85 -- cleanly rejects the flat-at-0.89 raw path). Exposed as
  CrossStreamBinder.binder_converged / .loss_ema / .chance_floor.
  Backward compatible: all changes inside the learned branch (fixed mode + OFF byte-
  identical). Smoke: FIXED byte-identical; LEARNED converges on the real 725 latent
  stream (loss_ema 3.33 < gate 3.54 -> binder_converged=True at default temp;
  binding_score matched 0.68 vs shuffled -0.10); un-normalized control stays at
  chance (0.885); 725 dry-run exit 0. Substrate status: implemented-but-non-
  functional -> REPAIRED (converges).
  Retest: V3-EXQ-725a (the 725 harness on the repaired binder) MUST gate on a HARD
  learned_binder_converged precondition (loss_ema < 0.85*log(batch)) REPLACING the
  vacuous learned_binder_trained (n_learn_steps>1) check.
