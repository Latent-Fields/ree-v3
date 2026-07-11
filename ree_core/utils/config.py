"""
Configuration classes for REE-v3 components.

V3 changes vs V2:
- LatentStackConfig: z_gamma split into z_self + z_world (SD-005)
  self_dim: proprioceptive/interoceptive channels (E2 domain)
  world_dim: exteroceptive world-state channels (E3/Hippocampal/ResidueField domain)
- E2Config: added action_object_dim for SD-004 action objects
- E3Config: dynamic precision via prediction error variance (ARC-016)
- HippocampalConfig: navigates action-object space O (SD-004)
- REEConfig: added multi-rate clock params (SD-006)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from ree_core.goal import GoalConfig
from ree_core.neuromodulation.serotonin import SerotoninConfig


@dataclass
class LatentStackConfig:
    """Configuration for the V3 latent stack (L-space).

    SD-005: z_gamma is split into z_self and z_world.

    Module responsibilities:
      E1:                    reads z_self + z_world (sensory prior, read-only)
      E2:                    writes z_self (motor-sensory domain)
      E3 / Hippocampal:      writes z_world (world model / planning domain)
      ResidueField:          operates over z_world (world_delta accumulation)

    z_beta (affective latent) remains shared — integrates self and world signals.
    z_theta (sequence context) remains shared.
    z_delta (regime/motivation) remains shared.
    """
    # SD-005: split encoder input dimensions
    body_obs_dim: int = 10     # proprioceptive + interoceptive channels → z_self encoder
    world_obs_dim: int = 54    # exteroceptive + env channels → z_world encoder
    observation_dim: int = 64  # total (body_obs_dim + world_obs_dim); for backward compat

    # Latent dimensions (SD-005)
    self_dim: int = 32         # z_self: proprioceptive/interoceptive (E2 domain)
    world_dim: int = 32        # z_world: exteroceptive world model (E3 domain)

    # Shared latent dimensions (unchanged from V2)
    beta_dim: int = 64         # z_beta: affective latent (shared)
    theta_dim: int = 32        # z_theta: sequence context (shared)
    delta_dim: int = 32        # z_delta: regime/motivation (shared)

    # MECH-099: lateral head for harm-salient features (hazard + contamination)
    # harm_dim=0 disables the lateral head (default: disabled for backward compat)
    harm_dim: int = 0

    # SD-008: per-channel EMA alpha for temporal smoothing in encode().
    # alpha_world should be >= 0.9 (or 1.0 = no blending) — MECH-089 theta buffer
    # already handles temporal integration; encoder EMA double-smoothing suppresses
    # event responses (root cause of EXQ-013/014/018/019 FAIL cluster).
    # alpha_self can remain low (body state is highly autocorrelated).
    alpha_world: float = 0.3   # SD-008: set to 0.9+ to fix event suppression
    alpha_self: float = 0.3

    # SELF-1 / DR-13 (self_model_v4): z_self temporal depth. When
    # use_self_recurrence is True, encode() replaces the z_self EMA smoothing
    # step ONLY with a dedicated gated self-recurrence (SelfRecurrenceCell,
    # a GRUCell over z_self whose hidden state is the previous stateful z_self),
    # optionally blended toward an E1 generative prediction of z_self
    # (self_e1_anchor, supplied by the agent from the cached E1 predicted-next
    # z_self). The blend weight is self_recurrence_e1_coupling:
    #   0.0 = pure self-recurrence (Option A; maximal stability-isolation)
    #   1.0 = pure E1-feedback anchor (Option B)
    #   ~0.15 (default when ON) = HYBRID, light coupling (preserves the
    #         stability-isolation benefit while staying consistent with the
    #         E-stream generative account of the body -- the recorded residual
    #         sub-question from ARC-081 notes 2026-06-14).
    # Default OFF (single-MLP + EMA stays the V3 self latent) -> bit-identical.
    # generation:v4 -- off the V3 closure path; promotes nothing.
    use_self_recurrence: bool = False
    self_recurrence_e1_coupling: float = 0.15

    # SD-010: dedicated harm stream (nociceptive separation, ARC-027).
    # use_harm_stream=False by default — backward compatible with all existing experiments.
    # When True, experiments should construct a HarmEncoder and pass harm_obs separately;
    # harm_obs layout: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1].
    # z_harm bypasses reafference correction by construction (not a perspective-dependent signal).
    use_harm_stream: bool = False
    harm_obs_dim: int = 51    # hazard_field(25) + resource_field(25) + harm_exposure(1)
    z_harm_dim: int = 32

    # SD-011: affective-motivational harm stream (C-fiber/paleospinothalamic analog, ARC-033).
    # Separate from use_harm_stream (SD-010 sensory-discriminative, Adelta-analog).
    # When True, experiments construct AffectiveHarmEncoder and maintain harm_obs_a as EMA
    # of recent harm_obs (tau=10-30 steps). z_harm_a is NOT counterfactually predicted --
    # it feeds E3 commit gating directly (motivational/urgency signal).
    # harm_obs_a layout: same as harm_obs_s by default (full proximity vector EMA);
    # or use harm_obs[50:] (1-dim harm_exposure EMA already emitted by env).
    use_affective_harm_stream: bool = False
    harm_obs_a_dim: int = 50  # hazard_field(25) + resource_field(25) -- no harm_exposure scalar
    z_harm_a_dim: int = 16    # smaller than z_harm_dim -- less spatial resolution needed

    # SD-011 second source: harm history rolling window for AffectiveHarmEncoder.
    # When > 0, env emits harm_history [harm_history_len] (FIFO of past harm_exposure
    # scalars), which is concatenated with harm_obs_a as encoder input. This gives
    # z_harm_a genuinely distinct temporal information that z_harm_s does not receive,
    # breaking the monotone redundancy confirmed by EXQ-241 (D3 reversal).
    # 0 = disabled (backward compat: encoder input dim = harm_obs_a_dim only).
    harm_history_len: int = 0
    # Auxiliary loss weight for harm accumulation prediction head on z_harm_a.
    # Forces encoder to integrate temporal harm info (not just spatial proximity).
    # Only active when harm_history_len > 0. Default 0.1.
    z_harm_a_aux_loss_weight: float = 0.1

    # SD-019a: harm_unpleasantness_channel
    # Non-trainable EMA of z_harm_s (medium timescale ~5-step rise).
    # NOT modulated by controllability (Loffler 2018 three-way dissociation).
    # Serves as input to MECH-219 hysteretic integrator (SD-019b) and redirects
    # AIC urgency + E3 short-horizon urgency_weight away from z_harm_a.
    use_harm_un: bool = False
    harm_un_ema_alpha: float = 0.2   # medium: ~5-step rise to z_harm_s=1.0

    # SD-007: ReafferencePredictor — perspective-shift correction for z_world.
    # Set reafference_action_dim = action_dim (e.g. 4) to enable.
    # 0 = disabled (default; backward compatible with EXQ-001-025).
    # When enabled, LatentStack.encode() accepts prev_action and applies:
    #   z_world_corrected = z_world_raw - ReafferencePredictor(z_self_prev, a_prev)
    # Trained on empty-space steps (transition_type=="none") only.
    # See docs/architecture/sd_004_sd_005_encoder_codesign.md §3.
    reafference_action_dim: int = 0

    # SD-009: event contrastive classifier head on z_world encoder (MECH-100).
    # When True, SplitEncoder.forward() returns event logits [batch, 3] and
    # training loop can apply CE loss with event_type labels from the environment.
    # Labels: 0=none, 1=env_caused_hazard, 2=agent_caused_hazard.
    use_event_classifier: bool = False

    # SD-018: resource proximity regression head on z_world encoder.
    # When True, SplitEncoder.forward() returns resource_prox_pred [batch, 1]
    # and training loop can apply MSE loss with max(resource_field_view) target.
    # Forces z_world to encode resource proximity, which SD-009 does NOT cover
    # (SD-009 discriminates event types but not resource saliency).
    # Without this, benefit_eval_head(z_world) produces R2=-0.004 (EXQ-085m).
    use_resource_proximity_head: bool = False
    # Auxiliary loss weight (scales MSE contribution to total loss).
    resource_proximity_weight: float = 0.5

    # Q-007 / EXQ-051b: volatility (NE/LC) signal injection into beta_encoder.
    # When > 0, beta_encoder input becomes cat(z_self_init, z_world_init, volatility_signal)
    # where volatility_signal [batch, volatility_signal_dim] is a running estimate of E3's
    # harm prediction error variance. Disabled (0) by default — backward compatible.
    # Biological basis: LC-NE encodes unexpected uncertainty (Yu & Dayan 2005 PMID 15944135):
    # z_beta's arousal dimension should reflect how unpredictable harm has been, not just
    # current sensory content. running_variance is the REE analog of HGF log-volatility (μ₃).
    # See claims: Q-007, MECH-093.
    volatility_signal_dim: int = 0

    # Ablation flag: fuse z_self and z_world into a single shared representation
    # after encoding. When True, both channels receive the average of z_self and
    # z_world, eliminating channel specialization. Used by EXQ-044 to test whether
    # the z_self/z_world split itself (vs. unified latent) provides efficiency gains.
    unified_latent_mode: bool = False

    # MECH-423 R2 (iterative-inference convergence readout; ARC-004 inference
    # machinery). The legacy encode() is a FIXED two-pass amortized recognition
    # (bottom-up init -> one top-down round): there is no settling loop, so the
    # per-inference-step ||delta z_shared|| the EXP-0380 R2 readiness check needs
    # has no source. When use_iterative_inference=True, encode() generalises the
    # single top-down round into a predictive-coding settling loop over the shared
    # z_beta -> z_theta -> z_delta stack (run BEFORE the EMA smoothing), tracking
    # per-round relative-delta and exposing a convergence readout on the returned
    # LatentState.inference_convergence (and agent.last_inference_convergence).
    # Grounds Gershman & Goodman 2014 (amortized inference; the amortization gap
    # makes "converged" measurable). Disabled by default -- bit-identical OFF
    # (inference_settle_iters is forced to 1 in the OFF path, i.e. the legacy
    # single round, and inference_convergence stays None).
    use_iterative_inference: bool = False
    # Maximum number of settling rounds when use_iterative_inference is True
    # (round 1 == the legacy top-down round; rounds 2..K are the extra settling
    # iterations). 1 == legacy behaviour even when the master flag is on.
    inference_settle_iters: int = 1
    # Relative-delta early-stop tolerance: the loop stops once
    # ||z_shared_k - z_shared_{k-1}|| / (||z_shared_k|| + eps) < this value, and
    # inference_convergence["converged"] reports whether that fixed point was
    # reached within inference_settle_iters rounds. EXP-0380 R2 asserts
    # final_rel_delta < 0.05.
    inference_convergence_rel_tol: float = 0.05

    # ARC-033: E2_harm_s forward model (sensory-discriminative harm stream predictor).
    # When True, experiments should construct an E2HarmSForward instance and train it
    # on z_harm_s transitions (P1 phase, after P0 HarmEncoder warmup).
    # Disabled by default -- backward compatible with all existing experiments.
    # See ree_core/predictors/e2_harm_s.py for the module and training protocol.
    use_e2_harm_s_forward: bool = False

    # SD-031: E2_world causal-footprint forward model + single-pass comparator
    # (the z_world instantiation of MECH-256; sibling to ARC-033 on z_harm_s).
    # When True, experiments should construct an E2WorldForward instance and
    # train it on z_world transitions (P1, after P0 z_world encoder warmup with
    # SD-009 + SD-018). The comparator's discriminative/attribution arm requires
    # world_dim >= 128 (E2WorldForward hard-asserts this; the 2026-06-06 cluster
    # autopsy established z_world at dim=32 lacks discriminative granularity).
    # Disabled by default -- backward compatible with all existing experiments.
    # See ree_core/predictors/e2_world.py for the module and training protocol.
    use_e2_world_forward: bool = False

    # SD-063 -- E2 conditional predictive-uncertainty head (distribution-free
    # quantile/pinball form). A SEPARATE readout head over (z_world, action) that
    # emits a per-input predictive spread tracking realized error, feeding E3
    # commitment gating (E3Config.use_conditional_precision_gate) in place of the
    # state-blind running-variance EMA. Winner of the V3-EXQ-712 diagnostic
    # (quantile CRPS 0.00486 vs point 0.00514; precision_error_corr 0.379 vs the
    # EMA null 0.0; Gaussian-family heads did WORSE than the point baseline).
    # CAVEAT (SD-031): the head is a STANDALONE module that reads DETACHED z_world
    # and shares NO parameters with E2WorldForward or the encoder -- its P1 loss
    # detaches both inputs and targets -- so it structurally cannot absorb/explain
    # away the E2WorldForward agency residual (MECH-256/SD-031). Like
    # use_e2_world_forward, this flag signals intent; the head is instantiated at
    # the experiment/agent level (LatentStack.encode() is untouched, so OFF is
    # byte-identical). Phased training: P0 encoder warmup -> P1 head on detached
    # z_world targets -> P2 eval. Disabled by default -- backward compatible.
    # See ree_core/predictors/e2_world_uncertainty.py; instantiates MECH-059.
    use_e2_world_uncertainty: bool = False
    e2_world_uncertainty_hidden_dim: int = 128   # trunk width (V3-EXQ-712 HEAD_HIDDEN)
    e2_world_uncertainty_lr: float = 1e-3        # head LR (V3-EXQ-712 HEAD_LR)

    # MECH-441 -- model-disagreement directed curiosity ensemble (RND / Plan2Explore
    # analog). A SMALL K-head ensemble of independent residual-delta predictors over
    # (z_world, action) -- a standalone ModelDisagreementEnsemble built at the agent
    # level (ree_core/curiosity/model_disagreement.py) when n_disagreement_heads >= 2.
    # Each head is a distinct-init delta predictor; per-candidate model DISAGREEMENT =
    # cross-head variance of the predicted next-z_world for that candidate's first
    # action, fed into E3 selection as a propagating per-candidate curiosity bonus
    # (E3Config.use_model_disagreement_curiosity). Phased training (mirrors the SD-031
    # P0/P1/P2): each head trains on the SAME frozen-z_world target
    # (target = z_world_next.detach()) in P1; disagreement_bootstrap_mask_prob > 0
    # gives each head a per-step Bernoulli mask so heads diverge in poorly-sampled
    # regions (epistemic disagreement) and converge where data is dense (the intrinsic
    # self-annealing). n_disagreement_heads <= 1 -> ensemble NOT built -> bit-identical
    # OFF. Built independently of use_e2_world_forward (it needs only z_world + action).
    n_disagreement_heads: int = 0
    disagreement_bootstrap_mask_prob: float = 0.0
    disagreement_learning_rate: float = 3e-4

    # SD-015 / MECH-112: dedicated ResourceEncoder for goal-directed navigation.
    # When True, LatentStack.encode() produces z_resource [batch, z_resource_dim]
    # from raw world_obs -- an object-type latent independent of spatial position.
    # GoalState.update() seeds from z_resource (not z_world) when this is enabled,
    # giving the goal system "what to seek" rather than "where it was."
    # Disabled by default -- backward compatible with all existing experiments.
    # See ree_core/latent/stack.py: ResourceEncoder.
    use_resource_encoder: bool = False
    z_resource_dim: int = 32  # must match GoalConfig.goal_dim for direct seeding

    # SD-049 Phase 2 hybrid identity classifier (Option C per verdict.md).
    # When True AND use_resource_encoder=True, ResourceEncoder gains an
    # identity_head: Linear(z_resource_dim, n_resource_types) supervised by
    # cross-entropy on obs_dict["resource_type_at_agent"] from the SD-049
    # multi_resource_heterogeneity substrate. The classifier head pulls on
    # the trunk during P0 supervised training (anti-collapse mitigation per
    # Levi 2021 + biology anchored to Ballesta-Padoa-Schioppa 2019 +
    # Quiroga 2005 + Schapiro 2017 hybrid CLS). z_resource shape unchanged
    # (still z_resource_dim); identity_logits exposed as separate
    # LatentState field for the loss computation. Disabled by default --
    # backward compatible.
    # Phased training: P0 enable + joint loss; P1 freeze identity_head
    # parameters; P2 evaluate identity-recovery + wanting!=liking + per-axis
    # drive ANOVA on z_goal cluster IDs (V3-EXQ-514 acceptance).
    use_identity_classifier: bool = False
    identity_classifier_n_types: int = 3  # default matches SD-049 default

    # Top-down conditioning
    topdown_dim: int = 16

    # Activation function
    activation: str = "relu"


@dataclass
class E1Config:
    """Configuration for E1 Deep Predictor.

    V3: E1 produces predictions over both z_self and z_world channels
    (associative prior). Unchanged in function from V2.

    Episode-boundary semantics: E1 maintains persistent hidden state
    across episode steps, reset via reset_hidden_state().

    SD-016: frontal cue-indexed integration.
    When sd016_enabled=True, E1 gains world_query_proj, cue_action_proj,
    cue_terrain_proj and exposes extract_cue_context(z_world) which queries
    ContextMemory with z_world alone (exteroceptive cues only) and produces
    action_bias [batch, action_object_dim] for E2 and terrain_weight [batch, 2]
    for E3. Disabled by default for full backward compatibility.
    """
    self_dim: int = 32         # z_self dimension
    world_dim: int = 32        # z_world dimension
    latent_dim: int = 64       # total = self_dim + world_dim
    hidden_dim: int = 128
    num_layers: int = 3
    prediction_horizon: int = 20   # Steps into future
    learning_rate: float = 1e-4

    # SD-016: frontal cue-indexed integration circuit (MECH-150/151/152, ARC-041)
    sd016_enabled: bool = False
    action_object_dim: int = 16    # must match E2Config.action_object_dim

    # SD-016 ContextMemory write-path mode (2026-04-25, EXQ-477 follow-up).
    # Controls whether observation-conditioned writes are made to the
    # ContextMemory slots. Values:
    #   "off"         - no writes (Part A only; legacy behaviour pre-write-path)
    #   "train_only"  - write inside REEAgent.compute_prediction_loss
    #                   (training-time hook, after E1 prediction error computed)
    #   "sense_only"  - write inside REEAgent.sense() (per-tick hook, gated
    #                   on _offline_mode at the call site)
    #   "both"        - both hooks active
    # Validation: V3-EXQ-418d (4-arm comparison; smallest slot_diversity
    # collapse + largest action_class_entropy ablation delta wins as default).
    # Default "off" preserves bit-identical observable behaviour for any
    # non-SD-016 experiment that does not opt in.
    # See REE_assembly/docs/architecture/context_memory_writepath_fix.md.
    sd016_writepath_mode: str = "off"

    # SD-016 Path 4 (V3-EXQ-418g): learnable attention temperature on the
    # z_world-only ContextMemory query inside extract_cue_context().
    # When True, exp(log_tau) replaces the fixed sqrt(memory_dim) divisor and
    # the parameter is exposed via self.sd016_log_temperature. Pair with
    # E1.compute_attention_entropy_loss() to apply gradient pressure toward
    # peaky attention (the EXQ-418e diagnosis: slot-side diversification
    # alone cannot move attention off the uniform rail because
    # world_query_proj has no gradient signal demanding selectivity).
    # Initialised at log(sqrt(memory_dim)) so step-0 behaviour matches the
    # fixed-temperature baseline. Default False = bit-identical to legacy.
    sd016_temperature_learnable: bool = False

    # SD-016 Path 3 (2026-06-05): feedforward cue->slot tagger.
    # V3-EXQ-418i established that the z_world-only q.k attention inside
    # extract_cue_context() is pinned at the uniform ln(num_slots) saddle
    # ("the attention bottleneck is categorically in query selectivity"):
    # key_proj(memory) with memory init 0.01 yields near-identical keys, so
    # softmax stays uniform and the softmax Jacobian at uniform is a flat
    # saddle that terrain_loss gradient cannot escape (Path 1 div-loss
    # exhausted across weights 1.0/2.0/5.0). Path 3 replaces ONLY the slot-
    # selection scores (not the slot-content value_proj/output_proj path)
    # with a fresh feedforward MLP cue_slot_tagger: z_world -> slot logits.
    # A random MLP sits off the saddle, so the existing cue_terrain_proj
    # terrain_loss gradient flows back into it and shapes contextual
    # selectivity. Requires sd016_enabled=True. Default False = the legacy
    # q.k attention branch runs verbatim (bit-identical).
    sd016_cue_slot_tagger: bool = False
    sd016_cue_slot_tagger_hidden: int = 32       # tagger MLP hidden width
    sd016_cue_slot_tagger_temperature: float = 1.0  # softmax temp on tagger logits

    # MECH-216: E1 predictive wanting (schema readout head).
    # When enabled, a Linear(hidden_dim, 1)+Sigmoid head reads E1's LSTM hidden state
    # and produces a scalar schema_salience in [0, 1]. High salience at positions where
    # E1's internal schemas predict resource proximity seeds VALENCE_WANTING before
    # direct resource contact (Pavlovian conditioned wanting, Zhang/Berridge 2009).
    schema_wanting_enabled: bool = False


@dataclass
class E2Config:
    """Configuration for E2 Fast Predictor.

    V3 extensions (SD-004, SD-005):
    - forward() operates on z_self (motor-sensory prediction)
    - world_forward() operates on z_world (for SD-003 attribution pipeline)
    - action_object() produces compressed world-effect representation o_t (SD-004)

    E2 trains on motor-sensory prediction error (z_self). NOT harm/goal error.
    rollout_horizon must exceed E1.prediction_horizon (30 > 20).

    action_object_dim: dimensionality of action object o_t produced by SD-004.
    Should be << world_dim to provide compression benefit.
    """
    self_dim: int = 32             # z_self dimension (E2 primary domain)
    world_dim: int = 32            # z_world dimension (for world_forward + action_object)
    action_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    rollout_horizon: int = 30      # Must exceed E1.prediction_horizon = 20
    num_candidates: int = 32
    action_object_dim: int = 16    # SD-004: compressed world-effect action object dim
    learning_rate: float = 3e-4

    # SD-056: action-conditional divergence preservation via auxiliary InfoNCE
    # contrastive loss on world_forward. Default OFF -- bit-identical to pre-SD-056.
    # See REE_assembly/docs/architecture/sd_056_e2_action_conditional_divergence.md.
    e2_action_contrastive_enabled: bool = False
    e2_action_contrastive_weight: float = 0.01
    e2_action_contrastive_temperature: float = 0.1
    e2_action_contrastive_min_batch_classes: int = 2

    # SD-056 multi-step rollout stability amend (2026-05-31).
    # Per V3-EXQ-569e autopsy: t=1 contrastive leaves get_world_state_sequence()
    # unbounded over the behavioural-runtime horizon (1e16+ magnitudes on most ON
    # seeds at P1 50ep/200step measurement). Two togglable levers (both default
    # OFF, bit-identical to SD-056-only path):
    #   (a) multi-step contrastive: extend the contrastive objective to h-step
    #       rollouts (Dreamer/PlaNet anchor; Srivastava 2021 contrastive RSSM
    #       lever B). Helper: E2FastPredictor.world_forward_contrastive_loss_multistep.
    #   (b) per-step output norm clamp inside E2.rollout_with_world (B2 anchor:
    #       clamp predicted ||z_world_{t+1}|| against ratio * ||z_world_0|| so
    #       the rollout is bounded by the initial-state scale, not by a
    #       compounding per-step ratio).
    e2_action_contrastive_multistep_enabled: bool = False
    e2_action_contrastive_horizon: int = 5
    e2_action_contrastive_horizon_weights_decay: float = 1.0
    e2_rollout_output_norm_clamp_enabled: bool = False
    e2_rollout_output_norm_clamp_ratio: float = 2.0

    # cross_stream_binding_substrate (2026-07-08): shared-latent-factor binding
    # coupling injected into BOTH streams inside E2.rollout_with_world so the
    # z_world<->z_self rollout deltas share a genuine common cause. This is the
    # substrate 641a needed: with these streams unbound, cross-stream coherence
    # C(tau) is reducible to E (failure_autopsy_V3-EXQ-641a_2026-06-06). All
    # no-op default -- OFF is byte-identical (the CrossStreamBinder submodule is
    # constructed only when cross_stream_binding_enabled is True, so OFF consumes
    # no parameters and no construction-time RNG). See
    # docs/architecture/sd_cross_stream_binding_substrate.md.
    cross_stream_binding_enabled: bool = False
    cross_stream_binding_dim: int = 16          # shared-factor dim (bind_dim)
    cross_stream_binding_strength: float = 0.15  # coupling scale (kappa)
    cross_stream_binding_theta_period: int = 4   # MECH-089 theta window period (steps)
    # LEARNED (plastic) binder (2026-07-09; cross_stream_binding_substrate V4
    # next-step per failure_autopsy_V3-EXQ-720_2026-07-09). The FIXED random
    # projection field (720, strength 0.5) lifted coherence-specificity 1/6->3/6
    # but did NOT clear the 4/6 SPEC gate: nothing SHAPES the coupling so real
    # cross-stream conjunctions beat a contrast-matched shuffle (n_rebind=0). The
    # learned binder trains phi_self/phi_world by contrastive co-encoding (InfoNCE
    # over within-tick positives vs in-batch shuffle negatives) so genuine
    # conjunctions bind and a shuffle collapses -- the load-bearing biology
    # divergence (binding-by-synchrony is LEARNED, not a fixed field). All no-op
    # default; learned=False preserves the fixed-field path byte-identical.
    # Requires phased training (P0 binder curriculum -> P1 frozen measurement).
    cross_stream_binding_learned: bool = False   # False = fixed field (720 path)
    cross_stream_binding_lr: float = 1e-3        # P0 binder optimizer LR
    cross_stream_binding_temperature: float = 0.5  # InfoNCE temperature
    cross_stream_binding_buffer_size: int = 512  # co-encoding pair buffer
    cross_stream_binding_batch: int = 64         # contrastive batch size
    # Convergence gate (failure_autopsy_V3-EXQ-725_2026-07-09 repair): the
    # substrate reports binder_converged = smoothed_loss < conv_frac*log(batch).
    # Report-only -- affects no dynamics. Default 0.85 cleanly rejects the raw
    # (un-normalized) curriculum that sat flat at ~0.89 of chance while accepting
    # the L2-normalized cosine-InfoNCE curriculum (0.65-0.80 of chance across
    # seeds; see evidence/planning/binder_convergence_probe_2026-07-09.md).
    cross_stream_binding_conv_frac: float = 0.85


@dataclass
class E3Config:
    """Configuration for E3 Trajectory Selector.

    V3 extensions:
    - harm_eval() method: evaluates harm of a z_world state (SD-003 V3 pipeline)
    - Dynamic precision: derived from E3 prediction error variance (ARC-016)
      replaces hardcoded precision_init/precision_max/precision_min
    - benefit_eval_head: Go channel for symmetric approach/avoidance (ARC-030, MECH-112)
    - novelty_bonus: E1 prediction error variance as exploration signal (MECH-111)
    - self_maintenance_weight: penalty on high z_self D_eff (MECH-113)

    Scoring equation J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) remains a working
    hypothesis — see ARCHITECTURE NOTE in e3_selector.py.
    """
    world_dim: int = 32        # z_world dimension (E3 domain)
    latent_dim: int = 64       # for backward compat (beta_dim)
    hidden_dim: int = 64

    # Scoring weights — placeholder parameters, not tuned constants
    lambda_ethical: float = 1.0
    rho_residue: float = 0.5

    # Dynamic precision (ARC-016): precision derived from prediction error variance
    # commit_threshold is in VARIANCE SPACE: committed when running_variance < threshold.
    # Recalibrated 2026-03-20: EXQ-038 shows running_variance converges to ~0.33 in
    # trained environments (not ~0.003 as assumed from EXQ-018 which used a different
    # z_world scale). threshold=0.40 lets trained agents commit (variance ~0.33 < 0.40)
    # while untrained agents don't (variance ~0.50 = precision_init). This is a pragmatic
    # fix; ARC-016 semantics will improve once ARC-027 separates z_harm from z_world,
    # reducing z_world's inherent variance and sharpening the stable/perturbed gap.
    # (Prior value 0.003 was 25,000× below actual running_variance → never committed.
    # EXQ-048/049 confirmed: beta gate never elevated, MECH-057b/090 could not be tested.)
    commitment_threshold: float = 0.40    # variance-space threshold
    precision_ema_alpha: float = 0.05     # EMA decay for running variance estimate
    precision_init: float = 0.5          # initial running variance (starts uncommitted)

    # SD-063: conditional predictive-precision commit gate. When True AND a
    # conditional_predictive_variance is passed to select(), the ARC-016 commit
    # decision compares that per-input predictive variance (from the SD-063
    # E2WorldUncertaintyHead over the leading candidate's (z_world, a) transition)
    # against effective_threshold, INSTEAD of the state-blind running_variance EMA.
    # This is the "gate on where THIS prediction is uncertain" realization of the
    # MECH-059 confidence channel. 0/None -> falls back to the EMA path
    # (byte-identical OFF). Does not affect the harm-variance-commit path.
    use_conditional_precision_gate: bool = False

    # ARC-030 / MECH-112: Go channel — benefit evaluation head.
    # benefit_eval_head maps z_world -> [0,1] (resource/goal proximity score).
    # When enabled, score_trajectory() subtracts benefit_weight * benefit_score,
    # creating competitive Go/NoGo evaluation of the same trajectory proposals.
    # benefit_eval_enabled=False by default — backward compatible with all prior experiments.
    benefit_eval_enabled: bool = False
    benefit_weight: float = 1.0

    # MECH-111: novelty bonus — E1 prediction error variance as exploration signal.
    # novelty_bonus_weight=0 disables (default). When > 0, score_trajectory()
    # subtracts novelty_bonus_weight * novelty_score, favouring novel states.
    # Novelty is tracked as an EMA of E1 MSE error per-trajectory final state.
    novelty_bonus_weight: float = 0.0

    # MECH-113: self-maintenance penalty — penalises high z_self D_eff.
    # D_eff = (sum|z_self|)^2 / sum(z_self^2) — participation ratio (epistemic-mapping).
    # self_maintenance_weight=0 disables (default). When > 0, training applies
    # an auxiliary loss penalising D_eff > self_maintenance_d_eff_target.
    # This is an agent.py training signal, not a trajectory scoring term.
    self_maintenance_weight: float = 0.0
    self_maintenance_d_eff_target: float = 1.5   # target max D_eff for z_self

    # MECH-112 / MECH-117: wanting signal weight in trajectory scoring.
    # score_trajectory() subtracts goal_weight * goal_proximity when goal_state is active.
    # Logically belongs here (it's an E3 scoring parameter) even though GoalConfig also
    # carries a copy for standalone goal experiments. Canonical source: this field.
    # 0.0 disables goal contribution (backward-compatible default).
    goal_weight: float = 0.0

    # SD-011: z_harm_a urgency modulation of commit threshold (ARC-016 reframe).
    # When > 0 and z_harm_a is provided to select(), effective_threshold is LOWERED
    # proportionally to z_harm_a.norm(), making the agent commit faster under threat
    # (D2 avoidance escape response). 0.0 disables (default, backward compat).
    urgency_weight: float = 0.0
    urgency_max: float = 0.5    # saturation cap: threshold never drops below 50% of base

    # SD-011: z_harm_a amplification of M(zeta) ethical cost.
    # lambda_eff = lambda_ethical * (1.0 + affective_harm_scale * z_harm_a_norm)
    # When accumulated threat is high, harm costs weigh more in trajectory scoring.
    # 0.0 disables (default, backward compat).
    affective_harm_scale: float = 0.0

    # MECH-091: urgency interrupt threshold -- phase-reset commitment on high harm signal.
    # When beta is elevated and z_harm_a.norm() exceeds this threshold, the commitment is
    # aborted (beta_gate.release(), _committed_step_idx reset), and a fresh E3 selection
    # is performed. This implements the salient-event phase-reset described in MECH-091.
    # 0.8 default: fires only on strong affective harm signal; 1e9 effectively disables.
    urgency_interrupt_threshold: float = 0.8

    # V3-EXQ-563c: experimental score-bias scale normalisation.
    # When True, score_bias passed to select() is rescaled by the ratio
    # raw_score_range / bias_range before application so that a given bias
    # magnitude exerts a consistent *relative* push regardless of the raw
    # score spread. Default False (backward compat: bias applied as-is).
    normalize_score_bias_to_e3_range: bool = False

    # modulatory-bias-selection-authority (2026-06-03): gap-relative scaling that
    # gives the composed modulatory score-bias (dACC + lateral_pfc + ofc + mech295
    # + MECH-314 curiosity + MECH-320 vigor) and the MECH-341 entropy bonus genuine
    # but BOUNDED authority at committed selection. Root cause (604a/624a/614d
    # cluster autopsy): fixed small bias magnitudes (~0.05-0.1) added to primary
    # scores whose raw_score_range is much larger never change the argmin.
    # When use_modulatory_selection_authority is True, the COMBINED modulatory
    # addition is rescaled so its range == modulatory_authority_gain * raw_score_range
    # before being added to scores. Primary scores are NOT modified (commit /
    # running_variance / softmax-temperature / urgency / MECH-090 admission
    # semantics unchanged). gain < 1.0 keeps the modulatory layer competitive in
    # near-tie regimes but subdominant when the primary harm/goal gap exceeds
    # gain * range (a clearly-harmful candidate stays rejected). Takes precedence
    # over normalize_score_bias_to_e3_range (the blunt gain=1.0 blob version) when
    # both are set. Default False (bit-identical OFF).
    use_modulatory_selection_authority: bool = False
    modulatory_authority_gain: float = 0.5
    modulatory_authority_min_range_floor: float = 1e-6

    # ARC-108 JOB-1 step-1 (learned dopamine-gated E3 gating, dopamine_into_gating
    # design 2026-06-22 secs 2-4): a single LEARNED per-channel selection-weight
    # vector w_chan over the modulatory channels that feed _modulatory_accum
    # (score_bias chain / MECH-341 entropy bonus / route bias), composed as
    # _modulatory_accum = sum_c softplus(w_chan[c]) * channel_bias_c (was the
    # unweighted sum). At init w_chan is set so softplus(w_chan[c]) == 1.0 for all
    # c -> BIT-IDENTICAL to the current unweighted accumulator. w_chan is updated
    # by a three-factor rule -- Delta w[c] = eta * delta_t * eligibility_c *
    # asym(delta_t) -- where delta_t is a SIGNED dopaminergic-RPE analog
    # (R_t - V-hat_t; R_t = benefit_eval - harm_eval at the realised state from the
    # ALREADY-TRAINED valuation heads; V-hat_t a slow EMA baseline), eligibility_c
    # is the decayed Hebbian co-activation trace |channel_bias_c[selected]|, and
    # asym renders the D1-LTP/D2-LTD asymmetry as a single asymmetric gain
    # (potentiation on delta_t>=0 faster than depression on delta_t<0). delta_t is
    # SIGNED and is explicitly NOT the unsigned ARC-016 prediction-error VARIANCE
    # (e3._running_variance) -- divergence B5; they are kept separate. Learning
    # composes INSIDE the F-bounded MECH-448/449 eligible set (it re-weights only
    # _modulatory_accum, never raw scores/F), so safety is inherited from the
    # envelope -- a learned weight can never re-admit a No-Go-suppressed candidate.
    # The update runs ONLY on the waking committed-selection path (a replay/DMN
    # simulation tick forms no delta_t and writes no w_chan). The MECH-450
    # recurrent-settling step (learned W_lat) is the SECOND factor and is OFF in
    # this build (W_lat == 0 -- the integration point is reserved, not enabled).
    # Default False -> bit-identical OFF.
    use_learned_channel_gating: bool = False
    learned_channel_gating_eta: float = 0.01           # three-factor learning rate
    learned_channel_gating_elig_decay: float = 0.9     # Hebbian eligibility-trace decay (last-K-ticks window)
    learned_channel_value_baseline_beta: float = 0.05  # V-hat_t slow-EMA baseline rate
    learned_channel_asym_potentiation: float = 1.0     # D1-LTP gain on delta_t >= 0
    learned_channel_asym_depression: float = 0.5       # D2-LTD gain on delta_t < 0 (slower)
    # ARC-108 sec-7 C3 ablation (divergence B5): the teaching signal that drives the
    # w_chan AND W_lat three-factor updates. "signed" (default) uses the SIGNED RPE
    # delta_t = R_t - V-hat_t -- bit-identical to the current substrate. "unsigned"
    # substitutes the UNSIGNED ARC-016 prediction-error magnitude (e3._running_variance,
    # always >= 0), removing the directional potentiate-vs-depress credit. This is the
    # C3 falsifier knob (the signed-RPE claim is refuted if unsigned converts just as
    # well -> route back to ARC-016). Only the LEARNING updates change; the signed
    # delta_t is kept intact for the JOB-2 habenula de-commit + the V-hat_t baseline EMA.
    learned_channel_rpe_mode: Literal["signed", "unsigned"] = "signed"
    # ARC-108 JOB-2 (d): the e3-level mirror of REEConfig.use_habenula_decommit.
    # post_action_update reads THIS (self.config is the E3Config) to decide whether
    # to compute the signed RPE delta_t = R_t - V-hat_t + advance the shared V-hat_t
    # on every waking post-action (NOT gated on a JOB-1 eligibility trace), and emit
    # habenula_delta_t for REEAgent.update_residue to route into the SD-034 habenula
    # abort. Set from the single from_dims param onto BOTH REEConfig (agent wiring /
    # operator) and config.e3 (this). Default False -> bit-identical.
    use_habenula_decommit: bool = False

    # MECH-451: finer-channel-granularity selection-gating (the cheap V3 rung
    # BETWEEN ARC-108's single global w_chan over the pre-COMPRESSED score_bias
    # blend and ARC-110's full V4 segregated loops). When True, the single
    # ARC-108 "score_bias" learned-gating channel is EXPLODED into its
    # constituents (OFC / dACC / lateral-PFC / vigour / liking / gated_policy +
    # a "residual" lump of everything else summed into score_bias), each with
    # its OWN learned w_chan_finer entry trained by the SAME ARC-108 signed-RPE
    # three-factor rule -- so the learner can discover SEPARATELY that different
    # control functions matter in different states, keeping ONE shared arena
    # (NOT ARC-110 per-loop competition). Tests whether the F-dominance
    # conversion ceiling (MECH-439) is REPRESENTATIONAL COMPRESSION (finer
    # channels convert non-motor influence to committed action) rather than a
    # need for full per-loop competition. Reuses the ARC-108 learning knobs
    # (eta / elig_decay / asym / rpe_mode) so A1_GLOBAL_WCHAN vs A2_FINER differ
    # ONLY in channel granularity. Default False -> bit-identical OFF; at init
    # softplus(w_chan_finer[c]) == 1.0 so the finer decomposition reproduces the
    # compressed blend EXACTLY even when ON. ARC-108 (use_learned_channel_gating)
    # path is UNTOUCHED (parallel buffer); the two are mutually exclusive at the
    # score_bias slot, finer takes precedence. See MECH-451 / EXP-0391.
    use_finer_channel_gating: bool = False

    # ARC-108 JOB-1 step-2 / MECH-450 (the coupled recurrent-settling step,
    # dopamine_into_gating design 2026-06-22 sec 4): the SECOND factor of the
    # learned-gating 2x2. A bounded recurrent lateral-inhibition SETTLING step over
    # the F-bounded eligible set BEFORE the within-eligible commit -- a few rounds R
    # of  a = softmax(-accum/T); accum += W_lat-coupled inhibition  on
    # _modulatory_accum[eligible_idx] -- then commit = argmin(settled accum)
    # (committed) / sample(softmax(-settled accum/T)) (uncommitted). Fixes divergence
    # B1 (one-shot argmin -> recurrent settling) AND B3-blend (additive
    # _modulatory_accum blend -> competitive winner-take-most) together. W_lat is the
    # LEARNED lateral-inhibition matrix over candidate first-action CLASSES (a stable
    # [C, C] object -- the per-candidate set is variable-size with no stable identity,
    # so the inhibition is parametrised by action class: the BG surround-inhibition
    # between competing motor programs, Mink 1996, which MECH-449 already grounds).
    # W_lat is a register_buffer (NOT nn.Parameter -- the three-factor plasticity is a
    # LOCAL update, never an optimizer/autograd target), init 0 -> the settling step
    # is a no-op -> BIT-IDENTICAL OFF and bit-identical at init. W_lat is learned by
    # the SAME signed-RPE three-factor rule as w_chan (sharing the JOB-1 delta_t /
    # V-hat_t / D1-D2 asym): Delta W_lat ~ eta * delta_t * asym(delta_t) * coact_trace,
    # where coact_trace is the decayed Hebbian co-activation of the settling-step class
    # activations. The settling composes INSIDE the F-bounded MECH-448/449 eligible set
    # (it transforms only _modulatory_accum[eligible_idx]; raw scores / F untouched), so
    # safety is inherited from the envelope -- a learned W_lat can never re-admit a
    # No-Go-excluded candidate. Waking-only (no settling / W_lat write on a simulation
    # tick, mirroring the JOB-1 gate). Default False -> bit-identical OFF.
    use_learned_settling_step: bool = False
    learned_settling_rounds: int = 3                   # R: mutual-inhibition rounds per tick
    learned_settling_temperature: float = 1.0          # T: softmax temperature for the support a
    learned_settling_eta: float = 0.01                 # W_lat three-factor learning rate
    learned_settling_elig_decay: float = 0.9           # cross-tick decay of the co-activation trace
    learned_settling_n_action_classes: int = 8         # W_lat dimension = first-action class count (clamped)

    # ============================================================== #
    # MECH-140 x MECH-450: disinhibitory soft-competitive settling    #
    # ============================================================== #
    #
    # The PARAMETER-FREE, always-graded complement to the learned W_lat settling
    # above. MECH-140 (soft-competitive disinhibition -- losing options
    # down-weighted but NOT silenced, rather than winner-take-all) x MECH-450 (a
    # minimal recurrent SETTLING step -- a few rounds of mutual/lateral inhibition
    # over the eligible set before commit, replacing the one-shot argmin). Unlike
    # W_lat (learned, no-op at init) this bites IMMEDIATELY when enabled -- it needs
    # no dopaminergic learning -- and so can flip the committed attractor at init,
    # the "recurrence gives the readout an attractor-flip the argmin lacks" property
    # MECH-450 asserts. Runs a few rounds of soft-competitive lateral inhibition on
    # the within-eligible field (COST units, lower = better): x = -field (activation,
    # higher=better); support = softmax(x / T) (graded, all > 0 -> never silenced);
    # inhib_i = gain * sum_j K_ij * support_j; x -= inhib_i; return -x. K_ij is the
    # PARAMETER-FREE class-surround kernel: 1.0 within the same first-action class,
    # `cross_class` (< 1) across classes -- surround-inhibition between competing
    # motor programs (Mink 1996; the SAME structure W_lat learns, here FIXED). The
    # structured kernel lets the settling REORDER (a candidate crowded by same-class
    # rivals loses to an isolated slightly-worse one), so the step is behaviourally
    # non-vacuous against the MECH-439 F-dominance conversion ceiling, not a
    # rank-preserving sharpen. Operates STRICTLY within the F + MECH-448/449 eligible
    # set (safety inherited -- can reorder within-eligible, never re-admit a
    # No-Go-excluded candidate; >= 1 survivor always). Waking-only (no settling on a
    # simulation/replay tick, MECH-094). No learned parameters, no autograd. Composes
    # with W_lat (learned within-loop) and the learned cross-loop arbitration
    # (M_cross, across loops) -- it is a within-eligible / within-loop transform at a
    # different level. Default False -> skipped -> BIT-IDENTICAL OFF; and gain 0.0 ->
    # inhib == 0 -> exact no-op even when the flag is on (byte-identical at default,
    # mirroring noisy_selection_sigma_init=0.0) -- a positive gain must be set to
    # activate. See REE_assembly/docs/architecture/soft_competitive_disinhibition_settling.md
    # and claims MECH-140 / MECH-450.
    use_soft_competitive_settling: bool = False
    soft_competitive_settling_rounds: int = 3           # R: mutual-inhibition rounds per tick
    soft_competitive_settling_gain: float = 0.0         # inhibition strength (0.0 = no-op even when ON)
    soft_competitive_settling_temperature: float = 1.0  # T: softmax temperature for the graded support
    soft_competitive_settling_cross_class: float = 0.25  # K_ij across first-action classes (within-class = 1.0)

    # ============================================================== #
    # ARC-110: parallel segregated cortico-BG-thalamic loops          #
    # (motor / associative-cognitive-set / limbic-motivational) + an  #
    # in-layer (eligibility/settling-field) committed-class null (S2). #
    # ARC-109 (D1/D2 population split) + MECH-452 (loop-local          #
    # eligibility traces) are the coupled co-requisites, built here.   #
    # ============================================================== #
    #
    # The V3 single E3 selection arena collapses the dACC/lPFC/OFC modulatory
    # channels into ONE F-dominated within-eligible argmin, so (a) F monopolises
    # ~88-89% of committed-selection variance (MECH-439) and (b) any same-layer
    # null is structurally inert on the committed-class DV (700b/704b/706b
    # autopsies). ARC-110 replaces that within-eligible arbitration with N>=3
    # parallel segregated loops: each loop runs its OWN within-loop competition
    # (its channel subset + optional per-loop settling + ARC-109 Go/No-Go
    # populations) FIRST, then cross-loop arbitration AFTER, so F can dominate
    # only the MOTOR loop and cannot drown the limbic "is this worth committing
    # to" value. Safety is unchanged: the loops arbitrate strictly WITHIN the
    # F-bounded MECH-448/449 eligible set (a No-Go-suppressed candidate is never
    # in the set), so a non-motor loop can FLIP the within-eligible winner but
    # can NEVER re-admit a suppressed candidate. Each loop's preference is
    # NORMALISED (zscore over the eligible set) before arbitration -- that
    # normalisation is what strips F's raw magnitude advantage (the conversion
    # mechanism). Default False -> the legacy single-arena within-eligible path
    # runs UNCHANGED (bit-identical OFF). Functional translation of the
    # Alexander/DeLong/Strick parallel-loop organisation (ARC-106; NOT anatomical
    # mimicry) integrated by Haber's ascending dopamine spiral. See
    # REE_assembly/docs/architecture/sd_v4_loop_segregation.md.
    use_loop_segregation: bool = False
    # Channel-name -> loop-name assignment. Empty -> the built-in default map in
    # e3_selector (_LOOP_DEFAULT_CHANNEL_MAP): dACC/lPFC -> associative;
    # OFC/curiosity(mech314)/liking(mech295)/vigour(mech320) -> limbic; everything
    # else -> loop_segregation_default_loop. The motor loop is F (raw_scores)
    # itself, not a modulatory-channel set.
    loop_segregation_channel_map: Dict[str, str] = field(default_factory=dict)
    loop_segregation_default_loop: str = "associative"  # unmapped modulatory channels
    # Haber ascending striato-nigro-striatal spiral coupling (limbic -> assoc ->
    # motor): the cross-loop arbitration weights each NON-motor loop's normalised
    # preference contributes to the committed score. motor_authority weights the
    # F (motor) loop. At all == 1.0 every loop contributes its zscore equally
    # (F no longer dominant) -- the ARC-110 conversion hypothesis.
    loop_segregation_spiral_gain_assoc: float = 1.0
    loop_segregation_spiral_gain_limbic: float = 1.0
    loop_segregation_motor_authority: float = 1.0
    # Per-loop preference normalisation before cross-loop arbitration. "zscore"
    # (default-when-ON) standardises each loop's within-eligible preference to
    # mean 0 / std 1 so F's raw magnitude carries no advantage; "range" scales to
    # unit range; "none" leaves raw (F re-dominates -- a degenerate control).
    loop_segregation_normalize: str = "zscore"
    # S2: in-layer (same-layer) committed-class null. When True, each NON-motor
    # loop's accumulator (the eligibility/settling field the per-loop settling
    # acts on) is REPLACED by a magnitude-matched random-structure perturbation
    # (gaussian, rescaled so its range == alpha * the real loop accumulator
    # range). Because it perturbs the SAME layer the loops settle on -- NOT policy
    # softmax temperature (the decoupled 700-lineage null) -- it can actually move
    # the committed-class DV, so noise_verified_lifting becomes a MEANINGFUL
    # non-vacuity precondition. Motor (F) is never nulled (it is the thing
    # conversion is tested against). Selection-only: writes nothing to memory
    # (MECH-094 not engaged). Default False -> no null injected.
    loop_segregation_noise_on: bool = False
    loop_segregation_noise_alpha: float = 1.0          # magnitude-match multiplier vs real loop range
    # ARC-110 C2 RELEASE (per-named-channel routing, 2026-06-28). The named
    # cortical bias HEADS (OFC/dACC/lPFC/vigour/liking) emit per-candidate-FLAT
    # output (their input representation carries cross-candidate range but their
    # consuming Linear collapses it -- the MECH-191 phasic-externalisation gap),
    # so under per-loop zscore a flat channel is inert and the limbic loop carries
    # no per-candidate competition (V3-EXQ-707: ARM_DROP_LIMBIC byte-identical to
    # A1_LOOPS). When True (and loop segregation + finer-channel gating on), each
    # named channel's loop-arbitration term is sourced from its per-candidate
    # REPRESENTATION routed through the parameter-free, range-preserving
    # project_channel_range projection (the SAME GAP-A path that keeps the lumped
    # `route` channel phasic) INSTEAD of its flattened bias-head scalar. This
    # affects ONLY _segregated_loop_arbitrate's view of the named channels -- the
    # _lcg_terms eligibility traces, the authority/shortlist _modulatory_accum
    # recompose, and the F/score commit path are all UNCHANGED. Selection-only;
    # writes nothing to memory (MECH-094 not engaged). Default False -> the loop
    # accumulates the flat bias scalars exactly as before (bit-identical OFF).
    use_named_channel_routing: bool = False

    # ARC-109: D1/D2 striatal population split with asymmetric dopamine gain.
    # When True (and loop segregation on), each loop's within-loop preference is
    # decomposed into TWO opponent populations instead of one additive scalar:
    # a Go/D1 population (the promote side: relu(-accum), reduces cost) potentiated
    # by dopamine (gain 1 + d1_da_gain * da) and a No-Go/D2 population (the suppress
    # side: relu(+accum), raises cost) DEPRESSED by dopamine (gain 1 - d2_da_gain *
    # da). The loop preference (cost, lower=better) = D2_activity - D1_activity.
    # This preserves a representational distinction the additive scalar destroys:
    # high-Go+high-No-Go (approach-avoidance CONFLICT) is dissociable from
    # low-Go+low-No-Go (indifference) -- the substrate the OCD/Parkinson/dyskinesia
    # CSTC disorder axis needs (ARC-106 EARNS). `da` is the bounded tonic-DA proxy
    # (the e3 value baseline V-hat_t, tanh-squashed). At da==0 and gains==1.0 the
    # net D2 - D1 == +accum - (-accum)... no: net == relu(accum) - relu(-accum) ==
    # accum exactly, so the split is bit-identical to the additive accum at da==0
    # (non-degeneracy comes from da != 0). Default False.
    use_d1_d2_population_split: bool = False
    d1_da_gain: float = 1.0                             # D1/Go LTP potentiation by DA
    d2_da_gain: float = 1.0                             # D2/No-Go LTD depression by DA

    # MECH-452: loop-local eligibility traces under a globally-broadcast dopamine
    # signal. When True (and loop segregation on), the single ARC-108/MECH-451
    # eligibility trace is PARTITIONED into one trace per loop, each armed and
    # credited independently: the shared signed-RPE delta_t updates each loop's
    # channels via that loop's OWN local trace, so credit stays loop-local even
    # though the teaching signal is one broadcast. A loop that did not vote for the
    # committed action earns no credit. Prevents a smeared trace from making
    # learned gating appear ineffective when the rule is correct. Default False ->
    # the single shared trace is used (bit-identical OFF).
    use_loop_local_eligibility_traces: bool = False

    # ARC-108 x ARC-110 COUPLING: LEARNED (dopamine-gated) CROSS-LOOP arbitration.
    # ------------------------------------------------------------------------ #
    # The named next attack on the F-dominance conversion ceiling (MECH-439)
    # after V3-EXQ-707b. 707b built ARC-110 loop segregation fully-live (all 6
    # non-degeneracy gates passed; limbic loop carried per-candidate range 1.414)
    # yet the limbic loop NEVER won: committed-class entropy A1_LOOPS 0.838 ~
    # A0_SINGLE_ARENA 0.914. The autopsy (failure_autopsy_V3-EXQ-707b_2026-06-29)
    # traced this to the CROSS-LOOP combine being STATIC ARITHMETIC -- the fixed
    # spiral gains (loop_segregation_spiral_gain_assoc/limbic +
    # loop_segregation_motor_authority) sum the per-loop normalised preferences
    # with constant weights, so the combine still inherits F's static dominance;
    # it cannot LEARN to let the limbic "is this worth committing to" value
    # override the motor/F loop. ARC-108 supplies learned WITHIN-arena gating
    # (w_chan) and MECH-450 the within-loop settling (W_lat); this flag adds the
    # missing piece the 2026-06-29 MECH-439 claim-synthesis named as the
    # ARC-108 x ARC-110 intersection: DA-gated three-factor plasticity operating
    # AT the cross-loop arbitration itself.
    #
    # Mechanism (full [3,3] cross-loop matrix, user-chosen 2026-07-01). The static
    # combine  final = m_a*motor_z + g_a*assoc_z + g_l*limbic_z  is replaced by a
    # LEARNED cross-loop matrix  W_cross = I + M_cross  ([3,3], loop order
    # motor/associative/limbic):  eff = W_cross @ [motor_z; assoc_z; limbic_z] ;
    # final = m_a*eff_motor + g_a*eff_assoc + g_l*eff_limbic . M_cross is a
    # register_buffer init 0 -> W_cross = I -> eff == the per-loop zscores -> final
    # is BIT-IDENTICAL to the static combine at init (and OFF -> the static combine
    # runs untouched). M_cross[i,j] is the learned directed influence of loop j on
    # loop i's effective preference; the diagonal is per-loop self-gain and the
    # off-diagonal is the ascending striato-nigro-striatal spiral (Haber) -- e.g.
    # M_cross[motor, limbic] is the learnable path by which the limbic value loop
    # comes to drive the motor commit. M_cross is SIGNED (a loop may invert
    # another's preference, mirroring the signed W_lat precedent) and is learned by
    # the SAME ARC-108 signed-RPE three-factor rule as w_chan/W_lat -- one shared
    # dopaminergic delta_t = R_t - V-hat_t / D1-D2 asymmetry (Haber's single
    # ascending spiral) -- via a standard outer-product Hebbian co-activation trace
    # coact[i,j] = post_i * pre_j, where pre_j = -loop_z_j[committed] (SIGNED: >0
    # when loop j preferred the committed candidate) and post_i = -eff_i[committed].
    # Delta M_cross = eta * delta_t * asym * coact_trace. Waking-only (a
    # replay/DMN simulation tick forms no delta_t and writes no M_cross; MECH-094).
    # SAFETY is unchanged: the arbitration runs STRICTLY within the F+MECH-448/449
    # eligible set, so a learned weight can reorder within-eligible candidates but
    # can NEVER re-admit a No-Go-suppressed candidate -- the orthogonal-to-F safety
    # guarantee is inherited from the envelope regardless of the weights. No
    # autograd (register_buffer + no_grad local update). Couples with MECH-450: the
    # per-loop settling shapes each loop's within-loop competition BEFORE the
    # cross-loop weights arbitrate across the settled loops, and both learned
    # objects ride the same shared delta_t in one post_action_update. NOTE (ARC-106
    # divergence, logged in the design doc): the forward map is linear, so the
    # committed selection depends on the effective column weights
    # w_eff[j] = sum_i gain_i * W_cross[i,j]; the [3,3] matrix's value is the
    # DIRECTED cross-loop credit structure it learns, not forward expressivity.
    # Default False -> bit-identical OFF. Requires use_loop_segregation on to act.
    use_learned_cross_loop_arbitration: bool = False
    learned_cross_loop_eta: float = 0.01               # M_cross three-factor learning rate
    # (elig_decay / asym_potentiation / asym_depression / rpe_mode / value_baseline_beta
    #  are SHARED with the ARC-108 learned_channel_* knobs -- one dopamine system.)

    # ARC-110 x ARC-108 ASCENDING-SPIRAL GAIN (V3-EXQ-709/710 loop-effective-weight
    # ceiling repair). The 709/710 autopsies found the learned cross-loop matrix
    # ENGAGES (M_cross moves off init) yet the ascending path M_cross[motor,limbic]
    # peaks ~0.03 -- functionally too weak to lift a non-motor (limbic) loop to the
    # motor loop's effective column weight, so limbic_loop_can_win met on only 1/4
    # divergent seeds. Biology (Haber 2000): the striato-nigro-striatal spiral is
    # anatomically ASYMMETRIC -- ascending (limbic -> associative -> motor) influence
    # is the developmentally-strengthened, load-bearing direction. Two knobs, both
    # scaling ONLY the ascending (upper-triangular row<col, in the motor/assoc/limbic
    # ordering) entries of M_cross, so w_eff[limbic] rises WITHOUT raising w_eff[motor]
    # (the motor column is diagonal + descending, un-amplified) -- this both
    # strengthens the ascending coupling AND implicitly de-pins the motor(F) default.
    #   1. forward gain: W_cross = I + (G_fwd .* M_cross), G_fwd upper-tri = spiral_gain
    #      -- the ANATOMICAL ascending-projection strength (untuned=1.0 in the 709/710
    #      substrate, functionally too weak). Keeps the forward map LINEAR (still a
    #      constant elementwise scaling of M_cross) so w_eff-collapsibility (CLA-3) and
    #      bit-identical-at-init both hold.
    #   2. plasticity maturation gain: the M_cross three-factor UPDATE's ascending
    #      entries are scaled by plasticity_gain -- the ascending SPIRAL MATURATION
    #      RATE (ascending credit accrues faster than descending, mirroring the
    #      developmental asymmetry). eta stays the base rate; this is the directional
    #      multiplier on ascending plasticity only.
    # F still fully owns the MOTOR loop (motor_pref unchanged); the gain only stops F
    # from drowning the limbic "is this worth committing to" value. Safety unchanged:
    # arbitration stays STRICTLY within the F+MECH-448/449 eligible set. Default
    # False / gains 1.0 -> BIT-IDENTICAL OFF (G matrices become all-ones; at init
    # M_cross==0 so gain*0==0 -> W_cross==I regardless of gain). Requires
    # use_learned_cross_loop_arbitration (hence use_loop_segregation) on to act.
    use_ascending_spiral_gain: bool = False
    loop_segregation_ascending_spiral_gain: float = 1.0       # forward W_cross ascending-entry gain (anatomical strength)
    loop_segregation_ascending_plasticity_gain: float = 1.0   # ascending-entry update gain (spiral maturation rate)

    # BOUNDED ascending-spiral gain -- target-PARITY controller (V3-EXQ-711 repair,
    # 2026-07-04). The raw scalar above has NO stable parity regime: sub-threshold
    # (709, coupling ~0.03, limbic never wins) or runaway (711, 20x-fwd x 5x-plasticity
    # compounding through the positive-feedback plastic loop -> M_cross range peak 4897.8,
    # w_eff[limbic] 10-2274x w_eff[motor] -- a limbic MONOPOLY that merely replaces the
    # F/motor-pinning). confirmed failure_autopsy_V3-EXQ-711_2026-07-04. This controller
    # replaces the unbounded multiply with actuator-saturated setpoint control:
    #   1. FORWARD (parity-ceiling): a per-step gain in [0, parity_forward_gain] on the
    #      ascending upper-tri M_cross entries, solved so the limbic effective column
    #      weight w_eff[limbic] is LIFTED toward but HARD-CAPPED at
    #      parity_ceiling_ratio * w_eff[motor]. The motor column has no upper-tri entries
    #      so w_eff[motor] is gain-invariant (the fixed parity reference). Bounds the
    #      w_eff[limbic]/w_eff[motor] RATIO -> a fair within-eligible reorder, never a
    #      monopoly. Applied via _ascending_gain_matrix -> map stays linear, bit-identical
    #      at init (M_cross==0 -> W_cross==I for any gain).
    #   2. MATURATION (bounded loop): the ascending three-factor update is scaled by the
    #      BOUNDED parity_plasticity_gain, then the ascending M_cross entries are clamped
    #      to [-m_cross_clamp, m_cross_clamp] -- an anti-windup clamp that stops the
    #      plastic positive-feedback loop from running away (the second 711 runaway source).
    # Takes PRECEDENCE over use_ascending_spiral_gain when both are on (the raw path is
    # retained only for 709/711 reproducibility). Safety unchanged: reorders STRICTLY
    # within the F+MECH-448/449 eligible set, never re-admits a No-Go-suppressed candidate.
    # Master switch False -> BIT-IDENTICAL OFF; sub-params default inert (forward_gain 1.0
    # = no lift, ceiling_ratio 0.0 / m_cross_clamp 0.0 = disabled) so the falsifier's ON
    # arm configures them explicitly. Requires use_learned_cross_loop_arbitration on to act.
    use_ascending_parity_controller: bool = False
    loop_segregation_parity_forward_gain: float = 1.0         # ascending lift strength g_raw (>=1 lifts; ceiling caps)
    loop_segregation_parity_ceiling_ratio: float = 0.0        # cap w_eff[limbic] <= ratio * w_eff[motor]; 0.0 = disabled
    loop_segregation_parity_plasticity_gain: float = 1.0      # bounded ascending maturation rate (paired with the clamp)
    loop_segregation_m_cross_clamp: float = 0.0               # |ascending M_cross| bound post-update; 0.0 = disabled

    # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a, 2026-06-10):
    # The 2026-06-03/06-06 authority rescales _modulatory_accum (the composed
    # score_bias chain + MECH-341 bonus). The 569f/661/654a cluster showed that a
    # channel whose REPRESENTATION carries genuine cross-candidate range
    # (world-summary spread 0.196; minted rule_state; coherence) still does not move
    # the committed argmax, because that range is flattened by the consuming bias
    # head before it reaches _modulatory_accum -- so the authority has nothing to
    # amplify. use_modulatory_channel_routing folds a parameter-free,
    # range-preserving projection of the channel-under-test's per-candidate
    # representation into _modulatory_accum BEFORE the authority's range
    # computation, guaranteeing the channel's range is ROUTED into the bias term the
    # authority rescales (extends the necessary-but-not-sufficient note one link).
    # P0 readiness gate: modulatory_channel_route_range (a diagnostic = range of the
    # routed bias, measured pre-rescale) lets a retest assert the modulatory bias
    # ITSELF carries cross-candidate range derived from the channel under test before
    # any behavioural falsifier is scored. Default False (bit-identical OFF).
    use_modulatory_channel_routing: bool = False
    modulatory_channel_route_min_range_floor: float = 1e-6
    modulatory_channel_route_weight: float = 1.0

    # modulatory-bias-selection-authority AMEND (CONVERSION, 569g/682, 2026-06-15):
    # V3-EXQ-682 (no_collapse_reproduced) confirmed the route-range amend SOLVED
    # REACH (ARM_1 applied in-arm route_range ~0.20, all upstream-collapse causes
    # ruled out incl seed 43); V3-EXQ-569g showed the residual gap is CONVERSION --
    # the gap-relative ADDITIVE authority at gain 0.5 (modrange = 0.5*raw_score_range)
    # is subdominant to the F-dominated primary (88-89% of E3 variance, V3-EXQ-571),
    # so the routed range flips only near-tie OUTLIERS, not near-decisive winners
    # (569g 1/3 seeds strict-above matched-noise). Two no-op-default conversion
    # levers let the routed range MOVE the committed argmax. See
    # failure_autopsy_V3-EXQ-569g_2026-06-14 + behavioral_diversity_isolation:GAP-A.
    #
    # (a) normalize_basis -- how the additive authority anchors its target range.
    #     "range" (default, legacy bit-identical): target = gain * raw_score_range
    #     (outlier-sensitive -> only near-tie). "std": target = gain * raw_score_std,
    #     rescaling by the modulatory STD (robust to outliers) so the structured
    #     channel competes against the TYPICAL primary spread (near-decisive
    #     candidates), not just the two outliers that set the range. gain stays
    #     sweepable. NOTE: additive gain >= 1.0 breaks the safety bound (modulatory
    #     can override a clearly-harmful rejection); keep gain < 1.0 on the additive
    #     path for the shipped config, or use the shortlist lever (b), which
    #     preserves safety at any internal strength.
    modulatory_authority_normalize_basis: str = "range"
    #
    # (b) shortlist-then-modulate -- the pre-registered architectural fallback. When
    #     True, F (raw primary scores) filters to a near-tie set (candidates within
    #     modulatory_shortlist_margin * raw_score_range of the best raw score), then
    #     the modulatory channel (_modulatory_accum) ARBITRATES the winner WITHIN
    #     that set. The structured channel is load-bearing within the shortlist
    #     without out-magnitude-ing F globally, and SAFETY is preserved at any
    #     internal strength (clearly-harmful candidates outside the shortlist are
    #     never selectable). Takes precedence over the additive-authority rescale +
    #     argmin/stratified selection at the selection site when enabled.
    #     Default False (bit-identical OFF).
    use_modulatory_shortlist_then_modulate: bool = False
    modulatory_shortlist_margin: float = 0.25
    #     TOP-K shortlist mode (569h-autopsy conversion amend, 2026-06-16). The
    #     V3-EXQ-684 margin shortlist admitted ~7/8 candidates
    #     (modulatory_shortlist_size_mean 6.25-8.54) = a near-whole, state-STABLE
    #     eligible set, so the committed-branch argmin collapsed to the modulatory
    #     channel's global favorite (ARM_SHORTLIST entropy 0.337 < proposer 0.549).
    #     "top_k" instead shortlists the k F-best candidates by primary score (k
    #     smallest raw_scores; lower-is-better). A SMALL fixed k gives an eligible
    #     set whose MEMBERSHIP rotates with state, so the within-set argmin of the
    #     routed modulatory channel converts the per-candidate range into
    #     committed-action diversity (and beats unstructured matched-noise = the
    #     C_R1B non-vacuity bar). Safety is preserved: only the k F-best are
    #     selectable, so clearly-harmful candidates are never eligible. "margin"
    #     (default) is the legacy bit-identical path.
    modulatory_shortlist_mode: str = "margin"
    modulatory_shortlist_k: int = 3
    #
    # CONVERSION-CEILING / F-DOMINANCE conflict-grade amend (MECH-439, 2026-06-18).
    # Two RENDERINGS OF ONE PRINCIPLE -- the BG hyperdirect conflict-grade: grade the
    # committed decision by the normalized top-F gap (gap_norm in [0,1], from raw_scores
    # / raw_score_range). F monopolises ~88-89%% of E3 committed-selection variance
    # (V3-EXQ-571), so a FIXED top-k=3 shortlist + a HARD argmin let the F-dominated
    # winner reassert except at near-ties. These two no-op-default levers grade BOTH the
    # eligibility width and the commit decisiveness by the per-tick F-gap.
    #
    # FACTOR A -- conflict-graded shortlist width (the safety-hard primary). When
    #   modulatory_shortlist_conflict_graded is True (and mode == "top_k"), the fixed k
    #   is replaced by k = clamp(round(k_max - (k_max-1)*gap_norm), 1, K): near-ties
    #   (small gap) -> wider k / slower commit (the STN threshold-raise); a decisive
    #   F-gap -> k -> 1 / fast commit. F still gates ELIGIBILITY only; it is ABSENT from
    #   the within-set arbitration (the routed modulatory channel argmin/sample picks
    #   inside the eligible set). SAFETY: because the eligible set is the k F-best, a
    #   clearly-harmful candidate (large F-gap above the best) is never admitted. Default
    #   False -> the fixed modulatory_shortlist_k path is bit-identical.
    modulatory_shortlist_conflict_graded: bool = False
    modulatory_shortlist_k_max: int = 6
    #
    # FACTOR B -- gap-scaled entropy-regularized commit (the complement). The committed
    #   selection is otherwise a HARD argmin over the F-dominated scores (or the routed
    #   modulatory channel within a shortlist). When use_gap_scaled_commit_temperature is
    #   True, the committed pick becomes multinomial(softmax(-q / T_eff)) over the
    #   eligible set, with T_eff = base_temperature + gap_scaled_commit_entropy_alpha *
    #   (1 - gap_norm): near-ties -> hotter (softer argmax); a decisive gap -> cold
    #   (T_eff -> base, preserves the decisive F-winner). q is the routed modulatory
    #   channel within an active Factor-A shortlist (F-bounded eligible set = the safety
    #   guarantee), else the F-dominated scores restricted to an F-eligibility envelope
    #   (candidates within gap_scaled_commit_harm_floor * raw_score_range of the best raw
    #   score) so a hot commit-T in a near-tie can NEVER softmax-promote a clearly-harmful
    #   candidate. This softens the COMMITTED hard-argmin (where the monostrategy lives);
    #   distinct from MECH-313 tonic-noise (gap-blind, pre-select) and from the existing
    #   uncommitted multinomial. A FLAT (gap-blind) commit-T reduces to the 569g
    #   temperature control that under-lifted -- the (1 - gap_norm) gap-scaling is
    #   load-bearing. Default False -> the hard argmin path is bit-identical.
    use_gap_scaled_commit_temperature: bool = False
    gap_scaled_commit_entropy_alpha: float = 1.0
    gap_scaled_commit_harm_floor: float = 0.25

    # MECH-448 / ARC-107 -- rank-preserving F->eligibility demotion (LEAD lever of
    # the basal-ganglia E3-selector constitution, 2026-06-20). The pallidal-permission
    # reading of the conversion ceiling: F (the primary harm/goal score) decides who is
    # ELIGIBLE to compete, NOT who wins. F monopolises ~88-89%% of E3 committed-selection
    # variance (V3-EXQ-571); the conflict-grade near-tie family (MECH-447 / 689a) is
    # exhausted, so this is the constitutional escalation routed by the 689a autopsy.
    #
    # F is renormalised against the COMPETING FIELD by a rank-preserving divisive-
    # normalisation analog (Carandini & Heeger 2012; value DN, Louie/Khaw/Glimcher 2013):
    #   merit[i]  = clamp(raw_scores.max() - raw_scores[i], min=0)   # lower-cost = higher
    #   pooled    = f_eligibility_dn_sigma + merit.sum()
    #   elig[i]   = merit[i] / pooled                                # share of the field
    #   eligible  = { i : elig[i] >= f_eligibility_envelope_floor }  # absolute share floor
    # The absolute-share floor (NOT a fraction-of-max, which would cancel the pooled term
    # and collapse to the margin shortlist) makes the envelope GRADED + CONFLICT-SCALED +
    # ENV-GENERAL: a candidate must command at least floor of the total competing merit to
    # be eligible, so a decisive F-winner narrows the envelope (others fall below floor)
    # and a near-tie widens it -- the hyperdirect conflict-grade emerging from the field
    # structure, not a hard top-k count. Order-preserving in merit -> RANK-PRESERVING in F
    # (the eligible set is an F-rank prefix). Then the existing _modulatory_accum arbitrates
    # the committed action WITHIN the eligible set with F REMOVED from the final argmin
    # (reuses the shortlist-then-modulate within-set arbitration). Requires a modulatory
    # channel (_modulatory_accum not None) -- with no modulation there is nothing to demote
    # F to, so the block is skipped (bit-identical, legacy F argmin).
    #
    # DIVERGENCE (ARC-106 ledger): canonical DN is ORDER-PRESERVING + POOLED-SYMMETRIC.
    # REE demotes ONLY F and removes it from the commit argmin (rank-ALTERING at COMMIT) --
    # this EXCEEDS canonical DN. LOAD-BEARING divergence (the QD/MAP-Elites justification),
    # validated by the 689a-successor falsifier (NOT queued here).
    # SAFETY: a clearly-harmful candidate has near-zero merit -> near-zero share -> below
    # floor -> excluded, so no global disinhibition (the envelope is itself the F-bound).
    # NON-DEGENERACY: the falsifier must show the envelope actually EXCLUDES on a divergent
    # pool (f_eligibility_excluded_count > 0); an all-admit envelope is a vacuous self-route.
    # Default False -> bit-identical OFF (the legacy argmin / shortlist path is untouched).
    use_f_eligibility_demotion: bool = False
    f_eligibility_envelope_floor: float = 0.30
    f_eligibility_dn_sigma: float = 0.0
    # CHANNEL-ADAPTIVE envelope amend (2026-06-21): the absolute share floor
    # (f_eligibility_envelope_floor=0.30) was tuned to PASS on the GAP-A foraging
    # bank (V3-EXQ-689d). Each downstream channel has a different F-merit
    # distribution, so the SAME fixed floor mis-fires: V3-EXQ-654h (arc_062) had
    # every share below 0.30 -> all-admit no-op (excluded_count==0, the lever
    # never engaged); V3-EXQ-485i (OFC) needed a bespoke per-seed floor
    # recalibration to engage. use_f_eligibility_adaptive_floor replaces the
    # fixed floor with a MEAN-RELATIVE one: a candidate is eligible iff its share
    # of the competing merit exceeds f_eligibility_adaptive_mean_factor times the
    # field's OWN mean share. Mean-relative -> scale-invariant across channel
    # F-distributions (auto-calibrates, no per-channel hand-tuning) AND retains
    # the MECH-448 conflict-grade (a decisive F-winner pulls the mean up so others
    # fall below -> narrow envelope; a near-tie sits near the mean -> wide). On any
    # non-uniform field at least one candidate is below the mean share, so the
    # envelope EXCLUDES (excluded_count > 0) by construction -- collapsing the
    # ~5 per-channel hand-floor dances (654h/485i/485j + pending 625/445/687) into
    # one global knob. Order-preserving in merit -> RANK-PRESERVING in F
    # (eligible set is still an F-rank prefix). Default False -> reads the legacy
    # fixed floor -> bit-identical OFF.
    use_f_eligibility_adaptive_floor: bool = False
    f_eligibility_adaptive_mean_factor: float = 1.0

    # MECH-449 / ARC-107: Go/No-Go eligibility constitution (2026-06-21). The
    # core opponency leg of the basal-ganglia E3-selector constitution -- the
    # bounded Go (eligibility-PROMOTION) + bounded No-Go (eligibility-SUPPRESSION)
    # pressure set governing which candidates may compete for the pallidal-like
    # permission-to-commit gate (the MECH-448 eligibility envelope / shortlist).
    # Generalises MECH-260 (dACC anti-recency No-Go) from a drowned score-bias
    # into an ELIGIBILITY-ACCESS gate (reuse-before-duplicate, ARC-106 G2: the
    # perseveration No-Go axis CONSUMES MECH-260's per-candidate suppression
    # vector; the other axes are genuinely new functions MECH-260 lacks).
    #
    # The gate acts on the eligible set AFTER the MECH-448 envelope / shortlist
    # builds it and BEFORE the within-eligible _modulatory_accum arbitration:
    #   No-Go suppresses a candidate (drops it from the eligible set) when ANY
    #     bounded axis exceeds its floor -- safety / staleness / perseveration /
    #     low-viability -- on an axis ORTHOGONAL to F-rank (rank-preserving
    #     demotion is order-preserving over F and structurally CANNOT exclude an
    #     F-eligible-but-undesirable candidate; only an active No-Go can; this is
    #     exactly what V3-EXQ-689f demonstrates).
    #   Go promotes a candidate that F demoted OUT of the envelope back into the
    #     eligible set when its go-evidence clears the threshold (bounded by
    #     gng_go_max_promote) -- lawful channel-specific ACCESS, not scalar
    #     F-dominance, decides.
    # SAFETY: a No-Go'd candidate is removed from the eligible set, so the
    # within-eligible argmin can never select it regardless of how large its
    # modulatory pull is (the orthogonal-to-F property; contract-verified).
    # FAIL-OPEN guard (gng_protect_min_eligible): No-Go never drops the eligible
    # set below this many survivors UNLESS the survivors are themselves
    # safety-No-Go'd -- guards the No-Go-over-pressure -> catatonia/avolition
    # failure pole (grounding synthesis 2.1/2.5) from deadlocking the gate.
    # Per-candidate signals are supplied via select(go_nogo_signals=...) (each an
    # optional [K] tensor); a missing axis is inert. Default master OFF ->
    # the gate block is skipped entirely -> bit-identical OFF. PROMOTES NOTHING.
    use_go_nogo_constitution: bool = False
    # No-Go axis floors (a candidate is No-Go'd if its signal on that axis
    # crosses the floor in the suppressing direction).
    gng_safety_floor: float = 0.5          # No-Go if safety-undesirability >= floor
    gng_staleness_floor: float = 0.5       # No-Go if staleness >= floor
    gng_perseveration_floor: float = 0.5   # No-Go if recency-share (MECH-260) >= floor
    gng_viability_floor: float = 0.1       # No-Go if viability < floor (LOW-viability)
    # Bounded Go promotion.
    gng_go_threshold: float = 0.5          # Go-promote a demoted candidate if go-evidence >= threshold
    gng_go_max_promote: int = 2            # at most this many Go promotions per tick (bounded)
    # Fail-open: No-Go keeps at least this many candidates eligible unless they
    # are safety-No-Go'd (safety is never overridden by the fail-open).
    gng_protect_min_eligible: int = 1

    # DR-12 (self_model_v4:SELF-4, FIRST V4 substrate build, 2026-06-17):
    # E2 forward prediction-error modulates E3 trajectory-scoring confidence.
    # E3 currently trusts the E2 rollout unconditionally; high E2 forward-PE in a
    # trajectory's region does NOT down-weight that trajectory's viability. This is
    # a NEW lever on EXISTING machinery -- E3 already consumes E1-novelty
    # (_novelty_ema, MECH-111) and world-forward running-variance PE
    # (_running_variance, ARC-016); DR-12 adds an E2-forward-PE confidence
    # down-weight alongside them. When use_pe_confidence_weighting is True AND a
    # per-trajectory e2_forward_pe is supplied to score_trajectory(), the score
    # (a COST; lower is better) gains a positive penalty = pe_confidence_weight *
    # monotone(e2_forward_pe), so a trajectory in a poorly-modelled (high-PE)
    # region is discounted. Threaded per-candidate via select()'s
    # e2_forward_pe_per_candidate so a varying PE can change the committed argmin
    # (a uniform scalar would be argmin-invariant -- the V3-EXQ-571 lesson).
    # Default False (bit-identical OFF). generation:v4, off the V3 critical path,
    # promotes nothing in V3. v1 source is caller-supplied (the DR-12 pilot
    # V4-EXQ-001 is a controlled probe); an ecological region-PE auto-source is the
    # documented follow-on. See docs/architecture/dr12_pe_conditioned_e3_confidence.md.
    use_pe_confidence_weighting: bool = False
    pe_confidence_weight: float = 0.0
    # monotone penalty form: "linear" (penalty = pe) | "saturating"
    # (penalty = 1 - exp(-pe / pe_confidence_scale), bounded [0,1) confidence-deficit).
    pe_confidence_mode: str = "linear"
    pe_confidence_scale: float = 1.0

    # DR-10 (self_model_v4:SELF-3): z_self enters E3 viability scoring. Sibling
    # lever to DR-12 (same machinery). When use_self_viability_weighting is True
    # AND a per-candidate self_viability COST is supplied to score_trajectory(),
    # the score (a COST; lower is better) gains a positive penalty =
    # self_viability_weight * monotone(self_viability), so a trajectory that is
    # LESS viable for THIS agent's current bodily state (capacity/affect/damage,
    # read from the DR-13 stateful z_self) is discounted. Threaded per-candidate
    # via select()'s self_viability_per_candidate so a varying signal can change
    # the committed argmin (a uniform scalar would be argmin-invariant -- the
    # V3-EXQ-571 lesson). Default False (bit-identical OFF). generation:v4, off
    # the V3 critical path, promotes nothing in V3. Depends on the DR-13 stateful
    # z_self (SELF-1) as the SUBJECT of the viability estimate. v1 source is
    # caller/agent-supplied (the DR-10 pilot is a controlled probe); an ecological
    # z_self-derived auto-source (allostatic z_self-deviation x per-candidate
    # demand, or a learned z_self->viability head) is the documented follow-on.
    # See docs/architecture/dr10_z_self_in_e3_viability.md.
    use_self_viability_weighting: bool = False
    self_viability_weight: float = 0.0
    # monotone penalty form: "linear" (penalty = self_viability) | "saturating"
    # (penalty = 1 - exp(-sv / self_viability_scale), bounded [0,1) viability-deficit).
    self_viability_mode: str = "linear"
    self_viability_scale: float = 1.0

    # MECH-440 -- state-conditioned self-annealing PROPAGATING noise floor (NoisyNet
    # learned-parametric-weight-noise analog; Fortunato et al. 2018). The mechanistic
    # refinement of MECH-313's tonic floor. MECH-313 lifts the softmax TEMPERATURE
    # (pre-commit); V3-EXQ-687 found that floor NON-PROPAGATING -- invisible to the
    # committed argmax (selected_action_entropy=0.0, the r1a_entropy_only_artefact).
    # MECH-440 injects factorised-Gaussian WEIGHT NOISE at the E3 selection head as a
    # per-candidate additive bias built from each candidate's first-action vector (a
    # per-candidate state-dependent activation that is differentiated across candidates
    # by construction -- robust to the z_world monostrategy collapse that sank 648/614e):
    #   bias[k] = (sigma_w (x) eps_w) . x_k + sigma_b (x) eps_b     (mu frozen at 0)
    # added INTO _modulatory_accum BEFORE the within-eligible argmin (and into the
    # segregated-loop final), so it (i) PROPAGATES into the committed action, (ii) is
    # STATE-CONDITIONED (the x_k action term modulates the noisy weight per candidate),
    # (iii) SELF-ANNEALS (sigma is scaled by a local confidence EMA -- falls where the
    # committed margin is consistently decisive, holds where near-ties recur). mu is
    # FROZEN at 0 so the head is a PURE exploration-noise injector (isolates the
    # falsifier: any committed-entropy lift is the noise, not a second learned mean
    # pathway). eps resampled per WAKING tick (rotates the within-eligible winner
    # tick-to-tick -> committed selected_action_entropy rises). MECH-094: zero
    # perturbation on a simulation tick. No-op default: flag False -> head not
    # instantiated -> bit-identical; flag True with sigma_init=0.0 -> output exactly 0
    # -> bit-identical (matches the sd stub's "sigma_init=0 = bit-identical").
    #
    # ARC-106 DIVERGENCES (both disclosed, logged in the grounding ledger):
    #   (1) per-parameter sigma is one description-level below biology's systems-level
    #       tonic/phasic mode gate (already in the decision record);
    #   (2) sigma self-anneals via REE's LOCAL confidence EMA, NOT NoisyNet's RL
    #       gradient -- REE does not backprop through E3 selection (w_chan above is the
    #       same local-update, never-autograd precedent).
    # See docs/architecture/state_conditioned_exploration_noise_floor.md (#mech-440).
    use_noisy_selection_head: bool = False
    noisy_selection_sigma_init: float = 0.0
    # overall scale on the per-candidate noisy bias before it enters _modulatory_accum.
    noisy_selection_weight: float = 1.0
    # local self-annealing: sigma_eff = sigma * (anneal_floor + (1-anneal_floor) *
    # ema(1 - gap_norm)). gap_norm ~ 0 (near-tie) -> keep full noise; gap_norm ~ 1
    # (decisive) -> decay toward anneal_floor. anneal_ema_alpha is the EMA rate on the
    # confidence signal. Set noisy_selection_anneal=False to freeze sigma at sigma_init
    # (the no-anneal control arm).
    noisy_selection_anneal: bool = True
    noisy_selection_anneal_floor: float = 0.1
    noisy_selection_anneal_ema_alpha: float = 0.01

    # MECH-441 -- model-disagreement directed curiosity (RND / Plan2Explore analog;
    # Burda 2018 / Sekar 2020). The complementary directed-curiosity leg of ARC-065
    # substrate (b). A per-candidate intrinsic signal from E2 forward-model
    # DISAGREEMENT (cross-head variance of a small K-head ensemble of E2.world_forward
    # delta predictors), supplied per-candidate to select() via
    # model_disagreement_per_candidate and added INTO _modulatory_accum as a BONUS
    # (a COST reduction; lower score = preferred), so it PROPAGATES per-candidate --
    # unlike the broadcast-novelty EMA channel found non-propagating (V3-EXQ-590a/141b).
    # Self-annealing is INTRINSIC to disagreement (Plan2Explore): as the ensemble
    # learns, cross-head variance -> 0 and the bonus vanishes. NON-DEGENERACY: a flat
    # cross-candidate disagreement cannot change the argmin, so the falsifier asserts a
    # supra-floor model_disagreement_range. Default False / weight 0 -> bit-identical.
    # Falsifier HELD (blocked_substrate) gated on ARC-110 validation V3-EXQ-707 (the
    # single-arena collapse, not the curiosity channel, is the binding constraint per
    # failure_autopsy_704b-706b-conversion-ceiling_2026-06-27). MECH-094: waking-only,
    # no_grad read. See docs/architecture/state_conditioned_exploration_noise_floor.md
    # (#mech-441).
    use_model_disagreement_curiosity: bool = False
    model_disagreement_weight: float = 0.0
    # monotone bonus form on the per-candidate disagreement variance: "linear"
    # (bonus = var) | "saturating" (bonus = 1 - exp(-var / scale), bounded).
    model_disagreement_mode: str = "linear"
    model_disagreement_scale: float = 1.0


@dataclass
class EventSegmenterScaleConfig:
    """Per-scale configuration for MECH-288 EventSegmenter.

    A scale defines a single boundary detector operating on a chosen set of
    latent streams with a chosen algorithm. Two scales (fast + slow) form the
    canonical hierarchical segmenter.
    """
    name: str = "fast"
    # Streams this scale watches; boundary decision is computed over the
    # concatenated / pooled signal from these stream names.
    streams: tuple = ("z_world", "z_self")
    # Algorithm: "pe_threshold" or "bocpd_gaussian".
    algorithm: str = "pe_threshold"
    # Minimum segment length (in ticks) before the detector is allowed to
    # fire again. Prevents one-tick boundary bursts.
    min_segment_length: int = 2
    # tau is an algorithm-hint (e.g. characteristic timescale for the scale);
    # present for API uniformity across detectors.
    tau: int = 2
    # PE-threshold params (unused by bocpd_gaussian).
    pe_threshold: float = 0.65
    pe_window_length: int = 200
    # BOCPD-Gaussian params (unused by pe_threshold).
    hazard: float = 1.0 / 40.0
    posterior_threshold: float = 0.5
    bocpd_top_k: int = 20
    bocpd_prior_var: float = 1.0


@dataclass
class EventSegmenterConfig:
    """MECH-288 hierarchical event-segmenter configuration.

    Canonical default is a two-level segmenter:
      fast: pe_threshold over (z_world, z_self) at tau=2, min_segment_length=2
      slow: bocpd_gaussian over (z_goal,) at hazard=1/40, posterior_threshold=0.5,
            min_segment_length=15
    Segment IDs are hierarchical ("{outer}.{inner}"): slow fire increments the
    outer index and forces an inner reset to 0; fast fire increments the inner
    index.

    Backward compatible: consumer code must gate on HippocampalConfig.use_event_segmenter.
    """
    scales: list = field(default_factory=lambda: [
        EventSegmenterScaleConfig(
            name="fast",
            streams=("z_world", "z_self"),
            algorithm="pe_threshold",
            min_segment_length=2,
            tau=2,
            pe_threshold=0.65,
            pe_window_length=200,
        ),
        EventSegmenterScaleConfig(
            name="slow",
            streams=("z_goal",),
            algorithm="bocpd_gaussian",
            min_segment_length=15,
            tau=40,
            hazard=1.0 / 40.0,
            posterior_threshold=0.5,
            bocpd_top_k=20,
            bocpd_prior_var=1.0,
        ),
    ])
    emit_to: tuple = ("mech_287_broadcast", "mech_269_anchor_set")
    scale_id_format: str = "{outer}.{inner}"
    slow_scale_name: str = "slow"


@dataclass
class InvalidationTriggerConfig:
    """MECH-287 broadcast invalidation trigger configuration (Phase 2 iv).

    Verdict-3 architectural commitment: the trigger is a BoundaryEvent
    subscriber (no independent comparator). Consumer of MECH-288's
    boundary queue; emits BroadcastEvent objects consumed by downstream
    MECH-269 anchor-reset (T3) and MECH-284 staleness accumulator (Phase
    3) wiring.

    Fields:
      gain:            multiplier applied to each BoundaryEvent posterior
                       (broadcast_strength = posterior * gain). No binary
                       thresholding on strength; gate is whole-broadcast
                       via the tonic guardrail.
      targets:         list of symbolic target names the broadcast is
                       addressed to. Default ("mech_269_anchor_set",) is
                       the planned T3 consumer subscription point.
      tonic_threshold: if the rolling-mean tonic estimate (measured BEFORE
                       the current tick) exceeds this value, the whole
                       tick's phasic broadcast is suppressed.
      tonic_window:    number of past ticks used for the rolling mean. A
                       longer window makes the tonic estimate slower-moving
                       and more tolerant of transient bursts.

    Backward compatible: consumer code must gate on
    HippocampalConfig.use_invalidation_trigger.
    """
    gain: float = 1.0
    targets: tuple = ("mech_269_anchor_set",)
    tonic_threshold: float = 0.5
    tonic_window: int = 50


@dataclass
class AnchorSetConfig:
    """MECH-269 Phase 2 (ii) anchor-set configuration.

    Scale-tagged anchor keys (scale, segment_id, stream_mixture) where
    stream_mixture is a Phase 2 stand-in: the tuple of per-stream V_s keys
    active at anchor-creation tick. Learned attribution head is deferred
    to Phase 3.

    Anchor reset policy (dual-trace, Bouton 2004): on reset_region the
    existing anchor is marked inactive (NOT erased); a parallel active
    anchor is installed for the new segment_id. Hysteresis: an anchor
    only resets once its V_s_anchor (V_s minus staleness proxy) has been
    below reset_threshold for hysteresis_k consecutive HippocampalModule
    ticks (Wills 2005 hard-switch + Colgin 2008 soft-reweight dual
    support, per V_s foundation lit-pull).

    Phase 2 (ii) stand-ins (flagged in claim notes; will evolve in Phase 3):
      stream_mixture: tuple(sorted(per_stream_vs.keys())) at write-time.
      Staleness: local tick-delta since last get_anchor / write; MECH-284
        staleness accumulator replaces this in Phase 3.
      Subscription source: drains MECH-288 BoundaryEvent queue. MECH-287
        BroadcastEvent consumption wires in when T3 lands.

    Fields:
      scales:            tuple of scale names this anchor set indexes (must
                         match EventSegmenterScaleConfig.name entries).
      reset_threshold:   V_s_anchor strict-less-than this -> reset-eligible.
      hysteresis_k:      consecutive below-threshold ticks required to fire
                         mark_inactive + new-active install.
      staleness_rate:    per-tick increase in the Phase 2 staleness proxy
                         subtracted from V_s to form V_s_anchor.
      staleness_clip:    maximum staleness contribution (prevents runaway).
      max_anchors_per_scale: soft cap per scale (FIFO on active anchors
                         when exceeded; inactive anchors retained).
      subscribe_to_boundary_events: drain MECH-288 BoundaryEvents on tick
                         (Phase 2 path).
    """
    scales: tuple = ("fast", "slow")
    reset_threshold: float = 0.3
    hysteresis_k: int = 5
    staleness_rate: float = 0.005
    staleness_clip: float = 1.0
    max_anchors_per_scale: int = 128
    subscribe_to_boundary_events: bool = True
    # SD-039 dual-trace anchor goal-snapshot payload (default OFF).
    # When True, write_anchor / mark_inactive / reset_region accept a
    # non-None goal_payload and attach it to the anchor; module-level
    # callers populate the payload from GoalState / VALENCE_WANTING /
    # amygdala arousal tags. With flag OFF, callers pass goal_payload
    # =None and behaviour is bit-identical to pre-SD-039. The flag is
    # carried on AnchorSetConfig (rather than HippocampalConfig)
    # because the substrate-side dataclass + query helper live on
    # AnchorSet; module-level write-site wiring is a follow-on session.
    use_sd039_anchor_payload: bool = False


@dataclass
class StalenessAccumulatorConfig:
    """MECH-284 Phase 3 staleness-accumulator configuration.

    Region-indexed scalar accumulator. Integrates MECH-287 BroadcastEvents
    with an attribution_weight credit assignment over the active anchor set
    and leaks per tick. Read as the staleness term in MECH-269's
    V_s_anchor = V_s(r) - staleness[r] hysteresis check (online read-out).
    The MECH-285 offline sleep-priority read-out is deferred.

    Fields:
      leak_factor:       per-tick multiplicative decay applied to all
                         region entries. 0.995 ~= 200-tick half-life.
      attribution_mode:  "equal" (default) divides strength uniformly
                         across the active anchor set; "stream_overlap"
                         weights each anchor by |set(source_sources) &
                         set(stream_mixture)| / max(|source_sources|, 1)
                         -- a cheap cosine-similarity surrogate over
                         stream-name sets.
      staleness_clip:    maximum per-region staleness. Matches AnchorSet
                         proxy range so V_s_anchor math stays in [-1, 1].
      drop_epsilon:      entries below this after leak_factor are removed
                         from the map to keep snapshot() bounded.

    Backward compatible: consumer code gates on
    HippocampalConfig.use_staleness_accumulator.
    """
    leak_factor: float = 0.995
    attribution_mode: str = "equal"   # "equal" | "stream_overlap"
    staleness_clip: float = 1.0
    drop_epsilon: float = 1e-6


@dataclass
class GhostGoalBankConfig:
    """MECH-292 ranked ghost-goal bank configuration.

    Derived view over the SD-039 dual-trace anchor pool. Per the MECH-292
    spec the bank carries no separate persistent store: each rank() call
    walks the existing anchor pool, scores each anchor on a four-term
    composite, and returns a sorted view. The anchor pool itself remains
    the source of truth.

    Ranking formula (per anchor a clearing goal_match_floor):

        wanting        = a.goal_payload.wanting_strength
        goal_match     = a.goal_match(current_z_goal)             [SD-039 cosine]
        staleness      = staleness_accumulator.lookup(region_key)
                         when accumulator present, else
                         (current_tick - a.last_accessed) * staleness_proxy_rate
        recoverability = clip_[0,1](a.goal_payload.last_vs)
                         when last_vs is not None, else
                         default_recoverability_when_unknown

        ghost_priority = ( wanting_weight       * wanting
                         + goal_match_weight    * goal_match
                         + staleness_weight     * staleness
                         + recoverability_weight * recoverability )

    All weights clamped to non-negative; absolute magnitude is irrelevant
    -- only ordering is consumed downstream. The goal_match_floor is the
    architectural commitment that pure rumination is excluded: anchors
    with no payload OR with goal_match < floor are invisible to the bank.

    Fields:
      wanting_weight:                 w_w on wanting term.
      goal_match_weight:              w_m on cosine(z_goal_snapshot, current).
      staleness_weight:               w_s on region staleness.
      recoverability_weight:          w_r on V_s-derived recoverability.
      goal_match_floor:               anchors with goal_match below this
                                      are excluded from the bank entirely
                                      (the rumination guard).
      top_k:                          cap on bank size returned per call.
                                      None -> no cap.
      default_recoverability_when_unknown:
                                      recoverability for anchors whose
                                      goal_payload.last_vs is None
                                      (e.g. MECH-269 phase 1/2 disabled).
                                      1.0 = treat as recoverable.
      include_inactive:               include the inactive (dual-trace
                                      preserved) half of the anchor pool.
                                      MECH-293 ghost-goal probes work
                                      primarily over inactive traces.
      include_active:                 include the currently-active half.
                                      Diagnostic / replay-prioritisation
                                      consumers may want this on.
      scale:                          optional scale filter ("fast",
                                      "slow"). None = all scales.
      staleness_proxy_rate:           when no staleness_accumulator is
                                      passed in, the per-tick fallback
                                      rate used to convert tick-delta
                                      since last_accessed into a [0,1]
                                      staleness scalar (clipped at 1.0).

    Backward compatible: consumer code gates on
    HippocampalConfig.use_mech292_ghost_bank.
    """
    wanting_weight: float = 1.0
    goal_match_weight: float = 1.0
    staleness_weight: float = 0.5
    recoverability_weight: float = 0.5
    goal_match_floor: float = 0.05
    top_k: Optional[int] = 32
    default_recoverability_when_unknown: float = 1.0
    include_inactive: bool = True
    include_active: bool = False
    scale: Optional[str] = None
    staleness_proxy_rate: float = 0.005
    # ---- MECH-339 Constraint 1: composite cue + outshining gate ----
    # The landed bank matches the retrieval cue by z_goal cosine only
    # (Anchor.goal_match). MECH-339 (ghost_goal_search.md Section 0.2
    # Constraint 1) asserts the cue must be composite: z_goal PLUS a
    # context channel built from the SD-039 payload fields the match
    # ignores, combined by an OUTSHINING gate (Smith & Vela 2001) so a
    # strong direct goal_match suppresses the context channel rather than
    # it being summed in with fixed weight. The smallest substrate step
    # sources the context channel from arousal_tag (the one SD-039 field
    # that is both stored and entirely unused by the bank). last_vs is
    # deferred (already consumed by the recoverability channel); a `cause`
    # tag is deferred (not present in the implemented AnchorGoalPayload --
    # design-sketch only, would need an SD-039 payload extension).
    #
    #   context_salience = 1 - exp(-arousal_tag / arousal_scale)   in [0, 1)
    #   gate             = clip_[0,1]((outshine_pivot - goal_match)
    #                                  / outshine_pivot)
    #   context_term     = context_weight * gate * context_salience
    #   ghost_priority  += context_term     (overall form stays additive;
    #                                        Constraint 2 unaffected)
    #
    # Defaults are no-op: master switch off and context_weight 0.0 ->
    # behaviour bit-identical to the pre-MECH-339 bank.
    use_composite_cue_outshining: bool = False
    context_weight: float = 0.0
    outshine_pivot: float = 0.5
    arousal_scale: float = 1.0
    # ---- MECH-340 persistence / efficacy gate ----
    # ARC-079 / MECH-340 (ghost_goal_search.md Section 0.3; Q-053
    # front-runner): persistence of an entry as an active re-probe target is
    # a GATED control/efficacy operation; disengagement is the default.
    # Smallest substrate step: a global control/efficacy *unattainability
    # appraisal passed into rank() -- NOT accumulated failure, NOT
    # recoverability/staleness/wanting (hard negatives per the pull).
    #
    #   license = clip_[0,1](control_efficacy * (1 - goal_unattainability))
    #
    # Anchors with license < persistence_floor are excluded from the bank
    # (they cease to be re-probe-eligible; the SD-039 trace is preserved).
    # Reengagement-coupled disengagement STATE is deferred (Klinger phase);
    # this step is the separable gate primitive only.
    #
    # Defaults are no-op: master switch off -> rank() ignores appraisal and
    # is bit-identical to the pre-MECH-340 bank.
    use_persistence_efficacy_gate: bool = False
    persistence_floor: float = 0.05
    persistence_default_when_appraisal_missing: float = 1.0


@dataclass
class PersistenceAppraisalComputeConfig:
    """Q-053 agent-side mapping into PersistenceAppraisal (MECH-340 consumer).

    Used by REEAgent when GhostGoalBankConfig.use_persistence_efficacy_gate
    is True. One-shot signals only; see persistence_appraisal_compute.py.
    """
    completion_weight: float = 0.6
    commitment_weight: float = 0.4


@dataclass
class HippocampalConfig:
    """Configuration for HippocampalModule.

    V3 changes (SD-004, Q-020 ARC-007 strict):
    - Navigates action-object space O, not raw z_world
    - action_object_dim: dimensionality of E2 action objects
    - terrain_prior now maps (z_world + e1_prior + residue_val) → action_object_mean
    - CEM operates in action-object space; E2.action_object() is called for scoring
    - No independent value head (ARC-007 strict: Q-020 resolved)

    SD-002 (preserved from V2): E1 prior wired into terrain search.
    """
    world_dim: int = 32            # z_world dimension
    action_dim: int = 4
    action_object_dim: int = 16    # SD-004: action object space dimensionality
    hidden_dim: int = 128
    horizon: int = 10
    num_candidates: int = 32
    num_cem_iterations: int = 3
    elite_fraction: float = 0.2
    # V3-EXQ-553 proposer-fix substrate (ARC-062 / MECH-269 cluster). When True,
    # the CEM inner-loop noise draw in propose_trajectories() is replaced with
    # an orthogonal-basis sample: the n noise vectors per iteration are stacked,
    # QR-decomposed, and the Q rows used (rescaled to N(0,1)-equivalent norm)
    # as the per-candidate noise. This gives maximally-distinct CEM proposals
    # at the seed-structure level, isolating "noise structure" as the variable
    # under test for the action-class-entropy cliff (V3-EXQ-550 / 551 / 551a /
    # 552 finding: pairwise_l2 ~ 1e-4, action_class_entropy = 0.0). When
    # n > flatten_dim (= horizon * action_object_dim), the basis is rank-
    # deficient and the implementation falls back to iid Gaussian for the
    # surplus candidates (with a diagnostic flag exposed in propose
    # diagnostics). Backward compatible: disabled by default; bit-identical
    # to legacy iid CEM when off.
    use_orthogonal_cem_seeding: bool = False
    # Diagnostic-only candidate support scaffold. When enabled,
    # HippocampalModule prepends one one-hot first-action candidate per
    # action class before returning candidates to E3, replacing the same
    # number of normal CEM candidates to preserve total K where possible.
    # Default False is bit-identical to the normal CEM proposer.
    use_action_class_scaffold_candidates: bool = False
    # Support-preserving CEM repair (ARC-065 hippocampal-trajectory-sampling
    # child). When enabled, CEM elite refit preserves a first-action class
    # floor when that support is present, and final candidate generation
    # injects minimal one-hot first actions only when the returned support
    # surface is collapsed.
    # MAIN-PATH DEFAULT (2026-05-17, SP-CEM landing): default True. The legacy
    # collapsing-CEM behaviour is intentionally NOT the default any more -- it
    # produced the monostrategy that left SD-029 / ARC-062 Rung 2 /
    # goal_pipeline / self_attribution experiments non_contributory. Validated
    # by V3-EXQ-567 PASS (ARM_1: selected_action_entropy 0.0124 -> 0.4965).
    # Bit-identical legacy opt-out: explicitly set this False together with
    # support_preserving_stratified_elites=False and
    # support_preserving_ao_std_floor=0.0.
    use_support_preserving_cem: bool = True
    support_preserving_min_first_action_classes: int = 2
    # Stratified elite selection: when True, the CEM elite refit always picks
    # the best-scoring candidate per action class first (across ALL present
    # classes), then fills remaining elite slots score-sorted. Stronger than
    # the support-preserving path which only activates when below the
    # class-count target.  Requires use_support_preserving_cem=True.
    # MAIN-PATH DEFAULT (2026-05-17): True (part of the V3-EXQ-567 ARM_1
    # validated combination). Legacy opt-out: set False.
    support_preserving_stratified_elites: bool = True
    # Per-class quota: when > 0, guarantees this many elite slots per
    # represented action class (capped by total elite budget and actual
    # candidates available).  0 = disabled.
    support_preserving_per_class_quota: int = 0
    # ao_std floor: after each CEM elite refit, clamp ao_std to at least this
    # value so the sampling distribution cannot collapse to a point.
    # MAIN-PATH DEFAULT (2026-05-17): 0.2 (V3-EXQ-567 ARM_1 value). Legacy
    # opt-out: set 0.0.
    support_preserving_ao_std_floor: float = 0.2
    # VALENCE_WANTING gradient: when > 0, trajectories toward high-wanting
    # (resource-proximal) regions score better during CEM selection.
    # Subtracted from terrain score (lower score = better in CEM).
    # Default 0.0 (backward compat). Set ~0.3-0.5 for goal-directed navigation.
    wanting_weight: float = 0.0
    # MECH-267: mode-conditioned hippocampal proposals (Pfeiffer & Foster 2013).
    # When enabled and an operating_mode dict is supplied to
    # propose_trajectories(), the per-mode noise multiplier is applied to the
    # CEM proposal std. Backwards compatible: disabled by default; when
    # disabled or operating_mode is None, propose_trajectories() behaves as
    # before. Mapping: external_task tight (exploitation), internal_planning
    # broader (exploration-biased), internal_replay tighter (consolidative),
    # offline_consolidation tightest (low-amplitude consolidation).
    mode_conditioning_enabled: bool = False
    mode_noise_scale: Dict[str, float] = field(default_factory=lambda: {
        "external_task":         1.0,
        "internal_planning":     1.3,
        "internal_replay":       0.5,
        "offline_consolidation": 0.3,
    })
    # SD-055: differentiable CEM selection approximation. When enabled, replaces
    # the non-differentiable argsort elite-selection step with a softmax-weighted
    # candidate mean so gradient can flow back to cue_action_proj (SD-016).
    # ARC-007 STRICT: scoring continues to use existing residue-terrain evaluation;
    # no value head is added. Opt-in; default False (legacy argmax path unchanged).
    use_differentiable_cem: bool = False
    differentiable_cem_temperature: float = 1.0
    # MECH-269 base substrate (Phase 1, 2026-04-22): per-stream verisimilitude
    # V_s scores tracked on the HippocampalModule. For each registered stream,
    # V_s = 1 - norm(z_hat - z_curr) / (norm(z_curr) + eps), EMA-smoothed with
    # per_stream_vs_tau. Provides the observable signal that the MECH-287
    # broadcast trigger and MECH-284 staleness accumulator will consume in
    # Phase 2/3. Backward compatible: disabled by default; when off,
    # update_per_stream_vs() is a no-op and per_stream_vs stays empty.
    # Forward-predictor routing (Phase 1 implementation):
    #   z_world  -> ReafferencePredictor (SD-007 / MECH-101) when available
    #   z_harm_s -> HarmForwardModel (SD-011) when available
    #   others   -> identity-prediction proxy (z_hat = z_prev)
    use_per_stream_vs: bool = False
    per_stream_vs_tau: float = 0.1
    per_stream_vs_streams: tuple = (
        "z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta",
    )
    # MECH-288: hierarchical event segmenter (Phase 2 of V_s invalidation runtime).
    # When enabled, a two-scale EventSegmenter ticks in agent.sense() after latent
    # encoding and before per-stream V_s update; BoundaryEvent objects are queued on
    # HippocampalModule for downstream MECH-287 broadcast / MECH-269 anchor-reset
    # consumers. Backward compatible: disabled by default (no queue writes, no ticks).
    use_event_segmenter: bool = False
    event_segmenter: EventSegmenterConfig = field(default_factory=EventSegmenterConfig)
    # MECH-287: broadcast invalidation trigger (Phase 2 iv of V_s invalidation runtime).
    # When enabled, subscribes to MECH-288 BoundaryEvents emitted in agent.sense()
    # and re-emits BroadcastEvent objects (strength = posterior * gain) onto
    # HippocampalModule._broadcast_event_queue for downstream MECH-269 anchor-reset
    # (T3) / MECH-284 staleness accumulator consumers. Phasic/tonic guardrail:
    # high tonic boundary activity suppresses the next phasic broadcast. Backward
    # compatible: disabled by default (no trigger ticks, no broadcast queue writes).
    use_invalidation_trigger: bool = False
    invalidation_trigger: InvalidationTriggerConfig = field(
        default_factory=InvalidationTriggerConfig
    )
    # MECH-269 Phase 2 (ii): scale-tagged anchor sets with dual-trace
    # preservation (mark_inactive, NOT erase) on remap. Keys:
    # (scale, segment_id, stream_mixture). Subscribes to MECH-288
    # BoundaryEvents. Backward compatible: disabled by default; when off,
    # anchor_set is None and no ticks/drains occur.
    use_anchor_sets: bool = False
    anchor_set: AnchorSetConfig = field(default_factory=AnchorSetConfig)
    # MECH-269 Phase 2 (iii, T4): per-region per-stream V_s readout.
    # Promotes the flat per_stream_vs dict to per_region_vs[(scale,
    # segment_id)][stream] -> float, keyed on AnchorSet active-anchor
    # keys (T3). V_s foundation lit-pull verdict 3: per-stream V_s is
    # the projection-readout of the integrated mixed-selectivity code;
    # per-region keying provides the partition. Orthogonal to
    # use_per_stream_vs -- per-region is a refinement, not a
    # replacement; both can be on simultaneously. When on without
    # use_anchor_sets, update is a no-op (per_region_vs stays empty).
    # MECH-287 broadcast events on (scale, segment_id) reset that
    # region's per_region_vs entries and mark_inactive the matching
    # anchor with k=5 hysteresis semantics (T3 logic). Backward
    # compatible: disabled by default.
    use_per_region_vs: bool = False

    # MECH-284 Phase 3: region-indexed staleness accumulator. Integrates
    # MECH-287 BroadcastEvents against the active anchor set, leaks per
    # tick, and is read by MECH-269 online anchor-reset hysteresis (when
    # use_mech284_hysteresis is also True). Backward compatible: disabled
    # by default; when off, StalenessAccumulator is None and hysteresis
    # uses the Phase 2 internal tick-delta proxy unchanged.
    use_staleness_accumulator: bool = False
    staleness_accumulator: StalenessAccumulatorConfig = field(
        default_factory=StalenessAccumulatorConfig
    )
    # MECH-269 Phase 3: switch AnchorSet.tick_hysteresis to consume the
    # MECH-284 staleness accumulator instead of its internal
    # (tick - last_accessed) * staleness_rate proxy. Only meaningful when
    # use_staleness_accumulator is also True; otherwise the hysteresis
    # lookup sees a zero-staleness accumulator and behaves identically to
    # the flag being off (V_s_anchor = V_s - 0).
    use_mech284_hysteresis: bool = False

    # MECH-269 / MECH-090 read-side hook: V_s -> commit release.
    # When True, REEAgent.select_action() snapshots the active AnchorSet
    # keys at commit entry (all three commit-entry paths). On each subsequent
    # tick while beta is elevated, if any snapshot key is no longer in the
    # current active_anchors() set, beta_gate.release() is called and
    # _committed_step_idx is reset (mirroring the MECH-091 urgency-interrupt
    # template). This is the read-side closure of the V_s invalidation
    # runtime: anchor invalidation events authored by MECH-287/MECH-288/
    # MECH-284 (write-side) become observable behavioural changes
    # (commitment release) only when this flag is on. Backward compatible:
    # disabled by default; with flag off, EXQ-478/480 wired-but-inert
    # behaviour reproduces. Requires use_anchor_sets=True to do anything
    # (no-op without an active anchor set to snapshot).
    use_vs_commit_release: bool = False

    # MECH-269b: Symmetric V_s gating on E1/E2 cortical rollouts (read-side
    # consumer of MECH-269 Phase 1 per_stream_vs).
    # When True, agent constructs a VsRolloutGate that snapshots per-stream
    # latent values when V_s[s] >= snapshot_refresh_threshold and substitutes
    # the snapshot for the current value when V_s[s] < per-side threshold at
    # the E1 sensory predictor and E2_harm_a per-tick forward call sites. Held
    # snapshots prevent cortical forward predictors from rolling forward off
    # stale-but-confident-looking inputs (the wired-but-inert failure mode of
    # EXQ-483 / EXQ-483a).
    # Requires use_per_stream_vs=True at agent build time (the gate has nothing
    # to read otherwise; agent.__init__ raises ValueError on that mismatch).
    # Backward compatible: disabled by default. With flag on but V_s seeded
    # at 1.0 the gate fires zero times until per-stream alignment drifts.
    # Per-stream override dicts (e1_threshold_per_stream / e2_threshold_per_stream)
    # are kept at the gate-config level (see VsRolloutGateConfig) and wired
    # only via the agent constructor; from_dims exposes the global side
    # thresholds, the master switch, the streams tuple, and the snapshot
    # refresh threshold.
    use_vs_rollout_gating: bool = False
    vs_gate_streams: tuple = (
        "z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta",
    )
    vs_gate_snapshot_refresh_threshold: float = 0.5
    vs_gate_e1_threshold: float = 0.4
    vs_gate_e2_threshold: float = 0.4
    vs_gate_unknown_stream_passes: bool = True
    # MECH-269b + MECH-284 wiring (Q-040b strong reading). When True, the
    # VsRolloutGate subtracts MECH-284 region staleness (aggregated to
    # per-stream by HippocampalModule.compute_per_stream_staleness via
    # max-over-active-anchors-whose-stream_mixture-includes-stream) from raw
    # V_s before the threshold comparison: effective_vs = raw_vs -
    # per_stream_staleness[s]. This makes the gate's stale-stream
    # discrimination testable at realistic V_s values without smoke-level
    # threshold overrides. Requires use_per_stream_vs=True (already
    # required by use_vs_rollout_gating) AND use_staleness_accumulator=True
    # AND use_anchor_sets=True; agent.__init__ raises ValueError on any
    # missing precondition. Backward compatible: default False; flag-OFF
    # gate behaviour is bit-identical to the legacy raw-V_s path.
    use_vs_gate_staleness_lookup: bool = False

    # MECH-292: ranked ghost-goal bank (derived view over the SD-039
    # dual-trace anchor pool). When enabled, HippocampalModule
    # instantiates a GhostGoalBank and exposes rank_ghost_goals().
    # Pure-arithmetic, non-trainable. Requires use_anchor_sets=True AND
    # AnchorSetConfig.use_sd039_anchor_payload=True (the bank reads
    # payloads written by the SD-039 population layer); HippocampalModule
    # __init__ raises ValueError on either mismatch. Backward compatible:
    # disabled by default; with flag OFF, ghost_goal_bank is None and
    # rank_ghost_goals() returns []. MECH-293 (waking ghost-goal probe
    # search) is the first behavioural consumer; this substrate is the
    # prerequisite for that wiring.
    use_mech292_ghost_bank: bool = False
    ghost_goal_bank_config: GhostGoalBankConfig = field(
        default_factory=GhostGoalBankConfig
    )
    # Q-053: agent computes PersistenceAppraisal when MECH-340 gate is on.
    persistence_appraisal_compute: PersistenceAppraisalComputeConfig = field(
        default_factory=PersistenceAppraisalComputeConfig
    )

    # MECH-293: waking ghost-goal probe search (read-side consumer of
    # MECH-292 ranked ghost-goal bank). When enabled, propose_trajectories()
    # adds a minority budget of CEM probes seeded around the highest-priority
    # bank entries' anchor.z_world rather than the agent's current z_world.
    # Each ghost trajectory carries hypothesis_tag=True and a metadata dict
    # tagging its source anchor + ghost_priority for downstream provenance.
    # Requires use_mech292_ghost_bank=True; HippocampalModule __init__ raises
    # ValueError on the mismatch (use_mech292 transitively guarantees
    # use_anchor_sets and use_sd039_anchor_payload). Backward compatible:
    # disabled by default; with flag OFF, the ghost branch is never entered.
    # mech293_replace_lowest_ranked=True (default) keeps the total candidate
    # count constant by dropping the highest-cost value-flat candidates
    # (CEM is lower-is-better). Setting False appends ghosts on top of the
    # value-flat pool (raises total) and is intended for diagnostic /
    # ablation use only.
    use_mech293_ghost_probes: bool = False
    mech293_ghost_fraction: float = 0.2
    mech293_min_ghost_candidates: int = 1
    mech293_max_ghost_candidates: int = 8
    mech293_replace_lowest_ranked: bool = True

    # MECH-290: backward trajectory credit sweep at goal arrival.
    # Biological basis: Foster & Wilson 2006 (Nature) -- reverse replay
    # fires at reward endpoint during waking, concurrent with dopamine.
    # Credit propagates backward from goal to trajectory start.
    # When enabled, HippocampalModule.backward_credit_sweep() is called
    # each time BetaGate releases via hippocampal completion signal
    # (ARC-028). The committed trajectory (stored by
    # record_committed_trajectory() at commit entry) is swept backward;
    # VALENCE_WANTING is updated at each z_world state proportional to:
    #   credit_t = outcome_quality * gamma^(T - t)
    # where T = trajectory length, t = step index.
    # No SD-006 dependency: fires synchronously on waking path.
    # Requires ResidueConfig.valence_enabled=True to write
    # VALENCE_WANTING; silently skips valence write if disabled.
    # Backward compatible: disabled by default.
    use_backward_credit_sweep: bool = False
    backward_sweep_gamma: float = 0.9        # temporal discount per step back
    backward_sweep_min_quality: float = 0.6  # only sweep on high-quality completions


@dataclass
class ResidueConfig:
    """Configuration for the Residue Field φ(z_world).

    V3 change (SD-005): operates over z_world, not z_gamma.
    Accumulates world_delta — z_world change attributable to agent action.
    Self-change (z_self_delta) does NOT drive residue accumulation.

    hypothesis_tag check: accumulate() refuses to accumulate when
    hypothesis_tag=True (MECH-094 — simulated/replay content cannot
    produce residue).
    """
    world_dim: int = 32        # z_world dimension (SD-005)
    hidden_dim: int = 64
    accumulation_rate: float = 0.1
    decay_rate: float = 0.0    # 0 = no decay (invariant: residue cannot be erased)
    num_basis_functions: int = 32
    kernel_bandwidth: float = 1.0
    integration_rate: float = 0.01
    # ARC-030 / MECH-117: benefit terrain (liking -- separate from z_goal wanting)
    benefit_terrain_enabled: bool = False
    # MECH-303: contextual passive safety terrain (separate from benefit_terrain and VALENCE_LIKING).
    # Accumulates harm-absence signal per step at current z_world; read in select_action()
    # to lower background avoidance commitment. Disabled by default (bit-identical OFF).
    safety_terrain_enabled: bool = False
    # SD-014: 4-component valence vector [wanting, liking, harm_discriminative, surprise].
    # When False, evaluate_valence() returns zeros and update_valence() is a no-op.
    # Used to ablate valence tracking in experiments that do not need replay prioritisation.
    # Prerequisite for ARC-036 (multidimensional valence map).
    valence_enabled: bool = True
    # MECH-334 (INV-074): critical-period closure / crystallization EWC
    # penalty. When ewc_enabled=True, snapshot_ewc_anchor() captures the
    # Phase-3 checkpoint (rbf_field centers/weights + an established-basin
    # Fisher proxy = |anchor_weight| * active_mask) and ewc_penalty()
    # returns ewc_lambda * sum(Fisher * (param - anchor)^2) for the
    # experiment to add to its loss. NOT a hard freeze: established basins
    # are protected proportionally to their accumulated strength while the
    # field keeps adapting elsewhere (Kirkpatrick 2017 write-protect;
    # faithful to MECH-334 "high resistance to overwriting established
    # basins"). Default OFF / lambda 0.0 = bit-identical no-op.
    ewc_enabled: bool = False
    ewc_lambda: float = 0.0


@dataclass
class HeartbeatConfig:
    """Configuration for the multi-rate clock (SD-006, ARC-023).

    Phase 1: time-multiplexed execution with explicit rate parameters.
    Each loop runs at its own temporal grain.

    MECH-089: E3 consumes theta-cycle-averaged z_world, never raw E1 output.
    MECH-090: beta_state gates E3 policy propagation (not E3 internal updating).
    MECH-091: phase_reset() on salient events synchronises next E3 tick.
    MECH-092: quiescent E3 cycles trigger hippocampal replay.
    MECH-093: e3_steps_per_tick varies with z_beta magnitude.
    """
    e1_steps_per_tick: int = 1     # E1 updates every env step
    e2_steps_per_tick: int = 3     # E2 updates every 3 env steps
    e3_steps_per_tick: int = 10    # E3 updates every 10 env steps (deliberation rate)
    theta_buffer_size: int = 10    # E1 steps per theta-cycle summary

    # MECH-093: z_beta magnitude → E3 rate scaling
    beta_rate_min_steps: int = 5   # Fastest E3 rate (high arousal)
    beta_rate_max_steps: int = 20  # Slowest E3 rate (low arousal)
    beta_magnitude_scale: float = 1.0  # Scale factor for z_beta magnitude

    # MECH-108: BreathOscillator — periodic uncommitted windows.
    # breath_period=0 disables (backward compatible). When > 0, the oscillator
    # creates periodic sweep phases that reduce the effective commit_threshold,
    # forcing uncommitted windows even after training converges variance below
    # the base threshold. Without this, the agent becomes permanently committed.
    # See clock.py for timing semantics.
    breath_period: int = 0           # 0 = disabled; e.g. 50 = sweep every 50 steps
    breath_sweep_amplitude: float = 0.25  # Fractional threshold reduction during sweep
    breath_sweep_duration: int = 5   # Duration of each sweep phase (steps)

    # MECH-090: bistable beta gate committed->uncommitted dynamics (SD-021 prerequisite).
    # False (default): legacy "for now" behavior -- elevate/release beta every E3 tick
    #   based on current commitment state. Fully backward compatible.
    # True (bistable): latch on commit entry; release only via hippocampal completion
    #   signal (BetaGate.receive_hippocampal_completion()) or explicit uncommit.
    #   Required for SD-021 (descending pain modulation) and proper MECH-090 dynamics.
    beta_gate_bistable: bool = False

    # MECH-090 commit-entry readiness conjunction (commitment_closure:GAP-4).
    # Synthesis: REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
    # synthesis.md (2026-05-28, lit-pull commit 9e68c5ca8a). R-c single-gate
    # conjunction reading: BetaGate entry requires precision-low AND motor-program
    # readiness above floor. V3-EXQ-592 seed 42 showed rv-only is satisfiable by
    # degenerate trivial-predictability (rv=2.7e-5 at nav_competence=0.0).
    # Cisek-Kalaska 2010 affordance-competition + Hanes-Schall 1996 accumulator-
    # to-threshold + Roesch-Calu-Schoenbaum 2007 readiness signal.
    # False (default): legacy rv-only entry. beta_gate.elevate() fires whenever
    #   E3SelectionResult.committed is True. Bit-identical to pre-MECH-090-R-c.
    # True: agent.py guards elevate() with
    #   should_admit_elevation(score_margin) -- elevation is blocked when the
    #   per-candidate E3 score margin (top1 - top2 over result.scores;
    #   lower-is-better so winner is argmin and margin = second_min - min)
    #   falls below commit_readiness_floor. Pure single-stage gate.
    use_commit_readiness_gate: bool = False
    # Floor on the per-candidate first-action margin (REE lower-is-better, so the
    # selected score is the minimum; margin = scores.sort()[1] - scores.min()).
    # Default 0.05 is small relative to the EXQ-608 mean_top2_class_gap range
    # 0.27-1.96 seen in unaugmented baselines, so the gate fires only on
    # genuinely degenerate near-tie scoring (V3-EXQ-592 seed-42 signature).
    # Q-053-style calibration is a follow-on substrate task, not a precondition
    # for the substrate landing.
    commit_readiness_floor: float = 0.05
    # When the candidate pool collapses to a single trajectory (top-k=1; some
    # ablation arms), the score-margin is undefined. False (default):
    # treat single-candidate ticks as readiness-above-floor (permissive, so
    # the legacy single-trajectory path still elevates). True: treat as
    # readiness-below-floor (strict, forces multi-candidate pools).
    # Diagnostic only -- production substrate stays permissive.
    commit_readiness_strict_single_candidate: bool = False


@dataclass
class EnvironmentConfig:
    """Configuration for CausalGridWorld V3."""
    size: int = 10
    num_resources: int = 5
    num_hazards: int = 3

    # Reward/harm signals
    resource_benefit: float = 1.0
    hazard_harm: float = -1.0

    # SD-005: observation channel dimensions
    body_obs_dim: int = 10    # proprioceptive channels (position, health, energy, footprint)
    world_obs_dim: int = 54   # exteroceptive channels (local view, contamination)


@dataclass
class REEConfig:
    """Master configuration for the complete REE-v3 agent."""
    latent: LatentStackConfig = field(default_factory=LatentStackConfig)
    e1: E1Config = field(default_factory=E1Config)
    e2: E2Config = field(default_factory=E2Config)
    e3: E3Config = field(default_factory=E3Config)
    hippocampal: HippocampalConfig = field(default_factory=HippocampalConfig)
    residue: ResidueConfig = field(default_factory=ResidueConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    serotonin: SerotoninConfig = field(default_factory=SerotoninConfig)

    device: str = "cpu"
    seed: Optional[int] = None

    offline_integration_frequency: int = 100

    # MECH-057a: action-loop completion gate (preserved from V2)
    action_loop_gate_enabled: bool = False

    # MECH-294: multi-content theta-burst packet (sibling to MECH-089 ThetaBuffer).
    # When True, a MultiContentThetaPacket binds a {goal_latent, action_proposal,
    # risk_estimate (z_harm_s + z_harm_a), state_summary} tuple within one theta
    # cycle (E3-heartbeat interval) and exposes it as agent.last_theta_packet.
    # Requires use_per_stream_vs=True (the packet consumes per-stream V_s for
    # vintaging). Default False -> agent.multi_content_theta_packet is None and
    # MECH-089 ThetaBuffer behaviour is byte-identical (bit-identical OFF).
    # See REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md.
    use_multi_content_theta_packet: bool = False
    # Binding regime: "joint" (MECH-294 hypothesis -- all four streams co-bound
    # within one cycle), "alternation" (Kay-2020 one-stream-per-cycle control),
    # "shuffled" (independent-content control -- each slot from a different cycle).
    theta_packet_binding_mode: str = "joint"
    # MECH-269b vintaging thresholds (reused, not re-invented): refresh the
    # per-stream snapshot when V_s >= refresh; substitute the held snapshot when
    # V_s < hold. 0.4-0.5 dead-band = MECH-269b Schmitt hysteresis.
    theta_packet_snapshot_refresh_threshold: float = 0.5
    theta_packet_hold_threshold: float = 0.4
    # Read-only-first discipline (memo S5): when False (default), the packet is
    # built / sealed / exposed / logged but does NOT touch E3 selection. When
    # True, joint_context biases E3 via a PARAMETER-FREE clamped arithmetic bias
    # (no trained head -> no phased training). The behavioural successor flips it.
    theta_packet_compose_into_e3_bias: bool = False
    theta_packet_bias_scale: float = 0.1
    # When True (default) the compose path gates the per-candidate action-grounding
    # bias by the sealed packet's within-cycle co-binding coherence (fraction of
    # co-temporally-current streams) -- so joint/alternation/shuffled reach E3
    # behaviour distinctly (the S6 discriminator). False recovers the legacy
    # action-only cosine (gate==1.0) bit-for-bit (the ablation). No-op unless
    # theta_packet_compose_into_e3_bias is also True.
    theta_packet_compose_use_joint_coherence: bool = True
    # MECH-294 per-candidate co-binding coherence (route-range amend, substrate-
    # ceiling-lifted triage 2026-06-19): currency_coherence() is a SCALAR
    # (rescale-invisible) so the modulatory selection authority + 569i top-k have
    # no per-candidate range to carve (V3-EXQ-661: committed-dist TV ~0 incl
    # gate-ON-vs-OFF). When True, the compose path produces a per-candidate
    # co-binding coherence bias whose cross-candidate RANGE is mode-distinct
    # (JOINT all four streams co-bound to this-cycle action -> full range;
    # ALTERNATION live + held-prior -> different pattern; SHUFFLED none co-bound
    # -> zeros). Routed via _bdc_coherence so the route-range authority rescales
    # it. No-op unless theta_packet_compose_into_e3_bias is also True; OFF (the
    # default) recovers the legacy scalar-gated action-only compose bit-for-bit.
    theta_packet_compose_per_candidate_coherence: bool = False
    # Weight on a held (alternation non-live) stream's PRIOR co-bound action in
    # the per-candidate coherence. 1.0 = held streams count as fully as the live
    # stream; 0.0 = pure currency-gating (only co-temporally-current streams).
    theta_packet_coherence_hold_weight: float = 0.5

    # MECH-205: surprise-gated replay. When True, prediction error from
    # E3.post_action_update() populates VALENCE_SURPRISE in the residue field,
    # and the replay drive_state surprise weight is set from recent PE magnitude.
    # Requires valence_enabled=True in residue config. Default False (backward compat).
    surprise_gated_replay: bool = False
    # MECH-205: PE EMA smoothing factor. Controls how fast the baseline tracks PE.
    # 0.02 = ~50-step effective window (default). 0.1 = ~10-step (too fast -- surprise
    # stays near zero because EMA tracks spikes immediately). Only active when
    # surprise_gated_replay=True.
    pe_ema_alpha: float = 0.02
    # MECH-205: minimum surprise magnitude to write to residue field. Filters out
    # negligible PE-EMA deltas that would accumulate as noise. Only active when
    # surprise_gated_replay=True.
    pe_surprise_threshold: float = 0.001

    # MECH-120: SHY-analog synaptic homeostasis in SWS
    shy_enabled: bool = False          # master switch (default off for backward compat)
    shy_decay_rate: float = 0.85       # EMA decay toward slot-mean; 0.85 per Tononi SHY lit

    # SD-017: minimal sleep-phase infrastructure (SWS-analog + REM-analog passes)
    # SWS-analog pass: hippocampus-to-cortex schema installation.
    # Writes compressed z_world prototypes from recent experience into ContextMemory
    # slots, installing differentiated context attractors (slot-formation, MECH-166).
    # Must be called after enter_sws_mode() which gates waking writes + runs SHY.
    sws_enabled: bool = False          # master switch (default off for backward compat)
    sws_consolidation_steps: int = 5   # schema-installation write passes per SWS call
    sws_schema_weight: float = 0.1     # EMA weight for ContextMemory slot installation
    # REM-analog pass: causal attribution replay (slot-filling, MECH-166).
    # Replays recent trajectory experience through the hippocampal module.
    # Evaluates residue terrain per trajectory; hypothesis_tag=True (no new residue).
    # ARC-045: forward + reverse replay both run (bidirectional information flow proxy).
    rem_enabled: bool = False          # master switch (default off for backward compat)
    rem_attribution_steps: int = 10    # attribution rollouts per REM call

    # MECH-165: reverse replay diversity scheduler
    replay_diversity_enabled: bool = False   # master switch (default off for backward compat)
    reverse_replay_fraction: float = 0.3     # fraction of replay calls using reverse mode
    random_replay_fraction: float = 0.2      # fraction using random action rollout
    exploration_buffer_len: int = 50         # max stored exploration trajectories (FIFO)

    # MECH-216: E1 predictive wanting (schema readout).
    # schema_wanting_threshold: minimum schema_salience to seed VALENCE_WANTING.
    # schema_wanting_gain: multiplier on schema_salience * drive_level -> wanting value.
    # Master switch is E1Config.schema_wanting_enabled (default False).
    # goal_stream_enabled: convenience marker/preset for the coherent z_goal +
    # schema wanting + MECH-295/307 bundle. Default False preserves legacy
    # partial-switch behavior.
    goal_stream_enabled: bool = False
    schema_wanting_threshold: float = 0.3
    schema_wanting_gain: float = 0.5

    # SD-016 Path 1 (V3-EXQ-418e): auxiliary diversification loss weight on
    # ContextMemory slots. When > 0 and E1Config.sd016_enabled=True, adds
    # `weight * context_memory.compute_diversification_loss()` to
    # REEAgent.compute_prediction_loss after the existing E1 loss. 0.0 = no-op.
    # See REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md.
    sd016_diversification_weight: float = 0.0

    # SD-019: affective harm non-redundancy constraint.
    # harm_nonredundancy_weight > 0 adds a cosine^2 penalty between z_harm_s and z_harm_a,
    # enforcing that the two streams encode non-redundant information.
    # 0.0 = disabled (default, backward compat). Typical range: 0.01 - 0.1.
    # harm_nonredundancy_precision_scale > 0 scales the penalty by E3 current_precision
    # (ARC-016 coupling: enforce non-redundancy more strongly when the agent is confident).
    # 0.0 = unscaled (default, backward compat).
    harm_nonredundancy_weight: float = 0.0
    harm_nonredundancy_precision_scale: float = 0.0

    # SD-020: precision-weighted prediction error target for z_harm_a training.
    # When True, compute_harm_accum_loss() trains z_harm_a on the PE signal
    # (|actual_harm - expected_harm| * precision_norm) rather than raw EMA harm.
    # Chen 2023: AIC encodes unsigned intensity prediction errors, not raw magnitude.
    # False = disabled (default, backward compat; keeps existing EMA target).
    harm_surprise_pe_enabled: bool = False
    # EMA smoothing alpha for expected-harm tracker (running average of harm_obs).
    # Smaller = slower adaptation (longer effective window for "expected").
    # Default 0.1 = ~10-step window.
    harm_obs_ema_alpha: float = 0.1

    # SD-021: descending pain modulation (commitment-gated z_harm attenuation).
    # When True, agent.sense() attenuates z_harm by descending_attenuation_factor
    # when beta_gate.is_elevated (E3 committed to a trajectory through expected harm).
    # z_harm_a is NOT attenuated (affective load persists regardless of commitment).
    # False = disabled (default, backward compat).
    harm_descending_mod_enabled: bool = False
    # Multiplier on z_harm when committed (0 < factor <= 1). 0.5 = 50% attenuation.
    descending_attenuation_factor: float = 0.5
    # SD-014 valence channel writes.
    # valence_harm_enabled: write post-attenuation z_harm_s.norm() to VALENCE_HARM_DISCRIMINATIVE
    #   on each sense() call. Uses post-SD-021 value so committed-state nodes get stale h.
    # valence_liking_enabled: write benefit_exposure to VALENCE_LIKING at consummatory contact
    #   (call update_liking() from experiment loop when resource is collected).
    # Both default False (backward compat).
    valence_harm_enabled: bool = False
    valence_liking_enabled: bool = False
    liking_threshold: float = 0.1

    # MECH-258: E2_harm_a affective-pain forward model (prerequisite for SD-032b).
    # When True, REEAgent instantiates an E2HarmAForward module predicting
    # z_harm_a_{t+1} = f(z_harm_a_t, a_t). Enables runtime z_harm_a_PE signal
    # (||z_harm_a_actual - pred||_2) for dACC consumption.
    # Training is external (experiment-loop driven), same as ARC-033 E2_harm_s.
    # False = disabled (default, backward compat).
    use_e2_harm_a: bool = False
    # Learning rate used when an experiment instantiates the E2_harm_a optimizer.
    # Parallel to E2HarmSConfig default; low because z_harm_a is low-dim, noisy.
    e2_harm_a_lr: float = 5e-4

    # ARC-058: harm_stream.shared_forward_trunk hypothesis.
    # Competing claim against the current ARC-033 implementation (which is a fully
    # independent E2_harm_s forward model). When True, REEAgent builds a single
    # HarmForwardTrunk instance and passes it to BOTH E2_harm_s and E2_harm_a so
    # they share the hidden-representation substrate with stream-specific heads.
    # False (default) = legacy ARC-033 independent-modules behaviour preserved.
    # Biological basis (Horing & Buchel 2022 PLoS Biol): anterior insula shows
    # modality-independent unsigned PE (shared trunk); dorsal posterior insula
    # shows modality-specific signed PE (per-stream head). See lit-pull
    # targeted_review_pain_predictive_coding_substrate (2026-04-19).
    use_shared_harm_trunk: bool = False

    # SD-032b: dACC/aMCC-analog adaptive control (minimum-viable cingulate substrate).
    # When True, REEAgent instantiates DACCAdaptiveControl which reads z_harm_a_PE,
    # z_conflict (top-vs-runner-up margin over E3 scores), and a control-demand
    # estimate, and emits a Croxson-style integration bundle (mode_ev, choice
    # difficulty, foraging value, harm interaction) that a stopgap adapter reduces
    # to a per-candidate score_bias[K] consumed by e3.select().
    # The stopgap adapter is explicitly a temporary wiring choice; SD-032a
    # (salience-network coordinator, not yet implemented) is the architectural
    # consumer of the bundle. Remove the adapter when SD-032a lands.
    # False = disabled (default, backward compat).
    use_dacc: bool = False
    # SD-057 phase-2 L7 (MECH-348): object-discriminative goal readout on the dACC
    # consumer. use_mech_consume gates feeding per-candidate goal_proximity (to the
    # SD-057-object-bound z_goal) into the dACC bundle; dacc_goal_readout_weight is
    # the adapter's bias weight on it. Both no-op default (bit-identical OFF).
    use_mech_consume: bool = False
    dacc_goal_readout_weight: float = 0.0
    # dACC output weighting (all zero by default -- no behavioural effect until set).
    # dacc_weight: scales -mode_ev[K] contribution to score_bias.
    dacc_weight: float = 0.0
    # dacc_interaction_weight: scales -harm_interaction[K] (Croxson 2009) contribution.
    dacc_interaction_weight: float = 0.0
    # dacc_foraging_weight: scales -foraging_value broadcast contribution
    # (uniform shift when switching away from committed trajectory is warranted).
    dacc_foraging_weight: float = 0.0
    # dacc_suppression_weight: scales MECH-260 recency-similarity counter-bias.
    dacc_suppression_weight: float = 0.0
    # MECH-260: recency memory window length. Recent actions are stored in a FIFO
    # deque; each candidate trajectory's first action is penalised by cosine
    # similarity to the recency-vector average. Small window = local recency only.
    dacc_suppression_memory: int = 8
    # MECH-268: pe saturation under repeated identical outcomes (habituation).
    # Distinct from dacc_pe_cap (absolute clamp) -- saturation is history-
    # conditioned: pe_saturated = pe_after_cap / (1 + strength * max(0, n_rec - grace))
    # where n_rec is the count of the current outcome class within the
    # last `dacc_saturation_window` outcomes. Enables the dACC conflict-
    # saturation falsification target (EXP-0159) and the closure-vs-
    # contradiction interaction (EXP-0164).
    dacc_saturation_enabled: bool = False
    dacc_saturation_window: int = 8
    dacc_saturation_strength: float = 0.3
    dacc_saturation_grace: int = 2
    # Precision normalisation scale (matches SD-020 pattern).
    # precision_norm = min(e3.current_precision / dacc_precision_scale, 3.0).
    # Higher scale -> more modest precision weighting.
    dacc_precision_scale: float = 500.0
    # Shenhav 2013 EVC: effort cost scalar (minimum-viable). Multiplies
    # control_required(K) when computing mode_ev[K] = payoff - control*effort.
    # Per-trajectory effort costs are a future refinement (Croxson, Kennerley 2006).
    dacc_effort_cost: float = 0.1
    # Scholl 2017: neuromodulator-tunable learning-rate gain. When
    # dacc_drive_coupling > 0, dACC heads' effective learning rate is scaled by
    # (1.0 + dacc_drive_coupling * drive_level) from SD-012 GoalState.drive_level.
    # 0.0 = disabled (default, backward compat).
    dacc_drive_coupling: float = 0.0
    # Absolute cap on total score_bias from DACCtoE3Adapter. 0.0 = no cap
    # (backward compat). Set 2.0 to prevent mode_ev dominating inter-candidate
    # variation when harm_eval_head raw scores are O(10-40).
    dacc_bias_max_abs: float = 0.0

    # SD-032a: salience-network coordinator.
    # Master switch -- when True, REEAgent instantiates a SalienceCoordinator
    # that aggregates the dACC bundle + drive_level + offline-mode flag into a
    # soft operating_mode probability vector and a discrete mode_switch_trigger
    # (MECH-259). Hosts the MECH-261 dict-keyed write-gate registry. Reads slots
    # for SD-032c/d/e signals (no-op until those land). False = disabled
    # (default, backward compat).
    use_salience_coordinator: bool = False
    # MECH-259 base switch threshold. effective_threshold = switch_threshold *
    # (1 + stability_scaling * pcc_stability). Trigger fires when salience
    # aggregate > effective_threshold AND argmax(operating_mode) != current_mode.
    salience_switch_threshold: float = 1.0
    # SD-032d (PCC stability) modulation strength on switch threshold.
    salience_stability_scaling: float = 1.0
    # Softmax temperature for the operating_mode vector. Higher -> more uniform.
    salience_softmax_temperature: float = 1.0
    # Bias added to external_task logit before softmax. Ensures default mode is
    # external_task when all inputs are zero (waking baseline).
    salience_external_task_bias: float = 1.0
    # Salience aggregate weights (urgency-relevant signals; separate from
    # mode-affinity weights -- "how loud is the alarm" vs "which mode does this
    # argue for"). dACC PE is the only live source in V3 (AIC salience is
    # SD-032c, no-op until landed).
    salience_dacc_pe_weight: float = 1.0
    salience_dacc_foraging_weight: float = 0.5
    # When True, the e3_policy write-gate value scales the dACC score_bias
    # before E3.select() (so that during internal_replay, dACC influence on
    # action selection is suppressed near zero). Default False = backward
    # compatible (dACC bias unaffected).
    salience_apply_to_dacc_bias: bool = False

    # mode-governance-engagement substrate (2026-06-13): external_task salience
    # SOURCE. On the 603n foraging substrate the SalienceCoordinator never sees a
    # signal that puts the agent in external_task mode (drive_level ~0.016, dACC
    # PE/foraging/difficulty all push internal_planning), so fraction_in_external_task
    # is structurally 0 and MECH-266's exit-rail has no contested mode to bind. When
    # use_external_task_drive=True, REEAgent injects a goal-pursuit-derived
    # "external_task_drive" signal into the coordinator (registered in BOTH
    # affinity_weights -> external_task AND salience_weights -> aggregate, so a switch
    # INTO external_task can fire), sourced from the committed-pursuit-of-an-active-goal
    # engagement scalar:
    #   engagement = goal_active ? clip(commit_w*float(beta_gate.is_elevated)
    #                                   + prox_w*goal_proximity(z_world), 0, 1) : 0
    # DYNAMIC by design (gated on an active goal, graded by commit x proximity) so it
    # releases toward internal_planning during deliberation / between-goals, preserving
    # genuine mode competition (NOT the 464b "100% external_task, 0 switches" saturation
    # degeneracy). All no-op default -> bit-identical OFF. See MECH-266 / SD-032a.
    use_external_task_drive: bool = False
    # engagement -> external_task affinity logit (mode SELECTION pull).
    external_task_drive_affinity_weight: float = 1.0
    # engagement -> salience_aggregate (enables the MECH-259 switch INTO external_task).
    external_task_drive_salience_weight: float = 1.0
    # beta-elevation (committed motor program) term in the engagement scalar.
    external_task_drive_commit_weight: float = 1.0
    # goal_proximity term in the engagement scalar.
    external_task_drive_proximity_weight: float = 1.0
    # gate engagement on goal_state.is_active() (no active goal -> engagement 0).
    external_task_drive_require_goal_active: bool = True

    # SD-032c: AIC-analog interoceptive-salience / urgency module.
    # Master switch -- when True, REEAgent instantiates an AICAnalog that
    # reads z_harm_a_norm + drive_level + beta_gate_elevated + operating_mode
    # (from the SD-032a coordinator's previous tick) and emits:
    #   aic_salience  -- fed to the coordinator via update_signal() as the
    #                    MECH-259 urgency-trigger source.
    #   harm_s_gain   -- drive- and mode-gated multiplier on z_harm_s that
    #                    REPLACES the raw beta_gate.is_elevated check in the
    #                    SD-021 descending pain-modulation path. This is the
    #                    subsumption rerouting -- SD-021 continues to live
    #                    behind harm_descending_mod_enabled but now routes
    #                    through the AIC gain when use_aic_analog=True.
    # False = disabled (default, backward compat).
    use_aic_analog: bool = False
    # EMA alpha for the interoceptive baseline on z_harm_a_norm. ~50-step
    # window matches MECH-205 pe_ema_alpha convention.
    aic_baseline_alpha: float = 0.02
    # drive_coupling: how strongly SD-012 drive_level scales aic_salience.
    # aic_salience = urgency_ratio * (1 + drive_coupling * drive_level)
    # Non-zero is REQUIRED for the SD-032c falsification signature (same
    # z_harm_a -> different mode-switch behaviour across drive regimes).
    aic_drive_coupling: float = 1.0
    # urgency_threshold: diagnostic threshold on aic_salience for the
    # urgency_signal flag (reporting only; coordinator's switch_threshold
    # is the authoritative MECH-259 trigger).
    aic_urgency_threshold: float = 1.0
    # base_attenuation: maximum descending attenuation of z_harm_s when the
    # agent is in fully committed external_task with no drive protection.
    # Matches the historical descending_attenuation_factor=0.5 default.
    aic_base_attenuation: float = 0.5
    # drive_protect_weight: alterable-configuration knob flagged in the
    # SD-032c spec. drive_protect = max(0, 1 - drive_protect_weight * drive).
    #  +1.0 (default): depleted agent -> no attenuation (preserve harm
    #    signal for the struggling agent, so SD-032c can still trigger).
    #   0.0: drive-independent attenuation (pure legacy SD-021 reading).
    #  -1.0: depleted agent -> more attenuation (opposite-sign reading,
    #    testable alternative).
    aic_drive_protect_weight: float = 1.0
    # Uniform weight on optional extra salient-event signals passed to
    # AICAnalog.tick(extra_salient=...) -- unexpected z_goal drop, reward
    # surprise, irreversibility. Zero = no-op (default).
    aic_extra_weight: float = 0.0

    # SD-032d: PCC-analog (attention partition / metastability).
    # Master switch -- when True, REEAgent instantiates a PCCAnalog that
    # produces a scalar pcc_stability in [0, 1] from a task-success EMA,
    # SD-012 drive_level (fatigue), and steps-since-last-offline-phase.
    # The scalar is fed to the salience coordinator via
    # update_signal("pcc_stability", ...) BEFORE coordinator.tick(), so the
    # MECH-259 effective_threshold is scaled by (1 + stability_scaling *
    # pcc_stability). High stability -> coordinator resists transitions;
    # low stability -> transitions happen at lower salience (rest-driven
    # relaxation when the agent is fatigued or has been held externally
    # for a long time). False = disabled (default, backward compat).
    use_pcc_analog: bool = False
    # EMA alpha for the task-success signal. Smaller = slower adaptation.
    # 0.02 matches the MECH-205 / SD-032c convention (~50-step window).
    pcc_success_alpha: float = 0.02
    # Centred contribution of (success_ema - 0.5) to stability. With
    # success_ema in [0, 1] this contribution sits in [-w/2, +w/2].
    pcc_success_weight: float = 0.5
    # Subtractive contribution of drive_level to stability. With
    # drive_level in [0, 1] this contribution sits in [-w, 0].
    pcc_fatigue_weight: float = 0.5
    # Steps over which steps-since-last-offline-phase saturates the
    # offline_recency factor at 1.0. ~500 steps ~ V3 inter-quiescence
    # interval from sleep experiments.
    pcc_offline_recency_window: int = 500
    # Subtractive contribution of offline_recency to stability. Drives
    # the rest-driven relaxation signature.
    pcc_offline_weight: float = 0.3
    # Additive baseline before clipping. With baseline=0.5 and all other
    # contributions zero, stability=0.5 (neutral metastability).
    pcc_stability_baseline: float = 0.5

    # SD-032e: pACC-analog slow autonomic write-back.
    # Master switch -- when True, REEAgent instantiates a PACCAnalog that
    # accumulates tanh-normalised z_harm_a magnitude into a bounded
    # drive_bias, gated by coordinator.write_gate("autonomic") (MECH-261
    # mode-conditioned: active in external_task, attenuated in planning /
    # replay / offline). drive_bias is then added to the base drive_level
    # passed to GoalState.update(), SalienceCoordinator.tick(), AICAnalog,
    # and PCCAnalog. Biological architectural path for chronic-pain-like
    # sensitisation (Baliki 2012) compressed into the V3 drive_level proxy.
    # False = disabled (default, backward compat).
    use_pacc_analog: bool = False
    # EMA alpha for the drive_bias accumulator. 0.002 ~ 347-step half-life,
    # multi-episode accumulation per Guo 2018 days-timescale ACC mGluR5 LTP.
    # Faster alphas (>=0.01) are "fast end of biological plausibility" per
    # the SD-032e scoping lit-pull -- only use them for diagnostic probes.
    pacc_drive_alpha: float = 0.002
    # Scale on the tanh-normalised z_harm_a target before accumulation.
    # 1.0 = full range; <1.0 attenuates the write magnitude.
    pacc_drive_scale: float = 1.0
    # Symmetric cap |drive_bias| <= this. Prevents runaway sensitisation.
    # 0.5 = can shift drive_level by up to +/-0.5 when effective_drive is
    # evaluated (then re-clipped to [0, 1]).
    pacc_drive_bias_cap: float = 0.5
    # Below this z_harm_a norm, the target is treated as zero (Guo 2018
    # reversibility -- no write in the absence of sustained affective input).
    pacc_z_harm_a_min: float = 0.0
    # Fractional decay on drive_bias per note_offline_entry() call. Default
    # 0.0 = NO offline decay (pACC accumulator is persistent across sleep).
    # A non-zero value instantiates a DISTINCT sleep-recalibration claim
    # that the SD-032e scoping lit-pull flagged as not yet grounded in
    # separate literature -- leave 0.0 unless explicitly queuing that
    # probe as its own experiment.
    pacc_offline_decay: float = 0.0

    # MECH-302 substrate: suffering-derivative comparator.
    # Non-trainable rolling-window descent detector on the z_harm_a norm stream.
    # Fires a relief_completion_event when the window shows a sustained drop
    # >= suffering_drop_threshold, provided the initial norm is above
    # suffering_min_initial_norm (prevents spurious fires on a quiet stream).
    # Event reuses the MECH-057a / MECH-091 pipeline: beta_gate.release() +
    # MECH-094 categorical tag write (VALENCE_LIKING) at the current z_world.
    # Simulation gating: tick() returns False when hypothesis_tag is set
    # (waking-stream signal only, per MECH-094).
    # False = disabled (default, backward compat; bit-identical to pre-MECH-302).
    use_suffering_derivative_comparator: bool = False
    # Rolling window length (ticks). Initial norm = norm_buffer[0]; total drop
    # = norm_buffer[0] - norm_buffer[-1] must reach drop_threshold to fire.
    suffering_window_length: int = 5
    # Required total norm drop across the window for event to fire.
    suffering_drop_threshold: float = 0.10
    # Minimum initial z_harm_a norm required for the window to be scored.
    # Below this the stream is already quiet; the event is suppressed.
    suffering_min_initial_norm: float = 0.05
    # Magnitude written to VALENCE_LIKING at the current z_world location when
    # the relief_completion_event fires (MECH-094 categorical tag write).
    relief_completion_weight: float = 1.0

    # SD-051 / MECH-304: cue-specific conditioned safety prediction store.
    # ConditionedSafetyStore maintains an EMA prototype of z_world at MECH-302
    # relief-completion event ticks. Cosine similarity to current z_world yields
    # safety_prediction; when above threshold + beta_gate elevated: release commitment.
    # Encoding pathway: EMA prototype (dorsal striatum / dlPFC analog).
    # Expression pathway: similarity gate -> beta_gate.release() (IL->CeA analog).
    # MECH-094 gating: update() returns 0.0 when hypothesis_tag set (waking only).
    # False = disabled (default, backward compat; bit-identical to pre-SD-051).
    use_conditioned_safety_store: bool = False
    # EMA update rate per MECH-302 event tick (0, 1].
    safety_store_ema_alpha: float = 0.1
    # Per-step prototype decay toward zero (forgetting without reinforcement).
    safety_store_decay_rate: float = 0.001
    # Prototype L2 norm below which query returns 0.0 (store not yet loaded).
    safety_store_min_norm: float = 0.1
    # Cosine similarity threshold for commitment release gate.
    safety_store_threshold: float = 0.5
    # Magnitude written to VALENCE_LIKING when safety gate releases commitment.
    safety_store_commitment_weight: float = 1.0

    # MECH-303: contextual passive safety terrain.
    # Accumulates harm-absence signal per-step at current z_world (sense()).
    # Evaluated in select_action() to release beta_gate when accumulated safety
    # exceeds contextual_safety_release_threshold in the current context.
    # Anatomical analog: vmPFC + vHipp-to-PL contextual safety store (Laing 2022,
    # Kreutzmann 2020, Meyer 2019). Separate from MECH-304 cue-specific store.
    # False = disabled (default, backward compat; bit-identical to pre-MECH-303).
    use_contextual_safety_terrain: bool = False
    # Increment added to safety terrain per harm-absent step. Small by design --
    # contextual safety builds slowly over repeated exposure (diffuse/passive update).
    contextual_safety_accum_weight: float = 0.01
    # z_harm_a norm below this threshold counts as "harm absent" for accumulation.
    contextual_safety_harm_threshold: float = 0.05
    # Safety terrain scalar at current z_world must exceed this threshold
    # (with beta_gate elevated) to trigger commitment release.
    contextual_safety_release_threshold: float = 1.0

    # MECH-095: TPJ agency comparator (self/other attribution on z_self).
    # When True, REEAgent stores an E2 efference-copy prediction at action
    # selection and compares it against the next observed z_self during sense().
    # This is a diagnostic/runtime ownership signal only; no automatic residue
    # gating is imposed by default because many experiment loops still call
    # update_residue() before the next observation has been sensed.
    use_tpj_comparator: bool = False
    tpj_agency_threshold: float = 0.5

    # SD-033a: Lateral-PFC-analog (rule/goal substrate, MECH-261 primary consumer).
    # Master switch -- when True, REEAgent instantiates a LateralPFCAnalog that
    # maintains a rule_state vector updated via gate-modulated EMA using
    # write_gate("sd_033a") as the effective-eta multiplier, and emits a
    # per-candidate score_bias (initial output = 0 because the head's last
    # Linear is zeroed at init -- backward-compatible).
    # False = disabled (default).
    use_lateral_pfc_analog: bool = False
    # Rule-state vector dimensionality. Small by intent (rule summary, not
    # a full working-memory buffer).
    lateral_pfc_rule_dim: int = 16
    # Base EMA rate; effective rate = lateral_pfc_update_eta * gate each
    # tick. 0.05 gives ~20-step rise time under gate=1.0.
    lateral_pfc_update_eta: float = 0.05
    # Blend weight on mean(z_world) in the rule-update source: source =
    # delta_proj(z_delta) + world_pool_weight * world_proj(z_world).
    lateral_pfc_world_pool_weight: float = 0.5
    # Clamp on |score_bias|; keeps rule-bias from dominating E3.
    lateral_pfc_bias_scale: float = 0.1
    # Bias-head hidden-layer width.
    lateral_pfc_hidden_dim: int = 32
    # ARC-062 GAP-C: route GatedPolicy discriminator output into rule_state
    # update source. Default False = no-op, bit-identical backward compat.
    lateral_pfc_use_discriminator_source: bool = False
    # Weight of discriminator contribution in the source formula.
    lateral_pfc_discriminator_pool_weight: float = 0.3
    # ARC-062 GAP-D: when True, rule_bias_head's last Linear is NOT zeroed at
    # init (keeps random init so gradient training works from tick 1).
    # Default False = last Linear zeroed (bit-identical landing behaviour).
    lateral_pfc_train_rule_bias_head: bool = False

    # ARC-063 v1: distributed CandidateRule field (the non-Bayesian rule-creator
    # resolving arc_062_rule_apprehension:GAP-B). Mints distinct subspace-
    # partitioned rule slots on detected recurring (context -> action-object ->
    # outcome) regularities; the available-AND-context-matched rules combine into
    # a differentiated rule_state vector handed to SD-033a LateralPFCAnalog. The
    # structural fix for the 543/598b rule_state collapse. All no-op default,
    # bit-identical OFF; requires use_lateral_pfc_analog=True (the consumer).
    # Design doc: REE_assembly/docs/architecture/arc_063_candidate_rule_field.md.
    use_candidate_rule_field: bool = False
    crf_n_slots: int = 16
    crf_rule_dim: int = 16
    crf_mint_recurrence_threshold: int = 3
    crf_tolerance_floor: float = 0.3
    crf_tolerance_conflict_gain: float = 1.0
    crf_availability_alpha: float = 0.1
    crf_availability_decay: float = 0.005
    crf_eligibility_window: int = 20
    crf_context_match_threshold: float = 0.5
    crf_seed_from_arc062: bool = True
    # ARC-062 amend (V3-EXQ-654 GAP-B maturity): when True, the field's reset()
    # does NOT clear the live rule pool / recurrence counters, so the pool
    # accumulates ACROSS the per-episode agent.reset() instead of cold-starting
    # every ~26-tick episode. No-op default = bit-identical per-episode wipe.
    crf_persist_rules_across_episode_reset: bool = False
    # ARC-063 amend (V3-EXQ-654b GAP-B maturity): mature-pool gate/credit/retire
    # dynamics so a differentiated, persistently-active pool of >=2 rules can form
    # (the 654/654a/654b crf_frac_active ~0.13 / crf_max_pairwise_rule_dist 0.0
    # churn fix). All consulted only when crf_mature_pool_dynamics=True; default
    # OFF -> bit-identical legacy dynamics.
    crf_mature_pool_dynamics: bool = False
    crf_mature_availability_decay: float = 0.001
    crf_mature_retire_floor: float = 0.05
    crf_mature_availability_alpha_negative: float = 0.02
    crf_mature_tolerance_floor: float = 0.15
    crf_mature_tolerance_conflict_gain: float = 0.25
    crf_mature_mint_block_threshold: float = 0.8
    crf_mature_mint_protection_ticks: int = 30
    # When True the CRF context (mint/match key) is sourced from
    # e2.world_forward(z_world, prev_action) instead of raw z_world, mirroring the
    # ARC-065 GAP-A re-sourcing so the mint-block does not collapse under low
    # raw-z_world spread. Default OFF -> raw z_world (bit-identical).
    crf_context_from_e2_world_forward: bool = False
    # --- crf-availability-maintenance (V3-EXQ-666 successor; ARC-063 amend).
    # Activity-silent maintenance of a differentiated rule pool: silence no longer
    # erodes availability (only exception/interference does), and the readiness
    # readout moves to the MAINTAINED pool. All consulted only when
    # crf_availability_maintenance=True; default False -> bit-identical legacy
    # path. Designed to run WITH crf_mature_pool_dynamics +
    # crf_context_from_e2_world_forward (the differentiation source). See
    # CandidateRuleFieldConfig for per-knob semantics.
    crf_availability_maintenance: bool = False
    crf_maintenance_floor: float = 0.45
    crf_maintenance_decay: float = 0.0
    crf_engaged_sustain: bool = False
    crf_engaged_sustain_rate: float = 0.1
    crf_maintained_reactivation_threshold: float = 0.0
    # CRF conflict-gate calibration amend (V3-EXQ-654d successor;
    # crf-availability-maintenance at the CRF locus, UNGATED from GAP-A). All no-op
    # default; consulted only under crf_mature_pool_dynamics. See the
    # CandidateRuleFieldConfig docstring for the three faults these address.
    crf_mature_context_match_threshold: float = -1.0
    crf_tolerance_conflict_cap: int = -1
    crf_maintenance_couple_to_theta: bool = False

    # SD-033b: OFC-analog (specific-outcome / task-structure substrate,
    # MECH-261 second consumer; MECH-263 falsification target). When True,
    # REEAgent instantiates an OFCAnalog that maintains a state_code vector
    # updated via gate-modulated EMA using write_gate("sd_033b") and emits
    # a per-candidate score_bias (initial output = 0 because the head's
    # last Linear is zeroed at init -- backward-compatible).
    use_ofc_analog: bool = False
    # state_code dimensionality. Small by intent (task-structure summary).
    ofc_state_dim: int = 16
    # Base EMA rate; effective rate = ofc_update_eta * gate each tick.
    ofc_update_eta: float = 0.05
    # Blend weight on mean(z_harm) in the state-update source: source =
    # world_proj(z_world) + outcome_pool_weight * outcome_proj(z_harm).
    # Set 0.0 to ablate the outcome-context contribution.
    ofc_outcome_pool_weight: float = 0.5
    # Clamp on |score_bias|.
    ofc_bias_scale: float = 0.1
    # Bias-head hidden-layer width.
    ofc_hidden_dim: int = 32
    # Dimensionality of z_harm consumed for the outcome-context source.
    # 0 disables the outcome projection entirely (state_code = world only).
    # When >0 must match LatentStackConfig.z_harm_dim.
    ofc_harm_dim: int = 0
    # When True, enables OFCAnalog.query_outcome() -- the MECH-263
    # specific-outcome oracle path. Delegates to E2HarmSForward on the
    # agent for prospective outcome prediction per candidate action.
    # Requires use_e2_harm_s_forward=True. Default False (backward compat).
    use_ofc_outcome_oracle: bool = False
    # SD-033b GAP-8 (mirror of SD-033a GAP-D): when True, OFCAnalog's
    # state_bias_head last Linear is NOT zeroed at init (random init preserved),
    # so the head can be added to an experiment optimizer and trained via the
    # E3 score-aggregation gradient. Default False = last Linear zeroed
    # (bit-identical to the original SD-033b landing; bias output stays 0 until
    # deliberately trained -- the deferred trained-OFC-head behavioural arm).
    ofc_train_state_bias_head: bool = False
    # SD-033b commitment_closure:GAP-8 devaluation-head DECOUPLE (failure_autopsy
    # V3-EXQ-485l, 2026-06-22). When True, OFCAnalog builds a SECOND output head
    # (devaluation_bias_head) sharing the state_code+candidate input but with its
    # OWN clamp (ofc_devaluation_bias_scale, independent of ofc_bias_scale), so the
    # devalued re-ranking magnitude is not traded against the C2 discrimination
    # range under the single +/-ofc_bias_scale clamp (the 485k saturate / 485l
    # undershoot bracket). Default False = no second head, bit-identical (existing
    # experiments read only compute_bias).
    use_ofc_devaluation_head: bool = False
    # Independent clamp on |devaluation_bias_head| output. Larger than ofc_bias_scale
    # by intent so an in-band re-ranking gain produces a supra-floor differentiated
    # devalued range without saturating. Consulted only when use_ofc_devaluation_head.
    ofc_devaluation_bias_scale: float = 2.0
    # Mirror of ofc_train_state_bias_head for the devaluation head: when True, the
    # devaluation_bias_head last Linear is NOT zeroed at init so it trains via the
    # E3 score-aggregation gradient (the 485-lineage behavioural retest optimizer).
    # Default False = last Linear zeroed (bias output stays 0 until trained).
    ofc_train_devaluation_head: bool = False

    # ----------------------------------------------------------------
    # ARC-062 (Phase 1, weak reading): gated-policy heads + learned
    # context discriminator. Substrate for the rule-apprehension layer
    # MECH-309 says is required for behavioural diversity in non-
    # stationary environments. Default-off; bit-identical OFF.
    # See evidence/planning/arc_062_rule_apprehension_plan.md Phase 1.
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates a GatedPolicy that
    # consumes (z_world, z_self, z_harm_a) for the discriminator and
    # per-candidate first-step z_world summaries for the heads, and
    # composes its per-candidate score_bias additively into the dACC /
    # lateral_pfc / ofc / mech295 chain before E3.select(). Phase 1
    # has NO connection to SD-033a (that wiring is Phase 3 of the
    # plan-of-record).
    use_gated_policy: bool = False
    # N=2 heads at Phase 1 (substrate-constrained by SD-054 reef-vs-
    # forage two-mode partition per Pull A R2 verdict). Multi-head
    # extension is Phase 4 / GAP-E.
    gated_policy_n_heads: int = 2
    # Context-discriminator hidden-layer width.
    gated_policy_disc_hidden: int = 24
    # Init scale for discriminator weights (small to prevent early
    # over-commitment to one head before either has differentiated).
    gated_policy_disc_init_scale: float = 0.1
    # Per-head MLP hidden-layer width.
    gated_policy_head_hidden: int = 32
    # Clamp on |gated_score_bias|. Mirrors lateral_pfc_bias_scale so
    # Phase 1 magnitudes are comparable to existing PFC contributions.
    gated_policy_bias_scale: float = 0.1
    # Symmetry-broken init magnitude on the heads' last-Linear bias
    # (head_0 +offset, head_1 -offset). Heads can differentiate from
    # step 0 even before the discriminator has training signal.
    gated_policy_head_init_bias_offset: float = 0.05
    # ARC-062 GAP-B option-2: head-input first-action one-hot augmentation.
    # When True, REEAgent.__init__ sets GatedPolicyConfig.use_first_action_onehot=True
    # and first_action_dim=config.e2.action_dim so each scoring head receives
    # [K, world_dim + action_dim] rather than [K, world_dim], bypassing the
    # E2 world-forward compression diagnosed in EXQ-543e (0.22% signal ratio).
    # Default False = no-op, bit-identical backward compat.
    gated_policy_use_first_action_onehot: bool = False
    # INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection
    # crystallization. Master toggle read by the infant-curriculum
    # experiment harness. When True: GatedPolicy is built with
    # crystallize_enabled=True and ResidueField with ewc_enabled=True,
    # and the experiment installs an on_phase3_entry callback that calls
    # agent.gated_policy.crystallize() + agent.residue_field.
    # snapshot_ewc_anchor() at the Phase 2->3 transition. The actual
    # crystallize()/snapshot fire only at Phase 3; with this flag False
    # everything is bit-identical (Nikishin 2023 plasticity injection +
    # Kirkpatrick 2017 EWC residue write-protect). Default False = no-op.
    crystallize_at_phase3: bool = False
    # Plastic expansion-MLP hidden width (Nikishin 2023 fresh-layer
    # channel added at crystallization). Mirrors gated_policy_head_hidden.
    gated_policy_crystallize_expansion_hidden: int = 32
    # ARC-062 differential-heads reparameterization (MECH-333). Default False.
    gated_policy_use_differential_heads: bool = False
    gated_policy_differential_bias_scale: float = 0.1
    # ARC-062 GAP-B mode-separation floor (543i autopsy). Default 0 = off.
    gated_policy_mode_separation_floor: float = 0.0
    # P1 aux: penalize discriminator w near 0.5 during outcome-coupled train.
    gated_policy_p1_w_deviation_aux_weight: float = 0.0
    # MECH-334 residue-field EWC penalty weight (passed to
    # ResidueConfig.ewc_lambda when crystallize_at_phase3=True). 0.0 =
    # anchor captured but penalty contributes nothing (safe default).
    residue_ewc_lambda: float = 0.0

    # ----------------------------------------------------------------
    # MECH-313 (ARC-065): stochastic_noise_floor (LC-NE tonic / SAC
    # max-entropy policy regularisation analog). State-independent
    # softmax-temperature lift that prevents complete deterministic
    # collapse of the policy. Distinct from MECH-260 dACC anti-recency
    # (state-dependent recency penalty); Q-045 falsifies whether they
    # collapse into a single substrate. See ree_core/policy/noise_floor.py
    # and Pull 1 SYNTHESIS verdicts (R1 BOTH-CHANNELS-NEEDED, R2 LC-NE
    # tonic load-bearing, R4 continuous every tick) for resolved defaults.
    use_noise_floor: bool = False
    # SAC-entropy-bonus analog; constant additive lift on the softmax
    # temperature. Default 0.1 = modest +10% of the baseline E3
    # temperature (1.0). Q-043 calibrates magnitudes via parametric
    # sweep (V3-EXQ-543b/c).
    noise_floor_alpha: float = 0.1
    # Hard lower bound on the effective softmax temperature (so the
    # policy never collapses to argmax even under annealing schedules
    # that drive the baseline below 1.0). Default 1.0 matches the
    # existing E3 baseline so well-formed callers clear the floor.
    noise_floor_min_temperature: float = 1.0

    # ----------------------------------------------------------------
    # MECH-314 (ARC-065): structured_curiosity_bonus. Frontopolar
    # exploration / EFE analog. Sibling to MECH-313 stochastic_noise_floor.
    # Three sub-flavours registered separately (Pull 1 SYNTHESIS R3 +
    # Q-044): MECH-314a striatal novelty (Wittmann 2008), MECH-314b
    # frontopolar uncertainty-driven curiosity (Daw 2006 / Friston EFE),
    # MECH-314c learning-progress / intrinsic motivation (Schmidhuber /
    # Pathak; least biologically anchored). See
    # ree_core/policy/structured_curiosity.py for the algorithm; Phase 1
    # honest-scoping note: 314a is genuinely per-candidate (RBF distance
    # to ResidueField centers), 314b and 314c are state-dependent global
    # scalars broadcast across [K] (per-candidate refinement deferred to
    # Phase 2 follow-on requiring an E1 forward-variance head).
    use_structured_curiosity: bool = False
    # Sub-flavour switches. Consulted only when use_structured_curiosity
    # is True. Defaults are True so flag-set Q-044 ablation is "turn the
    # master on, then flip individual sub-flavours off" (matches the
    # cluster-registration verdict NOT to collapse them prematurely).
    use_curiosity_novelty: bool = True
    use_curiosity_uncertainty: bool = True
    use_curiosity_learning_progress: bool = True
    # Per-sub-flavour magnitudes. Default 0.05 each; Q-043 (relative
    # weight calibration MECH-313 vs MECH-314) and Q-044 (sub-flavour
    # independence) are the empirical resolution paths.
    curiosity_novelty_weight: float = 0.05
    curiosity_uncertainty_weight: float = 0.05
    curiosity_learning_progress_weight: float = 0.05
    # Hard clamp on |total curiosity bias|. Mirrors lateral_pfc_bias_scale
    # so Phase 1 magnitudes are comparable to existing PFC-side score_bias
    # contributions and curiosity cannot dominate the dACC / lateral_pfc /
    # ofc / mech295 score-bias chain at extreme sub-signal magnitudes.
    curiosity_bias_scale: float = 0.1
    # 314c learning-progress EMA smoothing (~10-tick window at default).
    curiosity_lp_ema_alpha: float = 0.1
    # 314c |PE_t - PE_{t-K}| window K (Schmidhuber 1991 first-difference).
    curiosity_lp_window_k: int = 5

    # ----------------------------------------------------------------
    # MECH-314a Phase 2 (Candidate 5A): novelty-source selection +
    # first-action one-hot augmentation. Architecture doc
    # REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
    # section 6. Phase 1 sourced 314a novelty from the harm-coupled
    # ResidueField RBF centers only (empty on harm-free episodes -> zero
    # per-candidate spread, the V3-EXQ-571 false-positive). Phase 2 adds a
    # rolling z_world visitation buffer as an alternate novelty source (always
    # populated from tick 1) plus a substrate-robustness first-action one-hot
    # augmentation that engages when the un-augmented per-candidate spread
    # collapses (canonical case: SD-056 OFF). ALL DEFAULTS REPRODUCE PHASE 1
    # EXACTLY (bit-identical OFF) so in-flight baselines are uncontaminated.
    #
    # Novelty comparison set: "residue" (Phase-1 ResidueField centers; default,
    # bit-identical), "visitation" (rolling z_world buffer), "auto" (visitation
    # when the buffer is non-empty, else fall back to residue).
    curiosity_novelty_source: Literal["residue", "visitation", "auto"] = "residue"
    # Rolling z_world visitation buffer maxlen (waking-tick z_world states;
    # collections.deque on REEAgent). Consulted only when curiosity_novelty_source
    # is "visitation" or "auto".
    curiosity_visitation_buffer_len: int = 256
    # Master switch for the first-action one-hot augmentation leg. When True
    # (and the policy below is not "never"), per-candidate signatures are
    # augmented with the first-action one-hot before the RBF distance so the
    # action carries per-candidate spread by construction (ARC-062 GAP-B
    # bypass template). Default False -> no augmentation, bit-identical.
    curiosity_use_first_action_onehot: bool = False
    # When to engage augmentation: "never" (default, bypass), "always"
    # (augment every tick), "auto" (engage after the un-augmented per-candidate
    # spread stays below curiosity_min_spread_threshold for
    # curiosity_min_spread_consecutive_ticks; disengage when it recovers).
    curiosity_first_action_augmentation_policy: Literal[
        "never", "auto", "always"
    ] = "never"
    # Auto-policy: per-candidate spread (max-min of the un-augmented novelty
    # vector) below which a tick counts toward the consecutive-below streak.
    curiosity_min_spread_threshold: float = 0.01
    # Auto-policy: consecutive below-threshold ticks required to engage the
    # augmentation (and, symmetrically, an at-or-above tick disengages it).
    curiosity_min_spread_consecutive_ticks: int = 5
    # MECH-314a Phase-2 amend (V3-EXQ-648 autopsy 2026-06-07): source of the
    # per-candidate novelty signature fed to StructuredCuriosity. "proposer"
    # (default, bit-identical) uses the hippocampal proposer's first-step
    # z_world (trajectory.world_states[:,0,:]); its cross-candidate spread
    # collapses to <0.01 under monostrategy, zeroing both the 314a novelty
    # range and the auto-augmentation _candidate_spread key. "e2_world_forward"
    # rebuilds the signature from the SD-056-trained action-conditional
    # e2.world_forward(z0, a_i) predictions (cross-candidate spread ~0.11 when
    # SD-056 is trained), the representation the SD-056 readiness gate already
    # validates. Design-doc Candidate 1 source on the landed Candidate-5A
    # machinery.
    curiosity_candidate_source: Literal[
        "proposer", "e2_world_forward"
    ] = "proposer"

    # ARC-065 GAP-A (behavioral_diversity_isolation): source of the SHARED
    # per-candidate cand_world_summaries consumed by the E3-side bias channels
    # (lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor). "proposer"
    # (default, bit-identical) builds it from the collapsed proposer first-step
    # z_world trajectory.world_states[:,0,:], whose cross-candidate spread is
    # ~0 under monostrategy (cand_world_pairwise_dist=0.0000; V3-EXQ-614e
    # autopsy). "e2_world_forward" rebuilds it from the SD-056-trained
    # action-conditional e2.world_forward(z0, a_i) predictions so the candidate
    # pool carries class-level diversity into the bias channels -- the GAP-A
    # extension of the 2026-05-17 ARC-062 GAP-B first-action-onehot fix beyond
    # GatedPolicy, and the shared-channel sibling of curiosity_candidate_source
    # (MECH-314a Phase-2, which fixed the curiosity channel only). Composes with
    # the modulatory-bias-selection-authority gate (V3-EXQ-643a) so the
    # now-divergent bias reaches the committed argmin.
    candidate_summary_source: Literal[
        "proposer", "e2_world_forward"
    ] = "proposer"

    # ----------------------------------------------------------------
    # MECH-320 (ARC-066 child): tonic_vigor_coupling_score_bias. First child
    # mechanism for the non_deficit_action_drives family. Adds an additive
    # (or multiplicative-gain, falsifiable secondary) bias on E3 trajectory
    # scoring, biased toward action-trajectories and away from no-op
    # trajectories, scaled by a slow EWMA over the realised E3-score-receipt
    # stream gated by secondary internal-state modulators (energy / drive /
    # recent PE). See ree_core/policy/tonic_vigor.py and ARC-066 lit-pull
    # SYNTHESIS (evidence/literature/targeted_review_arc_066_tonic_vigor/
    # synthesis.md, lit_conf 0.789, supports). R1 substrate: mesolimbic DA-
    # vigor (Niv 2007 / Salamone & Correa 2012 / Beierholm 2013); LC-NE-
    # direction REJECTED per Kane et al. 2017. R3 form: additive primary,
    # multiplicative falsifiable secondary. R4 scalar: slow EWMA over
    # realised score-receipt (long-window avg-reward-rate per Niv 2007),
    # NOT internal-capacity composite.
    use_tonic_vigor: bool = False
    # EWMA half-life in ticks for the realised-score-receipt average. Long
    # window per R4 verdict (Niv 2007 long-run avg reward rate, not
    # short-window). Default 100 ticks ~= ~700-tick window to steady state.
    tonic_vigor_half_life: float = 100.0
    # Action-trajectory negative-bias weight (REE convention lower-is-better,
    # so favouring action means subtracting from action-trajectory scores).
    # Default 0.1; magnitudes calibrated empirically by V3-EXQ-547 successor.
    tonic_vigor_w_action: float = 0.1
    # No-op-trajectory positive-bias weight (opportunity-cost / ARC-068
    # complement). Default 0.1.
    tonic_vigor_w_passive: float = 0.1
    # Hard clamp on |per-candidate bias|. Mirrors lateral_pfc / curiosity
    # bias_scale so MECH-320 cannot dominate the dACC / lateral_pfc / ofc /
    # mech295 / curiosity score-bias chain at extreme reward histories.
    tonic_vigor_bias_scale: float = 0.1
    # Secondary modulator: energy reserve threshold below which vigor is
    # gated DOWN linearly. SD-012 owns z_goal pursuit in the deficit regime.
    tonic_vigor_gate_energy_min: float = 0.2
    # Secondary modulator: drive_level threshold above which vigor is
    # gated DOWN linearly. Above this the agent should be in z_goal-pursuit
    # mode; SD-012 + SD-037 handle the deficit-corner attribution.
    tonic_vigor_gate_drive_max: float = 0.7
    # Secondary modulator: recent prediction error threshold above which
    # vigor is gated DOWN linearly. Above this the agent should attend /
    # consolidate / sleep, not act vigorously.
    tonic_vigor_gate_pe_max: float = 1.0
    # Implementation-form selector. R3 verdict: additive primary;
    # multiplicative falsifiable secondary. Validated at TonicVigor
    # construction; "additive" or "multiplicative" only.
    tonic_vigor_form: str = "additive"
    # Action-class index treated as no-op. Default 0 matches MECH-279 PAG
    # freeze-gate convention.
    tonic_vigor_noop_class: int = 0
    # V3-EXQ-563 MECH-320: floor on the gated vigor scalar before per-
    # candidate bias computation. Default 0.0 = standard behaviour. Set > 0
    # to force a minimum positive tonic-vigor scalar for actuator tests.
    tonic_vigor_v_t_floor: float = 0.0

    # ----------------------------------------------------------------
    # MECH-341: e3_scoring_preserves_trajectory_class_diversity. Layer-B
    # diversity-preservation substrate. Triggered by V3-EXQ-608 P2 majority
    # R2a_e3_collapse_confirmed_large_gap finding (2026-05-26): CEM delivers
    # >=2 first-action classes (frac_pre_ge2=1.0) but E3 scoring collapses
    # to single class with mean_top2_class_gap 0.27-0.60 (large-gap, rules
    # out option 3 jittered tie-breaking). Two sub-flavours under one master,
    # mirroring MECH-314a/b/c precedent so Q-054 falsifier can dissociate
    # which one carries the load.
    #   Option 1 (entropy_bonus): per-candidate POSITIVE bias on candidates
    #     whose first-action class is over-represented in the pool.
    #     Composed into scores AFTER existing score_bias chain (dACC /
    #     lateral_pfc / ofc / mech295 / curiosity / tonic_vigor) and BEFORE
    #     softmax. Penalises homogenisation at the scoring step.
    #   Option 2 (stratified_select): partition candidates by first-action
    #     class, pick argmin within each class, sample across class-
    #     representatives via softmax. Replaces argmin in the committed
    #     selection path. Forces >= 2-class survival whenever the pool has
    #     >= 2 first-action classes.
    # Both sub-flavours bit-identical OFF by default. Behavioural validation
    # via R2.c rule in behavioral_diversity_isolation_plan.md.
    use_e3_score_diversity: bool = False
    use_e3_diversity_entropy_bonus: bool = True
    use_e3_diversity_stratified_select: bool = True
    e3_diversity_entropy_lambda: float = 0.5
    e3_diversity_entropy_bias_scale: float = 1.0
    e3_diversity_stratified_temperature: float = 1.0
    # MECH-341 amend (2026-06-01): within-class proportional sampling sharpness.
    # None = legacy argmin within each first-action class (bit-identical to
    # pre-amend MECH-341). When set to a positive float, sample within each
    # class via softmax(-class_scores / T) before the across-class softmax
    # step. Dissociates within-class diversity (Layer B sub-axis) from the
    # existing across-class diversity (`e3_diversity_stratified_temperature`).
    # See failure_autopsy_V3-EXQ-616_2026-05-31 Sections 7 + 10.
    e3_diversity_stratified_within_class_temperature: Optional[float] = None
    e3_diversity_min_classes_for_stratification: int = 2

    # modulatory-bias-selection-authority (2026-06-03): top-level mirror of the
    # E3Config master flag so build_from_ree_config can give the MECH-341
    # stratified across-class softmax authority too. When True, stratified_select
    # normalises class-representative scores to unit range before the
    # stratified_temperature softmax, so the diversity temperature acts on a
    # fixed scale (the 614d C2 failure was the across-class softmax collapsing
    # because the absolute class-representative score gap dominated the
    # temperature). Default False (bit-identical OFF). See E3Config fields
    # use_modulatory_selection_authority / modulatory_authority_gain.
    use_modulatory_selection_authority: bool = False

    # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a,
    # 2026-06-10). Top-level mirror of E3Config.use_modulatory_channel_routing
    # (set onto config.e3 by from_dims) PLUS the agent-side source/weight knobs
    # read in REEAgent.select_action via getattr. route_source selects which
    # channel-under-test's per-candidate representation is projected (parameter-
    # free, range-preserving) into the modulatory accumulator the authority
    # rescales: "none" (default, no routing, bit-identical) / "cand_world_summary"
    # (the [K, world_dim] world-summary channel -- ARC-065/569f cluster lead, the
    # genuine projection case) / "curiosity" / "gated_policy" / "mech295" /
    # "coherence" (each an already-computed per-candidate [K] bias, identity-
    # routed). route_weight sets the routed-channel proportion in _modulatory_accum
    # before the authority rescale.
    use_modulatory_channel_routing: bool = False
    modulatory_channel_route_source: str = "none"
    modulatory_channel_route_weight: float = 1.0

    # ----------------------------------------------------------------
    # ControlVector logging (recommendation B, four-signal control
    # adjudication 2026-06-07). Read-only, default-OFF telemetry that
    # assembles four separately-inspectable control signals each E3 tick:
    #   V_outcome -- primary value axis (E3 raw pre-bias scores)
    #   C_effort  -- dACC EVC effort term (control_required * candidate_effort)
    #   C_time    -- MECH-320 tonic-vigor +w_passive*v_t no-op / opportunity-
    #                cost-of-time half
    #   G_vigor   -- MECH-320 tonic-vigor -w_action*v_t action half + the
    #                MECH-313 noise-floor temperature lift
    # Purpose: make value / effort / opportunity-cost / vigor independently
    # inspectable and EXPOSE that C_time and G_vigor are both deterministic
    # functions of MECH-320's single v_t scalar (the ARC-068-vs-MECH-320
    # collapse). Pure telemetry: NO change to scoring or selection.
    # Surfaced on REEAgent._last_control_vector (read directly by experiment
    # scripts, the V3-EXQ-571 _last_score_bias_decomp pattern). Default False
    # -> bit-identical; the assembly block is skipped entirely.
    use_control_vector_logging: bool = False

    # ----------------------------------------------------------------
    # MECH-090 R-c conjunction: commit-entry predicate amendment from
    # rv-only to rv_low AND readiness_above_floor. Reading R-c per the
    # 2026-05-28 lit-pull synthesis (REE_assembly/evidence/literature/
    # targeted_review_connectome_mech_090/synthesis.md, commit 9e68c5ca8a;
    # Hanes & Schall 1996 accumulator-to-threshold + Cisek & Kalaska 2010
    # affordance competition + Roesch/Calu/Schoenbaum 2007 premature-
    # commit pathology -- all three load-bearing). Precipitating finding:
    # V3-EXQ-592 seed 42 (running_variance 2.7e-5 / nav_competence 0.0)
    # showed the agent satisfied the rv-only gate via degenerate trivial-
    # predictability. Substrate amendment lives at the BetaGate.elevate()
    # call sites in REEAgent.select_action (NOT in BetaGate.elevate
    # itself; that signature is preserved). See
    # REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
    # and ree_core/policy/commit_readiness.py.
    use_mech090_readiness_conjunction: bool = False
    # Readiness floor for the conjunction. Calibratable; lit-pull does
    # not pin a magnitude. Default 0.3 is the mid-low floor that V3-EXQ-
    # 592 seed 42's nav_competence=0.0 clearly fails to clear; a competent
    # agent reaching success rate >=30% clears. Validation experiment
    # V3-EXQ-592b sweeps this.
    mech090_readiness_floor: float = 0.3
    # CommitReadiness module master. Default False. Auto-armed True in
    # __post_init__ when use_mech090_readiness_conjunction is True (the
    # conjunction needs the readiness signal to consult). Setting this
    # True without the conjunction flag instantiates the module as a
    # passive diagnostic (readiness EMA advances but no commit-entry
    # gate consumes it).
    use_commit_readiness: bool = False
    # Nominal effective half-life (ticks) the EMA alpha targets.
    # Informational; alpha is the load-bearing knob. Default 20.
    commit_readiness_window: int = 20
    # Readiness EMA update rate. Default 0.1 (~10-tick half-life on the
    # EMA itself). Future Q-claim may sweep this.
    commit_readiness_ema_alpha: float = 0.1
    # Initial readiness value at construction / per-episode reset.
    # Default 1.0 (fail-open: an agent with no outcome history defaults
    # to "ready" so the conjunction reduces to rv-only behaviour until
    # real outcome data has been collected).
    commit_readiness_initial: float = 1.0

    # ----------------------------------------------------------------
    # MECH-342: maintenance-time readiness-driven commitment-release
    # coupling (B3b; commit-entry predicate is admission-only by design --
    # MECH-090 audit + motor-cessation lit-pull verdict 2026-06-02). The
    # SAME R-c readiness signals that MECH-090 AND-composes to admit a
    # commitment (score_margin + nav_competence) here drive a graded,
    # bounded-accumulation RELEASE of an already-elevated beta latch when
    # they degrade mid-commitment. Distinct from MECH-091 (threat),
    # ARC-028 (positive completion), MECH-269b/V_s (schema staleness),
    # MECH-340 (goal-level ghost-goal bank). Default OFF = bit-identical;
    # no consumer when agent.maintenance_release is None.
    # See ree_core/policy/commit_maintenance_release.py and
    # REE_assembly/docs/architecture/mech_342_commit_maintenance_release.md.
    # SD-061 difficulty-gated proposal-entropy regulator (MECH-343 blocker 2):
    # a stuck-state detector + a transient gain on the ARC-018 / CEM proposal
    # layer (wider candidate set + within-class CEM temperature), decaying as
    # the impasse clears. All no-op default -> bit-identical OFF. See
    # ree_core/cingulate/stuck_state_detector.py +
    # ree_core/policy/difficulty_gated_proposal_entropy.py +
    # REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md.
    use_difficulty_gated_proposal_entropy: bool = False
    # Stuck-state detector knobs.
    stuck_progress_window: int = 8
    stuck_progress_stall_eps: float = 0.01
    stuck_score_margin_floor: float = 0.05
    stuck_committed_diversity_window: int = 8
    stuck_committed_diversity_floor: float = 0.34
    stuck_choice_difficulty_ref: float = 0.05
    stuck_goal_salience_floor: float = 0.05
    stuck_ema_alpha_rise: float = 0.3
    stuck_ema_alpha_fall: float = 0.05
    stuck_threshold: float = 0.5
    stuck_combine_mode: str = "mean"
    # Difficulty-gated proposal-entropy regulator knobs.
    dgpe_candidate_widen_max: int = 8
    dgpe_temperature_gain_max: float = 1.0

    use_maintenance_release: bool = False
    # Decisiveness (score_margin) floor: at/below this the within-tick
    # decisiveness axis is "failing" and contributes release-pressure
    # deficit. Mirrors the MECH-090 commit_readiness_floor (0.05). Small
    # relative to the EXQ-608 mean_top2_class_gap range (0.27-1.96).
    maintenance_release_score_margin_floor: float = 0.05
    # Decisiveness reengage level (above the floor): at/above this the
    # decisiveness axis counts as "recovered" and contributes to leak
    # (the reengagement path). Hysteresis band = reengage - floor.
    maintenance_release_score_margin_reengage: float = 0.10
    # nav_competence (CommitReadiness EMA) floor: at/below this the
    # across-tick motor-readiness axis is "failing". Mirrors
    # mech090_readiness_floor (0.3).
    maintenance_release_nav_floor: float = 0.3
    # nav_competence reengage level (above the floor): at/above this the
    # nav axis counts as "recovered". Hysteresis band = reengage - floor.
    maintenance_release_nav_reengage: float = 0.5
    # Per-tick release-pressure accumulation rate. Pressure increments by
    # accumulation_rate * combined_deficit each maintenance tick where the
    # combined deficit is positive (conflict-scaled drift-to-bound,
    # Resulaj 2009 + Cavanagh/Frank 2011). At max deficit (1.0) it takes
    # release_bound/accumulation_rate = 5 sustained ticks to fire -- NOT a
    # one-shot Schmitt flag. Calibratable.
    maintenance_release_accumulation_rate: float = 0.2
    # Per-tick leak applied when BOTH axes have recovered (>= their
    # reengage level). Provides the reengagement path that guards the
    # premature-abort pole: a transient dip that recovers leaks back to 0
    # rather than committing to release. Default 0.1.
    maintenance_release_leak_rate: float = 0.1
    # Release fires when accumulated pressure reaches this bound.
    maintenance_release_bound: float = 1.0
    # Hard clamp on accumulated pressure (numerical guard against long
    # sustained-deficit runs overshooting the bound unboundedly).
    maintenance_release_pressure_cap: float = 1.5

    # ----------------------------------------------------------------
    # Commit/release-DURATION lever: graded natural-commit-occupancy release
    # (the rung-6 lever of f_dominance_conversion_ceiling -- the duration face,
    # PARALLEL to the selection-face MECH-448). Reduces the F-driven natural-
    # commit latch occupancy (~2400-2600 steps on strong seeds, V3-EXQ-460h) so
    # weak-natural-commit is the norm across seeds, dissolving the 460h MECH-445/
    # MECH-446 disjoint-certifier problem. Per BG-3 SYNTHESIS divergence D1 this
    # is a GRADED release (Thura/Cisek 2022 urgency + Jin 2014 behaviour-co-
    # extensive maintenance), NOT another fixed refractory clock. Pure-arithmetic
    # regulator (ree_core/policy/natural_commit_urgency.py) reusing
    # BetaGate.committed_run_length (no parallel latch module, ARC-106 G2). All
    # defaults no-op (bit-identical when use_natural_commit_urgency_release=False).
    # See REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
    use_natural_commit_urgency_release: bool = False
    # Sub-mode (1): Thura/Cisek graded urgency rising with the held duration,
    # rate scaled by the commit-entry decisiveness. Consulted only when master on.
    natural_commit_release_urgency_mode: bool = True
    # Sub-mode (2): Jin maintenance-co-extensive release (fire when the executed
    # action sequence completes). Consulted only when master on.
    natural_commit_release_action_extent_mode: bool = True
    # Per-tick base urgency increment (before the gap-scaled decisiveness scale).
    natural_commit_urgency_rate: float = 0.01
    # Urgency threshold at which the urgency-mode release fires.
    natural_commit_urgency_release_bound: float = 1.0
    # Hard clamp on accumulated urgency (numerical guard); must be >= bound.
    natural_commit_urgency_cap: float = 1.5
    # LOAD-BEARING: scales how strongly commit-entry decisiveness (gap_norm in
    # [0,1]) raises the urgency rate. >0 -> an F-decisive (monopolising) commit
    # accrues urgency faster -> the strongest-F holds are shortened most.
    # 0.0 -> a flat fixed-rate timeout (the contrasted "fixed refractory" control).
    natural_commit_gap_entry_sensitivity: float = 1.0
    # Grace ticks at the start of a committed run before urgency begins accruing.
    natural_commit_urgency_onset_ticks: int = 0
    # ----------------------------------------------------------------
    # SD-033e frontopolar-analog DE-COMMIT lever (V3-narrow MECH-264; 2026-07-09).
    # A DISTINCT de-commit-release lever for the DURATION face of the F-dominance
    # conversion ceiling (MECH-439), owed by failure_autopsy_MECH-445-cluster-
    # 715a-717_2026-07-07 ("the arming reliability gap needs a DISTINCT de-commit-
    # release lever, not more F-moderation"). Injects an ENTRY-RELATIVE, NON-F
    # counterfactual-improvement release-pressure term into the SAME
    # NaturalCommitUrgencyRelease accumulator (fires on urgency >= release_bound),
    # so it REQUIRES use_natural_commit_urgency_release=True to have any effect
    # (the agent instantiates FrontopolarAnalog independently, but the pressure it
    # emits is only consumed by NaturalCommitUrgencyRelease.tick). The counterfactual
    # value is the goal-proximity ADVANTAGE of the best foregone alternative over
    # the committed endpoint in z_world space -- sourced from goal-proximity, NOT F,
    # so it does not wash like the exhausted 709/711/713 input-reweighting route.
    # No-op when use_frontopolar_decommit is False (self.frontopolar is None) ->
    # bit-identical. See ree_core/pfc/frontopolar_analog.py section "SD-033e
    # V3-NARROW DE-COMMIT LANDING".
    use_frontopolar_decommit: bool = False
    # Gain on the entry-relative release pressure. 0.0 (default) => zero pressure =>
    # the frontopolar_gain=0 flat-urgency CONTRAST arm (bit-identical to the pure
    # NaturalCommitUrgencyRelease lever). The validation isolates the frontopolar
    # term by contrasting frontopolar_gain>0 (ON) vs frontopolar_gain=0.
    frontopolar_gain: float = 0.0
    # ----------------------------------------------------------------
    # Natural-commit LATCH-HOLD lever (rung-6 amend, 2026-06-21,
    # failure_autopsy_V3-EXQ-460i). SEPARATE from use_natural_commit_urgency_release
    # (the RELEASE): this is the HOLD that establishes a sustained natural-commit
    # beta-latch occupancy for the release to act on. V3-EXQ-460i found the lever
    # fired ZERO because the 460h sustained ~2400-step monolithic natural-commit
    # hold did not reproduce -- the active SD-034 de-commit control-plane fragments
    # the latch to ~1-tick blips even with the lever OFF, so there was no sustained
    # occupancy to shorten. When use_natural_commit_latch_hold=True, a natural commit
    # (result.committed) ARMS a hold; while armed AND the committed trajectory
    # persists, the beta latch is RE-ASSERTED each tick (kept elevated against the
    # de-commit churn) so the natural-commit occupancy sustains BY CONSTRUCTION. The
    # hold YIELDS to (does NOT override) the three PRINCIPLED releases: the MECH-091
    # genuine-threat urgency interrupt (safety -- never overridden), the rung-6
    # NaturalCommitUrgencyRelease's own duration release (so the lever CAN shorten
    # the hold -- the whole point), and an SD-034 closure de-commit (so the MECH-446
    # within-arm occupancy-drop DV stays measurable). Independent of the release
    # lever, so it arms in the ARM_LEVER_OFF baseline too (hold ON + release OFF ->
    # sustained reference; hold ON + graded urgency ON -> sustained then shortened).
    # Default False -> bit-identical (no arm, no re-assert). See
    # REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
    use_natural_commit_latch_hold: bool = False
    # Safety cap on how many ticks the hold re-asserts a single natural-commit run
    # before disarming (guards a degenerate config from latching forever).
    # 0 -> unbounded (the hold persists until a principled release / the committed
    # trajectory ends).
    natural_commit_latch_hold_max_ticks: int = 0
    # Closure-exclusive de-commit eval mode (rung-6 BUILD, 2026-06-22, the named
    # dissociable substrate from failure_autopsy_V3-EXQ-460j). The 460j ARM_LEVER_OFF
    # baseline showed the latch-hold NEVER armed (ncl_hold_reassert_total=0): it arms
    # only on a decisive natural commit (result.committed), which does not form on the
    # full closure-coupling substrate, so natural-commit and the SD-034 closure
    # de-commit were NON-DISSOCIABLE (no sustained occupancy for the de-commit to act
    # on). When closure_exclusive_decommit_eval=True, the eval makes beta elevation
    # CLOSURE-EXCLUSIVE -- _commit_for_beta is driven ONLY by _closure_commit_active
    # (the closure->beta coupling), the fragile F-driven result.committed path is
    # SUPPRESSED from beta elevation -- AND the natural-commit latch-hold arms on
    # _closure_commit_active. So a sustained commit occupancy forms via the closure
    # plane independently of the F-driven natural commit, and the SD-034 closure
    # de-commit (refractory) then acts on it (the existing yield-on-refractory at the
    # re-assertion site), making natural-commit and closure-de-commit DISSOCIABLE and
    # MECH-445 commit-intent + MECH-446 occupancy-drop co-measurable on the same seed.
    # PRECONDITIONS (enforced loud at REEAgent.__init__): requires
    # use_closure_commit_beta_coupling=True AND use_natural_commit_latch_hold=True.
    # Default False -> bit-identical (natural path unsuppressed, hold arms only on
    # result.committed). NOT a yield-clause patch (the refused 460k): this changes the
    # ARM SOURCE of the occupancy (the 460j root cause), not the release/yield logic.
    # See REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
    closure_exclusive_decommit_eval: bool = False

    # Closure-plane commit-ENTRY primitive (rung-6 amend, failure_autopsy_V3-EXQ-460k/460l
    # 2026-06-22; commitment_closure:GAP-4). The closure-exclusive de-commit eval above
    # arms the latch-hold ONLY via _closure_commit_active, which (agent.py:6365) gates on
    # e3._committed_trajectory is not None -- whose ONLY non-None writer is e3_selector.py
    # under `if committed:` (pure running_variance/F). On the 460j substrate the F-commit
    # never sustains (off_baseline_not_sustained), so _committed_trajectory is rarely
    # non-None -> the eval rarely arms (ncl_hold_closure_armed_total=0, the 460k/460l
    # signature). The design semantics ("closure-plane commitment forming, F-independent")
    # and the implementation (F-commit trajectory presence) contradict. When
    # use_closure_commit_entry=True an F-INDEPENDENT latch e3._closure_committed_active is
    # SET on a goal-active rule-directed commitment (goal_state.is_active() AND a trajectory
    # selected toward it AND lateral_pfc.rule_state norm above a floor -- faithful to SD-034:
    # closure governs rule-directed commitments) and CLEARED on the SD-034 closure fire /
    # de-commit refractory install / episode reset; _closure_commit_active (agent.py:6365)
    # is then the UNION (legacy F-commit OR the new closure-plane entry). So the
    # closure-exclusive eval can arm + sustain a closure-formed occupancy with ZERO
    # F-commits. PRECONDITIONS (enforced loud at REEAgent.__init__): requires
    # use_closure_commit_beta_coupling=True AND use_natural_commit_latch_hold=True. Default
    # False -> _closure_committed_active never set -> _closure_commit_active reduces to the
    # legacy `_committed_trajectory is not None` -> bit-identical for every existing run.
    # MECH-094: the latch is a waking control-state transition (no replay/memory write
    # surface); SET is a no-op under simulation/hypothesis_tag. See
    # REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
    use_closure_commit_entry: bool = False
    # rule_state norm floor above which a rule is considered "being followed" for the
    # closure-plane commit-entry SET predicate. Mirrors the closure_operator's
    # rule-stability precursor (completion_rule_delta_threshold * 10.0 trivially-stable
    # filter). Only read when use_closure_commit_entry=True.
    closure_commit_entry_rule_norm_floor: float = 0.01
    # Closure-plane commit-ENTRY TRAJECTORY primitive (rung-6 amend extension;
    # commitment_closure:GAP-4; failure_autopsy_V3-EXQ-460k/460l). The bool latch
    # use_closure_commit_entry above arms + SUSTAINS the beta occupancy (C-KEY) but a
    # bare bool cannot be STEPPED: the between-E3-tick path (agent.py) reads
    # e3._committed_trajectory to advance a committed program, so a closure-armed hold
    # with only a bool falls through to repeating _last_action (no closure-formed
    # program executes -- the C-STEP gap). When use_closure_commit_entry_trajectory is
    # on (REQUIRES use_closure_commit_entry), REEAgent.select_action ALSO installs the
    # goal/rule-directed result.selected_trajectory into a PARALLEL sticky latch
    # e3._closure_committed_trajectory; the between-tick stepping + the latch-hold
    # persistence + the _closure_commit_active arm gate all read the UNION
    # (_committed_trajectory OR _closure_committed_trajectory), so the closure-formed
    # occupancy advances an actual committed trajectory. CLEARED at the same de-commit /
    # closure-fire / reset sites as the bool latch. Default False -> the trajectory latch
    # is never set -> every union reduces to the bool-latch behaviour -> bit-identical to
    # the use_closure_commit_entry-only path. See
    # REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
    use_closure_commit_entry_trajectory: bool = False

    # ----------------------------------------------------------------
    # ARC-108 JOB-2 control-plane DRIVER pair (the dopaminergic driver of the
    # commit/maintain/de-commit machinery REE built but never gave its
    # neuromodulator; unified_dopamine_substrate_design_2026-06-22.md secs 3-6).
    # Both no-op-default -> bit-identical OFF. Pure-arithmetic / waking-only
    # (MECH-094); compose with MECH-090/342/SD-034 (gate + operator + refractory
    # kept as safety plumbing), no parallel module (ARC-106 G2). PROMOTES NOTHING.
    #
    # (c) rho_t MAINTENANCE RAMP -- REPLACES the flat-hold maintenance DRIVER of
    # the natural-commit latch-hold (the per-tick unconditional beta re-assert,
    # deviation B6 / the 460h ~2400-step monolithic hold). rho_t = goal-proximity
    # x value (reuse the goal/benefit valuation feeding F) ramps up while
    # approaching the goal and DECLINES past the proximity peak, so the hold
    # self-limits instead of running monolithically. Pure-arithmetic regulator
    # ree_core/policy/rho_maintenance_ramp.py. PRECONDITION (loud at __init__):
    # requires use_natural_commit_latch_hold=True (the hold this ramp drives).
    # Default False -> the latch-hold's flat re-assertion is unchanged
    # (bit-identical). See REE_assembly/docs/architecture/arc_108_job2_control_plane.md.
    use_rho_maintenance_ramp: bool = False
    # Below this rho_t the ramp releases (no value left to maintain).
    rho_hold_floor: float = 0.05
    # Release once rho_t has declined from its running peak by >= margin * peak
    # (the peaks-then-declines self-limit; the structural B6 fix).
    rho_release_margin: float = 0.5
    # Grace ticks at commit entry before the ramp may self-limit (let it rise to
    # its proximity peak first; guards against an early-tick spurious release).
    rho_onset_grace_ticks: int = 3
    #
    # (d) HABENULA negative-delta_t DE-COMMIT -- a new INTERNAL-scalar abort input
    # to the SD-034 ClosureOperator. A negative phasic RPE (delta_t = R_t - V-hat_t
    # below threshold = "worse than expected", the lateral-habenula analog) fires a
    # de-commit (closure) -- content-driven, dissociable from the latch's own
    # refractory state. ADDED alongside the existing refractory-timer release; the
    # operator/refractory/No-Go machinery is NOT replaced. Internal scalar only --
    # the routed GPi->habenula efferent drain stays V4. Reuses the SAME delta_t /
    # V-hat_t the ARC-108 JOB-1 learned-gating slice computes in
    # e3_selector.post_action_update (broadened to compute when JOB-1 OR this flag
    # is on). Requires use_closure_operator=True (forwarded onto
    # ClosureOperatorConfig.habenula_abort_enabled via the closure_decommit_hold_ticks
    # getattr-fallback pattern). Default False -> bit-identical OFF.
    use_habenula_decommit: bool = False
    # The habenula abort fires when delta_t < this (a negative "worse-than-expected"
    # margin). 0.0 = fire on any negative RPE; set more negative to require a larger
    # disappointment. Read only when use_habenula_decommit is True.
    habenula_decommit_delta_threshold: float = 0.0

    # ----------------------------------------------------------------
    # MECH-353: blocked-agency / control-failure affect stream (z_block).
    # Pure-arithmetic regulator (ree_core/affect/blocked_agency.py) that
    # integrates the SD-029 agency comparator applied to the action-outcome /
    # z_world channel (intended effect predicted by E2.world_forward, realised
    # effect diverges), behind an EXTERNAL-attribution gate (motor intact on
    # the z_self channel) and a CAPACITY gate (assert while capacity-belief
    # retained, hand off to z_harm_a as it collapses). All defaults no-op;
    # bit-identical when use_blocked_agency=False (agent.blocked_agency is None
    # and the new LatentState.z_block field stays None).
    # See REE_assembly/docs/architecture/mech_353_blocked_agency_zblock.md +
    # docs/architecture/affect_primitives.md (blocked_agency).
    use_blocked_agency: bool = False
    # Per-tick rise of z_block (alpha on outcome_mismatch) when an external
    # block is detected.
    blocked_agency_accumulation_rate: float = 0.2
    # Per-tick decay of z_block when the action succeeds (frustration leaks).
    blocked_agency_leak_rate: float = 0.1
    # Minimum normalised action-outcome comparator mismatch for a tick to count
    # as a block (below this the action produced ~its predicted effect).
    blocked_agency_outcome_mismatch_floor: float = 0.1
    # Minimum motor_agency (z_self efference-copy agency signal in (0,1]) for
    # the mismatch to be attributed to an EXTERNAL constraint rather than the
    # agent's own motor error (the attribution gate).
    blocked_agency_attribution_motor_floor: float = 0.5
    # Maps z_harm_a magnitude -> reduction in capacity-belief:
    # capacity = clip(1 - w * z_harm_a_norm, 0, 1). High suffering collapses
    # capacity-belief so the assert pole yields to the withdraw pole.
    blocked_agency_capacity_collapse_weight: float = 1.0
    # When True (default) z_block accumulates only while a goal is retained.
    blocked_agency_require_goal_active: bool = True
    # Hard clamp on the integrated z_block scalar.
    blocked_agency_z_block_cap: float = 1.5
    # ASSERT consumer weights (scaled by z_block_assert; REE lower-is-better).
    blocked_agency_assert_action_weight: float = 0.1     # negative bias on action (escalate effort)
    blocked_agency_assert_passive_weight: float = 0.1    # positive bias on no-op (penalise passivity)
    blocked_agency_assert_alt_action_weight: float = 0.1  # positive bias on the blocked action class (alt-action search)
    blocked_agency_assert_bias_scale: float = 0.1        # clamp on |assert bias|
    # DECOMMIT consumer: sustained asserting block -> release pressure; the
    # release itself is ARC-016-gated in agent.select_action.
    blocked_agency_decommit_bound: float = 1.0
    blocked_agency_decommit_consecutive_ticks: int = 5
    # ARC-016 gate on the z_block-driven decommit: release the commitment only
    # when E3 precision (confidence) is below this threshold (low confidence ->
    # the prefrontal-analogue permits abort). 0.0 = always permit (gate off).
    blocked_agency_decommit_arc016_precision_max: float = 0.0
    # Minimum predicted action-EFFECT magnitude (||E2.world_forward(zw,a) - zw||)
    # for the action-outcome comparator to fire at all. Below this the action was
    # not predicted to produce an effect, so there is nothing to be "blocked"
    # from (outcome_mismatch is forced to 0). This is the "intended,
    # predicted-to-succeed action" qualifier from the verdict, and it gates out
    # untrained-world_forward noise. The detector is delta-based:
    # outcome_mismatch = ||predicted_delta - realised_delta|| / (||predicted_delta||+eps).
    # NOTE: the action-outcome comparator is only DISCRIMINATIVE once
    # E2.world_forward is trained to be action-conditional (SD-056 contrastive
    # next-state loss); the validation experiment must train it in P0.
    blocked_agency_predicted_effect_floor: float = 0.05
    # Action class treated as no-op (matches MECH-279 / MECH-320 convention).
    blocked_agency_noop_class: int = 0

    # MECH-276: scientist-agent counterfactual-backed attribution feedstock.
    # The waking-phase mechanism that feeds the MECH-275 sleep-phase Bayesian
    # aggregator. On each waking tick the agent computes a counterfactual-backed
    # attribution from the SD-031 E2WorldForward (domain "place") and/or ARC-033
    # E2HarmSForward (domain "self") comparators and buffers it per (domain,
    # region); the sleep loop reads the buffered feedstock as the aggregator
    # evidence INSTEAD of the MECH-284 staleness scalar when this is on.
    # All defaults no-op; agent.scientist_attribution_buffer is None when off ->
    # bit-identical (the sleep loop sources staleness exactly as before).
    # PRECONDITION: requires use_e2_world_forward OR use_e2_harm_s_forward
    # (the buffer has no comparator to source attributions from otherwise).
    use_scientist_attribution: bool = False
    # Counterfactual-contrast threshold above which an attribution is treated as
    # counterfactual-backed (a discriminating intervention); below it the
    # attribution is correlational. The falsifiable distinction the MECH-275
    # claim turns on.
    scientist_attribution_cf_margin: float = 0.05
    # When True (default), correlational (contrast < cf_margin) records are NOT
    # buffered -> structured feedstock. False = correlational-control arm of the
    # readiness diagnostic (feed everything -> predicted noise-fit).
    scientist_attribution_only_counterfactual_backed: bool = True
    # EMA rate per record applied to the per-region attribution.
    scientist_attribution_ema_alpha: float = 0.3
    # Per-cycle multiplicative decay applied to each region EMA at sleep cycle
    # end (1.0 = no decay; parallels the BayesianAggregator decay_factor).
    scientist_attribution_decay: float = 1.0

    # SD-058 / MECH-357: instrumental-avoidance acquisition (ilPFC-analog
    # freeze-suppression gate + instrumental-avoidance action pathway +
    # eligibility-trace avoidance-efficacy learning). REE has the defensive
    # REACTION side (SD-035 amygdala salience + MECH-279 PAG freeze) but lacked
    # the instrumental-ACQUISITION side: active avoidance learning is the
    # resolution of a Pavlovian-instrumental conflict (Moscarello & LeDoux 2013)
    # -- learning to avoid REQUIRES ilPFC to suppress CeA-driven freezing.
    # Pure-arithmetic regulator (ree_core/pfc/infralimbic_avoidance_gate.py);
    # bit-identical when use_instrumental_avoidance=False (agent.instrumental_avoidance
    # is None and no consumer fires). See
    # REE_assembly/docs/architecture/sd_058_instrumental_avoidance_acquisition.md.
    use_instrumental_avoidance: bool = False
    # Eligibility-trace credit rate when a directed action under threat drops z_harm_a.
    avoidance_learn_rate: float = 0.05
    # Decay rate when the agent freezes / fails to avoid under threat.
    avoidance_leak_rate: float = 0.02
    # Freeze-default for the ilPFC-naive agent (0.0 = freezes until it learns).
    avoidance_initial_efficacy: float = 0.0
    # Protective-scaffold floor (curriculum sets > 0 and anneals it down;
    # effective_efficacy = max(avoidance_efficacy, scaffold_floor)).
    avoidance_scaffold_floor: float = 0.0
    # z_harm_a norm below which there is no threat to avoid.
    avoidance_threat_floor: float = 0.1
    # z_harm_a norm mapping to full threat_scale = 1.0.
    avoidance_threat_ref: float = 0.5
    # Min harm-drop counted as a successful avoidance (credit branch).
    avoidance_efficacy_reward_floor: float = 1e-4
    # Gain on the anti-passivity (penalise-no-op) instrumental-avoidance bias.
    avoidance_action_bias_gain: float = 0.1
    # Clamp on |avoidance bias| (mirrors lateral_pfc / curiosity / vigor).
    avoidance_bias_scale: float = 0.1
    # effective_efficacy * threat_scale above which the MECH-279 freeze no-op
    # is suppressed (ilPFC suppresses CeA-driven freezing).
    avoidance_suppression_threshold: float = 0.5
    # The passive / no-op action class (matches MECH-279 / MECH-320 convention).
    avoidance_noop_class: int = 0

    # ----------------------------------------------------------------
    # MECH-219 (SD-019b): affective-harm hysteretic integrator. Turns the
    # SD-019a medium-timescale unpleasantness channel (z_harm_un) into a slow,
    # persistent, controllability-gated SUFFERING load state (z_harm_suffering)
    # via an asymmetric (hysteretic) integrator. Pure-arithmetic regulator
    # (ree_core/affect/harm_suffering_accumulator.py); bit-identical when
    # use_harm_suffering_accumulator=False (agent.harm_suffering_accumulator is
    # None, LatentState.z_harm_suffering stays None, no consumer redirect fires).
    # The default escapability_mode=constant=1.0 gives g_t=0 -> s_t->0 so the
    # integrator is also inert/maximally-relieving even when explicitly enabled.
    # See REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md.
    use_harm_suffering_accumulator: bool = False
    # Asymmetric accumulation rates: alpha_rise >> alpha_fall is the hysteresis
    # (sticky suffering, slow recovery -- Baliki 2012 allostatic drift).
    harm_suffering_alpha_rise: float = 0.2
    harm_suffering_alpha_fall: float = 0.01
    # Escapability source mode: constant (default, dependency-free) /
    # avoidance_efficacy (SD-058 effective_efficacy(), the literal escapability
    # construct -- adds a soft dependency on the v3_pending SD-058 substrate) /
    # external (a validation experiment drives it via set_harm_suffering_escapability).
    harm_suffering_escapability_mode: str = "constant"
    # Escapability value used in the constant mode (1.0 = fully escapable -> g=0).
    harm_suffering_escapability_constant: float = 1.0
    # Initial value the external mode reports until a caller sets it.
    harm_suffering_external_escapability: float = 1.0
    # Hard clamp on the accumulated suffering scalar s_t.
    harm_suffering_s_cap: float = 2.0
    # SD-022 body-damage fold-in (memo Section 6 fork b): when > 0, ||z_harm_a||
    # is added to the drive so the SD-022 / EXQ-319 / EXQ-323a evidence is
    # preserved rather than orphaned. 0.0 -> pure z_harm_un drive.
    harm_suffering_body_damage_weight: float = 0.0
    # Optional SD-020 prediction-error driver (Q-036 secondary modulator); when
    # > 0, pe_gain * unsigned_PE is added to drive AFTER the controllability gate.
    harm_suffering_pe_gain: float = 0.0
    # Optional Schmitt bistable latch (the "distinct load STATE" reading).
    harm_suffering_use_bistable_latch: bool = False
    harm_suffering_theta_on: float = 0.5
    harm_suffering_theta_off: float = 0.3
    # Per-consumer z_harm_a -> z_harm_suffering redirect flags (memo Section 6).
    # Each redirects ONLY that consumer's scalar harm-magnitude read to
    # ||z_harm_suffering||; all default OFF so the migration is staged and
    # individually ablatable, and bit-identical when off. v1 wires the
    # urgency / PAG / interrupt consumers (AIC, PAG, MECH-091); the dACC/pACC
    # flags are reserved (their E2_harm_a forward models are keyed on the
    # current z_harm_a dim -- migrate last after measuring R^2; left on legacy
    # z_harm_a in v1 per memo R3, so these two flags are currently no-ops).
    harm_suffering_redirect_aic: bool = False
    harm_suffering_redirect_pag: bool = False
    harm_suffering_redirect_mech091: bool = False
    harm_suffering_redirect_dacc: bool = False
    harm_suffering_redirect_pacc: bool = False

    # ----------------------------------------------------------------
    # SD-059 / MECH-358: relief/safety escape-affordance bridge. Extends the
    # MECH-357 scalar avoidance_efficacy into a per-first-action-class credit
    # table (the DIRECTED escape MECH-357 lacks). Pure-arithmetic regulator
    # (ree_core/pfc/escape_affordance_bridge.py); bit-identical when
    # use_escape_affordance_bridge=False (agent.escape_affordance_bridge is None
    # and no consumer fires). Closes the V3-EXQ-603h Stage-H gap. See
    # REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md.
    use_escape_affordance_bridge: bool = False
    # Half switches (consulted only when the master flag is on); the 4-arm
    # validation toggles these to dissociate relief vs safety.
    use_escape_relief_credit: bool = True
    use_escape_safety_credit: bool = True
    # EMA credit rates for the relief / safety affordance traces.
    escape_relief_learn_rate: float = 0.1
    escape_safety_learn_rate: float = 0.1
    # Per-tick leak on both affordance tables (forgetting; pathological-loop guard).
    escape_bridge_leak_rate: float = 0.01
    # Minimum harm-drop counted as a relief event.
    escape_relief_reward_floor: float = 1e-4
    # Threat envelope (shared convention with the MECH-357 gate).
    escape_threat_floor: float = 0.1
    escape_threat_ref: float = 0.5
    # Approach-bonus gain + clamp (mirrors avoidance_bias_scale so the bridge
    # cannot dominate the additive score-bias chain).
    escape_approach_gain: float = 0.1
    escape_bias_scale: float = 0.1
    # The no-op / freeze action class (never gets an approach bonus).
    escape_noop_class: int = 0
    # 603i SAFETY-half fix: feed the trained MECH-303 (contextual safety terrain)
    # / MECH-304 (conditioned safety store) threat-absence prediction into the
    # bridge safety-credit path so the safety half can credit non-vacuously (on
    # 603i it credited 0/3 because the raw threat_scale<=0 check almost never
    # fires under Stage-H). OR-composed with the raw check; max() of whichever
    # trained predictors are enabled. Default OFF -> bit-identical to pre-603i.
    escape_use_trained_safety_signal: bool = False
    escape_safety_signal_threshold: float = 0.5

    # ----------------------------------------------------------------
    # ARC-006 / MECH-045: token-instance object-file / entity-persistence buffer.
    # The TOKEN projection of the ARC-080 type/token/anchor triad (the missing
    # third store; TYPE=SD-057 IncentiveTokenBank, ANCHOR=SD-039 ghost-goal bank).
    # Non-trainable stateful buffer (ree_core/entities/object_file_buffer.py)
    # that assigns label-free per-entity token ids by spatiotemporal-continuity
    # data-association over z_world-local features, with a precision-weighted
    # per-token feature buffer and attention-gated births. v1 lands STANDALONE
    # (no action-stream consumer) -> bit-identical OFF AND ON (only buffer state
    # changes; nothing reads it yet). OFF the GAP-7 V3-closure path. See
    # REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
    use_object_file_buffer: bool = False
    obf_max_tokens: int = 5            # FINST capacity cap (C4)
    obf_continuity_radius: float = 2.0  # motion gate radius in cells (C1)
    obf_w_motion: float = 1.0          # association cost weight (motion term)
    obf_w_feat: float = 1.0            # association cost weight (feature term)
    obf_feature_alpha: float = 0.3     # base feature EMA rate (C2/C5)
    obf_persist_ttl: int = 8           # ticks a token survives unseen (C3)
    obf_min_birth_salience: float = 0.0  # attention floor to open a new token (C4)
    obf_use_precision_weighting: bool = True  # C5 on/off

    # ----------------------------------------------------------------
    # Post-603i successor scaffold: trainable relief/safety escape-affordance
    # learner. This is a trainable-head sibling to the SD-059 arithmetic bridge,
    # not a replacement for the active V3-EXQ-603i validation path. Disabled by
    # default; when off, agent.trainable_escape_affordance_learner is None and no
    # update or score-bias consumer fires.
    use_trainable_escape_affordance_learner: bool = False
    use_trainable_relief_critic: bool = True
    use_trainable_safety_predictor: bool = True
    trainable_escape_bias_scale: float = 0.1
    trainable_escape_relief_learn_rate: float = 0.1
    trainable_escape_safety_learn_rate: float = 0.1
    trainable_escape_leak_rate: float = 0.01
    trainable_escape_relief_reward_floor: float = 1e-4
    trainable_escape_relief_target_scale: float = 0.3
    trainable_escape_threat_floor: float = 0.1
    trainable_escape_noop_class: int = 0
    trainable_escape_hidden_dim: int = 32
    trainable_escape_action_embedding_dim: int = 8
    trainable_escape_optimizer_lr: float = 0.03
    trainable_escape_prediction_floor: float = 0.02

    # ----------------------------------------------------------------
    # Post-603i E2 escape-affordance linker. A READOUT/linkage layer over the
    # EXISTING E2 (cerebellar-analog) action-consequence forward model
    # (E2.world_forward) -- NOT a duplicate forward predictor. Indexes E2 geometry
    # into escape-affordance viability readouts + a hippocampal-style viability
    # index, exposes escape_affordance_features for the relief/safety heads, and
    # (behind its own flag) emits a bounded threat-gated E3 score-bias. Disabled
    # by default; when off, agent.e2_escape_affordance_linker is None and no
    # consumer fires. See ree_core/pfc/e2_escape_affordance_linker.py +
    # docs/substrate_plans/post_603i_e2_escape_affordance_linkage.md.
    use_e2_escape_affordance_linker: bool = False
    use_e2_escape_linker_for_relief_safety: bool = False
    use_e2_escape_linker_e3_bias: bool = False
    escape_linker_learn_rate: float = 0.1
    escape_linker_optimizer_lr: float = 0.03
    escape_linker_leak_rate: float = 0.01
    escape_linker_hidden_dim: int = 32
    escape_linker_action_embedding_dim: int = 8
    escape_linker_bias_scale: float = 0.1
    escape_linker_threat_floor: float = 0.1
    escape_linker_threat_ref: float = 0.5
    escape_linker_noop_class: int = 0
    escape_linker_relief_reward_floor: float = 1e-4
    escape_linker_harm_delta_scale: float = 0.3
    escape_linker_prediction_floor: float = 0.02
    escape_linker_block_hypothesis_learning: bool = True

    # ----------------------------------------------------------------
    # V3-EXQ-563 diagnostic: forced_score_bias_per_class.
    # Hard-injects a per-action-class score bias vector, bypassing all
    # naturalistic signal generation (MECH-313/314/320). Used to verify
    # the score-bias -> action-change seam independently of CEM collapse.
    # None (default) = disabled; bit-identical to prior behaviour.
    # When set, each candidate's first-step action class is read and the
    # corresponding element of this list is added to dacc_score_bias.
    # List index = action class (0-based); out-of-range classes get 0.0.
    # E3 convention: NEGATIVE values favour a candidate; POSITIVE penalise.
    forced_score_bias_per_class: Optional[List[float]] = None

    # ----------------------------------------------------------------
    # MECH-319 (ARC-064 / arc_062 GAP-K): simulation_mode_rule_write_gate.
    # Substrate-level instantiation of MECH-094 at the rule-arbitration
    # layer. Unified categorical write gate that suppresses arbitration-
    # weight updates in MECH-312 sub-mechanisms (gated_policy,
    # lateral_pfc_analog, future arbitrators) during ghost / replay /
    # DMN passes. Substrate anchors: SWR machinery (Joo & Frank 2018) +
    # reverse-replay discriminable signature (Foster & Wilson 2006).
    # The categorical write-gate function at the arbitration layer is
    # REE-novel (Pull 3 SYNTHESIS R1 GENUINE-NOVELTY-CONFIRMED conf 0.72;
    # Pull 4 R3 KEEP-AS-IS verdict on MECH-094). See
    # ree_core/regulators/simulation_mode_rule_gate.py and
    # REE_assembly/docs/architecture/mech_319_simulation_mode_rule_gate.md.
    use_simulation_mode_rule_gate: bool = False
    # V3-EXQ-543c falsifier control. False (default) = MECH-319 normal
    # (simulation tag suppresses arbitration writes). True =
    # artificial-write-channel-routing mode -- simulation content IS
    # admitted into rule_state / gated_policy / future arbitrators.
    # Predicted to produce monomodal-collapse re-emergence per
    # MECH-094 / MECH-319 generalisation. Construction raises ValueError
    # when admit_writes=True without the master flag also on
    # (loud-not-silent guard against mis-configuration).
    simulation_mode_rule_gate_admit_writes: bool = False

    # ----------------------------------------------------------------
    # SD-034: governance.closure_operator (five-part "done" token)
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates a ClosureOperator
    # connecting BetaGate, DACCAdaptiveControl, ResidueField, SalienceCoordinator,
    # and LateralPFCAnalog. False = disabled (default, backward compatible).
    use_closure_operator: bool = False
    # Automatic detector threshold on ||rule_state_t - rule_state_{t-1}||.
    closure_rule_delta_threshold: float = 0.001
    # Number of consecutive stable ticks before automatic closure fires.
    closure_stable_ticks: int = 3
    # If True, automatic closure only fires while beta is elevated.
    closure_require_beta_elevated: bool = True
    # Minimum sd_033a write_gate value for closure (mode-conditioning).
    closure_min_sd033a_gate: float = 0.5
    # Number of copies of the completed action class to push onto the
    # MECH-260 suppression buffer when closure fires.
    closure_nogo_injection_count: int = 3
    # Multiplicative decay applied to RBF weights within the rule-domain
    # neighbourhood. Must be in (0, 1]; 1.0 disables discharge.
    closure_residue_discharge_factor: float = 0.5
    # Domain radius in RBF-bandwidth units for residue discharge.
    closure_residue_discharge_radius: float = 1.5
    # Closure event signal magnitude written to SalienceCoordinator.
    closure_signal_value: float = 1.0
    # If True, _pe_ema on DACCAdaptiveControl is reset on closure.
    closure_reset_pe_ema: bool = True
    # If set, installs an absolute cap on dACC pe after closure (MECH-268).
    closure_pe_cap_after: Optional[float] = None
    # MECH-268: if True, ClosureOperator clears the dACC outcome-history
    # FIFO on fire so the next rule-state starts with no habituation
    # accrued from the previous one. Independently ablatable from
    # closure_reset_pe_ema.
    closure_reset_outcome_history: bool = True
    # If nonzero, register closure_event affinity toward internal_planning
    # on the SalienceCoordinator (biases mode relaxation post-closure).
    closure_signal_affinity_internal_planning: float = 0.5
    # SD-034 commitment-closure-control-plane (2026-06-12).
    # Master switch for the explicit env-completion hook seam: when True,
    # REEAgent.notify_env_completion(action_class, z_world) routes the env's
    # transition_type == "sequence_complete" signal into
    # closure_operator.emit_closure (the explicit hook the closure docstring
    # describes but the *c cohort left unwired -> V3-EXQ-460c n_closures=0).
    # No-op (returns None) when False -> bit-identical. Requires
    # use_closure_operator=True to do anything.
    use_closure_env_completion_hook: bool = False
    # SD-034 de-commitment hold / refractory window in ticks. When > 0, a
    # closure fire installs a BetaGate refractory of this length so the
    # closure-driven release survives >1 tick and produces a measurable
    # latch-occupancy drop (V3-EXQ-468c committed_frac defect). Default 0 ->
    # no hold -> bit-identical.
    closure_decommit_hold_ticks: int = 0
    # SD-034 commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE amend
    # (2026-06-19, failure_autopsy_V3-EXQ-460f): the fixed Leg-B refractory
    # (closure_decommit_hold_ticks ~5) suppresses only ~20-35 tick-blocks of
    # latch occupancy, swamped by ~530-560 natural-commit elevated steps, so the
    # de-commit occupancy-drop DV is underpowered. This lever scales the
    # refractory installed at a closure fire by the committed-run length at the
    # moment of closure (the BetaGate.committed_run_length captured BEFORE the
    # closure's own release()): n = closure_decommit_hold_ticks +
    # round(closure_decommit_hold_scale_with_run * run_length), clamped to
    # closure_decommit_hold_max_ticks (0 = uncapped). So a long committed run --
    # exactly the source of the swamping occupancy -- triggers a proportionally
    # long post-closure hold, scaling the de-commit authority with the magnitude
    # it must overcome. Default 0.0 -> the refractory uses
    # closure_decommit_hold_ticks unchanged -> bit-identical (and with
    # closure_decommit_hold_ticks 0 too, no refractory at all).
    closure_decommit_hold_scale_with_run: float = 0.0
    # Hard clamp on the committed-run-scaled refractory length (ticks). 0 ->
    # uncapped. Read only when closure_decommit_hold_scale_with_run > 0.
    closure_decommit_hold_max_ticks: int = 0
    # SD-034 commitment-closure-control-plane BETA-ENGAGEMENT amend
    # (2026-06-17, failure_autopsy_V3-EXQ-460e): couple the closure-plane
    # installed commitment (e3._committed_trajectory is not None) to bistable
    # BetaGate elevation, decoupling the de-commit latch-occupancy DV from the
    # fragile natural running_variance crossing (result.committed, which fires
    # on only 1/3 seeds on the 603n foraging substrate -> commit-without-beta
    # dissociation, total_beta_elevated=0). When True, the bistable elevate
    # block treats (result.committed OR closure-plane commit active) as the
    # commit-for-beta trigger, so beta occupancy is non-zero on every seed where
    # a closure-plane commitment forms and the Leg-B de-commit refractory
    # produces a measurable ON<OFF latch-occupancy drop. Default False -> no-op
    # -> bit-identical (the bistable block keys on result.committed alone).
    use_closure_commit_beta_coupling: bool = False

    # ----------------------------------------------------------------
    # SD-035: Amygdala analogue (BLAAnalog + CeAAnalog peer modules)
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates BOTH BLAAnalog and
    # CeAAnalog unless the per-module switches below are set False. When
    # False (default), the amygdala layer is a no-op -- no instantiation,
    # no wiring, bit-identical to legacy behaviour.
    use_amygdala_analog: bool = False
    # Per-module switches. Gated by use_amygdala_analog (master switch
    # False overrides these). Default True once the master switch is on
    # -- both peer modules are expected to run together since each owns
    # different MECH-074 sub-behaviours.
    use_bla_analog: bool = True
    use_cea_analog: bool = True

    # -- BLAAnalog (SD-035 / MECH-074a / MECH-074b / MECH-074d) --
    # Inverted-U encoding-gain parameters (Roozendaal & McGaugh 2011).
    bla_encoding_gain_max: float = 2.5
    bla_encoding_gain_floor: float = 1.0
    bla_arousal_threshold_on: float = 0.4
    bla_arousal_peak: float = 0.7
    # Post-event decay window on the encoding gain. 18000 steps ~= 30 min
    # biological @ 100 ms per step; half-life 3600 steps ~= 6 min.
    bla_window_steps: int = 18000
    bla_window_half_life_steps: int = 3600
    # Retrieval bias (content-selective per-trace weight vector, NOT
    # scalar). w_i = 1 + alpha * arousal_tag_i; LaBar & Cabeza 2006
    # 0.3-1.0 range (midpoint default).
    bla_retrieval_bias_alpha: float = 0.6
    # Optional zero-sum compensation for untagged traces (0.0 default;
    # enable 0.1-0.3 for full zero-sum form).
    bla_retrieval_bias_compensation: float = 0.0
    # LaBar 2006 mandate: tag at encoding, not reconstructed at retrieval.
    # Set False only for the named-failure-signature ablation.
    bla_retrieval_tag_at_encoding: bool = True
    # Remap signal (MECH-074d). PE-zscore threshold over running std,
    # EMA alpha matches MECH-205 pe_ema_alpha convention, initial std
    # seeds a short burn-in window.
    bla_remap_pe_sigma_threshold: float = 1.0
    bla_remap_pe_ema_alpha: float = 0.02
    bla_remap_pe_std_init: float = 0.1
    # Moita 2004: ~30-35% of predictor-candidate codes perturbed.
    bla_remap_code_fraction: float = 0.33
    # Moita 2004 attribution-gate hard requirement. Set False only for
    # deliberate broadcast-remap ablation (named failure signature).
    bla_remap_requires_attribution: bool = True
    # ContextMemory slot blend applied when remap_signal targets a slot.
    # 0.5 = keep half the old slot code, half current observation-conditioned code.
    bla_context_remap_blend: float = 0.5

    # -- CeAAnalog (SD-035 / MECH-046 / MECH-074c) --
    # Fast-route threshold on the low-frequency magnitude projection of
    # z_harm_a. Below this, no fast-route fire (mode_prior=0, fast_prime
    # decays from whatever residual remains).
    cea_fast_route_threshold: float = 0.5
    # Use L1 mean-magnitude of z_harm_a as the low-frequency summary
    # (default True; set False for L2 norm on caller-supplied scalar
    # ablation).
    cea_fast_route_input_is_lowfreq: bool = True
    # Maximum absolute value of the pre-softmax additive log-odds bias
    # emitted as mode_prior. MUST be <= AIC/dACC log-odds ceiling so CeA
    # never over-rules cortex (synthesis.md).
    cea_mode_prior_log_odds_max: float = 0.8
    # Scale factor on the above cap at threshold-crossing.
    cea_mode_prior_gain: float = 1.0
    # Gating placement flag: pre-softmax additive vs post-softmax
    # multiplicative. True = correct biological reading per synthesis;
    # False reproduces the named failure signature.
    cea_pre_softmax_additive: bool = True
    # Fast-prime pulse peak amplitude; bounded by log_odds_max.
    cea_fast_prime_amplitude: float = 0.6
    # Decay half-life of the fast_prime pulse in sim steps.
    cea_fast_prime_decay_tau_steps: int = 4
    # Override window: cortical signals have this many steps to confirm
    # / extend the pulse.
    cea_fast_prime_override_window_steps: int = 8
    # Weight on cortical_confirmation during override window. 1.0 =
    # full cortical confirmation holds pulse; 0.0 = cortex cannot
    # sustain (pure fast-route timing).
    cea_cortical_confirmation_weight: float = 1.0

    # ----------------------------------------------------------------
    # SD-036: GABAergic cross-stream decay regulator + MECH-279 PAG freeze gate
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates a GABAergicDecayRegulator
    # that ticks each step (after LatentStack.encode and before mode arbitration)
    # and applies per-stream exponential decay to z_harm_s, z_harm_a, z_beta
    # (z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone)). Also gates instantiation of
    # the MECH-279 PAG freeze-gate. False = disabled (default, backward compat).
    use_gabaergic_decay: bool = False
    # Global GABAergic tonic multiplier in [gaba_tone_min, gaba_tone_max].
    # 1.0 = baseline; >1.0 benzo-analog faster decay; <1.0 withdrawal-analog
    # slower decay.
    gaba_tone: float = 1.0
    gaba_tone_min: float = 0.0
    gaba_tone_max: float = 2.0
    # Per-stream baseline decay rates.
    gaba_tau_z_harm_s: float = 0.05  # ~20-step half-life
    gaba_tau_z_harm_a: float = 0.02  # ~50-step half-life
    gaba_tau_z_beta: float = 0.03    # ~30-step half-life
    # Per-stream coverage flags (gated by master switch). All True by default
    # once master is on -- ablate per-stream by setting these False.
    gaba_decay_z_harm_s: bool = True
    gaba_decay_z_harm_a: bool = True
    gaba_decay_z_beta: bool = True
    # Per-stream input thresholds. Default 0.0 = always decay. Non-zero
    # suspends decay for the tick when the stream's magnitude change exceeds
    # the threshold (salient input drove the update).
    gaba_input_threshold_z_harm_s: float = 0.0
    gaba_input_threshold_z_harm_a: float = 0.0
    gaba_input_threshold_z_beta: float = 0.0

    # MECH-279: PAG freeze-gate. Master switch defaults False; when True, the
    # gate ticks each select_action() and constrains the action selector to
    # no-op / minimal-movement actions while freeze_active. The gate consumes
    # gaba_tone above to compute exit_threshold = theta_freeze * gaba_tone.
    use_pag_freeze_gate: bool = False
    pag_theta_freeze: float = 2.0
    pag_duration_input_threshold: float = 0.4
    pag_min_freeze_duration: int = 0
    pag_max_freeze_duration: int = 0
    # When freeze is active, the action selector emits a no-op action. The
    # no-op action class index defaults to 0 (typically a stay-in-place action
    # in CausalGridWorldV2). Override per env if the no-op action sits at a
    # different index.
    pag_freeze_noop_action_class: int = 0

    # ----------------------------------------------------------------
    # SD-037: Broadcast Override Regulator (orexin-analog)
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates a BroadcastOverrideRegulator
    # that ticks each sense() step. The regulator integrates drive_level (SD-012)
    # and a sustained-threat magnitude window over z_harm into a scalar
    # override_signal in [0, 1], EMA-smoothed. The signal is consumed at three
    # sites: PAG freeze-gate (theta_freeze scaling), SalienceCoordinator
    # (operating-mode reweight), and GoalState (drive -> z_goal seeding gate).
    # False = disabled (default, backward compat).
    use_broadcast_override: bool = False
    # Sigmoid recruitment threshold: override_raw = sigmoid(drive_w*drive +
    # harm_w*sustained_threat - threshold). At default 0.5, with both inputs
    # at ~0.5 and unit weights, the signal lifts off baseline.
    override_recruitment_threshold: float = 0.5
    # PAG freeze-gate scaling. exit_threshold_eff = theta_freeze *
    # (1 + alpha_override * override_signal) * gaba_tone. Only consumed when
    # use_pag_freeze_gate is also True.
    override_alpha_pag: float = 0.5
    # SalienceCoordinator reweight magnitude. Only consumed when
    # use_salience_coordinator is also True.
    override_salience_reweight_alpha: float = 0.3
    # Linear coefficients on the two driving signals before the sigmoid.
    override_drive_weight: float = 1.0
    override_harm_weight: float = 1.0
    # Sustained-threat magnitude window: rolling mean of z_harm.norm() over
    # the last sustained_threat_window ticks; normalised by
    # sustained_threat_threshold so values >= threshold map to ~1.0 input.
    override_sustained_threat_window: int = 12
    override_sustained_threat_threshold: float = 0.4
    # EMA decay rate on override_signal output (~20-tick smoothing at 0.05).
    override_decay_rate: float = 0.05
    # GoalState gate: drive -> z_goal seeding is amplified by this factor when
    # override_signal >= recruitment_threshold. 1.0 = legacy SD-012 path
    # unchanged; >1.0 = orexin-recruited drive seeding.
    override_goal_seeding_gain: float = 2.0

    # ----------------------------------------------------------------
    # SD-037 consumer-cascade (MECH-281 motor-coupling axis, 2026-05-30)
    # ----------------------------------------------------------------
    # Scalar (1 + gain * override_signal) multipliers on existing knobs at four
    # additional consumer sites named in MECH-281 implementation_note (PFC for
    # SD-033a deliberation, BLA / CeA for SD-035 amygdala arbitration, beta-gate
    # for MECH-090 motor-side escape-from-freeze). Defaults are 0.0 = bit-
    # identical OFF; each requires its parent substrate's master flag to also
    # be True (use_lateral_pfc_analog / use_amygdala_analog / nothing for the
    # urgency_interrupt path which is always available). Companion to the
    # already-wired override_alpha_pag (PAG freeze-gate exit threshold) and
    # override_goal_seeding_gain (GoalState seeding). See SD-037 design doc.
    override_pfc_eta_gain: float = 0.0
    override_bla_encoding_gain: float = 0.0
    override_cea_amplitude_gain: float = 0.0
    override_beta_interrupt_gain: float = 0.0

    # ----------------------------------------------------------------
    # MECH-282: LPB interoceptive harm routing
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent instantiates LPBInteroceptiveRouter,
    # masks resource-field channels out of harm_obs before HarmEncoder (external
    # threat only in z_harm), and populates LatentState.z_harm_intero from
    # drive_level + harm_obs_a resource slice. With use_broadcast_override=True,
    # override recruitment uses intero magnitude; external magnitude feeds PAG
    # freeze duration when use_pag_freeze_gate=True.
    use_lpb_interoceptive_routing: bool = False
    lpb_intero_z_dim: int = 16
    lpb_drive_weight: float = 1.0
    lpb_resource_weight: float = 1.0

    # ----------------------------------------------------------------
    # Sleep-aggregation cluster (MECH-272 / MECH-273 / MECH-275 / MECH-285)
    # ----------------------------------------------------------------
    # GAP-3 unified cluster master flag. The Phase A-E surface is gated by
    # eight independent default-False flags (use_sleep_loop, sws_enabled,
    # rem_enabled, use_mech285_sampler, use_mech272_routing,
    # use_mech272_routing_consumer, use_mech275_aggregator,
    # use_mech273_self_model). Before this flag existed, an experiment had to
    # set all eight by hand or the offline-consolidation pathway was silent
    # (sleep_substrate_plan.md GAP-3). When this flag is True, the eight
    # sub-flags are forced True via enable_sleep_aggregation_cluster()
    # (called from __post_init__ for direct construction and from from_dims()
    # for the factory path). Resolution is OR-only: it flips False -> True,
    # matching the use_mech307_conjunction resolver convention; an explicit
    # sub-flag set False alongside the master is overridden to True (use the
    # individual flags, not the master, when fine-grained opt-out is needed).
    # MECH-204 precision recalibration (use_rem_precision_recalibration) is a
    # separate sibling WRITEBACK step under sleep_substrate GAP-1 and is
    # intentionally NOT bundled here. Substrate prerequisites for the cluster
    # to actually fire (anchor sets for Phase B, e2_harm_s for Phase E) are
    # separate MECH-269/ARC-033 substrate switches, also not bundled here.
    # Default False: bit-identical to pre-GAP-3 behaviour.
    use_sleep_aggregation_cluster: bool = False
    # Phase A master flag. When True, REEAgent instantiates a SleepLoopManager
    # that drives a deterministic K-episode sleep cycle through the existing
    # SD-017 surface (run_sleep_cycle). Bit-identical OFF: the agent does not
    # construct the manager and reset() does not call notify_episode_end().
    # Phases B-E (replay sampler, routing gate, Bayesian aggregator,
    # self-model writeback) layer additional master flags on top of this one.
    use_sleep_loop: bool = False
    # Cycle period: sleep fires once per K completed episodes. Default 1 keeps
    # the smoke / contract surface simple; experiments override.
    sleep_loop_episodes_K: int = 1
    # When True, the SleepLoopManager refuses to fire if neither sws_enabled
    # nor rem_enabled is True on the agent config (no substrate to drive).
    # Set False only for diagnostic harness runs that exercise the manager
    # state machine without the SD-017 passes.
    sleep_loop_require_passes: bool = True

    # SD-MEL-CONSUMER (sleep_substrate:GAP-5b): adaptive sleep-cadence MEL
    # consumer. When True (and use_sleep_loop is True), the SleepLoopManager
    # reads accumulated waking Model Error Load (mean per-step e3 prediction
    # error over the wake window, the same signal V3-EXQ-701c measured) and
    # scales the offline-phase DURATION (sws_consolidation_steps and/or
    # rem_attribution_steps) -- and optionally the ENTRY timing -- by a
    # relative, scale-free factor. This is the INV-050 third / learning-demand
    # drive (MECH-180), DISTINCT from the SD-037 arousal entry gate
    # (use_mech286_sleep_onset_gate / GAP-5, V4). Master OFF -> the agent never
    # instantiates the consumer -> byte-identical to the K-episode-deterministic
    # scheduler. See REE_assembly/docs/architecture/sd_mel_consumer.md.
    use_mel_consumer: bool = False
    # Duration sensitivity: factor = clamp(1 + mel_gain*(mel/ref - 1), min, max).
    # Inert (factor stays 1.0) unless the master switch is on. gain 0.0 also
    # yields factor 1.0 (no-op) even with the master on.
    mel_gain: float = 1.0
    # Homeostatic MEL set-point (per-step PE units). 0.0 sentinel = auto-calibrate
    # to the first sleep cycle's MEL. Validation test-bed sets ~2e-5 (the 701c
    # converged-base per-step PE). Guarded by max(ref, mel_relative_floor).
    mel_reference: float = 0.0
    # Reference mode: "fixed" (constant set-point; correct for graded-novelty
    # ablation) or "ema" (slow per-cycle EMA; biologically faithful long-run).
    mel_reference_mode: str = "fixed"
    mel_ema_alpha: float = 0.1
    # Saturation clamp on the duration factor (control-saturation guard against a
    # transient PE spike blowing up / zeroing the offline phase).
    mel_duration_factor_min: float = 0.5
    mel_duration_factor_max: float = 3.0
    # Relative floor guarding ref ~ 0 in mel/ref. Recalibrated DOWN from the
    # 701c-inherited ABS_MEL_FLOOR=1e-4 (which was ~5x the converged-base signal)
    # to a divide-guard only -- the response is relative, never an absolute
    # spread gate. See sd_mel_consumer.md "Instrument-floor learning".
    mel_relative_floor: float = 1e-6
    # Which duration lever(s) the factor scales.
    mel_scale_sws: bool = True
    mel_scale_rem: bool = True
    # Secondary ENTRY-timing lever: fire a cycle as soon as accumulated MEL
    # crosses mel_entry_threshold (with the K-episode counter as a safety
    # backstop ceiling), instead of strictly every K episodes. Default off; the
    # duration lever is the primary validated mechanism.
    use_mel_entry: bool = False
    mel_entry_threshold: float = 0.0

    # Phase B (MECH-285): SleepReplaySampler offline arm. When True (and
    # use_sleep_loop is True), the SleepLoopManager freezes a
    # StalenessAccumulator snapshot at SLEEP_ENTRY and instantiates a
    # SleepReplaySampler that draws seeds from the broad AnchorSet pool
    # (active + inactive, dual-trace preserved) with priority
    # softmax(staleness[r] / temperature). Draws are recorded in
    # SleepCycleState.last_metrics for diagnostic inspection. No downstream
    # consumers wire in this phase -- routing gate (Phase C), Bayesian
    # aggregator (D), and self-model writeback (E) extend this manager
    # via additional master flags.
    use_mech285_sampler: bool = False
    # Number of draws per sleep cycle. Each draw is O(|seed_pool|) for the
    # softmax + one numpy.random.choice; default 50 keeps wallclock under
    # ~1ms per cycle for typical anchor counts.
    mech285_draws_per_cycle: int = 50
    # Softmax temperature. Lower concentrates replay on most-staled
    # regions; higher spreads. Default 1.0 per design doc verdict 2.
    mech285_temperature: float = 1.0
    # When True and the agent has no StalenessAccumulator attached, the
    # sampler falls back to a uniform distribution over the broad pool
    # (still broad-pool per MECH-285 verdict 1, just no staleness signal).
    # Set False to refuse sampling without the accumulator (raises
    # RuntimeError at draw time).
    mech285_allow_uniform_fallback: bool = True

    # Phase C (MECH-272): state-conditioned RoutingGate. When True (and
    # use_sleep_loop is True), the SleepLoopManager constructs a RoutingGate
    # whose anchor / probe channel weights flip across phases per the
    # design-doc table:
    #   WAKING       -> (anchor=1.0, probe=0.0)
    #   SWS_ANALOG   -> (anchor=0.6, probe=0.4)
    #   REM_ANALOG   -> (anchor=0.2, probe=0.8)
    # The gate emits RoutedEvent(anchor_channel, probe_channel) per replay
    # event. Phase C is a no-op consumer: downstream Phase D Bayesian
    # aggregator and the future E1 ContextMemory consolidation consumer
    # multiply their write strength by the channel weights, but neither
    # exists yet -- routed events land as diagnostics on
    # SleepCycleState.last_metrics.
    use_mech272_routing: bool = False
    mech272_waking_anchor_weight: float = 1.0
    mech272_waking_probe_weight: float = 0.0
    mech272_sws_anchor_weight: float = 0.6
    mech272_sws_probe_weight: float = 0.4
    mech272_rem_anchor_weight: float = 0.2
    mech272_rem_probe_weight: float = 0.8
    # Phase C downstream consumer (GAP-8): when True (and
    # use_mech272_routing is True), SleepLoopManager computes the mean
    # anchor_channel across the SWS draw batch and passes it as a
    # write_scale to run_sws_schema_pass(). The ContextMemory slot writes
    # are multiplied by this scale (0.6 at SWS_ANALOG row) so that
    # schema installation strength tracks the routing weight rather than
    # always writing at full strength regardless of phase.
    # Default False: bit-identical to pre-GAP-8 behaviour.
    use_mech272_routing_consumer: bool = False

    # Phase D (MECH-275): general Bayesian aggregator. When True (and
    # use_sleep_loop is True), the SleepLoopManager constructs a
    # BayesianAggregator that maintains per-domain per-region Gaussian
    # posteriors over residuals. Updates are gated on probe_channel from
    # MECH-272 RoutedEvents in the SWS pass and the REM re-route; a
    # snapshot is captured at PHASE_SWITCH so downstream Phase E
    # writeback (MECH-273) can read the SWS-only posterior. Phase D is a
    # NO-OP CONSUMER: posterior deltas land as diagnostics on
    # SleepCycleState.last_metrics. The "place" domain is the only
    # default domain in V3 (Bayesian upgrade of MECH-284's leaky
    # integrator); "self" domain is the MECH-273 specialisation in
    # Phase E. "object" / "other" are V4-deferred.
    use_mech275_aggregator: bool = False
    # Comma-separated tuple of domain names. Default is ("place",); V3
    # supports "place" and "self" (Phase E specialisation). Unknown
    # domain names are accepted as opaque keys.
    mech275_domains: tuple = ("place",)
    # Conjugate Gaussian posterior knobs (per-domain per-region).
    # Standard mean-and-variance update with Gaussian likelihood.
    mech275_prior_mean: float = 0.0
    mech275_prior_variance: float = 1.0
    mech275_likelihood_variance: float = 1.0
    # Per-cycle multiplicative decay applied to posterior precision
    # (1/variance) at PHASE_SWITCH. 1.0 = no decay. Lower values let
    # newer evidence overcome stale posteriors faster.
    mech275_decay_factor: float = 1.0
    # Probe-channel weight multiplier applied to each posterior update.
    # Update weight = probe_channel * mech275_probe_gain. When 0.0, the
    # aggregator runs but never updates -- diagnostic-only mode.
    mech275_probe_gain: float = 1.0

    # MECH-273: Phase E -- self-model writeback / specialisation.
    # SelfModelAggregator (subclass of BayesianAggregator specialised on
    # the SD-003 causal_sig posterior in the "self" domain) consumes the
    # SWS+REM posterior at WRITEBACK and runs a bounded offline gradient
    # pass on E2_harm_s using aggregator-corrected residuals as training
    # targets. MECH-094 simulation_mode tag is set throughout; the
    # E2_harm_s parameter update is the SINGLE EXPLICIT EXCEPTION,
    # gated to the writeback path. After the gradient pass,
    # StalenessAccumulator.partial_decay applies a multiplicative decay
    # to the staleness scalars at the regions touched by replay during
    # the cycle. SHY normalisation (MECH-120) is NOT consumed here --
    # explicitly out of V3 scope. Bit-identical OFF preserved.
    use_mech273_self_model: bool = False
    # MECH-204 Option A: precision recalibration consumer in WRITEBACK phase.
    # When True (and use_sleep_loop is True and serotonin tonic_5ht_enabled is
    # True and rem_enabled is True), the SleepLoopManager reads the zero-point
    # precision target captured by SerotoninModule at REM entry
    # (serotonin._precision_at_rem_entry) and nudges E3._running_variance
    # toward 1.0/target by mech204_recalibration_step. Sibling step in
    # WRITEBACK alongside MECH-273 self-model gradient + partial decay; runs
    # independently of MECH-273 (writeback fires for either reason). Per
    # Q-042 verdict, the broadcast read-site (Option B) is deferred to
    # Phase 7. Bit-identical OFF preserved.
    use_rem_precision_recalibration: bool = False
    # Step size for the linear interpolation toward target_variance:
    #   new_rv = (1 - step) * rv + step * (1.0 / target_precision)
    # Default 0.25 (high end of biologically defensible band per Q-042 Option A).
    # Bumped from 0.1 to 0.25 on 2026-05-09 after V3-EXQ-541c PASS confirmed
    # the F1 substrate works across the full {0.05, 0.10, 0.25, 0.50} sweep
    # with monotonically improving tracking_quality (0.842 -> 0.921) and zero
    # overshoot. The dose-response curve at 16 cycles per run showed:
    #   step 0.05 -> 0.90% cross-arm divergence
    #   step 0.10 -> 1.81%
    #   step 0.25 -> 4.51%  <- new default; just under 5% threshold
    #   step 0.50 -> 9.03%  <- above-band stress test, clears threshold
    # 0.25 is the strongest biologically-defensible default that balances
    # movement magnitude against overshoot risk; 0.05 / 0.10 are conservative
    # alternatives if an experiment needs gentler recalibration; 0.50 is
    # available as an above-band stress test (works empirically but is
    # outside the Q-042-supported band, so use only with explicit rationale).
    # See REE_assembly/evidence/planning/sleep_substrate_plan.md decision-log
    # entry 2026-05-09 ("MECH-204 V3 closure on F1") for full context.
    # See REE_assembly/evidence/literature/targeted_review_q_042/ for the
    # biological-defensibility band derivation.
    rem_precision_recalibration_step: float = 0.25
    # Offline learning-rate scale relative to the waking E2_harm_s LR.
    # Default 0.1 per the C6 architectural commitment.
    mech273_offline_lr_scale: float = 0.1
    # Bounded number of offline gradient steps per sleep cycle. Default
    # 100 per the C6 commitment. Set <= 0 to disable the pass while
    # leaving the aggregator wired (diagnostic mode).
    mech273_offline_n_steps: int = 100
    # Multiplicative staleness decay applied at WRITEBACK to the regions
    # touched by replay during the cycle. 1.0 = no decay; 0.5 = halve
    # staleness on replayed regions; 0.0 = clear them entirely.
    mech273_partial_decay_factor: float = 0.5

    # MECH-286: override-gated sleep-mode entry (SD-037 wake-stability axis).
    # When True (and use_sleep_loop is True), SleepLoopManager evaluates
    # joint permit conditions before run_sleep_cycle: override_signal below
    # theta_sleep_permit, MECH-284 max region staleness above
    # theta_sleep_recruit, z_harm_a tonic below threat_tonic_threshold.
    # Blocked cycles reset the K-episode counter and return gate diagnostics
    # without advancing cycle_index. Requires use_staleness_accumulator for
    # the staleness leg; use_broadcast_override for hyperarousal lesion tests.
    # Bit-identical OFF preserved.
    use_mech286_sleep_onset_gate: bool = False
    mech286_theta_sleep_permit: float = 0.5
    mech286_theta_sleep_recruit: float = 0.3
    mech286_threat_tonic_threshold: float = 0.4

    # ----------------------------------------------------------------
    # MECH-295: drive -> liking-stream -> approach_cue bridge (weak reading)
    # ----------------------------------------------------------------
    # Master switch. When True, REEAgent activates the MECH-295 bridge:
    #   (a) update_z_goal() writes an anticipatory liking pulse to
    #       VALENCE_LIKING at the goal location, scaled by
    #       drive_level * z_goal_norm * mech295_drive_to_liking_gain.
    #   (b) select_action() computes a per-candidate liking signal
    #       (drive * goal_proximity_to_candidate) and converts it to a
    #       per-candidate negative score_bias (E3 lower-is-better) so the
    #       liking-stream supplies the cue-side approach pull.
    # Weak reading (per claims.yaml MECH-295 functional_restatement):
    # baseline liking-stream activation is sufficient; what matters is
    # that the bridge wiring is intact -- if severed, drive amplification
    # produces no approach regardless of drive magnitude.
    # False = disabled (default, backward compat).
    use_mech295_liking_bridge: bool = False
    # (a) Multiplier on drive_level * z_goal_norm for the anticipatory
    # liking write at the goal location. Setting to 0 disables the
    # write side without touching the cue side.
    mech295_drive_to_liking_gain: float = 1.0
    # (b) Multiplier converting the per-candidate liking signal into the
    # additive approach-side score_bias. Lower-is-better in E3, so the
    # bias is negated internally; this gain controls the magnitude of
    # the approach pull. Setting to 0 disables the cue side without
    # touching the write side -- this is the "severed bridge" arm of
    # the MECH-295 weak-necessity test.
    mech295_liking_to_approach_cue_gain: float = 0.5
    # Drive floor below which the bridge is silent (avoids noisy writes
    # when sated). Lowered 2026-05-12 from 0.1 -> 0.01 per V3-EXQ-540c
    # probe finding: the legacy 0.1 floor was never crossed in standard
    # env configs (drive_level max=0.030 across 1087 bridge calls),
    # short-circuiting MECH-307 conjunction-bias evaluation on every
    # call. 0.01 preserves the "some unmet need" semantic while letting
    # the bridge fire under realistic drive levels.
    mech295_min_drive_to_fire: float = 0.01
    # Goal-norm floor below which the bridge does not fire (no goal
    # seeded; nothing to be congruent with).
    mech295_min_z_goal_norm_to_fire: float = 0.05

    # MECH-307 Anticipatory Affect Conjunction Architecture (registered 2026-05-08).
    # Four-gap substrate fix that makes excitement and dread emerge as derived
    # conjunction-states from the existing 4 valence channels rather than adding
    # new channels. All flags default False for bit-identical backward
    # compatibility with the pre-2026-05-08 substrate.
    #
    # Gap 1: signed VALENCE_SURPRISE write. The existing MECH-205 path stores
    #   only |PE| (max(0, pe_mag - pe_ema)). Biology stores signed RPE; harm-
    #   paired surprise (negative PE) and benefit/novel-cue surprise (positive
    #   PE) get collapsed identically into the residue field. With this flag,
    #   the write site stores -surprise when concurrent harm_signal < 0 and
    #   +surprise otherwise -- consumers reading sign get differential
    #   information, consumers reading magnitude get current behaviour.
    use_mech307_signed_pe: bool = False
    # Gap 1 Option-b (2026-05-11 user override): split VALENCE_SURPRISE into
    #   two separate channels VALENCE_POSITIVE_SURPRISE / VALENCE_NEGATIVE_SURPRISE
    #   in the residue field. Mutually exclusive with use_mech307_signed_pe
    #   (split takes precedence when both are True). Under the split path,
    #   the agent ALSO continues to write magnitude into the legacy
    #   VALENCE_SURPRISE channel so MECH-205 / SD-014 consumers reading the
    #   magnitude slot are bit-identical to the legacy substrate.
    #   When this flag is False, the residue field's indices 4-5 (positive /
    #   negative surprise) stay zeroed.
    use_mech307_split_surprise: bool = False
    # Gap 2 + Gap 3: MECH-216 schema readout writes to multiple channels. The
    #   existing path writes to VALENCE_WANTING only. With this flag, MECH-216
    #   ALSO writes proportional anticipatory VALENCE_LIKING and adds a
    #   salience-proportional pulse to z_beta (arousal). This implements the
    #   anatomical convergence at NAcc-anticipation that biology shows: cue-
    #   stage activation drives wanting + partial hedonic + ANS arousal jointly,
    #   not in isolation.
    use_mech307_schema_multichannel: bool = False
    # Gap 4: MECH-216 writes at predicted z_world (e1_prior), not at current
    #   z_world. With this flag, the schema-readout write target is the E1
    #   forward prediction cached during _e1_tick, falling back to current
    #   z_world if no cached prediction is available. This mirrors hippocampal
    #   place-cell preplay marking the predicted goal location, not the agent's
    #   current location.
    use_mech307_predicted_location_write: bool = False
    # Gain knobs (active only when the corresponding flag is True).
    # Multiplier on schema_salience x drive_level for the anticipatory
    # VALENCE_LIKING write under Gap 2.
    mech307_anticipatory_liking_gain: float = 0.5
    # Multiplier converting schema_salience into the z_beta arousal pulse
    # under Gap 2/3. Pulse is added to z_beta[..., 0] in-place.
    mech307_z_beta_schema_gain: float = 0.3
    # MECH-307 Path B: consumer-side conjunction read at MECH-295 bridge.
    #   The four-gap substrate fix populates the signals (signed surprise,
    #   anticipatory liking, z_beta pulse, predicted-location wanting). The
    #   legacy MECH-295 cue path is drive * goal_proximity ONLY -- it does
    #   not read those signals, so EXQ-539 fired all four substrate counters
    #   (C1-C4 PASS) without lifting approach_commit_rate (C5 FAIL). This
    #   flag wires the bridge's compute_conjunction_score_bias() into
    #   select_action(): when the four-way conjunction holds at a candidate's
    #   predicted-imminent location (wanting + liking + signed-positive
    #   surprise + z_beta arousal all above their thresholds), the candidate
    #   gets an additional negative bias of -mech307_conjunction_gain *
    #   drive_level. Default False -> bit-identical OFF.
    use_mech307_consumer_conjunction_read: bool = False
    # Conjunction predicate thresholds (match the doc's
    # is_excitement_state_at(...) defaults at lines 128-137).
    mech307_conjunction_wanting_threshold: float = 0.6
    mech307_conjunction_liking_threshold: float = 0.3
    # Lowered 2026-05-12 from 0.6 -> 0.3 per V3-EXQ-540c probe finding:
    # observed z_beta_arousal under use_mech307_conjunction has
    # max=0.545 mean=0.518 across 1087 bridge calls -- the legacy 0.6
    # floor sits above the achievable ceiling. The half-tier 0.3
    # cleared 94.66% of reads in the probe; 0.3 is the empirical sweet
    # spot.
    mech307_conjunction_z_beta_threshold: float = 0.3
    # Negative-score gain applied per candidate whose conjunction predicate
    # fires. Bias term = -mech307_conjunction_gain * drive_level when the
    # predicate holds; 0 otherwise.
    mech307_conjunction_gain: float = 1.0
    # MECH-307 master convenience flag (2026-05-11). When True, the
    # post-init resolver sets all four substrate-side sub-flags True:
    #   use_mech307_split_surprise        (Gap 1 Option-b)
    #   use_mech307_schema_multichannel   (Gap 2 + Gap 3)
    #   use_mech307_predicted_location_write (Gap 4)
    # The consumer-side flag use_mech307_consumer_conjunction_read is NOT
    # auto-set (it is a downstream consumer wiring decision, not a
    # substrate change). Setting this master flag overrides any explicit
    # False on the three sub-flags above. Bit-identical OFF by default.
    use_mech307_conjunction: bool = False

    # SD-049 Phase 3: SD-032 consumer cascade reading per_axis_drive directly.
    # Master switch (default False; bit-identical OFF). When True AND the env
    # has multi_resource_heterogeneity_enabled + per_axis_drive_enabled
    # (obs_dict["per_axis_drive"] is surfaced each tick), the agent threads
    # the per-axis drive vector through every SD-032 consumer tick(). Each
    # consumer then either collapses the vector to a scalar via its own
    # combiner knob (whole-organism control modules: AIC / PCC / pACC / dACC /
    # SalienceCoordinator / BroadcastOverrideRegulator -- biology predicts
    # scalar control output) or routes by goal-axis index (MECH-295 liking-
    # bridge -- biology predicts axis-matched wanting/liking dissociation,
    # the canonical use case the SD-049 design doc unblocks for MECH-229 /
    # MECH-117 / MECH-216 / Q-030).
    #
    # When this master flag is False, every consumer call site in agent.py
    # passes per_axis_drive=None and the legacy scalar drive_level path is
    # bit-identical. When True but env per-axis is OFF (no per_axis_drive
    # in obs_dict), the agent still passes None -- the gate is "vector
    # available AND master flag" rather than master-flag-only.
    #
    # Per-consumer combiner defaults are biology-anchored:
    #   AIC      "max"  -- urgency / interoceptive salience tracks worst axis.
    #   PCC      "mean" -- whole-organism fatigue integrates across axes.
    #   pACC     "sum"  -- allostatic-load style accumulation (Baliki 2012).
    #   dACC     "max"  -- control demand follows the most-pressing axis.
    #   SalCoord "max"  -- external-task affinity scales with worst axis.
    #   Override "max"  -- orexin recruits on the worst-deficit axis
    #                       (Mileykovskiy 2005).
    #   MECH-295 "max"  -- fallback when goal_axis_idx is None; primary
    #                       path is per_axis_drive[goal_axis_idx].
    # All combiners take values in {"max", "mean", "sum"}; ValueError raised
    # on unknown values via ree_core.utils.per_axis_drive.validate_combiner.
    #
    # See REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
    # Phase 3 section + ree_core/utils/per_axis_drive.py shared helper.
    use_sd049_per_axis_consumer_cascade: bool = False
    sd049_aic_per_axis_combiner: str = "max"
    sd049_pcc_per_axis_combiner: str = "mean"
    sd049_pacc_per_axis_combiner: str = "sum"
    sd049_dacc_per_axis_combiner: str = "max"
    sd049_salience_per_axis_combiner: str = "max"
    sd049_override_per_axis_combiner: str = "max"
    sd049_mech295_per_axis_combiner: str = "max"

    # MECH-423 R3 (module-tagged interleaved cross-module consolidation; MECH-121
    # consolidation cluster). The legacy MECH-121 offline pass trains e2_harm_s
    # ALONE (sleep/phase_manager.py) with region-keyed traces carrying no module
    # identity, so cross_module_replay_share is unmeasurable and the integrated
    # E1<->E2 representation cannot be acquired under an interleaved schedule.
    # When use_cross_module_consolidation=True, SleepLoopManager._run_cycle runs a
    # CrossModuleConsolidator pass (sleep/cross_module_consolidation.py) AFTER the
    # existing writeback: an interleaved (alternate-modules-step-by-step) gradient
    # schedule over E1 (world-model) + E2 (world_forward) sourced from the agent's
    # replay buffers, tagging each replayed trace by which modules it updated, and
    # merging {cross_module_consolidation_n_updates,
    # cross_module_consolidation_cross_module_replay_share,
    # cross_module_consolidation_interleaved, ...} into the sleep-cycle metrics.
    # MECH-094: this is the SAME explicit exception the e2_harm_s writeback uses --
    # a weight-update pass with optimisers constructed LOCALLY over only the named
    # modules' parameters; it writes NO residue / anchor / memory content.
    # Grounds McClelland 1995 + Kumaran 2016 CLS (interleaving is necessary for
    # shared-representation integration; a blocked schedule -> catastrophic
    # interference -> sub-additive artefact). Disabled by default -> bit-identical
    # OFF (consolidator not built; _run_cycle unchanged). EXP-0380 R3 asserts
    # n_updates > 0 AND cross_module_replay_share > 0 AND interleaved == True;
    # schedule="blocked" is the pre-registered control.
    use_cross_module_consolidation: bool = False
    cross_module_consolidation_schedule: str = "interleaved"  # "interleaved" | "blocked"
    cross_module_consolidation_steps: int = 0  # offline gradient steps per cycle (0 == none)
    cross_module_consolidation_lr: float = 1e-3
    cross_module_consolidation_batch: int = 16

    # MECH-457: first-class RPE-driven actor-critic action-learning substrate
    # (sd_actor_critic_action_learning). A dorsal-striatal-analog actor + value
    # critic, architecturally distinct from the lateral_pfc / ofc bias_head
    # REINFORCE readout. Routed from failure_autopsy_734-737 (V3-EXQ-737: a
    # trainable actor+critic over the FROZEN z_world scored below random -> the
    # action-learning loss must CO-SHAPE the representation). All defaults no-op:
    # use_actor_critic=False -> agent.action_critic is None, select_action
    # byte-identical. See ree_core/action_learning/actor_critic.py and
    # REE_assembly/docs/architecture/sd_actor_critic_action_learning.md.
    use_actor_critic: bool = False
    # The co-shaping ablation lever (the mandatory frozen-vs-co-trained arm). When
    # True the actor reads live z_world (gradient reaches the z_world encoder =
    # co-shaping); when False it reads z_world.detach() (= 737's frozen arm).
    # Consumed by REEAgent.actor_critic_step.
    actor_critic_cotrain_encoder: bool = False
    # Critic form: False -> plain learned value head V(z) (cand-A arms A0/A1);
    # True -> successor-feature critic V_SF = psi(z).w (cand-B arms A2/A3), w
    # grounded in the MECH-229 VALENCE_WANTING reward channel.
    actor_critic_use_sf_critic: bool = False
    actor_critic_hidden: int = 128
    actor_critic_sf_feature_dim: int = 32

    def __post_init__(self) -> None:
        # MECH-307 master flag resolver. When the convenience master flag
        # is set, force the three substrate-side sub-flags True so callers
        # can flip one switch rather than threading three. Explicit
        # sub-flags True without the master flag continue to work as
        # before (the resolver only flips False -> True under the master).
        if self.use_mech307_conjunction:
            self.use_mech307_split_surprise = True
            self.use_mech307_schema_multichannel = True
            self.use_mech307_predicted_location_write = True

        # GAP-3 sleep-aggregation cluster master flag resolver. Same OR-only
        # convention as MECH-307: flips the eight Phase A-E sub-flags
        # False -> True so a single switch lights the whole offline-
        # consolidation pathway. from_dims() handles the factory path
        # separately (it sets fields after cls(), so it re-invokes the
        # bundle via enable_sleep_aggregation_cluster()).
        if self.use_sleep_aggregation_cluster:
            self.enable_sleep_aggregation_cluster()

        # MECH-090 R-c conjunction master flag resolver. OR-only: when
        # the conjunction is enabled, the CommitReadiness module must be
        # instantiated (the gate has no readiness signal to consult
        # otherwise). Explicit use_commit_readiness=True without the
        # conjunction continues to work (passive diagnostic mode).
        if self.use_mech090_readiness_conjunction:
            self.use_commit_readiness = True

    def enable_sleep_aggregation_cluster(self) -> "REEConfig":
        """Enable the full Phase A-E sleep-aggregation cluster (GAP-3).

        Convenience preset, not a new mechanism. Forces the eight
        independent default-False master flags True so the offline-
        consolidation pathway (MECH-285 replay sampler -> MECH-272 routing
        gate + GAP-8 consumer -> MECH-275 Bayesian aggregator -> MECH-273
        self-model writeback) runs end-to-end from one switch. Before this,
        the cluster was silent unless an experiment set all eight by hand
        (sleep_substrate_plan.md GAP-3).

        OR-only, like the use_mech307_conjunction resolver: only flips
        False -> True. MECH-204 precision recalibration is a separate
        sibling step (GAP-1) and is deliberately not turned on here.
        Substrate prerequisites (anchor sets for Phase B, e2_harm_s for
        Phase E) are separate MECH-269/ARC-033 switches and are not
        bundled here -- the cluster flag governs the sleep-phase flags
        only, which is exactly GAP-3's scope.
        """
        self.use_sleep_aggregation_cluster = True
        self.use_sleep_loop = True               # Phase A
        self.sws_enabled = True                  # SD-017 SWS pass
        self.rem_enabled = True                  # SD-017 REM pass
        self.use_mech285_sampler = True          # Phase B
        self.use_mech272_routing = True          # Phase C gate
        self.use_mech272_routing_consumer = True  # Phase C consumer (GAP-8)
        self.use_mech275_aggregator = True       # Phase D
        self.use_mech273_self_model = True       # Phase E
        return self

    def enable_goal_stream(
        self,
        *,
        goal_weight: float = 0.5,
        wanting_weight: float = 0.5,
        benefit_threshold: float = 0.05,
        drive_weight: float = 2.0,
        schema_wanting_threshold: float = 0.10,
        schema_wanting_gain: float = 0.60,
        use_resource_encoder: bool = True,
        use_mech307: bool = True,
        use_consumer_conjunction_read: bool = True,
    ) -> "REEConfig":
        """Enable the canonical goal-stream heartbeat bundle.

        This is a convenience preset, not a new mechanism. It turns on the
        pieces that V3-EXQ-559 showed must be co-enabled for a live stream:
        contact-gated z_goal seeding, E3 goal scoring, schema wanting,
        hippocampal wanting reads, MECH-295 cue bridge, and the MECH-307
        write/read path.
        """
        self.goal_stream_enabled = True

        self.goal.z_goal_enabled = True
        self.goal.goal_weight = float(goal_weight)
        self.goal.benefit_threshold = float(benefit_threshold)
        self.goal.drive_weight = float(drive_weight)
        self.e3.goal_weight = float(goal_weight)

        self.e1.schema_wanting_enabled = True
        self.schema_wanting_threshold = float(schema_wanting_threshold)
        self.schema_wanting_gain = float(schema_wanting_gain)
        self.hippocampal.wanting_weight = float(wanting_weight)

        if use_resource_encoder:
            self.latent.use_resource_encoder = True
            self.latent.z_resource_dim = self.goal.goal_dim

        self.use_mech295_liking_bridge = True
        self.mech295_drive_to_liking_gain = 1.0
        self.mech295_liking_to_approach_cue_gain = 0.5
        self.mech295_min_drive_to_fire = 0.01
        self.mech295_min_z_goal_norm_to_fire = 0.03

        if use_mech307:
            self.surprise_gated_replay = True
            self.use_mech307_split_surprise = True
            self.use_mech307_schema_multichannel = True
            self.use_mech307_predicted_location_write = True
            self.use_mech307_consumer_conjunction_read = bool(
                use_consumer_conjunction_read
            )
            self.mech307_conjunction_gain = 1.0
            self.mech307_conjunction_wanting_threshold = 0.10
            self.mech307_conjunction_liking_threshold = 0.05
            self.mech307_conjunction_z_beta_threshold = 0.10

        return self

    @classmethod
    def from_dims(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        self_dim: int = 32,
        world_dim: int = 32,
        action_object_dim: int = 16,
        harm_dim: int = 0,
        alpha_world: float = 0.3,
        alpha_self: float = 0.3,
        reafference_action_dim: int = 0,
        use_event_classifier: bool = False,
        use_resource_proximity_head: bool = False,
        resource_proximity_weight: float = 0.5,
        use_harm_stream: bool = False,
        harm_obs_dim: int = 51,
        z_harm_dim: int = 32,
        # SD-011: affective-motivational harm stream
        use_affective_harm_stream: bool = False,
        harm_obs_a_dim: int = 50,
        z_harm_a_dim: int = 16,
        # SD-011 second source: harm history window
        harm_history_len: int = 0,
        z_harm_a_aux_loss_weight: float = 0.1,
        # SD-016: frontal cue-indexed integration
        sd016_enabled: bool = False,
        # SD-016 ContextMemory write-path mode (EXQ-477 follow-up):
        # "off" | "train_only" | "sense_only" | "both"
        sd016_writepath_mode: str = "off",
        # SD-016 Path 1 (V3-EXQ-418e): auxiliary diversification loss weight
        # on ContextMemory slots. 0.0 = no-op (legacy substrate). Recommended
        # 0.5 when sd016_enabled=True (mirrors LAMBDA_CUE_ACTION).
        # See REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md.
        sd016_diversification_weight: float = 0.0,
        # SD-016 Path 4 (V3-EXQ-418g): learnable attention temperature on the
        # z_world-only ContextMemory query inside extract_cue_context. Pair with
        # an attention-entropy minimisation loss term in the experiment script
        # (agent.e1.compute_attention_entropy_loss(z_world)). Default False =
        # bit-identical to legacy fixed sqrt(memory_dim) scale.
        sd016_temperature_learnable: bool = False,
        # SD-016 Path 3 (2026-06-05): feedforward cue->slot tagger replacing the
        # saddle-stuck q.k attention slot-selection in extract_cue_context.
        # Requires sd016_enabled=True. Default False = legacy attention branch.
        sd016_cue_slot_tagger: bool = False,
        sd016_cue_slot_tagger_hidden: int = 32,
        sd016_cue_slot_tagger_temperature: float = 1.0,
        # ARC-030 / MECH-111 / MECH-112 / MECH-113
        benefit_eval_enabled: bool = False,
        benefit_weight: float = 1.0,
        novelty_bonus_weight: float = 0.0,
        self_maintenance_weight: float = 0.0,
        self_maintenance_d_eff_target: float = 1.5,
        # SD-011: z_harm_a E3 integration
        urgency_weight: float = 0.0,
        urgency_max: float = 0.5,
        affective_harm_scale: float = 0.0,
        # MECH-112 / MECH-116: z_goal substrate
        z_goal_enabled: bool = False,
        alpha_goal: float = 0.05,
        decay_goal: float = 0.005,
        benefit_threshold: float = 0.1,
        goal_weight: float = 1.0,
        e1_goal_conditioned: bool = True,
        drive_weight: float = 2.0,  # SD-012: benefit amplification when depleted
        drive_ema_alpha: float = 1.0,  # SD-012 GAP-3 Option 1: sustained-drive EMA (1.0=OFF, bit-identical; 0.02~35-step half-life)
        drive_floor: float = 0.0,  # SD-012 GAP-3 Option 2: insatiability floor on drive_level (0.0=disabled, bit-identical)
        valence_wanting_floor: float = 0.0,  # MECH-186: minimum z_goal norm floor (0=disabled)
        z_goal_seeding_gain: float = 1.0,  # MECH-187: gain on seeding signal (1.0=no change)
        z_goal_inject: float = 0.0,  # MECH-188: PFC top-down injection norm floor (0=disabled)
        use_incentive_token_bank: bool = False,  # SD-057: object-bound incentive-salience bank (GAP-7 L2-L3-L4)
        incentive_decay: float = 0.005,  # SD-057 L3: per-object token slow decay
        incentive_value_alpha: float = 0.1,  # SD-057 L3: token revaluation EMA rate
        incentive_drive_kappa_weight: float = 2.0,  # SD-057 L3: relocated drive_weight for value x kappa(drive)
        incentive_drive_kappa_scale: float = 1.0,  # SD-049-PHASE-2 drive-coupling amend: no-op-default scale on effective kappa (1.0=bit-identical; >1 lets a realistic per-axis drive spread compete with real object base_value gaps)
        incentive_use_per_axis_drive: bool = True,  # SD-057 L3: drive-specific (per-axis) wanting
        use_cue_recall: bool = False,  # SD-057 phase-2 L6 (MECH-347): cue-triggered wanting path
        cue_recall_gain: float = 0.05,  # SD-057 L6: z_goal cue-pull strength
        cue_recall_min_proximity: float = 0.0,  # SD-057 L6: auto-perception proximity floor
        use_super_ordinal_goal_anchors: bool = False,  # MECH-189: ContextMemory writes substrate (super-ordinal goal anchors)
        super_ordinal_n_slots: int = 16,  # MECH-189: cue-indexed anchor slot count
        super_ordinal_salience_threshold: float = 0.5,  # MECH-189 WRITE gate (a): high-salience benefit floor
        super_ordinal_complexity_mode: str = "novelty",  # MECH-189 WRITE gate (b): "novelty" | "external" (DEV-NEED-024 adjudication)
        super_ordinal_complexity_threshold: float = 0.3,  # MECH-189 WRITE gate (b): min contextual complexity to write
        super_ordinal_merge_similarity: float = 0.8,  # MECH-189 WRITE: reinforce-vs-allocate cosine cutoff
        super_ordinal_write_alpha: float = 0.3,  # MECH-189 WRITE: EMA blend rate into an anchor slot
        super_ordinal_seed_below_norm: float = 0.4,  # MECH-189 READ: only seed when z_goal norm below this
        super_ordinal_seed_match_threshold: float = 0.3,  # MECH-189 READ: min retrieval cosine match to seed
        super_ordinal_seed_strength: float = 0.1,  # MECH-189 READ: cue-pull strength toward retrieved anchor
        use_mech_consume: bool = False,  # SD-057 phase-2 L7 (MECH-348): dACC object-discriminative goal readout
        dacc_goal_readout_weight: float = 0.0,  # SD-057 L7: dACC goal-readout bias weight
        # MECH-203/204: serotonergic neuromodulation
        tonic_5ht_enabled: bool = False,
        # MECH-204 F1: cross-cycle persistent zero-point EMA alpha
        precision_zero_point_ema_alpha: float = 0.1,
        # MECH-205: surprise-gated replay
        surprise_gated_replay: bool = False,
        pe_ema_alpha: float = 0.02,
        pe_surprise_threshold: float = 0.001,
        # MECH-120: SHY-analog synaptic homeostasis
        shy_enabled: bool = False,
        shy_decay_rate: float = 0.85,
        # SD-017: minimal sleep-phase infrastructure
        sws_enabled: bool = False,
        sws_consolidation_steps: int = 5,
        sws_schema_weight: float = 0.1,
        rem_enabled: bool = False,
        rem_attribution_steps: int = 10,
        # MECH-165: reverse replay diversity scheduler
        replay_diversity_enabled: bool = False,
        reverse_replay_fraction: float = 0.3,
        random_replay_fraction: float = 0.2,
        exploration_buffer_len: int = 50,
        # VALENCE_WANTING gradient in trajectory scoring
        wanting_weight: float = 0.0,
        # MECH-216: E1 predictive wanting (schema readout)
        schema_wanting_enabled: bool = False,
        schema_wanting_threshold: float = 0.3,
        schema_wanting_gain: float = 0.5,
        # ARC-033: E2_harm_s forward model (sensory-discriminative harm stream predictor)
        use_e2_harm_s_forward: bool = False,
        use_e2_world_forward: bool = False,
        # SD-056: E2 action-conditional divergence preservation
        # (auxiliary InfoNCE contrastive loss on world_forward)
        e2_action_contrastive_enabled: bool = False,
        e2_action_contrastive_weight: float = 0.01,
        e2_action_contrastive_temperature: float = 0.1,
        e2_action_contrastive_min_batch_classes: int = 2,
        # SD-056 multi-step rollout stability amend (2026-05-31)
        e2_action_contrastive_multistep_enabled: bool = False,
        e2_action_contrastive_horizon: int = 5,
        e2_action_contrastive_horizon_weights_decay: float = 1.0,
        e2_rollout_output_norm_clamp_enabled: bool = False,
        e2_rollout_output_norm_clamp_ratio: float = 2.0,
        # cross_stream_binding_substrate (2026-07-08); all no-op default.
        cross_stream_binding_enabled: bool = False,
        cross_stream_binding_dim: int = 16,
        cross_stream_binding_strength: float = 0.15,
        cross_stream_binding_theta_period: int = 4,
        # learned (plastic) binder (2026-07-09); all no-op default.
        cross_stream_binding_learned: bool = False,
        cross_stream_binding_lr: float = 1e-3,
        cross_stream_binding_temperature: float = 0.5,
        cross_stream_binding_buffer_size: int = 512,
        cross_stream_binding_batch: int = 64,
        cross_stream_binding_conv_frac: float = 0.85,
        # SD-022: directional limb damage
        limb_damage_enabled: bool = False,
        damage_increment: float = 0.15,
        failure_prob_scale: float = 0.3,
        heal_rate: float = 0.002,
        # SD-014: valence channel writes
        valence_harm_enabled: bool = False,
        valence_liking_enabled: bool = False,
        liking_threshold: float = 0.1,
        # MECH-258: E2_harm_a affective-pain forward model (SD-032b prerequisite)
        use_e2_harm_a: bool = False,
        e2_harm_a_lr: float = 5e-4,
        # ARC-058: shared-trunk harm forward hypothesis (competes with ARC-033)

        use_shared_harm_trunk: bool = False,
        # SD-032b: dACC/aMCC-analog adaptive control
        use_dacc: bool = False,
        dacc_weight: float = 0.0,
        dacc_interaction_weight: float = 0.0,
        dacc_foraging_weight: float = 0.0,
        dacc_suppression_weight: float = 0.0,
        dacc_suppression_memory: int = 8,
        dacc_precision_scale: float = 500.0,
        dacc_effort_cost: float = 0.1,
        dacc_drive_coupling: float = 0.0,
        dacc_bias_max_abs: float = 0.0,
        # MECH-268: history-conditioned PE saturation
        dacc_saturation_enabled: bool = False,
        dacc_saturation_window: int = 8,
        dacc_saturation_strength: float = 0.3,
        dacc_saturation_grace: int = 2,
        # SD-032a: salience-network coordinator
        use_salience_coordinator: bool = False,
        salience_switch_threshold: float = 1.0,
        salience_stability_scaling: float = 1.0,
        salience_softmax_temperature: float = 1.0,
        salience_external_task_bias: float = 1.0,
        salience_dacc_pe_weight: float = 1.0,
        salience_dacc_foraging_weight: float = 0.5,
        salience_apply_to_dacc_bias: bool = False,
        # mode-governance-engagement: external_task salience source (no-op default).
        use_external_task_drive: bool = False,
        external_task_drive_affinity_weight: float = 1.0,
        external_task_drive_salience_weight: float = 1.0,
        external_task_drive_commit_weight: float = 1.0,
        external_task_drive_proximity_weight: float = 1.0,
        external_task_drive_require_goal_active: bool = True,
        # SD-032c: AIC-analog interoceptive-salience / urgency
        use_aic_analog: bool = False,
        aic_baseline_alpha: float = 0.02,
        aic_drive_coupling: float = 1.0,
        aic_urgency_threshold: float = 1.0,
        aic_base_attenuation: float = 0.5,
        aic_drive_protect_weight: float = 1.0,
        aic_extra_weight: float = 0.0,
        # SD-032d: PCC-analog metastability scalar
        use_pcc_analog: bool = False,
        pcc_success_alpha: float = 0.02,
        pcc_success_weight: float = 0.5,
        pcc_fatigue_weight: float = 0.5,
        pcc_offline_recency_window: int = 500,
        pcc_offline_weight: float = 0.3,
        pcc_stability_baseline: float = 0.5,
        # SD-032e: pACC-analog autonomic write-back
        use_pacc_analog: bool = False,
        pacc_drive_alpha: float = 0.002,
        pacc_drive_scale: float = 1.0,
        pacc_drive_bias_cap: float = 0.5,
        pacc_z_harm_a_min: float = 0.0,
        pacc_offline_decay: float = 0.0,
        # MECH-095: TPJ comparator
        use_tpj_comparator: bool = False,
        tpj_agency_threshold: float = 0.5,
        # SD-033a: Lateral-PFC-analog (rule/goal substrate, MECH-261 consumer)
        use_lateral_pfc_analog: bool = False,
        lateral_pfc_rule_dim: int = 16,
        lateral_pfc_update_eta: float = 0.05,
        lateral_pfc_world_pool_weight: float = 0.5,
        lateral_pfc_bias_scale: float = 0.1,
        lateral_pfc_hidden_dim: int = 32,
        lateral_pfc_use_discriminator_source: bool = False,
        lateral_pfc_discriminator_pool_weight: float = 0.3,
        lateral_pfc_train_rule_bias_head: bool = False,
        # MECH-457 actor-critic action-learning substrate (all no-op default).
        use_actor_critic: bool = False,
        actor_critic_cotrain_encoder: bool = False,
        actor_critic_use_sf_critic: bool = False,
        actor_critic_hidden: int = 128,
        actor_critic_sf_feature_dim: int = 32,
        # ARC-063 v1: distributed CandidateRule field (GAP-B rule-creator)
        use_candidate_rule_field: bool = False,
        crf_n_slots: int = 16,
        crf_rule_dim: int = 16,
        crf_mint_recurrence_threshold: int = 3,
        crf_tolerance_floor: float = 0.3,
        crf_tolerance_conflict_gain: float = 1.0,
        crf_availability_alpha: float = 0.1,
        crf_availability_decay: float = 0.005,
        crf_eligibility_window: int = 20,
        crf_context_match_threshold: float = 0.5,
        crf_seed_from_arc062: bool = True,
        crf_persist_rules_across_episode_reset: bool = False,
        # ARC-063 amend (V3-EXQ-654b): mature-pool gate/credit/retire dynamics
        crf_mature_pool_dynamics: bool = False,
        crf_mature_availability_decay: float = 0.001,
        crf_mature_retire_floor: float = 0.05,
        crf_mature_availability_alpha_negative: float = 0.02,
        crf_mature_tolerance_floor: float = 0.15,
        crf_mature_tolerance_conflict_gain: float = 0.25,
        crf_mature_mint_block_threshold: float = 0.8,
        crf_mature_mint_protection_ticks: int = 30,
        crf_context_from_e2_world_forward: bool = False,
        # crf-availability-maintenance (V3-EXQ-666 successor; ARC-063 amend)
        crf_availability_maintenance: bool = False,
        crf_maintenance_floor: float = 0.45,
        crf_maintenance_decay: float = 0.0,
        crf_engaged_sustain: bool = False,
        crf_engaged_sustain_rate: float = 0.1,
        crf_maintained_reactivation_threshold: float = 0.0,
        # CRF conflict-gate calibration amend (V3-EXQ-654d successor; no-op default)
        crf_mature_context_match_threshold: float = -1.0,
        crf_tolerance_conflict_cap: int = -1,
        crf_maintenance_couple_to_theta: bool = False,
        # SD-033b: OFC-analog (specific-outcome / task-structure substrate)
        use_ofc_analog: bool = False,
        ofc_state_dim: int = 16,
        ofc_update_eta: float = 0.05,
        ofc_outcome_pool_weight: float = 0.5,
        ofc_bias_scale: float = 0.1,
        ofc_hidden_dim: int = 32,
        ofc_harm_dim: int = 0,
        use_ofc_outcome_oracle: bool = False,
        ofc_train_state_bias_head: bool = False,
        # SD-033b GAP-8 devaluation-head decouple (failure_autopsy V3-EXQ-485l)
        use_ofc_devaluation_head: bool = False,
        ofc_devaluation_bias_scale: float = 2.0,
        ofc_train_devaluation_head: bool = False,
        # ARC-062 Phase 1: gated-policy heads + context discriminator
        use_gated_policy: bool = False,
        gated_policy_n_heads: int = 2,
        gated_policy_disc_hidden: int = 24,
        gated_policy_disc_init_scale: float = 0.1,
        gated_policy_head_hidden: int = 32,
        gated_policy_bias_scale: float = 0.1,
        gated_policy_head_init_bias_offset: float = 0.05,
        gated_policy_use_first_action_onehot: bool = False,
        # INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection crystallization
        crystallize_at_phase3: bool = False,
        gated_policy_crystallize_expansion_hidden: int = 32,
        residue_ewc_lambda: float = 0.0,
        # MECH-313 (ARC-065): stochastic_noise_floor (LC-NE tonic / SAC analog)
        use_noise_floor: bool = False,
        noise_floor_alpha: float = 0.1,
        noise_floor_min_temperature: float = 1.0,
        # MECH-314 (ARC-065): structured_curiosity_bonus (frontopolar /
        # EFE analog) + 3 sub-flavour switches (314a/b/c)
        use_structured_curiosity: bool = False,
        use_curiosity_novelty: bool = True,
        use_curiosity_uncertainty: bool = True,
        use_curiosity_learning_progress: bool = True,
        curiosity_novelty_weight: float = 0.05,
        curiosity_uncertainty_weight: float = 0.05,
        curiosity_learning_progress_weight: float = 0.05,
        curiosity_bias_scale: float = 0.1,
        curiosity_lp_ema_alpha: float = 0.1,
        curiosity_lp_window_k: int = 5,
        # MECH-314a Phase 2 (Candidate 5A): novelty-source + augmentation
        curiosity_novelty_source: Literal[
            "residue", "visitation", "auto"
        ] = "residue",
        curiosity_visitation_buffer_len: int = 256,
        curiosity_use_first_action_onehot: bool = False,
        curiosity_first_action_augmentation_policy: Literal[
            "never", "auto", "always"
        ] = "never",
        curiosity_min_spread_threshold: float = 0.01,
        curiosity_min_spread_consecutive_ticks: int = 5,
        curiosity_candidate_source: Literal[
            "proposer", "e2_world_forward"
        ] = "proposer",
        # ARC-065 GAP-A: shared cand_world_summaries source for the E3-side
        # bias channels (lateral_pfc / ofc / mech295 / gated_policy / vigor).
        candidate_summary_source: Literal[
            "proposer", "e2_world_forward"
        ] = "proposer",
        # MECH-320 (ARC-066 child): tonic_vigor_coupling_score_bias
        # (mesolimbic-DA-vigor / avg-reward-rate / opportunity-cost regulator)
        use_tonic_vigor: bool = False,
        tonic_vigor_half_life: float = 100.0,
        tonic_vigor_w_action: float = 0.1,
        tonic_vigor_w_passive: float = 0.1,
        tonic_vigor_bias_scale: float = 0.1,
        tonic_vigor_gate_energy_min: float = 0.2,
        tonic_vigor_gate_drive_max: float = 0.7,
        tonic_vigor_gate_pe_max: float = 1.0,
        tonic_vigor_form: str = "additive",
        tonic_vigor_noop_class: int = 0,
        # V3-EXQ-563: forced floor on v_t; bypasses sign/scale gate for
        # actuator tests. Default 0.0 = standard behaviour.
        tonic_vigor_v_t_floor: float = 0.0,
        # SD-058 / MECH-357: instrumental-avoidance acquisition (ilPFC-analog
        # freeze-suppression + avoidance action pathway + efficacy learning).
        use_instrumental_avoidance: bool = False,
        avoidance_learn_rate: float = 0.05,
        avoidance_leak_rate: float = 0.02,
        avoidance_initial_efficacy: float = 0.0,
        avoidance_scaffold_floor: float = 0.0,
        avoidance_threat_floor: float = 0.1,
        avoidance_threat_ref: float = 0.5,
        avoidance_efficacy_reward_floor: float = 1e-4,
        avoidance_action_bias_gain: float = 0.1,
        avoidance_bias_scale: float = 0.1,
        avoidance_suppression_threshold: float = 0.5,
        avoidance_noop_class: int = 0,
        use_escape_affordance_bridge: bool = False,
        use_escape_relief_credit: bool = True,
        use_escape_safety_credit: bool = True,
        escape_relief_learn_rate: float = 0.1,
        escape_safety_learn_rate: float = 0.1,
        escape_bridge_leak_rate: float = 0.01,
        escape_relief_reward_floor: float = 1e-4,
        escape_threat_floor: float = 0.1,
        escape_threat_ref: float = 0.5,
        escape_approach_gain: float = 0.1,
        escape_bias_scale: float = 0.1,
        escape_noop_class: int = 0,
        escape_use_trained_safety_signal: bool = False,
        escape_safety_signal_threshold: float = 0.5,
        use_object_file_buffer: bool = False,
        obf_max_tokens: int = 5,
        obf_continuity_radius: float = 2.0,
        obf_w_motion: float = 1.0,
        obf_w_feat: float = 1.0,
        obf_feature_alpha: float = 0.3,
        obf_persist_ttl: int = 8,
        obf_min_birth_salience: float = 0.0,
        obf_use_precision_weighting: bool = True,
        use_trainable_escape_affordance_learner: bool = False,
        use_trainable_relief_critic: bool = True,
        use_trainable_safety_predictor: bool = True,
        trainable_escape_bias_scale: float = 0.1,
        trainable_escape_relief_learn_rate: float = 0.1,
        trainable_escape_safety_learn_rate: float = 0.1,
        trainable_escape_leak_rate: float = 0.01,
        trainable_escape_relief_reward_floor: float = 1e-4,
        trainable_escape_relief_target_scale: float = 0.3,
        trainable_escape_threat_floor: float = 0.1,
        trainable_escape_noop_class: int = 0,
        trainable_escape_hidden_dim: int = 32,
        trainable_escape_action_embedding_dim: int = 8,
        trainable_escape_optimizer_lr: float = 0.03,
        trainable_escape_prediction_floor: float = 0.02,
        # Post-603i E2 escape-affordance linker (readout over detached E2
        # action-consequence features; reuse, not a duplicate predictor).
        use_e2_escape_affordance_linker: bool = False,
        use_e2_escape_linker_for_relief_safety: bool = False,
        use_e2_escape_linker_e3_bias: bool = False,
        escape_linker_learn_rate: float = 0.1,
        escape_linker_optimizer_lr: float = 0.03,
        escape_linker_leak_rate: float = 0.01,
        escape_linker_hidden_dim: int = 32,
        escape_linker_action_embedding_dim: int = 8,
        escape_linker_bias_scale: float = 0.1,
        escape_linker_threat_floor: float = 0.1,
        escape_linker_threat_ref: float = 0.5,
        escape_linker_noop_class: int = 0,
        escape_linker_relief_reward_floor: float = 1e-4,
        escape_linker_harm_delta_scale: float = 0.3,
        escape_linker_prediction_floor: float = 0.02,
        escape_linker_block_hypothesis_learning: bool = True,
        # MECH-341 (ARC-065 Layer-B child): e3_scoring_preserves_trajectory_
        # class_diversity. Master + two togglable sub-flavours per V3-EXQ-608
        # R2a_e3_collapse_confirmed_large_gap routing (options 1 + 2).
        use_e3_score_diversity: bool = False,
        use_e3_diversity_entropy_bonus: bool = True,
        use_e3_diversity_stratified_select: bool = True,
        e3_diversity_entropy_lambda: float = 0.5,
        e3_diversity_entropy_bias_scale: float = 1.0,
        e3_diversity_stratified_temperature: float = 1.0,
        # MECH-341 amend (2026-06-01): within-class proportional sampling
        # sharpness; None = legacy argmin (bit-identical OFF). See
        # failure_autopsy_V3-EXQ-616 Sections 7 + 10.
        e3_diversity_stratified_within_class_temperature: Optional[float] = None,
        e3_diversity_min_classes_for_stratification: int = 2,
        # MECH-090 R-c conjunction (commitment_closure GAP-4): commit-entry
        # predicate amendment from rv-only to rv_low AND
        # readiness_above_floor. See ree_core/policy/commit_readiness.py and
        # REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md.
        use_mech090_readiness_conjunction: bool = False,
        mech090_readiness_floor: float = 0.3,
        use_commit_readiness: bool = False,
        commit_readiness_window: int = 20,
        commit_readiness_ema_alpha: float = 0.1,
        commit_readiness_initial: float = 1.0,
        # SD-061: difficulty-gated proposal-entropy regulator (MECH-343
        # blocker part 2). All no-op default -> bit-identical OFF.
        use_difficulty_gated_proposal_entropy: bool = False,
        stuck_progress_window: int = 8,
        stuck_progress_stall_eps: float = 0.01,
        stuck_score_margin_floor: float = 0.05,
        stuck_committed_diversity_window: int = 8,
        stuck_committed_diversity_floor: float = 0.34,
        stuck_choice_difficulty_ref: float = 0.05,
        stuck_goal_salience_floor: float = 0.05,
        stuck_ema_alpha_rise: float = 0.3,
        stuck_ema_alpha_fall: float = 0.05,
        stuck_threshold: float = 0.5,
        stuck_combine_mode: str = "mean",
        dgpe_candidate_widen_max: int = 8,
        dgpe_temperature_gain_max: float = 1.0,
        # MECH-342: maintenance-time readiness-driven commitment release
        # (B3b). The release-side complement to the MECH-090 admission
        # conjunction. See ree_core/policy/commit_maintenance_release.py.
        use_maintenance_release: bool = False,
        maintenance_release_score_margin_floor: float = 0.05,
        maintenance_release_score_margin_reengage: float = 0.10,
        maintenance_release_nav_floor: float = 0.3,
        maintenance_release_nav_reengage: float = 0.5,
        maintenance_release_accumulation_rate: float = 0.2,
        maintenance_release_leak_rate: float = 0.1,
        maintenance_release_bound: float = 1.0,
        maintenance_release_pressure_cap: float = 1.5,
        # Commit/release-DURATION lever: graded natural-commit-occupancy release
        # (rung-6 of f_dominance_conversion_ceiling; duration face, parallel to
        # MECH-448). All defaults no-op (bit-identical when the master is False).
        use_natural_commit_urgency_release: bool = False,
        natural_commit_release_urgency_mode: bool = True,
        natural_commit_release_action_extent_mode: bool = True,
        natural_commit_urgency_rate: float = 0.01,
        natural_commit_urgency_release_bound: float = 1.0,
        natural_commit_urgency_cap: float = 1.5,
        natural_commit_gap_entry_sensitivity: float = 1.0,
        natural_commit_urgency_onset_ticks: int = 0,
        # SD-033e frontopolar-analog DE-COMMIT lever (V3-narrow MECH-264). No-op
        # default (bit-identical when use_frontopolar_decommit is False).
        use_frontopolar_decommit: bool = False,
        frontopolar_gain: float = 0.0,
        use_natural_commit_latch_hold: bool = False,
        natural_commit_latch_hold_max_ticks: int = 0,
        closure_exclusive_decommit_eval: bool = False,
        # Closure-plane commit-ENTRY primitive (rung-6 amend; commitment_closure:GAP-4).
        use_closure_commit_entry: bool = False,
        closure_commit_entry_rule_norm_floor: float = 0.01,
        # Closure-plane commit-ENTRY TRAJECTORY extension (C-STEP; requires the bool flag).
        use_closure_commit_entry_trajectory: bool = False,
        # ARC-108 JOB-2 control-plane DRIVER pair (rho_t maintenance ramp +
        # habenula negative-delta_t de-commit); all no-op default, bit-identical OFF.
        use_rho_maintenance_ramp: bool = False,
        rho_hold_floor: float = 0.05,
        rho_release_margin: float = 0.5,
        rho_onset_grace_ticks: int = 3,
        use_habenula_decommit: bool = False,
        habenula_decommit_delta_threshold: float = 0.0,
        # MECH-276: scientist-agent counterfactual-backed attribution feedstock
        # (waking-phase mechanism feeding the MECH-275 sleep aggregator). All
        # no-op default; bit-identical when use_scientist_attribution=False.
        # See ree_core/attribution/scientist_attribution_buffer.py.
        use_scientist_attribution: bool = False,
        scientist_attribution_cf_margin: float = 0.05,
        scientist_attribution_only_counterfactual_backed: bool = True,
        scientist_attribution_ema_alpha: float = 0.3,
        scientist_attribution_decay: float = 1.0,
        # MECH-353: blocked-agency / control-failure affect stream (z_block).
        # Pure-arithmetic regulator on the SD-029 action-outcome comparator;
        # all defaults no-op (bit-identical when use_blocked_agency=False).
        # See ree_core/affect/blocked_agency.py.
        use_blocked_agency: bool = False,
        blocked_agency_accumulation_rate: float = 0.2,
        blocked_agency_leak_rate: float = 0.1,
        blocked_agency_outcome_mismatch_floor: float = 0.1,
        blocked_agency_attribution_motor_floor: float = 0.5,
        blocked_agency_capacity_collapse_weight: float = 1.0,
        blocked_agency_require_goal_active: bool = True,
        blocked_agency_z_block_cap: float = 1.5,
        blocked_agency_assert_action_weight: float = 0.1,
        blocked_agency_assert_passive_weight: float = 0.1,
        blocked_agency_assert_alt_action_weight: float = 0.1,
        blocked_agency_assert_bias_scale: float = 0.1,
        blocked_agency_decommit_bound: float = 1.0,
        blocked_agency_decommit_consecutive_ticks: int = 5,
        blocked_agency_decommit_arc016_precision_max: float = 0.0,
        blocked_agency_predicted_effect_floor: float = 0.05,
        blocked_agency_noop_class: int = 0,
        # MECH-219 (SD-019b): affective-harm hysteretic integrator. All defaults
        # no-op (bit-identical when use_harm_suffering_accumulator=False; also
        # inert under the default escapability_mode=constant=1.0).
        use_harm_suffering_accumulator: bool = False,
        harm_suffering_alpha_rise: float = 0.2,
        harm_suffering_alpha_fall: float = 0.01,
        harm_suffering_escapability_mode: str = "constant",
        harm_suffering_escapability_constant: float = 1.0,
        harm_suffering_external_escapability: float = 1.0,
        harm_suffering_s_cap: float = 2.0,
        harm_suffering_body_damage_weight: float = 0.0,
        harm_suffering_pe_gain: float = 0.0,
        harm_suffering_use_bistable_latch: bool = False,
        harm_suffering_theta_on: float = 0.5,
        harm_suffering_theta_off: float = 0.3,
        harm_suffering_redirect_aic: bool = False,
        harm_suffering_redirect_pag: bool = False,
        harm_suffering_redirect_mech091: bool = False,
        harm_suffering_redirect_dacc: bool = False,
        harm_suffering_redirect_pacc: bool = False,
        # V3-EXQ-563: hard-inject per-class score bias after all naturalistic
        # signal generation. None = disabled (standard behaviour).
        forced_score_bias_per_class: Optional[List[float]] = None,
        # MECH-319 (arc_062 GAP-K): simulation_mode_rule_write_gate
        # (substrate-level instantiation of MECH-094 at the rule-
        # arbitration layer). admit_writes is the V3-EXQ-543c falsifier.
        use_simulation_mode_rule_gate: bool = False,
        simulation_mode_rule_gate_admit_writes: bool = False,
        # SD-034: governance.closure_operator (five-part "done" token)
        use_closure_operator: bool = False,
        closure_rule_delta_threshold: float = 0.001,
        closure_stable_ticks: int = 3,
        closure_require_beta_elevated: bool = True,
        closure_min_sd033a_gate: float = 0.5,
        closure_nogo_injection_count: int = 3,
        closure_residue_discharge_factor: float = 0.5,
        closure_residue_discharge_radius: float = 1.5,
        closure_signal_value: float = 1.0,
        closure_reset_pe_ema: bool = True,
        closure_pe_cap_after: Optional[float] = None,
        closure_reset_outcome_history: bool = True,
        closure_signal_affinity_internal_planning: float = 0.5,
        # SD-034 commitment-closure-control-plane (2026-06-12); both no-op default.
        use_closure_env_completion_hook: bool = False,
        closure_decommit_hold_ticks: int = 0,
        closure_decommit_hold_scale_with_run: float = 0.0,
        closure_decommit_hold_max_ticks: int = 0,
        # SD-034 beta-engagement amend (2026-06-17); no-op default.
        use_closure_commit_beta_coupling: bool = False,
        # SD-035: amygdala analogue (BLAAnalog + CeAAnalog peer modules)
        use_amygdala_analog: bool = False,
        use_bla_analog: bool = True,
        use_cea_analog: bool = True,
        # BLAAnalog (MECH-074a/b/d)
        bla_encoding_gain_max: float = 2.5,
        bla_encoding_gain_floor: float = 1.0,
        bla_arousal_threshold_on: float = 0.4,
        bla_arousal_peak: float = 0.7,
        bla_window_steps: int = 18000,
        bla_window_half_life_steps: int = 3600,
        bla_retrieval_bias_alpha: float = 0.6,
        bla_retrieval_bias_compensation: float = 0.0,
        bla_retrieval_tag_at_encoding: bool = True,
        bla_remap_pe_sigma_threshold: float = 1.0,
        bla_remap_pe_ema_alpha: float = 0.02,
        bla_remap_pe_std_init: float = 0.1,
        bla_remap_code_fraction: float = 0.33,
        bla_remap_requires_attribution: bool = True,
        bla_context_remap_blend: float = 0.5,
        # CeAAnalog (MECH-046 / MECH-074c)
        cea_fast_route_threshold: float = 0.5,
        cea_fast_route_input_is_lowfreq: bool = True,
        cea_mode_prior_log_odds_max: float = 0.8,
        cea_mode_prior_gain: float = 1.0,
        cea_pre_softmax_additive: bool = True,
        cea_fast_prime_amplitude: float = 0.6,
        cea_fast_prime_decay_tau_steps: int = 4,
        cea_fast_prime_override_window_steps: int = 8,
        cea_cortical_confirmation_weight: float = 1.0,
        # SD-036: GABAergic cross-stream decay regulator + MECH-279 freeze gate
        use_gabaergic_decay: bool = False,
        gaba_tone: float = 1.0,
        gaba_tone_min: float = 0.0,
        gaba_tone_max: float = 2.0,
        gaba_tau_z_harm_s: float = 0.05,
        gaba_tau_z_harm_a: float = 0.02,
        gaba_tau_z_beta: float = 0.03,
        gaba_decay_z_harm_s: bool = True,
        gaba_decay_z_harm_a: bool = True,
        gaba_decay_z_beta: bool = True,
        gaba_input_threshold_z_harm_s: float = 0.0,
        gaba_input_threshold_z_harm_a: float = 0.0,
        gaba_input_threshold_z_beta: float = 0.0,
        use_pag_freeze_gate: bool = False,
        pag_theta_freeze: float = 2.0,
        pag_duration_input_threshold: float = 0.4,
        pag_min_freeze_duration: int = 0,
        pag_max_freeze_duration: int = 0,
        pag_freeze_noop_action_class: int = 0,
        # SD-037: Broadcast Override Regulator (orexin-analog)
        use_broadcast_override: bool = False,
        override_recruitment_threshold: float = 0.5,
        override_alpha_pag: float = 0.5,
        override_salience_reweight_alpha: float = 0.3,
        override_drive_weight: float = 1.0,
        override_harm_weight: float = 1.0,
        override_sustained_threat_window: int = 12,
        override_sustained_threat_threshold: float = 0.4,
        override_decay_rate: float = 0.05,
        override_goal_seeding_gain: float = 2.0,
        # SD-037 consumer-cascade (MECH-281 motor-coupling axis, 2026-05-30)
        override_pfc_eta_gain: float = 0.0,
        override_bla_encoding_gain: float = 0.0,
        override_cea_amplitude_gain: float = 0.0,
        override_beta_interrupt_gain: float = 0.0,
        # MECH-282: LPB interoceptive harm routing
        use_lpb_interoceptive_routing: bool = False,
        lpb_intero_z_dim: int = 16,
        lpb_drive_weight: float = 1.0,
        lpb_resource_weight: float = 1.0,
        # MECH-294: multi-content theta-burst packet (sibling to MECH-089)
        use_multi_content_theta_packet: bool = False,
        theta_packet_binding_mode: str = "joint",
        theta_packet_snapshot_refresh_threshold: float = 0.5,
        theta_packet_hold_threshold: float = 0.4,
        theta_packet_compose_into_e3_bias: bool = False,
        theta_packet_bias_scale: float = 0.1,
        theta_packet_compose_use_joint_coherence: bool = True,
        theta_packet_compose_per_candidate_coherence: bool = False,
        theta_packet_coherence_hold_weight: float = 0.5,
        # MECH-269 / MECH-287 / MECH-288: V_s invalidation runtime (Phase 1 + 2)
        use_per_stream_vs: bool = False,
        use_event_segmenter: bool = False,
        use_invalidation_trigger: bool = False,
        use_anchor_sets: bool = False,
        use_per_region_vs: bool = False,
        use_staleness_accumulator: bool = False,
        use_mech284_hysteresis: bool = False,
        use_vs_commit_release: bool = False,
        # SD-039: dual-trace anchor goal-snapshot payload (substrate flag
        # carried on AnchorSetConfig). Default False = bit-identical to
        # pre-SD-039; ON enables module-level write-site population in
        # REEAgent.sense() / HippocampalModule.build_goal_payload.
        use_sd039_anchor_payload: bool = False,
        # MECH-269b: V_s rollout gating on E1/E2 forward predictions
        use_vs_rollout_gating: bool = False,
        vs_gate_snapshot_refresh_threshold: float = 0.5,
        vs_gate_e1_threshold: float = 0.4,
        vs_gate_e2_threshold: float = 0.4,
        vs_gate_streams: tuple = (
            "z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta",
        ),
        vs_gate_unknown_stream_passes: bool = True,
        use_vs_gate_staleness_lookup: bool = False,
        # MECH-292: ranked ghost-goal bank (derived view over SD-039
        # dual-trace anchor pool). Requires use_anchor_sets=True AND
        # use_sd039_anchor_payload=True at agent build time.
        use_mech292_ghost_bank: bool = False,
        mech292_wanting_weight: float = 1.0,
        mech292_goal_match_weight: float = 1.0,
        mech292_staleness_weight: float = 0.5,
        mech292_recoverability_weight: float = 0.5,
        mech292_goal_match_floor: float = 0.05,
        mech292_top_k: Optional[int] = 32,
        # MECH-293: waking ghost-goal probe search (consumer of MECH-292
        # bank). Requires use_mech292_ghost_bank=True at agent build time.
        use_mech293_ghost_probes: bool = False,
        mech293_ghost_fraction: float = 0.2,
        mech293_min_ghost_candidates: int = 1,
        mech293_max_ghost_candidates: int = 8,
        mech293_replace_lowest_ranked: bool = True,
        # V3-EXQ-553 proposer-fix substrate: orthogonal CEM-candidate seeding.
        # When True, the CEM inner-loop noise is replaced with an orthogonal
        # basis (QR-decomposed) so that the n candidates per iteration are
        # maximally distinct at the proposal level. Backward compatible:
        # default False; bit-identical to legacy iid Gaussian when off.
        use_orthogonal_cem_seeding: bool = False,
        # V3-EXQ-563 follow-up: diagnostic candidate support scaffold.
        # When True, HippocampalModule ensures one one-hot first-action
        # candidate per action class reaches E3. Default False.
        use_action_class_scaffold_candidates: bool = False,
        # Support-preserving CEM (ARC-065). MAIN-PATH DEFAULT 2026-05-17
        # (SP-CEM landing, V3-EXQ-567 ARM_1): True / True / 0.2. Bit-identical
        # legacy opt-out: pass use_support_preserving_cem=False,
        # support_preserving_stratified_elites=False,
        # support_preserving_ao_std_floor=0.0. These from_dims defaults must
        # match the HippocampalConfig dataclass defaults (assigned unconditionally
        # below), otherwise from_dims-built agents silently revert to legacy CEM.
        use_support_preserving_cem: bool = True,
        support_preserving_min_first_action_classes: int = 2,
        support_preserving_stratified_elites: bool = True,
        support_preserving_per_class_quota: int = 0,
        support_preserving_ao_std_floor: float = 0.2,
        # V3-EXQ-563c: score/bias scale normalisation
        normalize_score_bias_to_e3_range: bool = False,
        use_modulatory_selection_authority: bool = False,
        modulatory_authority_gain: float = 0.5,
        modulatory_authority_min_range_floor: float = 1e-6,
        # ARC-108 JOB-1 step-1: learned per-channel selection-gating (no-op default)
        use_learned_channel_gating: bool = False,
        learned_channel_gating_eta: float = 0.01,
        learned_channel_gating_elig_decay: float = 0.9,
        learned_channel_value_baseline_beta: float = 0.05,
        learned_channel_asym_potentiation: float = 1.0,
        learned_channel_asym_depression: float = 0.5,
        learned_channel_rpe_mode: Literal["signed", "unsigned"] = "signed",
        # MECH-451: finer-channel-granularity selection-gating (no-op default)
        use_finer_channel_gating: bool = False,
        # ARC-108 JOB-1 step-2 / MECH-450 recurrent-settling step (learned W_lat)
        use_learned_settling_step: bool = False,
        learned_settling_rounds: int = 3,
        learned_settling_temperature: float = 1.0,
        learned_settling_eta: float = 0.01,
        learned_settling_elig_decay: float = 0.9,
        learned_settling_n_action_classes: int = 8,
        # MECH-140 x MECH-450 disinhibitory soft-competitive settling (parameter-free,
        # no-op default; bit-identical OFF, and gain 0.0 -> exact no-op even when ON)
        use_soft_competitive_settling: bool = False,
        soft_competitive_settling_rounds: int = 3,
        soft_competitive_settling_gain: float = 0.0,
        soft_competitive_settling_temperature: float = 1.0,
        soft_competitive_settling_cross_class: float = 0.25,
        # ARC-110 parallel segregated loops + S2 in-layer null + ARC-109 D1/D2
        # split + MECH-452 loop-local traces (all no-op default; bit-identical OFF)
        use_loop_segregation: bool = False,
        loop_segregation_channel_map: Optional[Dict[str, str]] = None,
        loop_segregation_default_loop: str = "associative",
        loop_segregation_spiral_gain_assoc: float = 1.0,
        loop_segregation_spiral_gain_limbic: float = 1.0,
        loop_segregation_motor_authority: float = 1.0,
        loop_segregation_normalize: str = "zscore",
        loop_segregation_noise_on: bool = False,
        loop_segregation_noise_alpha: float = 1.0,
        use_named_channel_routing: bool = False,
        use_d1_d2_population_split: bool = False,
        d1_da_gain: float = 1.0,
        d2_da_gain: float = 1.0,
        use_loop_local_eligibility_traces: bool = False,
        # ARC-108 x ARC-110 learned (dopamine-gated) cross-loop arbitration
        # (no-op default; bit-identical OFF; requires use_loop_segregation on):
        use_learned_cross_loop_arbitration: bool = False,
        learned_cross_loop_eta: float = 0.01,
        # ARC-110 ascending-spiral gain (V3-EXQ-709/710 loop-effective-weight repair):
        use_ascending_spiral_gain: bool = False,
        loop_segregation_ascending_spiral_gain: float = 1.0,
        loop_segregation_ascending_plasticity_gain: float = 1.0,
        use_ascending_parity_controller: bool = False,
        loop_segregation_parity_forward_gain: float = 1.0,
        loop_segregation_parity_ceiling_ratio: float = 0.0,
        loop_segregation_parity_plasticity_gain: float = 1.0,
        loop_segregation_m_cross_clamp: float = 0.0,
        # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a):
        use_modulatory_channel_routing: bool = False,
        modulatory_channel_route_min_range_floor: float = 1e-6,
        modulatory_channel_route_source: str = "none",
        modulatory_channel_route_weight: float = 1.0,
        # modulatory-bias-selection-authority AMEND (CONVERSION, 569g/682, 2026-06-15):
        modulatory_authority_normalize_basis: str = "range",
        use_modulatory_shortlist_then_modulate: bool = False,
        modulatory_shortlist_margin: float = 0.25,
        # modulatory-bias-selection-authority AMEND (TOP-K shortlist, 569h, 2026-06-16):
        modulatory_shortlist_mode: str = "margin",
        modulatory_shortlist_k: int = 3,
        # CONVERSION-CEILING / F-dominance conflict-grade amend (MECH-439, 2026-06-18):
        modulatory_shortlist_conflict_graded: bool = False,
        modulatory_shortlist_k_max: int = 6,
        use_gap_scaled_commit_temperature: bool = False,
        gap_scaled_commit_entropy_alpha: float = 1.0,
        gap_scaled_commit_harm_floor: float = 0.25,
        # MECH-448 / ARC-107 -- rank-preserving F->eligibility demotion (LEAD lever,
        # 2026-06-20). No-op default; bit-identical OFF.
        use_f_eligibility_demotion: bool = False,
        f_eligibility_envelope_floor: float = 0.30,
        f_eligibility_dn_sigma: float = 0.0,
        # Channel-adaptive envelope amend (2026-06-21): mean-relative floor so the
        # demotion auto-calibrates per channel. No-op default; bit-identical OFF.
        use_f_eligibility_adaptive_floor: bool = False,
        f_eligibility_adaptive_mean_factor: float = 1.0,
        # MECH-449 / ARC-107 Go/No-Go eligibility constitution (2026-06-21).
        # No-op default; bit-identical OFF. PROMOTES NOTHING.
        use_go_nogo_constitution: bool = False,
        gng_safety_floor: float = 0.5,
        gng_staleness_floor: float = 0.5,
        gng_perseveration_floor: float = 0.5,
        gng_viability_floor: float = 0.1,
        gng_go_threshold: float = 0.5,
        gng_go_max_promote: int = 2,
        gng_protect_min_eligible: int = 1,
        # DR-12 (self_model_v4:SELF-4, FIRST V4 substrate build, 2026-06-17):
        # E2 forward-PE -> E3 trajectory-scoring confidence down-weight. No-op default.
        use_pe_confidence_weighting: bool = False,
        pe_confidence_weight: float = 0.0,
        pe_confidence_mode: str = "linear",
        pe_confidence_scale: float = 1.0,
        # DR-10 (self_model_v4:SELF-3): z_self enters E3 viability scoring. No-op default.
        use_self_viability_weighting: bool = False,
        self_viability_weight: float = 0.0,
        self_viability_mode: str = "linear",
        self_viability_scale: float = 1.0,
        # MECH-440 NoisyNet propagating selection-head weight noise (2026-06-27).
        # No-op default; bit-identical OFF. PROMOTES NOTHING.
        use_noisy_selection_head: bool = False,
        noisy_selection_sigma_init: float = 0.0,
        noisy_selection_weight: float = 1.0,
        noisy_selection_anneal: bool = True,
        noisy_selection_anneal_floor: float = 0.1,
        noisy_selection_anneal_ema_alpha: float = 0.01,
        # MECH-441 model-disagreement directed curiosity (2026-06-27). E3-side
        # consumption levers; the ensemble itself is a LatentStackConfig field.
        # No-op default; bit-identical OFF. PROMOTES NOTHING.
        use_model_disagreement_curiosity: bool = False,
        model_disagreement_weight: float = 0.0,
        model_disagreement_mode: str = "linear",
        model_disagreement_scale: float = 1.0,
        # MECH-441 disagreement ensemble (LatentStackConfig). No-op default.
        n_disagreement_heads: int = 0,
        disagreement_bootstrap_mask_prob: float = 0.0,
        disagreement_learning_rate: float = 3e-4,
        # ControlVector logging (rec-B four-signal adjudication 2026-06-07):
        # read-only default-OFF telemetry; bit-identical when False.
        use_control_vector_logging: bool = False,
        # SD-055: differentiable CEM selection approximation
        use_differentiable_cem: bool = False,
        differentiable_cem_temperature: float = 1.0,
        # MECH-290: backward trajectory credit sweep
        use_backward_credit_sweep: bool = False,
        backward_sweep_gamma: float = 0.9,
        backward_sweep_min_quality: float = 0.6,
        # Sleep-aggregation cluster GAP-3 unified master flag (resolves the
        # eight Phase A-E sub-flags True via enable_sleep_aggregation_cluster()
        # at the end of from_dims). Default False: bit-identical pre-GAP-3.
        use_sleep_aggregation_cluster: bool = False,
        # Sleep-aggregation cluster Phase A master flag
        use_sleep_loop: bool = False,
        sleep_loop_episodes_K: int = 1,
        sleep_loop_require_passes: bool = True,
        # SD-MEL-CONSUMER (GAP-5b): adaptive sleep-cadence MEL consumer
        use_mel_consumer: bool = False,
        mel_gain: float = 1.0,
        mel_reference: float = 0.0,
        mel_reference_mode: str = "fixed",
        mel_ema_alpha: float = 0.1,
        mel_duration_factor_min: float = 0.5,
        mel_duration_factor_max: float = 3.0,
        mel_relative_floor: float = 1e-6,
        mel_scale_sws: bool = True,
        mel_scale_rem: bool = True,
        use_mel_entry: bool = False,
        mel_entry_threshold: float = 0.0,
        # Phase B: MECH-285 SleepReplaySampler
        use_mech285_sampler: bool = False,
        mech285_draws_per_cycle: int = 50,
        mech285_temperature: float = 1.0,
        mech285_allow_uniform_fallback: bool = True,
        # Phase C: MECH-272 RoutingGate
        use_mech272_routing: bool = False,
        mech272_waking_anchor_weight: float = 1.0,
        mech272_waking_probe_weight: float = 0.0,
        mech272_sws_anchor_weight: float = 0.6,
        mech272_sws_probe_weight: float = 0.4,
        mech272_rem_anchor_weight: float = 0.2,
        mech272_rem_probe_weight: float = 0.8,
        use_mech272_routing_consumer: bool = False,
        # Phase D: MECH-275 Bayesian aggregator
        use_mech275_aggregator: bool = False,
        mech275_domains: tuple = ("place",),
        mech275_prior_mean: float = 0.0,
        mech275_prior_variance: float = 1.0,
        mech275_likelihood_variance: float = 1.0,
        mech275_decay_factor: float = 1.0,
        mech275_probe_gain: float = 1.0,
        use_mech273_self_model: bool = False,
        # MECH-204 Option A: precision recalibration consumer in WRITEBACK
        use_rem_precision_recalibration: bool = False,
        # Default 0.25 (was 0.1 pre-2026-05-09); see field comment in REEConfig
        # dataclass for V3-EXQ-541c rationale + the biologically-defensible
        # band {0.05, 0.10, 0.25} per Q-042 Option A verdict.
        rem_precision_recalibration_step: float = 0.25,
        mech273_offline_lr_scale: float = 0.1,
        mech273_offline_n_steps: int = 100,
        mech273_partial_decay_factor: float = 0.5,
        # MECH-286: override-gated sleep onset
        use_mech286_sleep_onset_gate: bool = False,
        mech286_theta_sleep_permit: float = 0.5,
        mech286_theta_sleep_recruit: float = 0.3,
        mech286_threat_tonic_threshold: float = 0.4,
        # MECH-295: drive -> liking-stream -> approach_cue bridge
        use_mech295_liking_bridge: bool = False,
        mech295_drive_to_liking_gain: float = 1.0,
        mech295_liking_to_approach_cue_gain: float = 0.5,
        # Lowered 2026-05-12 from 0.1 -> 0.01 (V3-EXQ-540c finding); see
        # REEConfig.mech295_min_drive_to_fire field comment.
        mech295_min_drive_to_fire: float = 0.01,
        mech295_min_z_goal_norm_to_fire: float = 0.05,
        # MECH-302: suffering-derivative comparator substrate
        use_suffering_derivative_comparator: bool = False,
        suffering_window_length: int = 5,
        suffering_drop_threshold: float = 0.10,
        suffering_min_initial_norm: float = 0.05,
        relief_completion_weight: float = 1.0,
        use_conditioned_safety_store: bool = False,
        safety_store_ema_alpha: float = 0.1,
        safety_store_decay_rate: float = 0.001,
        safety_store_min_norm: float = 0.1,
        safety_store_threshold: float = 0.5,
        safety_store_commitment_weight: float = 1.0,
        # MECH-303: contextual passive safety terrain
        use_contextual_safety_terrain: bool = False,
        contextual_safety_accum_weight: float = 0.01,
        contextual_safety_harm_threshold: float = 0.05,
        contextual_safety_release_threshold: float = 1.0,
        # MECH-108: BreathOscillator -- periodic uncommitted windows.
        # breath_period=0 disables (legacy default). Set 50 to enable.
        # Biological basis: exhalation-phase respiratory coupling cyclically
        # modulates hippocampal/PFC gain (Zelano 2016, Karalis & Bhatt 2020).
        breath_period: int = 50,
        breath_sweep_amplitude: float = 0.25,
        breath_sweep_duration: int = 5,
        # SD-049 Phase 3: SD-032 consumer cascade reading per_axis_drive.
        # Master flag + 7 per-consumer combiner knobs. All default to no-op
        # (bit-identical OFF). See REEConfig.use_sd049_per_axis_consumer_cascade
        # field comment for full design rationale.
        use_sd049_per_axis_consumer_cascade: bool = False,
        sd049_aic_per_axis_combiner: str = "max",
        sd049_pcc_per_axis_combiner: str = "mean",
        sd049_pacc_per_axis_combiner: str = "sum",
        sd049_dacc_per_axis_combiner: str = "max",
        sd049_salience_per_axis_combiner: str = "max",
        sd049_override_per_axis_combiner: str = "max",
        sd049_mech295_per_axis_combiner: str = "max",
        # Goal-stream convenience bundle. Kept near **kwargs to preserve the
        # long-standing positional order of from_dims() arguments.
        goal_stream_enabled: bool = False,
        **kwargs,
    ) -> "REEConfig":
        """Create config from basic dimension specifications."""
        # SD-016 write-path mode validation (defensive; no silent typo->'off' fallback)
        _valid_sd016_modes = ("off", "train_only", "sense_only", "both")
        if sd016_writepath_mode not in _valid_sd016_modes:
            raise ValueError(
                f"sd016_writepath_mode must be one of {_valid_sd016_modes}; "
                f"got {sd016_writepath_mode!r}"
            )
        config = cls()

        # Observation dims
        config.latent.body_obs_dim = body_obs_dim
        config.latent.world_obs_dim = world_obs_dim
        config.latent.observation_dim = body_obs_dim + world_obs_dim

        # SD-005 split dims
        config.latent.self_dim = self_dim
        config.latent.world_dim = world_dim
        config.latent.harm_dim = harm_dim  # MECH-099: 0 = lateral head disabled

        # SD-008: temporal EMA alphas
        config.latent.alpha_world = alpha_world
        config.latent.alpha_self = alpha_self

        # SD-007: reafference correction
        config.latent.reafference_action_dim = reafference_action_dim

        # SD-009: event contrastive classifier
        config.latent.use_event_classifier = use_event_classifier

        # SD-018: resource proximity supervision
        config.latent.use_resource_proximity_head = use_resource_proximity_head
        config.latent.resource_proximity_weight = resource_proximity_weight

        # SD-010: dedicated harm stream
        config.latent.use_harm_stream = use_harm_stream
        config.latent.harm_obs_dim = harm_obs_dim
        config.latent.z_harm_dim = z_harm_dim

        # SD-011: affective-motivational harm stream
        config.latent.use_affective_harm_stream = use_affective_harm_stream
        config.latent.harm_obs_a_dim = harm_obs_a_dim
        config.latent.z_harm_a_dim = z_harm_a_dim
        config.latent.harm_history_len = harm_history_len
        config.latent.z_harm_a_aux_loss_weight = z_harm_a_aux_loss_weight

        # E1
        config.e1.self_dim = self_dim
        config.e1.world_dim = world_dim
        config.e1.latent_dim = self_dim + world_dim
        config.e1.sd016_enabled = sd016_enabled
        config.e1.sd016_writepath_mode = sd016_writepath_mode
        config.e1.sd016_temperature_learnable = sd016_temperature_learnable
        config.e1.sd016_cue_slot_tagger = sd016_cue_slot_tagger
        config.e1.sd016_cue_slot_tagger_hidden = sd016_cue_slot_tagger_hidden
        config.e1.sd016_cue_slot_tagger_temperature = sd016_cue_slot_tagger_temperature
        config.e1.action_object_dim = action_object_dim
        config.e1.schema_wanting_enabled = schema_wanting_enabled

        # E2
        config.e2.self_dim = self_dim
        config.e2.world_dim = world_dim
        config.e2.action_dim = action_dim
        config.e2.action_object_dim = action_object_dim

        # E3
        config.e3.world_dim = world_dim
        config.e3.latent_dim = config.latent.beta_dim  # E3 also uses beta for scoring
        config.e3.benefit_eval_enabled = benefit_eval_enabled
        config.e3.benefit_weight = benefit_weight
        config.e3.novelty_bonus_weight = novelty_bonus_weight
        config.e3.self_maintenance_weight = self_maintenance_weight
        config.e3.self_maintenance_d_eff_target = self_maintenance_d_eff_target
        config.e3.urgency_weight = urgency_weight
        config.e3.urgency_max = urgency_max
        config.e3.affective_harm_scale = affective_harm_scale
        config.e3.goal_weight = goal_weight

        # Hippocampal
        config.hippocampal.world_dim = world_dim
        config.hippocampal.action_dim = action_dim
        config.hippocampal.action_object_dim = action_object_dim
        config.hippocampal.horizon = config.e2.rollout_horizon
        config.hippocampal.wanting_weight = wanting_weight

        # Residue
        config.residue.world_dim = world_dim

        # Environment
        config.environment.body_obs_dim = body_obs_dim
        config.environment.world_obs_dim = world_obs_dim

        # GoalConfig fields -- wire goal_dim to world_dim by default
        goal_fields = {
            "z_goal_enabled", "alpha_goal", "decay_goal",
            "benefit_threshold", "goal_weight", "e1_goal_conditioned",
            "drive_weight",  # SD-012
            "drive_ema_alpha",  # SD-012 GAP-3 Option 1 sustained-drive EMA
            "drive_floor",  # SD-012 GAP-3 Option 2 insatiability floor
            "valence_wanting_floor",  # MECH-186
            "z_goal_seeding_gain",  # MECH-187
            "z_goal_inject",  # MECH-188
            "use_incentive_token_bank",  # SD-057 L2-L3-L4
            "incentive_decay",  # SD-057 L3
            "incentive_value_alpha",  # SD-057 L3
            "incentive_drive_kappa_weight",  # SD-057 L3
            "incentive_drive_kappa_scale",  # SD-049-PHASE-2 drive-coupling amend
            "incentive_use_per_axis_drive",  # SD-057 L3
            "use_cue_recall",  # SD-057 L6
            "cue_recall_gain",  # SD-057 L6
            "cue_recall_min_proximity",  # SD-057 L6
            "use_super_ordinal_goal_anchors",  # MECH-189
            "super_ordinal_n_slots",  # MECH-189
            "super_ordinal_salience_threshold",  # MECH-189
            "super_ordinal_complexity_mode",  # MECH-189
            "super_ordinal_complexity_threshold",  # MECH-189
            "super_ordinal_merge_similarity",  # MECH-189
            "super_ordinal_write_alpha",  # MECH-189
            "super_ordinal_seed_below_norm",  # MECH-189
            "super_ordinal_seed_match_threshold",  # MECH-189
            "super_ordinal_seed_strength",  # MECH-189
        }
        local_goal_vals = {
            "z_goal_enabled": z_goal_enabled,
            "alpha_goal": alpha_goal,
            "decay_goal": decay_goal,
            "benefit_threshold": benefit_threshold,
            "goal_weight": goal_weight,
            "e1_goal_conditioned": e1_goal_conditioned,
            "drive_weight": drive_weight,  # SD-012
            "drive_ema_alpha": drive_ema_alpha,  # SD-012 GAP-3 Option 1
            "drive_floor": drive_floor,  # SD-012 GAP-3 Option 2
            "valence_wanting_floor": valence_wanting_floor,  # MECH-186
            "z_goal_seeding_gain": z_goal_seeding_gain,  # MECH-187
            "z_goal_inject": z_goal_inject,  # MECH-188
            "use_incentive_token_bank": use_incentive_token_bank,  # SD-057
            "incentive_decay": incentive_decay,  # SD-057 L3
            "incentive_value_alpha": incentive_value_alpha,  # SD-057 L3
            "incentive_drive_kappa_weight": incentive_drive_kappa_weight,  # SD-057 L3
            "incentive_drive_kappa_scale": incentive_drive_kappa_scale,  # SD-049-PHASE-2 drive-coupling amend
            "incentive_use_per_axis_drive": incentive_use_per_axis_drive,  # SD-057 L3
            "use_cue_recall": use_cue_recall,  # SD-057 L6
            "cue_recall_gain": cue_recall_gain,  # SD-057 L6
            "cue_recall_min_proximity": cue_recall_min_proximity,  # SD-057 L6
            "use_super_ordinal_goal_anchors": use_super_ordinal_goal_anchors,  # MECH-189
            "super_ordinal_n_slots": super_ordinal_n_slots,  # MECH-189
            "super_ordinal_salience_threshold": super_ordinal_salience_threshold,  # MECH-189
            "super_ordinal_complexity_mode": super_ordinal_complexity_mode,  # MECH-189
            "super_ordinal_complexity_threshold": super_ordinal_complexity_threshold,  # MECH-189
            "super_ordinal_merge_similarity": super_ordinal_merge_similarity,  # MECH-189
            "super_ordinal_write_alpha": super_ordinal_write_alpha,  # MECH-189
            "super_ordinal_seed_below_norm": super_ordinal_seed_below_norm,  # MECH-189
            "super_ordinal_seed_match_threshold": super_ordinal_seed_match_threshold,  # MECH-189
            "super_ordinal_seed_strength": super_ordinal_seed_strength,  # MECH-189
        }
        for _key in goal_fields:
            if _key in local_goal_vals:
                setattr(config.goal, _key, local_goal_vals[_key])
        # Keep goal_dim in sync with world_dim
        if hasattr(config, "latent") and hasattr(config.latent, "world_dim"):
            config.goal.goal_dim = config.latent.world_dim

        # MECH-203/204: serotonin config
        config.serotonin.tonic_5ht_enabled = tonic_5ht_enabled
        # MECH-204 F1: cross-cycle persistent zero-point EMA tracking
        config.serotonin.precision_zero_point_ema_alpha = precision_zero_point_ema_alpha

        # MECH-205: surprise-gated replay
        config.surprise_gated_replay = surprise_gated_replay
        config.pe_ema_alpha = pe_ema_alpha
        config.pe_surprise_threshold = pe_surprise_threshold

        # MECH-294: multi-content theta-burst packet
        config.use_multi_content_theta_packet = use_multi_content_theta_packet
        config.theta_packet_binding_mode = theta_packet_binding_mode
        config.theta_packet_snapshot_refresh_threshold = theta_packet_snapshot_refresh_threshold
        config.theta_packet_hold_threshold = theta_packet_hold_threshold
        config.theta_packet_compose_into_e3_bias = theta_packet_compose_into_e3_bias
        config.theta_packet_bias_scale = theta_packet_bias_scale
        config.theta_packet_compose_use_joint_coherence = theta_packet_compose_use_joint_coherence
        config.theta_packet_compose_per_candidate_coherence = theta_packet_compose_per_candidate_coherence
        config.theta_packet_coherence_hold_weight = theta_packet_coherence_hold_weight

        # MECH-120: SHY normalization
        config.shy_enabled = shy_enabled
        config.shy_decay_rate = shy_decay_rate

        # SD-017: minimal sleep-phase infrastructure
        config.sws_enabled = sws_enabled
        config.sws_consolidation_steps = sws_consolidation_steps
        config.sws_schema_weight = sws_schema_weight
        config.rem_enabled = rem_enabled
        config.rem_attribution_steps = rem_attribution_steps

        # MECH-165: reverse replay diversity scheduler
        config.replay_diversity_enabled = replay_diversity_enabled
        config.reverse_replay_fraction = reverse_replay_fraction
        config.random_replay_fraction = random_replay_fraction
        config.exploration_buffer_len = exploration_buffer_len

        # MECH-216: E1 predictive wanting
        config.schema_wanting_threshold = schema_wanting_threshold
        config.schema_wanting_gain = schema_wanting_gain

        # ARC-033: E2_harm_s forward model
        config.latent.use_e2_harm_s_forward = use_e2_harm_s_forward
        config.latent.use_e2_world_forward = use_e2_world_forward

        # SD-056: E2 action-conditional divergence contrastive substrate.
        # Knobs land on config.e2 (E2Config); helpers live on E2FastPredictor.
        # Default OFF preserves bit-identical existing-experiment behaviour.
        config.e2.e2_action_contrastive_enabled = e2_action_contrastive_enabled
        config.e2.e2_action_contrastive_weight = e2_action_contrastive_weight
        config.e2.e2_action_contrastive_temperature = e2_action_contrastive_temperature
        config.e2.e2_action_contrastive_min_batch_classes = e2_action_contrastive_min_batch_classes

        # SD-056 multi-step rollout stability amend (2026-05-31).
        # Lever (a) multi-step contrastive + lever (b) per-step rollout norm
        # clamp. Both default OFF -> bit-identical to pre-amend SD-056 path.
        config.e2.e2_action_contrastive_multistep_enabled = e2_action_contrastive_multistep_enabled
        config.e2.e2_action_contrastive_horizon = e2_action_contrastive_horizon
        config.e2.e2_action_contrastive_horizon_weights_decay = e2_action_contrastive_horizon_weights_decay
        config.e2.e2_rollout_output_norm_clamp_enabled = e2_rollout_output_norm_clamp_enabled
        config.e2.e2_rollout_output_norm_clamp_ratio = e2_rollout_output_norm_clamp_ratio

        # cross_stream_binding_substrate (2026-07-08): shared-latent-factor
        # binding coupling in E2.rollout_with_world. All no-op default -> OFF is
        # byte-identical (binder submodule constructed only when enabled).
        config.e2.cross_stream_binding_enabled = cross_stream_binding_enabled
        config.e2.cross_stream_binding_dim = cross_stream_binding_dim
        config.e2.cross_stream_binding_strength = cross_stream_binding_strength
        config.e2.cross_stream_binding_theta_period = cross_stream_binding_theta_period
        # learned (plastic) binder (2026-07-09); all no-op default. learned=False
        # keeps the fixed-field path byte-identical.
        config.e2.cross_stream_binding_learned = cross_stream_binding_learned
        config.e2.cross_stream_binding_lr = cross_stream_binding_lr
        config.e2.cross_stream_binding_temperature = cross_stream_binding_temperature
        config.e2.cross_stream_binding_buffer_size = cross_stream_binding_buffer_size
        config.e2.cross_stream_binding_batch = cross_stream_binding_batch
        config.e2.cross_stream_binding_conv_frac = cross_stream_binding_conv_frac

        # SD-022: directional limb damage dim adjustments.
        # When enabled: harm_obs_a_dim is re-sourced from body damage state (7 dims).
        # body_obs_dim is passed by the caller reflecting the actual env dims
        # (CausalGridWorldV2.body_obs_dim returns 17 when limb_damage_enabled=True).
        # No body_obs_dim arithmetic needed here -- the env already reports the correct dim.
        if limb_damage_enabled:
            # Override harm_obs_a_dim to 7 (damage[4] + max_damage + mean_damage + residual_pain)
            config.latent.harm_obs_a_dim = 7

        # SD-014: valence channel writes
        config.valence_harm_enabled = valence_harm_enabled
        config.valence_liking_enabled = valence_liking_enabled
        config.liking_threshold = liking_threshold

        # MECH-258: E2_harm_a forward model (SD-032b prerequisite)
        config.use_e2_harm_a = use_e2_harm_a
        config.e2_harm_a_lr = e2_harm_a_lr

        # ARC-058: shared-trunk harm forward hypothesis
        config.use_shared_harm_trunk = use_shared_harm_trunk

        # SD-032b: dACC/aMCC-analog adaptive control
        config.use_dacc = use_dacc
        config.use_mech_consume = use_mech_consume  # SD-057 L7
        config.dacc_goal_readout_weight = dacc_goal_readout_weight  # SD-057 L7
        config.dacc_weight = dacc_weight
        config.dacc_interaction_weight = dacc_interaction_weight
        config.dacc_foraging_weight = dacc_foraging_weight
        config.dacc_suppression_weight = dacc_suppression_weight
        config.dacc_suppression_memory = dacc_suppression_memory
        config.dacc_precision_scale = dacc_precision_scale
        config.dacc_effort_cost = dacc_effort_cost
        config.dacc_drive_coupling = dacc_drive_coupling
        config.dacc_bias_max_abs = dacc_bias_max_abs

        # MECH-268: dACC conflict saturation (pe_saturated = f_sat(pe_raw, outcome_history))
        config.dacc_saturation_enabled = dacc_saturation_enabled
        config.dacc_saturation_window = dacc_saturation_window
        config.dacc_saturation_strength = dacc_saturation_strength
        config.dacc_saturation_grace = dacc_saturation_grace

        # SD-032a: salience-network coordinator
        config.use_salience_coordinator = use_salience_coordinator
        config.salience_switch_threshold = salience_switch_threshold
        config.salience_stability_scaling = salience_stability_scaling
        config.salience_softmax_temperature = salience_softmax_temperature
        config.salience_external_task_bias = salience_external_task_bias
        config.salience_dacc_pe_weight = salience_dacc_pe_weight
        config.salience_dacc_foraging_weight = salience_dacc_foraging_weight
        config.salience_apply_to_dacc_bias = salience_apply_to_dacc_bias
        # mode-governance-engagement: external_task salience source.
        config.use_external_task_drive = use_external_task_drive
        config.external_task_drive_affinity_weight = external_task_drive_affinity_weight
        config.external_task_drive_salience_weight = external_task_drive_salience_weight
        config.external_task_drive_commit_weight = external_task_drive_commit_weight
        config.external_task_drive_proximity_weight = external_task_drive_proximity_weight
        config.external_task_drive_require_goal_active = external_task_drive_require_goal_active

        # SD-032c: AIC-analog interoceptive-salience / urgency
        config.use_aic_analog = use_aic_analog
        config.aic_baseline_alpha = aic_baseline_alpha
        config.aic_drive_coupling = aic_drive_coupling
        config.aic_urgency_threshold = aic_urgency_threshold
        config.aic_base_attenuation = aic_base_attenuation
        config.aic_drive_protect_weight = aic_drive_protect_weight
        config.aic_extra_weight = aic_extra_weight

        # SD-032d: PCC-analog metastability scalar
        config.use_pcc_analog = use_pcc_analog
        config.pcc_success_alpha = pcc_success_alpha
        config.pcc_success_weight = pcc_success_weight
        config.pcc_fatigue_weight = pcc_fatigue_weight
        config.pcc_offline_recency_window = pcc_offline_recency_window
        config.pcc_offline_weight = pcc_offline_weight
        config.pcc_stability_baseline = pcc_stability_baseline

        # SD-032e: pACC-analog autonomic write-back
        config.use_pacc_analog = use_pacc_analog
        config.pacc_drive_alpha = pacc_drive_alpha
        config.pacc_drive_scale = pacc_drive_scale
        config.pacc_drive_bias_cap = pacc_drive_bias_cap
        config.pacc_z_harm_a_min = pacc_z_harm_a_min
        config.pacc_offline_decay = pacc_offline_decay

        # MECH-095: TPJ comparator
        config.use_tpj_comparator = use_tpj_comparator
        config.tpj_agency_threshold = tpj_agency_threshold

        # SD-033a: Lateral-PFC-analog
        config.use_lateral_pfc_analog = use_lateral_pfc_analog
        config.lateral_pfc_rule_dim = lateral_pfc_rule_dim
        config.lateral_pfc_update_eta = lateral_pfc_update_eta
        config.lateral_pfc_world_pool_weight = lateral_pfc_world_pool_weight
        config.lateral_pfc_bias_scale = lateral_pfc_bias_scale
        config.lateral_pfc_hidden_dim = lateral_pfc_hidden_dim
        config.lateral_pfc_use_discriminator_source = lateral_pfc_use_discriminator_source
        config.lateral_pfc_discriminator_pool_weight = lateral_pfc_discriminator_pool_weight
        config.lateral_pfc_train_rule_bias_head = lateral_pfc_train_rule_bias_head
        # MECH-457 actor-critic action-learning substrate.
        config.use_actor_critic = use_actor_critic
        config.actor_critic_cotrain_encoder = actor_critic_cotrain_encoder
        config.actor_critic_use_sf_critic = actor_critic_use_sf_critic
        config.actor_critic_hidden = actor_critic_hidden
        config.actor_critic_sf_feature_dim = actor_critic_sf_feature_dim

        # ARC-063 v1: distributed CandidateRule field (GAP-B rule-creator)
        config.use_candidate_rule_field = use_candidate_rule_field
        config.crf_n_slots = crf_n_slots
        config.crf_rule_dim = crf_rule_dim
        config.crf_mint_recurrence_threshold = crf_mint_recurrence_threshold
        config.crf_tolerance_floor = crf_tolerance_floor
        config.crf_tolerance_conflict_gain = crf_tolerance_conflict_gain
        config.crf_availability_alpha = crf_availability_alpha
        config.crf_availability_decay = crf_availability_decay
        config.crf_eligibility_window = crf_eligibility_window
        config.crf_context_match_threshold = crf_context_match_threshold
        config.crf_seed_from_arc062 = crf_seed_from_arc062
        config.crf_persist_rules_across_episode_reset = (
            crf_persist_rules_across_episode_reset
        )
        # ARC-063 amend (V3-EXQ-654b): mature-pool gate/credit/retire dynamics
        config.crf_mature_pool_dynamics = crf_mature_pool_dynamics
        config.crf_mature_availability_decay = crf_mature_availability_decay
        config.crf_mature_retire_floor = crf_mature_retire_floor
        config.crf_mature_availability_alpha_negative = (
            crf_mature_availability_alpha_negative
        )
        config.crf_mature_tolerance_floor = crf_mature_tolerance_floor
        config.crf_mature_tolerance_conflict_gain = (
            crf_mature_tolerance_conflict_gain
        )
        config.crf_mature_mint_block_threshold = crf_mature_mint_block_threshold
        config.crf_mature_mint_protection_ticks = (
            crf_mature_mint_protection_ticks
        )
        config.crf_context_from_e2_world_forward = (
            crf_context_from_e2_world_forward
        )
        # crf-availability-maintenance (V3-EXQ-666 successor; ARC-063 amend)
        config.crf_availability_maintenance = crf_availability_maintenance
        config.crf_maintenance_floor = crf_maintenance_floor
        config.crf_maintenance_decay = crf_maintenance_decay
        config.crf_engaged_sustain = crf_engaged_sustain
        config.crf_engaged_sustain_rate = crf_engaged_sustain_rate
        config.crf_mature_context_match_threshold = (
            crf_mature_context_match_threshold
        )
        config.crf_tolerance_conflict_cap = crf_tolerance_conflict_cap
        config.crf_maintenance_couple_to_theta = crf_maintenance_couple_to_theta
        config.crf_maintained_reactivation_threshold = (
            crf_maintained_reactivation_threshold
        )

        # SD-033b: OFC-analog
        config.use_ofc_analog = use_ofc_analog
        config.ofc_state_dim = ofc_state_dim
        config.ofc_update_eta = ofc_update_eta
        config.ofc_outcome_pool_weight = ofc_outcome_pool_weight
        config.ofc_bias_scale = ofc_bias_scale
        config.ofc_hidden_dim = ofc_hidden_dim
        config.ofc_harm_dim = ofc_harm_dim
        config.use_ofc_outcome_oracle = use_ofc_outcome_oracle
        config.ofc_train_state_bias_head = ofc_train_state_bias_head
        config.use_ofc_devaluation_head = use_ofc_devaluation_head
        config.ofc_devaluation_bias_scale = ofc_devaluation_bias_scale
        config.ofc_train_devaluation_head = ofc_train_devaluation_head

        # ARC-062 Phase 1: gated-policy heads + context discriminator
        config.use_gated_policy = use_gated_policy
        config.gated_policy_n_heads = gated_policy_n_heads
        config.gated_policy_disc_hidden = gated_policy_disc_hidden
        config.gated_policy_disc_init_scale = gated_policy_disc_init_scale
        config.gated_policy_head_hidden = gated_policy_head_hidden
        config.gated_policy_bias_scale = gated_policy_bias_scale
        config.gated_policy_head_init_bias_offset = gated_policy_head_init_bias_offset
        config.gated_policy_use_first_action_onehot = gated_policy_use_first_action_onehot

        # INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection
        # crystallization. The master toggle flows to the GatedPolicy /
        # ResidueField builds in REEAgent.__init__; here it also arms the
        # residue EWC config so the anchor/penalty are available when the
        # Phase-3 callback fires. lambda 0.0 keeps the penalty inert even
        # when armed (safe default).
        config.crystallize_at_phase3 = crystallize_at_phase3
        config.gated_policy_crystallize_expansion_hidden = (
            gated_policy_crystallize_expansion_hidden
        )
        config.residue_ewc_lambda = residue_ewc_lambda
        if crystallize_at_phase3:
            config.residue.ewc_enabled = True
            config.residue.ewc_lambda = residue_ewc_lambda

        # MECH-313 (ARC-065): stochastic_noise_floor
        config.use_noise_floor = use_noise_floor
        config.noise_floor_alpha = noise_floor_alpha
        config.noise_floor_min_temperature = noise_floor_min_temperature

        # MECH-314 (ARC-065): structured_curiosity_bonus
        config.use_structured_curiosity = use_structured_curiosity
        config.use_curiosity_novelty = use_curiosity_novelty
        config.use_curiosity_uncertainty = use_curiosity_uncertainty
        config.use_curiosity_learning_progress = use_curiosity_learning_progress
        config.curiosity_novelty_weight = curiosity_novelty_weight
        config.curiosity_uncertainty_weight = curiosity_uncertainty_weight
        config.curiosity_learning_progress_weight = curiosity_learning_progress_weight
        config.curiosity_bias_scale = curiosity_bias_scale
        config.curiosity_lp_ema_alpha = curiosity_lp_ema_alpha
        config.curiosity_lp_window_k = curiosity_lp_window_k
        # MECH-314a Phase 2 (Candidate 5A): novelty-source + augmentation
        config.curiosity_novelty_source = curiosity_novelty_source
        config.curiosity_visitation_buffer_len = curiosity_visitation_buffer_len
        config.curiosity_use_first_action_onehot = curiosity_use_first_action_onehot
        config.curiosity_first_action_augmentation_policy = (
            curiosity_first_action_augmentation_policy
        )
        config.curiosity_min_spread_threshold = curiosity_min_spread_threshold
        config.curiosity_min_spread_consecutive_ticks = (
            curiosity_min_spread_consecutive_ticks
        )
        config.curiosity_candidate_source = curiosity_candidate_source
        config.candidate_summary_source = candidate_summary_source

        # MECH-320 (ARC-066 child): tonic_vigor_coupling_score_bias
        config.use_tonic_vigor = use_tonic_vigor
        config.tonic_vigor_half_life = tonic_vigor_half_life
        config.tonic_vigor_w_action = tonic_vigor_w_action
        config.tonic_vigor_w_passive = tonic_vigor_w_passive
        config.tonic_vigor_bias_scale = tonic_vigor_bias_scale
        config.tonic_vigor_gate_energy_min = tonic_vigor_gate_energy_min
        config.tonic_vigor_gate_drive_max = tonic_vigor_gate_drive_max
        config.tonic_vigor_gate_pe_max = tonic_vigor_gate_pe_max
        config.tonic_vigor_form = tonic_vigor_form
        config.tonic_vigor_noop_class = tonic_vigor_noop_class
        config.tonic_vigor_v_t_floor = tonic_vigor_v_t_floor

        # SD-058 / MECH-357: instrumental-avoidance acquisition
        config.use_instrumental_avoidance = use_instrumental_avoidance
        config.avoidance_learn_rate = avoidance_learn_rate
        config.avoidance_leak_rate = avoidance_leak_rate
        config.avoidance_initial_efficacy = avoidance_initial_efficacy
        config.avoidance_scaffold_floor = avoidance_scaffold_floor
        config.avoidance_threat_floor = avoidance_threat_floor
        config.avoidance_threat_ref = avoidance_threat_ref
        config.avoidance_efficacy_reward_floor = avoidance_efficacy_reward_floor
        config.avoidance_action_bias_gain = avoidance_action_bias_gain
        config.avoidance_bias_scale = avoidance_bias_scale
        config.avoidance_suppression_threshold = avoidance_suppression_threshold
        config.avoidance_noop_class = avoidance_noop_class
        config.use_escape_affordance_bridge = use_escape_affordance_bridge
        config.use_escape_relief_credit = use_escape_relief_credit
        config.use_escape_safety_credit = use_escape_safety_credit
        config.escape_relief_learn_rate = escape_relief_learn_rate
        config.escape_safety_learn_rate = escape_safety_learn_rate
        config.escape_bridge_leak_rate = escape_bridge_leak_rate
        config.escape_relief_reward_floor = escape_relief_reward_floor
        config.escape_threat_floor = escape_threat_floor
        config.escape_threat_ref = escape_threat_ref
        config.escape_approach_gain = escape_approach_gain
        config.escape_bias_scale = escape_bias_scale
        config.escape_noop_class = escape_noop_class
        config.escape_use_trained_safety_signal = escape_use_trained_safety_signal
        config.escape_safety_signal_threshold = escape_safety_signal_threshold
        config.use_object_file_buffer = use_object_file_buffer
        config.obf_max_tokens = obf_max_tokens
        config.obf_continuity_radius = obf_continuity_radius
        config.obf_w_motion = obf_w_motion
        config.obf_w_feat = obf_w_feat
        config.obf_feature_alpha = obf_feature_alpha
        config.obf_persist_ttl = obf_persist_ttl
        config.obf_min_birth_salience = obf_min_birth_salience
        config.obf_use_precision_weighting = obf_use_precision_weighting
        config.use_trainable_escape_affordance_learner = use_trainable_escape_affordance_learner
        config.use_trainable_relief_critic = use_trainable_relief_critic
        config.use_trainable_safety_predictor = use_trainable_safety_predictor
        config.trainable_escape_bias_scale = trainable_escape_bias_scale
        config.trainable_escape_relief_learn_rate = trainable_escape_relief_learn_rate
        config.trainable_escape_safety_learn_rate = trainable_escape_safety_learn_rate
        config.trainable_escape_leak_rate = trainable_escape_leak_rate
        config.trainable_escape_relief_reward_floor = trainable_escape_relief_reward_floor
        config.trainable_escape_relief_target_scale = trainable_escape_relief_target_scale
        config.trainable_escape_threat_floor = trainable_escape_threat_floor
        config.trainable_escape_noop_class = trainable_escape_noop_class
        config.trainable_escape_hidden_dim = trainable_escape_hidden_dim
        config.trainable_escape_action_embedding_dim = trainable_escape_action_embedding_dim
        config.trainable_escape_optimizer_lr = trainable_escape_optimizer_lr
        config.trainable_escape_prediction_floor = trainable_escape_prediction_floor
        config.use_e2_escape_affordance_linker = use_e2_escape_affordance_linker
        config.use_e2_escape_linker_for_relief_safety = use_e2_escape_linker_for_relief_safety
        config.use_e2_escape_linker_e3_bias = use_e2_escape_linker_e3_bias
        config.escape_linker_learn_rate = escape_linker_learn_rate
        config.escape_linker_optimizer_lr = escape_linker_optimizer_lr
        config.escape_linker_leak_rate = escape_linker_leak_rate
        config.escape_linker_hidden_dim = escape_linker_hidden_dim
        config.escape_linker_action_embedding_dim = escape_linker_action_embedding_dim
        config.escape_linker_bias_scale = escape_linker_bias_scale
        config.escape_linker_threat_floor = escape_linker_threat_floor
        config.escape_linker_threat_ref = escape_linker_threat_ref
        config.escape_linker_noop_class = escape_linker_noop_class
        config.escape_linker_relief_reward_floor = escape_linker_relief_reward_floor
        config.escape_linker_harm_delta_scale = escape_linker_harm_delta_scale
        config.escape_linker_prediction_floor = escape_linker_prediction_floor
        config.escape_linker_block_hypothesis_learning = escape_linker_block_hypothesis_learning

        # MECH-341 (ARC-065 Layer-B child): e3_scoring_preserves_trajectory_
        # class_diversity
        config.use_e3_score_diversity = use_e3_score_diversity
        config.use_e3_diversity_entropy_bonus = use_e3_diversity_entropy_bonus
        config.use_e3_diversity_stratified_select = use_e3_diversity_stratified_select
        config.e3_diversity_entropy_lambda = e3_diversity_entropy_lambda
        config.e3_diversity_entropy_bias_scale = e3_diversity_entropy_bias_scale
        config.e3_diversity_stratified_temperature = e3_diversity_stratified_temperature
        config.e3_diversity_stratified_within_class_temperature = (
            e3_diversity_stratified_within_class_temperature
        )
        config.e3_diversity_min_classes_for_stratification = e3_diversity_min_classes_for_stratification

        # MECH-090 R-c conjunction: commit-entry readiness signal
        config.use_mech090_readiness_conjunction = use_mech090_readiness_conjunction
        config.mech090_readiness_floor = mech090_readiness_floor
        config.use_commit_readiness = use_commit_readiness
        config.commit_readiness_window = commit_readiness_window
        config.commit_readiness_ema_alpha = commit_readiness_ema_alpha
        config.commit_readiness_initial = commit_readiness_initial
        # __post_init__ auto-arm: when the conjunction is enabled, the
        # readiness module must be instantiated. from_dims sets fields
        # AFTER cls(), so __post_init__ ran before this assignment --
        # re-apply the OR-only resolver here so factory-built configs
        # behave identically to direct-construction configs.
        if use_mech090_readiness_conjunction:
            config.use_commit_readiness = True

        # SD-061: difficulty-gated proposal-entropy regulator.
        config.use_difficulty_gated_proposal_entropy = (
            use_difficulty_gated_proposal_entropy
        )
        config.stuck_progress_window = stuck_progress_window
        config.stuck_progress_stall_eps = stuck_progress_stall_eps
        config.stuck_score_margin_floor = stuck_score_margin_floor
        config.stuck_committed_diversity_window = stuck_committed_diversity_window
        config.stuck_committed_diversity_floor = stuck_committed_diversity_floor
        config.stuck_choice_difficulty_ref = stuck_choice_difficulty_ref
        config.stuck_goal_salience_floor = stuck_goal_salience_floor
        config.stuck_ema_alpha_rise = stuck_ema_alpha_rise
        config.stuck_ema_alpha_fall = stuck_ema_alpha_fall
        config.stuck_threshold = stuck_threshold
        config.stuck_combine_mode = stuck_combine_mode
        config.dgpe_candidate_widen_max = dgpe_candidate_widen_max
        config.dgpe_temperature_gain_max = dgpe_temperature_gain_max

        # MECH-342: maintenance-time readiness-driven commitment release.
        config.use_maintenance_release = use_maintenance_release
        config.maintenance_release_score_margin_floor = (
            maintenance_release_score_margin_floor
        )
        config.maintenance_release_score_margin_reengage = (
            maintenance_release_score_margin_reengage
        )
        config.maintenance_release_nav_floor = maintenance_release_nav_floor
        config.maintenance_release_nav_reengage = maintenance_release_nav_reengage
        config.maintenance_release_accumulation_rate = (
            maintenance_release_accumulation_rate
        )
        config.maintenance_release_leak_rate = maintenance_release_leak_rate
        config.maintenance_release_bound = maintenance_release_bound
        config.maintenance_release_pressure_cap = maintenance_release_pressure_cap

        # Commit/release-DURATION lever: graded natural-commit-occupancy release.
        config.use_natural_commit_urgency_release = (
            use_natural_commit_urgency_release
        )
        config.natural_commit_release_urgency_mode = (
            natural_commit_release_urgency_mode
        )
        config.natural_commit_release_action_extent_mode = (
            natural_commit_release_action_extent_mode
        )
        config.natural_commit_urgency_rate = natural_commit_urgency_rate
        config.natural_commit_urgency_release_bound = (
            natural_commit_urgency_release_bound
        )
        config.natural_commit_urgency_cap = natural_commit_urgency_cap
        config.natural_commit_gap_entry_sensitivity = (
            natural_commit_gap_entry_sensitivity
        )
        # SD-033e frontopolar-analog DE-COMMIT lever (V3-narrow MECH-264).
        config.use_frontopolar_decommit = use_frontopolar_decommit
        config.frontopolar_gain = frontopolar_gain
        config.use_natural_commit_latch_hold = use_natural_commit_latch_hold
        config.natural_commit_latch_hold_max_ticks = (
            natural_commit_latch_hold_max_ticks
        )
        config.closure_exclusive_decommit_eval = closure_exclusive_decommit_eval
        config.use_closure_commit_entry = use_closure_commit_entry
        config.closure_commit_entry_rule_norm_floor = (
            closure_commit_entry_rule_norm_floor
        )
        config.use_closure_commit_entry_trajectory = (
            use_closure_commit_entry_trajectory
        )
        config.natural_commit_urgency_onset_ticks = (
            natural_commit_urgency_onset_ticks
        )
        # ARC-108 JOB-2 control-plane DRIVER pair (rho_t ramp + habenula de-commit).
        config.use_rho_maintenance_ramp = use_rho_maintenance_ramp
        config.rho_hold_floor = rho_hold_floor
        config.rho_release_margin = rho_release_margin
        config.rho_onset_grace_ticks = rho_onset_grace_ticks
        config.use_habenula_decommit = use_habenula_decommit
        config.habenula_decommit_delta_threshold = (
            habenula_decommit_delta_threshold
        )
        # Mirror onto E3Config so e3_selector.post_action_update (which reads
        # self.config == E3Config) computes the signed RPE delta_t for the habenula.
        config.e3.use_habenula_decommit = use_habenula_decommit
        # MECH-276 scientist-agent attribution feedstock.
        config.use_scientist_attribution = use_scientist_attribution
        config.scientist_attribution_cf_margin = scientist_attribution_cf_margin
        config.scientist_attribution_only_counterfactual_backed = (
            scientist_attribution_only_counterfactual_backed
        )
        config.scientist_attribution_ema_alpha = scientist_attribution_ema_alpha
        config.scientist_attribution_decay = scientist_attribution_decay

        # MECH-353: blocked-agency / control-failure affect stream (z_block).
        config.use_blocked_agency = use_blocked_agency
        config.blocked_agency_accumulation_rate = blocked_agency_accumulation_rate
        config.blocked_agency_leak_rate = blocked_agency_leak_rate
        config.blocked_agency_outcome_mismatch_floor = (
            blocked_agency_outcome_mismatch_floor
        )
        config.blocked_agency_attribution_motor_floor = (
            blocked_agency_attribution_motor_floor
        )
        config.blocked_agency_capacity_collapse_weight = (
            blocked_agency_capacity_collapse_weight
        )
        config.blocked_agency_require_goal_active = blocked_agency_require_goal_active
        config.blocked_agency_z_block_cap = blocked_agency_z_block_cap
        config.blocked_agency_assert_action_weight = blocked_agency_assert_action_weight
        config.blocked_agency_assert_passive_weight = (
            blocked_agency_assert_passive_weight
        )
        config.blocked_agency_assert_alt_action_weight = (
            blocked_agency_assert_alt_action_weight
        )
        config.blocked_agency_assert_bias_scale = blocked_agency_assert_bias_scale
        config.blocked_agency_decommit_bound = blocked_agency_decommit_bound
        config.blocked_agency_decommit_consecutive_ticks = (
            blocked_agency_decommit_consecutive_ticks
        )
        config.blocked_agency_decommit_arc016_precision_max = (
            blocked_agency_decommit_arc016_precision_max
        )
        config.blocked_agency_predicted_effect_floor = (
            blocked_agency_predicted_effect_floor
        )
        config.blocked_agency_noop_class = blocked_agency_noop_class

        # MECH-219 (SD-019b): affective-harm hysteretic integrator.
        config.use_harm_suffering_accumulator = use_harm_suffering_accumulator
        config.harm_suffering_alpha_rise = harm_suffering_alpha_rise
        config.harm_suffering_alpha_fall = harm_suffering_alpha_fall
        config.harm_suffering_escapability_mode = harm_suffering_escapability_mode
        config.harm_suffering_escapability_constant = (
            harm_suffering_escapability_constant
        )
        config.harm_suffering_external_escapability = (
            harm_suffering_external_escapability
        )
        config.harm_suffering_s_cap = harm_suffering_s_cap
        config.harm_suffering_body_damage_weight = harm_suffering_body_damage_weight
        config.harm_suffering_pe_gain = harm_suffering_pe_gain
        config.harm_suffering_use_bistable_latch = harm_suffering_use_bistable_latch
        config.harm_suffering_theta_on = harm_suffering_theta_on
        config.harm_suffering_theta_off = harm_suffering_theta_off
        config.harm_suffering_redirect_aic = harm_suffering_redirect_aic
        config.harm_suffering_redirect_pag = harm_suffering_redirect_pag
        config.harm_suffering_redirect_mech091 = harm_suffering_redirect_mech091
        config.harm_suffering_redirect_dacc = harm_suffering_redirect_dacc
        config.harm_suffering_redirect_pacc = harm_suffering_redirect_pacc

        config.forced_score_bias_per_class = forced_score_bias_per_class

        # MECH-319: simulation_mode_rule_write_gate
        config.use_simulation_mode_rule_gate = use_simulation_mode_rule_gate
        config.simulation_mode_rule_gate_admit_writes = simulation_mode_rule_gate_admit_writes

        # SD-034: governance.closure_operator
        config.use_closure_operator = use_closure_operator
        config.closure_rule_delta_threshold = closure_rule_delta_threshold
        config.closure_stable_ticks = closure_stable_ticks
        config.closure_require_beta_elevated = closure_require_beta_elevated
        config.closure_min_sd033a_gate = closure_min_sd033a_gate
        config.closure_nogo_injection_count = closure_nogo_injection_count
        config.closure_residue_discharge_factor = closure_residue_discharge_factor
        config.closure_residue_discharge_radius = closure_residue_discharge_radius
        config.closure_signal_value = closure_signal_value
        config.closure_reset_pe_ema = closure_reset_pe_ema
        config.closure_pe_cap_after = closure_pe_cap_after
        config.closure_reset_outcome_history = closure_reset_outcome_history
        config.closure_signal_affinity_internal_planning = closure_signal_affinity_internal_planning
        # SD-034 commitment-closure-control-plane (2026-06-12).
        config.use_closure_env_completion_hook = use_closure_env_completion_hook
        config.closure_decommit_hold_ticks = closure_decommit_hold_ticks
        config.closure_decommit_hold_scale_with_run = (
            closure_decommit_hold_scale_with_run
        )
        config.closure_decommit_hold_max_ticks = closure_decommit_hold_max_ticks
        config.use_closure_commit_beta_coupling = use_closure_commit_beta_coupling

        # SD-035: amygdala analogue (BLAAnalog + CeAAnalog peer modules)
        config.use_amygdala_analog = use_amygdala_analog
        config.use_bla_analog = use_bla_analog
        config.use_cea_analog = use_cea_analog
        config.bla_encoding_gain_max = bla_encoding_gain_max
        config.bla_encoding_gain_floor = bla_encoding_gain_floor
        config.bla_arousal_threshold_on = bla_arousal_threshold_on
        config.bla_arousal_peak = bla_arousal_peak
        config.bla_window_steps = bla_window_steps
        config.bla_window_half_life_steps = bla_window_half_life_steps
        config.bla_retrieval_bias_alpha = bla_retrieval_bias_alpha
        config.bla_retrieval_bias_compensation = bla_retrieval_bias_compensation
        config.bla_retrieval_tag_at_encoding = bla_retrieval_tag_at_encoding
        config.bla_remap_pe_sigma_threshold = bla_remap_pe_sigma_threshold
        config.bla_remap_pe_ema_alpha = bla_remap_pe_ema_alpha
        config.bla_remap_pe_std_init = bla_remap_pe_std_init
        config.bla_remap_code_fraction = bla_remap_code_fraction
        config.bla_remap_requires_attribution = bla_remap_requires_attribution
        config.bla_context_remap_blend = bla_context_remap_blend
        config.cea_fast_route_threshold = cea_fast_route_threshold
        config.cea_fast_route_input_is_lowfreq = cea_fast_route_input_is_lowfreq
        config.cea_mode_prior_log_odds_max = cea_mode_prior_log_odds_max
        config.cea_mode_prior_gain = cea_mode_prior_gain
        config.cea_pre_softmax_additive = cea_pre_softmax_additive
        config.cea_fast_prime_amplitude = cea_fast_prime_amplitude
        config.cea_fast_prime_decay_tau_steps = cea_fast_prime_decay_tau_steps
        config.cea_fast_prime_override_window_steps = cea_fast_prime_override_window_steps
        config.cea_cortical_confirmation_weight = cea_cortical_confirmation_weight

        # SD-036: GABAergic cross-stream decay regulator + MECH-279 freeze gate
        config.use_gabaergic_decay = use_gabaergic_decay
        config.gaba_tone = gaba_tone
        config.gaba_tone_min = gaba_tone_min
        config.gaba_tone_max = gaba_tone_max
        config.gaba_tau_z_harm_s = gaba_tau_z_harm_s
        config.gaba_tau_z_harm_a = gaba_tau_z_harm_a
        config.gaba_tau_z_beta = gaba_tau_z_beta
        config.gaba_decay_z_harm_s = gaba_decay_z_harm_s
        config.gaba_decay_z_harm_a = gaba_decay_z_harm_a
        config.gaba_decay_z_beta = gaba_decay_z_beta
        config.gaba_input_threshold_z_harm_s = gaba_input_threshold_z_harm_s
        config.gaba_input_threshold_z_harm_a = gaba_input_threshold_z_harm_a
        config.gaba_input_threshold_z_beta = gaba_input_threshold_z_beta
        config.use_pag_freeze_gate = use_pag_freeze_gate
        config.pag_theta_freeze = pag_theta_freeze
        config.pag_duration_input_threshold = pag_duration_input_threshold
        config.pag_min_freeze_duration = pag_min_freeze_duration
        config.pag_max_freeze_duration = pag_max_freeze_duration
        config.pag_freeze_noop_action_class = pag_freeze_noop_action_class

        # SD-037: Broadcast Override Regulator (orexin-analog)
        config.use_broadcast_override = use_broadcast_override
        config.override_recruitment_threshold = override_recruitment_threshold
        config.override_alpha_pag = override_alpha_pag
        config.override_salience_reweight_alpha = override_salience_reweight_alpha
        config.override_drive_weight = override_drive_weight
        config.override_harm_weight = override_harm_weight
        config.override_sustained_threat_window = override_sustained_threat_window
        config.override_sustained_threat_threshold = override_sustained_threat_threshold
        config.override_decay_rate = override_decay_rate
        config.override_goal_seeding_gain = override_goal_seeding_gain
        # SD-037 consumer-cascade (MECH-281 motor-coupling axis, 2026-05-30)
        config.override_pfc_eta_gain = override_pfc_eta_gain
        config.override_bla_encoding_gain = override_bla_encoding_gain
        config.override_cea_amplitude_gain = override_cea_amplitude_gain
        config.override_beta_interrupt_gain = override_beta_interrupt_gain

        config.use_lpb_interoceptive_routing = use_lpb_interoceptive_routing
        config.lpb_intero_z_dim = lpb_intero_z_dim
        config.lpb_drive_weight = lpb_drive_weight
        config.lpb_resource_weight = lpb_resource_weight

        # MECH-269 / MECH-287 / MECH-288: V_s invalidation runtime Phase 1 + 2 flags
        config.hippocampal.use_per_stream_vs = use_per_stream_vs
        config.hippocampal.use_event_segmenter = use_event_segmenter
        config.hippocampal.use_invalidation_trigger = use_invalidation_trigger
        config.hippocampal.use_anchor_sets = use_anchor_sets
        config.hippocampal.use_per_region_vs = use_per_region_vs
        config.hippocampal.use_staleness_accumulator = use_staleness_accumulator
        config.hippocampal.use_mech284_hysteresis = use_mech284_hysteresis
        config.hippocampal.use_vs_commit_release = use_vs_commit_release
        # SD-039: master flag lives on AnchorSetConfig (cf. AnchorSetConfig
        # docstring). Propagate the from_dims kwarg to the nested config so
        # the anchor set's payload semantics fire when the substrate is on.
        config.hippocampal.anchor_set.use_sd039_anchor_payload = use_sd039_anchor_payload

        # MECH-269b: V_s rollout gating on E1/E2 forward predictions
        config.hippocampal.use_vs_rollout_gating = use_vs_rollout_gating
        config.hippocampal.vs_gate_snapshot_refresh_threshold = vs_gate_snapshot_refresh_threshold
        config.hippocampal.vs_gate_e1_threshold = vs_gate_e1_threshold
        config.hippocampal.vs_gate_e2_threshold = vs_gate_e2_threshold
        config.hippocampal.vs_gate_streams = vs_gate_streams
        config.hippocampal.vs_gate_unknown_stream_passes = vs_gate_unknown_stream_passes
        config.hippocampal.use_vs_gate_staleness_lookup = use_vs_gate_staleness_lookup

        # MECH-292: ranked ghost-goal bank (derived view over SD-039 anchor pool)
        config.hippocampal.use_mech292_ghost_bank = use_mech292_ghost_bank
        config.hippocampal.ghost_goal_bank_config.wanting_weight = mech292_wanting_weight
        config.hippocampal.ghost_goal_bank_config.goal_match_weight = mech292_goal_match_weight
        config.hippocampal.ghost_goal_bank_config.staleness_weight = mech292_staleness_weight
        config.hippocampal.ghost_goal_bank_config.recoverability_weight = mech292_recoverability_weight
        config.hippocampal.ghost_goal_bank_config.goal_match_floor = mech292_goal_match_floor
        config.hippocampal.ghost_goal_bank_config.top_k = mech292_top_k

        # MECH-293: waking ghost-goal probe search (consumer of MECH-292 bank)
        config.hippocampal.use_mech293_ghost_probes = use_mech293_ghost_probes
        config.hippocampal.mech293_ghost_fraction = mech293_ghost_fraction
        config.hippocampal.mech293_min_ghost_candidates = mech293_min_ghost_candidates
        config.hippocampal.mech293_max_ghost_candidates = mech293_max_ghost_candidates
        config.hippocampal.mech293_replace_lowest_ranked = mech293_replace_lowest_ranked

        # V3-EXQ-553 proposer-fix substrate: orthogonal CEM seeding
        config.hippocampal.use_orthogonal_cem_seeding = use_orthogonal_cem_seeding
        config.hippocampal.use_action_class_scaffold_candidates = (
            use_action_class_scaffold_candidates
        )
        config.hippocampal.use_support_preserving_cem = use_support_preserving_cem
        config.hippocampal.support_preserving_min_first_action_classes = (
            support_preserving_min_first_action_classes
        )
        config.hippocampal.support_preserving_stratified_elites = (
            support_preserving_stratified_elites
        )
        config.hippocampal.support_preserving_per_class_quota = (
            support_preserving_per_class_quota
        )
        config.hippocampal.support_preserving_ao_std_floor = (
            support_preserving_ao_std_floor
        )
        # SD-055: differentiable CEM selection approximation
        config.hippocampal.use_differentiable_cem = use_differentiable_cem
        config.hippocampal.differentiable_cem_temperature = differentiable_cem_temperature

        # V3-EXQ-563c: score/bias scale normalisation
        config.e3.normalize_score_bias_to_e3_range = normalize_score_bias_to_e3_range
        # modulatory-bias-selection-authority (2026-06-03): additive-bias sites
        # read from config.e3; the stratified across-class site reads the
        # top-level mirror via build_from_ree_config.
        config.e3.use_modulatory_selection_authority = use_modulatory_selection_authority
        config.e3.modulatory_authority_gain = modulatory_authority_gain
        config.e3.modulatory_authority_min_range_floor = modulatory_authority_min_range_floor
        config.use_modulatory_selection_authority = use_modulatory_selection_authority
        # ARC-108 JOB-1 step-1: learned per-channel selection-gating reads from
        # config.e3 (the e3_selector _modulatory_accum composition site).
        config.e3.use_learned_channel_gating = use_learned_channel_gating
        config.e3.learned_channel_gating_eta = learned_channel_gating_eta
        config.e3.learned_channel_gating_elig_decay = learned_channel_gating_elig_decay
        config.e3.learned_channel_value_baseline_beta = (
            learned_channel_value_baseline_beta
        )
        config.e3.learned_channel_asym_potentiation = learned_channel_asym_potentiation
        config.e3.learned_channel_asym_depression = learned_channel_asym_depression
        config.e3.learned_channel_rpe_mode = learned_channel_rpe_mode
        # MECH-451: finer-channel-granularity selection-gating reads from config.e3.
        config.e3.use_finer_channel_gating = use_finer_channel_gating
        # ARC-108 JOB-1 step-2 / MECH-450 recurrent-settling step (learned W_lat)
        config.e3.use_learned_settling_step = use_learned_settling_step
        config.e3.learned_settling_rounds = learned_settling_rounds
        config.e3.learned_settling_temperature = learned_settling_temperature
        config.e3.learned_settling_eta = learned_settling_eta
        config.e3.learned_settling_elig_decay = learned_settling_elig_decay
        config.e3.learned_settling_n_action_classes = learned_settling_n_action_classes
        # MECH-140 x MECH-450 disinhibitory soft-competitive settling (parameter-free).
        config.e3.use_soft_competitive_settling = use_soft_competitive_settling
        config.e3.soft_competitive_settling_rounds = soft_competitive_settling_rounds
        config.e3.soft_competitive_settling_gain = soft_competitive_settling_gain
        config.e3.soft_competitive_settling_temperature = soft_competitive_settling_temperature
        config.e3.soft_competitive_settling_cross_class = soft_competitive_settling_cross_class
        # ARC-110 segregated loops + S2 null + ARC-109 D1/D2 + MECH-452 traces.
        config.e3.use_loop_segregation = use_loop_segregation
        config.e3.loop_segregation_channel_map = (
            dict(loop_segregation_channel_map)
            if loop_segregation_channel_map else {}
        )
        config.e3.loop_segregation_default_loop = loop_segregation_default_loop
        config.e3.loop_segregation_spiral_gain_assoc = loop_segregation_spiral_gain_assoc
        config.e3.loop_segregation_spiral_gain_limbic = loop_segregation_spiral_gain_limbic
        config.e3.loop_segregation_motor_authority = loop_segregation_motor_authority
        config.e3.loop_segregation_normalize = loop_segregation_normalize
        config.e3.loop_segregation_noise_on = loop_segregation_noise_on
        config.e3.loop_segregation_noise_alpha = loop_segregation_noise_alpha
        # ARC-110 C2 release: per-named-channel range-preserving routing (2026-06-28).
        config.e3.use_named_channel_routing = use_named_channel_routing
        config.e3.use_d1_d2_population_split = use_d1_d2_population_split
        config.e3.d1_da_gain = d1_da_gain
        config.e3.d2_da_gain = d2_da_gain
        config.e3.use_loop_local_eligibility_traces = use_loop_local_eligibility_traces
        # ARC-108 x ARC-110 learned (dopamine-gated) cross-loop arbitration.
        config.e3.use_learned_cross_loop_arbitration = use_learned_cross_loop_arbitration
        config.e3.learned_cross_loop_eta = learned_cross_loop_eta
        config.e3.use_ascending_spiral_gain = use_ascending_spiral_gain
        config.e3.loop_segregation_ascending_spiral_gain = loop_segregation_ascending_spiral_gain
        config.e3.loop_segregation_ascending_plasticity_gain = loop_segregation_ascending_plasticity_gain
        config.e3.use_ascending_parity_controller = use_ascending_parity_controller
        config.e3.loop_segregation_parity_forward_gain = loop_segregation_parity_forward_gain
        config.e3.loop_segregation_parity_ceiling_ratio = loop_segregation_parity_ceiling_ratio
        config.e3.loop_segregation_parity_plasticity_gain = loop_segregation_parity_plasticity_gain
        config.e3.loop_segregation_m_cross_clamp = loop_segregation_m_cross_clamp
        # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a,
        # 2026-06-10): the e3_selector additive site reads
        # use_modulatory_channel_routing + min_range_floor from config.e3; the
        # agent reads route_source + route_weight from the top-level mirror.
        config.e3.use_modulatory_channel_routing = use_modulatory_channel_routing
        config.e3.modulatory_channel_route_min_range_floor = (
            modulatory_channel_route_min_range_floor
        )
        config.e3.modulatory_channel_route_weight = modulatory_channel_route_weight
        config.use_modulatory_channel_routing = use_modulatory_channel_routing
        config.modulatory_channel_route_source = modulatory_channel_route_source
        config.modulatory_channel_route_weight = modulatory_channel_route_weight
        # modulatory-bias-selection-authority CONVERSION amend (569g/682, 2026-06-15)
        config.e3.modulatory_authority_normalize_basis = (
            modulatory_authority_normalize_basis
        )
        config.e3.use_modulatory_shortlist_then_modulate = (
            use_modulatory_shortlist_then_modulate
        )
        config.e3.modulatory_shortlist_margin = modulatory_shortlist_margin
        # modulatory-bias-selection-authority TOP-K shortlist amend (569h, 2026-06-16)
        config.e3.modulatory_shortlist_mode = modulatory_shortlist_mode
        config.e3.modulatory_shortlist_k = modulatory_shortlist_k
        # CONVERSION-CEILING / F-dominance conflict-grade amend (MECH-439, 2026-06-18)
        config.e3.modulatory_shortlist_conflict_graded = (
            modulatory_shortlist_conflict_graded
        )
        config.e3.modulatory_shortlist_k_max = modulatory_shortlist_k_max
        config.e3.use_gap_scaled_commit_temperature = (
            use_gap_scaled_commit_temperature
        )
        config.e3.gap_scaled_commit_entropy_alpha = gap_scaled_commit_entropy_alpha
        config.e3.gap_scaled_commit_harm_floor = gap_scaled_commit_harm_floor
        # MECH-448 / ARC-107 rank-preserving F->eligibility demotion (2026-06-20)
        config.e3.use_f_eligibility_demotion = use_f_eligibility_demotion
        config.e3.f_eligibility_envelope_floor = f_eligibility_envelope_floor
        config.e3.f_eligibility_dn_sigma = f_eligibility_dn_sigma
        config.e3.use_f_eligibility_adaptive_floor = use_f_eligibility_adaptive_floor
        config.e3.f_eligibility_adaptive_mean_factor = f_eligibility_adaptive_mean_factor
        # MECH-449 / ARC-107 Go/No-Go eligibility constitution (2026-06-21).
        config.e3.use_go_nogo_constitution = use_go_nogo_constitution
        config.e3.gng_safety_floor = gng_safety_floor
        config.e3.gng_staleness_floor = gng_staleness_floor
        config.e3.gng_perseveration_floor = gng_perseveration_floor
        config.e3.gng_viability_floor = gng_viability_floor
        config.e3.gng_go_threshold = gng_go_threshold
        config.e3.gng_go_max_promote = gng_go_max_promote
        config.e3.gng_protect_min_eligible = gng_protect_min_eligible
        # DR-12 (self_model_v4:SELF-4, 2026-06-17): E2 forward-PE -> E3 confidence
        # down-weight. The score_trajectory penalty reads these from config.e3.
        config.e3.use_pe_confidence_weighting = use_pe_confidence_weighting
        config.e3.pe_confidence_weight = pe_confidence_weight
        config.e3.pe_confidence_mode = pe_confidence_mode
        config.e3.pe_confidence_scale = pe_confidence_scale
        config.e3.use_self_viability_weighting = use_self_viability_weighting
        config.e3.self_viability_weight = self_viability_weight
        config.e3.self_viability_mode = self_viability_mode
        config.e3.self_viability_scale = self_viability_scale
        # MECH-440 NoisyNet propagating selection-head weight noise (2026-06-27).
        config.e3.use_noisy_selection_head = use_noisy_selection_head
        config.e3.noisy_selection_sigma_init = noisy_selection_sigma_init
        config.e3.noisy_selection_weight = noisy_selection_weight
        config.e3.noisy_selection_anneal = noisy_selection_anneal
        config.e3.noisy_selection_anneal_floor = noisy_selection_anneal_floor
        config.e3.noisy_selection_anneal_ema_alpha = noisy_selection_anneal_ema_alpha
        # MECH-441 model-disagreement directed curiosity (2026-06-27).
        config.e3.use_model_disagreement_curiosity = use_model_disagreement_curiosity
        config.e3.model_disagreement_weight = model_disagreement_weight
        config.e3.model_disagreement_mode = model_disagreement_mode
        config.e3.model_disagreement_scale = model_disagreement_scale
        config.latent.n_disagreement_heads = n_disagreement_heads
        config.latent.disagreement_bootstrap_mask_prob = disagreement_bootstrap_mask_prob
        config.latent.disagreement_learning_rate = disagreement_learning_rate
        config.use_control_vector_logging = use_control_vector_logging

        # MECH-290: backward trajectory credit sweep
        config.hippocampal.use_backward_credit_sweep = use_backward_credit_sweep
        config.hippocampal.backward_sweep_gamma = backward_sweep_gamma
        config.hippocampal.backward_sweep_min_quality = backward_sweep_min_quality

        # Sleep-aggregation cluster Phase A
        config.use_sleep_loop = use_sleep_loop
        config.sleep_loop_episodes_K = sleep_loop_episodes_K
        config.sleep_loop_require_passes = sleep_loop_require_passes

        # SD-MEL-CONSUMER (sleep_substrate:GAP-5b)
        config.use_mel_consumer = use_mel_consumer
        config.mel_gain = mel_gain
        config.mel_reference = mel_reference
        config.mel_reference_mode = mel_reference_mode
        config.mel_ema_alpha = mel_ema_alpha
        config.mel_duration_factor_min = mel_duration_factor_min
        config.mel_duration_factor_max = mel_duration_factor_max
        config.mel_relative_floor = mel_relative_floor
        config.mel_scale_sws = mel_scale_sws
        config.mel_scale_rem = mel_scale_rem
        config.use_mel_entry = use_mel_entry
        config.mel_entry_threshold = mel_entry_threshold

        # Sleep-aggregation cluster Phase B (MECH-285)
        config.use_mech285_sampler = use_mech285_sampler
        config.mech285_draws_per_cycle = mech285_draws_per_cycle
        config.mech285_temperature = mech285_temperature
        config.mech285_allow_uniform_fallback = mech285_allow_uniform_fallback

        # Sleep-aggregation cluster Phase C (MECH-272)
        config.use_mech272_routing = use_mech272_routing
        config.mech272_waking_anchor_weight = mech272_waking_anchor_weight
        config.mech272_waking_probe_weight = mech272_waking_probe_weight
        config.mech272_sws_anchor_weight = mech272_sws_anchor_weight
        config.mech272_sws_probe_weight = mech272_sws_probe_weight
        config.mech272_rem_anchor_weight = mech272_rem_anchor_weight
        config.mech272_rem_probe_weight = mech272_rem_probe_weight
        config.use_mech272_routing_consumer = use_mech272_routing_consumer

        # Sleep-aggregation cluster Phase D (MECH-275)
        config.use_mech275_aggregator = use_mech275_aggregator
        config.mech275_domains = tuple(mech275_domains)
        config.mech275_prior_mean = mech275_prior_mean
        config.mech275_prior_variance = mech275_prior_variance
        config.mech275_likelihood_variance = mech275_likelihood_variance
        config.mech275_decay_factor = mech275_decay_factor
        config.mech275_probe_gain = mech275_probe_gain

        # Sleep-aggregation cluster Phase E (MECH-273)
        config.use_mech273_self_model = use_mech273_self_model
        # MECH-204 Option A: precision recalibration consumer
        config.use_rem_precision_recalibration = use_rem_precision_recalibration
        config.rem_precision_recalibration_step = rem_precision_recalibration_step
        config.mech273_offline_lr_scale = mech273_offline_lr_scale
        config.mech273_offline_n_steps = mech273_offline_n_steps
        config.mech273_partial_decay_factor = mech273_partial_decay_factor

        # MECH-286: override-gated sleep onset
        config.use_mech286_sleep_onset_gate = use_mech286_sleep_onset_gate
        config.mech286_theta_sleep_permit = mech286_theta_sleep_permit
        config.mech286_theta_sleep_recruit = mech286_theta_sleep_recruit
        config.mech286_threat_tonic_threshold = mech286_threat_tonic_threshold

        # MECH-295: drive -> liking-stream -> approach_cue bridge
        config.use_mech295_liking_bridge = use_mech295_liking_bridge
        config.mech295_drive_to_liking_gain = mech295_drive_to_liking_gain
        config.mech295_liking_to_approach_cue_gain = mech295_liking_to_approach_cue_gain
        config.mech295_min_drive_to_fire = mech295_min_drive_to_fire
        config.mech295_min_z_goal_norm_to_fire = mech295_min_z_goal_norm_to_fire

        # MECH-302: suffering-derivative comparator substrate
        config.use_suffering_derivative_comparator = use_suffering_derivative_comparator
        config.suffering_window_length = suffering_window_length
        config.suffering_drop_threshold = suffering_drop_threshold
        config.suffering_min_initial_norm = suffering_min_initial_norm
        config.relief_completion_weight = relief_completion_weight

        # SD-051 / MECH-304: conditioned safety store
        config.use_conditioned_safety_store = use_conditioned_safety_store
        config.safety_store_ema_alpha = safety_store_ema_alpha
        config.safety_store_decay_rate = safety_store_decay_rate
        config.safety_store_min_norm = safety_store_min_norm
        config.safety_store_threshold = safety_store_threshold
        config.safety_store_commitment_weight = safety_store_commitment_weight

        # MECH-303: contextual passive safety terrain
        config.use_contextual_safety_terrain = use_contextual_safety_terrain
        config.contextual_safety_accum_weight = contextual_safety_accum_weight
        config.contextual_safety_harm_threshold = contextual_safety_harm_threshold
        config.contextual_safety_release_threshold = contextual_safety_release_threshold
        if use_contextual_safety_terrain:
            config.residue.safety_terrain_enabled = True

        # MECH-108: BreathOscillator -- wire heartbeat params from from_dims().
        # breath_period=0 disables; default 50 enables periodic uncommitted windows.
        config.heartbeat.breath_period = breath_period
        config.heartbeat.breath_sweep_amplitude = breath_sweep_amplitude
        config.heartbeat.breath_sweep_duration = breath_sweep_duration

        if goal_stream_enabled:
            config.enable_goal_stream(
                goal_weight=goal_weight,
                wanting_weight=wanting_weight if wanting_weight > 0.0 else 0.5,
                benefit_threshold=benefit_threshold,
                drive_weight=drive_weight,
                schema_wanting_threshold=schema_wanting_threshold,
                schema_wanting_gain=schema_wanting_gain,
            )

        # GAP-3: resolve the unified sleep-aggregation cluster master flag
        # AFTER the individual Phase A-E flag assignments above so the master
        # wins (OR-only: forces the eight sub-flags True). __post_init__ ran
        # on cls() with the default False and was a no-op for this path.
        if use_sleep_aggregation_cluster:
            config.enable_sleep_aggregation_cluster()

        # SD-049 Phase 3: SD-032 consumer cascade. Surface the master flag +
        # 7 per-consumer combiners. Validation of combiner mode happens at
        # the helper (ree_core.utils.per_axis_drive.collapse_per_axis_drive)
        # the first time the agent invokes it -- we keep the assignment
        # cheap and defer validation until use, matching the SD-035 / SD-036
        # / MECH-313 / MECH-314 / MECH-320 pattern.
        config.use_sd049_per_axis_consumer_cascade = use_sd049_per_axis_consumer_cascade
        config.sd049_aic_per_axis_combiner = sd049_aic_per_axis_combiner
        config.sd049_pcc_per_axis_combiner = sd049_pcc_per_axis_combiner
        config.sd049_pacc_per_axis_combiner = sd049_pacc_per_axis_combiner
        config.sd049_dacc_per_axis_combiner = sd049_dacc_per_axis_combiner
        config.sd049_salience_per_axis_combiner = sd049_salience_per_axis_combiner
        config.sd049_override_per_axis_combiner = sd049_override_per_axis_combiner
        config.sd049_mech295_per_axis_combiner = sd049_mech295_per_axis_combiner

        # MECH-423 R2 + R3 readiness instrumentation. Surfaced via kwargs.pop
        # (signature-stable, matching the use_resource_encoder / goal-stream pop
        # precedent) so the from_dims() positional argument order is unchanged.
        # All default no-op -> bit-identical OFF.
        #   R2 (LatentStackConfig): iterative-inference convergence readout.
        config.latent.use_iterative_inference = bool(
            kwargs.pop("use_iterative_inference", False)
        )
        config.latent.inference_settle_iters = int(
            kwargs.pop("inference_settle_iters", 1)
        )
        config.latent.inference_convergence_rel_tol = float(
            kwargs.pop("inference_convergence_rel_tol", 0.05)
        )
        #   SELF-1 / DR-13 (LatentStackConfig): z_self temporal depth.
        #   Dedicated gated self-recurrence + E1-feedback anchor. no-op OFF.
        config.latent.use_self_recurrence = bool(
            kwargs.pop("use_self_recurrence", False)
        )
        config.latent.self_recurrence_e1_coupling = float(
            kwargs.pop("self_recurrence_e1_coupling", 0.15)
        )
        #   R3 (REEConfig): module-tagged interleaved cross-module consolidation.
        config.use_cross_module_consolidation = bool(
            kwargs.pop("use_cross_module_consolidation", False)
        )
        config.cross_module_consolidation_schedule = str(
            kwargs.pop("cross_module_consolidation_schedule", "interleaved")
        )
        config.cross_module_consolidation_steps = int(
            kwargs.pop("cross_module_consolidation_steps", 0)
        )
        config.cross_module_consolidation_lr = float(
            kwargs.pop("cross_module_consolidation_lr", 1e-3)
        )
        config.cross_module_consolidation_batch = int(
            kwargs.pop("cross_module_consolidation_batch", 16)
        )

        return config

    @classmethod
    def goal_stream(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        **kwargs,
    ) -> "REEConfig":
        """Create a config with the canonical goal-stream bundle enabled."""
        goal_weight = float(kwargs.pop("goal_weight", 0.5))
        wanting_weight = float(kwargs.pop("wanting_weight", 0.5))
        benefit_threshold = float(kwargs.pop("benefit_threshold", 0.05))
        drive_weight = float(kwargs.pop("drive_weight", 2.0))
        schema_wanting_threshold = float(
            kwargs.pop("schema_wanting_threshold", 0.10)
        )
        schema_wanting_gain = float(kwargs.pop("schema_wanting_gain", 0.60))
        use_resource_encoder = bool(kwargs.pop("use_resource_encoder", True))
        use_mech307 = bool(kwargs.pop("use_mech307", True))
        use_consumer_conjunction_read = bool(
            kwargs.pop("use_consumer_conjunction_read", True)
        )
        config = cls.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            z_goal_enabled=True,
            goal_weight=goal_weight,
            benefit_threshold=benefit_threshold,
            drive_weight=drive_weight,
            wanting_weight=wanting_weight,
            schema_wanting_enabled=True,
            schema_wanting_threshold=schema_wanting_threshold,
            schema_wanting_gain=schema_wanting_gain,
            use_mech295_liking_bridge=True,
            **kwargs,
        )
        return config.enable_goal_stream(
            goal_weight=goal_weight,
            wanting_weight=wanting_weight,
            benefit_threshold=benefit_threshold,
            drive_weight=drive_weight,
            schema_wanting_threshold=schema_wanting_threshold,
            schema_wanting_gain=schema_wanting_gain,
            use_resource_encoder=use_resource_encoder,
            use_mech307=use_mech307,
            use_consumer_conjunction_read=use_consumer_conjunction_read,
        )

    @classmethod
    def large(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        **kwargs,
    ) -> "REEConfig":
        """Config preset for GPU-beneficial scale (world_dim=128).

        Designed for NVIDIA Spark (GB10 Blackwell) or equivalent hardware.
        GPU becomes faster than CPU at world_dim >= 128 per V3 calibration data
        (Daniel-PC GTX 1050 Ti: CPU always wins at world_dim=32 at batch=1).

        Scaling choices:
          self_dim=128, world_dim=128 (SD-005 split, each 4x default)
          action_object_dim=64 (maintains SD-004 compression ratio vs world_dim)
          E1/E2/Hippocampal hidden_dim=256, E3 hidden_dim=128
          ResidueField: 64 basis functions, hidden_dim=128

        All feature flags (z_goal_enabled, benefit_eval_enabled, etc.) are
        forwarded to from_dims() via **kwargs.
        """
        config = cls.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            self_dim=128,
            world_dim=128,
            alpha_world=0.9,
            alpha_self=0.3,
            action_object_dim=64,
            **kwargs,
        )
        config.e1.hidden_dim = 256
        config.e2.hidden_dim = 256
        config.e3.hidden_dim = 128
        config.hippocampal.hidden_dim = 256
        config.residue.hidden_dim = 128
        config.residue.num_basis_functions = 64
        return config

    @classmethod
    def xlarge(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        **kwargs,
    ) -> "REEConfig":
        """Config preset for large-scale GPU experiments (world_dim=256).

        Designed for two linked NVIDIA Sparks (256 GB unified memory) or
        a datacenter GPU with >= 32 GB VRAM.

        Suitable for:
          - V5 multi-agent social synchronisation experiments
          - V4 sleep consolidation at scale (MECH-120/121/122/123)
          - SD-010 nociceptive separation (ARC-027) full implementation
          - SD-005/SD-004 co-design scale validation

        Scaling choices:
          self_dim=256, world_dim=256
          action_object_dim=64 (compression ratio maintained)
          E1/E2/Hippocampal hidden_dim=512, E3 hidden_dim=256
          ResidueField: 128 basis functions, hidden_dim=256

        All feature flags forwarded to from_dims() via **kwargs.
        """
        config = cls.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            self_dim=256,
            world_dim=256,
            alpha_world=0.9,
            alpha_self=0.3,
            action_object_dim=64,
            **kwargs,
        )
        config.e1.hidden_dim = 512
        config.e2.hidden_dim = 512
        config.e3.hidden_dim = 256
        config.hippocampal.hidden_dim = 512
        config.residue.hidden_dim = 256
        config.residue.num_basis_functions = 128
        return config
