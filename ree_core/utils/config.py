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
from typing import Dict, Optional

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

    # ARC-033: E2_harm_s forward model (sensory-discriminative harm stream predictor).
    # When True, experiments should construct an E2HarmSForward instance and train it
    # on z_harm_s transitions (P1 phase, after P0 HarmEncoder warmup).
    # Disabled by default -- backward compatible with all existing experiments.
    # See ree_core/predictors/e2_harm_s.py for the module and training protocol.
    use_e2_harm_s_forward: bool = False

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
    # Sleep-aggregation cluster (MECH-272 / MECH-273 / MECH-275 / MECH-285)
    # ----------------------------------------------------------------
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
    # when sated). At default 0.1 the bridge fires only when the agent
    # is at least mildly depleted.
    mech295_min_drive_to_fire: float = 0.1
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
    mech307_conjunction_z_beta_threshold: float = 0.6
    # Negative-score gain applied per candidate whose conjunction predicate
    # fires. Bias term = -mech307_conjunction_gain * drive_level when the
    # predicate holds; 0 otherwise.
    mech307_conjunction_gain: float = 1.0

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
        valence_wanting_floor: float = 0.0,  # MECH-186: minimum z_goal norm floor (0=disabled)
        z_goal_seeding_gain: float = 1.0,  # MECH-187: gain on seeding signal (1.0=no change)
        z_goal_inject: float = 0.0,  # MECH-188: PFC top-down injection norm floor (0=disabled)
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
        # SD-033b: OFC-analog (specific-outcome / task-structure substrate)
        use_ofc_analog: bool = False,
        ofc_state_dim: int = 16,
        ofc_update_eta: float = 0.05,
        ofc_outcome_pool_weight: float = 0.5,
        ofc_bias_scale: float = 0.1,
        ofc_hidden_dim: int = 32,
        ofc_harm_dim: int = 0,
        use_ofc_outcome_oracle: bool = False,
        # ARC-062 Phase 1: gated-policy heads + context discriminator
        use_gated_policy: bool = False,
        gated_policy_n_heads: int = 2,
        gated_policy_disc_hidden: int = 24,
        gated_policy_disc_init_scale: float = 0.1,
        gated_policy_head_hidden: int = 32,
        gated_policy_bias_scale: float = 0.1,
        gated_policy_head_init_bias_offset: float = 0.05,
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
        # MECH-290: backward trajectory credit sweep
        use_backward_credit_sweep: bool = False,
        backward_sweep_gamma: float = 0.9,
        backward_sweep_min_quality: float = 0.6,
        # Sleep-aggregation cluster Phase A master flag
        use_sleep_loop: bool = False,
        sleep_loop_episodes_K: int = 1,
        sleep_loop_require_passes: bool = True,
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
        # MECH-295: drive -> liking-stream -> approach_cue bridge
        use_mech295_liking_bridge: bool = False,
        mech295_drive_to_liking_gain: float = 1.0,
        mech295_liking_to_approach_cue_gain: float = 0.5,
        mech295_min_drive_to_fire: float = 0.1,
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
            "valence_wanting_floor",  # MECH-186
            "z_goal_seeding_gain",  # MECH-187
            "z_goal_inject",  # MECH-188
        }
        local_goal_vals = {
            "z_goal_enabled": z_goal_enabled,
            "alpha_goal": alpha_goal,
            "decay_goal": decay_goal,
            "benefit_threshold": benefit_threshold,
            "goal_weight": goal_weight,
            "e1_goal_conditioned": e1_goal_conditioned,
            "drive_weight": drive_weight,  # SD-012
            "valence_wanting_floor": valence_wanting_floor,  # MECH-186
            "z_goal_seeding_gain": z_goal_seeding_gain,  # MECH-187
            "z_goal_inject": z_goal_inject,  # MECH-188
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

        # SD-033b: OFC-analog
        config.use_ofc_analog = use_ofc_analog
        config.ofc_state_dim = ofc_state_dim
        config.ofc_update_eta = ofc_update_eta
        config.ofc_outcome_pool_weight = ofc_outcome_pool_weight
        config.ofc_bias_scale = ofc_bias_scale
        config.ofc_hidden_dim = ofc_hidden_dim
        config.ofc_harm_dim = ofc_harm_dim
        config.use_ofc_outcome_oracle = use_ofc_outcome_oracle

        # ARC-062 Phase 1: gated-policy heads + context discriminator
        config.use_gated_policy = use_gated_policy
        config.gated_policy_n_heads = gated_policy_n_heads
        config.gated_policy_disc_hidden = gated_policy_disc_hidden
        config.gated_policy_disc_init_scale = gated_policy_disc_init_scale
        config.gated_policy_head_hidden = gated_policy_head_hidden
        config.gated_policy_bias_scale = gated_policy_bias_scale
        config.gated_policy_head_init_bias_offset = gated_policy_head_init_bias_offset

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

        # MECH-290: backward trajectory credit sweep
        config.hippocampal.use_backward_credit_sweep = use_backward_credit_sweep
        config.hippocampal.backward_sweep_gamma = backward_sweep_gamma
        config.hippocampal.backward_sweep_min_quality = backward_sweep_min_quality

        # Sleep-aggregation cluster Phase A
        config.use_sleep_loop = use_sleep_loop
        config.sleep_loop_episodes_K = sleep_loop_episodes_K
        config.sleep_loop_require_passes = sleep_loop_require_passes

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

        return config

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
