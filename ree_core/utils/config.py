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
from typing import Optional

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

    # MECH-165: reverse replay diversity scheduler
    replay_diversity_enabled: bool = False   # master switch (default off for backward compat)
    reverse_replay_fraction: float = 0.3     # fraction of replay calls using reverse mode
    random_replay_fraction: float = 0.2      # fraction using random action rollout
    exploration_buffer_len: int = 50         # max stored exploration trajectories (FIFO)

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
        # MECH-205: surprise-gated replay
        surprise_gated_replay: bool = False,
        pe_ema_alpha: float = 0.02,
        pe_surprise_threshold: float = 0.001,
        # MECH-120: SHY-analog synaptic homeostasis
        shy_enabled: bool = False,
        shy_decay_rate: float = 0.85,
        # MECH-165: reverse replay diversity scheduler
        replay_diversity_enabled: bool = False,
        reverse_replay_fraction: float = 0.3,
        random_replay_fraction: float = 0.2,
        exploration_buffer_len: int = 50,
        # VALENCE_WANTING gradient in trajectory scoring
        wanting_weight: float = 0.0,
        **kwargs,
    ) -> "REEConfig":
        """Create config from basic dimension specifications."""
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
        config.e1.action_object_dim = action_object_dim

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

        # MECH-205: surprise-gated replay
        config.surprise_gated_replay = surprise_gated_replay
        config.pe_ema_alpha = pe_ema_alpha
        config.pe_surprise_threshold = pe_surprise_threshold

        # MECH-120: SHY normalization
        config.shy_enabled = shy_enabled
        config.shy_decay_rate = shy_decay_rate

        # MECH-165: reverse replay diversity scheduler
        config.replay_diversity_enabled = replay_diversity_enabled
        config.reverse_replay_fraction = reverse_replay_fraction
        config.random_replay_fraction = random_replay_fraction
        config.exploration_buffer_len = exploration_buffer_len

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
