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
    """
    self_dim: int = 32         # z_self dimension
    world_dim: int = 32        # z_world dimension
    latent_dim: int = 64       # total = self_dim + world_dim
    hidden_dim: int = 128
    num_layers: int = 3
    prediction_horizon: int = 20   # Steps into future
    learning_rate: float = 1e-4


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

    device: str = "cpu"
    seed: Optional[int] = None

    offline_integration_frequency: int = 100

    # MECH-057a: action-loop completion gate (preserved from V2)
    action_loop_gate_enabled: bool = False

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
        use_harm_stream: bool = False,
        harm_obs_dim: int = 51,
        z_harm_dim: int = 32,
        # ARC-030 / MECH-111 / MECH-112 / MECH-113
        benefit_eval_enabled: bool = False,
        benefit_weight: float = 1.0,
        novelty_bonus_weight: float = 0.0,
        self_maintenance_weight: float = 0.0,
        self_maintenance_d_eff_target: float = 1.5,
        # MECH-112 / MECH-116: z_goal substrate
        z_goal_enabled: bool = False,
        alpha_goal: float = 0.05,
        decay_goal: float = 0.005,
        benefit_threshold: float = 0.1,
        goal_weight: float = 1.0,
        e1_goal_conditioned: bool = True,
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

        # SD-010: dedicated harm stream
        config.latent.use_harm_stream = use_harm_stream
        config.latent.harm_obs_dim = harm_obs_dim
        config.latent.z_harm_dim = z_harm_dim

        # E1
        config.e1.self_dim = self_dim
        config.e1.world_dim = world_dim
        config.e1.latent_dim = self_dim + world_dim

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

        # Hippocampal
        config.hippocampal.world_dim = world_dim
        config.hippocampal.action_dim = action_dim
        config.hippocampal.action_object_dim = action_object_dim
        config.hippocampal.horizon = config.e2.rollout_horizon

        # Residue
        config.residue.world_dim = world_dim

        # Environment
        config.environment.body_obs_dim = body_obs_dim
        config.environment.world_obs_dim = world_obs_dim

        # GoalConfig fields -- wire goal_dim to world_dim by default
        goal_fields = {
            "z_goal_enabled", "alpha_goal", "decay_goal",
            "benefit_threshold", "goal_weight", "e1_goal_conditioned",
        }
        local_goal_vals = {
            "z_goal_enabled": z_goal_enabled,
            "alpha_goal": alpha_goal,
            "decay_goal": decay_goal,
            "benefit_threshold": benefit_threshold,
            "goal_weight": goal_weight,
            "e1_goal_conditioned": e1_goal_conditioned,
        }
        for _key in goal_fields:
            if _key in local_goal_vals:
                setattr(config.goal, _key, local_goal_vals[_key])
        # Keep goal_dim in sync with world_dim
        if hasattr(config, "latent") and hasattr(config.latent, "world_dim"):
            config.goal.goal_dim = config.latent.world_dim

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
