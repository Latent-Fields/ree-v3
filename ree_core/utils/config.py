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

    # SD-007: ReafferencePredictor — perspective-shift correction for z_world.
    # Set reafference_action_dim = action_dim (e.g. 4) to enable.
    # 0 = disabled (default; backward compatible with EXQ-001–025).
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
    # 0.02 = commit when env is stable (low prediction error); don't commit in
    # perturbed env (high prediction error). See e3_selector.py::variance_commit_threshold.
    # (Prior value was 0.7 which was on precision scale ~100 — always True. Fixed 2026-03-18.)
    commitment_threshold: float = 0.02    # variance-space threshold
    precision_ema_alpha: float = 0.05     # EMA decay for running variance estimate
    precision_init: float = 0.5          # initial running variance (starts uncommitted)


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

        return config
