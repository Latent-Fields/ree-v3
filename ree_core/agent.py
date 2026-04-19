"""
REE-v3 Agent Implementation

Integrates all V3 architectural components:
  - LatentStack (SD-005 split: z_self + z_world)
  - E1 DeepPredictor (associative prior over z_self + z_world)
  - E2 FastPredictor (SD-004: action objects; SD-005: world_forward)
  - E3 TrajectorySelector (harm_eval + dynamic precision)
  - HippocampalModule (action-object space navigation, MECH-092 replay)
  - ResidueField (z_world substrate, MECH-094 hypothesis_tag gate)
  - MultiRateClock (SD-006 phase 1: time-multiplexed multi-rate)
  - ThetaBuffer (MECH-089 cross-rate integration)
  - BetaGate (MECH-090 policy propagation gate)

V3 REE loop:
  1. SENSE     — split observation into body/world channels
  2. E1 TICK   — update E1 (every step); push z_self, z_world to ThetaBuffer
  3. E2 TICK   — update z_self via E2 motor-sensory model (every N_e2 steps)
  4. E3 TICK   — E3 consumes theta-cycle summary; proposes via HippocampalModule
                 (every N_e3 steps, or on phase reset)
  5. SELECT    — E3 selects trajectory; BetaGate controls policy propagation
  6. ACT       — execute first action from selected trajectory
  7. RESIDUE   — accumulate world_delta into ResidueField (MECH-094 gated)
  8. REPLAY    — quiescent E3 cycles trigger HippocampalModule.replay() (MECH-092)
  9. OFFLINE   — periodic sleep-like residue integration

Multi-rate execution:
  The clock fires e1_tick, e2_tick, e3_tick based on configured rates.
  E3 only fires at its tick; between ticks, the held action from the last
  E3 selection is repeated (MECH-057a: action-loop gate).
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import REEConfig
from ree_core.latent.stack import LatentStack, LatentState
from ree_core.goal import GoalState
from ree_core.latent.theta_buffer import ThetaBuffer
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.hippocampal.module import HippocampalModule
from ree_core.heartbeat.clock import MultiRateClock
from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.neuromodulation.serotonin import SerotoninModule
from ree_core.predictors.e2_harm_a import E2HarmAConfig, E2HarmAForward
from ree_core.cingulate import (
    AICAnalog,
    AICConfig,
    DACCAdaptiveControl,
    DACCConfig,
    DACCtoE3Adapter,
    PACCAnalog,
    PACCConfig,
    PCCAnalog,
    PCCConfig,
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)
from ree_core.latent.stack import HarmForwardTrunk
from ree_core.residue.field import (
    VALENCE_WANTING,
    VALENCE_LIKING,
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_SURPRISE,
)


@dataclass
class AgentState:
    """Complete state of the REE-v3 agent at a timestep."""
    latent_state: LatentState
    precision: float
    running_variance: float
    step: int
    harm_accumulated: float
    is_committed: bool
    beta_elevated: bool
    e3_steps_per_tick: int
    # MECH-203/204: serotonergic state (None when disabled)
    serotonin_state: Optional[Dict[str, Any]] = None


class REEAgent(nn.Module):
    """
    Reflective-Ethical Engine Agent — V3.

    Implements all V3 architectural invariants:
    1. Ethical cost is PERSISTENT (residue cannot be erased)
    2. Residue tracks WORLD-DELTA (z_world changes), not body-state changes
    3. E3 is the harm evaluator (harm_eval on z_world)
    4. E2 trains on motor-sensory error (z_self). NOT harm/goal error.
    5. HippocampalModule navigates action-object space O (SD-004)
    6. All replay/simulation carries hypothesis_tag=True (MECH-094)
    7. Precision is E3-derived (ARC-016). NOT hardcoded.
    8. E3 consumes theta-cycle summaries, never raw E1 output (MECH-089)
    9. Beta state gates policy propagation, not E3 internal updating (MECH-090)
    """

    def __init__(self, config: REEConfig):
        super().__init__()
        self.config = config

        # Core components
        self.latent_stack = LatentStack(config.latent)
        self.e1 = E1DeepPredictor(config.e1)
        self.e2 = E2FastPredictor(config.e2)
        self.residue_field = ResidueField(config.residue)
        self.e3 = E3TrajectorySelector(config.e3, self.residue_field)
        self.hippocampal = HippocampalModule(config.hippocampal, self.e2, self.residue_field)

        # SD-006: multi-rate clock
        self.clock = MultiRateClock(
            e1_steps_per_tick=config.heartbeat.e1_steps_per_tick,
            e2_steps_per_tick=config.heartbeat.e2_steps_per_tick,
            e3_steps_per_tick=config.heartbeat.e3_steps_per_tick,
            theta_buffer_size=config.heartbeat.theta_buffer_size,
            beta_rate_min_steps=config.heartbeat.beta_rate_min_steps,
            beta_rate_max_steps=config.heartbeat.beta_rate_max_steps,
            beta_magnitude_scale=config.heartbeat.beta_magnitude_scale,
            breath_period=config.heartbeat.breath_period,
            sweep_amplitude=config.heartbeat.breath_sweep_amplitude,
            sweep_duration=config.heartbeat.breath_sweep_duration,
        )

        # MECH-089: ThetaBuffer for cross-rate integration
        self.theta_buffer = ThetaBuffer(
            self_dim=config.latent.self_dim,
            world_dim=config.latent.world_dim,
            buffer_size=config.heartbeat.theta_buffer_size,
        )

        # MECH-090: BetaGate for policy propagation
        self.beta_gate = BetaGate()

        # MECH-203/204: serotonergic neuromodulation (sleep + benefit-salience)
        self.serotonin = SerotoninModule(config.serotonin)

        # MECH-258: E2_harm_a affective-pain forward model (SD-032b prerequisite).
        # ARC-058 shared-trunk path: if use_shared_harm_trunk, construct a single
        # HarmForwardTrunk and pass into E2HarmAForward. (E2HarmSForward in its
        # current form owns its trunk internally; the competing-claim experiment
        # builds E2HarmSForward via the same trunk at the experiment script level.)
        self.harm_forward_trunk: Optional[HarmForwardTrunk] = None
        self.e2_harm_a: Optional[E2HarmAForward] = None
        if getattr(config, "use_e2_harm_a", False):
            z_harm_a_dim = config.latent.z_harm_a_dim
            action_dim = config.e2.action_dim
            harm_a_cfg = E2HarmAConfig(
                z_harm_a_dim=z_harm_a_dim,
                action_dim=action_dim,
                learning_rate=config.e2_harm_a_lr,
            )
            shared_trunk = None
            if getattr(config, "use_shared_harm_trunk", False):
                # ARC-058: construct shared trunk. Uses z_harm_a_dim; experiments
                # that run the sensory stream through the same trunk must match
                # this dim via projection heads (see experiment scripts).
                self.harm_forward_trunk = HarmForwardTrunk(
                    z_harm_dim=z_harm_a_dim,
                    action_dim=action_dim,
                    hidden_dim=harm_a_cfg.hidden_dim,
                    action_enc_dim=harm_a_cfg.action_enc_dim,
                )
                shared_trunk = self.harm_forward_trunk
            self.e2_harm_a = E2HarmAForward(harm_a_cfg, shared_trunk=shared_trunk)

        # SD-032b: dACC/aMCC-analog adaptive control.
        self.dacc: Optional[DACCAdaptiveControl] = None
        self.dacc_adapter: Optional[DACCtoE3Adapter] = None
        if getattr(config, "use_dacc", False):
            dacc_cfg = DACCConfig(
                dacc_weight=config.dacc_weight,
                dacc_interaction_weight=config.dacc_interaction_weight,
                dacc_foraging_weight=config.dacc_foraging_weight,
                dacc_suppression_weight=config.dacc_suppression_weight,
                dacc_suppression_memory=config.dacc_suppression_memory,
                dacc_precision_scale=config.dacc_precision_scale,
                dacc_effort_cost=config.dacc_effort_cost,
                dacc_drive_coupling=config.dacc_drive_coupling,
            )
            self.dacc = DACCAdaptiveControl(dacc_cfg)
            # STOPGAP adapter -- still the score_bias source until SD-033
            # substrates consume operating_mode natively. With SD-032a active,
            # the adapter's bias may be scaled by the coordinator's e3_policy
            # write-gate (see select_action; gated by salience_apply_to_dacc_bias).
            self.dacc_adapter = DACCtoE3Adapter(dacc_cfg)

        # SD-032a: salience-network coordinator. Reads SD-032b dACC bundle +
        # drive_level + offline-mode flag; emits operating_mode soft vector +
        # mode_switch_trigger. Hosts MECH-261 write-gate registry.
        self.salience: Optional[SalienceCoordinator] = None
        if getattr(config, "use_salience_coordinator", False):
            sal_cfg = SalienceCoordinatorConfig(
                external_task_bias=config.salience_external_task_bias,
                softmax_temperature=config.salience_softmax_temperature,
                switch_threshold=config.salience_switch_threshold,
                stability_scaling=config.salience_stability_scaling,
            )
            sal_cfg.salience_weights = {
                "dacc_pe": config.salience_dacc_pe_weight,
                "dacc_foraging": config.salience_dacc_foraging_weight,
                "aic_salience": 1.0,
            }
            self.salience = SalienceCoordinator(sal_cfg)
        # Cache of last coordinator tick output (for diagnostics / experiments).
        self._salience_last_tick: Optional[Dict[str, object]] = None

        # SD-032c: AIC-analog interoceptive-salience / urgency module.
        # Emits aic_salience -> salience coordinator and harm_s_gain ->
        # descending z_harm_s attenuation path (subsumes SD-021 raw beta_gate
        # check). Reads operating_mode from the coordinator's previous tick
        # (one-step lag is biologically plausible; AIC->dACC->SAL is a circuit,
        # not instantaneous).
        self.aic: Optional[AICAnalog] = None
        if getattr(config, "use_aic_analog", False):
            aic_cfg = AICConfig(
                baseline_alpha=config.aic_baseline_alpha,
                drive_coupling=config.aic_drive_coupling,
                urgency_threshold=config.aic_urgency_threshold,
                base_attenuation=config.aic_base_attenuation,
                drive_protect_weight=config.aic_drive_protect_weight,
                extra_weight=config.aic_extra_weight,
            )
            self.aic = AICAnalog(aic_cfg)
        # Cache of last AIC tick output (for diagnostics / experiments).
        self._aic_last_tick: Optional[Dict[str, float]] = None

        # SD-032d: PCC-analog metastability scalar. Emits pcc_stability ->
        # SalienceCoordinator (multiplied into MECH-259 effective threshold).
        # Non-trainable arithmetic over success EMA + drive_level + steps-
        # since-last-offline-phase. Coordinates within-session (MECH-092) and
        # cross-session (INV-049) offline phases via enter_offline_mode().
        self.pcc: Optional[PCCAnalog] = None
        if getattr(config, "use_pcc_analog", False):
            pcc_cfg = PCCConfig(
                success_alpha=config.pcc_success_alpha,
                success_weight=config.pcc_success_weight,
                fatigue_weight=config.pcc_fatigue_weight,
                offline_recency_window=config.pcc_offline_recency_window,
                offline_weight=config.pcc_offline_weight,
                stability_baseline=config.pcc_stability_baseline,
            )
            self.pcc = PCCAnalog(pcc_cfg)
        # Cache of last PCC tick output (for diagnostics / experiments).
        self._pcc_last_tick: Optional[Dict[str, float]] = None

        # SD-032e: pACC-analog slow autonomic write-back. Accumulates
        # tanh-normalised z_harm_a magnitude into a bounded drive_bias that
        # shifts the effective drive_level passed into GoalState.update(),
        # SalienceCoordinator.tick(), SD-032c AIC, and SD-032d PCC. Gated by
        # coordinator.write_gate("autonomic") so the write is mode-conditioned
        # per MECH-261 (active in external_task, attenuated in planning /
        # replay / offline). Architectural path for chronic-pain-like
        # sensitisation (Baliki 2012). See scoping lit-pull synthesis.
        self.pacc: Optional[PACCAnalog] = None
        if getattr(config, "use_pacc_analog", False):
            pacc_cfg = PACCConfig(
                drive_alpha=config.pacc_drive_alpha,
                drive_scale=config.pacc_drive_scale,
                drive_bias_cap=config.pacc_drive_bias_cap,
                z_harm_a_min=config.pacc_z_harm_a_min,
                offline_decay=config.pacc_offline_decay,
            )
            self.pacc = PACCAnalog(pacc_cfg)
        # Cache of last PACC tick output (for diagnostics / experiments).
        self._pacc_last_tick: Optional[Dict[str, float]] = None

        # Observation encoders (maps raw body/world obs to latent input)
        # Body encoder: body_obs_dim → latent input for LatentStack
        self.body_obs_encoder = nn.Sequential(
            nn.Linear(config.latent.body_obs_dim, config.latent.body_obs_dim),
            nn.ReLU(),
        )
        self.world_obs_encoder = nn.Sequential(
            nn.Linear(config.latent.world_obs_dim, config.latent.world_obs_dim),
            nn.ReLU(),
        )

        # State tracking
        self._current_latent: Optional[LatentState] = None
        self._step_count: int = 0
        self._harm_this_episode: float = 0.0

        # Experience buffers for training
        self._self_experience_buffer: List[torch.Tensor] = []   # z_self history
        self._world_experience_buffer: List[torch.Tensor] = []  # z_world history
        self._e2_transition_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # MECH-057a: cached E3 candidates for action-loop gate
        self._committed_candidates: Optional[List[Trajectory]] = None
        # Last selected action (held between E3 ticks)
        self._last_action: Optional[torch.Tensor] = None
        # MECH-090: step index within committed trajectory (Layer 1 trajectory stepping).
        # Incremented each committed step so a0->a1->a2->... is executed in sequence.
        self._committed_step_idx: int = 0

        # SD-016: cached frontal cue signals (updated each E1 tick).
        # None when sd016_enabled=False (all existing experiments unaffected).
        self._cue_action_bias:    Optional[torch.Tensor] = None
        self._cue_terrain_weight: Optional[torch.Tensor] = None

        self.device = torch.device(config.device)

        # MECH-216: cached schema salience from E1 readout head
        self._schema_salience: Optional[torch.Tensor] = None

        # MECH-205: surprise-gated replay PE tracking
        self._pe_ema: float = 0.0  # EMA of prediction error magnitude
        self._pe_ema_alpha: float = config.pe_ema_alpha  # from config (default 0.02)
        self._surprise_write_count: int = 0  # diagnostic counter

        # SD-020: harm PE tracker for affective surprise target.
        # Running EMA of observed harm (expected harm estimate).
        # PE = |actual_harm_obs - _harm_obs_ema| used as z_harm_a training target.
        self._harm_obs_ema: float = 0.0

        # MECH-258 / SD-032b: previous-step z_harm_a for E2_harm_a rollout
        # and previous-step predicted z_harm_a for dACC PE computation.
        self._harm_a_prev: Optional[torch.Tensor] = None
        self._harm_a_pred_prev: Optional[torch.Tensor] = None
        # Diagnostic: last dACC bundle (for experiments).
        self._dacc_last_bundle: Optional[Dict[str, Any]] = None
        self._dacc_last_bias: Optional[torch.Tensor] = None

        # MECH-165: episode trajectory recording for exploration buffer
        self._episode_world_states: List[torch.Tensor] = []
        self._episode_self_states: List[torch.Tensor] = []
        self._episode_actions: List[torch.Tensor] = []

        # MECH-165: pass config to hippocampal for buffer sizing
        if self.config.replay_diversity_enabled:
            self.hippocampal._exploration_buffer_maxlen = self.config.exploration_buffer_len
            self.hippocampal._reverse_fraction = self.config.reverse_replay_fraction
            self.hippocampal._random_fraction = self.config.random_replay_fraction

        # MECH-112/116: persistent goal representation
        self.goal_state: Optional[GoalState] = None
        _goal_cfg = getattr(self.config, "goal", None)
        if _goal_cfg is not None and _goal_cfg.z_goal_enabled:
            self.goal_state = GoalState(_goal_cfg, self.device)

    @classmethod
    def from_config(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        self_dim: int = 32,
        world_dim: int = 32,
        **kwargs,
    ) -> "REEAgent":
        config = REEConfig.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            self_dim=self_dim,
            world_dim=world_dim,
        )
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config)

    def reset(self) -> None:
        """Reset agent for a new episode. Does NOT reset residue (invariant)."""
        # MECH-165: flush waking trajectory to exploration buffer before reset
        self._flush_exploration_episode()

        self._current_latent = self.latent_stack.init_state(
            batch_size=1, device=self.device
        )
        self.e1.reset_hidden_state()
        self._step_count = 0
        self._harm_this_episode = 0.0
        self._committed_candidates = None
        self._last_action = None
        self._committed_step_idx = 0
        self._cue_action_bias    = None
        self._cue_terrain_weight = None
        self.clock.reset()
        self.theta_buffer.reset()
        self.beta_gate.reset()
        self.serotonin.reset()
        self._pe_ema = 0.0
        # SD-032b: clear dACC state + previous-step harm_a cache.
        self._harm_a_prev = None
        self._harm_a_pred_prev = None
        self._dacc_last_bundle = None
        self._dacc_last_bias = None
        if self.dacc is not None:
            self.dacc.reset()
        # SD-032a: reset salience coordinator on episode boundary.
        if self.salience is not None:
            self.salience.reset()
        self._salience_last_tick = None

        # SD-032c: reset AIC-analog interoceptive baseline.
        if self.aic is not None:
            self.aic.reset()
        self._aic_last_tick = None

        # SD-032d: reset PCC-analog per-episode state. Note: does NOT reset
        # _steps_since_offline (cross-episode counter; only note_offline_entry
        # resets it -- a new episode starting does not constitute rest).
        if self.pcc is not None:
            self.pcc.reset()
        self._pcc_last_tick = None

        # SD-032e: reset PACC-analog diagnostics cache. Does NOT clear
        # _drive_bias -- SD-032e's architectural purpose is cross-episode
        # accumulation. Use non-zero pacc_offline_decay via offline entry
        # (or reinstantiate the module) for a hard reset.
        if self.pacc is not None:
            self.pacc.reset()
        self._pacc_last_tick = None

    def _record_exploration_state(self) -> None:
        """MECH-165: record current latent state for exploration trajectory."""
        if not self.config.replay_diversity_enabled:
            return
        if self._current_latent is None:
            return
        self._episode_world_states.append(
            self._current_latent.z_world.detach().clone()
        )
        self._episode_self_states.append(
            self._current_latent.z_self.detach().clone()
        )

    def _record_exploration_action(self, action: torch.Tensor) -> None:
        """MECH-165: record selected action for exploration trajectory."""
        if not self.config.replay_diversity_enabled:
            return
        self._episode_actions.append(action.detach().clone())

    def _flush_exploration_episode(self) -> None:
        """
        MECH-165: build Trajectory from accumulated episode data and
        store in HippocampalModule exploration buffer. Clears episode buffers.
        Minimum 5 steps required to form a useful trajectory.
        """
        if not self.config.replay_diversity_enabled:
            return
        min_steps = 5
        n_states = len(self._episode_world_states)
        n_actions = len(self._episode_actions)
        if n_states < min_steps or n_actions < min_steps:
            self._episode_world_states.clear()
            self._episode_self_states.clear()
            self._episode_actions.clear()
            return

        # Align: states[0..N], actions[0..N-1]
        # Trim to min(n_states, n_actions+1) for clean pairing
        n = min(n_states, n_actions + 1)
        states = self._episode_self_states[:n]
        world_states = self._episode_world_states[:n]
        actions_list = self._episode_actions[:n - 1]

        # Build actions tensor [batch, horizon, action_dim]
        actions_tensor = torch.stack(actions_list, dim=1)  # [batch, horizon, action_dim]

        traj = Trajectory(
            states=states,
            actions=actions_tensor,
            world_states=world_states,
            is_reverse=False,
        )
        self.hippocampal.record_exploration_trajectory(traj)

        self._episode_world_states.clear()
        self._episode_self_states.clear()
        self._episode_actions.clear()

    def sense(
        self,
        obs_body: torch.Tensor,
        obs_world: torch.Tensor,
        obs_harm: Optional[torch.Tensor] = None,
        obs_harm_a: Optional[torch.Tensor] = None,
        obs_harm_history: Optional[torch.Tensor] = None,
    ) -> LatentState:
        """
        SENSE + UPDATE step: encode split observation -> update latent state.

        Args:
            obs_body:   [batch, body_obs_dim] proprioceptive channels
            obs_world:  [batch, world_obs_dim] exteroceptive channels
            obs_harm:   SD-010 nociceptive channels [batch, harm_obs_dim] or None.
                        When provided and use_harm_stream=True in config, routes
                        through HarmEncoder to z_harm (bypasses reafference correction).
            obs_harm_a: SD-011 affective-motivational harm channels
                        [batch, harm_obs_a_dim] or None. EMA-accumulated proximity
                        signal from environment. When provided and
                        use_affective_harm_stream=True, routes through
                        AffectiveHarmEncoder to z_harm_a.
            obs_harm_history: SD-011 second source [batch, harm_history_len] or None.
                        Rolling window of past harm_exposure scalars. Concatenated
                        with harm_obs_a as AffectiveHarmEncoder input when
                        harm_history_len > 0.

        Returns:
            Updated LatentState
        """
        if obs_body.dim() == 1:
            obs_body  = obs_body.unsqueeze(0)
            obs_world = obs_world.unsqueeze(0)
        obs_body  = obs_body.to(self.device).float()
        obs_world = obs_world.to(self.device).float()

        enc_body  = self.body_obs_encoder(obs_body)
        enc_world = self.world_obs_encoder(obs_world)

        # Concatenate for LatentStack (which splits internally)
        enc_combined = torch.cat([enc_body, enc_world], dim=-1)
        # SD-007: pass last action as prev_action for reafference correction.
        # _last_action is None on the first step (no correction applied).
        # Q-007: pass volatility estimate (var(rv) over sliding window) to
        # z_beta encoder.  Raw rv converges to near-zero in both stable and
        # volatile environments once E2 adapts.  var(rv) captures how much rv
        # *fluctuates*: high in volatile envs (rv spikes on hazard moves),
        # low in stable envs (rv is flat).  LC-NE tonic firing analog.
        vol_signal = None
        if self.config.latent.volatility_signal_dim > 0:
            vol_signal = self.e3.volatility_estimate
        new_latent = self.latent_stack.encode(
            enc_combined, self._current_latent,
            prev_action=self._last_action,
            harm_obs=obs_harm,       # SD-010: nociceptive stream (None = disabled)
            harm_obs_a=obs_harm_a,   # SD-011: affective harm stream (None = disabled)
            harm_history=obs_harm_history,  # SD-011 second source (None = disabled)
            volatility_signal=vol_signal,
        )
        # SD-032c: tick the AIC-analog interoceptive-salience module.
        # Reads z_harm_a_norm + drive_level + beta_gate_elevated + the
        # coordinator's previous operating_mode readout. Produces aic_salience
        # (fed to the coordinator below in select_action) and harm_s_gain
        # (descending attenuation multiplier that subsumes the raw SD-021
        # beta_gate check).
        if self.aic is not None:
            if new_latent.z_harm_a is not None:
                aic_z_norm = float(new_latent.z_harm_a.norm().item())
            else:
                aic_z_norm = 0.0
            aic_drive = 0.0
            if self.goal_state is not None:
                aic_drive = float(getattr(self.goal_state, "_last_drive_level", 0.0))
            # SD-032e: if pACC is active, AIC sees the effective (sensitised)
            # drive_level with one-step lag -- the pACC tick in select_action()
            # produced drive_bias from the previous step's z_harm_a, which is
            # the correct causal ordering (pACC accumulates first, then AIC
            # reads the resulting baseline).
            if self.pacc is not None:
                aic_drive = self.pacc.effective_drive(aic_drive)
            aic_op_mode = None
            if self.salience is not None:
                aic_op_mode = self.salience.operating_mode
            self._aic_last_tick = self.aic.tick(
                z_harm_a_norm=aic_z_norm,
                drive_level=aic_drive,
                beta_gate_elevated=self.beta_gate.is_elevated,
                operating_mode=aic_op_mode,
            )

        # SD-021: descending pain modulation (commitment-gated sensory harm attenuation).
        # When harm_descending_mod_enabled=True, attenuate z_harm (sensory-discriminative
        # stream) before it reaches E3 / E2_harm_s. z_harm_a is NOT attenuated: affective
        # load (C-fiber) persists regardless of commitment -- you can tolerate expected
        # pain but it still matters motivationally.
        #
        # Two gating paths (selected by use_aic_analog):
        #   use_aic_analog=True  -- multiplier is self.aic.harm_s_gain (drive- and
        #     operating-mode-gated). This is the SD-032c-subsumed path -- resolves the
        #     EXQ-325a FAIL by making the descending branch a genuinely different
        #     function of state (not a redundant beta_gate flag).
        #   use_aic_analog=False -- legacy path: if beta_gate is elevated, multiplier is
        #     config.descending_attenuation_factor; otherwise 1.0.
        #
        # Biological basis: pgACC / AIC -> PAG -> RVM descending inhibitory pathway
        # provides endogenous analgesia during committed escape/approach (Basbaum 1984,
        # Keltner 2006). The AIC-gated routing is the biologically correct one
        # (Craig 2009 AIC as interoceptive-salience hub with descending efferents);
        # the legacy raw-beta_gate routing is retained for backward compatibility.
        # MECH-094: applies to waking observation stream (not replay content).
        if (
            getattr(self.config, "harm_descending_mod_enabled", False)
            and new_latent.z_harm is not None
        ):
            if self.aic is not None:
                attn = float(self.aic.harm_s_gain)
                if attn < 1.0:
                    new_latent.z_harm = new_latent.z_harm * attn
            elif self.beta_gate.is_elevated:
                attn = getattr(self.config, "descending_attenuation_factor", 0.5)
                new_latent.z_harm = new_latent.z_harm * attn

        # SD-014 h-component: write post-attenuation z_harm_s norm to VALENCE_HARM_DISCRIMINATIVE.
        # Using the post-attenuation value means SD-021 commitment gating automatically produces
        # a stale (attenuated) h at committed-state nodes -- exactly the analgesia-as-underestimated-h
        # signature needed for the SD-021/SD-014 cross-connection (surprise spike post-execution).
        if (
            getattr(self.config, "valence_harm_enabled", False)
            and new_latent.z_harm is not None
            and new_latent.z_world is not None
            and hasattr(self.residue_field, "update_valence")
        ):
            h_val = float(new_latent.z_harm.norm().item())
            self.residue_field.update_valence(
                new_latent.z_world,
                component=VALENCE_HARM_DISCRIMINATIVE,
                value=h_val,
                hypothesis_tag=False,
            )

        # Detach before storing: prevents EMA from linking computational graphs
        # across time steps. Without detach, optimizer.step() modifies weights
        # in-place, invalidating the old graph's version -- causing RuntimeError
        # "modified by an inplace operation" on the next backward() call.
        self._current_latent = new_latent.detach()
        # MECH-165: record state for exploration trajectory
        self._record_exploration_state()

        # MECH-258 / SD-032b: cache current z_harm_a so the next step's dACC
        # computation has access to both z_harm_a_prev (input to E2_harm_a) and
        # z_harm_a_current (realised target). Detached to avoid graph retention.
        if new_latent.z_harm_a is not None:
            self._harm_a_prev = new_latent.z_harm_a.detach().clone()
        return new_latent

    def sense_flat(self, observation: torch.Tensor) -> LatentState:
        """
        SENSE step with flat observation (backward compat / pre-SD-005 wiring).

        Splits observation into [body, world] based on configured dims.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device).float()
        body_dim = self.config.latent.body_obs_dim
        obs_body  = observation[:, :body_dim]
        obs_world = observation[:, body_dim:]
        return self.sense(obs_body, obs_world)

    def _e1_tick(self, latent_state: LatentState) -> torch.Tensor:
        """
        E1 tick: run E1 prediction and push to ThetaBuffer (MECH-089).

        Returns E1 world-domain prior for HippocampalModule (SD-002).
        """
        total_state = torch.cat([latent_state.z_self, latent_state.z_world], dim=-1)

        # Store in experience buffers
        self._self_experience_buffer.append(latent_state.z_self.detach().clone())
        self._world_experience_buffer.append(latent_state.z_world.detach().clone())
        for buf in [self._self_experience_buffer, self._world_experience_buffer]:
            if len(buf) > 1000:
                del buf[:-1000]

        # Run E1 for prior generation
        _z_goal_input = None
        _goal_cfg = getattr(self.config, "goal", None)
        if (self.goal_state is not None
                and _goal_cfg is not None
                and _goal_cfg.e1_goal_conditioned
                and self.goal_state.is_active()):
            _z_goal_input = self.goal_state.z_goal
        _, e1_prior = self.e1(total_state, z_goal=_z_goal_input)

        # MECH-089: push z_self, z_world estimates to ThetaBuffer
        self.theta_buffer.update(latent_state.z_world, latent_state.z_self)

        # MECH-093: update E3 rate from z_beta magnitude
        self.clock.update_e3_rate_from_beta(latent_state.z_beta)

        # MECH-216: cache schema salience from E1 readout head.
        schema_sal = self.e1.get_schema_salience()
        self._schema_salience = schema_sal.detach() if schema_sal is not None else None

        # SD-016 (MECH-150/151/152): frontal cue-indexed integration.
        # Extract action_bias and terrain_weight from z_world-only ContextMemory query.
        # Detached: cue signals are modulation inputs, not part of current-step gradient graph.
        # Cached until next E1 tick (same theta-cycle rate as generate_prior).
        if hasattr(self.e1, 'world_query_proj'):
            action_bias, terrain_weight = self.e1.extract_cue_context(
                latent_state.z_world.detach()
            )
            self._cue_action_bias    = action_bias.detach()
            self._cue_terrain_weight = terrain_weight.detach()
        else:
            self._cue_action_bias    = None
            self._cue_terrain_weight = None

        return e1_prior

    def _e3_tick(
        self,
        latent_state: LatentState,
        e1_prior: torch.Tensor,
        num_candidates: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        E3 tick: propose trajectories via HippocampalModule, select via E3.

        E3 receives theta-cycle summary (MECH-089), not raw z_world.

        Returns candidate trajectories.
        """
        # MECH-089: E3 consumes theta-cycle summary
        z_world_for_e3 = self.theta_buffer.summary()  # theta-averaged z_world

        # HippocampalModule proposes in action-object space (SD-004).
        # SD-016 (MECH-151): pass cached action_bias so each action_object()
        # call in the CEM loop is contextually biased by z_world cue retrieval.
        candidates = self.hippocampal.propose_trajectories(
            z_world=z_world_for_e3,
            z_self=latent_state.z_self,
            num_candidates=num_candidates,
            e1_prior=e1_prior,
            action_bias=self._cue_action_bias,
        )
        self._committed_candidates = candidates

        # ARC-028 / MECH-105: hippocampal completion signal -> BetaGate release.
        # compute_completion_signal() scores all proposals; high-quality trajectory
        # found -> dopamine analog -> beta drops -> gate opens (Lisman & Grace 2005).
        # MECH-090 bistable: release triggered by completion, not by variance re-eval.
        if self.config.heartbeat.beta_gate_bistable and self.beta_gate.is_elevated:
            completion = self.hippocampal.compute_completion_signal(candidates)
            self.beta_gate.receive_hippocampal_completion(completion)

        return candidates

    def generate_trajectories(
        self,
        latent_state: LatentState,
        e1_prior: torch.Tensor,
        ticks: dict,
        num_candidates: Optional[int] = None,
        sequence_in_progress: bool = False,
    ) -> List[Trajectory]:
        """
        GENERATE step: propose candidate trajectories.

        Multi-rate: only re-generate when e3_tick fires.
        Between ticks: return cached candidates (MECH-057a gate).
        """
        # If E3 hasn't ticked and we have cached candidates, use them
        if (
            not ticks["e3_tick"]
            and self._committed_candidates is not None
        ):
            return self._committed_candidates

        # If gate enabled and sequence in progress and no new tick, hold
        if (
            self.config.action_loop_gate_enabled
            and sequence_in_progress
            and not ticks["e3_tick"]
            and self._committed_candidates is not None
        ):
            return self._committed_candidates

        return self._e3_tick(latent_state, e1_prior, num_candidates)

    def select_action(
        self,
        candidates: List[Trajectory],
        ticks: dict,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        SELECT step: E3 selects trajectory; BetaGate controls propagation.

        If E3 hasn't ticked, return the held action (MECH-090).

        Layer 1 (MECH-090 trajectory stepping): when committed, step through
        a0->a1->a2->... using _committed_step_idx instead of repeating a0.

        Layer 2 (MECH-091 urgency interrupt): when beta is elevated and z_harm_a
        norm exceeds urgency_interrupt_threshold, abort commitment and re-select.
        """
        # SD-011: extract z_harm_a for E3 urgency gating and ethical cost amplification.
        z_harm_a = None
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a = self._current_latent.z_harm_a

        # MECH-091: urgency interrupt -- phase-reset commitment on high harm signal.
        # When beta is elevated (committed) and affective harm load is extreme,
        # abort the committed trajectory and fall through to fresh E3 selection.
        if self.beta_gate.is_elevated and z_harm_a is not None:
            urgency_threshold = getattr(
                self.config.e3, "urgency_interrupt_threshold", 0.8
            )
            if float(z_harm_a.norm().item()) > urgency_threshold:
                self.beta_gate.release()
                self._committed_step_idx = 0

        if not ticks["e3_tick"] and self._last_action is not None:
            # Between E3 ticks: step through committed trajectory (Layer 1) or hold.
            if self.beta_gate.is_elevated and self.e3._committed_trajectory is not None:
                traj = self.e3._committed_trajectory
                horizon = traj.actions.shape[1]
                step_idx = min(self._committed_step_idx, horizon - 1)
                action = traj.actions[:, step_idx, :]
                self._committed_step_idx += 1
            else:
                action = self._last_action
            # MECH-165: record held/stepped action for exploration trajectory (every step)
            self._record_exploration_action(action)
            self._last_action = action
            return action

        # SD-016 (MECH-152): pass cached terrain_weight so harm/goal scoring
        # precision reflects current z_world cue context.
        # MECH-108: pass BreathOscillator sweep reduction during sweep phase.
        # Creates periodic uncommitted windows even when running_variance has
        # converged below base commit_threshold after training.
        sweep_reduction = self.clock.sweep_amplitude if self.clock.sweep_active else 0.0
        # MECH-188: PFC top-down z_goal injection (EXQ-253).
        # When z_goal_inject > 0, apply a constant norm floor to z_goal for
        # action selection ONLY. Does not modify the persistent attractor.
        # Simulates DRN-mPFC top-down goal persistence bypassing terrain seeding.
        _goal_inject = getattr(
            getattr(self.config, "goal", None), "z_goal_inject", 0.0
        )
        _goal_state_for_select = self.goal_state
        if _goal_inject > 0.0 and self.goal_state is not None:
            _goal_state_for_select = self.goal_state.with_injection(_goal_inject)

        # SD-032e: tick pACC-analog before dACC / salience consumers so that
        # effective_drive_level (base + drive_bias) is available for all SD-032
        # downstream modules. Gate on coordinator's previous-tick "autonomic"
        # write_gate (one-step lag; the pACC->autonomic->sensitisation circuit
        # is slow enough that instantaneous coupling is not required). When
        # salience is disabled the gate defaults to 1.0 so the drift remains
        # observable under ablation. MECH-094: hypothesis_tag=False here
        # (waking action selection); simulation/replay paths do not route
        # through select_action().
        if self.pacc is not None:
            _zha_norm = 0.0
            if z_harm_a is not None:
                _zha_norm = float(z_harm_a.norm().item())
            _auto_gate = 1.0
            if self.salience is not None:
                _auto_gate = float(self.salience.write_gate("autonomic"))
            self._pacc_last_tick = self.pacc.tick(
                z_harm_a_norm=_zha_norm,
                write_gate=_auto_gate,
                hypothesis_tag=False,
            )

        # SD-032e: resolve the effective drive_level once, used by dACC,
        # salience coordinator, AIC, PCC. Falls back to base when pACC is off.
        _base_drive_level = float(
            getattr(self.goal_state, "_last_drive_level", 0.0)
        ) if self.goal_state is not None else 0.0
        if self.pacc is not None:
            _effective_drive_level = self.pacc.effective_drive(_base_drive_level)
        else:
            _effective_drive_level = _base_drive_level

        # SD-032b: compute dACC bundle + stopgap-adapter score bias before E3.select.
        # The bundle reads the (precision-weighted) affective-pain PE from the last
        # forward-model prediction, and mixes in per-candidate payoff/effort terms.
        dacc_score_bias: Optional[torch.Tensor] = None
        if self.dacc is not None and z_harm_a is not None:
            # Per-candidate payoff proxy: negative of E3 running candidate score if
            # available (lower score = better, so payoff = -score). Falls back to
            # a zero payoff vector until E3 has run at least once.
            K = len(candidates)
            if self.e3.last_scores is not None and self.e3.last_scores.numel() == K:
                payoffs = -self.e3.last_scores.detach().float()
            else:
                payoffs = torch.zeros(K, device=self.device)
            # Per-candidate effort proxy: trajectory length / horizon. Future
            # refinement (Croxson): harm-forward rollout cost.
            effort = torch.tensor(
                [float(c.actions.shape[1]) for c in candidates],
                dtype=payoffs.dtype,
                device=self.device,
            )
            # Action-class tags for MECH-260 suppression: argmax of first action.
            action_classes = [
                int(c.actions[0, 0].argmax().item()) for c in candidates
            ]
            # GoalState does not persist drive_level (it is passed per-update);
            # SD-032e: use effective drive (base + pACC bias) so dACC sees the
            # chronic-pain-sensitised drive regime.
            drive_level = _effective_drive_level
            bundle = self.dacc(
                z_harm_a=z_harm_a.squeeze(0) if z_harm_a.dim() > 1 else z_harm_a,
                z_harm_a_pred=self._harm_a_pred_prev,
                candidate_payoffs=payoffs,
                candidate_effort=effort,
                candidate_action_classes=action_classes,
                precision=float(self.e3.current_precision),
                drive_level=drive_level,
            )
            self._dacc_last_bundle = bundle
            if self.dacc_adapter is not None:
                dacc_score_bias = self.dacc_adapter(bundle)
                self._dacc_last_bias = dacc_score_bias.detach().clone()

        # SD-032a: tick the salience-network coordinator. Aggregates the dACC
        # bundle (live), drive_level (live), and offline-mode flag (proxy for
        # SD-032d stability) into the operating_mode soft vector and the
        # MECH-259 mode_switch_trigger. SD-032c/d/e signal slots remain at zero
        # until those subdivisions land.
        if self.salience is not None:
            # SD-032e: use effective drive (base + pACC bias) so salience,
            # AIC, and PCC all see the sensitised drive regime coherently.
            sal_drive = _effective_drive_level
            sal_offline = bool(getattr(self.e1, "_offline_mode", False))
            sal_bundle = self._dacc_last_bundle  # may be None if dACC is off
            # SD-032c: inject AIC salience into the coordinator BEFORE tick so
            # the MECH-259 urgency-trigger source is live on this cycle.
            if self.aic is not None:
                self.salience.update_signal(
                    "aic_salience", float(self.aic.aic_salience)
                )
            # SD-032d: tick PCC-analog and inject pcc_stability into the
            # coordinator BEFORE tick so MECH-259 effective_threshold is
            # modulated on this cycle. High stability -> harder to switch;
            # low stability (depleted / no recent rest / failing) -> easier.
            if self.pcc is not None:
                self._pcc_last_tick = self.pcc.tick(drive_level=sal_drive)
                self.salience.update_signal(
                    "pcc_stability", float(self.pcc.pcc_stability)
                )
            self._salience_last_tick = self.salience.tick(
                dacc_bundle=sal_bundle,
                drive_level=sal_drive,
                is_offline=sal_offline,
            )
            # Optional: scale dACC score_bias by the e3_policy write-gate, so
            # that during internal_replay the dACC bias is suppressed near zero
            # (per MECH-261 table). Default-off (backward compatible).
            if (
                self.config.salience_apply_to_dacc_bias
                and dacc_score_bias is not None
            ):
                e3_gate = self.salience.write_gate("e3_policy")
                dacc_score_bias = dacc_score_bias * float(e3_gate)
                self._dacc_last_bias = dacc_score_bias.detach().clone()

        result = self.e3.select(
            candidates, temperature,
            goal_state=_goal_state_for_select,
            terrain_weight=self._cue_terrain_weight,
            sweep_threshold_reduction=sweep_reduction,
            z_harm_a=z_harm_a,
            score_bias=dacc_score_bias,
        )
        action = result.selected_action

        # MECH-090: gate policy propagation based on beta state.
        bistable = self.config.heartbeat.beta_gate_bistable
        if bistable:
            # Bistable latch: only elevate on commit ENTRY (not every tick).
            # Release is triggered by hippocampal completion signal in _e3_tick(),
            # not by variance re-evaluation. This prevents flickering when variance
            # hovers near the commit threshold.
            if result.committed and not self.beta_gate.is_elevated:
                self.beta_gate.elevate()
                self._committed_step_idx = 0  # reset step counter on new commitment
            # If not committed and beta already released: no-op (already open).
            # If committed and beta already elevated: no-op (stay latched).
        else:
            # Legacy behavior (backward compat): re-evaluate every E3 tick.
            if result.committed:
                if not self.beta_gate.is_elevated:
                    self._committed_step_idx = 0  # reset on new commitment
                self.beta_gate.elevate()
            else:
                if self.beta_gate.is_elevated:
                    self._committed_step_idx = 0  # reset when gate opens
                self.beta_gate.release()

        propagated = self.beta_gate.propagate(action)
        if propagated is None:
            # Beta elevated: step through committed trajectory (Layer 1).
            if self.e3._committed_trajectory is not None:
                traj = self.e3._committed_trajectory
                horizon = traj.actions.shape[1]
                step_idx = min(self._committed_step_idx, horizon - 1)
                action = traj.actions[:, step_idx, :]
                self._committed_step_idx += 1
            elif self._last_action is not None:
                action = self._last_action

        self._last_action = action
        # MECH-165: record action for exploration trajectory
        self._record_exploration_action(action)

        # MECH-258 / SD-032b: roll E2_harm_a forward for the chosen action, so
        # the next step's dACC tick has a prediction to compute PE against.
        if (
            self.e2_harm_a is not None
            and self._harm_a_prev is not None
            and action is not None
        ):
            with torch.no_grad():
                a_in = action if action.dim() > 1 else action.unsqueeze(0)
                z_in = self._harm_a_prev
                if z_in.dim() == 1:
                    z_in = z_in.unsqueeze(0)
                pred = self.e2_harm_a(z_in, a_in)
                self._harm_a_pred_prev = pred.squeeze(0).detach().clone()

        # MECH-260: record chosen action class in dACC recency history.
        if self.dacc is not None and action is not None:
            try:
                a_row = action[0] if action.dim() > 1 else action
                self.dacc.record_action(int(a_row.argmax().item()))
            except Exception:
                # One-hot discretisation fallback: hash raw action. Silent to
                # preserve backward-compatible select_action control flow.
                pass

        return action

    def act(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> torch.Tensor:
        """
        Complete V3 REE action loop.

        Accepts flat observation — auto-splits into body/world channels.

        Returns action tensor.
        """
        # SENSE + UPDATE
        latent_state = self.sense_flat(observation)

        # Advance clock
        ticks = self.clock.advance()

        # E1 tick
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )

        # GENERATE
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )

        # SELECT (multi-rate + beta gate)
        action = self.select_action(candidates, ticks, temperature)

        # MECH-092: replay on quiescent E3 cycles
        if ticks.get("e3_quiescent", False):
            self._do_replay(latent_state)

        self._step_count += 1
        return action

    def act_with_split_obs(
        self,
        obs_body: torch.Tensor,
        obs_world: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> torch.Tensor:
        """Act with pre-split observations (preferred V3 interface)."""
        latent_state = self.sense(obs_body, obs_world)
        ticks = self.clock.advance()
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )
        action = self.select_action(candidates, ticks, temperature)
        if ticks.get("e3_quiescent", False):
            self._do_replay(latent_state)
        self._step_count += 1
        return action

    def act_with_log_prob(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Action selection + log-prob for REINFORCE."""
        latent_state = self.sense_flat(observation)
        ticks = self.clock.advance()
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )
        z_harm_a = None
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a = self._current_latent.z_harm_a
        result = self.e3.select(candidates, temperature, z_harm_a=z_harm_a)
        self._last_action = result.selected_action
        # MECH-165: record action for exploration trajectory
        self._record_exploration_action(result.selected_action)
        self._step_count += 1
        return result.selected_action, result.log_prob

    def _do_replay(self, latent_state: LatentState) -> None:
        """
        MECH-092: SWR-equivalent replay on quiescent E3 cycles.

        All replay content carries hypothesis_tag=True — cannot produce residue.
        MECH-203: when serotonin is enabled, passes drive_state to replay()
        for valence-weighted start point selection (balanced consolidation).
        """
        recent = self.theta_buffer.recent
        if recent is None:
            return
        # MECH-203 + MECH-205: build drive_state for valence-weighted replay
        drive_state = None
        if self.serotonin.enabled or self.config.surprise_gated_replay:
            # Drive state = [wanting_weight, liking_weight, harm_weight, surprise_weight]
            t5ht = self.serotonin.tonic_5ht if self.serotonin.enabled else 0.0
            # MECH-205: surprise weight scales with recent PE magnitude
            surprise_weight = 0.3
            if self.config.surprise_gated_replay and self._pe_ema > 0:
                surprise_weight = min(1.0, self._pe_ema * 5.0)
            drive_state = torch.tensor(
                [t5ht, 0.5, 1.0 - t5ht, surprise_weight],
                device=self.device,
            )
        # MECH-165: use diverse replay scheduler when enabled
        if self.config.replay_diversity_enabled:
            replay_trajs = self.hippocampal.diverse_replay(
                recent, drive_state=drive_state, mode="auto",
            )
        else:
            replay_trajs = self.hippocampal.replay(recent, drive_state=drive_state)
        # hypothesis_tag=True: these trajectories cannot update residue
        # (MECH-094 -- enforced in ResidueField.accumulate)

    def update_residue(
        self,
        harm_signal: float,
        world_delta: Optional[float] = None,
        hypothesis_tag: bool = False,
        owned: bool = True,
    ) -> Dict[str, Any]:
        """
        RESIDUE step: accumulate residue at z_world location after action.

        V3 changes:
        - Residue accumulates on z_world (SD-005), not z_gamma
        - world_delta from SD-003 attribution pipeline scales accumulation
        - hypothesis_tag=True blocks accumulation (MECH-094)

        Args:
            harm_signal:   Harm from environment (negative = harm)
            world_delta:   Optional |z_world_actual - z_world_cf| from SD-003
            hypothesis_tag: If True, no residue (MECH-094)
            owned:         If False, no residue (not agent-attributed harm)
        """
        metrics: Dict[str, Any] = {}

        # ARC-016: update running variance on EVERY step so rv tracks world
        # prediction error continuously.  Previously gated on harm_signal < 0,
        # which meant rv only updated on harm steps -- too sparse for
        # volatility tracking (Q-007) and caused rv deadlock.
        if self._current_latent is not None:
            z_world = self._current_latent.z_world
            e3_metrics = self.e3.post_action_update(
                actual_z_world=z_world,
                harm_occurred=(harm_signal < 0),
            )
            metrics.update({f"e3_{k}": v for k, v in e3_metrics.items()})

            # MECH-205: populate VALENCE_SURPRISE on residue field
            if self.config.surprise_gated_replay:
                pe_val = e3_metrics.get("prediction_error")
                if pe_val is not None:
                    pe_mag = float(pe_val.detach())
                    self._pe_ema = (1 - self._pe_ema_alpha) * self._pe_ema + self._pe_ema_alpha * pe_mag
                    surprise = max(0.0, pe_mag - self._pe_ema)
                    # Gate: only write genuine surprises above threshold
                    if surprise > self.config.pe_surprise_threshold:
                        self.residue_field.update_valence(
                            z_world, VALENCE_SURPRISE, surprise, hypothesis_tag=False
                        )
                        self._surprise_write_count += 1
                    metrics["mech205_pe_mag"] = pe_mag
                    metrics["mech205_pe_ema"] = self._pe_ema
                    metrics["mech205_surprise"] = surprise
                    metrics["mech205_write_count"] = self._surprise_write_count

        if harm_signal < 0:
            harm_magnitude = abs(harm_signal)
            self._harm_this_episode += harm_magnitude

            if owned and not hypothesis_tag and self._current_latent is not None:
                z_world = self._current_latent.z_world
                residue_metrics = self.residue_field.accumulate(
                    z_world,
                    harm_magnitude=harm_magnitude,
                    world_delta=world_delta,
                    hypothesis_tag=hypothesis_tag,
                )
                metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})

                # MECH-091: harm is salient -> phase reset
                self.clock.phase_reset()

        metrics["harm_signal"] = harm_signal
        metrics["harm_this_episode"] = self._harm_this_episode
        return metrics

    def record_transition(
        self,
        z_self_t: torch.Tensor,
        action: torch.Tensor,
        z_self_t1: torch.Tensor,
    ) -> None:
        """Record (z_self_t, a_t, z_self_{t+1}) for E2 motor-sensory training."""
        # Detach all three: replay buffer entries must not retain computation graphs.
        # action from E3 select() is a slice (AsStrided) of trajectory tensors that
        # may have E3 parameter grad_fns. Without detach, compute_e2_loss() backward()
        # propagates through E3's graph and version-increments E3 tensors mid-backward,
        # causing "modified by an inplace operation" RuntimeError on the next call.
        self._e2_transition_buffer.append((
            z_self_t.detach().clone(),
            action.detach().clone(),
            z_self_t1.detach().clone(),
        ))
        if len(self._e2_transition_buffer) > 1000:
            self._e2_transition_buffer = self._e2_transition_buffer[-1000:]

    def compute_prediction_loss(self) -> torch.Tensor:
        """
        E1 world-model prediction loss for training.

        Samples from experience buffer; restores hidden state after.
        """
        zero_loss = next(self.e1.parameters()).sum() * 0.0
        if len(self._world_experience_buffer) < 2:
            return zero_loss

        buf_len = len(self._world_experience_buffer)
        horizon = self.e1.config.prediction_horizon
        max_start = max(1, buf_len - 1)
        start_idx = int(torch.randint(0, max_start, (1,)).item())
        end_idx = min(start_idx + horizon + 1, buf_len)

        if end_idx - start_idx < 2:
            return zero_loss

        # Build [z_self, z_world] sequence for E1
        self_seq  = self._self_experience_buffer[start_idx:end_idx]
        world_seq = self._world_experience_buffer[start_idx:end_idx]
        combined = [torch.cat([s.squeeze(0), w.squeeze(0)]) for s, w in zip(self_seq, world_seq)]
        sequence = torch.stack(combined).unsqueeze(0)  # [1, seq_len, total_dim]

        saved_hidden = self.e1._hidden_state
        self.e1.reset_hidden_state()

        initial = sequence[:, 0, :]
        horizon_len = sequence.shape[1] - 1
        predictions = self.e1.predict_long_horizon(initial, horizon=horizon_len)
        targets = sequence[:, 1:, :]
        loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)

        self.e1._hidden_state = saved_hidden
        return loss

    def compute_benefit_eval_loss(
        self,
        benefit_exposure: torch.Tensor,
    ) -> torch.Tensor:
        """
        ARC-030 / MECH-112: Train benefit_eval_head from benefit_exposure signal.

        benefit_exposure = body_state[11] (CausalGridWorldV2 use_proxy_fields=True).
        Supervises E3.benefit_eval_head to predict resource proximity.
        Gradient flows through E3's benefit_eval_head only.

        Args:
            benefit_exposure: [batch, 1] or scalar tensor in [0, 1]

        Returns:
            MSE loss scalar (zero if current_latent is not set)
        """
        zero_loss = next(self.e3.parameters()).sum() * 0.0
        if self._current_latent is None:
            return zero_loss
        z_world = self._current_latent.z_world.detach()
        benefit_pred = self.e3.benefit_eval(z_world)  # [batch, 1]
        target = benefit_exposure.to(z_world.device).float()
        if target.dim() == 0:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 1:
            target = target.unsqueeze(-1)
        return F.mse_loss(benefit_pred, target.expand_as(benefit_pred))

    # SD-009 / MECH-100: event label mapping.
    # obs_t is labeled with the transition_type that PRODUCED obs_t --
    # pass info_{t-1}["transition_type"] from the previous env.step(), not the current one.
    _EVENT_LABEL_MAP = {"none": 0, "env_caused_hazard": 1, "agent_caused_hazard": 2}

    def compute_event_contrastive_loss(
        self,
        transition_type: str,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-009 / MECH-100: Event contrastive CE loss on z_world encoder.

        Trains the SplitEncoder event_classifier head to distinguish transition
        types (none / env_caused_hazard / agent_caused_hazard) from z_world,
        forcing the world encoder to represent harm-relevance distinctions.
        Without this, E1/E2/reconstruction losses are invariant to event type
        and z_world converges to a harm-agnostic representation.

        Requires use_event_classifier=True in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        CE loss back through the world encoder. Usage:

            latent = agent.sense(obs_body, obs_world)
            # ... end of step t ... record transition_type from info ...
            # at step t+1, label the t+1 encoding with t's transition_type:
            latent_next = agent.sense(obs_body_next, obs_world_next)
            loss += lambda_event * agent.compute_event_contrastive_loss(
                prev_ttype, latent_next)

        Labeling convention (from EXQ-020): obs_t is labeled with the transition
        that PRODUCED obs_t. Pass info_{t-1}["transition_type"], not the current
        step's info. On the first step (no prev_ttype), skip or pass "none".

        Args:
            transition_type: string from env info dict, one of
                "none", "env_caused_hazard", "agent_caused_hazard".
                Unknown strings map to label 0 (none).
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            CE loss scalar. Gradient flows through latent_stack encoder.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_event_classifier", False):
            return zero_loss
        if latent_state.event_logits is None:
            return zero_loss
        label = self._EVENT_LABEL_MAP.get(transition_type, 0)
        label_t = torch.tensor([label], dtype=torch.long,
                               device=latent_state.event_logits.device)
        # event_logits: [batch, 3]; F.cross_entropy expects [N, C]
        logits = latent_state.event_logits
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return F.cross_entropy(logits, label_t)

    def compute_resource_proximity_loss(
        self,
        resource_proximity_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-018: Resource proximity regression loss on z_world encoder.

        Trains the SplitEncoder resource_proximity_head to predict
        max(resource_field_view) from z_world, forcing the world encoder to
        represent resource proximity. Without this, benefit_eval_head(z_world)
        produces R2=-0.004 (EXQ-085m) because E1/reconstruction losses are
        invariant to resource saliency.

        Requires use_resource_proximity_head=True in LatentStackConfig. Returns
        zero otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        MSE loss back through the world encoder.

        Labeling: target is max(resource_field_view) from the observation that
        produced this LatentState. The environment provides this in
        obs_dict["resource_field_view"] when use_proxy_fields=True.

        Args:
            resource_proximity_target: float in [0, 1], the peak resource
                proximity from the agent's current 5x5 view.
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            MSE loss scalar. Gradient flows through latent_stack encoder.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_resource_proximity_head", False):
            return zero_loss
        if latent_state.resource_prox_pred is None:
            return zero_loss
        pred = latent_state.resource_prox_pred  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        target = torch.tensor(
            [[resource_proximity_target]],
            dtype=torch.float32,
            device=pred.device,
        )
        return F.mse_loss(pred, target)

    def compute_resource_encoder_loss(
        self,
        resource_proximity_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-015 / MECH-112: auxiliary loss for ResourceEncoder object-type supervision.

        Trains the ResourceEncoder's resource_prox_head to predict resource proximity
        from z_resource. Same supervision signal as SD-018 (max(resource_field_view)),
        but backpropagates through the separate ResourceEncoder rather than through
        z_world. This forces z_resource to represent object-type features (what is
        present) rather than spatial position.

        Requires use_resource_encoder=True in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        Args:
            resource_proximity_target: scalar in [0, 1]; typically max(resource_field_view)
                from the environment observation.
            latent_state: current LatentState (from sense()); must contain resource_prox_pred_r.

        Returns:
            MSE loss scalar (zero when ResourceEncoder is disabled).
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_resource_encoder", False):
            return zero_loss
        if latent_state.resource_prox_pred_r is None:
            return zero_loss
        pred = latent_state.resource_prox_pred_r  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        target = torch.tensor(
            [[resource_proximity_target]],
            dtype=torch.float32,
            device=pred.device,
        )
        return F.mse_loss(pred, target)

    def compute_harm_nonredundancy_loss(
        self,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-019: Affective harm non-redundancy constraint loss.

        Penalises cosine alignment between z_harm_s and z_harm_a using a squared
        cosine similarity penalty. This enforces that the two harm streams encode
        non-redundant information, consistent with the A-delta/C-fiber distinction:
        sensory-discriminative (z_harm_s) encodes immediate proximity/intensity,
        affective-motivational (z_harm_a) encodes accumulated threat burden with
        different temporal scope and persistence.

        Penalty = cos_sim(z_harm_s, z_harm_a)^2
        - Penalises both positive AND negative alignment (symmetric).
        - Zero gradient when streams are orthogonal.
        - Large gradient when streams are parallel (redundant).

        ARC-016 coupling: when harm_nonredundancy_precision_scale > 0, the penalty
        is scaled by normalised E3 precision, enforcing non-redundancy more strongly
        in high-confidence (low-variance) states. Biological grounding: Barlow 1961
        redundancy reduction; C-fiber/A-delta distinct temporal integration profiles.

        Returns zero when harm_nonredundancy_weight=0.0 (default, backward compat)
        or when either harm stream is absent from the current latent state.

        Args:
            latent_state: current LatentState (from sense()); must contain z_harm and z_harm_a.

        Returns:
            Penalty loss scalar (>= 0).
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        weight = getattr(self.config, "harm_nonredundancy_weight", 0.0)
        if weight <= 0.0:
            return zero_loss
        if latent_state.z_harm is None or latent_state.z_harm_a is None:
            return zero_loss
        z_s = latent_state.z_harm    # [batch, harm_dim]
        z_a = latent_state.z_harm_a  # [batch, z_harm_a_dim]
        # Project to common dim if needed (use the smaller)
        min_dim = min(z_s.shape[-1], z_a.shape[-1])
        z_s_proj = z_s[..., :min_dim]
        z_a_proj = z_a[..., :min_dim]
        cos_sim = F.cosine_similarity(z_s_proj, z_a_proj, dim=-1).mean()
        penalty = cos_sim.pow(2)
        # ARC-016 coupling: scale by normalised precision when enabled
        prec_scale = getattr(self.config, "harm_nonredundancy_precision_scale", 0.0)
        if prec_scale > 0.0:
            precision_norm = min(self.e3.current_precision / 500.0, 2.0)
            penalty = penalty * (1.0 + prec_scale * precision_norm)
        return weight * penalty

    def compute_harm_accum_loss(
        self,
        accumulated_harm_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-011 second source: auxiliary loss for harm accumulation prediction.

        Trains the AffectiveHarmEncoder harm_accum_head to predict accumulated
        harm exposure from z_harm_a, forcing the affective encoder to integrate
        temporal harm information that z_harm_s does not receive. This creates
        gradient pressure for genuine stream divergence (resolving EXQ-241 D3
        reversal where z_harm_a was monotonically redundant with z_harm_s).

        Requires harm_history_len > 0 in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        MSE loss back through the affective harm encoder.

        Args:
            accumulated_harm_target: float in [0, 1], running average of
                harm_exposure over the episode. From obs_dict["accumulated_harm"].
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            Weighted MSE loss scalar. Weight = z_harm_a_aux_loss_weight.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if getattr(self.config.latent, "harm_history_len", 0) <= 0:
            return zero_loss
        if latent_state.harm_accum_pred is None:
            return zero_loss
        pred = latent_state.harm_accum_pred  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)

        # SD-020: optional precision-weighted PE target for z_harm_a training.
        # When harm_surprise_pe_enabled=True, replace EMA accumulated_harm_target
        # with |actual_harm - expected_harm| * precision_norm (affective surprise).
        # Chen 2023: AIC encodes unsigned intensity prediction errors, not raw magnitude.
        # precision_norm = min(E3.current_precision / 500, 3.0) provides ARC-016 coupling.
        harm_surprise_pe_enabled = getattr(self.config, "harm_surprise_pe_enabled", False)
        if harm_surprise_pe_enabled:
            alpha = getattr(self.config, "harm_obs_ema_alpha", 0.1)
            # Update running expected harm
            self._harm_obs_ema = (1.0 - alpha) * self._harm_obs_ema + alpha * accumulated_harm_target
            # PE = |actual - expected|
            harm_pe = abs(accumulated_harm_target - self._harm_obs_ema)
            # ARC-016: scale by normalised precision
            precision_norm = min(self.e3.current_precision / 500.0, 3.0)
            surprise_target = harm_pe * precision_norm
            target_val = surprise_target
        else:
            target_val = accumulated_harm_target

        target = torch.tensor(
            [[target_val]],
            dtype=torch.float32,
            device=pred.device,
        )
        weight = getattr(self.config.latent, "z_harm_a_aux_loss_weight", 0.1)
        return weight * F.mse_loss(pred, target)

    def update_z_goal(self, benefit_exposure: float, drive_level: float = 1.0) -> None:
        """Update z_goal from benefit signal (MECH-112 wanting update).

        SD-015: when ResourceEncoder is enabled and z_resource is populated in the
        current latent state, seeds z_goal from z_resource (object-type latent) rather
        than z_world (full scene latent). z_resource encodes "what kind of resource is
        present" independent of spatial position -- resources respawn at random locations,
        so z_world at contact has no predictive value for future resource locations.

        Args:
            benefit_exposure: scalar benefit this step (obs_body[11] in proxy mode)
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         Compute as: 1.0 - float(obs_body[0, 3]) where obs_body[3]=energy.
        """
        if self.goal_state is None or self._current_latent is None:
            return
        # SD-015: use z_resource if available (object-type seeding), else z_world
        use_resource = (
            getattr(self.config.latent, "use_resource_encoder", False)
            and self._current_latent.z_resource is not None
        )
        seed_latent = (
            self._current_latent.z_resource if use_resource
            else self._current_latent.z_world
        )
        # SD-032: cache base drive_level on goal_state so AIC / PCC / pACC /
        # salience consumers can read a live value via getattr. Convention:
        # stored value is the BASE drive_level (no pACC bias); SD-032e
        # consumers apply pacc.effective_drive() themselves to avoid double-
        # counting.
        self.goal_state._last_drive_level = float(drive_level)
        # SD-032e: the wanting gain scaling inside GoalState.update() should
        # see the SENSITISED drive regime. Sustained affective pain ->
        # elevated drive_bias -> stronger wanting pull toward resources.
        effective_drive = drive_level
        if self.pacc is not None:
            effective_drive = self.pacc.effective_drive(drive_level)
        self.goal_state.update(
            seed_latent,
            benefit_exposure,
            drive_level=effective_drive,
        )

    def serotonin_step(self, benefit_exposure: float) -> None:
        """
        MECH-203: Update tonic 5-HT and dynamically modulate GoalConfig.

        Extracts z_harm_a norm from current latent state. Updates tonic_5ht,
        then writes current_seeding_gain and current_wanting_floor into
        the GoalConfig so that goal.update() uses dynamic serotonergic values.

        No-op when serotonin.enabled is False (default).
        """
        if not self.serotonin.enabled:
            return

        z_harm_a_norm = 0.0
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a_norm = float(self._current_latent.z_harm_a.norm().item())

        self.serotonin.serotonin_step(benefit_exposure, z_harm_a_norm)

        # Dynamically modulate GoalConfig parameters
        if self.goal_state is not None:
            self.goal_state.config.z_goal_seeding_gain = self.serotonin.current_seeding_gain()
            self.goal_state.config.valence_wanting_floor = self.serotonin.current_wanting_floor()

    def update_benefit_salience(self, benefit_exposure: float) -> None:
        """
        MECH-203 (SR-2): Tag current residue field location with benefit salience.

        Uses SD-014 VALENCE_WANTING infrastructure: writes benefit_salience into
        the residue field's valence vector at the current z_world position.

        No-op when serotonin is disabled or no current latent state.
        """
        if not self.serotonin.enabled or self._current_latent is None:
            return

        salience = self.serotonin.benefit_salience(benefit_exposure)
        if salience <= 0.0:
            return

        z_world = self._current_latent.z_world
        if hasattr(self.residue_field, 'update_valence'):
            # Write benefit_salience into VALENCE_WANTING (index 0) at current z_world
            self.residue_field.update_valence(
                z_world,
                component=VALENCE_WANTING,
                value=salience,
                hypothesis_tag=False,
            )

    def update_liking(self, benefit_exposure: float) -> None:
        """SD-014 l-component: write consummatory benefit signal to VALENCE_LIKING.

        Berridge liking = hedonic impact at resource contact (opioid-mediated).
        Unlike wanting (anticipatory), liking fires on consummatory contact only.
        Threshold gating prevents proximity gradients from contaminating the liking map.
        No-op when valence_liking_enabled is False or no current latent state.
        """
        if not getattr(self.config, "valence_liking_enabled", False):
            return
        if self._current_latent is None:
            return
        threshold = getattr(self.config, "liking_threshold", 0.1)
        if benefit_exposure < threshold:
            return
        z_world = self._current_latent.z_world
        if hasattr(self.residue_field, "update_valence"):
            self.residue_field.update_valence(
                z_world,
                component=VALENCE_LIKING,
                value=benefit_exposure,
                hypothesis_tag=False,
            )

    @staticmethod
    def compute_drive_level(obs_body: torch.Tensor) -> float:
        """SD-012: Compute homeostatic drive from body observation.

        drive_level = 1.0 - energy (obs_body[3]). 0=sated, 1=depleted.
        Canonical formula -- single source of truth for SD-012.
        """
        if obs_body.dim() == 2:
            energy = float(obs_body[0, 3])
        else:
            energy = float(obs_body[3])
        return max(0.0, 1.0 - energy)

    def update_schema_wanting(self, drive_level: float = 1.0) -> None:
        """MECH-216: seed VALENCE_WANTING from E1 schema salience.

        Zhang/Berridge: W_m = kappa (drive_level) x V_hat (schema_salience).
        Only writes when schema_salience >= threshold and schema_wanting is enabled.
        """
        if self._schema_salience is None or self._current_latent is None:
            return
        if not getattr(self.config.e1, 'schema_wanting_enabled', False):
            return
        sal_val = float(self._schema_salience.squeeze())
        threshold = getattr(self.config, 'schema_wanting_threshold', 0.3)
        if sal_val < threshold:
            return
        gain = getattr(self.config, 'schema_wanting_gain', 0.5)
        wanting_value = sal_val * gain * max(drive_level, 0.1)
        z_world = self._current_latent.z_world
        if hasattr(self.residue_field, 'update_valence'):
            self.residue_field.update_valence(
                z_world, component=VALENCE_WANTING,
                value=wanting_value, hypothesis_tag=False,
            )

    def compute_schema_readout_loss(
        self, resource_proximity_target: float
    ) -> torch.Tensor:
        """MECH-216: passthrough to E1.compute_schema_readout_loss."""
        if not getattr(self.config.e1, 'schema_wanting_enabled', False):
            return torch.tensor(0.0)
        target = torch.tensor([[resource_proximity_target]], dtype=torch.float32)
        return self.e1.compute_schema_readout_loss(target)

    def compute_goal_maintenance_diagnostic(self) -> dict:
        """Goal state metrics for MECH-116 (EXQ-076 criterion C1)."""
        if self.goal_state is None or self._current_latent is None:
            return {"goal_norm": 0.0, "goal_proximity": 0.0, "is_active": False}
        prox = float(
            self.goal_state.goal_proximity(
                self._current_latent.z_world
            ).mean().item()
        )
        return {
            "goal_norm": self.goal_state.goal_norm(),
            "goal_proximity": prox,
            "is_active": self.goal_state.is_active(),
        }

    def compute_z_self_d_eff(self) -> Optional[float]:
        """
        MECH-113: Compute z_self participation ratio (D_eff).

        D_eff = (sum|z_self|)^2 / sum(z_self^2)
        From epistemic-mapping: measures effective dimensionality of the
        self-model. High D_eff = distributed/uncertain; low D_eff = focused.
        Self-maintenance loss penalises D_eff exceeding the target threshold.

        Returns:
            D_eff scalar or None if no latent state available.
        """
        if self._current_latent is None:
            return None
        z = self._current_latent.z_self.detach().squeeze(0)  # [self_dim]
        abs_z = z.abs()
        numerator = abs_z.sum().pow(2)
        denominator = z.pow(2).sum()
        if denominator.item() < 1e-8:
            return None
        return (numerator / denominator).item()

    def compute_self_maintenance_loss(self) -> torch.Tensor:
        """
        MECH-113: Self-maintenance auxiliary loss on z_self D_eff.

        Penalises D_eff > self_maintenance_d_eff_target.
        This creates a homeostatic pressure to keep z_self representations
        focused (low D_eff = coherent self-model). Disabled when weight=0.

        Returns:
            Scalar loss tensor (zero if disabled or no latent state).
        """
        weight = self.config.e3.self_maintenance_weight
        zero_loss = next(self.e3.parameters()).sum() * 0.0
        if weight == 0.0 or self._current_latent is None:
            return zero_loss
        z = self._current_latent.z_self  # [batch, self_dim] — keep in graph
        abs_z = z.abs()
        numerator = abs_z.sum(dim=-1).pow(2)     # [batch]
        denominator = z.pow(2).sum(dim=-1)        # [batch]
        d_eff = numerator / (denominator + 1e-8)  # [batch]
        target = self.config.e3.self_maintenance_d_eff_target
        excess = F.relu(d_eff - target)           # penalise only excess
        return weight * excess.mean()

    def compute_e2_loss(self, batch_size: int = 16) -> torch.Tensor:
        """E2 motor-sensory forward-model loss (z_self domain)."""
        zero_loss = next(self.e2.parameters()).sum() * 0.0
        if len(self._e2_transition_buffer) < 2:
            return zero_loss

        n = min(batch_size, len(self._e2_transition_buffer))
        indices = torch.randperm(len(self._e2_transition_buffer))[:n].tolist()
        batch = [self._e2_transition_buffer[i] for i in indices]
        z_t_list, a_list, z_t1_list = zip(*batch)

        z_t  = torch.cat(z_t_list, dim=0)
        acts = torch.cat(a_list, dim=0)
        z_t1 = torch.cat(z_t1_list, dim=0)

        z_t1_pred = self.e2.predict_next_self(z_t, acts)
        return F.mse_loss(z_t1_pred, z_t1)

    def offline_integration(self) -> Dict[str, float]:
        """Offline integration (residue contextualisation + E1 replay)."""
        metrics: Dict[str, float] = {}
        if len(self._world_experience_buffer) > 10:
            e1_metrics = self.e1.integrate_experience(
                [torch.cat([s, w]) for s, w in
                 zip(self._self_experience_buffer, self._world_experience_buffer)]
            )
            metrics.update({f"e1_{k}": v for k, v in e1_metrics.items()})
        residue_metrics = self.residue_field.integrate()
        metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})
        return metrics

    def should_integrate(self) -> bool:
        return self._step_count % self.config.offline_integration_frequency == 0

    def enter_offline_mode(self) -> None:
        """
        MECH-120 / SD-017: Gate E1 context_memory writes during offline phases.

        Sets e1._offline_mode = True, which suppresses context_memory.write()
        calls in E1.update_from_observation(). This prevents new waking
        observations from overwriting schema slots installed during the
        SWS-analog pass (Phase 0 sensory gate from offline_phases.md).

        Call before running SWS/REM-analog passes.
        Paired with exit_offline_mode() to resume normal waking writes.

        SD-032d: also resets the PCC steps_since_offline counter so the
        coordinator's effective_threshold relaxes after rest. Both MECH-092
        within-session quiescence and INV-049 cross-session sleep paths
        funnel through here, giving SD-032d a single integration point.
        """
        self.e1._offline_mode = True
        if self.pcc is not None:
            self.pcc.note_offline_entry()
        # SD-032e: optional offline decay of drive_bias (default 0.0 = no-op).
        # Non-zero decay would instantiate a distinct sleep-recalibration
        # claim per the SD-032e scoping synthesis; default preserves the
        # cross-session pACC accumulator untouched.
        if self.pacc is not None:
            self.pacc.note_offline_entry()

    def exit_offline_mode(self) -> None:
        """Resume normal waking context_memory writes (undo enter_offline_mode)."""
        self.e1._offline_mode = False

    def note_task_outcome(self, outcome: float) -> None:
        """SD-032d: feed a per-step task-outcome scalar into the PCC success EMA.

        Convenience pass-through to self.pcc.note_task_outcome(outcome). No-op
        when use_pcc_analog=False. Experiment loops choose what counts as a
        task outcome (e.g., 1.0 on benefit-collection, 0.0 on harm event,
        0.5 otherwise). Without any calls, the PCC success channel stays
        neutral and contributes zero to stability.
        """
        if self.pcc is not None:
            self.pcc.note_task_outcome(outcome)

    # -- Sleep mode convenience methods (SD-017 + MECH-203/204) --

    def enter_sws_mode(self) -> None:
        """
        Enter SWS-analog phase.

        Composes: enter_offline_mode() + SHY normalisation + serotonin.enter_sws().
        Phase 0: gate waking writes. Phase 1: SHY normalisation (MECH-120).
        Phase 2 (replay) runs after this method returns.
        """
        self.enter_offline_mode()
        # MECH-120: SHY normalization before replay (Phase 1)
        if self.config.shy_enabled:
            self.e1.shy_normalise(decay=self.config.shy_decay_rate)
        self.serotonin.enter_sws()

    def enter_rem_mode(self) -> None:
        """
        Enter REM-analog phase.

        Composes: enter_offline_mode() + serotonin.enter_rem().
        SR-3: captures current E3 precision as zero-point reference.
        """
        self.enter_offline_mode()
        self.serotonin.enter_rem(current_precision=self.e3.current_precision)

    def exit_sleep_mode(self) -> None:
        """
        Exit sleep, return to waking mode.

        Composes: exit_offline_mode() + serotonin.exit_sleep().
        Restores E1 context_memory writes and pre-sleep 5-HT level.
        """
        self.exit_offline_mode()
        self.serotonin.exit_sleep()

    # -- SD-017: SWS-analog and REM-analog passes --

    def run_sws_schema_pass(self) -> Dict[str, float]:
        """
        SD-017 SWS-analog pass: hippocampus-to-cortex schema installation.

        Installs differentiated context attractors in E1.ContextMemory by
        writing compressed z_world prototypes from recent waking experience.
        This is the slot-formation phase (MECH-166): before this pass, the
        ContextMemory slots are undifferentiated (cosine_sim -> 1.0); after,
        they form attractors that slot-filling (REM-analog) can populate.

        Must be called after enter_sws_mode() (which gates waking writes and
        runs MECH-120 SHY normalisation). The offline gate prevents new waking
        observations from overwriting slots installed here.

        Algorithm:
        1. Sample sws_consolidation_steps prototype z_worlds from the world
           experience buffer (diverse sampling: early, mid, and recent windows).
        2. For each prototype, construct the full E1 input [z_self_mean, z_world]
           and write it directly to ContextMemory bypassing the offline gate.
           (The offline gate suppresses waking writes; this pass writes offline
           schema content, which is the intended action during SWS.)
        3. Compute slot diversity (mean pairwise cosine distance) as a metric
           for context differentiation quality.

        Args: none (uses self._world_experience_buffer internally)

        Returns:
            dict with keys:
              sws_n_writes: number of prototype writes attempted
              sws_slot_diversity: mean pairwise cosine distance of ContextMemory
                                  slots after pass (higher = more differentiated)
              sws_buffer_size: size of experience buffer used
        """
        metrics: Dict[str, float] = {
            "sws_n_writes": 0.0,
            "sws_slot_diversity": 0.0,
            "sws_buffer_size": 0.0,
        }

        if not self.config.sws_enabled:
            return metrics

        wb = self._world_experience_buffer
        sb = self._self_experience_buffer
        n_buf = len(wb)
        metrics["sws_buffer_size"] = float(n_buf)

        if n_buf < 2:
            return metrics

        # Temporarily lift offline gate for schema writes
        # (gate suppresses waking obs; schema installation is intentional offline write)
        was_offline = self.e1._offline_mode
        self.e1._offline_mode = False

        n_steps = min(self.config.sws_consolidation_steps, n_buf)
        # Diverse sampling: spread across buffer windows
        if n_steps >= n_buf:
            indices = list(range(n_buf))
        else:
            # Sample from early, mid, recent thirds plus random fill
            step = max(1, n_buf // n_steps)
            indices = list(range(0, n_buf, step))[:n_steps]

        n_writes = 0
        self_dim = self.config.latent.self_dim

        for idx in indices:
            z_world = wb[idx].detach()     # [1, world_dim] or [world_dim]
            if z_world.dim() == 1:
                z_world = z_world.unsqueeze(0)

            # Pair with corresponding z_self if available, else zeros
            if idx < len(sb):
                z_self = sb[idx].detach()
                if z_self.dim() == 1:
                    z_self = z_self.unsqueeze(0)
            else:
                z_self = torch.zeros(1, self_dim, device=self.device)

            # Full E1 input: [z_self, z_world] concatenated
            e1_input = torch.cat([z_self, z_world], dim=-1)  # [1, self_dim+world_dim]

            # Write to ContextMemory (offline gate lifted for this block only)
            self.e1.context_memory.write(e1_input)
            n_writes += 1

        # Restore gate
        self.e1._offline_mode = was_offline

        metrics["sws_n_writes"] = float(n_writes)

        # Compute slot diversity: mean pairwise cosine distance across memory slots
        with torch.no_grad():
            mem = self.e1.context_memory.memory  # [num_slots, memory_dim]
            num_slots = mem.shape[0]
            if num_slots > 1:
                # Normalise rows
                norms = mem.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                normed = mem / norms  # [num_slots, memory_dim]
                # Cosine similarity matrix [num_slots, num_slots]
                sim_mat = torch.mm(normed, normed.t())
                # Mask diagonal (self-similarity = 1.0)
                mask = torch.eye(num_slots, device=sim_mat.device, dtype=torch.bool)
                off_diag = sim_mat[~mask]
                # Diversity = mean pairwise distance (1 - cosine similarity)
                diversity = float((1.0 - off_diag).mean().item())
                metrics["sws_slot_diversity"] = diversity

        return metrics

    def run_rem_attribution_pass(self) -> Dict[str, float]:
        """
        SD-017 REM-analog pass: causal attribution replay (slot-filling, MECH-166).

        Replays recent trajectory experience through the hippocampal module in
        both forward and reverse temporal order (ARC-045 bidirectional flow proxy).
        Evaluates residue terrain per trajectory segment WITHOUT accumulating new
        residue (hypothesis_tag=True per MECH-094). This is the slot-filling phase:
        with schema attractors installed by the SWS pass, trajectory evidence can
        now be attributed to differentiated context slots.

        Must be called after run_sws_schema_pass() (slots must exist before
        filling). The offline gate should still be active (enter_rem_mode()
        ensures this).

        Algorithm:
        1. Take recent z_world from theta_buffer (waking experience).
        2. For each attribution step:
           a. Forward replay: hippocampal.replay() -- random rollout from recent z_world.
           b. Reverse replay: hippocampal.reverse_replay() on a stored trajectory.
           c. Evaluate residue terrain on each trajectory (read-only, no writes).
        3. Aggregate attribution metrics: mean harm terrain, benefit terrain (if enabled),
           context differentiation proxy (variance of terrain scores across rollouts).

        Returns:
            dict with keys:
              rem_n_rollouts: number of rollouts attempted
              rem_mean_harm_terrain: mean residue terrain score across rollouts
              rem_terrain_variance: variance of terrain scores (context differentiation proxy)
              rem_n_reverse: number of reverse-order rollouts included
        """
        metrics: Dict[str, float] = {
            "rem_n_rollouts": 0.0,
            "rem_mean_harm_terrain": 0.0,
            "rem_terrain_variance": 0.0,
            "rem_n_reverse": 0.0,
        }

        if not self.config.rem_enabled:
            return metrics

        recent = self.theta_buffer.recent
        if recent is None:
            return metrics

        n_steps = self.config.rem_attribution_steps
        terrain_scores: List[float] = []
        n_reverse = 0

        # Forward replay pass: hypothesis_tag=True (read-only, no residue writes)
        forward_trajs = self.hippocampal.replay(
            recent,
            num_replay_steps=max(1, n_steps // 2),
            drive_state=None,  # attribution mode: drive-neutral
        )

        for traj in forward_trajs:
            score = self.hippocampal._score_trajectory(traj)
            terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

        # Reverse replay pass (ARC-045 bidirectional flow proxy)
        # Only if exploration buffer has stored trajectories
        n_reverse_steps = max(1, n_steps - len(forward_trajs))
        if len(self.hippocampal._exploration_buffer) > 0:
            reverse_trajs = self.hippocampal.diverse_replay(
                recent,
                num_replay_steps=n_reverse_steps,
                drive_state=None,
                mode="reverse",
            )
            for traj in reverse_trajs:
                score = self.hippocampal._score_trajectory(traj)
                terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))
                n_reverse += 1
        else:
            # No exploration buffer yet: extra forward rollouts
            extra = self.hippocampal.replay(recent, num_replay_steps=n_reverse_steps)
            for traj in extra:
                score = self.hippocampal._score_trajectory(traj)
                terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

        if terrain_scores:
            metrics["rem_n_rollouts"] = float(len(terrain_scores))
            metrics["rem_mean_harm_terrain"] = float(sum(terrain_scores) / len(terrain_scores))
            if len(terrain_scores) > 1:
                mean_s = metrics["rem_mean_harm_terrain"]
                var_s = sum((s - mean_s) ** 2 for s in terrain_scores) / len(terrain_scores)
                metrics["rem_terrain_variance"] = var_s
            metrics["rem_n_reverse"] = float(n_reverse)

        return metrics

    def run_sleep_cycle(self) -> Dict[str, float]:
        """
        SD-017: Run a complete SWS->REM sleep cycle.

        Convenience method: calls enter_sws_mode(), run_sws_schema_pass(),
        enter_rem_mode(), run_rem_attribution_pass(), then exit_sleep_mode().

        Returns merged metrics from both passes.

        IMPORTANT: Only call this when sws_enabled=True or rem_enabled=True.
        If both are False this is a no-op returning empty metrics.
        """
        all_metrics: Dict[str, float] = {}
        if not self.config.sws_enabled and not self.config.rem_enabled:
            return all_metrics

        # SWS phase
        if self.config.sws_enabled:
            self.enter_sws_mode()
            sws_metrics = self.run_sws_schema_pass()
            all_metrics.update(sws_metrics)
            self.exit_sleep_mode()

        # REM phase (requires slots installed by SWS; safe to run standalone too)
        if self.config.rem_enabled:
            self.enter_rem_mode()
            rem_metrics = self.run_rem_attribution_pass()
            all_metrics.update(rem_metrics)
            self.exit_sleep_mode()

        return all_metrics

    def get_state(self) -> AgentState:
        return AgentState(
            latent_state=self._current_latent,
            precision=self.e3.current_precision,
            running_variance=self.e3._running_variance,
            step=self._step_count,
            harm_accumulated=self._harm_this_episode,
            is_committed=self.e3._committed_trajectory is not None,
            beta_elevated=self.beta_gate.is_elevated,
            e3_steps_per_tick=self.clock.e3_steps_per_tick,
            serotonin_state=self.serotonin.get_state() if self.serotonin.enabled else None,
        )

    def get_residue_statistics(self) -> Dict[str, torch.Tensor]:
        return self.residue_field.get_statistics()

    def forward(self, observation: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return self.act(observation, temperature)
