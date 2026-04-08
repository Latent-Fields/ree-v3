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
from ree_core.residue.field import VALENCE_WANTING, VALENCE_SURPRISE


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
        self._cue_action_bias    = None
        self._cue_terrain_weight = None
        self.clock.reset()
        self.theta_buffer.reset()
        self.beta_gate.reset()
        self.serotonin.reset()
        self._pe_ema = 0.0

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
        # Detach before storing: prevents EMA from linking computational graphs
        # across time steps. Without detach, optimizer.step() modifies weights
        # in-place, invalidating the old graph's version — causing RuntimeError
        # "modified by an inplace operation" on the next backward() call.
        self._current_latent = new_latent.detach()
        # MECH-165: record state for exploration trajectory
        self._record_exploration_state()
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
        """
        if not ticks["e3_tick"] and self._last_action is not None:
            # MECH-165: record held action for exploration trajectory (every step)
            self._record_exploration_action(self._last_action)
            return self._last_action

        # SD-016 (MECH-152): pass cached terrain_weight so harm/goal scoring
        # precision reflects current z_world cue context.
        # MECH-108: pass BreathOscillator sweep reduction during sweep phase.
        # Creates periodic uncommitted windows even when running_variance has
        # converged below base commit_threshold after training.
        sweep_reduction = self.clock.sweep_amplitude if self.clock.sweep_active else 0.0
        # SD-011: extract z_harm_a for E3 urgency gating and ethical cost amplification.
        z_harm_a = None
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a = self._current_latent.z_harm_a
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
        result = self.e3.select(
            candidates, temperature,
            goal_state=_goal_state_for_select,
            terrain_weight=self._cue_terrain_weight,
            sweep_threshold_reduction=sweep_reduction,
            z_harm_a=z_harm_a,
        )
        action = result.selected_action

        # MECH-090: gate policy propagation based on beta state
        # For now: if committed, elevate beta; if not, release
        if result.committed:
            self.beta_gate.elevate()
        else:
            self.beta_gate.release()

        propagated = self.beta_gate.propagate(action)
        if propagated is None:
            # Beta elevated: hold previous action
            if self._last_action is not None:
                action = self._last_action

        self._last_action = action
        # MECH-165: record action for exploration trajectory
        self._record_exploration_action(action)
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
        target = torch.tensor(
            [[accumulated_harm_target]],
            dtype=torch.float32,
            device=pred.device,
        )
        weight = getattr(self.config.latent, "z_harm_a_aux_loss_weight", 0.1)
        return weight * F.mse_loss(pred, target)

    def update_z_goal(self, benefit_exposure: float, drive_level: float = 1.0) -> None:
        """Update z_goal from benefit signal (MECH-112 wanting update).

        Args:
            benefit_exposure: scalar benefit this step (obs_body[11] in proxy mode)
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         Compute as: 1.0 - float(obs_body[0, 3]) where obs_body[3]=energy.
        """
        if self.goal_state is None or self._current_latent is None:
            return
        self.goal_state.update(
            self._current_latent.z_world,
            benefit_exposure,
            drive_level=drive_level,
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
        """
        self.e1._offline_mode = True

    def exit_offline_mode(self) -> None:
        """Resume normal waking context_memory writes (undo enter_offline_mode)."""
        self.e1._offline_mode = False

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
