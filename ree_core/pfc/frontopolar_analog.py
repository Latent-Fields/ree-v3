"""SD-033e: Frontopolar-analog (BA 10 / rostral pole).

V4-scope module for the FULL substrate (learned MECH-264/MECH-265 heads +
gateway + relative-importance monitor). The learned path is still a STUB --
every V4 method is a no-op when the master flag `use_frontopolar_analog` is
False (default), and raises NotImplementedError when enabled.

--------------------------------------------------------------------------------
SD-033e V3-NARROW DE-COMMIT LANDING (2026-07-09)

A DELIBERATE, low-odds-but-owed V3-narrow slice landed under the SEPARATE flag
`use_frontopolar_decommit` (orthogonal to the V4 `use_frontopolar_analog`). It
instantiates ONLY function (1) MECH-264 counterfactual-value + function (4)
disengagement-for-exploration, in a PARAMETER-FREE GEOMETRIC form (no learned
head, no training phase), as a commit-LATCH DE-COMMIT / switch-propensity lever
for the DURATION face of the F-dominance conversion ceiling (MECH-439).

WHY (frontier context): on strong-F seeds the primary harm/goal score F forms a
monolithic natural-commit latch (~2400-2600 steps, V3-EXQ-460h) that nothing
releases -- readiness stays healthy so MECH-342 is silent, no closure fires so
SD-034 is silent. Input-side score-bias reweighting is EXHAUSTED (V3-EXQ-709/
711/713). The arming-reliability residual (failure_autopsy_MECH-445-cluster-
715a-717_2026-07-07) needs a DISTINCT de-commit-release lever, not more
F-moderation. This IS that lever.

HOW: compute_counterfactual_value (use_frontopolar_decommit branch) returns the
NON-F goal-proximity ADVANTAGE ||z_chosen - z_goal|| - ||z_alt - z_goal||;
compute_decommit_pressure applies the ENTRY-RELATIVE derivative
frontopolar_gain * max(0, cfv_now - cfv_at_entry); the agent injects that scalar
into NaturalCommitUrgencyRelease.tick(frontopolar_pressure=...) so it fires on
the SAME urgency >= release_bound as the existing rung-6 duration release
(reusing the agent.py release block + its _ncl_lever_fired latch-hold yield).
Non-F source + entry-relative form are the two mitigations that keep this a
genuine lever rather than an F-derivative that washes like 709/711/713.

V3-NARROW SCOPE: function (2) MECH-265 relative-importance and the gateway mode
stay V4 stubs (single-resource env can't supply K>=2 goals);
compute_disengagement_bias / compute_relative_importance are NOT on the
de-commit path. See REE_assembly/docs/architecture/sd_033_pfc_subdivision_
architecture.md section SD-033e and failure_autopsy_V3-EXQ-460h_2026-06-20.md.

BIT-IDENTICAL OFF: `use_frontopolar_decommit` defaults False; the agent does not
instantiate FrontopolarAnalog unless it is set, and both de-commit methods
return 0.0 / zeros when it is False.

--------------------------------------------------------------------------------
ORIGINAL V4 STUB CONTRACT (learned heads; unchanged)

Every V4 method is a no-op when `use_frontopolar_analog` is False (default), and
raises NotImplementedError when enabled until the corresponding implementation
lands.

Purpose of the stub:
  (1) Reserve the module surface so SalienceCoordinator can treat
      `parallel_goal_deliberation` and a `sd_033e` write-gate target as
      first-class without schema churn when V4 lands.
  (2) Pin down the interface the SD-033e design doc will instantiate against
      (MECH-264 counterfactual-value head, MECH-265 relative-importance
      monitor), so the Prong D literature pull and subsequent design doc
      produce concrete architectural choices rather than fresh API design.
  (3) Keep backward compatibility: nothing in this file is reachable on the
      default V3 path. `use_frontopolar_analog=False` is always the live
      branch until the design doc lands and lifts the guard.

--------------------------------------------------------------------------------
DESIGN LANDING STATE (2026-04-21)

- Prong D literature pull queued in task_inbox.md (line 21) covering Boorman
  2009, Burgess/Dumontheil/Gilbert 2007, Mansouri 2015 / 2017, Koechlin &
  Summerfield 2007, Christoff, Badre & Nee 2018. The four open synthesis
  questions (pairwise vs K>=2 counterfactual; gateway continuous vs discrete;
  lesion specificity; training signal for counterfactual heads) are
  prerequisites for authoring the SD-033e design doc.
- SD-033a (LateralPFCAnalog) is the template for this module's structure
  and DESIGN ALTERNATIVES section. SD-033e mirrors it at a higher level of
  abstraction (counterfactual-value + parallel-goal monitoring instead of
  rule-state + bias).

--------------------------------------------------------------------------------
INTENDED FOUR CONVERGING FUNCTIONS (per SD-033e claim in claims.yaml:18632)

  (1) MECH-264 counterfactual-value tracking (Boorman 2009):
      For the currently chosen trajectory, estimate the value of the single
      best unchosen alternative. Emits `counterfactual_value` scalar (or [K-1]
      vector if pairwise structure is preserved). Feeds E3 as a
      switch-propensity signal.
  (2) MECH-265 relative-importance monitoring (Mansouri 2017):
      Over K>=2 simultaneously active goals (V4 dual-goal env prerequisite),
      emit a [K] relative-importance vector. Used to modulate GoalState goal
      weighting and SalienceCoordinator `parallel_goal_deliberation` mode.
  (3) Gateway / engagement mode (Burgess 2007):
      Continuous or discrete switch between medial (external task engagement)
      and lateral (internal cognition / counterfactual search) FPC modes.
      Interacts with SD-032a operating_mode.
  (4) Disengagement for exploration (Mansouri 2015 lesion signature):
      When the counterfactual-value signal is high and relative-importance
      variance is low (i.e., current goal is not dominant), emit a
      disengagement bias that makes committed-state release (MECH-090) more
      likely. Interacts with SD-034 closure operator.

--------------------------------------------------------------------------------
PREREQUISITES (all V4-scope)

  - V3 full-completion gate: MECH-163 hippocampal multi-step trajectory
    planning. Without multi-step plans, counterfactual values are
    structurally 1-step and MECH-264 collapses to an immediate-reward
    comparator (i.e., reduces to MECH-257 dACC function).
  - CausalGridWorldV2 dual-active-goal extension: the current env emits a
    single resource cue. MECH-265 relative-importance over K>=2 goals is
    structurally inaccessible until this env extension lands.
  - SalienceCoordinator `parallel_goal_deliberation` mode registration.
    This is a V4 extension to DEFAULT_MODE_NAMES. The registry already
    accepts arbitrary mode keys via register_target (see
    ree_core/cingulate/salience_coordinator.py:274), so this requires no
    schema change -- only activation behind a
    `salience_enable_v4_modes` flag (deferred pending stale MECH-267
    claim resolution; see TASK_CLAIMS.json).

--------------------------------------------------------------------------------
BACKWARD COMPAT CONTRACT

  - `use_frontopolar_analog` defaults False. When False, every method is
    a no-op or returns a zero tensor matching the shape the enabled path
    would return. The module is safe to instantiate unconditionally.
  - When True, every method currently raises NotImplementedError with a
    pointer to the Prong D lit-pull task and the SD-033e design doc
    that will land it. This is deliberate: the STUB must not silently
    return plausible-looking numbers, because that would contaminate any
    future ablation baseline.
  - The zero-init pattern used by SD-033a (frozen-random with last Linear
    zeroed at init, exact-zero initial output) will be adopted here once
    MECH-264 and MECH-265 heads are authored. At that point the guard
    will be lifted and the no-op return paths become the bit-identical
    backward-compat path.

--------------------------------------------------------------------------------
MECH-094

Hypothesis tag handling is deferred to the SD-033e design doc. Following
SD-033a, the default plan is to route counterfactual-value and relative-
importance updates through the MECH-261 write-gate registry under target
key `sd_033e`. Gate under internal_replay / offline_consolidation modes
will be low; waking external_task / parallel_goal_deliberation high. See
DEFAULT_GATE_WEIGHTS comment stub (deferred edit in salience_coordinator.py).

See also:
  - REE_assembly/docs/claims/claims.yaml -- SD-033e, MECH-264, MECH-265
  - REE_assembly/docs/architecture/sd_033_pfc_subdivision_architecture.md
  - REE_assembly/evidence/planning/task_inbox.md -- Prong D lit-pull task
  - ree_core/pfc/lateral_pfc_analog.py -- SD-033a template
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


_NOT_IMPL_MSG = (
    "SD-033e FrontopolarAnalog is a V4-scope stub. Enabling "
    "use_frontopolar_analog=True without an implementation is not supported. "
    "Prerequisite: Prong D literature synthesis (see REE_assembly/evidence/"
    "planning/task_inbox.md line 21) and SD-033e design doc. The stub is "
    "provided to pin interface shape; the design doc will instantiate it."
)


@dataclass
class FrontopolarConfig:
    """SD-033e configuration (STUB; see module docstring).

    Attributes:
        use_frontopolar_analog: master switch (default False, backward-compatible).
            When False the module is a no-op. When True, methods raise
            NotImplementedError until the SD-033e design doc lands.
        counterfactual_value_dim: dimensionality of the MECH-264 head output.
            Scalar (1) if pairwise best-unchosen only; K if per-alternative.
            Default 1 matches the Boorman 2009 pairwise reading; the design
            doc will commit based on the Prong D synthesis.
        importance_vector_dim: dimensionality of the MECH-265 relative-
            importance vector. Matches the number of active goals. Default
            2 (dual-goal minimal case). V4 env extension will wire the
            dynamic K from GoalState.
        gateway_mode: reserved for medial/lateral BA 10 gateway selection
            (Burgess 2007). Either "continuous" (modulatory gain) or
            "discrete" (switch). Default "continuous"; design doc will
            commit.
        hidden_dim: hidden layer width for MECH-264 / MECH-265 heads
            (matches SD-033a hidden_dim default).
        disengagement_scale: clamp on |disengagement_bias| returned by
            compute_disengagement_bias(). Matches SD-033a bias_scale
            pattern. Default 0.1.
    """

    use_frontopolar_analog: bool = False
    counterfactual_value_dim: int = 1
    importance_vector_dim: int = 2
    gateway_mode: str = "continuous"
    hidden_dim: int = 32
    disengagement_scale: float = 0.1
    # ------------------------------------------------------------------
    # SD-033e V3-NARROW DE-COMMIT LEVER (2026-07-09). A DISTINCT flag from the
    # V4 master `use_frontopolar_analog` above. Instantiates ONLY function (1)
    # MECH-264 counterfactual-value + function (4) disengagement-for-exploration,
    # in a parameter-free geometric (NON-learned) form, as a commit-LATCH
    # de-commit / switch-propensity lever for the DURATION face of the
    # F-dominance conversion ceiling (MECH-439). See module docstring section
    # "SD-033e V3-NARROW DE-COMMIT LANDING". When False (default) every method
    # is bit-identical to the V4 stub. use_frontopolar_analog stays the V4
    # learned-head gate and is orthogonal; if BOTH are set, use_frontopolar_decommit
    # takes precedence in compute_counterfactual_value (the geometric V3 lever).
    use_frontopolar_decommit: bool = False
    # Gain on the entry-relative counterfactual-improvement release-pressure term
    # injected into NaturalCommitUrgencyRelease. 0.0 (default) => zero pressure =>
    # the flat-urgency contrast arm (bit-identical to the frontopolar-OFF lever).
    frontopolar_gain: float = 0.0


class FrontopolarAnalog(nn.Module):
    """SD-033e frontopolar-analog: counterfactual-value + importance monitor.

    V4-scope STUB. See module docstring for the full DESIGN LANDING STATE
    and interface rationale. Every method is a no-op when
    use_frontopolar_analog is False; every method raises NotImplementedError
    when True and called, until the SD-033e design doc lands.
    """

    def __init__(
        self,
        world_dim: int,
        goal_dim: int,
        action_dim: int,
        config: Optional[FrontopolarConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else FrontopolarConfig()
        self.world_dim = world_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim

        # Interface-shape parameter stubs. Registered (not optimized) so that
        # future design-doc instantiation can lift the guard without changing
        # the module's parameter tree. All Linear weights will be zeroed at
        # init to match the SD-033a zero-init contract. Until the design doc
        # commits to the exact topology these are placeholders only.
        hidden_dim = self.config.hidden_dim

        # MECH-264 counterfactual-value head:
        #   concat([z_world_chosen, z_world_alt, z_goal]) -> scalar (or [K-1]).
        self._cfv_input_dim = 2 * world_dim + goal_dim
        self._cfv_output_dim = self.config.counterfactual_value_dim
        self.counterfactual_value_head = nn.Sequential(
            nn.Linear(self._cfv_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._cfv_output_dim),
        )

        # MECH-265 relative-importance monitor:
        #   concat([z_world, z_goals_stack]) -> [K] softmax.
        self._imp_input_dim = world_dim + goal_dim * self.config.importance_vector_dim
        self._imp_output_dim = self.config.importance_vector_dim
        self.importance_monitor_head = nn.Sequential(
            nn.Linear(self._imp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._imp_output_dim),
        )

        # Zero-init last Linear layers (matches SD-033a contract).
        with torch.no_grad():
            self.counterfactual_value_head[-1].weight.zero_()
            self.counterfactual_value_head[-1].bias.zero_()
            self.importance_monitor_head[-1].weight.zero_()
            self.importance_monitor_head[-1].bias.zero_()

        # Diagnostics (not persisted, not part of state_dict contract).
        self._last_gate: float = 0.0
        self._last_counterfactual_value: float = 0.0
        self._last_importance_entropy: float = 0.0
        self._last_disengagement_bias: float = 0.0

    # ------------------------------------------------------------------
    # MECH-264 counterfactual-value head (STUB)
    # ------------------------------------------------------------------
    def compute_counterfactual_value(
        self,
        z_world_chosen: torch.Tensor,
        z_world_alt: torch.Tensor,
        z_goal: torch.Tensor,
        gate: float = 1.0,
    ) -> torch.Tensor:
        """MECH-264: counterfactual-value of unchosen alternative (STUB).

        Args:
            z_world_chosen: [batch, world_dim] latent of chosen trajectory endpoint.
            z_world_alt: [batch, world_dim] latent of best unchosen alternative.
            z_goal: [batch, goal_dim] current active goal representation.
            gate: write_gate("sd_033e") from SalienceCoordinator (V4).

        Returns:
            counterfactual_value: [batch, counterfactual_value_dim] tensor.
            In the V3-narrow de-commit path (use_frontopolar_decommit=True), a
            parameter-free NON-F goal-proximity ADVANTAGE (see below). In the
            no-op path (both flags False), zeros matching that shape. In the V4
            learned path (use_frontopolar_analog=True), raises NotImplementedError.
        """
        # SD-033e V3-NARROW DE-COMMIT LEVER (checked FIRST; precedence over the V4
        # learned path). MECH-264 counterfactual value = goal-proximity ADVANTAGE
        # of the best foregone alternative over the chosen trajectory endpoint, in
        # z_world space (z_goal is a z_world-space attractor -> goal_dim == world_dim):
        #     cfv = ||z_world_chosen - z_goal|| - ||z_world_alt - z_goal||
        # cfv > 0  <=>  the foregone alternative sits CLOSER to the active goal than
        # the committed trajectory endpoint. Parameter-free geometry: NO learned
        # head, NO training phase. The switch signal is sourced ENTIRELY from
        # goal-proximity, so it is NOT F-monotone -- the load-bearing mitigation
        # against the 709/711/713 input-side-reweighting wash-out (an F-derivative
        # signal washes one layer down; this is orthogonal to F). Boorman 2009
        # counterfactual-value reading. The chosen-vs-alt ADVANTAGE (not raw
        # alt-proximity) controls for a global goal-approach confound where the
        # whole world drifts toward the goal.
        if self.config.use_frontopolar_decommit:
            chosen = (
                z_world_chosen
                if z_world_chosen.dim() > 1
                else z_world_chosen.unsqueeze(0)
            )
            alt = z_world_alt if z_world_alt.dim() > 1 else z_world_alt.unsqueeze(0)
            goal = z_goal if z_goal.dim() > 1 else z_goal.unsqueeze(0)
            d_chosen = torch.linalg.vector_norm(chosen - goal, dim=-1)
            d_alt = torch.linalg.vector_norm(alt - goal, dim=-1)
            cfv = (d_chosen - d_alt).reshape(-1, 1)
            self._last_counterfactual_value = float(cfv.reshape(-1)[0].item())
            self._last_gate = float(gate)
            return cfv
        if not self.config.use_frontopolar_analog:
            batch = z_world_chosen.shape[0] if z_world_chosen.dim() > 1 else 1
            return torch.zeros(
                batch,
                self._cfv_output_dim,
                device=z_world_chosen.device,
                dtype=z_world_chosen.dtype,
            )
        raise NotImplementedError(_NOT_IMPL_MSG)

    # ------------------------------------------------------------------
    # SD-033e function (4): disengagement-for-exploration (V3-narrow scalar).
    # ------------------------------------------------------------------
    def compute_decommit_pressure(
        self, cfv_now: float, cfv_at_entry: float
    ) -> float:
        """Entry-relative counterfactual-improvement de-commit release pressure.

        The MECH-090-release-facilitating "disengagement for exploration" function
        (Mansouri 2015 lesion signature), rendered V3-narrow as a scalar pressure
        term injected into NaturalCommitUrgencyRelease.tick():

            pressure = frontopolar_gain * max(0, cfv_now - cfv_at_entry)

        ENTRY-RELATIVE / DERIVATIVE form (load-bearing): fires only when the best
        foregone alternative has IMPROVED in goal-proximity terms RELATIVE to the
        commit moment -- NOT when the alternative out-scores F in absolute level. A
        level-based form is an F-derivative and washes one layer down (the
        709/711/713 failure mode); the entry-relative derivative does not.

        Returns 0.0 when the lever is disabled (use_frontopolar_decommit False) or
        frontopolar_gain is 0.0 -- so the frontopolar-OFF / gain=0 contrast arm is
        bit-identical to the pure NaturalCommitUrgencyRelease lever.
        """
        if not self.config.use_frontopolar_decommit:
            self._last_disengagement_bias = 0.0
            return 0.0
        pressure = float(self.config.frontopolar_gain) * max(
            0.0, float(cfv_now) - float(cfv_at_entry)
        )
        self._last_disengagement_bias = pressure
        return pressure

    # ------------------------------------------------------------------
    # MECH-265 relative-importance monitor (STUB)
    # ------------------------------------------------------------------
    def compute_relative_importance(
        self,
        z_world: torch.Tensor,
        z_goals: torch.Tensor,
        gate: float = 1.0,
    ) -> torch.Tensor:
        """MECH-265: relative-importance over K>=2 active goals (STUB).

        Args:
            z_world: [batch, world_dim] current world latent.
            z_goals: [batch, K, goal_dim] stack of active goal latents.
            gate: write_gate("sd_033e") from SalienceCoordinator (V4).

        Returns:
            importance_vector: [batch, K] softmax over active goals.
            In no-op path, returns uniform 1/K. In enabled path, currently
            raises NotImplementedError.
        """
        if not self.config.use_frontopolar_analog:
            batch = z_world.shape[0] if z_world.dim() > 1 else 1
            k = self.config.importance_vector_dim
            return torch.full(
                (batch, k),
                1.0 / k,
                device=z_world.device,
                dtype=z_world.dtype,
            )
        raise NotImplementedError(_NOT_IMPL_MSG)

    # ------------------------------------------------------------------
    # Four-function downstream: disengagement bias (STUB)
    # ------------------------------------------------------------------
    def compute_disengagement_bias(
        self,
        counterfactual_value: torch.Tensor,
        importance_vector: torch.Tensor,
        gate: float = 1.0,
    ) -> torch.Tensor:
        """Disengagement-for-exploration bias (Mansouri 2015 lesion signature; STUB).

        Args:
            counterfactual_value: output of compute_counterfactual_value.
            importance_vector: output of compute_relative_importance.
            gate: write_gate("sd_033e") from SalienceCoordinator (V4).

        Returns:
            disengagement_bias: [batch] scalar in [-disengagement_scale, +].
            Positive bias lowers commit threshold (facilitates MECH-090
            release); negative bias raises it. In no-op path, returns zeros.
            In enabled path, currently raises NotImplementedError.
        """
        if not self.config.use_frontopolar_analog:
            batch = counterfactual_value.shape[0] if counterfactual_value.dim() > 0 else 1
            return torch.zeros(
                batch,
                device=counterfactual_value.device,
                dtype=counterfactual_value.dtype,
            )
        raise NotImplementedError(_NOT_IMPL_MSG)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset FP diagnostics. Called on episode boundary via REEAgent.reset()."""
        self._last_gate = 0.0
        self._last_counterfactual_value = 0.0
        self._last_importance_entropy = 0.0
        self._last_disengagement_bias = 0.0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "last_gate": self._last_gate,
            "last_counterfactual_value": self._last_counterfactual_value,
            "last_importance_entropy": self._last_importance_entropy,
            # V3-narrow de-commit: last entry-relative release pressure emitted.
            "last_decommit_pressure": self._last_disengagement_bias,
            "decommit_enabled": bool(self.config.use_frontopolar_decommit),
            "frontopolar_gain": float(self.config.frontopolar_gain),
            # V4 learned-head path still a stub unless use_frontopolar_analog.
            "stub": not self.config.use_frontopolar_decommit,
        }
