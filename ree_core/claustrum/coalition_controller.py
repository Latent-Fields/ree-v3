"""
CoalitionController -- SD-091/MECH-481 minimum-viable coalition-control
primitive (star-topology recruit/suppress/gain, G_t).

Spec: REE_assembly/docs/architecture/sd_091_coalition_topology_control.md,
Section 1 ("Minimum-viable coalition-control primitive") and the
"Minimum-viable V3 implementation path" (steps 1-3 of that path -- this
module + control_demand.py + coalition_templates.py; steps 4-5, wiring real
consumer read sites and REEAgent.select_action integration, are explicitly
NOT done here, see the module-level "Scope" note below).

Guardrails (verbatim intent from sd_091_coalition_topology_control.md
Section 4, restated here as structural properties this module enforces --
NOT reminders, contract-tested in
tests/contracts/test_claustrum_coalition_controller.py):

  - Does NOT claim the claustrum is the seat of consciousness or
    metacognition. The claustrum correspondence motivating this design is
    functional/hypothesis-generating (Krimmel et al Network Instantiation
    in Cognitive Control account), NOT anatomical, and not asserted to BE
    metacognition -- only to plausibly contribute to its enactment side.
  - Does NOT collapse coalition control into global broadcasting:
    `participating`/`suppressed` are always sparse per-type dicts (only the
    targets a specific ControlDemandType's template names), never an
    all-target default. If CoalitionController's own un-configured/disabled
    state ever behaved like "everything recruited," that would be the exact
    failure mode MECH-481's Arm 3 (untyped coalition) exists to test
    AGAINST -- see test_c_no_global_broadcast_default.
  - Does NOT grant a central router unrestricted representational or
    governance authority: this class never touches representational
    content, never computes reward, and the write-gate composition formula
    (coalition_gate below) is structurally attenuation-only -- participating
    and suppressed weights are clamped to [0, 1] at CoalitionState
    construction, so `coalition_gate(target) <= 1.0` always, for every
    target, in every reachable state. A coalition can lower or leave
    unchanged a target's effective write-through; it can never raise it
    above the mode-gate-only baseline. This is what makes "coalition
    suppression of the E3 commitment monitor target can only lower
    commit-readiness, never force-open the commit boundary" (SD-034/
    MECH-090 preservation) a property of the arithmetic, not a convention
    callers have to remember.
  - Persistence/dissolution (Gamma_t) is real and configurable, not a
    permanent no-op: a coalition dissolves when EITHER
    completion_condition(agent_state) returns True OR
    (current_tick - opened_tick) >= max_duration_ticks, whichever fires
    first (tick() below). The MVP's falsifier does not score persistence
    dynamics directly (scenario 5 is explicitly deferred per MECH-481's own
    scoping) but both conditions are real and measured here.
  - Keeps coalition instantiation distinguishable from mode classification:
    CoalitionController never writes into a SalienceCoordinator (this
    module does not import or reference ree_core.cingulate at all -- the
    one-directional dependency the doc's Section 3 requires is enforced by
    this module simply having no path to reach back into
    SalienceCoordinator's mode selection or update_signal() registry).

Scope (why this pass stops after steps 1-3): sd_091_coalition_topology_
control.md's implementation path also calls for (4) wiring named consumer
read sites (E1/E2 sensory path, hippocampal anchor/persistence, BetaGate/
MECH-090) to call `coalition.write_gate(name)` and compose it with
SalienceCoordinator.write_gate(name), and (5) integrating
`coalition.tick()` into REEAgent.select_action. Those steps touch the live
agent loop across several existing modules and are deliberately left for a
follow-up pass -- this module is fully self-contained and inert until
something calls request_coalition(), so landing steps 1-3 alone carries no
behavioural risk to the existing agent (nothing in ree_core imports
ree_core.claustrum yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from ree_core.claustrum.control_demand import ControlDemandType
from ree_core.claustrum.coalition_templates import COALITION_TEMPLATES


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _clamp01_dict(d: Dict[str, float]) -> Dict[str, float]:
    return {k: _clamp01(v) for k, v in d.items()}


def _clamp_nonneg_dict(d: Dict[str, float]) -> Dict[str, float]:
    return {k: (float(v) if v >= 0.0 else 0.0) for k, v in d.items()}


@dataclass
class CoalitionState:
    """One live, per-tick coalition instance (G_t's star-topology payload).

    participating / suppressed weights are clamped to [0, 1] in
    __post_init__ regardless of caller input -- this is the structural
    enforcement of the "attenuation-only" guardrail (see module docstring).
    channel_gain is clamped to >= 0.0 only (it is a temporary gain
    multiplier scoped to this coalition, a PARAMETRIC side-effect the
    coalition may request -- see coalition_templates.py's
    e3_candidate_count example -- not part of the write-gate axis, so it is
    not subject to the <= 1.0 ceiling).
    """

    demand_type: ControlDemandType
    participating: Dict[str, float] = field(default_factory=dict)
    suppressed: Dict[str, float] = field(default_factory=dict)
    channel_gain: Dict[str, float] = field(default_factory=dict)
    opened_tick: int = 0
    max_duration_ticks: int = 1
    completion_condition: Optional[Callable[[Any], bool]] = None

    def __post_init__(self) -> None:
        self.participating = _clamp01_dict(self.participating)
        self.suppressed = _clamp01_dict(self.suppressed)
        self.channel_gain = _clamp_nonneg_dict(self.channel_gain)
        if self.max_duration_ticks < 1:
            self.max_duration_ticks = 1

    def coalition_gate(self, target_name: str) -> float:
        """coalition_gate(target) = participating.get(target, 1.0) *
        (1 - suppressed.get(target, 0.0)) -- the exact composition formula
        from sd_091_coalition_topology_control.md Section 1. Absence from
        both dicts is a no-op multiplier of 1.0.
        """
        p = self.participating.get(target_name, 1.0)
        s = self.suppressed.get(target_name, 0.0)
        return _clamp01(p * (1.0 - s))

    def should_dissolve(self, current_tick: int, agent_state: Any = None) -> bool:
        if agent_state is not None and self.completion_condition is not None:
            try:
                if bool(self.completion_condition(agent_state)):
                    return True
            except Exception:
                # A misbehaving completion_condition must not crash the
                # agent loop -- fall through to the timeout floor, which is
                # the Gamma_t "hard timeout" this class exists to guarantee.
                pass
        return (current_tick - self.opened_tick) >= self.max_duration_ticks


@dataclass
class CoalitionControllerConfig:
    """Master switch defaults to False -- bit-identical off (nothing in
    ree_core calls CoalitionController yet, so this is currently inert
    regardless, but the internal switch is kept so the module is
    self-testably no-op even once a future consumer imports it before the
    REEConfig-level `use_coalition_controller` flag from the doc's Section 3
    is wired -- that REEConfig flag is a follow-up-pass concern, not
    duplicated here).
    """

    enabled: bool = False
    types_enabled: FrozenSet[ControlDemandType] = field(
        default_factory=lambda: frozenset(COALITION_TEMPLATES.keys())
    )
    # Placeholder magnitude -- doc Section 3: "default TBD by whoever builds
    # this, informed by typical trial length in the MECH-481 battery." No
    # battery exists yet (that is step 7, explicitly out of scope here), so
    # this is a round placeholder, overridable per-request via
    # request_coalition(..., max_duration_ticks=...).
    default_max_duration_ticks: int = 50
    channel_gain_scale: float = 1.0


class CoalitionController:
    """Injection-only coalition controller (doc Section 1: "Injection, not
    derivation"). Does NOT implement Monitor/Classify (MECH-481 sequence
    steps 1-2) -- request_coalition() is the single entry point; whatever
    upstream signal wants to trigger a coalition calls it.
    """

    def __init__(self, config: Optional[CoalitionControllerConfig] = None):
        self._config = config or CoalitionControllerConfig()
        self._active: List[CoalitionState] = []
        self._current_tick: int = 0
        # Diagnostic counter, not a crash path: request_coalition() on an
        # unregistered/untemplated ControlDemandType increments this and
        # returns None (doc Section 2: "callers requesting an unregistered
        # type get a no-op with a diagnostic counter, not a crash, mirroring
        # the codebase's safe no-op default convention").
        self.unregistered_request_count: int = 0

    @property
    def config(self) -> CoalitionControllerConfig:
        return self._config

    def reset(self) -> None:
        """Clear all active coalitions and the tick clock on episode boundary
        (mirrors SalienceCoordinator.reset()/BetaGate.reset() -- added when
        this module was wired into REEAgent.select_action; not needed while
        the module was self-contained and unimported). Does not clear
        unregistered_request_count (a lifetime diagnostic counter, matching
        the convention of similar counters elsewhere in ree_core)."""
        self._active = []
        self._current_tick = 0

    @property
    def active_coalitions(self) -> List[CoalitionState]:
        return list(self._active)

    # -- MECH-481 injection API --

    def request_coalition(
        self,
        demand_type: ControlDemandType,
        tick: int,
        context: Any = None,
        max_duration_ticks: Optional[int] = None,
        completion_condition: Optional[Callable[[Any], bool]] = None,
    ) -> Optional[CoalitionState]:
        """Instantiate a coalition from COALITION_TEMPLATES[demand_type].

        Returns None (no-op) if the controller is disabled, if
        `demand_type` is not in `types_enabled`, or if no template exists
        for it yet (8 of the 10 ControlDemandType members -- see
        control_demand.py). `context` is accepted for forward-compatibility
        with a future typed-uncertainty classifier but is not read by this
        MVP -- the doc's own scope statement: "this claim commits only to
        the architectural necessity of the graph-valued output existing at
        all," not an automatic classifier.
        """
        if not self._config.enabled:
            return None
        if demand_type not in self._config.types_enabled:
            self.unregistered_request_count += 1
            return None
        template = COALITION_TEMPLATES.get(demand_type)
        if template is None:
            self.unregistered_request_count += 1
            return None

        duration = (
            max_duration_ticks
            if max_duration_ticks is not None
            else self._config.default_max_duration_ticks
        )
        scale = self._config.channel_gain_scale
        state = CoalitionState(
            demand_type=demand_type,
            participating=dict(template.participating),
            suppressed=dict(template.suppressed),
            channel_gain={k: v * scale for k, v in template.channel_gain.items()},
            opened_tick=tick,
            max_duration_ticks=duration,
            completion_condition=completion_condition,
        )
        self._active.append(state)
        return state

    # -- Gamma_t: persistence / dissolution --

    def tick(self, current_tick: int, agent_state: Any = None) -> None:
        """Advance the controller's clock and dissolve any coalition whose
        completion_condition(agent_state) fires or whose
        max_duration_ticks timeout has elapsed -- whichever comes first.
        No-op when disabled (mirrors request_coalition's master switch, so
        a disabled controller never accumulates or dissolves state).

        Intended call site (doc Section 3, NOT wired by this pass):
        REEAgent.select_action calls `coalition.tick(current_tick)` after
        `coordinator.tick()` (SalienceCoordinator), before consumer sites
        resolve their effective recruitment weight.
        """
        if not self._config.enabled:
            return
        self._current_tick = current_tick
        self._active = [
            state
            for state in self._active
            if not state.should_dissolve(current_tick, agent_state)
        ]

    # -- Consumer-facing accessors --

    def write_gate(self, target_name: str) -> float:
        """Deliberately named to match SalienceCoordinator.write_gate(name)
        (MECH-261) -- doc Section 3: "two independent registries with the
        same call shape, composed multiplicatively at each consumer site,
        not merged into one dict." Returns 1.0 (full no-op pass) when
        disabled or when no active coalition names `target_name`, and the
        product of every active coalition's coalition_gate(target_name)
        otherwise -- each factor is already clamped to [0, 1]
        (CoalitionState.__post_init__), so the product is too.
        """
        if not self._config.enabled or not self._active:
            return 1.0
        gate = 1.0
        for state in self._active:
            gate *= state.coalition_gate(target_name)
        return _clamp01(gate)

    def channel_gain(self, target_name: str) -> float:
        """Product of every active coalition's channel_gain multiplier for
        `target_name` (default 1.0 = no adjustment). This is the PARAMETRIC
        side-effect axis (doc Section 2's e3_candidate_count example), kept
        deliberately separate from write_gate()'s attenuation-only
        composition -- a gain multiplier can legitimately exceed 1.0.
        """
        if not self._config.enabled or not self._active:
            return 1.0
        gain = 1.0
        for state in self._active:
            gain *= state.channel_gain.get(target_name, 1.0)
        return gain if gain >= 0.0 else 0.0

    def recruited_targets(self) -> FrozenSet[str]:
        """Union of every active coalition's `participating` keys -- the
        qualitative "which subsystems are in the currently active
        coalition" signal (doc's falsification signature #3: "log
        participating/suppressed per trial and diff"). Empty when disabled
        or no coalition is active -- never an implicit all-target set (see
        the no-global-broadcast guardrail in the module docstring).
        """
        if not self._config.enabled:
            return frozenset()
        out: set = set()
        for state in self._active:
            out.update(state.participating.keys())
        return frozenset(out)

    def suppressed_targets(self) -> FrozenSet[str]:
        """Union of every active coalition's `suppressed` keys. See
        recruited_targets() docstring."""
        if not self._config.enabled:
            return frozenset()
        out: set = set()
        for state in self._active:
            out.update(state.suppressed.keys())
        return frozenset(out)
