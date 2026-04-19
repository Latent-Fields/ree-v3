"""
SalienceCoordinator -- SD-032a salience-network coordinator.

Network-level coordinator that binds the cingulate subdivisions (SD-032a/b/c/d/e)
into a coherent salience network. Reads signals from all five subdivisions and
emits two outputs:

  (i)  operating_mode: a soft probability vector over
       {external_task, internal_planning, internal_replay, offline_consolidation}.
       Soft-weighted, NOT one-hot (Carr/Jadhav/Frank 2011 mixed SWR
       subpopulations; Tambini & Davachi 2019 cross-state persistence).
  (ii) mode_switch_trigger: boolean. Fires when (salience-weighted aggregate
       exceeds the MECH-259 threshold) AND argmax(operating_mode) differs from
       the current discrete mode. Modulated by the SD-032d stability scalar.

Also hosts MECH-261 mode-conditioned write gating: a dict-keyed registry of
per-target gate weights. Each target reads a soft-weighted sum over the mode
vector to produce a scalar gate value in [0, 1]. Dict-keyed on mode names so
that V4 parallel_goal_deliberation (SD-033e) can be added without schema
changes (see SD-033e note in claims.yaml MECH-261).

Architectural scope (SD-032a v3 minimum-viable):
  Inputs (live in V3):
    - SD-032b dACC bundle  : pe, foraging_value, choice_difficulty
    - SD-012 (proxy SD-032c): drive_level
    - Offline-mode flag (proxy SD-032d): bool
  Inputs (registered slots, default no-op until SD-032c/d/e land):
    - aic_salience    (SD-032c)  : scalar urgency salience
    - pcc_stability   (SD-032d)  : scalar in [0, 1]; high resists transitions
    - pacc_autonomic  (SD-032e)  : scalar autonomic write-back

Outputs:
    operating_mode      Dict[str, float]    soft probability vector
    current_mode        str                 discrete mode (hysteresis on switch)
    mode_switch_trigger bool                fires on threshold crossing + mode flip
    write_gate(target)  float in [0, 1]     MECH-261 per-target gate value

Biological grounding (REE_assembly/evidence/literature/
targeted_review_cingulate_integration_substrate/synthesis.md):
  Menon & Uddin 2010   -- AIC-dACC salience network switching DMN <-> CEN.
  Craig 2009           -- AIC as interoceptive-salience hub; mode-switch source.
  Carr/Jadhav/Frank 2011 -- soft-boundary write subpopulations during awake SWRs.
  Tambini & Davachi 2019 -- cross-state persistence, forward propagation bias.

Non-trainable: pure arithmetic over inputs. No gradient flow.

MECH-094: not produced here (coordinator does not author replay content). MECH-261
generalises MECH-094 -- per-target write-gate weights govern when content can
write to specific substrates as a function of operating mode.

See CLAUDE.md: SD-032a, MECH-259, MECH-261. Spec:
REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
"""

from dataclasses import dataclass, field
from math import exp
from typing import Dict, List, Optional


# Mode names. Stored as a list (not a fixed-arity tuple) so V4 modes can be
# appended without changing data-structure schemas downstream.
DEFAULT_MODE_NAMES: List[str] = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]


# MECH-261 default per-target gate weights. Replicates the table in
# sd_032_cingulate_integration_substrate.md -- "write" ~ 1.0, "consolidative"
# ~ 0.3 (slow low-rate write), "speculative write" / "reduced-gain write"
# ~ 0.5, "suppressed" ~ 0.05 (near-zero per spec).
DEFAULT_GATE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # SD-033 PFC subdivisions (see sd_033_pfc_subdivision_architecture.md)
    "sd_033a": {  # lateral-PFC rule/goal
        "external_task": 1.0,
        "internal_planning": 1.0,
        "internal_replay": 0.05,           # "suppressed" -- protect held rule
        "offline_consolidation": 0.3,
    },
    "sd_033b": {  # OFC state-space
        "external_task": 1.0,              # "read" -- treat as full read/write gate
        "internal_planning": 0.5,          # "speculative write"
        "internal_replay": 0.05,
        "offline_consolidation": 0.3,
    },
    "sd_033c": {  # vmPFC value
        "external_task": 1.0,
        "internal_planning": 0.5,
        "internal_replay": 0.5,            # "reduced-gain write"
        "offline_consolidation": 0.3,
    },
    "sd_033d": {  # premotor/SMA sequence (writes are MECH-094 tagged)
        "external_task": 1.0,
        "internal_planning": 1.0,
        "internal_replay": 0.05,           # "suppressed unless tag set"
        "offline_consolidation": 0.3,
    },
    # Hippocampal viability map (ARC-038)
    "hc_viability": {
        "external_task": 1.0,
        "internal_planning": 0.5,          # "read"
        "internal_replay": 1.0,
        "offline_consolidation": 0.3,
    },
    # Cortical-sensory buffer (Rothschild/Eban/Frank 2017 cortical replay)
    "sensory_buffer": {
        "external_task": 0.3,              # "read"
        "internal_planning": 0.3,          # "read"
        "internal_replay": 1.0,
        "offline_consolidation": 1.0,
    },
    # SD-032e autonomic coupling (active waking, attenuated otherwise)
    "autonomic": {
        "external_task": 1.0,
        "internal_planning": 0.3,
        "internal_replay": 0.3,
        "offline_consolidation": 0.3,
    },
    # E3 policy update direct gate
    "e3_policy": {
        "external_task": 1.0,
        "internal_planning": 0.5,          # "gated"
        "internal_replay": 0.05,           # "near-zero" -- propagate via SD-033a
        "offline_consolidation": 0.3,
    },
}


@dataclass
class SalienceCoordinatorConfig:
    """Configuration for SD-032a salience-network coordinator.

    All weights default to backward-compatible values:
      - Without inputs (zeros), softmax gives a near-uniform vector heavily
        biased toward external_task (the waking baseline).
      - mode_switch_trigger fires only when salience exceeds switch_threshold.
    """

    # Mode-affinity weighting per input signal. Each entry maps an input-signal
    # name to a per-mode logit contribution (added to the affinity logits before
    # softmax). Allows tuning without code change.
    # Default weights produce sensible v3 behaviour:
    #   - dACC pe        -> bias toward internal_planning (large PE = need to revise)
    #   - dACC foraging  -> bias toward internal_planning
    #   - dACC difficulty-> bias toward internal_planning
    #   - drive_level    -> bias toward external_task (high drive = must act)
    #   - is_offline     -> strong bias toward offline_consolidation
    # (No SD-032c/d/e mappings yet; those slots are no-op until landed.)
    affinity_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "dacc_pe":          {"internal_planning": 1.0, "internal_replay": 0.5},
        "dacc_foraging":    {"internal_planning": 1.0, "internal_replay": 0.3},
        "dacc_difficulty":  {"internal_planning": 0.5},
        "drive_level":      {"external_task": 1.0},
        "is_offline":       {"offline_consolidation": 5.0},
        # Stubs for future signals:
        "aic_salience":     {"internal_planning": 1.0},
        "pcc_stability":    {},   # only modulates threshold, not affinity
        "pacc_autonomic":   {"external_task": 0.3},
    })

    # Bias added to external_task logit before softmax. Ensures default mode is
    # external_task when all inputs are zero.
    external_task_bias: float = 1.0

    # Softmax temperature. Higher -> more uniform; lower -> sharper.
    softmax_temperature: float = 1.0

    # MECH-259 base switch threshold. mode_switch_trigger fires when
    # salience_aggregate > effective_threshold AND argmax(operating_mode) !=
    # current_mode.
    switch_threshold: float = 1.0

    # SD-032d stability scaling. effective_threshold = switch_threshold *
    # (1.0 + stability_scaling * pcc_stability). Higher stability -> higher
    # threshold -> harder to flip mode.
    stability_scaling: float = 1.0

    # Salience-aggregate weights -- which signals contribute to the salience
    # magnitude compared against the threshold. Separate from affinity_weights
    # because salience is "how loud is the alarm", whereas affinity is "which
    # mode does this argue for". Defaults: dACC PE + AIC salience drive the
    # urgency-interrupt pathway (per spec: SD-032c is the trigger source).
    salience_weights: Dict[str, float] = field(default_factory=lambda: {
        "dacc_pe":      1.0,
        "aic_salience": 1.0,
    })


class SalienceCoordinator:
    """SD-032a salience-network coordinator.

    Stateful over steps:
      _input_signals  -- registered named inputs, updated each tick
      _operating_mode -- last computed soft probability vector (Dict)
      _current_mode   -- discrete mode (str); flips only on threshold crossing
      _last_trigger   -- last mode_switch_trigger value
      _n_ticks        -- diagnostic counter
      _n_switches     -- diagnostic counter

    No gradient flow. Reset state via .reset() on episode boundaries.
    """

    def __init__(
        self,
        config: Optional[SalienceCoordinatorConfig] = None,
        mode_names: Optional[List[str]] = None,
        gate_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.config = config or SalienceCoordinatorConfig()
        self.mode_names: List[str] = list(mode_names) if mode_names else list(DEFAULT_MODE_NAMES)
        # Deep-ish copy of the gate registry so callers can mutate per-instance.
        src = gate_weights if gate_weights is not None else DEFAULT_GATE_WEIGHTS
        self._gate_weights: Dict[str, Dict[str, float]] = {
            target: dict(weights) for target, weights in src.items()
        }
        # Default input signal map: every known affinity / salience input
        # registered at zero so missing inputs are treated as no-op.
        self._input_signals: Dict[str, float] = {
            name: 0.0 for name in set(self.config.affinity_weights.keys())
            | set(self.config.salience_weights.keys())
        }
        # Initial operating mode: pure external_task (v3 waking baseline).
        self._operating_mode: Dict[str, float] = {
            m: (1.0 if m == "external_task" else 0.0) for m in self.mode_names
        }
        self._current_mode: str = "external_task"
        self._last_trigger: bool = False
        self._n_ticks: int = 0
        self._n_switches: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        for k in self._input_signals:
            self._input_signals[k] = 0.0
        self._operating_mode = {
            m: (1.0 if m == "external_task" else 0.0) for m in self.mode_names
        }
        self._current_mode = "external_task"
        self._last_trigger = False

    def update_signal(self, name: str, value: float) -> None:
        """Register or update a named input signal.

        Unknown names are accepted -- callers can register custom inputs that
        are then mapped via affinity_weights / salience_weights extension.
        """
        self._input_signals[name] = float(value)

    def register_target(self, name: str, weights: Dict[str, float]) -> None:
        """Add or replace a MECH-261 write-gate target.

        Allows V4 substrates (e.g., SD-033e parallel_goal_deliberation) to
        register their own gate profile without coordinator schema changes.
        """
        self._gate_weights[name] = dict(weights)

    # -- Tick: main per-step computation --

    def tick(
        self,
        dacc_bundle: Optional[dict] = None,
        drive_level: float = 0.0,
        is_offline: bool = False,
        extra_signals: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Compute operating_mode, mode_switch_trigger for this step.

        Args:
            dacc_bundle: SD-032b DACCAdaptiveControl.forward() output dict, or
                None if dACC is disabled.
            drive_level: SD-012 GoalState drive_level scalar in [0, 1].
            is_offline: True if the agent is in an offline phase
                (SWS / REM / quiescence). Strong affinity for
                offline_consolidation mode.
            extra_signals: Optional dict of additional named signals
                (aic_salience, pcc_stability, pacc_autonomic, ...).

        Returns a dict:
            operating_mode      Dict[str, float]   soft prob vector
            current_mode        str                discrete mode
            mode_switch_trigger bool               fired this tick
            salience_aggregate  float              raw salience magnitude
            effective_threshold float              MECH-259 threshold
        """
        self._n_ticks += 1

        # Pull dACC bundle scalars (zero if no bundle).
        if dacc_bundle is not None:
            self._input_signals["dacc_pe"] = float(dacc_bundle.get("pe", 0.0))
            self._input_signals["dacc_foraging"] = float(dacc_bundle.get("foraging_value", 0.0))
            self._input_signals["dacc_difficulty"] = float(dacc_bundle.get("choice_difficulty", 0.0))
        else:
            self._input_signals["dacc_pe"] = 0.0
            self._input_signals["dacc_foraging"] = 0.0
            self._input_signals["dacc_difficulty"] = 0.0
        self._input_signals["drive_level"] = float(drive_level)
        self._input_signals["is_offline"] = 1.0 if is_offline else 0.0
        if extra_signals:
            for k, v in extra_signals.items():
                self._input_signals[k] = float(v)

        # Per-mode affinity logit = external_task_bias on external_task only +
        # sum over signals of (signal_value * affinity_weights[signal][mode]).
        logits: Dict[str, float] = {m: 0.0 for m in self.mode_names}
        logits["external_task"] += self.config.external_task_bias
        for signal_name, mode_map in self.config.affinity_weights.items():
            value = self._input_signals.get(signal_name, 0.0)
            if value == 0.0:
                continue
            for mode, weight in mode_map.items():
                if mode in logits:
                    logits[mode] += value * weight

        # Softmax over logits with configurable temperature.
        operating_mode = self._softmax(logits, self.config.softmax_temperature)
        self._operating_mode = operating_mode

        # MECH-259 salience aggregate: weighted sum of urgency-relevant signals.
        salience_aggregate = 0.0
        for signal_name, weight in self.config.salience_weights.items():
            salience_aggregate += weight * self._input_signals.get(signal_name, 0.0)

        # SD-032d stability modulation of threshold.
        pcc_stability = self._input_signals.get("pcc_stability", 0.0)
        effective_threshold = self.config.switch_threshold * (
            1.0 + self.config.stability_scaling * pcc_stability
        )

        # MECH-259 switch trigger: salience exceeds threshold AND the soft
        # vector's argmax differs from the currently committed discrete mode.
        soft_argmax = max(operating_mode.items(), key=lambda kv: kv[1])[0]
        trigger = bool(
            salience_aggregate > effective_threshold
            and soft_argmax != self._current_mode
        )
        if trigger:
            self._current_mode = soft_argmax
            self._n_switches += 1
        self._last_trigger = trigger

        return {
            "operating_mode": dict(operating_mode),
            "current_mode": self._current_mode,
            "mode_switch_trigger": trigger,
            "salience_aggregate": float(salience_aggregate),
            "effective_threshold": float(effective_threshold),
        }

    # -- MECH-261 write-gating --

    def write_gate(self, target_name: str) -> float:
        """Soft-weighted gate value for a MECH-261 target.

        Returns sum_{mode} operating_mode[mode] * gate_weights[target][mode],
        clamped to [0, 1]. Returns 0.0 if target is not registered (deny-by-
        default for unknown substrates).
        """
        weights = self._gate_weights.get(target_name)
        if weights is None:
            return 0.0
        gate = 0.0
        for mode, prob in self._operating_mode.items():
            gate += prob * weights.get(mode, 0.0)
        if gate < 0.0:
            return 0.0
        if gate > 1.0:
            return 1.0
        return gate

    def write_gates(self) -> Dict[str, float]:
        """Compute all registered gates at once (for logging / diagnostics)."""
        return {name: self.write_gate(name) for name in self._gate_weights}

    # -- Read-only accessors --

    @property
    def operating_mode(self) -> Dict[str, float]:
        return dict(self._operating_mode)

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def last_trigger(self) -> bool:
        return self._last_trigger

    @property
    def gate_targets(self) -> List[str]:
        return list(self._gate_weights.keys())

    @property
    def diagnostics(self) -> Dict[str, int]:
        return {"n_ticks": self._n_ticks, "n_switches": self._n_switches}

    # -- Internal helpers --

    @staticmethod
    def _softmax(logits: Dict[str, float], temperature: float) -> Dict[str, float]:
        """Numerically stable softmax over a dict of logits."""
        if temperature <= 0:
            temperature = 1e-6
        max_logit = max(logits.values())
        exps = {k: exp((v - max_logit) / temperature) for k, v in logits.items()}
        z = sum(exps.values())
        if z <= 0:
            n = len(logits)
            return {k: 1.0 / n for k in logits}
        return {k: v / z for k, v in exps.items()}
