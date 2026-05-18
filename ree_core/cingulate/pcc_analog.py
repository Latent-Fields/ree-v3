"""
PCCAnalog -- SD-032d posterior-cingulate-analog (attention partition / metastability).

SD-032d does NOT trigger mode switches directly (that is SD-032c's job). It
emits a scalar stability parameter in [0, 1] that modulates the MECH-259
switch threshold inside SD-032a. High stability -> coordinator resists mode
transitions; low stability -> transitions happen at lower salience.

The stability scalar is a function of three signals the agent already has:
  - recent task success (EMA, supplied by the experiment loop via
    note_task_outcome(); defaults to neutral 0.5 -- no signal -- when no
    outcome is fed in)
  - fatigue (SD-012 drive_level)
  - time since the last offline phase (auto-tracked; reset to 0 when
    note_offline_entry() is called from agent.enter_offline_mode())

Coordinates within-session (MECH-092 micro-quiescence) and cross-session
(INV-049 sleep) offline phases at a single integration point: both call
agent.enter_offline_mode() -> pcc.note_offline_entry().

Falsification signature (sd_032_cingulate_integration_substrate.md SD-032d):
  Ablating SD-032d (use_pcc_analog=False) makes the SalienceCoordinator
  effective_threshold insensitive to fatigue / time-since-offline. Agent
  over-commits to external_task without rest-driven relaxation of the
  threshold. The contrast is observable: with PCC ON, drive_level rises
  -> stability falls -> effective_threshold falls -> mode_switch_trigger
  rate rises under matched salience input.

Biological grounding (use conservatively -- Leech & Sharp 2013 is a
proposal, not consensus):
  Leech & Sharp 2013 (Brain 136:2013) -- "Arousal, Balance, Breadth"
    proposal: PCC tracks the global stability of the current cognitive
    set vs the need to broaden attentional sampling. The scalar reading
    here is the most conservative computational interpretation: a [0, 1]
    metastability index that biases the threshold for any mode change
    without committing to attention-partition geometry.
  Cross-session offline-phase coordination: Frankland & Bontempi 2005
    systems-consolidation framing -- the stability scalar drops with
    time since the last offline phase, so the system biases toward
    requesting offline consolidation when held externally for too long.

Non-trainable: pure arithmetic over scalars. EMA over success outcomes
plus a step counter. No gradient flow.

MECH-094: not applicable (waking arithmetic, no replay content authored).

See CLAUDE.md: SD-032d. Spec:
REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PCCConfig:
    """Configuration for SD-032d PCC-analog stability scalar.

    Defaults are conservative and produce a neutral stability around 0.5
    until the agent supplies task-outcome signals via note_task_outcome().
    """

    # EMA alpha for the task-success signal. ~50-step window matches the
    # MECH-205 / SD-032c convention.
    success_alpha: float = 0.02

    # Initial value of the success EMA. 0.5 = neutral (no information).
    # Avoids biasing the agent toward "successful" or "failing" at episode
    # start before any outcomes have been fed in.
    success_init: float = 0.5

    # Window length (in steps) over which steps-since-last-offline saturates
    # the offline_recency factor at 1.0. Smaller = stability decays faster
    # without rest. 500 steps is roughly the MECH-092 inter-quiescence
    # interval from V3 sleep experiments.
    offline_recency_window: int = 500

    # Centred contribution of success_ema to stability:
    #   stability += success_weight * (success_ema - 0.5)
    # Centred at 0.5 so a neutral history (no outcomes fed in) contributes
    # zero. With success_ema in [0, 1], this contribution is in
    # [-success_weight/2, +success_weight/2].
    success_weight: float = 0.5

    # Subtractive contribution of drive_level (SD-012 fatigue) to stability:
    #   stability -= fatigue_weight * drive_level
    # drive_level in [0, 1].
    fatigue_weight: float = 0.5

    # Subtractive contribution of offline_recency to stability:
    #   offline_recency = min(1.0, steps_since_offline / window)
    #   stability -= offline_weight * offline_recency
    # Drives the "need rest" signature -- long time since the last offline
    # phase lowers stability, lowering the switch threshold.
    offline_weight: float = 0.3

    # Additive baseline before clipping. With baseline=0.5 and all other
    # contributions zero, stability = 0.5 (neutral metastability).
    stability_baseline: float = 0.5


class PCCAnalog:
    """SD-032d posterior-cingulate-analog stability scalar producer.

    Stateful:
      _success_ema             -- EMA over task-outcome scalars in [0, 1].
      _steps_since_offline     -- step counter; reset by note_offline_entry().
      _last_pcc_stability      -- last computed stability scalar in [0, 1].
      _last_offline_recency    -- last computed offline_recency in [0, 1].
      _last_drive_level        -- last drive_level seen on tick().
      _n_ticks                 -- diagnostic counter.
      _n_offline_resets        -- diagnostic counter.
      _outcomes_fed            -- diagnostic counter (note_task_outcome calls).

    No gradient flow. Reset per episode via .reset(). Reset offline-recency
    on offline-mode entry via .note_offline_entry(). Feed task outcomes via
    .note_task_outcome(value).
    """

    def __init__(self, config: Optional[PCCConfig] = None):
        self.config = config or PCCConfig()
        self._success_ema: float = float(self.config.success_init)
        self._steps_since_offline: int = 0
        self._last_pcc_stability: float = float(self.config.stability_baseline)
        self._last_offline_recency: float = 0.0
        self._last_drive_level: float = 0.0
        self._n_ticks: int = 0
        self._n_offline_resets: int = 0
        self._outcomes_fed: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset().

        Does NOT reset _steps_since_offline -- offline-recency is a
        cross-episode signal (the agent did not just rest because a new
        episode began). It IS reset by note_offline_entry().
        """
        self._success_ema = float(self.config.success_init)
        self._last_pcc_stability = float(self.config.stability_baseline)
        self._last_offline_recency = 0.0
        self._last_drive_level = 0.0

    def note_offline_entry(self) -> None:
        """Reset the offline-recency counter.

        Called from agent.enter_offline_mode(); both MECH-092 within-session
        quiescence and INV-049 cross-session sleep paths flow through that
        single integration point.
        """
        self._steps_since_offline = 0
        self._n_offline_resets += 1

    def note_task_outcome(self, outcome: float) -> None:
        """Feed a per-step task-outcome scalar into the success EMA.

        Args:
            outcome: scalar in [0, 1]; 1.0 = clear success this step,
                0.0 = clear failure, 0.5 = neutral / no signal. Out-of-range
                values are clipped to [0, 1].

        Experiment loops choose what counts as success (e.g., 1.0 on
        benefit-collection, 0.0 on harm event, 0.5 otherwise). Without any
        calls, the EMA stays at success_init=0.5 and the success channel
        contributes zero to stability (neutral).
        """
        v = float(outcome)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        alpha = float(self.config.success_alpha)
        self._success_ema = (1.0 - alpha) * self._success_ema + alpha * v
        self._outcomes_fed += 1

    # -- Tick: main per-step computation --

    def tick(self, drive_level: float = 0.0) -> Dict[str, float]:
        """Compute pcc_stability for this step.

        Args:
            drive_level: SD-012 GoalState drive_level in [0, 1]. Higher =
                more depleted = lower stability.

        Returns a dict (all plain floats):
            pcc_stability     scalar in [0, 1]
            success_ema       current success EMA
            offline_recency   min(1, steps_since_offline / window)
            drive_level       drive_level (echo, for diagnostics)
            steps_since_offline  int counter (echo)
        """
        self._n_ticks += 1
        self._steps_since_offline += 1

        drive = float(drive_level)
        if drive < 0.0:
            drive = 0.0
        elif drive > 1.0:
            drive = 1.0
        self._last_drive_level = drive

        window = max(1, int(self.config.offline_recency_window))
        offline_recency = float(self._steps_since_offline) / float(window)
        if offline_recency > 1.0:
            offline_recency = 1.0
        self._last_offline_recency = offline_recency

        stability = float(self.config.stability_baseline)
        stability += float(self.config.success_weight) * (
            self._success_ema - 0.5
        )
        stability -= float(self.config.fatigue_weight) * drive
        stability -= float(self.config.offline_weight) * offline_recency

        if stability < 0.0:
            stability = 0.0
        elif stability > 1.0:
            stability = 1.0
        self._last_pcc_stability = stability

        return {
            "pcc_stability": float(stability),
            "success_ema": float(self._success_ema),
            "offline_recency": float(offline_recency),
            "drive_level": float(drive),
            "steps_since_offline": float(self._steps_since_offline),
        }

    # -- Read-only accessors --

    @property
    def pcc_stability(self) -> float:
        return float(self._last_pcc_stability)

    @property
    def success_ema(self) -> float:
        return float(self._success_ema)

    @property
    def offline_recency(self) -> float:
        return float(self._last_offline_recency)

    @property
    def steps_since_offline(self) -> int:
        return int(self._steps_since_offline)

    @property
    def diagnostics(self) -> Dict[str, int]:
        return {
            "n_ticks": self._n_ticks,
            "n_offline_resets": self._n_offline_resets,
            "outcomes_fed": self._outcomes_fed,
        }
