"""MECH-219: affective-harm hysteretic integrator (z_harm_suffering).

The tier-2 -> tier-3 step of the harm-affect hierarchy. SD-019a builds the
medium-timescale unpleasantness channel z_harm_un (a symmetric EMA of the
sensory harm stream z_harm_s). MECH-219 turns that "make it stop" unpleasantness
into a slow, persistent, controllability-gated SUFFERING load state via an
asymmetric (hysteretic) integrator -- the thing SD-019b
(harm_stream.suffering_accumulator) names and is blocked on.

```
z_harm_s   (SD-010/011)   fast nociception                       BUILT
   |  EMA alpha=0.2 (SD-019a, harm_un_ema)
z_harm_un  (SD-019a)      medium "make it stop" unpleasantness    BUILT
   |  MECH-219 controllability-gated hysteretic integration       THIS MODULE
z_harm_a   (suffering)    slow, persistent, controllability-gated z_harm_suffering
```

DISTINCTNESS (anti-duplication, see design memo Section 5):
  vs SD-019a z_harm_un (EMA): z_harm_un is SYMMETRIC and controllability-
    INDEPENDENT. MECH-219 is ASYMMETRIC/hysteretic (alpha_rise >> alpha_fall) and
    controllability-GATED (g = 1 - escapability). Turning escapability to the
    default constant=1.0 collapses g to 0 -> suffering does not accrue even at
    high unpleasantness -> bit-identical OFF.
  vs SD-022 body-damage z_harm_a (current): body-damage is an env-sourced slow
    EMA, NOT controllability-gated. MECH-219 re-sources the suffering signal from
    z_harm_un under a control gate; the body-damage contribution is FOLDED INTO
    the drive (body_damage_weight) rather than discarded, preserving the
    SD-022 / EXQ-319 / EXQ-323a non-redundancy evidence.
  vs MECH-353 blocked-agency (z_block): SAME controllability axis, OPPOSITE pole.
    MECH-353 is capacity-RETAINED ASSERT (high capacity_belief -> act); MECH-219
    is capacity-COLLAPSED WITHDRAW (low escapability -> suffer). They
    anti-correlate under a controllability sweep.

CONTROLLABILITY GATE g_t = 1 - escapability_t (Salomons 2004 / Loffler 2018):
  under full control (escapability=1) drive_t=0 -> suffering does not accrue even
  when unpleasantness u_t is high. This is the falsifiable dissociation. The
  escapability SOURCE is selected by the caller (constant / avoidance_efficacy /
  external); this regulator takes the resolved escapability scalar as an argument.

HYSTERESIS alpha_rise >> alpha_fall: suffering is sticky (fast to build under
  uncontrollable harm, slow to release on relief). The recovery-failure /
  persistence signature emerges HERE; it is NOT a separate input. Lit anchor:
  Baliki 2012 corticostriatal chronic-pain drift.

This is a pure-arithmetic regulator: no nn.Module, no learned parameters, no
gradient flow. Mirrors the SD-019a EMA-buffer precedent and the MECH-313 /
MECH-320 / MECH-342 / MECH-353 regulator pattern.

MECH-094: update() is a no-op under simulation_mode=True (replay / DMN must not
accumulate suffering on imagined outcomes); the prior s_t is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


VALID_ESCAPABILITY_MODES = ("constant", "avoidance_efficacy", "external")


@dataclass
class HarmSufferingAccumulatorConfig:
    """MECH-219 hysteretic-integrator configuration.

    All defaults produce the inert / maximally-relieving regime
    (escapability_mode=constant=1.0 -> g_t=0 -> s_t -> 0; latch off; pe_gain 0;
    body_damage_weight 0) so the integrator is bit-identical OFF.

    Attributes:
        use_harm_suffering_accumulator: master switch. False = disabled (default,
            backward-compatible); REEAgent does not instantiate the accumulator.
        alpha_rise: building rate when drive_t > s_{t-1}. Fast-up under
            uncontrollable harm. ~0.2 -> ~5-step build (matches z_harm_un).
        alpha_fall: releasing rate when drive_t <= s_{t-1}. Slow recovery.
            alpha_fall << alpha_rise is the hysteresis. ~0.01 -> ~100-step
            release (allostatic / chronic-pain signature).
        escapability_mode: source-selection label, validated against
            VALID_ESCAPABILITY_MODES. The caller resolves the actual escapability
            scalar per mode and passes it to update(); this label is stored for
            diagnostics and so a future session can swap sources without a module
            refactor. constant (default) is dependency-free; avoidance_efficacy
            reads SD-058 effective_efficacy() (the literal escapability construct);
            external lets a validation experiment drive it.
        escapability_constant: escapability value used by the caller in the
            constant mode (1.0 = fully escapable -> g=0 -> inert).
        s_cap: hard clamp on the accumulated suffering scalar s_t.
        body_damage_weight: SD-022 fold-in (design memo Section 6 fork b). When
            > 0, the body-damage magnitude (||z_harm_a||) is added to the drive
            u_t so the SD-022 non-redundancy evidence is preserved rather than
            orphaned. 0.0 (default) -> pure z_harm_un drive.
        pe_gain: optional SD-020 prediction-error driver (Q-036 secondary
            modulator). When > 0, pe_gain * unsigned_PE is added to drive_t AFTER
            the controllability gate. 0.0 (default) keeps PE out of the core
            commitment.
        use_bistable_latch: optional Schmitt-style latch (the "distinct load
            STATE" reading). When True, the output is s_t * latched where latched
            flips True above theta_on and False below theta_off. Default False
            (graded s_t alone).
        theta_on: Schmitt latch ON threshold (latched -> True when s_t > theta_on).
        theta_off: Schmitt latch OFF threshold (latched -> False when s_t <
            theta_off). Must be < theta_on for a hysteresis band.
    """

    use_harm_suffering_accumulator: bool = False
    alpha_rise: float = 0.2
    alpha_fall: float = 0.01
    escapability_mode: str = "constant"
    escapability_constant: float = 1.0
    s_cap: float = 2.0
    body_damage_weight: float = 0.0
    pe_gain: float = 0.0
    use_bistable_latch: bool = False
    theta_on: float = 0.5
    theta_off: float = 0.3


@dataclass
class HarmSufferingAccumulatorOutput:
    """Diagnostic snapshot for one HarmSufferingAccumulator.update() call."""

    s: float = 0.0                 # output suffering scalar (s_raw * latch)
    s_raw: float = 0.0             # graded accumulator before the optional latch
    drive: float = 0.0            # g_t * u_t + pe_gain * unsigned_PE
    u: float = 0.0                # uncontrolled unpleasantness magnitude (+ body)
    g: float = 0.0                # controllability gate 1 - escapability
    escapability: float = 1.0
    building: bool = False         # True when drive_t > s_{t-1} (rise branch)
    latched: bool = False
    simulation_skipped: bool = False


class HarmSufferingAccumulator:
    """MECH-219 controllability-gated hysteretic suffering integrator.

    Pure-arithmetic, no learned parameters, no nn.Module inheritance. Maintains
    the scalar suffering state s_t across waking ticks. The agent builds the
    z_harm_suffering LatentState vector (same dim as z_harm_un) by scaling the
    z_harm_un direction to magnitude s_t; this regulator owns only the scalar
    dynamics.
    """

    def __init__(
        self, config: Optional[HarmSufferingAccumulatorConfig] = None
    ) -> None:
        self.config = config if config is not None else HarmSufferingAccumulatorConfig()
        c = self.config
        # Validate (loud, not silent).
        if not (0.0 <= c.alpha_rise <= 1.0):
            raise ValueError(
                f"alpha_rise must be in [0, 1]. Got {c.alpha_rise}."
            )
        if not (0.0 <= c.alpha_fall <= 1.0):
            raise ValueError(
                f"alpha_fall must be in [0, 1]. Got {c.alpha_fall}."
            )
        if c.escapability_mode not in VALID_ESCAPABILITY_MODES:
            raise ValueError(
                "escapability_mode must be one of "
                f"{VALID_ESCAPABILITY_MODES}. Got {c.escapability_mode!r}."
            )
        if c.s_cap <= 0.0:
            raise ValueError(f"s_cap must be > 0. Got {c.s_cap}.")
        if c.body_damage_weight < 0.0:
            raise ValueError(
                f"body_damage_weight must be >= 0. Got {c.body_damage_weight}."
            )
        if c.pe_gain < 0.0:
            raise ValueError(f"pe_gain must be >= 0. Got {c.pe_gain}.")
        if c.use_bistable_latch and not (c.theta_off < c.theta_on):
            raise ValueError(
                "theta_off must be < theta_on for a Schmitt hysteresis band. "
                f"Got theta_off={c.theta_off}, theta_on={c.theta_on}."
            )
        # State.
        self._s: float = 0.0
        self._latched: bool = False
        self._last_output: HarmSufferingAccumulatorOutput = (
            HarmSufferingAccumulatorOutput()
        )
        # Diagnostics.
        self._n_waking_updates: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        unpleasantness_norm: float,
        escapability: float,
        body_damage_norm: float = 0.0,
        unsigned_pe: float = 0.0,
        simulation_mode: bool = False,
    ) -> HarmSufferingAccumulatorOutput:
        """Advance the suffering accumulator for one waking tick.

        Args:
            unpleasantness_norm: ||z_harm_un|| -- the SD-019a medium-timescale
                unpleasantness magnitude (>= 0). The drive.
            escapability: resolved controllability scalar in [0, 1] (1 = fully
                controllable -> g=0 -> no suffering accrual). The caller selects
                the source per escapability_mode and passes the value here.
            body_damage_norm: ||z_harm_a|| -- SD-022 body-damage magnitude folded
                into the drive when body_damage_weight > 0 (memo Section 6 fork b).
            unsigned_pe: optional SD-020 unsigned aversive prediction error, added
                to the drive scaled by pe_gain (Q-036 secondary modulator).
            simulation_mode: MECH-094 gate. When True, no state advances and the
                prior s_t is returned (replay must not accumulate suffering).
        """
        c = self.config

        if simulation_mode:
            self._n_simulation_skips += 1
            prev = self._last_output
            out = HarmSufferingAccumulatorOutput(
                s=self._s * (1.0 if (not c.use_bistable_latch or self._latched) else 0.0),
                s_raw=self._s,
                drive=0.0,
                u=0.0,
                g=prev.g,
                escapability=float(escapability),
                building=False,
                latched=self._latched,
                simulation_skipped=True,
            )
            return out

        # Drive: uncontrolled unpleasantness (+ optional body-damage fold-in).
        u = max(0.0, float(unpleasantness_norm))
        if c.body_damage_weight > 0.0:
            u = u + c.body_damage_weight * max(0.0, float(body_damage_norm))
        esc = max(0.0, min(1.0, float(escapability)))
        g = 1.0 - esc
        drive = g * u
        if c.pe_gain > 0.0:
            drive = drive + c.pe_gain * max(0.0, float(unsigned_pe))

        # Asymmetric (hysteretic) accumulation toward the drive target.
        building = drive > self._s
        alpha = c.alpha_rise if building else c.alpha_fall
        self._s = self._s + alpha * (drive - self._s)
        self._s = max(0.0, min(c.s_cap, self._s))

        # Optional Schmitt-style bistable latch (default off).
        if c.use_bistable_latch:
            if self._s > c.theta_on:
                self._latched = True
            elif self._s < c.theta_off:
                self._latched = False
            s_out = self._s if self._latched else 0.0
        else:
            s_out = self._s

        self._n_waking_updates += 1
        out = HarmSufferingAccumulatorOutput(
            s=s_out,
            s_raw=self._s,
            drive=drive,
            u=u,
            g=g,
            escapability=esc,
            building=building,
            latched=self._latched,
            simulation_skipped=False,
        )
        self._last_output = out
        return out

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_suffering(self) -> float:
        """Current output suffering scalar (s_raw * latch when latch enabled)."""
        if self.config.use_bistable_latch:
            return self._s if self._latched else 0.0
        return self._s

    def last_output(self) -> HarmSufferingAccumulatorOutput:
        return self._last_output

    def reset(self) -> None:
        """Reset per-episode state + diagnostic counters."""
        self._s = 0.0
        self._latched = False
        self._last_output = HarmSufferingAccumulatorOutput()
        self._n_waking_updates = 0
        self._n_simulation_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        o = self._last_output
        return {
            "mech219_s": self._s,
            "mech219_latched": self._latched,
            "mech219_last_s_out": o.s,
            "mech219_last_drive": o.drive,
            "mech219_last_u": o.u,
            "mech219_last_g": o.g,
            "mech219_last_escapability": o.escapability,
            "mech219_last_building": o.building,
            "mech219_escapability_mode": self.config.escapability_mode,
            "mech219_n_waking_updates": self._n_waking_updates,
            "mech219_n_simulation_skips": self._n_simulation_skips,
        }
