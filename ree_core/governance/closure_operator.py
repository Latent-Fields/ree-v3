"""SD-034: governance.closure_operator -- five-part "done" token.

When a committed rule-state reaches resolution, the ClosureOperator emits a
coordinated signal that:

  (a) releases the MECH-090 BetaGate commitment latch (if elevated),
  (b) installs a targeted No-Go on the just-completed action class via the
      MECH-260 dACC recency-suppression buffer,
  (c) discharges rule-domain residue (decay weights at nearby RBF centers --
      bounded multiplicative attenuation, NOT erasure; preserves the
      ResidueField invariant "cannot be erased" while implementing OCD-memo
      closure semantics),
  (d) emits a "closure_event" signal to the SD-032a SalienceCoordinator so
      the operating-mode vector can relax out of the committed regime,
  (e) caps and rebaselines the MECH-268 dACC precision-weighted PE buffer
      so the just-ended episode's residual PE does not continue driving
      control demand after closure.

MODE-CONDITIONING (critical falsifiability constraint per SD-034 spec):
Closure must only fire from rule-states installed in an operating mode that
gates sd_033a writes. Rule-states accumulated during internal_replay have
write_gate("sd_033a") ~= 0.05; their rule_state barely moves, so their
"completion" signal is weak by construction. We additionally require that
the current operating_mode be in allowed_closure_modes (default: external_task,
internal_planning) so that a rule installed during external_task cannot be
closed by a spurious completion during internal_replay.

COMPLETION DETECTOR:
Two entry points:
  1. Automatic: tick() monitors rule_state delta across consecutive calls.
     When ||rule_state_curr - rule_state_prev|| falls below
     completion_rule_delta_threshold for completion_stable_ticks consecutive
     ticks AND beta_gate is elevated AND mode-conditioning is satisfied, fires.
  2. Explicit: emit_closure(action_class, z_world) forces a closure, used by
     experiment scripts (EXP-0156 verified-but-not-released baseline) and by
     environment completion hooks.

FALSIFIABILITY (per SD-034 spec):
If any combination of MECH-090 / MECH-260 / MECH-094 tuning WITHOUT closure
produces the closure signature (beta release + targeted No-Go + residue
discharge + mode relaxation + PE cap all co-occur at rule-end), SD-034 is
over-specification. V3-EXQ-460 (verified-but-not-released) and V3-EXQ-466
(satisficing/residue discharge) test this by contrasting closure-ON vs
closure-OFF arms at the same event boundaries.

MECH-094: closure is a waking control-plane operation on real rule-states.
Internal_replay content cannot produce closure because the mode-conditioning
gate rejects it. Simulation/replay ticks are silently skipped.

Non-trainable: pure arithmetic and state mutation. No gradient flow.

See claim SD-034 (REE_assembly/docs/claims/claims.yaml) and plan
REE_assembly/evidence/planning/sd033_governance_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch


# Sentinel object used when the caller does not pass a parameter yet wants to
# distinguish "not provided" from "explicitly None". Keeps the tick() API
# tolerant to substrate absence without silent semantic drift.
class _Unset:
    pass


_UNSET = _Unset()


@dataclass
class ClosureOperatorConfig:
    """Configuration for SD-034 closure operator.

    All defaults preserve backward-compatible behaviour: when
    use_closure_operator=False (default on REEConfig) the operator is not
    instantiated and no hook in REEAgent.select_action() runs.

    Attributes
    ----------
    use_closure_operator : bool
        Master switch. Default False.
    completion_rule_delta_threshold : float
        Automatic-detector threshold on ||rule_state_t - rule_state_{t-1}||.
        Below this, the rule is considered stable. Default 0.001.
    completion_stable_ticks : int
        Number of consecutive stable ticks required before automatic closure
        fires. Prevents firing on single-step plateaus. Default 3.
    require_beta_elevated : bool
        If True, automatic closure only fires while beta is elevated
        (committed sequence). Default True.
    allowed_closure_modes : tuple[str, ...]
        Operating modes from which closure may fire. Default
        ("external_task", "internal_planning"). Replay / offline modes are
        excluded to enforce mode-conditioning (SD-034 falsifiability).
    min_sd033a_write_gate : float
        Additional gate on write_gate("sd_033a"). Closure only fires when
        the SD-033a write gate is at least this value at the time of
        firing. Default 0.5.
    nogo_injection_count : int
        How many copies of the completed action class to push onto the
        MECH-260 _action_history buffer. Default 3. Clipped by history
        memory (dacc_suppression_memory).
    residue_discharge_factor : float
        Multiplicative decay applied to RBF weights within the closure
        domain. 0.5 -> weights halved. Default 0.5. Must be in (0, 1].
        factor=1.0 disables discharge (acts as an ablation knob).
    residue_discharge_radius : float
        Radius (in RBF-bandwidth units) around the closure z_world within
        which centers are attenuated. Default 1.5.
    closure_signal_name : str
        Input-signal name written on the SalienceCoordinator when closure
        fires. Default "closure_event". Coordinator accepts arbitrary
        named signals; callers register affinity / salience weights for
        this signal via ClosureOperator.register_on_coordinator().
    closure_signal_value : float
        Magnitude of the closure_event signal. Default 1.0.
    reset_pe_ema : bool
        If True, the MECH-268 pe_ema on DACCAdaptiveControl is reset to
        None on closure (rebaselined). Default True.
    pe_cap_after_closure : Optional[float]
        If set, dACC pe is clamped to this absolute value from the tick
        after closure onward (via DACCConfig.dacc_pe_cap). Default None
        (no cap).
    diagnostic_logging : bool
        If True, closure events are logged to self._event_log. Default
        True (cheap -- just a list of ClosureEvent snapshots).
    """

    use_closure_operator: bool = False
    completion_rule_delta_threshold: float = 0.001
    completion_stable_ticks: int = 3
    require_beta_elevated: bool = True
    allowed_closure_modes: Tuple[str, ...] = (
        "external_task",
        "internal_planning",
    )
    min_sd033a_write_gate: float = 0.5
    nogo_injection_count: int = 3
    residue_discharge_factor: float = 0.5
    residue_discharge_radius: float = 1.5
    closure_signal_name: str = "closure_event"
    closure_signal_value: float = 1.0
    reset_pe_ema: bool = True
    pe_cap_after_closure: Optional[float] = None
    diagnostic_logging: bool = True


@dataclass
class ClosureEvent:
    """Diagnostic record of a single closure firing.

    Returned by tick() and emit_closure(). Fields fully describe which of
    the 5 signal components actually executed (e.g., beta release is a
    no-op if beta was not elevated; still recorded as fired=False).
    """

    fired: bool
    tick_index: int
    reason: str                       # "auto-stable" | "explicit" | "skipped:<cause>"
    action_class: Optional[int] = None
    current_mode: Optional[str] = None
    sd033a_gate: float = 0.0
    beta_was_elevated: bool = False
    beta_released: bool = False
    nogo_pushed: int = 0
    residue_centers_discharged: int = 0
    salience_signal_written: bool = False
    pe_ema_reset: bool = False
    pe_cap_applied: Optional[float] = None


class ClosureOperator:
    """SD-034 closure operator -- five-part coordinated "done" signal.

    Instantiated by REEAgent when config.use_closure_operator=True. Holds
    references to the participating substrates (beta_gate, dacc, residue,
    salience, lateral_pfc) and invokes their respective mutation methods
    when fire() is called.

    Stateful across ticks for automatic completion detection:
      _last_rule_state_norm     -- previous tick's rule_state norm
      _stable_tick_count        -- consecutive ticks below delta threshold
      _n_ticks / _n_closures    -- diagnostic counters
      _event_log                -- List[ClosureEvent] (if diagnostic_logging)

    The operator does NOT own rule_state, beta state, or residue state --
    it only reads / triggers writes on its references. All mutation is
    through documented public methods on those modules (beta_gate.release,
    dacc.inject_nogo, dacc.reset_episode_pe, residue.discharge_domain,
    salience.update_signal).
    """

    def __init__(
        self,
        config: Optional[ClosureOperatorConfig] = None,
        beta_gate=None,
        dacc=None,
        residue=None,
        salience=None,
        lateral_pfc=None,
    ) -> None:
        self.config = config if config is not None else ClosureOperatorConfig()
        self.beta_gate = beta_gate
        self.dacc = dacc
        self.residue = residue
        self.salience = salience
        self.lateral_pfc = lateral_pfc

        self._last_rule_state: Optional[torch.Tensor] = None
        self._stable_tick_count: int = 0
        self._n_ticks: int = 0
        self._n_closures: int = 0
        self._event_log: list[ClosureEvent] = []
        # Used to propagate pe_cap_after_closure into DACCConfig after firing.
        self._active_pe_cap: Optional[float] = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode detector state (keep config and refs)."""
        self._last_rule_state = None
        self._stable_tick_count = 0
        # _n_ticks / _n_closures / _event_log are cross-episode diagnostics.
        # Reset pe-cap side effect -- caller decides if cap persists across
        # episodes by re-setting config.pe_cap_after_closure.
        self._active_pe_cap = None

    # ------------------------------------------------------------------
    # Tick: automatic completion detector
    # ------------------------------------------------------------------
    def tick(
        self,
        current_z_world: torch.Tensor,
        current_action_class: Optional[int],
        current_mode: Optional[str] = None,
        sd033a_gate: Optional[float] = None,
        hypothesis_tag: bool = False,
    ) -> ClosureEvent:
        """Per-step closure-detection tick.

        Must be called AFTER lateral_pfc.update() so the rule_state buffer
        reflects the post-update value for this step. Typically called at
        the end of REEAgent.select_action() once per tick.

        Args:
            current_z_world: [batch, world_dim] or [world_dim]. Location at
                which residue discharge would be centred if closure fires.
                Passed through to fire() untouched.
            current_action_class: int action class (argmax of action[0])
                used as the MECH-260 No-Go injection target. None -> No-Go
                injection is skipped but other signals still fire.
            current_mode: str current discrete operating mode. Used for
                allowed_closure_modes gating. None -> read from salience
                coordinator if available, else assume external_task.
            sd033a_gate: float write_gate("sd_033a") at firing time.
                None -> read from salience if available.
            hypothesis_tag: MECH-094 replay gate. True -> automatic tick
                is a no-op (replay content cannot produce closure).

        Returns:
            ClosureEvent with fired=False when detector conditions are not
            met, True when the 5-part signal was emitted.
        """
        self._n_ticks += 1

        if not self.config.use_closure_operator:
            return ClosureEvent(
                fired=False, tick_index=self._n_ticks, reason="skipped:disabled"
            )

        if hypothesis_tag:
            return ClosureEvent(
                fired=False, tick_index=self._n_ticks, reason="skipped:hypothesis_tag"
            )

        # Read rule_state if available
        if self.lateral_pfc is None or not hasattr(self.lateral_pfc, "rule_state"):
            return ClosureEvent(
                fired=False, tick_index=self._n_ticks, reason="skipped:no_lateral_pfc"
            )

        rule_state = self.lateral_pfc.rule_state.detach()

        # Compute delta and update stability counter
        if self._last_rule_state is None:
            self._last_rule_state = rule_state.clone()
            return ClosureEvent(
                fired=False, tick_index=self._n_ticks, reason="skipped:first_tick"
            )

        delta_norm = float((rule_state - self._last_rule_state).norm().item())
        rule_norm = float(rule_state.norm().item())
        self._last_rule_state = rule_state.clone()

        # Require rule_state to have meaningful magnitude -- a zero
        # rule_state is trivially stable. This filters out "closure at
        # init" before any rule has been installed.
        if rule_norm < self.config.completion_rule_delta_threshold * 10.0:
            self._stable_tick_count = 0
            return ClosureEvent(
                fired=False,
                tick_index=self._n_ticks,
                reason="skipped:rule_state_unset",
            )

        if delta_norm < self.config.completion_rule_delta_threshold:
            self._stable_tick_count += 1
        else:
            self._stable_tick_count = 0

        if self._stable_tick_count < self.config.completion_stable_ticks:
            return ClosureEvent(
                fired=False,
                tick_index=self._n_ticks,
                reason=f"skipped:not_stable_yet({self._stable_tick_count}/{self.config.completion_stable_ticks})",
            )

        # Beta-elevation gate
        beta_elevated = False
        if self.beta_gate is not None:
            beta_elevated = bool(getattr(self.beta_gate, "is_elevated", False))
        if self.config.require_beta_elevated and not beta_elevated:
            return ClosureEvent(
                fired=False,
                tick_index=self._n_ticks,
                reason="skipped:beta_not_elevated",
                beta_was_elevated=False,
            )

        # Mode-conditioning: resolve current mode and gate
        mode = current_mode
        gate = sd033a_gate
        if self.salience is not None:
            if mode is None:
                mode = getattr(self.salience, "current_mode", "external_task")
            if gate is None:
                gate = self.salience.write_gate("sd_033a")
        if mode is None:
            mode = "external_task"
        if gate is None:
            gate = 1.0

        if mode not in self.config.allowed_closure_modes:
            return ClosureEvent(
                fired=False,
                tick_index=self._n_ticks,
                reason=f"skipped:mode_disallowed({mode})",
                current_mode=mode,
                sd033a_gate=float(gate),
                beta_was_elevated=beta_elevated,
            )
        if gate < self.config.min_sd033a_write_gate:
            return ClosureEvent(
                fired=False,
                tick_index=self._n_ticks,
                reason=f"skipped:gate_below_min({gate:.3f}<{self.config.min_sd033a_write_gate})",
                current_mode=mode,
                sd033a_gate=float(gate),
                beta_was_elevated=beta_elevated,
            )

        # Fire
        self._stable_tick_count = 0
        event = self._fire(
            reason="auto-stable",
            action_class=current_action_class,
            z_world=current_z_world,
            current_mode=mode,
            sd033a_gate=float(gate),
            beta_was_elevated=beta_elevated,
        )
        return event

    # ------------------------------------------------------------------
    # Explicit entry point
    # ------------------------------------------------------------------
    def emit_closure(
        self,
        action_class: Optional[int],
        z_world: torch.Tensor,
        current_mode: Optional[str] = None,
        sd033a_gate: Optional[float] = None,
        bypass_mode_conditioning: bool = False,
    ) -> ClosureEvent:
        """Force an immediate closure firing (experiment / env hook).

        Bypasses the automatic detector's stability check but DOES enforce
        mode-conditioning by default. Pass bypass_mode_conditioning=True
        for controlled ablations where closure must be emitted regardless
        of operating mode (e.g. EXP-0156 baseline arm forces closure at
        known event boundaries to dissociate detector latency from signal
        effect).
        """
        self._n_ticks += 1

        if not self.config.use_closure_operator:
            return ClosureEvent(
                fired=False, tick_index=self._n_ticks, reason="skipped:disabled"
            )

        mode = current_mode
        gate = sd033a_gate
        if self.salience is not None:
            if mode is None:
                mode = getattr(self.salience, "current_mode", "external_task")
            if gate is None:
                gate = self.salience.write_gate("sd_033a")
        if mode is None:
            mode = "external_task"
        if gate is None:
            gate = 1.0

        beta_elevated = False
        if self.beta_gate is not None:
            beta_elevated = bool(getattr(self.beta_gate, "is_elevated", False))

        if not bypass_mode_conditioning:
            if mode not in self.config.allowed_closure_modes:
                return ClosureEvent(
                    fired=False,
                    tick_index=self._n_ticks,
                    reason=f"skipped:mode_disallowed({mode})",
                    current_mode=mode,
                    sd033a_gate=float(gate),
                    beta_was_elevated=beta_elevated,
                )
            if gate < self.config.min_sd033a_write_gate:
                return ClosureEvent(
                    fired=False,
                    tick_index=self._n_ticks,
                    reason=f"skipped:gate_below_min({gate:.3f})",
                    current_mode=mode,
                    sd033a_gate=float(gate),
                    beta_was_elevated=beta_elevated,
                )

        return self._fire(
            reason="explicit" if not bypass_mode_conditioning else "explicit_bypass",
            action_class=action_class,
            z_world=z_world,
            current_mode=mode,
            sd033a_gate=float(gate),
            beta_was_elevated=beta_elevated,
        )

    # ------------------------------------------------------------------
    # Internal: the 5-part signal
    # ------------------------------------------------------------------
    def _fire(
        self,
        reason: str,
        action_class: Optional[int],
        z_world: torch.Tensor,
        current_mode: str,
        sd033a_gate: float,
        beta_was_elevated: bool,
    ) -> ClosureEvent:
        """Execute the 5-part coordinated signal. Records a ClosureEvent."""
        self._n_closures += 1

        beta_released = False
        nogo_pushed = 0
        residue_discharged = 0
        salience_written = False
        pe_ema_reset = False
        pe_cap_applied: Optional[float] = None

        # (a) MECH-090 BetaGate release
        if self.beta_gate is not None and beta_was_elevated:
            self.beta_gate.release()
            beta_released = True

        # (b) MECH-260 targeted No-Go injection
        if self.dacc is not None and action_class is not None:
            if hasattr(self.dacc, "inject_nogo"):
                nogo_pushed = int(
                    self.dacc.inject_nogo(
                        int(action_class), self.config.nogo_injection_count
                    )
                )
            elif hasattr(self.dacc, "record_action"):
                # Fallback: repeated record_action calls.
                for _ in range(self.config.nogo_injection_count):
                    self.dacc.record_action(int(action_class))
                nogo_pushed = self.config.nogo_injection_count

        # (c) Rule-domain residue discharge
        if self.residue is not None and hasattr(self.residue, "discharge_domain"):
            if self.config.residue_discharge_factor < 1.0:
                residue_discharged = int(
                    self.residue.discharge_domain(
                        z_world,
                        factor=self.config.residue_discharge_factor,
                        radius=self.config.residue_discharge_radius,
                    )
                )

        # (d) SalienceCoordinator closure_event signal
        if self.salience is not None and hasattr(self.salience, "update_signal"):
            self.salience.update_signal(
                self.config.closure_signal_name,
                self.config.closure_signal_value,
            )
            salience_written = True

        # (e) MECH-268 dACC pe cap / rebaseline
        if self.dacc is not None:
            if self.config.reset_pe_ema and hasattr(self.dacc, "reset_episode_pe"):
                self.dacc.reset_episode_pe()
                pe_ema_reset = True
            if self.config.pe_cap_after_closure is not None:
                # Install the cap on the DACCConfig for subsequent ticks.
                if hasattr(self.dacc, "config") and hasattr(self.dacc.config, "dacc_pe_cap"):
                    self.dacc.config.dacc_pe_cap = self.config.pe_cap_after_closure
                    pe_cap_applied = self.config.pe_cap_after_closure
                    self._active_pe_cap = pe_cap_applied

        event = ClosureEvent(
            fired=True,
            tick_index=self._n_ticks,
            reason=reason,
            action_class=action_class,
            current_mode=current_mode,
            sd033a_gate=sd033a_gate,
            beta_was_elevated=beta_was_elevated,
            beta_released=beta_released,
            nogo_pushed=nogo_pushed,
            residue_centers_discharged=residue_discharged,
            salience_signal_written=salience_written,
            pe_ema_reset=pe_ema_reset,
            pe_cap_applied=pe_cap_applied,
        )

        if self.config.diagnostic_logging:
            self._event_log.append(event)

        return event

    # ------------------------------------------------------------------
    # Optional: register closure_event on the salience coordinator so it
    # contributes to affinity logits. Callers wire this at agent init
    # if they want the closure signal to bias mode selection toward
    # internal_planning / offline_consolidation (mode relaxation).
    # ------------------------------------------------------------------
    def register_on_coordinator(
        self,
        affinity_modes: Optional[dict] = None,
        salience_weight: float = 0.0,
    ) -> None:
        """Register closure_event affinity + salience weights on SalienceCoordinator.

        Args:
            affinity_modes: dict[mode_name, logit_weight]. When closure
                fires, the coordinator's soft mode vector shifts toward
                these modes. Default: {"internal_planning": 0.5}
                (relaxes away from external_task toward internal reflection).
            salience_weight: contribution of closure_event to salience
                aggregate. Default 0.0 (closure does not trigger a switch
                on its own -- it biases mode affinity only).
        """
        if self.salience is None:
            return
        if affinity_modes is None:
            affinity_modes = {"internal_planning": 0.5}
        # SalienceCoordinator.config.affinity_weights is a Dict[str, Dict[str, float]]
        if hasattr(self.salience, "config"):
            aff = getattr(self.salience.config, "affinity_weights", None)
            if aff is not None:
                aff[self.config.closure_signal_name] = dict(affinity_modes)
            sal = getattr(self.salience.config, "salience_weights", None)
            if sal is not None and salience_weight != 0.0:
                sal[self.config.closure_signal_name] = float(salience_weight)
        # Also register the signal name on the input-signals dict so it is
        # seen as known and zeroed between firings.
        if hasattr(self.salience, "_input_signals"):
            self.salience._input_signals.setdefault(
                self.config.closure_signal_name, 0.0
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "n_ticks": self._n_ticks,
            "n_closures": self._n_closures,
            "stable_tick_count": self._stable_tick_count,
            "active_pe_cap": self._active_pe_cap,
            "last_event": (
                self._event_log[-1].__dict__ if self._event_log else None
            ),
        }

    @property
    def event_log(self) -> Sequence[ClosureEvent]:
        return tuple(self._event_log)
