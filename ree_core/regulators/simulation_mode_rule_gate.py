"""
MECH-319: Simulation-Mode Rule-Write Gate (Categorical Replay Tag).

Architectural commitment (see REE_assembly/docs/architecture/
mech_319_simulation_mode_rule_gate.md):

  Substrate-level instantiation of MECH-094's hypothesis-tag categorical
  write gate at the rule-arbitration layer. MECH-094 names the
  architectural principle (categorical phi(z) write gate; simulation-
  mode-suppressed write channel); MECH-319 names the specific substrate
  instantiation: a unified categorical write gate, keyed to a
  simulation-mode tag, that suppresses arbitration-weight updates in
  the MECH-312 sub-mechanisms (gated_policy, lateral_pfc_analog,
  future arbitrators) during ghost / replay / DMN passes.

  Substrate-availability premise (well-anchored):
    Joo & Frank 2018 (Nat Rev Neurosci) -- SWR review; the discriminable
      sharp-wave-ripple machinery is the biological substrate that
      carries the categorical "this is replay, not waking" signal.
    Foster & Wilson 2006 (Nature) -- reverse replay; concrete structural
      marker (temporal-reverse compressed sequences during awake
      quiescence) demonstrating the substrate-level identifiability
      that MECH-319's categorical tag relies on.

  REE-novel functional consequence:
    The architectural commitment that downstream rule-write machinery
    uses the discriminable replay signature to suppress arbitration-
    weight updates is REE-specific and biologically unverified at the
    cellular / synaptic level. Pull 3 SYNTHESIS R1 verdict
    GENUINE-NOVELTY-CONFIRMED (conf 0.72).

DESIGN

  Single primitive method: effective_simulation_mode(simulation_mode,
  site) -> bool. Translates a caller-supplied simulation_mode tag and
  the configured falsifier flag into the final admit/block decision the
  arbitration-write site should observe.

  Truth table (master_on, admit_writes, caller_sim) -> output:
    OFF, *,     *      -> caller_sim                  (identity; bit-identical)
    ON,  False, False  -> False (admit waking write)
    ON,  False, True   -> True  (block simulation write -- MECH-319 normal)
    ON,  True,  False  -> False (admit waking; flag has no effect on waking)
    ON,  True,  True   -> False (admit simulation write -- V3-EXQ-543c
                                  falsifier; predicted to produce
                                  monomodal-collapse re-emergence)

  The gate is idempotent for waking calls (always returns False),
  so wiring it into existing waking call sites is bit-identical
  regardless of admit_writes. The falsifier-control asymmetry surfaces
  only when caller_sim=True (replay / ghost-goal probe / DMN paths).

  Per-site diagnostic counters expose the gate's firing pattern for
  experiment manifests:
    n_calls_total, n_waking_admitted, n_simulation_blocked,
    n_simulation_admitted (the falsifier path),
    plus per-site breakdown across {gated_policy, lateral_pfc, default}.

CALL-SITE WIRING (in REEAgent.select_action)

  GatedPolicy block: replace the literal simulation_mode=False with
    gate.effective_simulation_mode(False, site="gated_policy"). With
    MECH-319 OFF the call is identity; with MECH-319 ON + waking
    caller, identity again. The seam exists for V3-EXQ-543c to flip
    admit_writes and route a replay-driven invocation through the
    gated_policy.forward path.

  LateralPFCAnalog block: consult gate via
    eff_sim = gate.effective_simulation_mode(False, site="lateral_pfc")
    if eff_sim: skip lateral_pfc.update(...) for this tick
    else: proceed with the existing MECH-261 mode-conditioned EMA.
    Bit-identical for waking; falsifier wiring exposed for the
    V3-EXQ-543c artificial-write-channel-routing test.

MECH-094 INVARIANCE

  This substrate does NOT modify MECH-094, GatedPolicy.forward's
  simulation_mode argument semantics, or LateralPFCAnalog.update.
  The gate is a pre-call coordinator that wraps the simulation_mode
  argument that callers ALREADY pass. With MECH-319 disabled, every
  arbitration-write call site behaves bit-identically to its
  pre-MECH-319 form.

SCOPE BOUNDARY

  Substrate landing only. The V3-EXQ-543c falsifier experiment
  (artificial-write-channel-routing flag flipped + replay-driven call
  site) is downstream of this substrate AND the MECH-313 / MECH-314 /
  MECH-318 sibling substrates; that experiment is queued in a separate
  session per the arc_062_rule_apprehension_plan.md status table
  (GAP-K row, 2026-05-10).

Master switch: REEConfig.use_simulation_mode_rule_gate (default False).
With the flag off, agent.simulation_mode_rule_gate is None and every
integration site is a no-op. Backward compatible.

Inverse-debug flag: REEConfig.simulation_mode_rule_gate_admit_writes
(default False). True = V3-EXQ-543c artificial-write-channel-routing
falsifier control. Construction raises ValueError when admit_writes=True
without the master flag also on -- loud-not-silent guard against
mis-configuration (admit_writes is meaningless without the substrate
to gate).

Non-trainable: pure boolean / counter arithmetic, no gradient flow.

Reset per episode: clears diagnostic counters. The gate has no
persistent state across ticks beyond the counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Canonical call-site labels. Keep stable for diagnostic comparability
# across experiments. New consumers can pass arbitrary site strings;
# they will appear in the per-site breakdown.
SITE_GATED_POLICY = "gated_policy"
SITE_LATERAL_PFC = "lateral_pfc"
SITE_DEFAULT = "default"


@dataclass
class SimulationModeRuleGateConfig:
    """MECH-319 configuration.

    Attributes:
        use_simulation_mode_rule_gate : master switch. False = disabled
            (default, backward-compatible -- agent does not instantiate
            the gate; all call sites are no-op).
        admit_writes : V3-EXQ-543c falsifier control. False (default) =
            MECH-319 normal behaviour (simulation-mode tag suppresses
            arbitration writes). True = artificial-write-channel-routing
            mode -- simulation content IS admitted into rule_state /
            gated_policy / future arbitrators. Predicted to produce
            monomodal-collapse re-emergence per MECH-094 /
            MECH-319 generalisation. Validated only via paired
            ON / OFF arms in V3-EXQ-543c-successor experiments AFTER the
            MECH-313 / MECH-314 / MECH-318 substrates have landed.
    """

    use_simulation_mode_rule_gate: bool = False
    admit_writes: bool = False


@dataclass
class SimulationModeRuleGateDiagnostics:
    """Per-tick + per-site counters for experiment manifests.

    Aggregate counters across all sites:
      n_calls_total: every effective_simulation_mode call.
      n_waking_admitted: caller_sim=False -> output=False (waking write
        admitted; default and bit-identical for the current call sites).
      n_simulation_blocked: caller_sim=True, admit_writes=False ->
        output=True (MECH-319 normal: simulation write blocked).
      n_simulation_admitted: caller_sim=True, admit_writes=True ->
        output=False (V3-EXQ-543c falsifier: simulation write admitted
        despite the tag).

    Per-site dicts mirror the same counters keyed on site string.
    """

    n_calls_total: int = 0
    n_waking_admitted: int = 0
    n_simulation_blocked: int = 0
    n_simulation_admitted: int = 0
    per_site_calls: Dict[str, int] = field(default_factory=dict)
    per_site_waking_admitted: Dict[str, int] = field(default_factory=dict)
    per_site_simulation_blocked: Dict[str, int] = field(default_factory=dict)
    per_site_simulation_admitted: Dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.n_calls_total = 0
        self.n_waking_admitted = 0
        self.n_simulation_blocked = 0
        self.n_simulation_admitted = 0
        self.per_site_calls = {}
        self.per_site_waking_admitted = {}
        self.per_site_simulation_blocked = {}
        self.per_site_simulation_admitted = {}


class SimulationModeRuleGate:
    """MECH-319: unified simulation-mode write gate at the rule-arbitration layer.

    Pure-arithmetic regulator (no nn.Module inheritance, no learned
    parameters). Sibling to GABAergicDecayRegulator (SD-036) and
    BroadcastOverrideRegulator (SD-037) in the regulators package.

    Construction raises ValueError when admit_writes=True without the
    master flag on (loud-not-silent guard).
    """

    def __init__(self, config: Optional[SimulationModeRuleGateConfig] = None) -> None:
        self.config = config if config is not None else SimulationModeRuleGateConfig()
        if self.config.admit_writes and not self.config.use_simulation_mode_rule_gate:
            raise ValueError(
                "MECH-319: simulation_mode_rule_gate_admit_writes=True requires "
                "use_simulation_mode_rule_gate=True (master flag must be on for the "
                "falsifier control to have any effect). Mis-configured -- enable the "
                "master switch or set admit_writes=False."
            )
        self.diagnostics = SimulationModeRuleGateDiagnostics()

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------
    def effective_simulation_mode(
        self,
        simulation_mode: bool,
        site: str = SITE_DEFAULT,
    ) -> bool:
        """Translate caller_sim + admit_writes into the final admit/block decision.

        Args:
            simulation_mode : the caller-supplied tag indicating whether
                this write originates from a replay / ghost-goal / DMN
                path (True) or from waking action selection (False).
            site : free-form label for per-site diagnostic breakdown
                (canonical: SITE_GATED_POLICY, SITE_LATERAL_PFC,
                SITE_DEFAULT). New consumer call sites can pass arbitrary
                strings; they appear in per_site_* dicts.

        Returns:
            bool : the simulation_mode value the downstream call site
                should observe (True = block write, False = admit).

        Truth table (master_on, admit_writes, caller_sim) -> output:
            OFF, *,     *     -> caller_sim                (identity)
            ON,  False, False -> False                     (waking admit)
            ON,  False, True  -> True                      (sim block, normal)
            ON,  True,  False -> False                     (waking admit)
            ON,  True,  True  -> False                     (sim admit, V3-EXQ-543c)
        """
        # Master OFF: identity; do not advance counters (the gate is
        # architecturally absent in this regime, so its diagnostics are
        # not meaningful).
        if not self.config.use_simulation_mode_rule_gate:
            return bool(simulation_mode)

        caller_sim = bool(simulation_mode)
        # MECH-319 normal: simulation -> block. Falsifier: simulation -> admit.
        if caller_sim:
            output = not self.config.admit_writes
        else:
            # Waking call: always admit. Falsifier flag has no effect here.
            output = False

        # Update counters AFTER the decision is made.
        self._record_call(site=site, caller_sim=caller_sim, output=output)
        return output

    # ------------------------------------------------------------------
    # Diagnostics bookkeeping
    # ------------------------------------------------------------------
    def _record_call(self, site: str, caller_sim: bool, output: bool) -> None:
        d = self.diagnostics
        d.n_calls_total += 1
        d.per_site_calls[site] = d.per_site_calls.get(site, 0) + 1
        if not caller_sim:
            # Waking admit (output should be False; assertion-grade invariant).
            d.n_waking_admitted += 1
            d.per_site_waking_admitted[site] = d.per_site_waking_admitted.get(site, 0) + 1
        else:
            if output:
                # Simulation blocked (MECH-319 normal).
                d.n_simulation_blocked += 1
                d.per_site_simulation_blocked[site] = (
                    d.per_site_simulation_blocked.get(site, 0) + 1
                )
            else:
                # Simulation admitted (V3-EXQ-543c falsifier).
                d.n_simulation_admitted += 1
                d.per_site_simulation_admitted[site] = (
                    d.per_site_simulation_admitted.get(site, 0) + 1
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Per-episode reset: clears diagnostic counters.

        The gate has no persistent state across ticks beyond the
        counters, so per-episode reset is purely a diagnostic boundary.
        """
        self.diagnostics.reset()

    def get_state(self) -> Dict[str, object]:
        """Diagnostic snapshot for experiment manifests."""
        d = self.diagnostics
        return {
            "use_simulation_mode_rule_gate": self.config.use_simulation_mode_rule_gate,
            "admit_writes": self.config.admit_writes,
            "n_calls_total": d.n_calls_total,
            "n_waking_admitted": d.n_waking_admitted,
            "n_simulation_blocked": d.n_simulation_blocked,
            "n_simulation_admitted": d.n_simulation_admitted,
            "per_site_calls": dict(d.per_site_calls),
            "per_site_waking_admitted": dict(d.per_site_waking_admitted),
            "per_site_simulation_blocked": dict(d.per_site_simulation_blocked),
            "per_site_simulation_admitted": dict(d.per_site_simulation_admitted),
        }
