"""
Cingulate integration substrate for REE-v3 (SD-032 cluster).

Currently implements:
  - DACCAdaptiveControl (SD-032b, MECH-258, MECH-260): dACC/aMCC-analog
    adaptive-control module. Emits a Croxson-style integration bundle
    (mode_ev, choice_difficulty, foraging_value, harm_interaction)
    computed from the precision-weighted affective-pain PE (MECH-258) and
    adds a recency/monostrategy suppression bias (MECH-260).
  - DACCtoE3Adapter: STOPGAP adapter pending full SD-033 substrate landing.
    Maps the integration bundle onto a per-candidate score bias for
    E3.select(). With SD-032a now implemented, the coordinator can scale
    this bias via its e3_policy gate, but the adapter itself remains as
    the score_bias source until SD-033 substrates consume operating_mode
    natively.
  - SalienceCoordinator (SD-032a, MECH-259, MECH-261): network-level
    coordinator that aggregates the dACC bundle and homeostatic / offline
    signals into a soft operating-mode probability vector and a discrete
    mode-switch trigger. Hosts the MECH-261 dict-keyed write-gate
    registry. Reads slots for SD-032c/d/e signals (no-op until those land).
  - AICAnalog (SD-032c): anterior-insula-analog interoceptive-salience /
    urgency-interrupt module. Emits aic_salience (fed to the coordinator
    as the urgency-trigger source per MECH-259) and harm_s_gain (which
    subsumes SD-021 descending pain modulation: z_harm_s attenuation now
    gated on operating_mode + drive_level rather than raw beta_gate).
  - PCCAnalog (SD-032d): posterior-cingulate-analog metastability scalar.
    Non-trainable arithmetic over a success-outcome EMA + drive_level +
    steps-since-last-offline-phase. Emits pcc_stability in [0, 1] which
    the SalienceCoordinator multiplies into its MECH-259 effective
    threshold (high stability -> harder to switch). Coordinates within-
    session (MECH-092) and cross-session (INV-049) offline phases via the
    enter_offline_mode integration point.

Future SD-032 siblings (not implemented here):
  SD-032e  pACC-analog autonomic write-back to SD-012
"""

from ree_core.cingulate.aic_analog import AICAnalog, AICConfig
from ree_core.cingulate.dacc import (
    DACCAdaptiveControl,
    DACCConfig,
    DACCtoE3Adapter,
)
from ree_core.cingulate.pcc_analog import PCCAnalog, PCCConfig
from ree_core.cingulate.salience_coordinator import (
    DEFAULT_GATE_WEIGHTS,
    DEFAULT_MODE_NAMES,
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)

__all__ = [
    "AICAnalog",
    "AICConfig",
    "DACCAdaptiveControl",
    "DACCConfig",
    "DACCtoE3Adapter",
    "PCCAnalog",
    "PCCConfig",
    "SalienceCoordinator",
    "SalienceCoordinatorConfig",
    "DEFAULT_GATE_WEIGHTS",
    "DEFAULT_MODE_NAMES",
]
