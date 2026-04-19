"""
Cingulate integration substrate for REE-v3 (SD-032 cluster).

Currently implements:
  - DACCAdaptiveControl (SD-032b, MECH-258, MECH-260): dACC/aMCC-analog
    adaptive-control module. Emits a Croxson-style integration bundle
    (mode_ev, choice_difficulty, foraging_value, harm_interaction)
    computed from the precision-weighted affective-pain PE (MECH-258) and
    adds a recency/monostrategy suppression bias (MECH-260).
  - DACCtoE3Adapter: STOPGAP adapter pending SD-032a (salience-network
    coordinator). Maps the integration bundle onto a per-candidate score
    bias for E3.select(). Marked for removal when SD-032a lands.

Future SD-032 siblings (not implemented here):
  SD-032a  vACC / salience-network coordinator
  SD-032c  pgACC / sgACC emotional regulation
  SD-032d  posterior cingulate / precuneus default-mode anchoring
  SD-032e  retrosplenial scene-context integration
"""

from ree_core.cingulate.dacc import (
    DACCAdaptiveControl,
    DACCConfig,
    DACCtoE3Adapter,
)

__all__ = ["DACCAdaptiveControl", "DACCConfig", "DACCtoE3Adapter"]
