"""
Neuromodulation subsystem for REE-v3.

Currently implements:
  - SerotoninModule (MECH-203, MECH-204): tonic 5-HT state variable,
    benefit-salience tagging, and sleep-phase dynamics.
"""

from ree_core.neuromodulation.serotonin import SerotoninModule, SerotoninConfig

__all__ = ["SerotoninModule", "SerotoninConfig"]
