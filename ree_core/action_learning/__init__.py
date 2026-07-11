"""Action-learning substrate (MECH-457).

First-class RPE-driven actor-critic action-learning system -- a dorsal-striatal-analog
actor + value-baseline critic (plain or successor-feature), architecturally distinct
from the lateral_pfc / ofc bias_head REINFORCE readout. Routed from
failure_autopsy_734-737-conversion-ceiling-competence_2026-07-11 (V3-EXQ-737: a
trainable actor+critic over the FROZEN z_world scored below random -> the frozen
prediction latent is action-inadequate; the action-learning loss must co-shape the
representation). Design doc:
REE_assembly/docs/architecture/sd_actor_critic_action_learning.md.
"""

from ree_core.action_learning.actor_critic import (
    ActorCriticPolicy,
    ActorCriticStep,
)

__all__ = [
    "ActorCriticPolicy",
    "ActorCriticStep",
]
