"""Q-053 / MECH-340: agent-side persistence appraisal computation.

Maps one-shot goal-level signals into PersistenceAppraisal
(control_efficacy, goal_unattainability) for GhostGoalBank.rank().

Front-runner form (ghost_goal_search.md Section 0.3; Q-053 narrowed
2026-05-19): internal control/efficacy unattainability *appraisal*.
  - control_efficacy: prior hippocampal completion (planning quality) plus
    E3 commitment state (control / precision), not accumulated failure.
  - goal_unattainability: 1 - goal_proximity (instantaneous world-goal
    alignment appraisal); external/world invalidation demoted to an input
    here, not a separate gate.

Hard negatives (API shape): does NOT read staleness, recoverability,
wanting, failure tallies, or effort-cost proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ree_core.hippocampal.ghost_goal_bank import PersistenceAppraisal


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class PersistenceAppraisalComputeConfig:
    """Weights for agent-side PersistenceAppraisal (Q-053 smallest step)."""

    # Prior-tick hippocampal completion signal weight (planning/control).
    completion_weight: float = 0.6
    # E3 commitment weight (held plan or variance-below-threshold now).
    commitment_weight: float = 0.4


def compute_agent_persistence_appraisal(
    *,
    goal_active: bool,
    goal_proximity: Optional[float],
    prior_completion_signal: float,
    e3_is_committed: bool,
    e3_committed_now: bool,
    cfg: PersistenceAppraisalComputeConfig,
) -> PersistenceAppraisal:
    """Build a global PersistenceAppraisal from agent tick signals.

    When goal_active is False, returns the disengagement default
    (control=0, unattainability=1) per ARC-079 / MECH-340.

    When goal_active is True:
      goal_unattainability = clip(1 - goal_proximity)
      control_efficacy = clip(w_c * completion + w_k * commitment_signal)
    where commitment_signal is 1.0 if a trajectory is committed, else 1.0
    if variance is below the commit threshold this tick, else 0.0.
    """
    if not goal_active:
        return PersistenceAppraisal(control_efficacy=0.0, goal_unattainability=1.0)

    proximity = _clip01(goal_proximity if goal_proximity is not None else 0.0)
    unattainability = _clip01(1.0 - proximity)

    if e3_is_committed:
        commit_signal = 1.0
    elif e3_committed_now:
        commit_signal = 1.0
    else:
        commit_signal = 0.0

    completion = _clip01(prior_completion_signal)
    control_raw = (
        float(cfg.completion_weight) * completion
        + float(cfg.commitment_weight) * commit_signal
    )
    control = _clip01(control_raw)

    return PersistenceAppraisal(
        control_efficacy=control,
        goal_unattainability=unattainability,
    )
