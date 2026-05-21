"""Q-053 agent-side PersistenceAppraisal computation contracts."""

from __future__ import annotations

import pytest

from ree_core.hippocampal.persistence_appraisal_compute import (
    PersistenceAppraisalComputeConfig,
    compute_agent_persistence_appraisal,
)
from ree_core.utils.config import (
    GhostGoalBankConfig,
    HippocampalConfig,
    PersistenceAppraisalComputeConfig,
)


def test_q053_inactive_goal_disengage_default() -> None:
    cfg = PersistenceAppraisalComputeConfig()
    out = compute_agent_persistence_appraisal(
        goal_active=False,
        goal_proximity=None,
        prior_completion_signal=1.0,
        e3_is_committed=True,
        e3_committed_now=True,
        cfg=cfg,
    )
    assert out.control_efficacy == 0.0
    assert out.goal_unattainability == 1.0


def test_q053_high_proximity_low_unattainability() -> None:
    cfg = PersistenceAppraisalComputeConfig()
    out = compute_agent_persistence_appraisal(
        goal_active=True,
        goal_proximity=0.9,
        prior_completion_signal=0.0,
        e3_is_committed=False,
        e3_committed_now=False,
        cfg=cfg,
    )
    assert out.goal_unattainability == pytest.approx(0.1)
    assert out.control_efficacy == 0.0


def test_q053_completion_and_commitment_raise_control() -> None:
    cfg = PersistenceAppraisalComputeConfig(
        completion_weight=0.5,
        commitment_weight=0.5,
    )
    out = compute_agent_persistence_appraisal(
        goal_active=True,
        goal_proximity=0.5,
        prior_completion_signal=1.0,
        e3_is_committed=True,
        e3_committed_now=False,
        cfg=cfg,
    )
    assert out.control_efficacy == 1.0
    assert out.goal_unattainability == 0.5


def test_q053_hippocampal_config_has_compute_block() -> None:
    cfg = HippocampalConfig()
    assert isinstance(cfg.persistence_appraisal_compute, PersistenceAppraisalComputeConfig)
    assert isinstance(cfg.ghost_goal_bank_config, GhostGoalBankConfig)
