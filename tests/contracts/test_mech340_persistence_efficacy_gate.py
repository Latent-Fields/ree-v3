"""MECH-340 persistence/efficacy gate contracts (ree-v3 substrate)."""

from __future__ import annotations

import torch

from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload, AnchorSet
from ree_core.hippocampal.ghost_goal_bank import (
    GhostGoalBank,
    GhostGoalBankConfig,
    PersistenceAppraisal,
)
from ree_core.utils.config import AnchorSetConfig


def _anchor(seg: str, zsnap: list[float], last_vs: float = 0.6) -> Anchor:
    a = Anchor(key=("fast", seg, ("s",)), z_world=torch.zeros(4), active=False)
    a.goal_payload = AnchorGoalPayload(
        z_goal_snapshot=torch.tensor(zsnap, dtype=torch.float32).unsqueeze(0),
        wanting_strength=0.3,
        arousal_tag=0.1,
        last_vs=last_vs,
    )
    return a


def _bank(anchors: list[Anchor], cfg: GhostGoalBankConfig) -> GhostGoalBank:
    s = AnchorSet(AnchorSetConfig())
    s._all = {a.key: a for a in anchors}
    return GhostGoalBank(cfg, s)


Z_GOAL = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_mech340_gate_off_bit_identical_priorities() -> None:
    """Master switch off ignores appraisal and matches pre-MECH-340 ranks."""
    a = _anchor("A", [1.0, 0.0, 0.0, 0.0])
    b = _anchor("B", [0.2, 1.0, 0.0, 0.0])
    cfg = GhostGoalBankConfig()
    bank = _bank([a, b], cfg)
    off = bank.rank(Z_GOAL)
    disengage = bank.rank(
        Z_GOAL,
        persistence_appraisal=PersistenceAppraisal(
            control_efficacy=0.0,
            goal_unattainability=1.0,
        ),
    )
    assert len(off) == 2
    assert len(disengage) == 2
    assert off[0].ghost_priority == disengage[0].ghost_priority
    assert off[1].ghost_priority == disengage[1].ghost_priority
    assert "persistence_license" not in off[0].components


def test_mech340_high_control_low_unattainability_admits() -> None:
    """Persistence licensed -> anchors above floor remain in the bank."""
    a = _anchor("A", [1.0, 0.0, 0.0, 0.0])
    cfg = GhostGoalBankConfig(use_persistence_efficacy_gate=True)
    bank = _bank([a], cfg)
    entries = bank.rank(
        Z_GOAL,
        persistence_appraisal=PersistenceAppraisal(
            control_efficacy=1.0,
            goal_unattainability=0.0,
        ),
    )
    assert len(entries) == 1
    assert entries[0].components["persistence_license"] == 1.0


def test_mech340_disengaged_excludes_all() -> None:
    """Default-disengage appraisal excludes every anchor."""
    a = _anchor("A", [1.0, 0.0, 0.0, 0.0])
    cfg = GhostGoalBankConfig(
        use_persistence_efficacy_gate=True,
        persistence_floor=0.05,
    )
    bank = _bank([a], cfg)
    assert bank.rank(
        Z_GOAL,
        persistence_appraisal=PersistenceAppraisal(
            control_efficacy=0.0,
            goal_unattainability=1.0,
        ),
    ) == []
    diag = bank.get_diagnostics()
    assert diag["n_below_persistence"] == 1


def test_mech340_recoverability_does_not_affect_gate() -> None:
    """Same appraisal -> same admission regardless of recoverability."""
    hi = _anchor("Hi", [1.0, 0.0, 0.0, 0.0], last_vs=0.95)
    lo = _anchor("Lo", [1.0, 0.0, 0.0, 0.0], last_vs=0.05)
    cfg = GhostGoalBankConfig(use_persistence_efficacy_gate=True)
    appraisal = PersistenceAppraisal(
        control_efficacy=0.0,
        goal_unattainability=1.0,
    )
    bank = _bank([hi, lo], cfg)
    assert len(bank.rank(Z_GOAL, persistence_appraisal=appraisal)) == 0
    # Stuck-on failure mode: unattainable but control=1 -> license 0; excludes.
    stuck = PersistenceAppraisal(control_efficacy=1.0, goal_unattainability=1.0)
    cfg_stuck = GhostGoalBankConfig(
        use_persistence_efficacy_gate=True,
        persistence_floor=0.5,
    )
    bank2 = _bank([hi, lo], cfg_stuck)
    assert len(bank2.rank(Z_GOAL, persistence_appraisal=stuck)) == 0
