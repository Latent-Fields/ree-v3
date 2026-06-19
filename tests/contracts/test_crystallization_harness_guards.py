"""Contract: the crystallization-necessity harness guards (MECH-334 / INV-074;
experiments/_metrics.py) FIRE on a deliberately-broken config and PASS on a
correct one.

These guards extract the 655-lineage `_assert_fixes_wired` preflight into the
shared harness so the next MECH-334 retest (a copy-and-modify of 655) cannot
silently re-introduce the 610c-655 no-op:
  (1) assert_policy_trained        -- policy genuinely trained before crystallize()
  (2) assert_ewc_penalty_live      -- EWC penalty is a live differentiable term
      assert_ewc_term_in_loss      -- EWC penalty is actually added to the loss
  (3) assert_true_negative_arm0    -- ARM_0 control has every diversity floor OFF
      assert_d2_control_collapsed  -- (post-run) the control measurably collapsed

The load-bearing assertions in this file are the FIRE-ON-BROKEN cases: each broken
config raises HarnessGuardError. The pass-on-correct cases pin that the guard does
not false-positive a well-wired retest.
"""

import math
import sys
from pathlib import Path

# Repo-root on sys.path so `from experiments._metrics import ...` works both under
# pytest (conftest also does this) and when run standalone as a smoke script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from experiments._metrics import (  # noqa: E402
    HarnessGuardError,
    assert_policy_trained,
    assert_ewc_penalty_live,
    assert_ewc_term_in_loss,
    assert_true_negative_arm0,
    assert_d2_control_collapsed,
    TRUE_NEGATIVE_ARM0_CONTRACT,
)


# --------------------------------------------------------------------------- #
# Minimal throwaway residue-field stand-in for the EWC-live guard.            #
# Quadratic EWC penalty around the snapshotted anchor, depends on rbf weights #
# + centers, so backward() puts a non-zero gradient on the params.            #
# --------------------------------------------------------------------------- #

class _FakeRBF:
    def __init__(self, n: int = 4, dim: int = 8):
        self.centers = torch.zeros(n, dim, requires_grad=True)
        self.weights = torch.zeros(n, requires_grad=True)
        self.active_mask = torch.ones(n, dtype=torch.bool)
        self._anchor_w = self.weights.detach().clone()
        self._anchor_c = self.centers.detach().clone()


class _FakeResidueField:
    def __init__(self, *, anchored: bool = True, lam: float = 0.1):
        self.rbf_field = _FakeRBF()
        self.ewc_anchored = anchored
        self._lam = float(lam)

    def ewc_penalty(self):
        rbf = self.rbf_field
        dw = (rbf.weights - rbf._anchor_w) * rbf.active_mask.float()
        dc = rbf.centers - rbf._anchor_c
        return self._lam * ((dw ** 2).sum() + (dc ** 2).sum())


def _make_params_and_snapshot():
    """A tiny policy + its pre-train snapshot."""
    lin = torch.nn.Linear(8, 4)
    params = [p for p in lin.parameters() if p.requires_grad]
    snapshot = [p.detach().clone() for p in params]
    return lin, params, snapshot


# --------------------------------------------------------------------------- #
# Guard 1: assert_policy_trained                                              #
# --------------------------------------------------------------------------- #

def test_policy_trained_passes_on_moved_weights():
    lin, params, snapshot = _make_params_and_snapshot()
    with torch.no_grad():  # simulate a real training step moving the weights
        for p in params:
            p.add_(0.05)
    out = assert_policy_trained(params, snapshot, grad_seen=True,
                                trained_action_entropy=0.90,
                                untrained_entropy_ceiling=1.04)
    assert out["policy_trained"] is True
    assert out["policy_weight_delta"] > 1e-4


def test_policy_trained_FIRES_on_frozen_policy():
    # 610c/610d signature: crystallize() fired on a policy that never moved.
    lin, params, snapshot = _make_params_and_snapshot()
    # params unchanged -> zero weight delta.
    with pytest.raises(HarnessGuardError, match="weight delta"):
        assert_policy_trained(params, snapshot, grad_seen=True)


def test_policy_trained_FIRES_on_no_gradient_seen():
    lin, params, snapshot = _make_params_and_snapshot()
    with torch.no_grad():
        for p in params:
            p.add_(0.05)
    with pytest.raises(HarnessGuardError, match="grad_seen=False"):
        assert_policy_trained(params, snapshot, grad_seen=False)


def test_policy_trained_FIRES_on_uniform_action_distribution():
    # Weights moved but the learned action distribution stayed ~uniform (ln(5)).
    lin, params, snapshot = _make_params_and_snapshot()
    with torch.no_grad():
        for p in params:
            p.add_(0.05)
    with pytest.raises(HarnessGuardError, match="action entropy"):
        assert_policy_trained(params, snapshot, grad_seen=True,
                              trained_action_entropy=math.log(5),
                              untrained_entropy_ceiling=1.04)


def test_policy_trained_FIRES_on_empty_param_list():
    with pytest.raises(HarnessGuardError, match="empty parameter list"):
        assert_policy_trained([], [], grad_seen=True)


# --------------------------------------------------------------------------- #
# Guard 2a: assert_ewc_penalty_live                                          #
# --------------------------------------------------------------------------- #

def test_ewc_penalty_live_passes_when_armed():
    field = _FakeResidueField(anchored=True, lam=0.1)
    out = assert_ewc_penalty_live(field)  # perturbs weights to force a penalty
    assert out["ewc_penalty_live"] is True
    assert out["ewc_penalty_value"] > 0.0
    assert out["ewc_residue_grad_sum"] > 0.0


def test_ewc_penalty_live_FIRES_when_not_anchored():
    # snapshot_ewc_anchor() never called.
    field = _FakeResidueField(anchored=False, lam=0.1)
    with pytest.raises(HarnessGuardError, match="ewc_anchored"):
        assert_ewc_penalty_live(field)


def test_ewc_penalty_live_FIRES_when_lambda_zero():
    # residue_ewc_lambda=0 -> penalty inert -> adding to loss is a no-op.
    field = _FakeResidueField(anchored=True, lam=0.0)
    with pytest.raises(HarnessGuardError, match="not > "):
        assert_ewc_penalty_live(field)


# --------------------------------------------------------------------------- #
# Guard 2b: assert_ewc_term_in_loss                                          #
# --------------------------------------------------------------------------- #

def test_ewc_term_in_loss_passes_when_added():
    out = assert_ewc_term_in_loss(loss_without_ewc=2.0, ewc_term=0.5,
                                  total_loss=2.5)
    assert out["ewc_term_in_loss"] is True


def test_ewc_term_in_loss_passes_with_tensors():
    l0 = torch.tensor(2.0)
    et = torch.tensor(0.5)
    lt = l0 + et
    out = assert_ewc_term_in_loss(l0, et, lt)
    assert out["ewc_term_value"] == pytest.approx(0.5)


def test_ewc_term_in_loss_FIRES_when_term_inert():
    with pytest.raises(HarnessGuardError, match="ewc_term="):
        assert_ewc_term_in_loss(loss_without_ewc=2.0, ewc_term=0.0,
                                total_loss=2.0)


def test_ewc_term_in_loss_FIRES_when_term_dropped():
    # 610c/610d: the penalty was computed but never summed into the total loss.
    with pytest.raises(HarnessGuardError):
        assert_ewc_term_in_loss(loss_without_ewc=2.0, ewc_term=0.5,
                                total_loss=2.0)


# --------------------------------------------------------------------------- #
# Guard 3: assert_true_negative_arm0 + assert_d2_control_collapsed            #
# --------------------------------------------------------------------------- #

def _clean_arm0():
    return dict(label="ARM_0_stripped_control", **TRUE_NEGATIVE_ARM0_CONTRACT)


def test_true_negative_arm0_passes_on_clean_control():
    out = assert_true_negative_arm0(_clean_arm0())
    assert out["arm0_is_true_negative"] is True
    assert out["arm0_violations"] == []


def test_true_negative_arm0_FIRES_on_noise_floor_on():
    # 610e confound: the "control" quietly carried a MECH-313 noise floor.
    arm = _clean_arm0()
    arm["use_noise_floor"] = True
    with pytest.raises(HarnessGuardError, match="use_noise_floor"):
        assert_true_negative_arm0(arm)


def test_true_negative_arm0_FIRES_on_entropy_bonus_on():
    arm = _clean_arm0()
    arm["entropy_bonus_phase3"] = 0.02
    with pytest.raises(HarnessGuardError, match="entropy_bonus_phase3"):
        assert_true_negative_arm0(arm)


def test_true_negative_arm0_FIRES_on_e3_diversity_on():
    arm = _clean_arm0()
    arm["use_e3_diversity"] = True
    with pytest.raises(HarnessGuardError, match="use_e3_diversity"):
        assert_true_negative_arm0(arm)


def test_true_negative_arm0_FIRES_on_crystallize_on():
    arm = _clean_arm0()
    arm["crystallize"] = True
    with pytest.raises(HarnessGuardError, match="crystallize"):
        assert_true_negative_arm0(arm)


def test_d2_control_collapsed_passes_on_genuine_collapse():
    out = assert_d2_control_collapsed(end_phase2_entropy=0.80,
                                      end_phase3_entropy=0.60)
    assert out["d2_collapsed"] is True
    assert out["d2_delta"] == pytest.approx(0.20)


def test_d2_control_collapsed_FIRES_on_no_collapse():
    # 655 substrate_ceiling signature: control did not collapse (delta < 0.10).
    with pytest.raises(HarnessGuardError, match="control-collapse delta"):
        assert_d2_control_collapsed(end_phase2_entropy=0.80,
                                    end_phase3_entropy=0.75)


if __name__ == "__main__":  # standalone smoke runner (ASCII-only output)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_fire = sum(1 for f in fns if "FIRES" in f.__name__)
    failed = []
    for f in fns:
        try:
            f()
        except Exception as exc:  # noqa: BLE001
            failed.append((f.__name__, repr(exc)))
    if failed:
        for name, exc in failed:
            print("FAIL", name, exc)
        print("[crystallization-harness-guards] %d/%d FAILED" % (len(failed), len(fns)))
        sys.exit(1)
    print("[crystallization-harness-guards] %d/%d PASS (%d fire-on-broken)"
          % (len(fns), len(fns), n_fire))
