"""Contract tests for SD-100 (ARC-032 / MECH-089): phase-aware ThetaBuffer.summary().

Design memo (the contract):
  REE_assembly/docs/architecture/sd_100_theta_buffer_phase_aware_summary.md

Contracts:
  C1: default-OFF bit-identical. With use_phase_weighted_summary=False,
      summary() is byte-identical to the pre-SD-100 flat mean, for both an
      empty buffer and a partially/fully filled one.
  C2: ON is order-sensitive. Two windows holding the SAME multiset of
      z_world values in a DIFFERENT order produce the SAME flat mean but a
      DIFFERENT phase-weighted summary -- the exact property a flat mean
      cannot provide by construction (the failure this SD fixes).
  C3: ON reduces toward uniform weighting as phase_concentration -> 0 (the
      mathematical link between the new kernel and the old flat mean).
  C4: single-entry buffer (T=1) does not divide by zero and returns that
      entry unchanged, both ON and OFF.
  C5: weights are strictly monotonically increasing with recency (t) for
      phase_concentration > 0 -- this is what guarantees C2 holds for an
      arbitrary reordering, not just the specific one tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.latent.theta_buffer import ThetaBuffer


def _buffer(use_phase_weighted_summary=False, phase_concentration=4.0, buffer_size=10):
    return ThetaBuffer(
        self_dim=4,
        world_dim=6,
        buffer_size=buffer_size,
        use_phase_weighted_summary=use_phase_weighted_summary,
        phase_concentration=phase_concentration,
    )


def _push(buf: ThetaBuffer, world_vecs):
    for v in world_vecs:
        buf.update(z_world=v, z_self=torch.zeros(1, buf.self_dim))


# ----------------------------------------------------------------------
# C1: default-OFF bit-identical
# ----------------------------------------------------------------------
def test_off_empty_buffer_matches_legacy_zeros():
    buf = _buffer(use_phase_weighted_summary=False)
    result = buf.summary()
    assert torch.equal(result, torch.zeros(1, buf.world_dim))


def test_off_summary_is_bit_identical_to_flat_mean():
    torch.manual_seed(0)
    vecs = [torch.randn(1, 6) for _ in range(7)]

    buf = _buffer(use_phase_weighted_summary=False)
    _push(buf, vecs)
    got = buf.summary()

    expected = torch.stack(vecs, dim=0).mean(dim=0)
    assert torch.equal(got, expected)


def test_off_default_constructor_arg_matches_explicit_false():
    # Constructor default (no kwarg passed) must behave identically to
    # explicitly passing use_phase_weighted_summary=False.
    torch.manual_seed(1)
    vecs = [torch.randn(1, 6) for _ in range(5)]

    buf_default = ThetaBuffer(self_dim=4, world_dim=6, buffer_size=10)
    buf_explicit = _buffer(use_phase_weighted_summary=False)
    _push(buf_default, vecs)
    _push(buf_explicit, vecs)

    assert torch.equal(buf_default.summary(), buf_explicit.summary())


# ----------------------------------------------------------------------
# C2: ON is order-sensitive (the core fix)
# ----------------------------------------------------------------------
def test_reordering_changes_summary_when_phase_weighted_on():
    torch.manual_seed(2)
    vecs = [torch.randn(1, 6) for _ in range(6)]
    reordered = list(reversed(vecs))

    # Sanity: the flat mean is (numerically, up to float summation-order
    # rounding) identical for both orderings -- confirm the defect being
    # fixed still holds for the OFF path. Not torch.equal: summing floats in
    # a different order is not bit-exact even though it is the same multiset.
    buf_off_a = _buffer(use_phase_weighted_summary=False)
    buf_off_b = _buffer(use_phase_weighted_summary=False)
    _push(buf_off_a, vecs)
    _push(buf_off_b, reordered)
    assert torch.allclose(buf_off_a.summary(), buf_off_b.summary(), atol=1e-6)

    # ON: same multiset, different order -> different summary.
    buf_on_a = _buffer(use_phase_weighted_summary=True, phase_concentration=4.0)
    buf_on_b = _buffer(use_phase_weighted_summary=True, phase_concentration=4.0)
    _push(buf_on_a, vecs)
    _push(buf_on_b, reordered)

    summary_a = buf_on_a.summary()
    summary_b = buf_on_b.summary()
    assert not torch.allclose(summary_a, summary_b), (
        "phase-weighted summary must differ for differently-ordered windows "
        "holding the same z_world values"
    )
    # Both should still differ from the (order-invariant) flat mean.
    flat_mean = torch.stack(vecs, dim=0).mean(dim=0)
    assert not torch.allclose(summary_a, flat_mean)


# ----------------------------------------------------------------------
# C3: kappa -> 0 reduces to uniform weighting (mathematically = flat mean)
# ----------------------------------------------------------------------
def test_zero_concentration_matches_flat_mean_up_to_softmax_rounding():
    torch.manual_seed(3)
    vecs = [torch.randn(1, 6) for _ in range(8)]

    buf_on = _buffer(use_phase_weighted_summary=True, phase_concentration=0.0)
    _push(buf_on, vecs)
    on_result = buf_on.summary()

    flat_mean = torch.stack(vecs, dim=0).mean(dim=0)
    assert torch.allclose(on_result, flat_mean, atol=1e-6)


# ----------------------------------------------------------------------
# C4: single-entry buffer does not divide by zero
# ----------------------------------------------------------------------
def test_single_entry_buffer_both_modes():
    vec = torch.randn(1, 6)

    buf_off = _buffer(use_phase_weighted_summary=False)
    _push(buf_off, [vec])
    assert torch.equal(buf_off.summary(), vec)

    buf_on = _buffer(use_phase_weighted_summary=True, phase_concentration=4.0)
    _push(buf_on, [vec])
    assert torch.equal(buf_on.summary(), vec)


# ----------------------------------------------------------------------
# C5: weights strictly increase with recency for kappa > 0
# ----------------------------------------------------------------------
def test_phase_weights_strictly_increase_with_recency():
    buf = _buffer(use_phase_weighted_summary=True, phase_concentration=4.0)
    T = 5
    # Use orthogonal-ish unit vectors so each position's contribution to the
    # summary is separable: push e_0, e_1, ..., e_{T-1} (one-hot-ish) so the
    # resulting summary's entries ARE (proportional to) the position weights.
    vecs = [torch.eye(6)[i % 6].unsqueeze(0) for i in range(T)]
    _push(buf, vecs)
    result = buf.summary().squeeze(0)  # [6]

    # entries 0..T-1 of result correspond to weights w_0..w_{T-1} (one-hot
    # basis vectors were pushed in order, non-overlapping for T<=6).
    weights = result[:T]
    diffs = weights[1:] - weights[:-1]
    assert torch.all(diffs > 0), f"expected strictly increasing weights, got {weights.tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
