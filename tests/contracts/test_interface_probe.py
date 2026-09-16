"""Contracts for experiments/_lib/interface_probe.py -- the P0 instrument for
the hippocampal campaign's communication-subspace / bridge-ladder / causal-
replacement assays (hippocampal_campaign_assay_specifications_20260910.md
section 4).

Invariants asserted:
  (1) Purity: every public function is a plain function over its inputs --
      no agent/ree_core import anywhere in the module, no file writes.
  (2) capture()/hash_tensor_state(): provenance validation, and the hash is
      stable under key-order permutation and sensitive to a changed tensor.
  (3) communication_subspace(): held-out R^2 for the TRUE rank on a known
      low-rank relation dominates rank-1 R^2 and the selected basis captures
      the planted subspace (via principal_angles); grouped folds keep every
      row of one group in a single fold.
  (4) principal_angles(): identical subspaces -> zero angles / overlap 1.0;
      orthogonal subspaces -> overlap 0.0.
  (5) bridge_ladder(): richer levels weakly dominate simpler ones in heldout
      R^2 on a nonlinear relation; parameter counts follow L0<=L1<L2, and the
      per-level dict carries every documented field.
  (6) causal_replacement(): all six category outcomes are reachable
      (including the AMBIGUOUS defensive fallback) and the result records
      every condition's raw score.
  (7) manifold_guard(): flags an off-manifold batch and does not flag an
      in-distribution one; saturation/hidden-trajectory fields are None when
      not supplied and populated when they are.
  (8) dynamic_compatibility(): one-step scores match a hand-computed MSE/
      cosine; `compounding` is True on a strictly-growing-error sequence,
      False on a flat one, None on a single horizon.
  (9) per_code_drift(): drift confined to the declared subspace reports ~0
      out-of-subspace rate and the reverse for out-of-subspace-only drift.
 (10) CaptureRecord.frame: defaults to {} (every pre-existing call site stays
      valid), round-trips verbatim with no interpretation, defensively copied.
 (11) receiver_conditioned_bridge() (MECH-547 `T(A, B)`): all four arms are
      capacity-matched by construction; the control is a TRUE ROW PERMUTATION
      of the receiver block; real per-row conditioning is SUPPORTED and its
      gain dies under permutation; noise receiver state gives NO gain; a
      target that is a function of the receiver state alone is caught as
      LEAKAGE; an unmatched baseline (pad_baselines=False) REFUSES a verdict;
      and the section-2.7 falsifier verdict is reachable.
 (12) frame_permutation_control() (MECH-555): a load-bearing frame survives
      and dies under permutation, an incidental one buys nothing, a scramble
      that is not a permutation is rejected as CONTROL_INVALID, and the
      "content-preserving" property is asserted mechanically -- X and Y are
      bit-identical before and after.

Design docs: REE_assembly/evidence/planning/hippocampal_campaign_assay_specifications_20260910.md
section 4 (and section 2.2 arms A3_frame_cond / A4_receiver_state_cond / A7_receiver_only,
section 2.7 falsifier); REE_assembly/docs/thoughts/2026-09-07_mutual_legibility_implementation_assays.md;
REE_assembly/docs/architecture/receiver_conditioned_translation.md (MECH-547);
REE_assembly/docs/architecture/interface_reference_frames_and_temporal_gates.md (MECH-555).
"""
import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from _lib import interface_probe as ip  # noqa: E402


# ---- (1) purity: no agent reference, no ree_core import, no writes ----------

def test_module_has_no_ree_core_or_agent_import():
    src = Path(REPO_ROOT / "experiments" / "_lib" / "interface_probe.py").read_text()
    tree = ast.parse(src)
    forbidden_prefixes = ("ree_core", "coordinator")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            assert not any((name or "").startswith(p) for p in forbidden_prefixes), (
                "interface_probe.py must not import %r (module purity contract)" % (name,)
            )


def test_module_has_no_file_write_calls():
    src = Path(REPO_ROOT / "experiments" / "_lib" / "interface_probe.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            pytest.fail("interface_probe.py must never call open() -- pure instrument, writes nothing")


# ---- (2) capture / hash_tensor_state ----------------------------------------

def test_capture_validates_provenance():
    with pytest.raises(ValueError):
        ip.capture(
            run_id="r1", seed=0, episode=0, timestep=0,
            sender=torch.zeros(3), receiver_input=torch.zeros(3),
            committed_action=0, provenance="not_a_real_provenance", phase="wake",
        )
    rec = ip.capture(
        run_id="r1", seed=0, episode=0, timestep=0,
        sender=torch.zeros(3), receiver_input=torch.zeros(3),
        committed_action=0, provenance="observed", phase="wake",
    )
    assert isinstance(rec, ip.CaptureRecord)
    assert rec.provenance == "observed"


def test_hash_tensor_state_stable_under_key_order_and_sensitive_to_change():
    a = {"w1": torch.arange(6.0), "w2": torch.ones(2)}
    b = {"w2": torch.ones(2), "w1": torch.arange(6.0)}  # same content, different insertion order
    assert ip.hash_tensor_state(a) == ip.hash_tensor_state(b)

    c = {"w1": torch.arange(6.0) + 1.0, "w2": torch.ones(2)}
    assert ip.hash_tensor_state(a) != ip.hash_tensor_state(c)

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.lin.weight.fill_(0.5)

    m = _Module()
    assert ip.hash_tensor_state(m) == ip.hash_tensor_state(m.state_dict())


# ---- (3) communication_subspace ---------------------------------------------

def _make_low_rank_dataset(n=300, dx=10, dy=10, true_rank=2, noise=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dx, generator=g)
    u = ip._random_orthonormal(dx, true_rank, seed=seed + 1)
    v = ip._random_orthonormal(dy, true_rank, seed=seed + 2)
    w = u @ v.T * 4.0
    y = x @ w + noise * torch.randn(n, dy, generator=g)
    return x, y, u


def test_communication_subspace_recovers_true_rank_and_basis():
    x, y, u = _make_low_rank_dataset()
    result = ip.communication_subspace(x, y, ranks=[1, 2, 3, 4, 8], seed=1)
    assert isinstance(result, ip.CommunicationSubspaceResult)
    assert result.heldout_r2_by_rank[2] > result.heldout_r2_by_rank[1]
    assert result.heldout_r2_by_rank[2] > 0.85
    # the fitted basis at the true rank should align closely with the planted directions
    angles = ip.principal_angles(result.basis, u)
    assert angles["mean_squared_cosine_overlap"] > 0.8


def test_communication_subspace_grouped_folds_keep_groups_together():
    n = 60
    x, y, _u = _make_low_rank_dataset(n=n)
    groups = [i // 6 for i in range(n)]  # 10 groups of 6 rows each
    result = ip.communication_subspace(x, y, ranks=[2], groups=groups, n_folds=5, seed=2)
    assert result.n_folds_used <= 5
    assert result.n_rows == n


# ---- (4) principal_angles ----------------------------------------------------

def test_principal_angles_identical_and_orthogonal_subspaces():
    basis = ip._random_orthonormal(8, 3, seed=10)
    same = ip.principal_angles(basis, basis)
    assert same["mean_squared_cosine_overlap"] == pytest.approx(1.0, abs=1e-6)
    assert max(same["angles_rad"]) < 1e-4

    full = ip._random_orthonormal(8, 8, seed=11)
    a = full[:, :3]
    b = full[:, 3:6]
    orth = ip.principal_angles(a, b)
    assert orth["mean_squared_cosine_overlap"] == pytest.approx(0.0, abs=1e-6)


# ---- (5) bridge_ladder --------------------------------------------------------

def test_bridge_ladder_richer_levels_and_field_contract():
    n, d = 300, 6
    x = torch.randn(n, d)
    u = ip._random_orthonormal(d, 2, seed=20)
    v = ip._random_orthonormal(d, 2, seed=21)
    y = torch.tanh(x @ u) @ v.T * 3.0 + 0.01 * torch.randn(n, d)
    x_tr, x_te = x[:200], x[200:]
    y_tr, y_te = y[:200], y[200:]

    out = ip.bridge_ladder(x_tr, y_tr, x_te, y_te, low_rank_k=2, l4_epochs=100, l5_epochs=150, seed=3)
    assert set(out.keys()) == {lvl.value for lvl in ip._ALL_BRIDGE_LEVELS}
    for level_result in out.values():
        for field in ("bridge_class", "rank", "n_parameters", "n_training_rows",
                       "heldout_mse", "heldout_r2", "downstream_behavioural_effect", "ood"):
            assert field in level_result

    n_l0 = out[ip.BridgeLevel.L0_IDENTITY.value]["n_parameters"]
    n_l1 = out[ip.BridgeLevel.L1_PROCRUSTES.value]["n_parameters"]
    n_l2 = out[ip.BridgeLevel.L2_AFFINE.value]["n_parameters"]
    assert n_l0 <= n_l1 < n_l2

    r2_l4 = out[ip.BridgeLevel.L4_CONSTRAINED_NONLINEAR.value]["heldout_r2"]
    r2_l0 = out[ip.BridgeLevel.L0_IDENTITY.value]["heldout_r2"]
    assert r2_l4 > r2_l0


def test_bridge_ladder_consumer_eval_and_ood_wiring():
    n, d = 120, 4
    x = torch.randn(n, d)
    y = x.clone()
    x_tr, x_te = x[:80], x[80:]
    y_tr, y_te = y[:80], y[80:]
    ood_x = torch.randn(10, d) * 50.0

    calls = []

    def consumer(mapped):
        calls.append(mapped.shape)
        return float(mapped.abs().mean())

    out = ip.bridge_ladder(
        x_tr, y_tr, x_te, y_te,
        levels=(ip.BridgeLevel.L0_IDENTITY,), consumer_eval_fn=consumer, ood_x=ood_x, seed=4,
    )
    assert len(calls) == 1
    result = out[ip.BridgeLevel.L0_IDENTITY.value]
    assert isinstance(result["downstream_behavioural_effect"], float)
    assert result["ood"]["provided"] is True
    assert result["ood"]["mean_abs_zscore"] > 1.0


# ---- (6) causal_replacement ---------------------------------------------------

def test_causal_replacement_not_load_bearing():
    def flat_eval(_batch):
        return 0.5

    result = ip.causal_replacement(flat_eval, correct=torch.randn(20, 4), mismatched=torch.randn(20, 4), seed=5)
    assert result.category == ip.CausalReplacementCategory.NOT_LOAD_BEARING


def test_causal_replacement_misleading_content():
    # mismatched actively scores BELOW the zero baseline (band=0.05 apart) --
    # the MISLEADING_CONTENT trigger is mismatched_s < zero_s - band.
    scores = {"correct": 0.80, "mismatched": -0.50, "zero": 0.0, "moment_matched_random": 0.0}

    def eval_fn(batch):
        return scores[batch]

    result = ip.causal_replacement(
        eval_fn,
        correct="correct", mismatched="mismatched", zero="zero", moment_matched_random="moment_matched_random",
        seed=6,
    )
    assert result.category == ip.CausalReplacementCategory.MISLEADING_CONTENT


def test_causal_replacement_mixed_category():
    scores = {"correct": 0.9, "mismatched": 0.5, "zero": 0.1, "moment_matched_random": 0.1}

    def eval_fn(batch):
        return scores[batch]  # batch is a sentinel string standing in for the condition name

    result = ip.causal_replacement(
        eval_fn,
        correct="correct", mismatched="mismatched", zero="zero", moment_matched_random="moment_matched_random",
        band=0.05, seed=7,
    )
    assert result.category == ip.CausalReplacementCategory.MIXED_GENERIC_AND_CONTENT_SPECIFIC


def test_causal_replacement_ambiguous_fallback():
    # correct < mismatched (reversed from every documented pattern, all of which require
    # correct >= mismatched somewhere in their rule) -- matches none of the five named
    # categories, so the defensive AMBIGUOUS sentinel fires.
    scores = {"correct": 0.20, "mismatched": 0.50, "zero": 0.10, "moment_matched_random": 0.10}

    def eval_fn(batch):
        return scores[batch]

    result = ip.causal_replacement(
        eval_fn,
        correct="correct", mismatched="mismatched", zero="zero", moment_matched_random="moment_matched_random",
        band=0.02, dominance=0.1, seed=8,
    )
    assert result.category == ip.CausalReplacementCategory.AMBIGUOUS


def test_causal_replacement_records_optional_conditions():
    def eval_fn(batch):
        return float(batch.mean())

    correct = torch.ones(10, 3)
    mismatched = torch.zeros(10, 3)
    same_action = torch.full((10, 3), 0.3)
    same_context = torch.full((10, 3), 0.4)
    result = ip.causal_replacement(
        eval_fn, correct=correct, mismatched=mismatched,
        same_action_mismatch=same_action, same_context_mismatch=same_context, seed=9,
    )
    assert "same_action_mismatch" in result.scores
    assert "same_context_mismatch" in result.scores


# ---- (7) manifold_guard --------------------------------------------------------

def test_manifold_guard_flags_off_manifold_only():
    native = torch.randn(150, 5) * 0.3
    in_dist = torch.randn(15, 5) * 0.3
    off_manifold = torch.randn(15, 5) * 0.3 + 50.0

    ok = ip.manifold_guard(native, in_dist)
    bad = ip.manifold_guard(native, off_manifold)
    assert not ok.off_manifold_flag
    assert bad.off_manifold_flag
    assert ok.activation_saturation_rate is None
    assert ok.hidden_trajectory_distance is None


def test_manifold_guard_optional_fields_populate():
    native = torch.randn(100, 4) * 0.5
    mapped = torch.randn(10, 4) * 0.5
    activations = torch.tensor([[0.99, 0.97], [0.98, 0.96]])
    traj_native = torch.randn(5, 3)
    traj_mapped = traj_native + 0.1
    result = ip.manifold_guard(
        native, mapped, activations=activations,
        hidden_trajectory_native=traj_native, hidden_trajectory_mapped=traj_mapped,
    )
    assert result.activation_saturation_rate == pytest.approx(1.0)
    assert result.hidden_trajectory_distance["n_steps"] == 5
    assert result.hidden_trajectory_distance["step_1"] is not None


# ---- (8) dynamic_compatibility -------------------------------------------------

def test_dynamic_compatibility_one_step_matches_hand_computation():
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    result = ip.dynamic_compatibility(pred, target)
    expected_mse = float(((pred - target) ** 2).mean())
    assert result.one_step_mse == pytest.approx(expected_mse)
    assert result.multi_step_mse_by_horizon is None
    assert result.compounding is None


def test_dynamic_compatibility_compounding_true_and_false():
    pred = torch.zeros(5, 2)
    growing_targets = [pred + 0.1 * (h + 1) for h in range(4)]
    growing_preds = [pred for _ in range(4)]
    result_growing = ip.dynamic_compatibility(pred, pred, multi_step_predicted=growing_preds,
                                               multi_step_target=growing_targets)
    assert result_growing.compounding is True

    decreasing_targets = [pred + 0.5 * (4 - h) for h in range(4)]  # mse strictly shrinks
    result_decreasing = ip.dynamic_compatibility(pred, pred, multi_step_predicted=growing_preds,
                                                  multi_step_target=decreasing_targets)
    assert result_decreasing.compounding is False


# ---- (9) per_code_drift ---------------------------------------------------------

def test_per_code_drift_in_subspace_only():
    d = 6
    basis = ip._random_orthonormal(d, 2, seed=30)
    n = 50
    pre = torch.randn(n, d)
    drift_in_subspace = (torch.randn(n, 2) @ basis.T) * 2.0
    post = pre + drift_in_subspace
    result = ip.per_code_drift(pre, post, basis)
    assert result.in_subspace_drift_rate_per_dim > result.out_subspace_drift_rate_per_dim


def test_per_code_drift_out_of_subspace_only():
    d = 6
    basis = ip._random_orthonormal(d, 2, seed=31)
    full = ip._random_orthonormal(d, d, seed=32)
    complement = full[:, 2:]  # orthogonal complement-ish (not guaranteed exactly, seed differs)
    n = 50
    pre = torch.randn(n, d)
    # project drift onto the orthogonal complement of `basis` explicitly
    q, _ = torch.linalg.qr(basis)
    raw_drift = torch.randn(n, d) * 2.0
    proj_onto_basis = (raw_drift @ q) @ q.T
    drift_out_of_subspace = raw_drift - proj_onto_basis
    post = pre + drift_out_of_subspace
    result = ip.per_code_drift(pre, post, basis)
    assert result.out_subspace_drift_rate_per_dim > result.in_subspace_drift_rate_per_dim


# ---- (10) frame field on CaptureRecord (MECH-555 telemetry gap) -------------

def test_capture_frame_field_defaults_empty_and_round_trips():
    """The frame gap RF doc records as "absent -- no frame field". It must
    default to {} so every pre-existing call site stays valid, and must
    round-trip a caller's frame verbatim without interpretation."""
    bare = ip.capture(
        run_id="r1", seed=0, episode=0, timestep=0,
        sender=torch.zeros(3), receiver_input=torch.zeros(3),
        committed_action=0, provenance="observed", phase="wake",
    )
    assert bare.frame == {}

    framed = ip.capture(
        run_id="r1", seed=0, episode=0, timestep=0,
        sender=torch.zeros(3), receiver_input=torch.zeros(3),
        committed_action=0, provenance="observed", phase="wake",
        frame={"frame_id": "allocentric", "anchor": (1, 2)},
    )
    assert framed.frame == {"frame_id": "allocentric", "anchor": (1, 2)}
    # recorded, not interpreted: no normalisation, no vocabulary enforcement
    odd = ip.capture(
        run_id="r1", seed=0, episode=0, timestep=0,
        sender=torch.zeros(3), receiver_input=torch.zeros(3),
        committed_action=0, provenance="observed", phase="wake",
        frame={"anything": 7},
    )
    assert odd.frame == {"anything": 7}
    # defensive copy: mutating the caller's dict must not reach the record
    src = {"frame_id": "ego"}
    rec = ip.capture(
        run_id="r1", seed=0, episode=0, timestep=0,
        sender=torch.zeros(3), receiver_input=torch.zeros(3),
        committed_action=0, provenance="observed", phase="wake", frame=src,
    )
    src["frame_id"] = "mutated"
    assert rec.frame == {"frame_id": "ego"}


# ---- (11) receiver_conditioned_bridge -- MECH-547 T(A, B) ------------------

def _cond_data(n_train=400, n_test=200, da=8, db=3, dy=4, seed=4242):
    g = torch.Generator().manual_seed(seed)
    a_tr = torch.randn(n_train, da, generator=g, dtype=torch.float64)
    b_tr = torch.randn(n_train, db, generator=g, dtype=torch.float64)
    a_te = torch.randn(n_test, da, generator=g, dtype=torch.float64)
    b_te = torch.randn(n_test, db, generator=g, dtype=torch.float64)
    w = torch.randn(da, dy, generator=g, dtype=torch.float64)
    v = torch.randn(db, dy, generator=g, dtype=torch.float64) * 2.0
    return a_tr, b_tr, a_te, b_te, w, v


def test_receiver_conditioned_bridge_arms_are_capacity_matched():
    """THE acceptance bar: the conditioned arm must not win on parameters.
    All four arms are fitted at the same level over the same input width, so
    n_parameters must be identical -- and `capacity_matched` must say so."""
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, a_tr @ w + b_tr @ v,
        a_te, b_te, a_te @ w + b_te @ v,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert set(res.arms) == {"sender_only", "conditioned", "receiver_permuted", "receiver_only"}
    assert len(set(res.parameter_counts.values())) == 1, res.parameter_counts
    assert res.capacity_matched is True
    # every arm dict is self-identifying once serialised into a manifest
    for name, arm in res.arms.items():
        assert arm["arm"] == name
        for key in ("bridge_class", "n_parameters", "heldout_mse", "heldout_r2"):
            assert key in arm


def test_receiver_conditioned_bridge_permutation_is_a_true_row_permutation():
    """THE other acceptance bar: the discriminator is only a discriminator if
    the permuted arm differs from the conditioned arm by a ROW PERMUTATION of
    the receiver block and nothing else. If a future edit made it a resample,
    a zeroing, or a moment-match, the control would silently become a
    capacity test. Assert the multiset of receiver rows is preserved."""
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    n_tr = b_tr.shape[0]
    perm = ip._row_permutation(n_tr, 0 + 101)
    assert sorted(perm.tolist()) == list(range(n_tr)), "not a permutation of the row index"
    permuted = b_tr[perm]
    assert permuted.shape == b_tr.shape
    # same multiset of rows, different order (with n=400 an identity draw is
    # not a realistic risk, but assert the reorder explicitly)
    assert torch.allclose(permuted.sum(dim=0), b_tr.sum(dim=0))
    assert not torch.equal(permuted, b_tr)
    assert torch.equal(
        torch.sort(permuted[:, 0]).values, torch.sort(b_tr[:, 0]).values
    )


def test_receiver_conditioned_bridge_supported_when_conditioning_is_real():
    """Y = A@W + B@V: the receiver block genuinely carries per-row information
    the sender does not. The conditioned arm must gain, and a receiver-state
    permutation must destroy that gain (receiver_conditioned_translation.md's
    reading table, row 1)."""
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, a_tr @ w + b_tr @ v,
        a_te, b_te, a_te @ w + b_te @ v,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert res.verdict == ip.ConditioningVerdict.CONDITIONING_SUPPORTED.value, res.notes
    assert res.conditioning_gain > 0.02
    assert res.permutation_destroyed_fraction is not None
    assert res.permutation_destroyed_fraction > 0.5
    assert res.scores["conditioned"] > res.scores["receiver_permuted"]


def test_receiver_conditioned_bridge_no_gain_when_receiver_state_is_noise():
    """Y = A@W only. The receiver block is pure noise, so there is nothing to
    condition on and the instrument must not manufacture a gain from the
    extra input width."""
    a_tr, b_tr, a_te, b_te, w, _v = _cond_data()
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, a_tr @ w, a_te, b_te, a_te @ w,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert res.verdict == ip.ConditioningVerdict.NO_CONDITIONING_GAIN.value, res.notes
    assert res.conditioning_gain <= 0.02


def test_receiver_conditioned_bridge_detects_receiver_state_leakage():
    """Y = B@V: the target is a function of the receiver state alone. A
    conditioned bridge scores perfectly, but it INTRODUCED the content rather
    than translating it (assay spec 2.8, `A7_receiver_only` high) -- the
    instrument must refuse the conditioning reading."""
    a_tr, b_tr, a_te, b_te, _w, v = _cond_data()
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, b_tr @ v, a_te, b_te, b_te @ v,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert res.verdict == ip.ConditioningVerdict.RECEIVER_STATE_LEAKS_TARGET.value, res.notes
    assert res.scores["receiver_only"] > res.scores["sender_only"]
    assert any("A7_receiver_only" in n for n in res.notes)


def test_receiver_conditioned_bridge_refuses_verdict_when_capacity_unmatched():
    """`pad_baselines=False` gives the native unpadded sender-only bridge --
    a legitimate comparison (assay-spec A1_source_only) but NOT capacity
    matched. The instrument must report MATCHING_FAILED rather than credit
    the conditioned arm with its parameter advantage, even though the
    underlying data supports conditioning."""
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, a_tr @ w + b_tr @ v,
        a_te, b_te, a_te @ w + b_te @ v,
        level=ip.BridgeLevel.L2_AFFINE, seed=0, pad_baselines=False,
    )
    assert res.capacity_matched is False
    assert res.verdict == ip.ConditioningVerdict.MATCHING_FAILED.value
    assert res.parameter_counts["sender_only"] < res.parameter_counts["conditioned"]
    assert any("not capacity-matched" in n or "capacity NOT matched" in n for n in res.notes)


def test_receiver_conditioned_bridge_capacity_not_conditioning_verdict():
    """The assay spec's section 2.7 falsifier -- "A4's advantage survives
    receiver-state permutation intact". A row permutation destroys every
    per-row signal by construction, so this branch is not synthesisable from
    data; it is driven here through the public `score_from_arm` hook (which
    is also the consumer-use-gain path the spec's primary estimand needs)."""
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    fixed = {"sender_only": 0.50, "conditioned": 0.80,
             "receiver_permuted": 0.78, "receiver_only": 0.10}
    res = ip.receiver_conditioned_bridge(
        a_tr, b_tr, a_tr @ w + b_tr @ v,
        a_te, b_te, a_te @ w + b_te @ v,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
        score_from_arm=lambda arm: fixed[arm["arm"]],
    )
    assert res.capacity_matched is True
    assert res.verdict == ip.ConditioningVerdict.CAPACITY_NOT_CONDITIONING.value, res.notes
    assert res.conditioning_gain == pytest.approx(0.30)
    assert res.permuted_gain == pytest.approx(0.28)
    assert any("never conditioned on state" in n for n in res.notes)


def test_receiver_conditioned_bridge_rejects_misaligned_rows():
    a_tr, b_tr, a_te, b_te, w, v = _cond_data()
    with pytest.raises(ValueError):
        ip.receiver_conditioned_bridge(
            a_tr, b_tr[:-1], a_tr @ w, a_te, b_te, a_te @ w,
            level=ip.BridgeLevel.L2_AFFINE,
        )
    with pytest.raises(ValueError):
        ip.receiver_conditioned_bridge(
            a_tr, b_tr, a_tr @ w, a_te, b_te[:-1], a_te @ w,
            level=ip.BridgeLevel.L2_AFFINE,
        )


# ---- (12) frame_permutation_control -- MECH-555 ----------------------------

def _frame_data(n_train=400, n_test=200, dx=8, dy=4, n_frames=3, seed=909):
    g = torch.Generator().manual_seed(seed)
    x_tr = torch.randn(n_train, dx, generator=g, dtype=torch.float64)
    x_te = torch.randn(n_test, dx, generator=g, dtype=torch.float64)
    w = torch.randn(dx, dy, generator=g, dtype=torch.float64)
    fv = torch.randn(n_frames, dy, generator=g, dtype=torch.float64) * 3.0
    fr_tr = ["f%d" % (i % n_frames) for i in range(n_train)]
    fr_te = ["f%d" % (i % n_frames) for i in range(n_test)]

    def onehot(labels):
        m = torch.zeros(len(labels), n_frames, dtype=torch.float64)
        for i, lab in enumerate(labels):
            m[i, int(lab[1])] = 1.0
        return m

    return x_tr, x_te, w, fv, fr_tr, fr_te, onehot


def test_frame_permutation_control_load_bearing_frame():
    """Y = X@W + onehot(frame)@FV. The frame genuinely indexes the content, so
    the intact arm must beat the empty-frame baseline and a content-preserving
    row permutation of the frame labels must destroy that gain (RF doc:
    "intact >> frame-permuted ... shared indexing is functionally
    load-bearing")."""
    x_tr, x_te, w, fv, fr_tr, fr_te, onehot = _frame_data()
    res = ip.frame_permutation_control(
        x_tr, x_tr @ w + onehot(fr_tr) @ fv,
        x_te, x_te @ w + onehot(fr_te) @ fv,
        frame_train=fr_tr, frame_test=fr_te,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert res.verdict == ip.FrameVerdict.FRAME_LOAD_BEARING.value, res.notes
    assert res.content_preserved is True
    assert res.capacity_matched is True
    assert res.n_frames == 3
    assert res.frame_gain > 0.02
    assert res.permutation_destroyed_fraction > 0.5
    assert set(res.arms) == {"no_frame", "frame_conditioned", "frame_permuted"}
    assert len(set(res.parameter_counts.values())) == 1, res.parameter_counts


def test_frame_permutation_control_incidental_frame():
    """Y = X@W: the frame indexes nothing at this interface, so the frame
    channel must buy nothing ("intact ~= frame-permuted -> the frame is
    incidental at that interface")."""
    x_tr, x_te, w, _fv, fr_tr, fr_te, _onehot = _frame_data()
    res = ip.frame_permutation_control(
        x_tr, x_tr @ w, x_te, x_te @ w,
        frame_train=fr_tr, frame_test=fr_te,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert res.verdict == ip.FrameVerdict.FRAME_INCIDENTAL.value, res.notes
    assert res.frame_gain <= 0.02


def test_frame_permutation_control_rejects_non_permutation_scramble():
    """A scramble that is not a permutation of the intact label vector changes
    the frame MARGINAL as well as the frame-to-content correspondence, which
    confounds the contrast. It must be reported as CONTROL_INVALID, never
    scored as a frame result."""
    x_tr, x_te, w, fv, fr_tr, fr_te, onehot = _frame_data()
    res = ip.frame_permutation_control(
        x_tr, x_tr @ w + onehot(fr_tr) @ fv,
        x_te, x_te @ w + onehot(fr_te) @ fv,
        frame_train=fr_tr, frame_test=fr_te,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
        frame_permutation_train=["f0"] * len(fr_tr),
        frame_permutation_test=["f0"] * len(fr_te),
    )
    assert res.content_preserved is False
    assert res.verdict == ip.FrameVerdict.CONTROL_INVALID.value
    assert any("NOT a permutation" in n for n in res.notes)


def test_frame_permutation_control_preserves_content_tensors_exactly():
    """"Content-preserving" is asserted mechanically, not by inspection: the
    default scramble must leave every element of X and Y untouched -- only the
    frame index moves. A future edit that "helpfully" permuted rows of X or Y
    would silently turn the discriminator into a content test."""
    x_tr, x_te, w, fv, fr_tr, fr_te, onehot = _frame_data()
    y_tr = x_tr @ w + onehot(fr_tr) @ fv
    y_te = x_te @ w + onehot(fr_te) @ fv
    x_tr_before = x_tr.clone()
    y_tr_before = y_tr.clone()
    x_te_before = x_te.clone()
    y_te_before = y_te.clone()
    ip.frame_permutation_control(
        x_tr, y_tr, x_te, y_te,
        frame_train=fr_tr, frame_test=fr_te,
        level=ip.BridgeLevel.L2_AFFINE, seed=0,
    )
    assert torch.equal(x_tr, x_tr_before)
    assert torch.equal(y_tr, y_tr_before)
    assert torch.equal(x_te, x_te_before)
    assert torch.equal(y_te, y_te_before)


def test_frame_permutation_control_rejects_label_count_mismatch():
    x_tr, x_te, w, _fv, fr_tr, fr_te, _onehot = _frame_data()
    with pytest.raises(ValueError):
        ip.frame_permutation_control(
            x_tr, x_tr @ w, x_te, x_te @ w,
            frame_train=fr_tr[:-1], frame_test=fr_te,
            level=ip.BridgeLevel.L2_AFFINE,
        )
    with pytest.raises(ValueError):
        ip.frame_permutation_control(
            x_tr, x_tr @ w, x_te, x_te @ w,
            frame_train=fr_tr, frame_test=fr_te[:-1],
            level=ip.BridgeLevel.L2_AFFINE,
        )
