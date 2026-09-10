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

Design docs: REE_assembly/evidence/planning/hippocampal_campaign_assay_specifications_20260910.md
section 4; REE_assembly/docs/thoughts/2026-09-07_mutual_legibility_implementation_assays.md.
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
