"""Read-only communication-subspace / bridge-ladder / causal-replacement instrument.

Prerequisite P0 for the hippocampal campaign's assays A and B -- see

    REE_assembly/evidence/planning/hippocampal_campaign_assay_specifications_20260910.md
    section 4

and its parent design document

    REE_assembly/docs/thoughts/2026-09-07_mutual_legibility_implementation_assays.md

whose section 0.2 confirmed that `ree-v3` contained NO communication-subspace
estimator, bridge ladder, principal-angle metric, causal-replacement harness,
receiver-manifold guard, or dynamic-compatibility scorer anywhere -- so both
assays were instrument-blocked, not substrate-blocked, and this module is the
missing instrument.

Scope, per section 4.4 of the spec above (do not widen this file past it):

  - This module changes NO `ree_core` file and adds NO module to the agent.
    It operates only over already-collected tensors, frozen checkpoints, and
    recorded trajectories that a CALLER supplies.
  - Every public function here is PURE over its inputs, holds no agent
    reference, and writes nothing (no manifest, no queue entry, no EXQ id --
    this module never registers or runs an experiment on its own).
  - No `LatentBridge` (or any other permanent bridge module) is added to
    `ree_core`. A successful bridge fitted by this module is a DIAGNOSTIC
    finding about the two frozen endpoints it was fitted between, never
    architecture.
  - No global CKA/RSA alignment objective; no optimisation of a communication-
    subspace rank without a competence constraint; L5 ("high-capacity upper
    bound") success is reported as an information-in-principle statement and
    must never be read as evidence that a source is adequate as an interface,
    nor substituted for a causal content test.
  - No task-specific semantics are baked into this generic layer (mutual-
    legibility doc section 15: "Do not bake task-specific semantics such as
    `resource_direction` into the generic analysis layer"). Every function
    below takes plain tensors; a caller wanting a decision-relevant subspace,
    an oracle label, or a consumer network supplies it itself.

Deliberate reimplementation, not reuse, for two small helpers
---------------------------------------------------------------
The spec's reuse table (section 2.10/3.10) names `_random_orthonormal` and
`_inv_sqrt_psd` from `v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis.py`
as existing symbols. That file is a full EXPERIMENT SCRIPT (it imports
`x734`/`x1002`, builds environments, and is meant to be run, not imported as a
library), so importing it here would make this pure instrument module
transitively depend on, and pay the import cost of, an experiment driver --
exactly what "holds no agent reference" and "experiment-layer only" argue
against for a module every future assay script will import. `_random_orthonormal`
and `_inv_sqrt_psd` are each a handful of lines with no external state; they
are reimplemented locally below (`_random_orthonormal`, `_inv_sqrt_psd`),
identically in substance to the 1008 originals. `reset_all_rng` from
`experiments/_lib/arm_fingerprint.py` IS a `_lib` module already meant to be
imported broadly (it is "pure and side-effect-free" by its own docstring), so
the self-test below imports it directly rather than reimplementing it.

The seven-rung reporting contract (INV-105) and everything about how these
functions compose into assays A and B lives in the spec document above, not
here -- this module supplies the primitives; the assay scripts (a later
`/queue-experiment` build, not this one) supply the wiring, the oracle labels,
the consumer network, and the manifest.

Conditioned rungs (added 2026-09-16)
------------------------------------
The original P0 ladder fits X -> Y UNCONDITIONALLY, which left two of the
assay spec's own arms without an instrument: `A4_receiver_state_cond`
(MECH-547's `T(A, B)`) and `A3_frame_cond` (MECH-555's reference-frame
mediation). Section 5b below supplies both, plus the `frame` field on
`CaptureRecord` that the frame arm needs in telemetry. Both follow the same
discipline and it is the load-bearing part: a conditioned arm is compared only
against a MATCHED-CAPACITY baseline, and the claim is carried not by the gain
but by a PERMUTATION control that destroys it. A gain a permutation cannot
touch is capacity, not conditioning -- and both functions report that outcome
as a first-class verdict rather than as an error.
"""

from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Data capture (spec section 4.1 `capture(...)`, mutual-legibility doc
#    section 2 "Data capture contract")
# ---------------------------------------------------------------------------

_VALID_PROVENANCE = ("observed", "replayed", "simulated")


@dataclass(frozen=True)
class CaptureRecord:
    """One aligned sender/receiver observation, per the data-capture contract.

    The CRITICAL RULE from the mutual-legibility doc section 2 is that
    `receiver_input` must be the signal the LIVE consumer actually reads at
    its point of consumption -- not a convenient neighbouring tensor. This
    dataclass does not and cannot enforce that; it is the caller's obligation
    at the call site that builds each `capture(...)` record.

    `frame` records the REFERENCE FRAME this row's content is indexed against
    (MECH-555; `REE_assembly/docs/architecture/interface_reference_frames_and_temporal_gates.md`).
    It is free-form and defaults to `{}` so every pre-existing call site stays
    valid. The conventional key is `frame_id` (a hashable label, the thing
    `frame_permutation_control` below permutes); anchors, origins or any other
    frame parameters may be carried alongside it. It is recorded, never
    interpreted, here: this module bakes in no task-specific frame semantics
    (see the scope note in the module docstring), so the assay supplies both
    the labels and their meaning.
    """

    run_id: str
    seed: int
    episode: int
    timestep: int
    env_id: Any
    layout_id: Any
    sender: torch.Tensor
    receiver_input: torch.Tensor
    receiver_pre_activation: Optional[torch.Tensor]
    receiver_post_activation: Optional[torch.Tensor]
    candidate_id: Any
    task_targets: Dict[str, Any]
    committed_action: Any
    behaviour_vars: Dict[str, Any]
    provenance: str
    phase: str
    sender_weight_hash: Optional[str]
    receiver_weight_hash: Optional[str]
    active_gates: Dict[str, float]
    frame: Dict[str, Any] = field(default_factory=dict)


def hash_tensor_state(state: Union[Dict[str, torch.Tensor], "nn.Module"]) -> str:
    """Stable sha256 over a state_dict's tensor bytes, sorted by key.

    Accepts either a raw `state_dict()`-shaped mapping or an `nn.Module`
    directly (its `.state_dict()` is used). Order-independent of dict
    insertion order, so two structurally-identical checkpoints hash
    identically regardless of construction order. Used to snapshot-and-verify
    frozen endpoint weights per the spec's section 1.2 "Frozen endpoints"
    contract: hash before the first evaluation, re-verify at every boundary,
    abort the cell on a mismatch.
    """
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        h.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def capture(
    *,
    run_id: str,
    seed: int,
    episode: int,
    timestep: int,
    sender: torch.Tensor,
    receiver_input: torch.Tensor,
    committed_action: Any,
    provenance: str,
    phase: str,
    env_id: Any = None,
    layout_id: Any = None,
    receiver_pre_activation: Optional[torch.Tensor] = None,
    receiver_post_activation: Optional[torch.Tensor] = None,
    candidate_id: Any = None,
    task_targets: Optional[Dict[str, Any]] = None,
    behaviour_vars: Optional[Dict[str, Any]] = None,
    sender_weight_hash: Optional[str] = None,
    receiver_weight_hash: Optional[str] = None,
    active_gates: Optional[Dict[str, float]] = None,
    frame: Optional[Dict[str, Any]] = None,
) -> CaptureRecord:
    """Construct one `CaptureRecord`. Pure: validates and packages, writes nothing.

    `provenance` must be one of "observed" / "replayed" / "simulated" (the
    MECH-094 hypothesis-tag taxonomy this codebase uses everywhere else --
    reused here rather than inventing a fourth spelling of the same concept).
    """
    if provenance not in _VALID_PROVENANCE:
        raise ValueError(
            "provenance must be one of %r, got %r" % (_VALID_PROVENANCE, provenance)
        )
    return CaptureRecord(
        run_id=str(run_id),
        seed=int(seed),
        episode=int(episode),
        timestep=int(timestep),
        env_id=env_id,
        layout_id=layout_id,
        sender=sender,
        receiver_input=receiver_input,
        receiver_pre_activation=receiver_pre_activation,
        receiver_post_activation=receiver_post_activation,
        candidate_id=candidate_id,
        task_targets=dict(task_targets or {}),
        committed_action=committed_action,
        behaviour_vars=dict(behaviour_vars or {}),
        provenance=provenance,
        phase=str(phase),
        sender_weight_hash=sender_weight_hash,
        receiver_weight_hash=receiver_weight_hash,
        active_gates=dict(active_gates or {}),
        frame=dict(frame or {}),
    )


# ---------------------------------------------------------------------------
# 2. Small private math helpers (deliberately reimplemented, not imported --
#    see the module docstring "Deliberate reimplementation" section)
# ---------------------------------------------------------------------------


def _random_orthonormal(in_dim: int, k: int, seed: int) -> torch.Tensor:
    """A [in_dim, k] matrix with orthonormal columns, deterministic in `seed`.

    Same construction as v3_exq_1008's `_random_orthonormal`: QR of a
    Gaussian random matrix, using a dedicated `torch.Generator` so this draw
    never perturbs the caller's own global RNG stream.
    """
    g = torch.Generator().manual_seed(int(seed))
    q, _r = torch.linalg.qr(torch.randn(int(in_dim), int(k), generator=g))
    return q.contiguous()


def _inv_sqrt_psd(mat: torch.Tensor, ridge: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Inverse matrix square root of a PSD matrix, ridge-regularised.

    Same construction as v3_exq_1008's `_inv_sqrt_psd`: eigendecompose,
    clamp eigenvalues at 0 (PSD by construction, but floating-point can push
    a near-zero eigenvalue slightly negative), add the ridge, invert the
    square root. Returns the transform `T` such that `T @ T @ mat ~= I`
    (`T` is symmetric, `T = mat^(-1/2)`), plus a small diagnostics dict.
    """
    lam, v = torch.linalg.eigh(mat)
    lam = lam.clamp(min=0.0)
    t = v @ torch.diag(1.0 / torch.sqrt(lam + ridge)) @ v.T
    cond_scatter = float((lam.max() + ridge) / (lam.min() + ridge)) if lam.numel() else 1.0
    return t, {
        "eig_min": float(lam.min()) if lam.numel() else None,
        "eig_max": float(lam.max()) if lam.numel() else None,
        "ridge": float(ridge),
        "condition_number_scatter": cond_scatter,
        "condition_number": float(np.sqrt(cond_scatter)),
    }


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination, computed over ALL elements of `y_true`/
    `y_pred` jointly (not averaged per-column then re-averaged), matching the
    single-scalar `_r2`/`_lin_decode_r2` convention already used in
    v3_exq_1008. Returns 0.0 (not NaN) when `y_true` is exactly constant,
    since R^2 is undefined there and 0.0 is the conservative "no better than
    the mean" reading; a caller that needs to distinguish "undefined" from
    "genuinely zero" should check `y_true`'s variance itself.
    """
    yt = y_true.reshape(-1).to(torch.float64)
    yp = y_pred.reshape(-1).to(torch.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(F.mse_loss(y_pred.to(torch.float64), y_true.to(torch.float64)))


def _cosine_sim_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean per-row cosine similarity between two [N, d] tensors."""
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return float((num / den).mean())


def _center(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    return x - mean, mean


def _kfold_indices(n: int, n_folds: int, seed: int,
                    groups: Optional[Sequence[Any]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs for `n_folds` folds over `n` rows.

    When `groups` is supplied (e.g. episode ids), folds are built over the
    DISTINCT group values and every row of a group lands entirely in one
    fold -- a grouped k-fold, matching the mutual-legibility doc section 3
    "held-out environment/layout family, held-out episode" split doctrine
    rather than a naive per-row split (adjacent rows are strongly correlated
    here; a per-row split leaks). Without `groups`, folds are a plain
    per-row split, deterministic in `seed`.
    """
    rng = np.random.default_rng(int(seed))
    n_folds = max(1, min(int(n_folds), n))
    if groups is not None:
        uniq = np.unique(np.asarray(groups))
        rng.shuffle(uniq)
        group_folds = np.array_split(uniq, n_folds)
        groups_arr = np.asarray(groups)
        out = []
        for gf in group_folds:
            if len(gf) == 0:
                continue
            test_mask = np.isin(groups_arr, gf)
            test_idx = np.nonzero(test_mask)[0]
            train_idx = np.nonzero(~test_mask)[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            out.append((train_idx, test_idx))
        if out:
            return out
        # fall through to the per-row split if grouping degenerated (e.g. one group)
    order = rng.permutation(n)
    row_folds = np.array_split(order, n_folds)
    out = []
    for i in range(n_folds):
        test_idx = row_folds[i]
        train_idx = np.concatenate([row_folds[j] for j in range(n_folds) if j != i]) if n_folds > 1 else order
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        out.append((train_idx, test_idx))
    return out


# ---------------------------------------------------------------------------
# 3. communication_subspace -- cross-validated reduced-rank regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommunicationSubspaceResult:
    ranks: List[int]
    heldout_r2_by_rank: Dict[int, float]
    selected_rank: int
    selected_heldout_r2: float
    basis: torch.Tensor  # [d_sender, selected_rank], orthonormal columns, fit on ALL rows
    n_folds_used: int
    n_rows: int


def _rrr_fit_full_data(x: torch.Tensor, y: torch.Tensor, rank: int,
                        ridge: float = 1e-6) -> Dict[str, Any]:
    """One reduced-rank-regression fit at a fixed rank, on already-centered
    `x`/`y`. Standard construction (Izenman 1975 / the RRR form Semedo et al.
    2019 use for "communication subspace" estimation): OLS/ridge for the
    full-rank coefficient, SVD of the resulting prediction, keep the top-`rank`
    right singular vectors of the PREDICTION to get a rank-constrained
    coefficient matrix. `basis` (the sender-side communication directions,
    orthonormalised via QR) is what `principal_angles` compares across seeds
    or checkpoints.
    """
    dx = x.shape[1]
    xtx = x.T @ x
    b_full = torch.linalg.solve(xtx + ridge * torch.eye(dx, dtype=x.dtype), x.T @ y)
    y_hat_full = x @ b_full
    _u, _s, vt = torch.linalg.svd(y_hat_full, full_matrices=False)
    r = max(1, min(int(rank), vt.shape[0]))
    v_r = vt[:r, :].T  # [dy, r]
    coeff_r = b_full @ v_r @ v_r.T  # [dx, dy], rank <= r
    sender_directions = b_full @ v_r  # [dx, r], not orthonormal in general
    if sender_directions.shape[1] > 0:
        basis, _ = torch.linalg.qr(sender_directions)
    else:
        basis = torch.zeros(dx, 0, dtype=x.dtype)
    return {"coeff": coeff_r, "basis": basis, "predict": lambda xn: xn @ coeff_r}


def communication_subspace(
    X: torch.Tensor,
    Y: torch.Tensor,
    ranks: Optional[Sequence[int]] = None,
    *,
    groups: Optional[Sequence[Any]] = None,
    n_folds: int = 5,
    ridge: float = 1e-6,
    seed: int = 0,
) -> CommunicationSubspaceResult:
    """Cross-validated reduced-rank regression from sender `X` [N, dx] to
    receiver `Y` [N, dy], per hippocampal_campaign_assay_specifications
    section 4.1 and mutual-legibility doc section 5.

    Ranks default to `1..min(16, min(dx, dy))` if not supplied, matching the
    spec's "ranks 1..min(16, dim)" contract. Held-out R^2 per rank is
    averaged across `n_folds` folds (grouped by `groups`, e.g. episode id, if
    supplied -- see `_kfold_indices`). `selected_rank` is the rank with the
    highest held-out R^2, smallest rank breaking ties, so a caller does not
    need to separately re-derive "smallest rank at or near the best score."

    The returned `basis` is refit on ALL rows at the selected rank (not one
    fold's basis), giving the caller a single canonical basis to feed into
    `principal_angles` for cross-seed/cross-checkpoint stability comparisons.
    """
    x = X.to(torch.float64)
    y = Y.to(torch.float64)
    n, dx = x.shape
    dy = y.shape[1]
    if ranks is None:
        ranks = list(range(1, max(1, min(16, dx, dy)) + 1))
    ranks = [int(r) for r in ranks]

    folds = _kfold_indices(n, n_folds, seed, groups=groups)
    if not folds:
        folds = [(np.arange(n), np.arange(n))]  # degenerate: evaluate in-sample

    heldout_r2_by_rank: Dict[int, float] = {}
    for r in ranks:
        fold_scores = []
        for train_idx, test_idx in folds:
            x_tr, x_te = x[train_idx], x[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            x_tr_c, x_mean = _center(x_tr)
            y_tr_c, y_mean = _center(y_tr)
            fit = _rrr_fit_full_data(x_tr_c, y_tr_c, r, ridge=ridge)
            y_pred = fit["predict"](x_te - x_mean) + y_mean
            fold_scores.append(_r2(y_te, y_pred))
        heldout_r2_by_rank[r] = float(np.mean(fold_scores)) if fold_scores else float("nan")

    selected_rank = min(
        ranks, key=lambda r: (-heldout_r2_by_rank[r], r)
    )
    x_c, _xm = _center(x)
    y_c, _ym = _center(y)
    full_fit = _rrr_fit_full_data(x_c, y_c, selected_rank, ridge=ridge)
    return CommunicationSubspaceResult(
        ranks=ranks,
        heldout_r2_by_rank=heldout_r2_by_rank,
        selected_rank=selected_rank,
        selected_heldout_r2=heldout_r2_by_rank[selected_rank],
        basis=full_fit["basis"],
        n_folds_used=len(folds),
        n_rows=n,
    )


# ---------------------------------------------------------------------------
# 4. principal_angles
# ---------------------------------------------------------------------------


def principal_angles(U: torch.Tensor, V: torch.Tensor) -> Dict[str, Any]:
    """Principal angles and subspace overlap between the column spaces of
    `U` [d, k1] and `V` [d, k2]. Columns need not be pre-orthonormalised --
    a defensive QR pass is applied first.

    `mean_squared_cosine_overlap` in [0, 1] is the simple scalar summary used
    for pre/post-drift or cross-seed stability comparisons (1.0 = identical
    subspaces, 0.0 = orthogonal). Angles are returned in radians, ascending
    (smallest angle first, i.e. the most-aligned direction pair first).
    """
    u = U.to(torch.float64)
    v = V.to(torch.float64)
    u_o, _ = torch.linalg.qr(u)
    v_o, _ = torch.linalg.qr(v)
    m = u_o.T @ v_o
    svals = torch.linalg.svdvals(m)
    svals = torch.clamp(svals, -1.0, 1.0)
    angles = torch.arccos(svals)
    order = torch.argsort(angles)
    angles = angles[order]
    svals = svals[order]
    overlap = float((svals ** 2).mean()) if svals.numel() else 0.0
    return {
        "angles_rad": [float(a) for a in angles],
        "cosines": [float(c) for c in svals],
        "mean_squared_cosine_overlap": overlap,
        "k1": int(U.shape[1]),
        "k2": int(V.shape[1]),
    }


# ---------------------------------------------------------------------------
# 5. bridge_ladder -- L0 identity through L5 high-capacity upper bound
# ---------------------------------------------------------------------------


class BridgeLevel(str, enum.Enum):
    L0_IDENTITY = "L0_identity_native"
    L1_PROCRUSTES = "L1_orthogonal_procrustes"
    L2_AFFINE = "L2_affine"
    L3_LOW_RANK_AFFINE = "L3_low_rank_affine"
    L4_CONSTRAINED_NONLINEAR = "L4_constrained_nonlinear"
    L5_HIGH_CAPACITY = "L5_high_capacity_upper_bound"


_ALL_BRIDGE_LEVELS: Tuple[BridgeLevel, ...] = (
    BridgeLevel.L0_IDENTITY,
    BridgeLevel.L1_PROCRUSTES,
    BridgeLevel.L2_AFFINE,
    BridgeLevel.L3_LOW_RANK_AFFINE,
    BridgeLevel.L4_CONSTRAINED_NONLINEAR,
    BridgeLevel.L5_HIGH_CAPACITY,
)


def _pad_or_truncate(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """The mutual-legibility doc section 8 L1 note: "If dimensions differ,
    use a predeclared dimensional reduction/augmentation procedure rather
    than silently allowing arbitrary capacity." This is that procedure:
    truncate columns beyond `target_dim`, or zero-pad up to it. Deterministic
    and parameter-free, so it never adds bridge capacity on its own.
    """
    d = x.shape[1]
    if d == target_dim:
        return x
    if d > target_dim:
        return x[:, :target_dim]
    pad = torch.zeros(x.shape[0], target_dim - d, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _fit_l0(x_tr: torch.Tensor, y_tr: torch.Tensor) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    dy = y_tr.shape[1]
    return (lambda x: _pad_or_truncate(x, dy)), 0


def _fit_l1_procrustes(x_tr: torch.Tensor, y_tr: torch.Tensor) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Orthogonal Procrustes: R minimising ||X R - Y||_F, R^T R = I."""
    dy = y_tr.shape[1]
    x_matched = _pad_or_truncate(x_tr, dy)
    m = x_matched.T @ y_tr
    u, _s, vt = torch.linalg.svd(m, full_matrices=False)
    r = u @ vt  # [dy, dy] orthogonal
    return (lambda x: _pad_or_truncate(x, dy) @ r), int(r.numel())


def _fit_l2_affine(x_tr: torch.Tensor, y_tr: torch.Tensor, ridge: float = 1e-3
                    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    dx = x_tr.shape[1]
    dy = y_tr.shape[1]
    ones = torch.ones(x_tr.shape[0], 1, dtype=x_tr.dtype)
    x_aug = torch.cat([x_tr, ones], dim=1)  # [n, dx+1]
    reg = ridge * torch.eye(dx + 1, dtype=x_tr.dtype)
    reg[-1, -1] = 0.0  # never penalise the bias term
    w_aug = torch.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_tr)  # [dx+1, dy]

    def predict(x: torch.Tensor) -> torch.Tensor:
        ones_ = torch.ones(x.shape[0], 1, dtype=x.dtype)
        return torch.cat([x, ones_], dim=1) @ w_aug

    return predict, int(dx * dy + dy)


def _fit_l3_low_rank_affine(x_tr: torch.Tensor, y_tr: torch.Tensor, rank: int,
                             ridge: float = 1e-3) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Affine map with the linear part rank-constrained: `W = U V^T`,
    `rank(W) <= rank`. Fit the unconstrained affine map first, then truncate
    its linear part via SVD (the same "fit full, truncate by SVD" recipe as
    `communication_subspace`'s RRR, applied here to a fitted bridge rather
    than to the raw sender/receiver covariance)."""
    dx = x_tr.shape[1]
    dy = y_tr.shape[1]
    full_predict, _n = _fit_l2_affine(x_tr, y_tr, ridge=ridge)
    x_mean = x_tr.mean(dim=0, keepdim=True)
    y_mean = y_tr.mean(dim=0, keepdim=True)
    # recover the LINEAR part W from the fitted affine map: W = predict(x+e_j) - predict(x)
    # cheaper: refit W directly via centered ridge regression, and bias = y_mean - x_mean @ W
    x_c = x_tr - x_mean
    y_c = y_tr - y_mean
    reg = ridge * torch.eye(dx, dtype=x_tr.dtype)
    w_full = torch.linalg.solve(x_c.T @ x_c + reg, x_c.T @ y_c)  # [dx, dy]
    r = max(1, min(int(rank), min(dx, dy)))
    u, s, vt = torch.linalg.svd(w_full, full_matrices=False)
    w_r = u[:, :r] * s[:r] @ vt[:r, :]

    def predict(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean) @ w_r + y_mean

    n_params = int(r * (dx + dy) + dy)  # U: dx*r, V: dy*r, bias: dy
    return predict, n_params


class _BottleneckMLP(nn.Module):
    """A small generic feed-forward regressor: Linear -> ReLU -> Linear,
    narrow bottleneck, no recurrence, purely a regression head between two
    arbitrary tensor spaces -- NOT the discrete-action decoder classes used
    elsewhere in this codebase (those assume a fixed action-space softmax
    output and a cross-entropy objective; this bridges an arbitrary
    continuous sender representation into an arbitrary continuous receiver
    representation, which is what L4/L5 need)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_nonlinear_bridge(x_tr: torch.Tensor, y_tr: torch.Tensor, *, hidden: int,
                           weight_decay: float, epochs: int, seed: int, lr: float = 1e-2
                           ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    g = torch.Generator().manual_seed(int(seed))
    torch_seed_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))  # nn.Linear init draws from the global RNG; scope it narrowly
    try:
        net = _BottleneckMLP(x_tr.shape[1], y_tr.shape[1], hidden)
    finally:
        torch.random.set_rng_state(torch_seed_state)
    x_mean = x_tr.mean(dim=0, keepdim=True)
    x_std = x_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_c = ((x_tr - x_mean) / x_std).to(torch.float32)
    y_tr32 = y_tr.to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = x_c.shape[0]
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=g)
        pred = net(x_c[perm])
        loss = F.mse_loss(pred, y_tr32[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()

    def predict(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xc = ((x.to(torch.float32) - x_mean.to(torch.float32)) / x_std.to(torch.float32))
            return net(xc).to(torch.float64)

    n_params = int(sum(p.numel() for p in net.parameters()))
    return predict, n_params


def _ood_zscore(mapped: torch.Tensor, reference: torch.Tensor) -> Optional[float]:
    if mapped is None or mapped.shape[0] == 0:
        return None
    ref_mean = reference.mean(dim=0, keepdim=True)
    ref_std = reference.std(dim=0, keepdim=True).clamp(min=1e-6)
    z = (mapped - ref_mean) / ref_std
    return float(z.abs().mean())


def bridge_ladder(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    levels: Sequence[BridgeLevel] = _ALL_BRIDGE_LEVELS,
    low_rank_k: int = 4,
    l4_bottleneck: Optional[int] = None,
    l4_epochs: int = 200,
    l4_weight_decay: float = 1e-2,
    l5_hidden: int = 128,
    l5_epochs: int = 400,
    consumer_eval_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ood_x: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Fit and evaluate the L0-L5 frozen-endpoint bridge ladder, per spec
    section 4.1. Returns one dict per requested level, keyed by
    `BridgeLevel.value`, each with:

        bridge_class, rank, n_parameters, n_training_rows,
        heldout_mse, heldout_r2, downstream_behavioural_effect, ood

    `consumer_eval_fn`, if supplied, is called once per level as
    `consumer_eval_fn(mapped_x_test)` and its raw return value is stored
    under `downstream_behavioural_effect` -- this module never invents its
    own notion of "behavioural effect" (that is assay-specific: it is
    whatever the caller's frozen consumer does with the mapped input).
    `ood_x`, if supplied, is mapped through the SAME fitted bridge and
    reported as a mean-absolute-z-score against the training receiver
    distribution (`ood.mean_abs_zscore`) -- a cheap out-of-distribution
    receiver-state indicator; `manifold_guard` below gives the fuller
    Mahalanobis/kNN treatment when a caller needs it.

    `L5 success is reported as an information-in-principle statement and is
    never a plausible-interface claim` (spec section 4.4) -- this function
    does not decide that for the caller; it is a documentation obligation on
    whoever reads `L5_high_capacity_upper_bound`'s result.
    """
    dy = y_train.shape[1]
    out: Dict[str, Dict[str, Any]] = {}
    for level in levels:
        if level == BridgeLevel.L0_IDENTITY:
            predict, n_params = _fit_l0(x_train, y_train)
        elif level == BridgeLevel.L1_PROCRUSTES:
            predict, n_params = _fit_l1_procrustes(x_train, y_train)
        elif level == BridgeLevel.L2_AFFINE:
            predict, n_params = _fit_l2_affine(x_train, y_train)
        elif level == BridgeLevel.L3_LOW_RANK_AFFINE:
            predict, n_params = _fit_l3_low_rank_affine(x_train, y_train, rank=low_rank_k)
        elif level == BridgeLevel.L4_CONSTRAINED_NONLINEAR:
            bottleneck = l4_bottleneck if l4_bottleneck is not None else max(2, min(8, dy))
            predict, n_params = _fit_nonlinear_bridge(
                x_train, y_train, hidden=bottleneck, weight_decay=l4_weight_decay,
                epochs=l4_epochs, seed=seed,
            )
        elif level == BridgeLevel.L5_HIGH_CAPACITY:
            predict, n_params = _fit_nonlinear_bridge(
                x_train, y_train, hidden=l5_hidden, weight_decay=0.0,
                epochs=l5_epochs, seed=seed + 1,
            )
        else:  # pragma: no cover -- exhaustive over BridgeLevel
            raise ValueError("unknown bridge level: %r" % (level,))

        y_pred_test = predict(x_test)
        behavioural = consumer_eval_fn(y_pred_test) if consumer_eval_fn is not None else None
        ood_score = None
        if ood_x is not None:
            ood_score = _ood_zscore(predict(ood_x), y_train)

        out[level.value] = {
            "bridge_class": level.value,
            "rank": (int(low_rank_k) if level == BridgeLevel.L3_LOW_RANK_AFFINE else None),
            "n_parameters": int(n_params),
            "n_training_rows": int(x_train.shape[0]),
            "heldout_mse": _mse(y_test, y_pred_test),
            "heldout_r2": _r2(y_test, y_pred_test),
            "downstream_behavioural_effect": behavioural,
            "ood": {"provided": ood_x is not None, "mean_abs_zscore": ood_score},
        }
    return out


# ---------------------------------------------------------------------------
# 5b. Conditioned bridge rungs -- receiver-state conditioning T(A, B)
#     (MECH-547) and reference-frame mediation (MECH-555)
#
#     `bridge_ladder` above fits X -> Y UNCONDITIONALLY. Both claims below say
#     the translation may take a SECOND argument -- the receiver's own state
#     (MECH-547) or the frame the content is indexed against (MECH-555) -- and
#     both are discriminated from "we simply gave the bridge more capacity" by
#     a PERMUTATION control at MATCHED CAPACITY, never by a raw gain.
#
#     Design sources, read before changing anything here:
#       REE_assembly/docs/architecture/receiver_conditioned_translation.md
#         ("a sender-only matched-capacity baseline T(A)"; "a receiver-state
#          permutation control (the discriminator: if permutation barely
#          changes the gain, the gain is capacity, not conditioning)")
#       REE_assembly/docs/architecture/interface_reference_frames_and_temporal_gates.md
#         ("the discriminator is a content-preserving frame permutation";
#          "intact ~= frame-permuted -> the frame is incidental at that interface")
#       REE_assembly/evidence/planning/hippocampal_campaign_assay_specifications_20260910.md
#         section 2.2 arms A3_frame_cond / A4_receiver_state_cond / A7_receiver_only,
#         and section 2.7's falsifier: "A4's advantage survives receiver-state
#         permutation intact" -- i.e. CAPACITY_NOT_CONDITIONING below is not an
#         error path, it is the pre-registered way this claim dies.
#
#     Capacity matching is achieved STRUCTURALLY, not by post-hoc adjustment:
#     every arm is fitted by the SAME `bridge_ladder` call at the SAME level
#     over an input of the SAME width, so the arms differ only in what the
#     conditioning block CONTAINS. `n_parameters` is then compared across arms
#     and a mismatch REFUSES a verdict rather than reporting an adjusted one.
# ---------------------------------------------------------------------------


class ConditioningVerdict(str, enum.Enum):
    """Outcome of `receiver_conditioned_bridge`."""

    CONDITIONING_SUPPORTED = "receiver_conditioning_supported"
    CAPACITY_NOT_CONDITIONING = "capacity_not_conditioning"
    NO_CONDITIONING_GAIN = "no_conditioning_gain"
    RECEIVER_STATE_LEAKS_TARGET = "receiver_state_leaks_target"
    MATCHING_FAILED = "matching_failed"


class FrameVerdict(str, enum.Enum):
    """Outcome of `frame_permutation_control`."""

    FRAME_LOAD_BEARING = "frame_load_bearing"
    FRAME_INCIDENTAL = "frame_incidental"
    CONTROL_INVALID = "control_invalid"
    MATCHING_FAILED = "matching_failed"


@dataclass(frozen=True)
class ConditionedBridgeResult:
    """Result of the MECH-547 receiver-conditioned bridge contrast.

    `arms` holds the full per-arm `bridge_ladder` dict (so every capacity and
    held-out number stays inspectable); `scores` is the single scalar each arm
    was judged on. `verdict` is only meaningful when `capacity_matched` is
    True -- see `ConditioningVerdict.MATCHING_FAILED`.
    """

    arms: Dict[str, Dict[str, Any]]
    scores: Dict[str, float]
    parameter_counts: Dict[str, int]
    capacity_matched: bool
    conditioning_gain: float
    permuted_gain: float
    permutation_destroyed_fraction: Optional[float]
    verdict: str
    notes: List[str]


@dataclass(frozen=True)
class FramePermutationResult:
    """Result of the MECH-555 content-preserving frame-permutation contrast."""

    arms: Dict[str, Dict[str, Any]]
    scores: Dict[str, float]
    parameter_counts: Dict[str, int]
    capacity_matched: bool
    content_preserved: bool
    frame_gain: float
    permuted_gain: float
    permutation_destroyed_fraction: Optional[float]
    n_frames: int
    verdict: str
    notes: List[str]


def _moment_matched_random(block: torch.Tensor, seed: int) -> torch.Tensor:
    """A random block with `block`'s per-column mean and std, and no
    information about it. The matched-capacity filler for a baseline arm:
    same shape, same first two moments, zero mutual information with the
    quantity it stands in for (the "moment-matched random control" named in
    receiver_conditioned_translation.md's assay arm 6).
    """
    g = torch.Generator().manual_seed(int(seed))
    mean = block.mean(dim=0, keepdim=True)
    std = block.std(dim=0, keepdim=True)
    noise = torch.randn(block.shape, generator=g, dtype=block.dtype)
    return noise * std + mean


def _row_permutation(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(n), generator=g)


def _default_arm_score(arm: Dict[str, Any]) -> float:
    return float(arm["heldout_r2"])


def _fit_one_arm(name: str, x_tr: torch.Tensor, y_tr: torch.Tensor,
                 x_te: torch.Tensor, y_te: torch.Tensor,
                 *, level: BridgeLevel, seed: int,
                 ladder_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Every arm goes through the SAME `bridge_ladder` entry point at the SAME
    level and seed. Routing all arms through one fitter is what makes the
    capacity comparison meaningful -- do not "optimise" this into per-arm
    bespoke fitting.

    The arm's own name is stamped into the returned dict under `arm`, so a
    record stays self-identifying once an assay serialises it into a manifest,
    and so a caller-supplied `score_from_arm` can key on which arm it is
    scoring (the consumer-use-gain path the assay spec's primary estimand
    needs).
    """
    arm = dict(bridge_ladder(
        x_tr, y_tr, x_te, y_te,
        levels=[level], seed=seed, **ladder_kwargs
    )[level.value])
    arm["arm"] = str(name)
    return arm


def receiver_conditioned_bridge(
    sender_train: torch.Tensor,
    receiver_state_train: torch.Tensor,
    y_train: torch.Tensor,
    sender_test: torch.Tensor,
    receiver_state_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    level: BridgeLevel = BridgeLevel.L3_LOW_RANK_AFFINE,
    score_from_arm: Optional[Callable[[Dict[str, Any]], float]] = None,
    gain_threshold: float = 0.02,
    permutation_destruction_frac: float = 0.5,
    leak_frac: float = 0.9,
    pad_baselines: bool = True,
    seed: int = 0,
    **ladder_kwargs: Any,
) -> ConditionedBridgeResult:
    """MECH-547's `T(A, B)` rung with its receiver-state permutation control.

    Four arms, fitted identically and (by construction) at matched capacity:

      `sender_only`        `concat(A, moment_matched_random(B))` -- the
                           matched-capacity baseline `T(A)`. The conditioning
                           block is present, so the parameter count matches,
                           but it carries no information about the receiver.
      `conditioned`        `concat(A, B)` -- `T(A, B)` itself.
      `receiver_permuted`  `concat(A, B[perm])` -- THE DISCRIMINATOR. The
                           receiver block keeps its exact marginal
                           distribution and only its PAIRING with the sender
                           is destroyed, so any gain that survives here was
                           never conditioning.
      `receiver_only`      `concat(moment_matched_random(A), B)` -- assay
                           spec arm `A7_receiver_only`, the leakage guard: if
                           the receiver state alone already predicts the
                           target, the conditioned gain is uninterpretable
                           because the bridge introduced rather than
                           translated content.

    `pad_baselines=False` replaces `sender_only` with the NATIVE, unpadded
    `A -> Y` bridge. That is a legitimate thing to look at (it is the assay
    spec's `A1_source_only`), but it is NOT capacity-matched, so this function
    then reports `MATCHING_FAILED` and refuses a conditioning verdict rather
    than quietly crediting the conditioned arm with a capacity advantage.

    The verdict ladder, in the order it is evaluated:

      1. capacity mismatch                     -> MATCHING_FAILED
      2. receiver state alone predicts Y       -> RECEIVER_STATE_LEAKS_TARGET
      3. conditioning gain below threshold     -> NO_CONDITIONING_GAIN
      4. permutation preserves the gain        -> CAPACITY_NOT_CONDITIONING
      5. permutation destroys the gain         -> CONDITIONING_SUPPORTED

    Only (5) supports MECH-547. (4) is the assay spec's section 2.7 falsifier
    and is a first-class outcome, not a failure of the instrument.

    A property of the control worth recording, because it bears on how (4) is
    read: a ROW permutation destroys every PER-ROW conditioning signal by
    construction, so CAPACITY_NOT_CONDITIONING cannot be produced by any
    designed per-row dependence of Y on B. When it fires on real data the gain
    is coming from capacity or optimisation geometry (extra input width,
    ridge conditioning, over-parameterised fitting), which is exactly the
    reading the architecture doc gives it -- "extra capacity, not
    conditioning". It is therefore not synthesisable in a contract test from
    data alone; the verdict ladder is exercised there through the public
    `score_from_arm` hook instead.

    Pure: fits, scores and returns; writes nothing and holds no agent state.
    """
    score_fn = score_from_arm if score_from_arm is not None else _default_arm_score
    notes: List[str] = []

    if sender_train.shape[0] != receiver_state_train.shape[0]:
        raise ValueError(
            "sender_train and receiver_state_train must have the same number of rows, got %d vs %d"
            % (sender_train.shape[0], receiver_state_train.shape[0])
        )
    if sender_test.shape[0] != receiver_state_test.shape[0]:
        raise ValueError(
            "sender_test and receiver_state_test must have the same number of rows, got %d vs %d"
            % (sender_test.shape[0], receiver_state_test.shape[0])
        )

    perm_tr = _row_permutation(receiver_state_train.shape[0], seed + 101)
    perm_te = _row_permutation(receiver_state_test.shape[0], seed + 102)
    rand_b_tr = _moment_matched_random(receiver_state_train, seed + 201)
    rand_b_te = _moment_matched_random(receiver_state_test, seed + 202)
    rand_a_tr = _moment_matched_random(sender_train, seed + 301)
    rand_a_te = _moment_matched_random(sender_test, seed + 302)

    cat = lambda a, b: torch.cat([a, b], dim=1)

    arm_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "conditioned": (cat(sender_train, receiver_state_train),
                        cat(sender_test, receiver_state_test)),
        "receiver_permuted": (cat(sender_train, receiver_state_train[perm_tr]),
                              cat(sender_test, receiver_state_test[perm_te])),
        "receiver_only": (cat(rand_a_tr, receiver_state_train),
                          cat(rand_a_te, receiver_state_test)),
    }
    if pad_baselines:
        arm_inputs["sender_only"] = (cat(sender_train, rand_b_tr),
                                     cat(sender_test, rand_b_te))
    else:
        arm_inputs["sender_only"] = (sender_train, sender_test)
        notes.append(
            "pad_baselines=False: sender_only is the NATIVE unpadded A->Y bridge "
            "(assay-spec A1_source_only), which is not capacity-matched to the "
            "conditioned arm -- no conditioning verdict is issued."
        )

    arms: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    parameter_counts: Dict[str, int] = {}
    for name, (x_tr, x_te) in arm_inputs.items():
        arm = _fit_one_arm(name, x_tr, y_train, x_te, y_test,
                           level=level, seed=seed, ladder_kwargs=ladder_kwargs)
        arms[name] = arm
        scores[name] = float(score_fn(arm))
        parameter_counts[name] = int(arm["n_parameters"])

    distinct_counts = set(parameter_counts.values())
    capacity_matched = len(distinct_counts) == 1
    if not capacity_matched:
        notes.append(
            "capacity NOT matched across arms: %r" % (dict(parameter_counts),)
        )

    baseline = scores["sender_only"]
    conditioning_gain = scores["conditioned"] - baseline
    permuted_gain = scores["receiver_permuted"] - baseline
    destroyed_fraction: Optional[float] = None
    if conditioning_gain > 0.0:
        destroyed_fraction = float(1.0 - (permuted_gain / conditioning_gain))

    receiver_only_score = scores["receiver_only"]
    leaks = (
        receiver_only_score >= leak_frac * scores["conditioned"]
        and receiver_only_score > baseline + gain_threshold
    )

    if not capacity_matched:
        verdict = ConditioningVerdict.MATCHING_FAILED.value
    elif leaks:
        verdict = ConditioningVerdict.RECEIVER_STATE_LEAKS_TARGET.value
        notes.append(
            "receiver_only reaches %.4f vs conditioned %.4f -- the receiver state "
            "alone carries the target; the bridge would be introducing content, "
            "not translating it (assay spec 2.8 'A7_receiver_only high')."
            % (receiver_only_score, scores["conditioned"])
        )
    elif conditioning_gain <= gain_threshold:
        verdict = ConditioningVerdict.NO_CONDITIONING_GAIN.value
    elif permuted_gain > permutation_destruction_frac * conditioning_gain:
        verdict = ConditioningVerdict.CAPACITY_NOT_CONDITIONING.value
        notes.append(
            "permutation preserved %.0f%% of the gain -- an advantage a receiver-state "
            "permutation cannot touch was never conditioned on state (assay spec 2.7)."
            % (100.0 * (permuted_gain / conditioning_gain))
        )
    else:
        verdict = ConditioningVerdict.CONDITIONING_SUPPORTED.value

    return ConditionedBridgeResult(
        arms=arms,
        scores=scores,
        parameter_counts=parameter_counts,
        capacity_matched=capacity_matched,
        conditioning_gain=float(conditioning_gain),
        permuted_gain=float(permuted_gain),
        permutation_destroyed_fraction=destroyed_fraction,
        verdict=verdict,
        notes=notes,
    )


def _one_hot(labels: Sequence[Any], vocab: Sequence[Any],
             dtype: torch.dtype) -> torch.Tensor:
    index = {lab: i for i, lab in enumerate(vocab)}
    out = torch.zeros(len(labels), len(vocab), dtype=dtype)
    for row, lab in enumerate(labels):
        out[row, index[lab]] = 1.0
    return out


def frame_permutation_control(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    frame_train: Sequence[Any],
    frame_test: Sequence[Any],
    level: BridgeLevel = BridgeLevel.L3_LOW_RANK_AFFINE,
    score_from_arm: Optional[Callable[[Dict[str, Any]], float]] = None,
    gain_threshold: float = 0.02,
    permutation_destruction_frac: float = 0.5,
    frame_permutation_train: Optional[Sequence[Any]] = None,
    frame_permutation_test: Optional[Sequence[Any]] = None,
    seed: int = 0,
    **ladder_kwargs: Any,
) -> FramePermutationResult:
    """MECH-555's content-preserving frame permutation (assay arm `A3_frame_cond`).

    Three arms, all fitted at identical width and capacity, differing only in
    what the frame channel carries:

      `no_frame`          `concat(X, zeros)` -- the frame channel is present
                          (so the parameter count matches) but empty.
      `frame_conditioned` `concat(X, onehot(frame))` -- the intact frame.
      `frame_permuted`    `concat(X, onehot(scrambled frame))` -- THE
                          DISCRIMINATOR. The scramble is a ROW PERMUTATION of
                          the frame label vector, so the marginal frame
                          distribution is preserved exactly and only the
                          frame-to-content correspondence is destroyed.

    Why a row permutation and not a relabelling: a CONSISTENT relabelling of
    the frame vocabulary (frame g -> sigma(g) for every row of g) permutes the
    one-hot COLUMNS, and a linear or MLP fit is exactly invariant to that -- it
    would be a degenerate control that can never fail. The row permutation is
    the non-degenerate form, and it is content-preserving in the strict sense
    the architecture doc requires: no element of `x_train`, `y_train`,
    `x_test` or `y_test` is altered by any arm.

    `content_preserved` additionally verifies that the scrambled label vector
    is a genuine permutation of the intact one (same multiset). A caller that
    overrides `frame_permutation_train` / `frame_permutation_test` with a
    vector that is NOT a permutation (e.g. a constant) has changed the frame
    MARGINAL as well as the correspondence, which confounds the contrast --
    that is reported as `CONTROL_INVALID`, not silently scored.

    Verdicts:
      capacity mismatch or a non-permutation scramble -> MATCHING_FAILED / CONTROL_INVALID
      intact gain over `no_frame` below threshold     -> FRAME_INCIDENTAL
      permutation preserves the gain                  -> FRAME_INCIDENTAL
      permutation destroys the gain                   -> FRAME_LOAD_BEARING

    Note the asymmetry with `receiver_conditioned_bridge`: "intact ~= permuted"
    and "no gain at all" are BOTH `FRAME_INCIDENTAL` here, because the
    architecture doc's reading table gives them the same interpretation ("the
    frame is incidental at that interface"); the `notes` field records which
    of the two produced the verdict.

    Pure: fits, scores and returns; writes nothing.
    """
    score_fn = score_from_arm if score_from_arm is not None else _default_arm_score
    notes: List[str] = []

    frame_train = list(frame_train)
    frame_test = list(frame_test)
    if len(frame_train) != x_train.shape[0]:
        raise ValueError(
            "frame_train has %d labels for %d training rows"
            % (len(frame_train), x_train.shape[0])
        )
    if len(frame_test) != x_test.shape[0]:
        raise ValueError(
            "frame_test has %d labels for %d test rows"
            % (len(frame_test), x_test.shape[0])
        )

    vocab = sorted(set(frame_train) | set(frame_test), key=repr)
    n_frames = len(vocab)

    if frame_permutation_train is None:
        perm_tr = _row_permutation(len(frame_train), seed + 401)
        scrambled_train = [frame_train[int(i)] for i in perm_tr]
    else:
        scrambled_train = list(frame_permutation_train)
    if frame_permutation_test is None:
        perm_te = _row_permutation(len(frame_test), seed + 402)
        scrambled_test = [frame_test[int(i)] for i in perm_te]
    else:
        scrambled_test = list(frame_permutation_test)

    content_preserved = (
        sorted(map(repr, scrambled_train)) == sorted(map(repr, frame_train))
        and sorted(map(repr, scrambled_test)) == sorted(map(repr, frame_test))
    )
    if not content_preserved:
        notes.append(
            "the supplied frame scramble is NOT a permutation of the intact label "
            "vector -- it changes the frame marginal as well as the "
            "frame-to-content correspondence, which confounds the contrast."
        )
        vocab = sorted(set(vocab) | set(scrambled_train) | set(scrambled_test), key=repr)
        n_frames = len(vocab)

    dtype = x_train.dtype
    oh_tr = _one_hot(frame_train, vocab, dtype)
    oh_te = _one_hot(frame_test, vocab, dtype)
    oh_tr_perm = _one_hot(scrambled_train, vocab, dtype)
    oh_te_perm = _one_hot(scrambled_test, vocab, dtype)
    zeros_tr = torch.zeros(x_train.shape[0], n_frames, dtype=dtype)
    zeros_te = torch.zeros(x_test.shape[0], n_frames, dtype=dtype)

    cat = lambda a, b: torch.cat([a, b], dim=1)
    arm_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "no_frame": (cat(x_train, zeros_tr), cat(x_test, zeros_te)),
        "frame_conditioned": (cat(x_train, oh_tr), cat(x_test, oh_te)),
        "frame_permuted": (cat(x_train, oh_tr_perm), cat(x_test, oh_te_perm)),
    }

    arms: Dict[str, Dict[str, Any]] = {}
    scores: Dict[str, float] = {}
    parameter_counts: Dict[str, int] = {}
    for name, (x_tr, x_te) in arm_inputs.items():
        arm = _fit_one_arm(name, x_tr, y_train, x_te, y_test,
                           level=level, seed=seed, ladder_kwargs=ladder_kwargs)
        arms[name] = arm
        scores[name] = float(score_fn(arm))
        parameter_counts[name] = int(arm["n_parameters"])

    capacity_matched = len(set(parameter_counts.values())) == 1
    if not capacity_matched:
        notes.append("capacity NOT matched across arms: %r" % (dict(parameter_counts),))

    baseline = scores["no_frame"]
    frame_gain = scores["frame_conditioned"] - baseline
    permuted_gain = scores["frame_permuted"] - baseline
    destroyed_fraction: Optional[float] = None
    if frame_gain > 0.0:
        destroyed_fraction = float(1.0 - (permuted_gain / frame_gain))

    if not capacity_matched:
        verdict = FrameVerdict.MATCHING_FAILED.value
    elif not content_preserved:
        verdict = FrameVerdict.CONTROL_INVALID.value
    elif frame_gain <= gain_threshold:
        verdict = FrameVerdict.FRAME_INCIDENTAL.value
        notes.append(
            "no frame gain over the empty-frame baseline (%.4f <= %.4f): the frame "
            "channel bought nothing at this interface." % (frame_gain, gain_threshold)
        )
    elif permuted_gain > permutation_destruction_frac * frame_gain:
        verdict = FrameVerdict.FRAME_INCIDENTAL.value
        notes.append(
            "permutation preserved %.0f%% of the frame gain -- intact ~= frame-permuted, "
            "so the frame is incidental at this interface."
            % (100.0 * (permuted_gain / frame_gain))
        )
    else:
        verdict = FrameVerdict.FRAME_LOAD_BEARING.value

    return FramePermutationResult(
        arms=arms,
        scores=scores,
        parameter_counts=parameter_counts,
        capacity_matched=capacity_matched,
        content_preserved=content_preserved,
        frame_gain=float(frame_gain),
        permuted_gain=float(permuted_gain),
        permutation_destroyed_fraction=destroyed_fraction,
        n_frames=int(n_frames),
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 6. causal_replacement -- pairing-specific causal audit
# ---------------------------------------------------------------------------


class CausalReplacementCategory(str, enum.Enum):
    """The interpretive categories of mutual-legibility doc section 10
    ("Minimum interpretive categories"). `AMBIGUOUS` is a defensive sixth
    sentinel NOT in that section -- returned when the measured scores do not
    cleanly match any of the five documented patterns, rather than forcing a
    wrong classification. A caller must treat `AMBIGUOUS` as "uninterpretable
    under this rule", never silently as one of the five.
    """
    PAIRING_SPECIFIC_CONTENT_USE = "pairing_specific_content_use"
    GENERIC_CHANNEL_EFFECT = "generic_channel_effect"
    MIXED_GENERIC_AND_CONTENT_SPECIFIC = "mixed_generic_and_content_specific"
    MISLEADING_CONTENT = "misleading_content"
    NOT_LOAD_BEARING = "not_load_bearing"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CausalReplacementResult:
    category: CausalReplacementCategory
    scores: Dict[str, float]
    band: float
    dominance: float


def _approx_equal(a: float, b: float, band: float) -> bool:
    return abs(a - b) <= band


def _much_greater(a: float, b: float, dominance: float) -> bool:
    return (a - b) >= dominance


def causal_replacement(
    eval_fn: Callable[[torch.Tensor], float],
    *,
    correct: torch.Tensor,
    mismatched: torch.Tensor,
    zero: Optional[torch.Tensor] = None,
    moment_matched_random: Optional[torch.Tensor] = None,
    same_action_mismatch: Optional[torch.Tensor] = None,
    same_context_mismatch: Optional[torch.Tensor] = None,
    band: float = 0.05,
    dominance: Optional[float] = None,
    seed: int = 0,
) -> CausalReplacementResult:
    """Pairing-specific causal audit, per spec section 4.1 and mutual-
    legibility doc section 10.

    `eval_fn` is called ONCE PER CONDITION on the whole batch for that
    condition and must return a single scalar score for it (e.g. mean
    consumer-use agreement over the batch) -- this module has no consumer of
    its own, so scoring is entirely the caller's `eval_fn`.

    `correct` and `mismatched` are required (every one of the doc's five
    categories is defined in terms of them). `zero` defaults to
    `torch.zeros_like(correct)`; `moment_matched_random` defaults to a
    per-dimension-moment-matched Gaussian draw from `correct` (seeded, so a
    repeated call with the same `seed` reproduces the same random condition).
    `same_action_mismatch`/`same_context_mismatch` are optional REFINEMENTS
    (doc section 10 lists them as "optional") -- their scores are recorded in
    the result but do not themselves change the five-category classification,
    which is defined purely from correct/mismatched/zero/random.

    `band` is the "~=" (approximately equal) tolerance and `dominance`
    (default `2 * band`) is the ">>"/"much greater than" tolerance used by
    the classification rule below -- both explicit, tunable parameters, not
    claims about a universally correct magnitude.
    """
    if dominance is None:
        dominance = 2.0 * band
    if zero is None:
        zero = torch.zeros_like(correct)
    if moment_matched_random is None:
        mean = correct.mean(dim=0, keepdim=True)
        std = correct.std(dim=0, keepdim=True)
        g = torch.Generator().manual_seed(int(seed))
        moment_matched_random = mean + std * torch.randn(correct.shape, generator=g)

    scores: Dict[str, float] = {
        "correct": float(eval_fn(correct)),
        "mismatched": float(eval_fn(mismatched)),
        "zero": float(eval_fn(zero)),
        "moment_matched_random": float(eval_fn(moment_matched_random)),
    }
    if same_action_mismatch is not None:
        scores["same_action_mismatch"] = float(eval_fn(same_action_mismatch))
    if same_context_mismatch is not None:
        scores["same_context_mismatch"] = float(eval_fn(same_context_mismatch))

    correct_s = scores["correct"]
    mismatched_s = scores["mismatched"]
    zero_s = scores["zero"]
    random_s = scores["moment_matched_random"]
    low_baseline = min(zero_s, random_s)

    if _approx_equal(correct_s, zero_s, band):
        category = CausalReplacementCategory.NOT_LOAD_BEARING
    elif mismatched_s < zero_s - band:
        category = CausalReplacementCategory.MISLEADING_CONTENT
    elif _much_greater(correct_s, mismatched_s, dominance) and _approx_equal(mismatched_s, random_s, band):
        category = CausalReplacementCategory.PAIRING_SPECIFIC_CONTENT_USE
    elif _approx_equal(correct_s, mismatched_s, band) and _much_greater(mismatched_s, zero_s, dominance):
        category = CausalReplacementCategory.GENERIC_CHANNEL_EFFECT
    elif correct_s > mismatched_s > low_baseline:
        category = CausalReplacementCategory.MIXED_GENERIC_AND_CONTENT_SPECIFIC
    else:
        category = CausalReplacementCategory.AMBIGUOUS

    return CausalReplacementResult(category=category, scores=scores, band=band, dominance=dominance)


# ---------------------------------------------------------------------------
# 7. manifold_guard -- receiver-manifold off-manifold detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifoldGuardResult:
    mahalanobis_mean: float
    mahalanobis_max: float
    knn_distance_mean: float
    knn_distance_max: float
    mapped_mean_abs_zscore: float
    mapped_std_ratio_mean: float
    activation_saturation_rate: Optional[float]
    hidden_trajectory_distance: Optional[Dict[str, float]]
    off_manifold_flag: bool


def manifold_guard(
    native_states: torch.Tensor,
    mapped_states: torch.Tensor,
    *,
    k_neighbors: int = 5,
    ridge: float = 1e-3,
    mahalanobis_threshold: float = 6.0,
    activations: Optional[torch.Tensor] = None,
    saturation_abs_threshold: float = 0.95,
    hidden_trajectory_native: Optional[torch.Tensor] = None,
    hidden_trajectory_mapped: Optional[torch.Tensor] = None,
) -> ManifoldGuardResult:
    """Detect off-manifold bridge outputs, per spec section 4.1 and
    mutual-legibility doc section 9.

    `native_states` [N, d] is the distribution the receiver actually
    occupies (e.g. a held-out sample of true receiver inputs);
    `mapped_states` [M, d] is the candidate/bridged states under test.

    `activations`, if supplied, is the receiver's own activation tensor on
    `mapped_states` (already computed by the caller); saturation rate is the
    fraction of entries with `abs(value) >= saturation_abs_threshold`, which
    assumes a BOUNDED activation (tanh/sigmoid-like) -- documented rather
    than silently wrong for an unbounded one (ReLU, etc.); pass `None` there
    if the receiver's activation is unbounded.

    `hidden_trajectory_native`/`hidden_trajectory_mapped`, if both supplied,
    are [T, d_h] consumer hidden-state trajectories (native vs. driven by the
    mapped states) already produced by the caller's own rollout; this
    function only measures the per-step distance between them (step 1 and
    the final step), per the doc's "consumer hidden-state trajectory
    distance after one and after N steps" requirement -- it performs no
    rollout itself (this module holds no agent/consumer reference).

    `off_manifold_flag` is True when the mean Mahalanobis distance of the
    mapped states exceeds `mahalanobis_threshold` (default 6.0 standard
    deviations under the native covariance -- generous on purpose, since the
    doc explicitly warns "do not require exact in-distribution identity; a
    bridge may validly expose states not frequently visited natively. The
    guard is to detect EXTREME off-manifold shortcuts").
    """
    native = native_states.to(torch.float64)
    mapped = mapped_states.to(torch.float64)
    mean = native.mean(dim=0, keepdim=True)
    centered_native = native - mean
    cov = (centered_native.T @ centered_native) / max(1, native.shape[0] - 1)
    inv_sqrt, _diag = _inv_sqrt_psd(cov, ridge)
    delta = (mapped - mean) @ inv_sqrt
    mahal = delta.norm(dim=-1)

    # kNN distance: for each mapped row, mean distance to its k nearest native rows.
    # O(M*N) pairwise distance -- fine at the row counts this instrument is used at
    # (thousands, not millions); no approximate-NN dependency needed.
    dmat = torch.cdist(mapped, native)
    k = max(1, min(int(k_neighbors), native.shape[0]))
    knn_vals, _ = torch.topk(dmat, k, dim=-1, largest=False)
    knn_mean_per_row = knn_vals.mean(dim=-1)

    native_std = native.std(dim=0, keepdim=True).clamp(min=1e-8)
    mapped_std = mapped.std(dim=0, keepdim=True)
    mapped_zscore = ((mapped.mean(dim=0, keepdim=True) - mean) / native_std).abs().mean()
    std_ratio = (mapped_std / native_std).mean()

    saturation_rate = None
    if activations is not None:
        saturation_rate = float((activations.abs() >= saturation_abs_threshold).to(torch.float64).mean())

    hidden_distance = None
    if hidden_trajectory_native is not None and hidden_trajectory_mapped is not None:
        t = min(hidden_trajectory_native.shape[0], hidden_trajectory_mapped.shape[0])
        diffs = (hidden_trajectory_native[:t] - hidden_trajectory_mapped[:t]).to(torch.float64)
        per_step = diffs.norm(dim=-1)
        hidden_distance = {
            "step_1": float(per_step[0]) if t >= 1 else None,
            "step_n": float(per_step[-1]) if t >= 1 else None,
            "n_steps": int(t),
        }

    mahal_mean = float(mahal.mean()) if mahal.numel() else 0.0
    return ManifoldGuardResult(
        mahalanobis_mean=mahal_mean,
        mahalanobis_max=float(mahal.max()) if mahal.numel() else 0.0,
        knn_distance_mean=float(knn_mean_per_row.mean()) if knn_mean_per_row.numel() else 0.0,
        knn_distance_max=float(knn_mean_per_row.max()) if knn_mean_per_row.numel() else 0.0,
        mapped_mean_abs_zscore=float(mapped_zscore),
        mapped_std_ratio_mean=float(std_ratio),
        activation_saturation_rate=saturation_rate,
        hidden_trajectory_distance=hidden_distance,
        off_manifold_flag=bool(mahal_mean > mahalanobis_threshold),
    )


# ---------------------------------------------------------------------------
# 8. dynamic_compatibility -- one-step and multi-step transition agreement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicCompatibilityResult:
    one_step_mse: float
    one_step_cosine: float
    multi_step_mse_by_horizon: Optional[List[float]]
    multi_step_cosine_by_horizon: Optional[List[float]]
    compounding: Optional[bool]


def dynamic_compatibility(
    predicted_next: torch.Tensor,
    target_next: torch.Tensor,
    *,
    multi_step_predicted: Optional[Sequence[torch.Tensor]] = None,
    multi_step_target: Optional[Sequence[torch.Tensor]] = None,
) -> DynamicCompatibilityResult:
    """One-step and multi-step transition-compatibility scoring (MECH-539),
    per spec section 4.1 and mutual-legibility doc section 11.

    One-step: compares `predicted_next = receiver_transition(T(x_t), a_t)`
    against `target_next = T(x_{t+1})`, both [N, d], already computed by the
    caller (this module performs no rollout of its own).

    Multi-step, if `multi_step_predicted`/`multi_step_target` are supplied
    (each a sequence of [N, d] tensors, one per horizon 1, 2, 4, ... as the
    caller's own rollout produced them): reports per-horizon MSE and cosine
    agreement, and `compounding` -- True when the per-horizon MSE is
    monotonically non-decreasing across ALL horizons (a compounding-error
    signature), False when it is not, `None` when fewer than two horizons
    were supplied to judge a trend from.
    """
    one_step_mse = _mse(target_next, predicted_next)
    one_step_cosine = _cosine_sim_mean(target_next, predicted_next)

    multi_mse: Optional[List[float]] = None
    multi_cos: Optional[List[float]] = None
    compounding: Optional[bool] = None
    if multi_step_predicted is not None and multi_step_target is not None:
        h = min(len(multi_step_predicted), len(multi_step_target))
        multi_mse = [_mse(multi_step_target[i], multi_step_predicted[i]) for i in range(h)]
        multi_cos = [_cosine_sim_mean(multi_step_target[i], multi_step_predicted[i]) for i in range(h)]
        if h >= 2:
            compounding = all(multi_mse[i + 1] >= multi_mse[i] for i in range(h - 1))

    return DynamicCompatibilityResult(
        one_step_mse=one_step_mse,
        one_step_cosine=one_step_cosine,
        multi_step_mse_by_horizon=multi_mse,
        multi_step_cosine_by_horizon=multi_cos,
        compounding=compounding,
    )


# ---------------------------------------------------------------------------
# 9. per_code_drift -- transfer-relevant vs transfer-irrelevant drift rate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerCodeDriftResult:
    in_subspace_drift_norm_mean: float
    out_subspace_drift_norm_mean: float
    in_subspace_dim: int
    out_subspace_dim: int
    in_subspace_drift_rate_per_dim: float
    out_subspace_drift_rate_per_dim: float


def per_code_drift(
    pre: torch.Tensor,
    post: torch.Tensor,
    decision_subspace_basis: torch.Tensor,
) -> PerCodeDriftResult:
    """Drift rate in the transfer-relevant directions of a declared decision
    subspace versus its orthogonal (transfer-irrelevant) complement, per
    spec section 4.1 and tranche-3 debt 9.

    `pre`/`post` are [N, d] representations of the SAME rows at two points
    in time (before/after a drift-inducing interval). `decision_subspace_basis`
    is a [d, k] basis of the transfer-relevant directions, supplied by the
    caller -- this module does not compute what "decision-relevant" means
    for any particular oracle/task (mutual-legibility doc section 15: no
    task-specific semantics in the generic layer). A caller wanting one can
    build it with a task-specific helper such as v3_exq_1008's
    `_decision_subspace_retention`, or any other declared linear map's
    column space.

    Drift rate is normalised per-dimension (mean projected drift norm
    divided by the subspace's own dimensionality) so the in-subspace and
    out-of-subspace rates are comparable even when the two subspaces have
    very different dimensionality.
    """
    d = pre.shape[1]
    basis, _ = torch.linalg.qr(decision_subspace_basis.to(torch.float64))
    k = basis.shape[1]
    drift = (post - pre).to(torch.float64)
    proj_in = drift @ basis  # [N, k]
    proj_full = proj_in @ basis.T  # [N, d], the in-subspace component reconstructed
    proj_out = drift - proj_full

    in_norm_mean = float(proj_in.norm(dim=-1).mean()) if proj_in.numel() else 0.0
    out_norm_mean = float(proj_out.norm(dim=-1).mean()) if proj_out.numel() else 0.0
    out_dim = max(1, d - k)
    return PerCodeDriftResult(
        in_subspace_drift_norm_mean=in_norm_mean,
        out_subspace_drift_norm_mean=out_norm_mean,
        in_subspace_dim=int(k),
        out_subspace_dim=int(d - k),
        in_subspace_drift_rate_per_dim=(in_norm_mean / k) if k > 0 else 0.0,
        out_subspace_drift_rate_per_dim=(out_norm_mean / out_dim),
    )


# ---------------------------------------------------------------------------
# 10. --selftest -- also the per-cell wall-time cost meter (spec section 4.3)
# ---------------------------------------------------------------------------


def _run_selftest() -> bool:
    """Run every assertion spec section 4.3 requires, on SYNTHETIC data with
    a known invertible relation, and print per-cell wall time (the cost
    meter that converts the spec's sections 2.9/3.9 compute estimates into
    schedulable numbers). Returns True iff every assertion passed; prints
    ASCII-only, per project convention.
    """
    import time

    from experiments._lib.arm_fingerprint import reset_all_rng

    reset_all_rng(20260910)
    ok = True
    cell_times: Dict[str, float] = {}

    def _cell(name: str, fn: Callable[[], None]) -> None:
        nonlocal ok
        t0 = time.time()
        try:
            fn()
            print("  [PASS] %s" % name)
        except AssertionError as exc:
            ok = False
            print("  [FAIL] %s -- %s" % (name, exc))
        cell_times[name] = time.time() - t0

    n, d = 400, 8

    # -- L1 recovers a pure rotation ------------------------------------------------
    def _l1_rotation() -> None:
        x = torch.randn(n, d)
        r0 = _random_orthonormal(d, d, seed=1)
        y = x @ r0
        x_tr, x_te = x[: n // 2], x[n // 2 :]
        y_tr, y_te = y[: n // 2], y[n // 2 :]
        levels = (BridgeLevel.L0_IDENTITY, BridgeLevel.L1_PROCRUSTES)
        res = bridge_ladder(x_tr, y_tr, x_te, y_te, levels=levels, seed=2)
        l1_r2 = res[BridgeLevel.L1_PROCRUSTES.value]["heldout_r2"]
        assert l1_r2 > 0.99, "L1 heldout_r2=%.4f, expected > 0.99 on a pure rotation" % l1_r2

    _cell("L1 recovers a pure rotation", _l1_rotation)

    # -- L0 fails and L3 succeeds on a known low-rank relation ----------------------
    def _l0_fails_l3_succeeds() -> None:
        x = torch.randn(n, d)
        u = _random_orthonormal(d, 2, seed=3)
        v = _random_orthonormal(d, 2, seed=4)
        w_lowrank = u @ v.T * 3.0  # rank-2, dx=dy=d so L0 (identity) is well-defined but wrong
        noise = 0.01 * torch.randn(n, d)
        y = x @ w_lowrank + noise
        x_tr, x_te = x[: n // 2], x[n // 2 :]
        y_tr, y_te = y[: n // 2], y[n // 2 :]
        levels = (BridgeLevel.L0_IDENTITY, BridgeLevel.L3_LOW_RANK_AFFINE)
        res = bridge_ladder(x_tr, y_tr, x_te, y_te, levels=levels, low_rank_k=4, seed=5)
        l0_r2 = res[BridgeLevel.L0_IDENTITY.value]["heldout_r2"]
        l3_r2 = res[BridgeLevel.L3_LOW_RANK_AFFINE.value]["heldout_r2"]
        assert l0_r2 < 0.5, "L0 heldout_r2=%.4f, expected well below 0.5 (identity is wrong here)" % l0_r2
        assert l3_r2 > 0.9, "L3 heldout_r2=%.4f, expected > 0.9 on a known rank-2 relation" % l3_r2

    _cell("L0 fails and L3 succeeds on a known low-rank relation", _l0_fails_l3_succeeds)

    # -- causal_replacement: correct >> mismatched ~= random on a pairing-specific --
    def _causal_pairing_specific() -> None:
        m = 64
        codebook = torch.randn(m, d) * 5.0  # well-separated per-row codes

        def nn_match_accuracy(batch: torch.Tensor) -> float:
            dmat = torch.cdist(batch, codebook)
            nearest = dmat.argmin(dim=-1)
            return float((nearest == torch.arange(m)).to(torch.float64).mean())

        shuffle = torch.randperm(m, generator=torch.Generator().manual_seed(6))
        result = causal_replacement(
            nn_match_accuracy,
            correct=codebook,
            mismatched=codebook[shuffle],
            seed=7,
        )
        cat = result.category
        assert cat == CausalReplacementCategory.PAIRING_SPECIFIC_CONTENT_USE, (
            "expected PAIRING_SPECIFIC_CONTENT_USE, got %s (scores=%r)" % (cat, result.scores)
        )
        assert result.scores["correct"] > result.scores["mismatched"], result.scores

    _cell("causal_replacement: correct >> mismatched ~= random (pairing-specific)", _causal_pairing_specific)

    # -- causal_replacement: correct ~= mismatched >> zero on a generic-channel ------
    def _causal_generic_channel() -> None:
        m = 64
        presence_threshold = 0.5

        def presence_score(batch: torch.Tensor) -> float:
            return float((batch.norm(dim=-1) > presence_threshold).to(torch.float64).mean())

        correct = torch.randn(m, d) + 3.0  # any nonzero content triggers "presence"
        shuffle = torch.randperm(m, generator=torch.Generator().manual_seed(8))
        mismatched = correct[shuffle]
        result = causal_replacement(
            presence_score, correct=correct, mismatched=mismatched, seed=9,
        )
        cat = result.category
        assert cat == CausalReplacementCategory.GENERIC_CHANNEL_EFFECT, (
            "expected GENERIC_CHANNEL_EFFECT, got %s (scores=%r)" % (cat, result.scores)
        )

    _cell("causal_replacement: correct ~= mismatched >> zero (generic-channel)", _causal_generic_channel)

    # -- manifold_guard fires on a deliberately off-manifold map ---------------------
    def _manifold_guard_fires() -> None:
        native = torch.randn(200, d) * 0.5
        mapped_ok = torch.randn(20, d) * 0.5  # in-distribution -- should NOT fire
        mapped_off = torch.randn(20, d) * 0.5 + 100.0  # deliberately off-manifold

        guard_ok = manifold_guard(native, mapped_ok)
        guard_off = manifold_guard(native, mapped_off)
        assert not guard_ok.off_manifold_flag, "in-distribution mapped states incorrectly flagged"
        assert guard_off.off_manifold_flag, "deliberately off-manifold states were NOT flagged"
        assert guard_off.mahalanobis_mean > guard_ok.mahalanobis_mean

    _cell("manifold_guard fires on a deliberately off-manifold map", _manifold_guard_fires)

    # ---- 5b rungs: receiver conditioning (MECH-547) and frames (MECH-555) ----

    def _receiver_conditioning() -> None:
        gg = torch.Generator().manual_seed(4242)
        a_tr = torch.randn(400, 8, generator=gg, dtype=torch.float64)
        b_tr = torch.randn(400, 3, generator=gg, dtype=torch.float64)
        a_te = torch.randn(200, 8, generator=gg, dtype=torch.float64)
        b_te = torch.randn(200, 3, generator=gg, dtype=torch.float64)
        w = torch.randn(8, 4, generator=gg, dtype=torch.float64)
        v = torch.randn(3, 4, generator=gg, dtype=torch.float64) * 2.0
        res = receiver_conditioned_bridge(
            a_tr, b_tr, a_tr @ w + b_tr @ v,
            a_te, b_te, a_te @ w + b_te @ v,
            level=BridgeLevel.L2_AFFINE, seed=0,
        )
        assert res.capacity_matched, res.parameter_counts
        assert res.verdict == ConditioningVerdict.CONDITIONING_SUPPORTED.value, res.verdict
        assert res.permutation_destroyed_fraction is not None
        assert res.permutation_destroyed_fraction > 0.5, res.permutation_destroyed_fraction
        null = receiver_conditioned_bridge(
            a_tr, b_tr, a_tr @ w, a_te, b_te, a_te @ w,
            level=BridgeLevel.L2_AFFINE, seed=0,
        )
        assert null.verdict == ConditioningVerdict.NO_CONDITIONING_GAIN.value, null.verdict

    _cell("receiver_conditioned_bridge: real conditioning survives, permutation destroys it",
          _receiver_conditioning)

    def _frame_mediation() -> None:
        gg = torch.Generator().manual_seed(909)
        x_tr = torch.randn(400, 8, generator=gg, dtype=torch.float64)
        x_te = torch.randn(200, 8, generator=gg, dtype=torch.float64)
        w = torch.randn(8, 4, generator=gg, dtype=torch.float64)
        fv = torch.randn(3, 4, generator=gg, dtype=torch.float64) * 3.0
        fr_tr = ["f%d" % (i % 3) for i in range(400)]
        fr_te = ["f%d" % (i % 3) for i in range(200)]

        def onehot(labels: Sequence[str]) -> torch.Tensor:
            m = torch.zeros(len(labels), 3, dtype=torch.float64)
            for i, lab in enumerate(labels):
                m[i, int(lab[1])] = 1.0
            return m

        res = frame_permutation_control(
            x_tr, x_tr @ w + onehot(fr_tr) @ fv,
            x_te, x_te @ w + onehot(fr_te) @ fv,
            frame_train=fr_tr, frame_test=fr_te,
            level=BridgeLevel.L2_AFFINE, seed=0,
        )
        assert res.capacity_matched, res.parameter_counts
        assert res.content_preserved
        assert res.verdict == FrameVerdict.FRAME_LOAD_BEARING.value, res.verdict
        incidental = frame_permutation_control(
            x_tr, x_tr @ w, x_te, x_te @ w,
            frame_train=fr_tr, frame_test=fr_te,
            level=BridgeLevel.L2_AFFINE, seed=0,
        )
        assert incidental.verdict == FrameVerdict.FRAME_INCIDENTAL.value, incidental.verdict

    _cell("frame_permutation_control: load-bearing frame survives, incidental frame does not",
          _frame_mediation)

    print("\nPer-cell wall time (seconds):")
    total = 0.0
    for name, secs in cell_times.items():
        print("  %6.3f  %s" % (secs, name))
        total += secs
    print("  %6.3f  TOTAL" % total)
    print("\nSELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run the P0 self-test and exit")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if _run_selftest() else 1)
    parser.print_help()
