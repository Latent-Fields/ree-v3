"""GOV-MATCHAUX-1 matched arbitrary-auxiliary targets for the SD-070 P0a proximity channel.

WHY THIS EXISTS. GOV-MATCHAUX-1 (claims.yaml, candidate governance rule) admits a result as
evidence that a representation objective ORGANISES a latent -- rather than that extra supervision
helps -- only against a control that matches the objective's head capacity, loss budget, update
frequency and training examples while targeting learnable but organism-IRRELEVANT structure. The
SD-070 P0a recipe already has one caller-supplied scalar channel -- `trainer.observe(world_obs,
resource_proximity_target)` -> `split_encoder.resource_proximity_head` under `proximity_weight` --
so the cheapest exact match is a DIFFERENT scalar through the SAME channel. This module builds that
scalar. `run_zworld_p0(..., target_fn=MatchedArbitraryTarget(...))` is the seam that feeds it.

WHAT THE TARGET IS. A fixed random linear functional of the local view's NON-consequential
one-hot channels (empty / wall / and the remaining non-resource, non-hazard entity slots), chosen
among K seeded candidate directions as the one LEAST linearly correlated with the SD-018 resource
proximity target on a calibration rollout, then QUANTILE-MAPPED onto the proximity target's own
empirical distribution on that rollout. So, by construction:

  * same head, same MSE, same weight, same cadence, same examples as the regulatory arm
    (those are the trainer's, untouched);
  * the marginal distribution of the target -- hence its variance and entropy, the two things
    GOV-MATCHAUX-1 says must be matched or REPORTED -- equals the proximity target's on the
    calibration distribution (KS distance reported on a held-out calibration half, never assumed);
  * the target carries learnable structure (a monotone function of a linear function of the
    observation the encoder itself receives), with no organism meaning (walls and empty cells do not
    feed, harm, or drain the agent; resource and hazard channels are excluded from the functional,
    and the residual linear association with resource proximity is minimised by selection and
    REPORTED as `pearson_r_with_prox`, not assumed zero).

DECORRELATION IS AGAINST THE LINEAR PREDICTOR, NOT THE TARGET. After selecting the least-correlated
candidate direction d, the functional is adjusted to d' = d - beta * w_hat, where w_hat is the ridge
regression of the proximity target on the kept channels (fitted on the calibration fit-half) and
beta the slope of d.x on w_hat.x. d'.x is STILL a pure linear function of the observation -- it
never reads the proximity target at call time (which would make the control a function of the
organism-relevant scalar, with a negative coefficient, and leak exactly the content it must not
carry) -- but it is linearly uncorrelated with the best linear prox predictor available from those
channels. The residual Pearson r with prox itself is REPORTED on the held-out half, never assumed.

WHAT IT IS NOT. Not a substrate change: nothing in `ree_core/` is touched. Not a claim that the
control is information-free about the organism: it is a claim that the association is measured and
small, which is what GOV-MATCHAUX-1 asks ("any entropy/difficulty mismatch REPORTED").

ASCII-only (repo rule). Lives under `experiments/_lib/**`, so it is folded into `substrate_hash`
(an edit here correctly refuses a stale banked arm).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ree_core.latent.zworld_p0 import (
    HAZARD_ENTITY_INDEX,
    LOCAL_VIEW_CELLS,
    LOCAL_VIEW_ENTITY_STRIDE,
    RESOURCE_ENTITY_INDEX,
)
from experiments._lib.zworld_p0_warmup import resource_prox_target

__all__ = ["MatchedArbitraryTarget", "collect_calibration_obs", "ks_distance", "action_decodability"]

N_CANDIDATE_DIRECTIONS = 64
RIDGE = 1.0e-2   # ridge on the linear prox-predictor fit (125 dims, a few hundred rows)


def _kept_indices(exclude_entities: Sequence[int]) -> List[int]:
    """world_obs indices of the local-view one-hot slots whose entity is NOT excluded."""
    keep = []
    for c in range(LOCAL_VIEW_CELLS):
        for e in range(LOCAL_VIEW_ENTITY_STRIDE):
            if e in exclude_entities:
                continue
            keep.append(c * LOCAL_VIEW_ENTITY_STRIDE + e)
    return keep


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance (sup |F_a - F_b|); 0 = identical marginals."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


N_ACTION_BINS = 8


def _bin_fit(values: np.ndarray, actions: np.ndarray, n_bins: int = N_ACTION_BINS
             ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quantile-bin a scalar and record the majority action per bin (fitted on one split)."""
    qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    edges = np.unique(qs)
    bins = np.searchsorted(edges, values, side="right")
    n_actions = int(actions.max()) + 1 if actions.size else 1
    maj = np.zeros(len(edges) + 1, dtype=np.int64)
    global_maj = int(np.bincount(actions, minlength=n_actions).argmax()) if actions.size else 0
    for b in range(len(edges) + 1):
        m = bins == b
        maj[b] = int(np.bincount(actions[m], minlength=n_actions).argmax()) if m.any() else global_maj
    return edges, maj, global_maj


def _bin_acc(edges: np.ndarray, maj: np.ndarray, values: np.ndarray, actions: np.ndarray) -> float:
    if actions.size == 0:
        return float("nan")
    bins = np.searchsorted(edges, values, side="right")
    return float(np.mean(maj[bins] == actions))


def action_decodability(fit_values: np.ndarray, fit_actions: np.ndarray,
                        eval_values: np.ndarray, eval_actions: np.ndarray) -> Dict[str, float]:
    """How much of the oracle's ACTION a scalar target predicts, beyond the majority class:
    bins + per-bin majorities fitted on one split, accuracy scored on another. The leak witness
    for a 'matched' control (finding 5 of the V3-EXQ-1017 red-team): decorrelation from the
    proximity SCALAR does not by itself bound association with the DV's label."""
    if fit_actions.size < 8 or eval_actions.size < 8:
        return {"acc": float("nan"), "majority_baseline": float("nan"), "elevation": float("nan")}
    edges, maj, gmaj = _bin_fit(fit_values, fit_actions)
    acc = _bin_acc(edges, maj, eval_values, eval_actions)
    base = float(np.mean(eval_actions == gmaj))
    return {"acc": acc, "majority_baseline": base, "elevation": acc - base}


def _world_vec(obs_dict: Dict[str, Any]) -> Optional[np.ndarray]:
    w = obs_dict.get("world_state") if isinstance(obs_dict, dict) else None
    if w is None:
        return None
    w = w.detach().cpu().numpy() if torch.is_tensor(w) else np.asarray(w)
    return w.reshape(-1).astype(np.float64)


class MatchedArbitraryTarget:
    """Callable `obs_dict -> Optional[float]` for `run_zworld_p0(target_fn=...)`.

    Usage:
        tgt = MatchedArbitraryTarget(seed)
        report = tgt.fit(calibration_obs)      # list of obs_dicts from a RANDOM-policy rollout
        run_zworld_p0(..., config=cfg, target_fn=tgt)
    `report` carries the matching diagnostics the manifest must record.
    """

    name = "matched_arbitrary_auxiliary"

    def __init__(self, seed: int,
                 exclude_entities: Sequence[int] = (RESOURCE_ENTITY_INDEX, HAZARD_ENTITY_INDEX),
                 n_candidates: int = N_CANDIDATE_DIRECTIONS,
                 feature_mode: str = "counts") -> None:
        """feature_mode:
          "counts"    -- per-entity COUNTS over the 25 view cells for the kept entity slots
                         (cell-PERMUTATION-INVARIANT, hence direction-blind by construction:
                         the oracle's action is a direction, and no symmetric function of the
                         cells can point). The default since the V3-EXQ-1017 red-team: a per-cell
                         functional of wall/empty geometry predicted the oracle's action 5-17
                         points above chance on the training rollout (walls block moves), the
                         proximity scalar ~0 -- the "matched" control leaked the DV's own label.
          "cells"     -- the per-cell one-hot functional (kept for the record; leaks direction).
        """
        self.seed = int(seed)
        self.exclude_entities = tuple(int(e) for e in exclude_entities)
        self.n_candidates = int(n_candidates)
        self.feature_mode = str(feature_mode)
        self._idx = np.asarray(_kept_indices(self.exclude_entities), dtype=np.int64)
        self._kept_entities = [e for e in range(LOCAL_VIEW_ENTITY_STRIDE) if e not in self.exclude_entities]
        n_feat = len(self._kept_entities) if self.feature_mode == "counts" else len(self._idx)
        g = torch.Generator().manual_seed(20260909 + self.seed)
        cands = torch.randn(self.n_candidates, n_feat, generator=g)
        cands = cands / cands.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self._candidates = cands.numpy().astype(np.float64)
        self._direction: Optional[np.ndarray] = None
        self._raw_grid: Optional[np.ndarray] = None
        self._prox_grid: Optional[np.ndarray] = None
        self.report: Dict[str, Any] = {"fitted": False}

    # -- the raw functional -------------------------------------------------------------
    def _features(self, W: np.ndarray) -> np.ndarray:
        """[n, n_feat] features of world_obs rows W: per-entity counts (default) or kept cells."""
        if W.ndim == 1:
            W = W[None, :]
        if self.feature_mode == "counts":
            view = W[:, :LOCAL_VIEW_CELLS * LOCAL_VIEW_ENTITY_STRIDE].reshape(
                W.shape[0], LOCAL_VIEW_CELLS, LOCAL_VIEW_ENTITY_STRIDE)
            return view[:, :, self._kept_entities].sum(axis=1)
        return W[:, self._idx]

    def _raw(self, w: np.ndarray, direction: np.ndarray) -> float:
        return float(np.dot(self._features(w)[0], direction))

    # -- fit ------------------------------------------------------------------------------
    def fit(self, calibration_obs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ws, prox, eps, acts = [], [], [], []
        n_prox_none = 0
        for i, o in enumerate(calibration_obs):
            w = _world_vec(o)
            if w is None:
                continue
            p = resource_prox_target(o)
            if p is None or not np.isfinite(p):
                n_prox_none += 1
                continue
            ws.append(w)
            prox.append(float(p))
            eps.append(int(o.get("_calib_episode", -1)) if isinstance(o, dict) else -1)
            acts.append(int(o.get("_calib_oracle_action", -1)) if isinstance(o, dict) else -1)
        n = len(ws)
        if n < 16:
            raise ValueError(
                "MatchedArbitraryTarget.fit needs >= 16 calibration steps with a finite "
                "proximity target, got %d (%d had none)" % (n, n_prox_none))
        W = np.stack(ws)
        P = np.asarray(prox, dtype=np.float64)
        E = np.asarray(eps, dtype=np.int64)
        # Fit on EVEN calibration episodes, evaluate matching on ODD ones: the report must not be
        # certified on the data the map was fitted to, and consecutive steps are correlated, so the
        # split is by EPISODE (not by step, not by first/second half -- a first/second-half split
        # confounds the check with episode-order drift). Falls back to halves if no episode tag.
        if np.unique(E).size >= 2:
            fit_mask = (E % 2 == 0)
        else:
            fit_mask = np.arange(n) < (n // 2)
        if int(fit_mask.sum()) < 8 or int((~fit_mask).sum()) < 8:
            fit_mask = np.arange(n) < (n // 2)
        W_fit, P_fit, W_eval, P_eval = W[fit_mask], P[fit_mask], W[~fit_mask], P[~fit_mask]
        X_fit, X_eval = self._features(W_fit), self._features(W_eval)

        raws_fit = X_fit @ self._candidates.T      # [n_fit, K]
        rs = np.array([abs(_pearson(raws_fit[:, k], P_fit)) for k in range(self.n_candidates)])
        # A candidate with zero variance on the calibration distribution carries no learnable
        # structure; refuse it even if its |r| is trivially 0.
        var_ok = raws_fit.std(axis=0) > 1e-9
        rs = np.where(var_ok, rs, np.inf)
        if not np.isfinite(rs).any():
            raise ValueError("MatchedArbitraryTarget.fit: no candidate direction has variance "
                             "on the calibration distribution")
        k = int(np.argmin(rs))
        d = self._candidates[k].copy()
        # Ridge fit of the proximity target on the kept channels (the best LINEAR prox predictor
        # from what the functional can see), then remove d's component along it in the sense of
        # the regression slope of d.x on w_hat.x. The result is still a pure linear functional of x.
        Xc = X_fit - X_fit.mean(axis=0, keepdims=True)
        Pc = P_fit - P_fit.mean()
        A = Xc.T @ Xc + RIDGE * float(Xc.shape[0]) * np.eye(Xc.shape[1])
        w_hat = np.linalg.solve(A, Xc.T @ Pc)
        q_fit = Xc @ w_hat
        r_raw_prox_fit = _pearson(raws_fit[:, k], P_fit)
        vq = float(np.dot(q_fit, q_fit))
        beta = float(np.dot(Xc @ d, q_fit) / vq) if vq > 1e-12 else 0.0
        d_adj = d - beta * w_hat
        if float(np.std(X_fit @ d_adj)) <= 1e-9:
            d_adj = d        # degenerate adjustment: keep the selected direction, report it
            beta = 0.0
        self._direction = d_adj
        raw_fit = X_fit @ self._direction
        # Quantile map: empirical CDF of raw on the fit half -> empirical quantile function of prox.
        self._raw_grid = np.sort(raw_fit)
        self._prox_grid = np.sort(P_fit)

        mapped_eval = np.array([self._map(self._raw(w, self._direction)) for w in W_eval])
        raw_eval = X_eval @ self._direction
        mapped_fit = np.array([self._map(r) for r in raw_fit])
        # Kept for certify(): the fit-half values + oracle actions the bin majorities are fitted on.
        A = np.asarray(acts, dtype=np.int64)
        self._fit_mapped, self._fit_prox = mapped_fit, P_fit
        self._fit_actions = A[fit_mask]
        have_actions = bool((A >= 0).all()) and A.size > 0
        act_m = (action_decodability(mapped_fit, A[fit_mask], mapped_eval, A[~fit_mask])
                 if have_actions else None)
        act_p = (action_decodability(P_fit, A[fit_mask], P_eval, A[~fit_mask])
                 if have_actions else None)
        self.report = {
            "action_decodability_matched_eval": act_m,
            "action_decodability_prox_eval": act_p,
            "fitted": True,
            "name": self.name,
            "n_calibration_steps": int(n),
            "n_fit": int(fit_mask.sum()),
            "n_eval": int((~fit_mask).sum()),
            "split": "episode_parity" if np.unique(E).size >= 2 else "halves",
            "residualised_against_linear_prox_predictor": bool(beta != 0.0),
            "residualisation_beta": float(beta),
            "pearson_r_selected_raw_with_prox_fit": float(r_raw_prox_fit),
            "n_prox_none_dropped": int(n_prox_none),
            "n_candidate_directions": int(self.n_candidates),
            "chosen_candidate": k,
            "candidate_abs_r_with_prox_fit": [float(x) if np.isfinite(x) else None for x in rs],
            "feature_mode": self.feature_mode,
            "n_features": int(X_fit.shape[1]),
            "kept_entity_slots": list(self._kept_entities),
            "excluded_entity_slots": list(self.exclude_entities),
            # the two GOV-MATCHAUX-1 matching witnesses, on the HELD-OUT calibration half
            "ks_distance_mapped_vs_prox_eval": ks_distance(mapped_eval, P_eval),
            "pearson_r_with_prox_eval": _pearson(mapped_eval, P_eval),
            "pearson_r_raw_with_prox_eval": _pearson(raw_eval, P_eval),
            "mapped_mean_eval": float(np.mean(mapped_eval)),
            "mapped_std_eval": float(np.std(mapped_eval)),
            "prox_mean_eval": float(np.mean(P_eval)),
            "prox_std_eval": float(np.std(P_eval)),
            "prox_unique_values_eval": int(np.unique(np.round(P_eval, 6)).size),
        }
        return dict(self.report)

    def certify(self, rollout_obs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """The matching witnesses on a DIFFERENT rollout -- pass the observations the P0a warmup
        will actually train on (same env seed + same RandomPolicy seed reproduce the identical
        random walk), so the gates certify the training distribution rather than the calibration
        draw the direction was selected against (V3-EXQ-1017 red-team finding 7)."""
        if self._direction is None:
            raise RuntimeError("MatchedArbitraryTarget.certify before fit()")
        mapped, prox, acts = [], [], []
        for o in rollout_obs:
            t = self(o)
            p = resource_prox_target(o)
            if t is None or p is None or not np.isfinite(p):
                continue
            mapped.append(float(t))
            prox.append(float(p))
            acts.append(int(o.get("_calib_oracle_action", -1)) if isinstance(o, dict) else -1)
        M, P, A = (np.asarray(mapped, dtype=np.float64), np.asarray(prox, dtype=np.float64),
                   np.asarray(acts, dtype=np.int64))
        out: Dict[str, Any] = {
            "n_rollout_steps": int(M.size),
            "ks_distance_mapped_vs_prox": ks_distance(M, P) if M.size else float("nan"),
            "pearson_r_with_prox": _pearson(M, P) if M.size else float("nan"),
            "mapped_mean": float(M.mean()) if M.size else None,
            "mapped_std": float(M.std()) if M.size else None,
            "prox_mean": float(P.mean()) if P.size else None,
            "prox_std": float(P.std()) if P.size else None,
        }
        if A.size and bool((A >= 0).all()) and getattr(self, "_fit_actions", None) is not None \
                and self._fit_actions.size and bool((self._fit_actions >= 0).all()):
            out["action_decodability_matched"] = action_decodability(
                self._fit_mapped, self._fit_actions, M, A)
            out["action_decodability_prox"] = action_decodability(
                self._fit_prox, self._fit_actions, P, A)
        else:
            out["action_decodability_matched"] = None
            out["action_decodability_prox"] = None
        return out

    def _map(self, raw: float) -> float:
        assert self._raw_grid is not None and self._prox_grid is not None
        # empirical CDF position of raw among the fit-half raws, then that quantile of prox
        u = np.searchsorted(self._raw_grid, raw, side="right") / self._raw_grid.size
        u = min(max(u, 0.0), 1.0)
        j = min(int(u * (self._prox_grid.size - 1) + 0.5), self._prox_grid.size - 1)
        return float(self._prox_grid[j])

    # -- the callable the trainer sees ------------------------------------------------------
    def __call__(self, obs_dict: Dict[str, Any]) -> Optional[float]:
        if self._direction is None:
            raise RuntimeError("MatchedArbitraryTarget used before fit()")
        w = _world_vec(obs_dict)
        if w is None or w.size <= int(self._idx.max()):
            return None
        return self._map(self._raw(w, self._direction))


def collect_calibration_obs(env: Any, policy: Any, episodes: int, steps: int,
                            label_policy: Any = None) -> List[Dict[str, Any]]:
    """Roll `policy` on `env` and return the per-step obs_dicts (cloned), for `fit()`/`certify()`.

    Pass a DEDICATED env instance (the rollout consumes env RNG) and a policy independent of the
    agent -- the calibration must sample the same state distribution the warmup rollout will, which
    for the SD-070 recipe is a random walk (`capability_eval.RandomPolicy`). `label_policy`
    (e.g. `LocalViewGreedyPolicy`) is queried at every step WITHOUT acting, to record the DV's
    label (`_calib_oracle_action`) for the action-decodability leak witness; it reads env state
    only and owns its own RNG, so it leaves the rollout untouched.
    """
    out: List[Dict[str, Any]] = []
    for _ep in range(int(episodes)):
        _flat, obs = env.reset()
        if hasattr(policy, "reset"):
            policy.reset(env)
        if label_policy is not None and hasattr(label_policy, "reset"):
            label_policy.reset(env)
        for _t in range(int(steps)):
            rec = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
            rec["_calib_episode"] = int(_ep)
            if label_policy is not None:
                rec["_calib_oracle_action"] = int(label_policy.act(env, obs))
            out.append(rec)
            a = policy.act(env, obs)
            with torch.no_grad():
                _f, _h, done, _i, obs = env.step(a)
            if done:
                break
    return out
