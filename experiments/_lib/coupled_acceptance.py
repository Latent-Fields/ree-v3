"""Coupled loop-repair campaign acceptance instruments (plan node I1; instrument-only).

Plan of record: REE_assembly/evidence/planning/coupled_loop_repair_campaign_plan.md
sections 3 (I1) and 5 (A1 pre-registration). Every campaign member gate and every A1
precondition reads one of these. They are ports of the committed probes under
REE_assembly/evidence/planning/probes/{rollout,babble,evaluation}/ and of the Worker J
CEM trace (probes/decoderprobe/cem_trace_diag.py). This module changes no behaviour: it
reads agents/envs, never trains them, and (where it wraps a live method to record calls)
restores the original in a ``finally``.

INSTRUMENTS (numbering follows plan section 3, I1):
  1. action_discrimination     swap-first-action executed-closest disc_h over the 4-class
                               (0..3) and 5-class sets + fidelity k, on a held-out
                               uniform-random test set (babble_probe.evaluate /
                               rollout_fidelity_probe.m2_fidelity).
  2. cloned-env probe states   true next z per action class through sense()'s own encoder,
                               side-effect-free, validated against the next tick's sensed
                               z_world; env-Q (6 random 4-step continuations)
                               (balanced_replay_probe / partitioned_repair_probe_r2).
  3. E3 decomposition          variance share from steps > d; truncation + deep-shuffle flip
                               rates; pick-in-best + Spearman(J_pred, J_true) at FULL /
                               depth-1 / any chosen aggregation; head-swap pick-flip rate.
  4. codec trace               per-CEM-iteration O-norm and decoded norm, iteration-0 range
                               ratio, round-trip accuracy, pool Q-best coverage, proposal m4.
  5. stratum classifier        hazard-trapped vs benign from the NATIVE arm alone; the
                               sidecar is written before any other arm may be read.
  6. outcome decomposition     by env transition_type: true contacts, proximity steps,
                               consumptions, approach steps, each separately.

NEGATIVE-INSTRUMENT CONTRACT (REE_Working CLAUDE.md, "Negative instruments"). Each
instrument returns a dict carrying ``verdict``: one of
  CANNOT_DETERMINE  the derivation was empty or degenerate (no starts, a static ground
                    truth, an unseen class, an unknown transition_type, a failed encoder
                    validation, ...). ``reason`` says which. NEVER read as PASS.
  PASS / FAIL       only when the caller passes a bar (``bar=...``) and the value is
                    determinable.
  MEASURED          determinable, no bar given.
Each instrument also has a pinned CANARY (``canary_*`` below): a known-baseline finding
that must keep reproducing. tests/contracts/test_coupled_acceptance_instruments.py pins
them and shows each test FAILS against a deliberately broken instrument.

DEVIATIONS FROM THE PROBES (deliberate, stated):
  - disc_h credit is TIE-FAIR: a start scores 1/|argmin set| when the executed class is in
    the argmin set (tolerance 1e-9 relative), 0 otherwise. For a continuous predictor this
    equals the probes' argmin credit; for an action-INDEPENDENT predictor it equals chance
    exactly, where the probes' bare argmin credited class 0 (index order) on every tie.
  - The 5-class disc is reported at every requested h, not only h=1.

ASCII-only output (nothing here prints).
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PASS = "PASS"
FAIL = "FAIL"
MEASURED = "MEASURED"
CANNOT_DETERMINE = "CANNOT_DETERMINE"
VERDICTS = (PASS, FAIL, MEASURED, CANNOT_DETERMINE)

CLASSES_4 = (0, 1, 2, 3)   # the Phase-0 generator's action space (argmax % 4)
_TIE_TOL = 1e-9


def _cd(reason: str, **extra: Any) -> Dict[str, Any]:
    out = {"verdict": CANNOT_DETERMINE, "reason": reason}
    out.update(extra)
    return out


def _judge(value: Optional[float], bar: Optional[float], higher_is_better: bool = True) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return CANNOT_DETERMINE
    if bar is None:
        return MEASURED
    ok = value >= bar if higher_is_better else value <= bar
    return PASS if ok else FAIL


def _mean(v: Iterable[Optional[float]]) -> Optional[float]:
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Rank correlation (ordinal ranks, as the probes). None when either side is constant."""
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _tie_fair_hit(errs: Sequence[float], target: int) -> Tuple[float, bool]:
    """(credit, was_tie). credit = 1/|argmin set| if target in it, else 0."""
    e = np.asarray(errs, dtype=float)
    m = float(e.min())
    tol = _TIE_TOL * max(1.0, abs(m))
    best = np.nonzero(e <= m + tol)[0]
    tie = len(best) > 1
    return ((1.0 / len(best)) if target in set(best.tolist()) else 0.0), tie


def onehot(c: int, action_dim: int) -> torch.Tensor:
    v = torch.zeros(1, action_dim)
    v[0, int(c)] = 1.0
    return v


# =============================================================== 1. action discrimination
Predictor = Callable[[torch.Tensor, torch.Tensor], Sequence[torch.Tensor]]


def e2_world_predictor(e2: Any, z_self: Optional[torch.Tensor] = None) -> Predictor:
    """Adapter: E2's native open-loop world rollout as a ``predict(x0, acts)`` callable.

    Returns world_states (index 0 = x0, index h = prediction after h actions). z_self is
    held at zeros (the babble probe's convention) unless given.
    """
    self_dim = int(getattr(e2.config, "self_dim", 32))

    @torch.no_grad()
    def predict(x0: torch.Tensor, acts: torch.Tensor) -> Sequence[torch.Tensor]:
        zs = z_self if z_self is not None else torch.zeros(1, self_dim)
        return e2.rollout_with_world(zs, x0, acts, compute_action_objects=False).world_states

    return predict


@torch.no_grad()
def action_discrimination(
    predict: Predictor,
    episodes: Sequence[Dict[str, torch.Tensor]],
    *,
    action_dim: int,
    horizons: Sequence[int] = (1, 3, 5),
    fidelity_horizon: int = 10,
    classes: Sequence[int] = CLASSES_4,
    max_starts: int = 300,
    seed: int = 0,
    bar_disc4_h1: Optional[float] = None,
) -> Dict[str, Any]:
    """Swap-first-action executed-closest discrimination + fidelity k.

    episodes: held-out, uniform-random-action test episodes, each
      {"z": Tensor[T+1, D] encoded states, "a": LongTensor[T] executed classes}.
    For every start t (with t + fidelity_horizon <= T), the executed action sequence is
    rolled out; for h in ``horizons`` the FIRST action is swapped for each class and the
    executed class must give the prediction closest to the actual z_{t+h}.
      disc4_h{h}: over ``classes`` (chance 1/len(classes)); disc5_h{h}: over all
      action_dim classes (chance 1/action_dim).
      k: deepest h (contiguous from 1) at which median rollout error beats persistence.
    CANNOT_DETERMINE when: no starts; a non-finite prediction; the ground truth is static
    (median persistence 0 at h=1, so no action can be discriminated); or a class in
    ``classes`` is never executed among the starts (the test set is not uniform).
    """
    H = int(max(max(horizons), fidelity_horizon))
    g = np.random.default_rng(seed + 77)
    starts: List[Tuple[int, int]] = []
    for i, e in enumerate(episodes):
        z, a = e["z"], e["a"]
        if z.shape[0] != a.shape[0] + 1:
            return _cd("episode %d: len(z)=%d must equal len(a)+1=%d" % (i, z.shape[0], a.shape[0] + 1))
        starts.extend((i, t) for t in range(a.shape[0] - H + 1))
    if not starts:
        return _cd("no start with a full %d-step window in the test set" % H, n_starts=0)
    if len(starts) > max_starts:
        starts = [starts[j] for j in sorted(g.choice(len(starts), max_starts, replace=False))]
    cls = [int(c) for c in classes]
    rows = {h: {"err": [], "pers": []} for h in range(1, H + 1)}
    d4 = {h: [] for h in horizons}
    d5 = {h: [] for h in horizons}
    ties4 = 0
    exec_counts: Counter = Counter()
    for i, t in starts:
        x, a = episodes[i]["z"], episodes[i]["a"]
        acts = F.one_hot(a[t:t + H].long(), action_dim).float().unsqueeze(0)
        x0 = x[t:t + 1]
        ws = predict(x0, acts)
        for h in range(1, H + 1):
            p = ws[h]
            if not bool(torch.isfinite(p).all()):
                return _cd("non-finite prediction at h=%d" % h)
            y = x[t + h:t + h + 1]
            rows[h]["err"].append(float((p - y).norm()))
            rows[h]["pers"].append(float((x0 - y).norm()))
        ex = int(a[t])
        exec_counts[ex] += 1
        for h in horizons:
            y = x[t + h:t + h + 1]
            errs = []
            for c in range(action_dim):
                a2 = acts[:, :h].clone()
                a2[0, 0] = 0.0
                a2[0, 0, c] = 1.0
                errs.append(float((predict(x0, a2)[h] - y).norm()))
            credit5, _t5 = _tie_fair_hit(errs, ex)
            d5[h].append(credit5)
            if ex in cls:
                credit4, t4 = _tie_fair_hit([errs[c] for c in cls], cls.index(ex))
                d4[h].append(credit4)
                ties4 += int(t4 and h == horizons[0])
    k = 0
    eop = {}
    for h in range(1, H + 1):
        me, mp = float(np.median(rows[h]["err"])), float(np.median(rows[h]["pers"]))
        eop[h] = me / mp if mp > 0 else None
        if mp > 0 and me < mp and k == h - 1:
            k = h
    out: Dict[str, Any] = {
        "n_starts": len(starts), "k": k, "chance4": 1.0 / len(cls), "chance5": 1.0 / action_dim,
        "executed_class_counts": {str(c): exec_counts.get(c, 0) for c in range(action_dim)},
        "tie_frac_first_h": ties4 / max(1, len(d4[horizons[0]])),
        "err_over_pers": {str(h): eop[h] for h in (1, H)},
    }
    for h in horizons:
        out["disc4_h%d" % h] = float(np.mean(d4[h])) if d4[h] else None
        out["disc5_h%d" % h] = float(np.mean(d5[h])) if d5[h] else None
    if float(np.median(rows[1]["pers"])) <= 0.0:
        out.update(_cd("static ground truth: median persistence error 0 at h=1"))
        return out
    missing = [c for c in cls if exec_counts.get(c, 0) == 0]
    if missing:
        out.update(_cd("classes never executed among starts: %s" % missing))
        return out
    out["verdict"] = _judge(out.get("disc4_h1"), bar_disc4_h1)
    return out


@torch.no_grad()
def sense_latent(agent: Any, obs: Dict[str, Any]) -> Any:
    """The native read path (agent.sense) on an env obs dict. MUTATES the agent's latent."""
    return agent.sense(obs["body_state"], obs["world_state"], obs_harm=obs.get("harm_obs"),
                       obs_harm_a=obs.get("harm_obs_a"), obs_harm_history=obs.get("harm_history"))


@torch.no_grad()
def collect_uniform_random_episodes(agent: Any, env: Any, n_steps: int, seed: int,
                                    classes: Sequence[int] = CLASSES_4) -> List[Dict[str, torch.Tensor]]:
    """Held-out test set: uniform-random actions over ``classes``, z through agent.sense().

    Use a seed disjoint from any training data. The agent's latent state is reset at
    every episode boundary (babble_probe.encode_segs convention) and is left mutated.
    """
    g = np.random.default_rng(seed)
    eps: List[Dict[str, torch.Tensor]] = []
    _f, obs = env.reset()
    agent.reset()
    zs = [sense_latent(agent, obs).z_world.detach().reshape(-1).clone()]
    acts: List[int] = []
    for _ in range(int(n_steps)):
        a = int(classes[int(g.integers(0, len(classes)))])
        _f, _h, done, _i, obs = env.step(a)
        acts.append(a)
        zs.append(sense_latent(agent, obs).z_world.detach().reshape(-1).clone())
        if done:
            eps.append({"z": torch.stack(zs), "a": torch.tensor(acts)})
            _f, obs = env.reset()
            agent.reset()
            zs = [sense_latent(agent, obs).z_world.detach().reshape(-1).clone()]
            acts = []
    if acts:
        eps.append({"z": torch.stack(zs), "a": torch.tensor(acts)})
    return eps


# =============================================================== 2. cloned-env probe states
def env_q_values(env: Any, action_dim: int, rng: np.random.Generator, n_cont: int = 6,
                 cont_len: int = 4) -> np.ndarray:
    """ADDENDUM 3 outcome-3 estimator: Q(c) = mean over n_cont deep-copied env rollouts of
    the summed env reward of [step c, then cont_len uniform-random steps]."""
    q = []
    for c in range(action_dim):
        tot = 0.0
        for _ in range(n_cont):
            e = copy.deepcopy(env)
            _f, h, done, _i, _o = e.step(c)
            s = float(h)
            for _k in range(cont_len):
                if done:
                    break
                _f, h, done, _i, _o = e.step(int(rng.integers(0, action_dim)))
                s += float(h)
            tot += s
        q.append(tot / n_cont)
    return np.asarray(q)


@torch.no_grad()
def encode_next_side_effect_free(agent: Any, obs: Dict[str, Any], prev_latent: Any,
                                 prev_action: torch.Tensor) -> torch.Tensor:
    """sense()'s encoder call (agent.py sense -> latent_stack.encode) without mutating the
    agent. Must track agent.sense; ``probe_state_validation`` detects drift (max |diff|)."""
    dev = next(agent.parameters()).device
    ob = torch.as_tensor(obs["body_state"]).float().reshape(1, -1).to(dev)
    ow = torch.as_tensor(obs["world_state"]).float().reshape(1, -1).to(dev)
    enc = torch.cat([agent.body_obs_encoder(ob), agent.world_obs_encoder(ow)], dim=-1)
    vol = agent.e3.volatility_estimate if agent.config.latent.volatility_signal_dim > 0 else None
    harm = obs.get("harm_obs")
    if agent.lpb_router is not None and harm is not None:
        harm = agent.lpb_router.mask_external_harm_obs(harm)
    anchor = agent._e1_predicted_next_z_self if getattr(agent.config.latent, "use_self_recurrence", False) else None
    lat = agent.latent_stack.encode(enc, prev_latent, prev_action=prev_action, harm_obs=harm,
                                    harm_obs_a=obs.get("harm_obs_a"), harm_history=obs.get("harm_history"),
                                    volatility_signal=vol, self_e1_anchor=anchor)
    return lat.z_world.detach().clone()


def collect_probe_states(agent: Any, env: Any, steps: int, every: int, seed: int, *,
                         action_dim: Optional[int] = None, with_q: bool = True, n_cont: int = 6,
                         cont_len: int = 4, with_pool: bool = True) -> List[Dict[str, Any]]:
    """Native waking run (StepHarness, train_mode=False). Every ``every`` steps, at the
    on_action hook (after select, before env.step): for each action class c, deep-copy the
    env, step c, and encode the TRUE next z_world side-effect-free; optionally the env-Q
    vector and a native CEM pool (torch RNG saved/restored). The next tick's sensed
    z_world is recorded against the executed class for ``probe_state_validation``.
    NOTE: the validation consumes the following tick, so probe states are >= 2 apart."""
    from experiments._harness import StepHarness, StepHooks

    A = int(action_dim if action_dim is not None else agent.e2.config.action_dim)
    g = np.random.default_rng(seed + 31)
    states: List[Dict[str, Any]] = []
    pend: Dict[str, int] = {}

    def on_action(agent, latent, action, obs_dict, ticks, step, **_k):
        if step % every != 0:
            return
        lat_s = agent._current_latent
        z0, s0 = latent.z_world.detach().clone(), latent.z_self.detach().clone()
        zt = []
        for c in range(A):
            e = copy.deepcopy(env)
            zt.append(encode_next_side_effect_free(agent, e.step(c)[4], lat_s, onehot(c, A)))
        a_ex = action.detach().reshape(1, -1).float()
        e = copy.deepcopy(env)
        st: Dict[str, Any] = {"z0": z0, "s0": s0, "z_true": zt,
                              "executed": int(a_ex.argmax()),
                              "exec_is_onehot": bool(float(a_ex.max()) > 0.99 and float(a_ex.abs().sum()) < 1.01),
                              "z_exact_exec": encode_next_side_effect_free(agent, e.step(action)[4], lat_s, a_ex)}
        if with_q:
            st["q"] = env_q_values(env, A, g, n_cont, cont_len)
        if with_pool:
            rs = torch.get_rng_state()
            with torch.no_grad():
                pool = agent.hippocampal.propose_trajectories(z0, z_self=s0)
            torch.set_rng_state(rs)
            st["pool_actions"] = [p.actions.detach().clone() for p in pool]
        pend["i"] = len(states)
        states.append(st)

    h = StepHarness(agent, env, train_mode=False, seed=seed, hooks=StepHooks(on_action=on_action))
    _f, obs = env.reset(); agent.reset(); h.reset()
    n = 0
    while n < steps:
        pend.pop("i", None)
        r = h.step(obs); n += 1
        i = pend.get("i")
        obs = r.next_obs_dict
        if i is not None and not r.done and n < steps:
            r2 = h.step(obs); n += 1
            st = states[i]
            st["validation_exact_maxabs"] = float((r2.latent.z_world - st["z_exact_exec"]).abs().max())
            if st["exec_is_onehot"]:
                st["validation_maxabs"] = float((r2.latent.z_world - st["z_true"][st["executed"]]).abs().max())
            obs = r2.next_obs_dict
            r = r2
        if r.done:
            _f, obs = env.reset(); agent.reset(); h.reset()
    return states


def probe_state_validation(states: Sequence[Dict[str, Any]], tol: float = 1e-5) -> Dict[str, Any]:
    """Canary for instrument 2: the side-effect-free encode of a deep-copied env step must
    reproduce the next tick's sensed z_world (ADDENDUM 2: max |diff| 0.0). Checked on the
    EXACT executed action (always available; the native action may be a continuous decoded
    vector) and, where the executed action was one-hot, on its class's z_true entry.
    FAIL means encode_next_side_effect_free has drifted from agent.sense (or the env clone
    is not faithful) and every z_true is suspect. CANNOT_DETERMINE: nothing validated."""
    ve = [s["validation_exact_maxabs"] for s in states if "validation_exact_maxabs" in s]
    vc = [s["validation_maxabs"] for s in states if "validation_maxabs" in s]
    if not ve:
        return _cd("no validated probe state (no probe state was followed by a next tick)", n_states=len(states))
    mx = float(max(ve + vc))
    return {"verdict": PASS if mx <= tol else FAIL, "n_validated_exact": len(ve), "n_validated_class": len(vc),
            "max_abs_diff": mx, "tol": tol}


# =============================================================== 3. E3 decomposition
ScoreFn = Callable[[Sequence[Any], Optional[int]], np.ndarray]


def e3_score_fn(agent: Any) -> ScoreFn:
    """Adapter: E3's own score_trajectory, optionally truncated to the first ``n_steps``
    world steps via SD-081's live knob (_score_depth_limit = n_steps + 1). Restores it."""
    e3 = agent.e3

    @torch.no_grad()
    def score(trajs: Sequence[Any], n_steps: Optional[int] = None) -> np.ndarray:
        prev = e3._score_depth_limit
        e3._score_depth_limit = None if n_steps is None else int(n_steps) + 1
        try:
            return np.asarray([float(e3.score_trajectory(t).mean()) for t in trajs])
        finally:
            e3._score_depth_limit = prev

    return score


def _with_world_states(traj: Any, ws: List[torch.Tensor]) -> Any:
    t2 = copy.copy(traj)
    t2.world_states = ws
    return t2


def e3_depth_structure(score: ScoreFn, pools: Sequence[Sequence[Any]], *, depths: Sequence[int] = (1, 3, 5),
                       deep_var_depth: int = 5, shuffle_after: int = 3, seed: int = 0) -> Dict[str, Any]:
    """Where E3's cross-candidate choice lives along the rollout.
      trunc_flip_rate[d]  P(argmin J changes when only the first d world steps are scored)
      deep_var_share_p50  median share of FULL-J cross-candidate variance NOT reproduced by
                          the first ``deep_var_depth`` steps (variance from steps > d)
      shuffle_flip_rate   P(argmin changes when world steps > ``shuffle_after`` are replaced
                          by a deranged other candidate's steps)
    CANNOT_DETERMINE when no pool has >= 2 candidates or FULL J is constant in every pool."""
    g = np.random.default_rng(seed)
    flips: Dict[int, List[bool]] = {d: [] for d in depths}
    deep: List[float] = []
    shuf: List[bool] = []
    n_const = 0
    for trajs in pools:
        C = len(trajs)
        if C < 2:
            continue
        full = score(trajs, None)
        if float(np.ptp(full)) == 0.0:
            n_const += 1
            continue
        af = int(np.argmin(full))
        for d in depths:
            flips[d].append(int(np.argmin(score(trajs, d))) != af)
        tr = score(trajs, deep_var_depth)
        vf = float(full.var())
        deep.append(float((full - tr).var() / vf))
        perm = np.roll(np.arange(C), 1 + int(g.integers(0, C - 1)))
        sh = []
        for i in range(C):
            ws = list(trajs[i].world_states)
            dws = trajs[int(perm[i])].world_states
            for j in range(shuffle_after + 1, len(ws)):
                ws[j] = dws[j]
            sh.append(_with_world_states(trajs[i], ws))
        shuf.append(int(np.argmin(score(sh, None))) != af)
    if not shuf:
        return _cd("no pool with >= 2 candidates and non-constant FULL J", n_constant_pools=n_const)
    return {"verdict": MEASURED, "n_pools": len(shuf), "n_constant_pools": n_const,
            "trunc_flip_rate": {str(d): float(np.mean(v)) for d, v in flips.items()},
            "deep_var_share_p50": float(np.median(deep)), "deep_var_depth": deep_var_depth,
            "shuffle_flip_rate": float(np.mean(shuf)), "shuffle_after": shuffle_after}


def e3_choice_quality(pred_by_arm: Dict[str, Sequence[np.ndarray]], truth: Sequence[np.ndarray], *,
                      truth_lower_is_better: bool = True, bar_arm: Optional[str] = None,
                      bar_margin_over_chance: Optional[float] = None) -> Dict[str, Any]:
    """Pick-in-best and Spearman(J_pred, J_true) per scoring arm (e.g. FULL, DEPTH1, chosen).

    pred_by_arm[arm][s]: E3's J per action class at state s (argmin = E3's pick).
    truth[s]: the grounded value per class -- E3's own score on the TRUE 1-step trajectory
      (lower better) or the env-Q vector (pass truth_lower_is_better=False).
    Only states whose truth is non-constant count. Chance = mean(|best set| / A).
    CANNOT_DETERMINE when no state is informative. With bar_arm and
    bar_margin_over_chance the verdict is PASS iff that arm's pick_in_best exceeds chance
    by more than the margin."""
    t_all = [np.asarray(t, dtype=float) * (1.0 if truth_lower_is_better else -1.0) for t in truth]
    inf_idx = [i for i, t in enumerate(t_all) if float(np.ptp(t)) > 1e-9 * max(1.0, float(np.abs(t).max()))]
    if not inf_idx:
        return _cd("no state with non-constant truth", n_states=len(truth))
    chance = []
    best_sets = {}
    for i in inf_idx:
        t = t_all[i]
        best = set(np.nonzero(t <= t.min() + 1e-9 * max(1.0, abs(t.min())))[0].tolist())
        best_sets[i] = best
        chance.append(len(best) / len(t))
    out: Dict[str, Any] = {"n_states": len(truth), "n_informative": len(inf_idx), "chance": float(np.mean(chance)),
                           "arms": {}}
    for arm, preds in pred_by_arm.items():
        hits, rhos, picks = [], [], Counter()
        for i in inf_idx:
            p = np.asarray(preds[i], dtype=float)
            pk = int(np.argmin(p))
            picks[pk] += 1
            hits.append(pk in best_sets[i])
            rhos.append(spearman(p, t_all[i]))
        out["arms"][arm] = {"pick_in_best": float(np.mean(hits)), "spearman_mean": _mean(rhos),
                            "pick_counts": {str(k): v for k, v in sorted(picks.items())}}
    if bar_arm is not None and bar_margin_over_chance is not None:
        v = out["arms"][bar_arm]["pick_in_best"] - out["chance"]
        out["margin_over_chance"] = v
        out["verdict"] = PASS if v > bar_margin_over_chance else FAIL
    else:
        out["verdict"] = MEASURED
    return out


def head_swap_flip_rate(scores_a: Sequence[np.ndarray], scores_b: Sequence[np.ndarray]) -> Dict[str, Any]:
    """P(E3's pick changes when head A's rollouts are replaced by head B's) over the same
    candidate sets. W4 gate (c): the shuffled head must move the pick LESS often than the
    real head (compare two calls). CANNOT_DETERMINE when no paired state has >= 2 candidates."""
    pairs = [(np.asarray(a), np.asarray(b)) for a, b in zip(scores_a, scores_b) if len(a) >= 2 and len(a) == len(b)]
    if not pairs:
        return _cd("no paired state with >= 2 candidates")
    flips = [int(np.argmin(a)) != int(np.argmin(b)) for a, b in pairs]
    return {"verdict": MEASURED, "n_states": len(pairs), "flip_rate": float(np.mean(flips))}


@torch.no_grad()
def scaffold_scores(agent: Any, states: Sequence[Dict[str, Any]], *, depths: Sequence[Optional[int]] = (None, 1),
                    action_dim: Optional[int] = None) -> Dict[str, Any]:
    """Per probe state: one candidate per first-action class c (native pool[0]'s action
    sequence with step 0 replaced by one-hot c), rolled out by the live E2 head, scored by E3
    at each depth (None = FULL). Also J_true = E3 on the TRUE 1-step trajectory
    [z0, z_true(c)]. Returns {"pred": {depth_key: [array per state]}, "j_true": [...], "q": [...]}."""
    from ree_core.predictors.e2_fast import Trajectory

    A = int(action_dim if action_dim is not None else agent.e2.config.action_dim)
    score = e3_score_fn(agent)
    pred: Dict[str, List[np.ndarray]] = {("FULL" if d is None else "DEPTH%d" % d): [] for d in depths}
    j_true, q = [], []
    for st in states:
        base = st["pool_actions"][0].clone()
        trajs, true_trajs = [], []
        for c in range(A):
            a = base.clone(); a[:, 0, :] = 0.0; a[:, 0, c] = 1.0
            trajs.append(agent.e2.rollout_with_world(st["s0"], st["z0"], a, compute_action_objects=False))
            true_trajs.append(Trajectory(states=[st["s0"], st["s0"]], actions=onehot(c, A).unsqueeze(1),
                                         world_states=[st["z0"], st["z_true"][c]]))
        for d in depths:
            pred["FULL" if d is None else "DEPTH%d" % d].append(score(trajs, d))
        j_true.append(score(true_trajs, None))
        if "q" in st:
            q.append(np.asarray(st["q"]))
    return {"pred": pred, "j_true": j_true, "q": q}


# =============================================================== 4. codec trace
def proposal_m4(first_classes_by_state: Sequence[Sequence[Optional[int]]]) -> Dict[str, Any]:
    """Proposal state-dependence (rollout probe M4): per state, the pool's first-action class
    distribution; across states, how often the majority class is the modal one.
    frac_states_with_modal_majority = 1.0 means the proposal is state-INVARIANT."""
    rows = []
    for cls in first_classes_by_state:
        cls = [c for c in cls if c is not None]
        if not cls:
            continue
        cc = Counter(cls)
        maj, nmaj = cc.most_common(1)[0]
        p = np.asarray(list(cc.values()), dtype=float) / len(cls)
        rows.append({"maj": int(maj), "maj_share": nmaj / len(cls), "n_classes": len(cc),
                     "entropy": float(-(p * np.log(p)).sum())})
    if len(rows) < 2:
        return _cd("fewer than 2 states with a non-empty pool", n_states=len(rows))
    majs = Counter(r["maj"] for r in rows)
    return {"verdict": MEASURED, "n_states": len(rows),
            "maj_share_mean": float(np.mean([r["maj_share"] for r in rows])),
            "n_classes_mean": float(np.mean([r["n_classes"] for r in rows])),
            "entropy_mean": float(np.mean([r["entropy"] for r in rows])),
            "majority_class_counts_across_states": {str(k): v for k, v in sorted(majs.items())},
            "frac_states_with_modal_majority": majs.most_common(1)[0][1] / len(rows)}


def pool_qbest_coverage(first_classes_by_state: Sequence[Sequence[Optional[int]]],
                        q_by_state: Sequence[np.ndarray]) -> Dict[str, Any]:
    """W1 gate (e) statistic: P(the proposal pool contains a candidate whose first action is
    in the env-Q-best set), over states where Q is not constant. Chance-free by itself --
    the gate compares real vs label-shuffled codec. CANNOT_DETERMINE if no informative state."""
    hits = []
    for cls, q in zip(first_classes_by_state, q_by_state):
        q = np.asarray(q, dtype=float)
        if float(np.ptp(q)) < 1e-9:
            continue
        best = set(np.nonzero(q >= q.max() - 1e-9)[0].tolist())
        hits.append(any(c in best for c in cls if c is not None))
    if not hits:
        return _cd("no state with non-constant env-Q", n_states=len(q_by_state))
    return {"verdict": MEASURED, "n_informative": len(hits), "coverage": float(np.mean(hits))}


@torch.no_grad()
def cem_codec_trace(agent: Any, latents: Sequence[Tuple[torch.Tensor, torch.Tensor]], *, seed: int) -> Dict[str, Any]:
    """Worker J's cem_trace_diag, as a function. Wraps the live
    HippocampalModule._decode_action_objects (restored in ``finally``) and records, per CEM
    iteration, the median sampled action-object (O) norm fed in, the median and max decoded
    norm (first step), and the first-step argmax class counts; plus each pool's first-action
    classes (candidate_first_action_class) for proposal_m4 / pool_qbest_coverage.
    latents: (z_world, z_self) per state. growth[i] = decoded median at iteration i+1 / i."""
    hip = agent.hippocampal
    iters = int(hip.config.num_cem_iterations)
    orig = hip._decode_action_objects
    rec: List[Tuple[float, float, int]] = []

    def wrapped(ao, _o=orig):
        y = _o(ao)
        rec.append((float(ao[:, 0, :].norm(dim=-1).median()), float(y[:, 0, :].norm(dim=-1).median()),
                    int(y[0, 0, :].argmax())))
        return y

    per_it = [{"in": [], "out": [], "cls": Counter()} for _ in range(iters)]
    pool_classes: List[List[Optional[int]]] = []
    uneven = 0
    had_instance_attr = "_decode_action_objects" in vars(hip)
    hip._decode_action_objects = wrapped
    try:
        for j, (zw, zs) in enumerate(latents):
            rec.clear()
            torch.manual_seed(seed * 1000 + j)
            pool = hip.propose_trajectories(zw, z_self=zs)
            if iters and len(rec) % iters:
                uneven += 1
            n = len(rec) // iters if iters else 0
            for it in range(iters):
                for (i_n, o_n, c) in rec[it * n:(it + 1) * n]:
                    per_it[it]["in"].append(i_n); per_it[it]["out"].append(o_n); per_it[it]["cls"][c] += 1
            pool_classes.append([hip.candidate_first_action_class(tr) for tr in pool])
    finally:
        if had_instance_attr:
            hip._decode_action_objects = orig
        else:
            del hip._decode_action_objects
    if iters == 0 or not per_it[0]["out"]:
        return _cd("no decode call recorded (CEM loop did not route through _decode_action_objects)",
                   num_cem_iterations=iters, pool_first_classes=pool_classes)
    rows = [{"iter": it, "ao_input_norm_median": float(np.median(p["in"])),
             "decoded_norm_median": float(np.median(p["out"])), "decoded_norm_max": float(np.max(p["out"])),
             "first_class_counts": {str(k): v for k, v in sorted(p["cls"].items())}}
            for it, p in enumerate(per_it) if p["out"]]
    growth = [rows[i + 1]["decoded_norm_median"] / rows[i]["decoded_norm_median"]
              for i in range(len(rows) - 1) if rows[i]["decoded_norm_median"] > 0]
    return {"verdict": MEASURED, "num_cem_iterations": iters, "n_states": len(latents),
            "states_with_uneven_call_count": uneven, "per_iteration": rows,
            "decoded_growth_per_iteration": growth,
            "decoded_growth_max": float(max(growth)) if growth else None,
            "pool_first_classes": pool_classes}


@torch.no_grad()
def encoder_image_norm(e2: Any, z_worlds: Sequence[torch.Tensor], action_dim: int) -> Optional[float]:
    """Median norm of E2's action-object image action_object(z, onehot(c)) over states x
    classes (the reference scale for W1 gate (d))."""
    norms = [float(e2.action_object(z.reshape(1, -1), onehot(c, action_dim)).norm())
             for z in z_worlds for c in range(action_dim)]
    return float(np.median(norms)) if norms else None


def codec_ranges(trace: Dict[str, Any], image_norm: Optional[float], onehot_norm: float = 1.0) -> Dict[str, Any]:
    """W1 gates (c) and (d) read off a cem_codec_trace:
      iter0_range_ratio   = median iteration-0 sampled O-norm / encoder-image median norm
                            (gate (d): within [0.5, 2]).
      decoded_ratio[i]    = median decoded norm at iteration i / one-hot norm
                            (gate (c): all within [0.5, 2], no iteration-on-iteration growth).
    CANNOT_DETERMINE when the trace is, or the image norm is missing/zero."""
    if trace.get("verdict") == CANNOT_DETERMINE:
        return _cd("trace CANNOT_DETERMINE: %s" % trace.get("reason"))
    if not image_norm or image_norm <= 0:
        return _cd("encoder image norm missing or zero")
    rows = trace["per_iteration"]
    r0 = rows[0]["ao_input_norm_median"] / image_norm
    dec = [r["decoded_norm_median"] / onehot_norm for r in rows]
    in_band = all(0.5 <= d <= 2.0 for d in dec)
    no_growth = all(g <= 1.0 + 1e-6 for g in trace["decoded_growth_per_iteration"])
    return {"verdict": PASS if (0.5 <= r0 <= 2.0 and in_band and no_growth) else FAIL,
            "iter0_range_ratio": r0, "gate_d_pass": bool(0.5 <= r0 <= 2.0),
            "decoded_ratio_by_iteration": dec, "gate_c_pass": bool(in_band and no_growth)}


@torch.no_grad()
def codec_roundtrip_accuracy(agent: Any, z_worlds: Sequence[torch.Tensor], *, action_dim: Optional[int] = None,
                             bar_per_class: Optional[float] = None) -> Dict[str, Any]:
    """W1 gate (b): per class c, P(argmax decoder(action_object(z, onehot c)) == c) over
    held-out z. This is the codec's OWN round trip, measured as an instrument -- it is NOT an
    action source (see HippocampalModule._decode_action_objects). PASS iff every class
    meets bar_per_class. CANNOT_DETERMINE with no states."""
    if not z_worlds:
        return _cd("no held-out z_world")
    A = int(action_dim if action_dim is not None else agent.e2.config.action_dim)
    dec = agent.hippocampal.action_object_decoder
    acc = {}
    for c in range(A):
        hits = [int(dec(agent.e2.action_object(z.reshape(1, -1), onehot(c, A))).argmax()) == c for z in z_worlds]
        acc[str(c)] = float(np.mean(hits))
    worst = min(acc.values())
    return {"verdict": _judge(worst, bar_per_class), "per_class": acc, "min_class": worst, "n_states": len(z_worlds)}


# =============================================================== 5. stratum classifier
class StratumOrderError(RuntimeError):
    """Raised when a non-NATIVE arm is read before its seed's NATIVE stratum sidecar exists,
    or a sidecar write would be informed by (or overwrite with) another arm's data."""


TRAPPED = "hazard_trapped"
BENIGN = "benign"


def classify_stratum(dones: Sequence[bool], *, window: int = 600, early_len: int = 200,
                     min_early: int = 10, max_episode_steps: Optional[int] = None) -> Dict[str, Any]:
    """A1 stratum from the NATIVE arm alone (plan sec 5): hazard-trapped iff >= ``min_early``
    episodes TERMINATE with length < ``early_len`` steps within NATIVE's first ``window``
    steps; benign otherwise. ``dones``: the per-step done flag of NATIVE's closed-loop steps,
    in order. An episode still running at the window edge is not counted.
    CANNOT_DETERMINE when fewer than ``window`` steps are given, or when the env's
    max_episode_steps < early_len (every episode would count as early)."""
    if max_episode_steps is not None and int(max_episode_steps) < early_len:
        return _cd("max_episode_steps %d < early_len %d: every episode is early" % (max_episode_steps, early_len))
    if len(dones) < window:
        return _cd("only %d NATIVE steps observed, need %d" % (len(dones), window), n_steps=len(dones))
    n_early, n_ep, length = 0, 0, 0
    lengths = []
    for d in list(dones)[:window]:
        length += 1
        if d:
            n_ep += 1
            lengths.append(length)
            if length < early_len:
                n_early += 1
            length = 0
    stratum = TRAPPED if n_early >= min_early else BENIGN
    return {"verdict": MEASURED, "stratum": stratum, "n_early_terminations": n_early, "n_episodes_ended": n_ep,
            "episode_lengths": lengths, "criteria": {"window": window, "early_len": early_len, "min_early": min_early}}


def write_stratum_sidecar(path: str, *, seed: int, result: Dict[str, Any], arm: str = "NATIVE",
                          arms_already_read: Sequence[str] = ()) -> Dict[str, Any]:
    """Write the per-seed stratum sidecar BEFORE any other arm is read. Refuses (raises
    StratumOrderError) if ``arm`` is not NATIVE, if any non-NATIVE arm was already read, if
    the result is CANNOT_DETERMINE, or if a sidecar with a DIFFERENT stratum exists
    (write-once; an identical rewrite is a no-op)."""
    if arm != "NATIVE":
        raise StratumOrderError("stratum must be classified from the NATIVE arm, got %r" % arm)
    foreign = [a for a in arms_already_read if a != "NATIVE"]
    if foreign:
        raise StratumOrderError("non-NATIVE arm(s) already read before classification: %s" % foreign)
    if result.get("verdict") == CANNOT_DETERMINE:
        raise StratumOrderError("refusing to write a CANNOT_DETERMINE stratum: %s" % result.get("reason"))
    rec = {"seed": int(seed), "stratum": result["stratum"], "source_arm": "NATIVE",
           "n_early_terminations": result["n_early_terminations"], "criteria": result["criteria"],
           "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        if old.get("stratum") != rec["stratum"] or int(old.get("seed", -1)) != rec["seed"]:
            raise StratumOrderError("sidecar %s already records seed %s stratum %s; refusing to overwrite with %s"
                                    % (path, old.get("seed"), old.get("stratum"), rec["stratum"]))
        return old
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    return rec


def require_stratum_sidecar(path: str, seed: int) -> str:
    """Call BEFORE reading any non-NATIVE arm of ``seed``. Returns the stratum or raises."""
    if not os.path.exists(path):
        raise StratumOrderError("no NATIVE stratum sidecar for seed %d at %s: classify NATIVE first" % (seed, path))
    with open(path) as fh:
        rec = json.load(fh)
    if int(rec.get("seed", -1)) != int(seed) or rec.get("stratum") not in (TRAPPED, BENIGN):
        raise StratumOrderError("sidecar %s does not carry a valid stratum for seed %d" % (path, seed))
    return rec["stratum"]


def admit_seeds(strata_in_screen_order: Sequence[Tuple[int, str]], *, n_per_stratum: int = 5,
                ceiling: int = 80) -> Dict[str, Any]:
    """Admit the first ``n_per_stratum`` seeds of each stratum, screening at most ``ceiling``
    seeds in order. A stratum with fewer than n admitted is CANNOT_DETERMINE (plan sec 5)."""
    adm: Dict[str, List[int]] = {BENIGN: [], TRAPPED: []}
    screened = 0
    for seed, stratum in strata_in_screen_order[:ceiling]:
        screened += 1
        if stratum in adm and len(adm[stratum]) < n_per_stratum:
            adm[stratum].append(int(seed))
        if all(len(v) >= n_per_stratum for v in adm.values()):
            break
    per = {s: (MEASURED if len(v) >= n_per_stratum else CANNOT_DETERMINE) for s, v in adm.items()}
    return {"admitted": adm, "n_screened": screened, "stratum_verdict": per,
            "verdict": MEASURED if all(v == MEASURED for v in per.values()) else CANNOT_DETERMINE}


# =============================================================== 6. outcome decomposition
TRUE_CONTACT_TYPES = ("agent_caused_hazard", "env_caused_hazard", "env_caused_multisource")
PROXIMITY_TYPES = ("hazard_approach", "harm_gradient")
CONSUMPTION_TYPES = ("resource",)
APPROACH_TYPES = ("benefit_approach",)
# Known, deliberately UNSCORED transition types. A type outside every tuple here makes the
# decomposition CANNOT_DETERMINE: an env that grew a new contact type must not have it
# silently counted as "other".
OTHER_KNOWN_TYPES = ("none", "action_blocked", "resource_contact", "waypoint", "sequence_complete",
                     "zone_c_ambient")
_CATEGORIES = (("true_contacts", TRUE_CONTACT_TYPES), ("proximity_steps", PROXIMITY_TYPES),
               ("consumptions", CONSUMPTION_TYPES), ("approach_steps", APPROACH_TYPES))


def outcome_decomposition(transition_types: Sequence[Optional[str]],
                          rewards: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Counts and per-100-step rates by category (true contacts / proximity / consumptions /
    approach steps), each separately, plus reward per 100 when ``rewards`` is given.
    CANNOT_DETERMINE when there are no steps, any step lacks a transition_type (the env did
    not surface it), or an unknown type appears (listed in ``unknown_types``)."""
    n = len(transition_types)
    if n == 0:
        return _cd("no steps")
    missing = sum(1 for t in transition_types if t is None)
    counts = Counter(t for t in transition_types if t is not None)
    known = set(OTHER_KNOWN_TYPES)
    for _name, types in _CATEGORIES:
        known.update(types)
    unknown = sorted(t for t in counts if t not in known)
    out: Dict[str, Any] = {"n_steps": n, "type_counts": dict(sorted(counts.items()))}
    for name, types in _CATEGORIES:
        c = int(sum(counts.get(t, 0) for t in types))
        out[name] = c
        out[name + "_per_100"] = 100.0 * c / n
        out[name + "_by_type"] = {t: int(counts.get(t, 0)) for t in types}
    if rewards is not None:
        if len(rewards) != n:
            return _cd("len(rewards)=%d != n_steps=%d" % (len(rewards), n))
        out["reward_per_100"] = 100.0 * float(np.sum(rewards)) / n
    if missing:
        out.update(_cd("%d step(s) carry no transition_type" % missing))
        return out
    if unknown:
        out.update(_cd("unknown transition_type(s)", unknown_types=unknown))
        return out
    out["verdict"] = MEASURED
    return out


def step_outcome(info: Dict[str, Any]) -> Optional[str]:
    """transition_type from an env.step info dict / StepResult.info (None if absent)."""
    t = info.get("transition_type") if isinstance(info, dict) else None
    return None if t is None else str(t)


# =============================================================== canaries (pinned findings)
def _toy_world(seed: int, n_eps: int = 12, T: int = 40, D: int = 6, action_dim: int = 5, noise: float = 0.05,
               world_seed: int = 0):
    """Toy linear world: z_{t+1} = z_t + delta[a] + noise; uniform actions over CLASSES_4.
    ``world_seed`` fixes the dynamics (delta); ``seed`` the sampled episodes."""
    delta = torch.randn(action_dim, D, generator=torch.Generator().manual_seed(world_seed))
    g = torch.Generator().manual_seed(seed + 1)
    eps = []
    for _ in range(n_eps):
        a = torch.randint(0, 4, (T,), generator=g)
        z = [torch.randn(D, generator=g)]
        for t in range(T):
            z.append(z[-1] + delta[a[t]] + noise * torch.randn(D, generator=g))
        eps.append({"z": torch.stack(z), "a": a})
    return eps, delta


def _fit_delta_head(eps, action_dim: int, shuffle_seed: Optional[int] = None) -> torch.Tensor:
    """Least-squares per-class mean displacement; labels permuted if shuffle_seed is given."""
    dz = torch.cat([e["z"][1:] - e["z"][:-1] for e in eps])
    a = torch.cat([e["a"] for e in eps])
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        a = a[torch.randperm(a.shape[0], generator=g)]
    W = torch.zeros(action_dim, dz.shape[1])
    for c in range(action_dim):
        m = a == c
        W[c] = dz[m].mean(0) if bool(m.any()) else dz.mean(0)
    return W


def _delta_predictor(W: torch.Tensor) -> Predictor:
    def predict(x0, acts):
        out = [x0]
        for h in range(acts.shape[1]):
            out.append(out[-1] + acts[:, h] @ W)
        return out
    return predict


def canary_action_discrimination(seed: int = 0) -> Dict[str, Any]:
    """Pinned: on a toy world with a real per-action effect, a head fit on TRUE labels scores
    disc4_h1 ~ 1 and k >= 3; the same head fit on LABEL-SHUFFLED transitions scores disc at
    chance (mean over 16 shuffles, 0.25 +/- 0.12), and an action-blind head scores exactly chance by tie-fair credit."""
    train, _ = _toy_world(seed, world_seed=seed)
    test, _ = _toy_world(seed + 1000, world_seed=seed)
    A = 5
    real = action_discrimination(_delta_predictor(_fit_delta_head(train, A)), test, action_dim=A, seed=seed)
    # one shuffled head maps each executed class to a fixed nearest row (a random function),
    # so a SINGLE shuffle scores 0.25 x (its number of fixed points); average 16 shuffles.
    shuf_runs = [action_discrimination(_delta_predictor(_fit_delta_head(train, A, shuffle_seed=seed * 100 + j)),
                                       test, action_dim=A, seed=seed) for j in range(16)]
    shuf = {"disc4_h1": float(np.mean([r["disc4_h1"] for r in shuf_runs])), "n_shuffles": len(shuf_runs)}
    blind = action_discrimination(_delta_predictor(torch.zeros(A, test[0]["z"].shape[1])), test, action_dim=A,
                                  seed=seed)
    ok = (real["disc4_h1"] >= 0.95 and real["k"] >= 3 and abs(shuf["disc4_h1"] - 0.25) <= 0.12
          and abs(blind["disc4_h1"] - 0.25) < 1e-9)
    return {"reproduced": bool(ok), "real": real, "shuffled": shuf, "blind": blind}


def canary_stratum() -> Dict[str, Any]:
    """Pinned: 12 deaths at 40 steps then survival -> trapped; 3 full 200-step episodes -> benign;
    9 early deaths -> benign (threshold is >= 10); 599 steps -> CANNOT_DETERMINE."""
    def ep(n, done=True):
        return [False] * (n - 1) + [done]
    trapped = classify_stratum(ep(40) * 12 + [False] * 200)
    benign = classify_stratum(ep(200) * 3 + [False] * 10)
    nine = classify_stratum(ep(40) * 9 + [False] * 300)
    short = classify_stratum([False] * 599)
    ok = (trapped.get("stratum") == TRAPPED and benign.get("stratum") == BENIGN and nine.get("stratum") == BENIGN
          and short["verdict"] == CANNOT_DETERMINE)
    return {"reproduced": bool(ok), "trapped": trapped["stratum"], "benign": benign["stratum"],
            "nine_early": nine["stratum"], "short": short["verdict"]}


def canary_outcome_decomposition() -> Dict[str, Any]:
    """Pinned: a stream of 2 true contacts, 3 proximity, 1 consumption, 4 approach in 100 steps
    decomposes to exactly those counts (proximity NEVER counted as contact); an unknown type
    is CANNOT_DETERMINE."""
    tt = (["agent_caused_hazard", "env_caused_hazard"] + ["hazard_approach"] * 2 + ["harm_gradient"]
          + ["resource"] + ["benefit_approach"] * 4 + ["none"] * 90)
    r = outcome_decomposition(tt)
    bad = outcome_decomposition(tt[:-1] + ["new_contact_type"])
    ok = (r["verdict"] == MEASURED and r["true_contacts"] == 2 and r["proximity_steps"] == 3
          and r["consumptions"] == 1 and r["approach_steps"] == 4 and bad["verdict"] == CANNOT_DETERMINE)
    return {"reproduced": bool(ok), "result": r, "unknown_verdict": bad["verdict"]}


def canary_e3_structure(seed: int = 0) -> Dict[str, Any]:
    """Pinned: with J = sum of world-step values and all candidates identical on the first
    step, depth-1 truncation flips the pick (flip rate 1.0 vs deep choice) and the deep
    variance share is ~1; with the value only on step 1, truncation never flips (0.0) and
    the deep share is 0."""
    class _T:
        def __init__(self, ws):
            self.world_states = ws

    def mk(vals_by_cand):
        return [_T([torch.zeros(1, 1)] + [torch.full((1, 1), float(v)) for v in vals]) for vals in vals_by_cand]

    def score(trajs, n_steps=None):
        return np.asarray([float(sum(w.item() for w in t.world_states[1:(None if n_steps is None else n_steps + 1)]))
                           for t in trajs])
    g = np.random.default_rng(seed)
    deep_pools, shallow_pools = [], []
    for _ in range(10):
        dv = g.normal(size=4)
        # step-1 identical, choice lives at steps 2..6; the depth-1 argmin (all ties -> index 0) differs
        # from the FULL argmin whenever FULL's best is not candidate 0
        order = np.argsort(dv)
        dv = dv[np.r_[order[1:], order[0]]] if order[0] == 0 else dv
        deep_pools.append(mk([[0.0] + [float(v)] * 5 for v in dv]))
        sv = g.normal(size=4)
        shallow_pools.append(mk([[float(v)] + [0.0] * 5 for v in sv]))
    deep = e3_depth_structure(score, deep_pools, depths=(1,), deep_var_depth=1, seed=seed)
    shallow = e3_depth_structure(score, shallow_pools, depths=(1,), deep_var_depth=1, seed=seed)
    ok = (deep["trunc_flip_rate"]["1"] == 1.0 and deep["deep_var_share_p50"] > 0.99
          and shallow["trunc_flip_rate"]["1"] == 0.0 and shallow["deep_var_share_p50"] < 1e-9)
    return {"reproduced": bool(ok), "deep": deep, "shallow": shallow}


def canary_codec_trace(agent: Any, latents: Sequence[Tuple[torch.Tensor, torch.Tensor]], *, seed: int = 1,
                       gain: float = 8.0) -> Dict[str, Any]:
    """Pinned on an UNTRAINED agent (regime B scale, world_dim 32). Two signatures must
    reproduce (Worker J / plan W1 '(2) and (3)'):
      CONTRACTION (native untrained decoder): the iteration-0 sampled O-norm is far above the
        encoder image (documented ~12x; pinned > 5x) and the decoded norm does not grow
        (first iteration-on-iteration ratio < 1, none > 1.1).
      DIVERGENCE (the same decoder with its Linear weights scaled by ``gain``, a stand-in for
        a trained high-gain decoder): the decoded norm grows every iteration (all ratios > 1.2).
    The decoder is restored afterwards."""
    dec = agent.hippocampal.action_object_decoder
    z_worlds = [zw for zw, _zs in latents]
    img = encoder_image_norm(agent.e2, z_worlds, int(agent.e2.config.action_dim))
    native = cem_codec_trace(agent, latents, seed=seed)
    rng = codec_ranges(native, img)
    saved = {k: v.clone() for k, v in dec.state_dict().items()}
    try:
        with torch.no_grad():
            for m in dec.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.mul_(gain)
        boosted = cem_codec_trace(agent, latents, seed=seed)
    finally:
        dec.load_state_dict(saved)
    g_n = native.get("decoded_growth_per_iteration") or []
    g_b = boosted.get("decoded_growth_per_iteration") or []
    contraction = bool(rng.get("iter0_range_ratio", 0.0) > 5.0 and g_n and g_n[0] < 1.0 and max(g_n) <= 1.1)
    divergence = bool(g_b and min(g_b) > 1.2)
    return {"reproduced": contraction and divergence, "contraction": contraction, "divergence": divergence,
            "iter0_range_ratio": rng.get("iter0_range_ratio"), "native_growth": g_n, "boosted_growth": g_b,
            "image_norm": img}
