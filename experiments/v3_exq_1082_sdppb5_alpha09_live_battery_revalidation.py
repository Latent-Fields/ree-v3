"""V3-EXQ-1082 -- SD-PP-B5 re-validation at alpha_world 0.9 on a LIVE-agent battery.
Does the interventional margin loss raise e2.world_forward's action read, when z_world
is not EMA-damped and the evaluation battery is collected from an agent that is alive?

PURPOSE: diagnostic (validates_substrate SD-PP-B5-z-world-per-step-displacement-range,
re-validation). Non-contributory to governance confidence by design; claim_ids EMPTY,
bears_on names the claims.

SLEEP DRIVER: not applicable (no sleep machinery; P0 world_forward training only).

RED-TEAM (Step 4.5, fable): CONTESTED -> fixed. F1 FAIL-a split so an inert margin and a
margin that LOWERS the action read get different labels; F2 A3' guard removed (with mse_init
shared across arms it collapsed to MSE(ON)/MSE(OFF) <= 1.0), A4 is the only gate, 1075's A3
recorded; F3 rung scale vs per-step displacement recorded per cell (M001=0.01 already sits
below the ~0.031 L2 step at alpha 0.9; ladder kept as 1075's).

WHY THIS RUN EXISTS
-------------------
SD-PP-B5's governance-ratified validated-NEGATIVE verdict was QUALIFIED (not reversed)
on 2026-09-23 on the confirmed failure_autopsy_V3-EXQ-1079_2026-09-23. It rests on
V3-EXQ-1073/1075, which are confounded on TWO independent axes:
  (i)  both inherited alpha_world 0.3 from REEConfig.from_dims
       (ree_core/utils/config.py:7562; EMA at ree_core/latent/stack.py:1584) against
       SD-008's stable floor of >= 0.9;
  (ii) the 1075/1079 battery collector never reads `done` (env.step's 3rd return), so
       with a 30-step burn-in every battery row was POST-DEATH (health_depleted at
       step 32/27/22; |world_state| grows without bound thereafter).
The autopsy routes: "re-validate the SD-PP-B5 question at alpha_world >= 0.9 on a
battery that resets on done. A NEW question, not a lettered re-run of 1075."

THE TRAP THIS DESIGN AVOIDS (chip-20260923-sdppb5-revalidate-alpha09): V3-EXQ-1079
does NOT show alpha 0.9 makes the head better. model_r2 FALLS 0.88->0.72 while
persistence_r2 collapses 0.93->0.55 -- the persistence-relative 'readiness' flip is the
BASELINE collapsing. This run therefore reports model_r2 next to every
persistence-relative number and never routes on persistence skill.

DESIGN
------
alpha_world = 0.9 EXPLICITLY at from_dims, asserted threaded, recorded per cell.
5 arms x 3 seeds = 15 cells -- V3-EXQ-1075's arm set, the ONLY between-arm difference
being the interventional margin term in the P0 objective:
  ARM_OFF      use_world_interventional=False
  ARM_ON_M001  margin 0.01
  ARM_ON_M005  margin 0.05
  ARM_ON_M010  margin 0.10 (config default)
  ARM_ON_M050  margin 0.50
Env, dims, budget (3600 P0 steps), optimiser, batch sampling and seeds are 1075's. Two
rng streams (1075): `rng` drives rollout + batch sampling identically in every arm;
`coin` draws the interventional-fraction decision and is consumed only by ON arms.

DONE-HANDLING (pre-flight AMBER change 2): ported from V3-EXQ-1073
(_sample_probe_battery, v3_exq_1073...py:1054-1056) into BOTH the P0 rollout and the
battery collector: unpack `done` from env.step, and on done -> env.reset(), prev=None,
so no transition spanning a death/reset is ever recorded. The battery collector
additionally skips the first POST_RESET_SKIP transitions after every reset (EMA
residual 0.1^3 = 1e-3 at alpha 0.9) and records per-row health; the precondition
`battery_rows_all_live` requires zero post-death rows.

THE ROUTED DV -- d_act, ANALYTIC null (chip requirement 3):
  d_act = (SSE_swap - SSE_true) / (SSE_swap + SSE_true)
on a LIVE battery of 512 transitions, SSE_swap being the per-row error averaged over the
a_dim-1 alternative actions (errors averaged, not predictions -> no Jensen term), so an
action-blind head scores EXACTLY 0. The battery is collected by a FRESH auxiliary agent
built with the OFF config at the same seed and alpha, with its own rng, so it is
bit-identical across the 5 arms of a seed (asserted by battery hash) and the ON-vs-OFF
contrast is a PAIRED row bootstrap. The encoder is not trained (the optimiser holds
only agent.e2 parameters) and the margin flag adds no parameters, so the auxiliary
agent's encoder equals every arm's encoder (asserted, precondition).

PRE-REGISTERED ROUTES (both directions, CLAUDE.md)
--------------------------------------------------
Per seed s and ON rung r: lift = d_act(ON_r) - d_act(OFF), paired bootstrap CI.
  grows      lift CI lower > 0
  excluded   lift CI upper < DELTA_EQUIV (0.05): a lift of practical size is ruled out
             (DELTA_EQUIV ~10x the untrained-head floor 0.002-0.006 seen in 1079)
  reads      d_act(ON_r) CI lower > 0
  shrinks    lift CI upper < 0
  guard      reconstruction intact: A4 (1075, verbatim) battery MSE(ON_r)/MSE(OFF) <=
             2.0 on the same rows. 1075's A3 (conv_rel_drop >= 0.99) is RECORDED only:
             mse_init is shared across a seed's arms, so any OFF-relative A3 reduces to an
             MSE ratio cap, and the absolute 0.99 (calibrated at alpha 0.3) is failed by
             OFF itself at 0.9 on some seeds.
A seed is VALID for routing only if its rows are aligned across all 5 arms and its
positive control is live (ridge action-aware predictor on the OFF cell's own P0 buffer
scores d_act CI lower > 0 on the same battery).

  PASS    margin_raises_action_read -- some rung r has grows AND reads AND guard on
          >= 2 valid seeds. Label split (interpretation only): OFF reads its action at
          0.9 on >= 2 seeds -> margin_amplifies_present_action_read, else
          margin_makes_blind_head_read_action.
  FAIL-b  margin_read_bought_by_reconstruction -- no PASS rung, but some rung has
          grows AND reads with guard VIOLATED on >= 2 valid seeds (V3-EXQ-701b
          direction on the margin form).
  FAIL-a  margin_does_not_raise_action_read -- THE MANIPULATION GENUINELY DOES NOT
          WORK: every rung is `excluded` on >= 2 valid seeds (preconditions met,
          instrument live). Label split (interpretation only, counts unchanged): some
          rung `shrinks` on >= 2 valid seeds -> margin_lowers_action_read (the loss
          actively degrades the read), else margin_does_not_raise_action_read (inert). The SD-PP-B5 negative then stands at SD-008's operating
          point on a live agent. Secondary field off_arm_premise records whether the
          OFF head already reads its action at 0.9 (off_reads_action -> the margin
          loss is unneeded rather than ineffective; off_blind -> ineffective and the
          blind-head premise holds).
  UNDETERMINED margin_effect_undetermined -- none of the above (CIs too wide / mixed).
          Not evidence either way.
  substrate_not_ready_requeue -- any precondition unmet.
outcome PASS iff PASS route. model_r2, persistence_r2, persistence_relative_skill and
skill_vs_identity are RECORDED per cell and never gate (MECH-573 readability remains
deferred per rec-20260923-cb59ede6). The untrained-head d_act is recorded per seed as
the empirical floor of the analytic null.

DV-SYMMETRY (Step 3.5): d_act is a normalised difference of two SSEs on a frozen,
arm-invariant battery; the manipulation is a training-time loss term changing only the
head's parameters. No symmetry of the DV absorbs it: a shared additive shift of the
predictions changes SSE_true and SSE_swap differently unless the head ignores its
action, which is exactly the null the statistic scores as 0.

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): no recorded manifest carries a margin-ON
head at alpha_world 0.9 (1075 is 0.3-only; 1079 is OFF-only) and every recorded battery
in this lineage is post-death. Not recoverable -> run.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome
from experiments._lib.action_sensitivity_gate import identity_predictor_mse
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.persistence_skill_gate import (
    BOOTSTRAP_SEED, CI_LEVEL, N_BOOTSTRAP, per_row_squared_error, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1082_sdppb5_alpha09_live_battery_revalidation"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1082"
# DELIBERATELY EMPTY, as in V3-EXQ-1075/1079: a substrate re-validation. The claims it
# bears on reach governance through the SD-PP-B5 substrate_queue entry, not scoring.
CLAIM_IDS: List[str] = []
BEARS_ON = ["MECH-573", "MECH-574", "SD-008"]
VALIDATES_SUBSTRATE = "SD-PP-B5-z-world-per-step-displacement-range"
AUTOPSY = "failure_autopsy_V3-EXQ-1079_2026-09-23"
# Counting anchors (rows, distinct actions, alignment, live rows) are reachable by
# construction under fixed-length uniform-random collection with reset-on-done; the
# positive-control anchor uses the SAME _d_act_ci code that computes the routed DV.
ANCHOR_REACHABILITY_EXEMPT = (
    "counting anchors reachable by construction (random-action fixed-length live "
    "batteries with reset-on-done); positive-control anchor uses the shipped _d_act_ci "
    "predicate itself")
# Load-bearing bars are seed COUNTS over three-valued bootstrap CIs; DV room to move is
# certified per seed by the positive-control precondition.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "load-bearing bars are seed counts over three-valued bootstrap CIs; DV range "
    "certified per seed by the positive-control precondition")

# ---- operating point: COPIED from V3-EXQ-1075 (itself from 1073) ------------------
SEEDS = [42, 123, 456]
SELF_DIM = 16
WORLD_DIM = 16
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
P0_STEPS = 3600
STEPS_PER_EPISODE = 90
EPISODES_PER_RUN = P0_STEPS // STEPS_PER_EPISODE      # 40 -- the [train] denominator
BATCH_K = 8
MIN_BUF_BEFORE_TRAIN = 16
BUF_CAP = 4096
LR = 1e-3
MAX_GRAD_NORM = 1.0
INTERVENTIONAL_FRACTION = 0.3
ARMS: List[Tuple[str, Optional[float]]] = [
    ("ARM_OFF", None),
    ("ARM_ON_M001", 0.01),
    ("ARM_ON_M005", 0.05),
    ("ARM_ON_M010", 0.10),
    ("ARM_ON_M050", 0.50),
]

# ---- the two pre-flight changes -------------------------------------------------
ALPHA_WORLD = 0.9          # SD-008 stable floor; EXPLICIT, never inherited
LIVE_BATTERY_N = 512
POST_RESET_SKIP = 3        # transitions skipped after each reset (EMA residual 1e-3)
BATTERY_RNG_XOR = 0xB477E2

# ---- PRE-REGISTERED thresholds (constants, never derived from this run) ----------
DELTA_EQUIV = 0.05         # lift CI upper below this -> practical lift excluded
CONV_REL_DROP_BAR = 0.99   # A3 (1075), RECORDED only (see docstring)
MSE_INFLATION_BAR = 2.0    # A4 (1075), verbatim
SEEDS_REQUIRED = 2         # of 3
MIN_BATTERY_ROWS = 32      # persistence_skill_gate.MIN_ROWS
MIN_DISTINCT_ACTIONS = 2
RIDGE_REL_LAMBDA = 1e-3

_ZG = ZGoalStreamAccumulator()


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
                             num_resources=N_RESOURCES, use_proxy_fields=True)


def _config_slice(margin: Optional[float]) -> Dict[str, Any]:
    """Declares ONLY what the cell's computation reads."""
    return {
        "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES, "use_proxy_fields": True},
        "dims": {"self_dim": SELF_DIM, "world_dim": WORLD_DIM},
        "latent": {"alpha_world": float(ALPHA_WORLD)},
        "schedule": {"p0_steps": P0_STEPS, "batch_k": BATCH_K, "lr": LR,
                     "min_buf": MIN_BUF_BEFORE_TRAIN, "max_grad_norm": MAX_GRAD_NORM,
                     "buf_cap": BUF_CAP, "reset_on_done": True,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "battery": {"n_live": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                    "rng_xor": BATTERY_RNG_XOR},
        "positive_control": {"ridge_rel_lambda": RIDGE_REL_LAMBDA},
        "margin": {"use_world_interventional": margin is not None,
                   "world_interventional_margin": margin,
                   "world_interventional_fraction":
                       INTERVENTIONAL_FRACTION if margin is not None else None},
    }


def _build(seed: int, margin: Optional[float]) -> Tuple[CausalGridWorldV2, REEAgent]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    kw: Dict[str, Any] = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        alpha_world=float(ALPHA_WORLD))
    if margin is not None:
        kw.update(use_world_interventional=True,
                  world_interventional_margin=float(margin),
                  world_interventional_fraction=INTERVENTIONAL_FRACTION)
    cfg = REEConfig.from_dims(**kw)
    # MECH-307 guard: from_dims silently swallows unknown kwargs. Assert they landed.
    if float(getattr(cfg.latent, "alpha_world", -1.0)) != float(ALPHA_WORLD):
        raise RuntimeError("from_dims did not thread alpha_world")
    if margin is None:
        if bool(getattr(cfg.e2, "use_world_interventional", False)):
            raise RuntimeError("ARM_OFF must not carry the interventional margin")
    else:
        if not getattr(cfg.e2, "use_world_interventional", False):
            raise RuntimeError("from_dims did not thread use_world_interventional")
        if float(getattr(cfg.e2, "world_interventional_margin", -1.0)) != float(margin):
            raise RuntimeError("from_dims did not thread world_interventional_margin")
    return env, REEAgent(cfg)


def _sense_zworld(agent: REEAgent, obs: Dict[str, Any]) -> torch.Tensor:
    obs_harm = obs.get("harm_obs", None)
    return agent.sense(_to_b(obs["body_state"], agent.device),
                       _to_b(obs["world_state"], agent.device),
                       obs_harm=(_to_b(obs_harm, agent.device)
                                 if obs_harm is not None else None)
                       ).z_world.detach().reshape(-1).clone()


def _reset(env: CausalGridWorldV2) -> Dict[str, Any]:
    obs = env.reset()
    return obs[-1] if isinstance(obs, tuple) else obs


def _onehot(idx: int, a_dim: int) -> torch.Tensor:
    a = torch.zeros(a_dim, dtype=torch.float32)
    a[idx] = 1.0
    return a


def _collect_live_battery(agent: REEAgent, env: CausalGridWorldV2, rng: random.Random,
                          n: int) -> Dict[str, Any]:
    """LIVE battery. done-handling ported from V3-EXQ-1073 (:1054-1056): unpack done,
    on done -> env.reset() and prev=None, so the dying transition is never recorded.
    Also skips the first POST_RESET_SKIP transitions of every episode."""
    obs = _reset(env)
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    health: List[Tuple[float, float]] = []
    ws_norm: List[float] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None
    since_reset = 0
    n_resets = 0
    causes: Dict[str, int] = {}
    guard = 0
    while len(rows) < n and guard < n * 20:
        guard += 1
        z = _sense_zworld(agent, obs)
        h_now = float(env.agent_health)
        if (prev is not None and since_reset > POST_RESET_SKIP
                and bool(torch.isfinite(z).all())):
            rows.append((prev[0], prev[1], z))
            health.append((prev[2], h_now))
            ws_norm.append(float(torch.as_tensor(obs["world_state"]).float().norm()))
        idx = rng.randrange(env.action_dim)
        a = _onehot(idx, env.action_dim)
        _, _, done, info, obs = env.step(a.unsqueeze(0).to(agent.device))
        prev = (z, a, h_now)
        since_reset += 1
        if done:
            c = str((info or {}).get("done_cause", ""))
            causes[c] = causes.get(c, 0) + 1
            obs = _reset(env)
            prev = None
            since_reset = 0
            n_resets += 1
    z0 = torch.stack([r[0] for r in rows])
    acts = torch.stack([r[1] for r in rows])
    z1 = torch.stack([r[2] for r in rows])
    n_dead = sum(1 for h0, h1 in health if h0 <= 0.0 or h1 <= 0.0)
    return {"z0": z0, "acts": acts, "z1": z1, "n_resets": n_resets, "causes": causes,
            "n_postdeath_rows": n_dead, "min_health": min(min(h) for h in health),
            "max_world_state_norm": max(ws_norm), "mean_world_state_norm":
            sum(ws_norm) / len(ws_norm)}


def _draw_cf(acts: torch.Tensor) -> torch.Tensor:
    """a_cf from the COMPLEMENT of a_actual, per row (V3-EXQ-1075 verbatim)."""
    a_dim = acts.shape[-1]
    idx = acts.argmax(-1)
    offset = torch.randint(1, a_dim, idx.shape) if a_dim > 1 else torch.zeros_like(idx)
    return F.one_hot((idx + offset) % a_dim, a_dim).float()


def _battery_mse(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                 z1: torch.Tensor) -> float:
    with torch.no_grad():
        return float(((head(z0, acts) - z1) ** 2).mean().item())


def _swap_errors(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row squared error at the TRUE action, and the per-row MEAN squared error
    over every OTHER action (V3-EXQ-1079 verbatim). Same rows, same targets."""
    a_dim = int(acts.shape[-1])
    idx = acts.argmax(-1)
    with torch.no_grad():
        e_true = per_row_squared_error(head(z0, acts), z1)
        alts = [per_row_squared_error(
                    head(z0, F.one_hot((idx + k) % a_dim, a_dim).float()), z1)
                for k in range(1, a_dim)]
        e_swap = torch.stack(alts).mean(dim=0)
    return e_true.double(), e_swap.double()


def _d_from(e_true: torch.Tensor, e_swap: torch.Tensor) -> Optional[float]:
    den = float(e_swap.sum() + e_true.sum())
    return (float(e_swap.sum() - e_true.sum()) / den) if den > 0 else None


def _pct(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _classify(lo: Optional[float], hi: Optional[float], pos: str, neg: str) -> str:
    if lo is None or hi is None:
        return "cannot_determine"
    if lo > 0.0:
        return pos
    if hi < 0.0:
        return neg
    return "cannot_determine"


def _d_act_ci(e_true: torch.Tensor, e_swap: torch.Tensor, seed_offset: int
              ) -> Dict[str, Any]:
    """Paired row bootstrap (V3-EXQ-1079 verbatim). Analytic null 0."""
    n = int(e_true.shape[0])
    d = _d_from(e_true, e_swap)
    out: Dict[str, Any] = {"d_act": d, "ci_low": None, "ci_high": None,
                           "n_rows": n, "n_bootstrap": 0}
    if d is None or n < MIN_BATTERY_ROWS:
        out["status"] = "cannot_determine"
        return out
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    idx = torch.randint(0, n, (N_BOOTSTRAP, n), generator=gen)
    bt, bs = e_true[idx].sum(1), e_swap[idx].sum(1)
    den = bs + bt
    ok = den > 0
    vals = sorted(((bs - bt)[ok] / den[ok]).tolist())
    if len(vals) < N_BOOTSTRAP // 2:
        out["status"] = "cannot_determine"
        return out
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"], "reads_action",
                              "swap_better")
    return out


def _contrast_ci(et_hi: torch.Tensor, es_hi: torch.Tensor, et_lo: torch.Tensor,
                 es_lo: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """CI on d_act(ON) - d_act(OFF); rows identical across arms, so ROW INDICES are
    resampled JOINTLY (V3-EXQ-1079 construction, labels grows/shrinks)."""
    n = int(et_hi.shape[0])
    d_hi, d_lo = _d_from(et_hi, es_hi), _d_from(et_lo, es_lo)
    out: Dict[str, Any] = {"delta": None if (d_hi is None or d_lo is None)
                           else d_hi - d_lo, "ci_low": None, "ci_high": None,
                           "n_rows": n, "n_bootstrap": 0}
    if out["delta"] is None or n < MIN_BATTERY_ROWS or int(et_lo.shape[0]) != n:
        out["status"] = "cannot_determine"
        return out
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    idx = torch.randint(0, n, (N_BOOTSTRAP, n), generator=gen)
    th, sh, tl, sl = et_hi[idx].sum(1), es_hi[idx].sum(1), et_lo[idx].sum(1), es_lo[idx].sum(1)
    dh, dl = sh + th, sl + tl
    ok = (dh > 0) & (dl > 0)
    vals = sorted((((sh - th) / dh) - ((sl - tl) / dl))[ok].tolist())
    if len(vals) < N_BOOTSTRAP // 2:
        out["status"] = "cannot_determine"
        return out
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"], "grows", "shrinks")
    return out


def _ridge_head(buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Any:
    """POSITIVE CONTROL (V3-EXQ-1079 verbatim): z1_hat = z0 + [1, z0, a, z0 (x) a] @ W,
    ridge-fitted on the cell's own P0 buffer. Action-aware by construction."""
    z0 = torch.stack([b[0] for b in buf]).double()
    a = torch.stack([b[1] for b in buf]).double()
    z1 = torch.stack([b[2] for b in buf]).double()

    def phi(z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        cross = (z.unsqueeze(-1) * act.unsqueeze(-2)).reshape(z.shape[0], -1)
        return torch.cat([torch.ones(z.shape[0], 1, dtype=z.dtype), z, act, cross], -1)

    P = phi(z0, a)
    G = P.T @ P
    lam = RIDGE_REL_LAMBDA * float(torch.diagonal(G).mean())
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), P.T @ (z1 - z0))

    def head(z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        zd, ad = z.double(), act.double()
        return (zd + phi(zd, ad) @ W).float()
    return head


def _action_sep(head: Any, z0: torch.Tensor, acts: torch.Tensor) -> float:
    """Mean over rows and alternative actions of ||f(z,a) - f(z,a_alt)||_2."""
    a_dim = int(acts.shape[-1])
    idx = acts.argmax(-1)
    with torch.no_grad():
        p = head(z0, acts)
        d = [(p - head(z0, F.one_hot((idx + k) % a_dim, a_dim).float())).norm(dim=-1)
             for k in range(1, a_dim)]
    return float(torch.stack(d).mean().item())


def _encoder_equal(a: REEAgent, b: REEAgent) -> bool:
    sa, sb = a.latent_stack.state_dict(), b.latent_stack.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _hash(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().double().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _run_cell(arm: str, margin: Optional[float], seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(margin),
                  script_path=Path(__file__), config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(seed, margin)
        rng = random.Random(seed)
        coin = random.Random(seed ^ 0x5D99B5)

        # --- P0 training, reset-on-done (1073 pattern) ---------------------------
        opt = torch.optim.Adam(agent.e2.parameters(), lr=LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_CAP)
        obs = _reset(env)
        prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        n_train_resets = 0
        train_causes: Dict[str, int] = {}
        margin_steps = 0
        margin_loss_sum = 0.0
        for step in range(1, P0_STEPS + 1):
            if step % STEPS_PER_EPISODE == 0:
                ep = step // STEPS_PER_EPISODE
                print(f"  [train] {arm} seed={seed} ep {ep}/{EPISODES_PER_RUN} "
                      f"phase=P0", flush=True)
            z = _sense_zworld(agent, obs)
            if prev is not None and bool(torch.isfinite(z).all()):
                buf.append((prev[0], prev[1], z))
            a = _onehot(rng.randrange(env.action_dim), env.action_dim)
            _, _, done, info, obs = env.step(a.unsqueeze(0).to(agent.device))
            prev = (z, a)
            if done:
                c = str((info or {}).get("done_cause", ""))
                train_causes[c] = train_causes.get(c, 0) + 1
                obs = _reset(env)
                prev = None
                n_train_resets += 1
            if len(buf) < MIN_BUF_BEFORE_TRAIN:
                continue
            pool = list(buf)
            batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
            b0 = torch.stack([t[0] for t in batch]).to(agent.device)
            ba = torch.stack([t[1] for t in batch]).to(agent.device)
            b1 = torch.stack([t[2] for t in batch]).to(agent.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(agent.e2.world_forward(b0, ba), b1)
            if margin is not None and coin.random() < INTERVENTIONAL_FRACTION:
                m = agent.e2.compute_world_interventional_loss(b0, ba, _draw_cf(ba))
                margin_loss_sum += float(m.detach().item())
                margin_steps += 1
                loss = loss + m
            if not math.isfinite(float(loss.detach().item())):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()

        # --- LIVE battery from a FRESH OFF-config agent (arm-invariant rows) -------
        _, ag_b = _build(seed, None)
        enc_eq = _encoder_equal(agent, ag_b)
        bat = _collect_live_battery(ag_b, _make_env(seed),
                                    random.Random(seed ^ BATTERY_RNG_XOR), LIVE_BATTERY_N)
        bz0, bacts, bz1 = bat["z0"], bat["acts"], bat["z1"]
        head = agent.e2.world_forward
        untrained = ag_b.e2.world_forward

        et, es = _swap_errors(head, bz0, bacts, bz1)
        d_head = _d_act_ci(et, es, seed_offset=seed)
        pc = _ridge_head(list(buf))
        pt, ps = _swap_errors(pc, bz0, bacts, bz1)
        d_pc = _d_act_ci(pt, ps, seed_offset=seed + 7)
        ut, us = _swap_errors(untrained, bz0, bacts, bz1)
        mse_init = _battery_mse(untrained, bz0, bacts, bz1)
        mse_final = _battery_mse(head, bz0, bacts, bz1)
        id_mse = identity_predictor_mse(bz0, bz1)
        with torch.no_grad():
            pv = persistence_verdict(head(bz0, bacts), bz1, bz0)
        print(f"  [d_act] {arm} seed={seed} alpha={ALPHA_WORLD} d_act={d_head['d_act']} "
              f"CI=[{d_head['ci_low']}, {d_head['ci_high']}] status={d_head['status']} | "
              f"pc={d_pc['status']} | model_r2={pv.model_r2} persistence_r2="
              f"{pv.persistence_r2} | postdeath_rows={bat['n_postdeath_rows']} "
              f"resets={bat['n_resets']}", flush=True)

        row: Dict[str, Any] = {
            "arm": arm, "margin": margin, "seed": seed,
            "alpha_world": float(agent.config.latent.alpha_world),
            # routed DV
            "d_act": d_head["d_act"], "d_act_ci_low": d_head["ci_low"],
            "d_act_ci_high": d_head["ci_high"], "d_act_status": d_head["status"],
            "pc_d_act": d_pc["d_act"], "pc_d_act_ci_low": d_pc["ci_low"],
            "pc_d_act_ci_high": d_pc["ci_high"], "pc_d_act_status": d_pc["status"],
            "pc_live": bool(d_pc["ci_low"] is not None and d_pc["ci_low"] > 0.0),
            "d_act_untrained_head": _d_from(ut, us),
            # reconstruction guards
            "battery_mse_init": mse_init, "battery_mse_final": mse_final,
            "conv_rel_drop": (1.0 - mse_final / mse_init) if mse_init > 0 else None,
            "identity_predictor_mse": id_mse,
            "skill_vs_identity": (1.0 - mse_final / id_mse) if id_mse > 0 else None,
            # model_r2 ALWAYS next to persistence-relative skill (chip requirement 4)
            "model_r2": pv.model_r2, "persistence_r2": pv.persistence_r2,
            "persistence_relative_skill": pv.relative_skill,
            "persistence_status": pv.status,
            "persistence_ci_low": pv.ci_low, "persistence_ci_high": pv.ci_high,
            # live-battery hygiene
            "encoder_equal_to_battery_agent": enc_eq,
            "battery_hash": _hash(bz0, bacts, bz1),
            "n_rows_battery": int(bz0.shape[0]),
            "n_distinct_actions_battery": int(torch.unique(bacts, dim=0).shape[0]),
            "battery_n_resets": bat["n_resets"], "battery_done_causes": bat["causes"],
            "battery_n_postdeath_rows": bat["n_postdeath_rows"],
            "battery_min_health": bat["min_health"],
            "battery_max_world_state_norm": bat["max_world_state_norm"],
            "battery_mean_world_state_norm": bat["mean_world_state_norm"],
            "rms_dz_per_dim_battery": float(((bz1 - bz0) ** 2).mean().sqrt().item()),
            # F3: the margin is an L2 hinge in z_world units; record the scale it faces
            "transition_l2_mean": float((bz1 - bz0).norm(dim=-1).mean().item()),
            "head_action_separation_l2_mean": _action_sep(head, bz0, bacts),
            "train_n_resets": n_train_resets, "train_done_causes": train_causes,
            "p0_buffer_rows": len(buf),
            "margin_steps_applied": margin_steps,
            "margin_loss_mean": (margin_loss_sum / margin_steps) if margin_steps else None,
            # per-row errors so every contrast is re-derivable post hoc
            "per_row_se_true": [float(x) for x in et.tolist()],
            "per_row_se_swap_mean": [float(x) for x in es.tolist()],
        }
        cell.stamp(row)
        _ZG.observe(agent)
    print(f"verdict: {'PASS' if d_head['status'] == 'reads_action' else 'FAIL'}", flush=True)
    return row


def _t(r: Dict[str, Any], k: str) -> torch.Tensor:
    return torch.tensor(r[k], dtype=torch.float64)


def _seed_analysis(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    by = {r["arm"]: r for r in rows if r["seed"] == seed}
    off = by.get("ARM_OFF")
    res: Dict[str, Any] = {"seed": seed, "rungs": {}}
    if off is None or len(by) != len(ARMS):
        res.update(valid=False, reason="cell missing")
        return res
    aligned = len({r["battery_hash"] for r in by.values()}) == 1
    enc = all(r["encoder_equal_to_battery_agent"] for r in by.values())
    res.update(rows_aligned=aligned, encoder_equal=enc, pc_live=bool(off["pc_live"]),
               off_reads_action=bool(off["d_act_status"] == "reads_action"),
               off_d_act=off["d_act"], off_model_r2=off["model_r2"])
    res["valid"] = bool(aligned and enc and off["pc_live"])
    for arm, margin in ARMS[1:]:
        on = by[arm]
        c = _contrast_ci(_t(on, "per_row_se_true"), _t(on, "per_row_se_swap_mean"),
                         _t(off, "per_row_se_true"), _t(off, "per_row_se_swap_mean"),
                         seed_offset=seed + int(round(margin * 1000)) + 31)
        infl = (on["battery_mse_final"] / off["battery_mse_final"]
                if off["battery_mse_final"] > 0 else float("inf"))
        a3 = bool(on["conv_rel_drop"] is not None
                  and on["conv_rel_drop"] >= CONV_REL_DROP_BAR)
        a4 = bool(infl <= MSE_INFLATION_BAR)
        res["rungs"][arm] = {
            "margin": margin, "lift": c["delta"], "lift_ci_low": c["ci_low"],
            "lift_ci_high": c["ci_high"], "lift_status": c["status"],
            "grows": c["status"] == "grows",
            "excluded": bool(c["ci_high"] is not None and c["ci_high"] < DELTA_EQUIV),
            "shrinks": c["status"] == "shrinks",
            "margin_over_step_l2": (margin / on["transition_l2_mean"]
                                    if on["transition_l2_mean"] else None),
            "reads": on["d_act_status"] == "reads_action",
            "a3_1075_recorded": a3, "a4_mse_inflation": infl, "a4_ok": a4,
            "guard": a4, "model_r2": on["model_r2"],
            "model_r2_minus_off": (on["model_r2"] - off["model_r2"]
                                   if on["model_r2"] is not None
                                   and off["model_r2"] is not None else None)}
    return res


def _worst(rows: List[Dict[str, Any]], key: str, lowest: bool = True) -> Tuple[float, str]:
    w = (min if lowest else max)(rows, key=lambda r: r[key])
    return float(w[key]), f"{w['arm']}/seed={w['seed']}"


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    # Dry run: 2 seeds x ALL 5 arms at the FULL budget (SEEDS_REQUIRED stays reachable).
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm, margin in ARMS:
            rows.append(_run_cell(arm, margin, seed))

    per_seed = [_seed_analysis(rows, s) for s in seeds]
    valid = [s for s in per_seed if s["valid"]]
    for s in per_seed:
        for arm, rr in s["rungs"].items():
            print(f"  [lift] seed={s['seed']} {arm} vs OFF: lift={rr['lift']} CI=["
                  f"{rr['lift_ci_low']}, {rr['lift_ci_high']}] {rr['lift_status']} "
                  f"reads={rr['reads']} guard={rr['guard']} excluded={rr['excluded']} "
                  f"model_r2-off={rr['model_r2_minus_off']}", flush=True)

    n_valid = len(valid)
    n_aligned = sum(1 for s in per_seed if s.get("rows_aligned"))
    n_enc = sum(1 for s in per_seed if s.get("encoder_equal"))
    n_live = sum(1 for s in per_seed if s.get("pc_live"))
    worst_rows, worst_rows_cell = _worst(rows, "n_rows_battery")
    worst_distinct, worst_distinct_cell = _worst(rows, "n_distinct_actions_battery")
    worst_dead, worst_dead_cell = _worst(rows, "battery_n_postdeath_rows", lowest=False)
    worst_alpha = min(r["alpha_world"] for r in rows)
    preconditions = [
        {"name": "alpha_world_explicit_0p9", "description":
         "lowest alpha_world read back from any cell's built config (must equal 0.9)",
         "measured": worst_alpha, "threshold": float(ALPHA_WORLD), "direction": "lower",
         "control": "explicit from_dims kwarg + MECH-307 assert",
         "met": worst_alpha >= ALPHA_WORLD},
        {"name": "battery_rows_all_live", "description":
         "worst cell's count of battery rows recorded at or after agent_health <= 0 "
         "(the V3-EXQ-1075/1079 defect; must be 0)",
         "measured": worst_dead, "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_dead_cell, "control": "reset-on-done (1073 pattern)",
         "met": worst_dead <= 0},
        {"name": "battery_rows", "description": "worst cell's live-battery row count",
         "measured": worst_rows, "threshold": float(MIN_BATTERY_ROWS), "direction": "lower",
         "offending_cell": worst_rows_cell, "control": "fixed-length collection",
         "met": worst_rows >= MIN_BATTERY_ROWS},
        {"name": "battery_distinct_actions", "description":
         "worst cell's distinct actions in the battery (1 makes the swap undefined)",
         "measured": worst_distinct, "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cell,
         "control": "uniform random actions", "met": worst_distinct >= MIN_DISTINCT_ACTIONS},
        {"name": "rows_aligned_seeds", "description":
         "seeds whose battery is bit-identical across all 5 arms (paired contrast)",
         "measured": float(n_aligned), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "fresh OFF-config battery agent, own rng",
         "met": n_aligned >= SEEDS_REQUIRED},
        {"name": "encoder_equal_seeds", "description":
         "seeds where every arm's latent_stack equals the battery agent's",
         "measured": float(n_enc), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "encoder untrained; margin adds no params",
         "met": n_enc >= SEEDS_REQUIRED},
        {"name": "positive_control_live_seeds", "description":
         "READINESS (same statistic): seeds where a ridge action-aware predictor fitted "
         "on the OFF cell's own P0 buffer scores d_act CI lower > 0 on the battery",
         "measured": float(n_live), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "ridge z0 + [1, z0, a, z0 x a] @ W",
         "met": n_live >= SEEDS_REQUIRED},
    ]
    all_pre_met = all(p["met"] for p in preconditions) and n_valid >= SEEDS_REQUIRED

    rung_counts: Dict[str, Dict[str, int]] = {}
    for arm, _m in ARMS[1:]:
        rs = [s["rungs"][arm] for s in valid]
        rung_counts[arm] = {
            "pass": sum(1 for r in rs if r["grows"] and r["reads"] and r["guard"]),
            "bought": sum(1 for r in rs if r["grows"] and r["reads"] and not r["guard"]),
            "excluded": sum(1 for r in rs if r["excluded"]),
            "shrinks": sum(1 for r in rs if r["shrinks"]),
            "grows": sum(1 for r in rs if r["grows"])}
    n_off_reads = sum(1 for s in valid if s["off_reads_action"])
    off_premise = ("off_reads_action" if n_off_reads >= SEEDS_REQUIRED else
                   "off_blind" if (n_valid - n_off_reads) >= SEEDS_REQUIRED else "off_mixed")
    best_pass = max(c["pass"] for c in rung_counts.values())
    best_bought = max(c["bought"] for c in rung_counts.values())
    min_excluded = min(c["excluded"] for c in rung_counts.values())

    if not all_pre_met:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif best_pass >= SEEDS_REQUIRED:
        label = ("margin_amplifies_present_action_read" if off_premise == "off_reads_action"
                 else "margin_makes_blind_head_read_action")
        outcome = "PASS"
    elif best_bought >= SEEDS_REQUIRED:
        label, outcome = "margin_read_bought_by_reconstruction", "FAIL"
    elif min_excluded >= SEEDS_REQUIRED:
        max_shrinks = max(c["shrinks"] for c in rung_counts.values())
        label = ("margin_lowers_action_read" if max_shrinks >= SEEDS_REQUIRED
                 else "margin_does_not_raise_action_read")
        outcome = "FAIL"
    else:
        label, outcome = "margin_effect_undetermined", "FAIL"

    criteria = [
        {"name": "C1_margin_raises_action_read", "load_bearing": True,
         "measured": float(best_pass), "threshold": float(SEEDS_REQUIRED),
         "description": "max over ON rungs of valid seeds with lift CI lower > 0 AND ON "
                        "d_act CI lower > 0 AND reconstruction guard (A4) intact",
         "passed": bool(all_pre_met and best_pass >= SEEDS_REQUIRED)},
        {"name": "C2_read_bought_by_reconstruction", "load_bearing": False,
         "measured": float(best_bought), "threshold": float(SEEDS_REQUIRED),
         "description": "max over rungs of valid seeds with lift grows AND reads but "
                        "guard violated (FAIL-b)",
         "passed": bool(best_bought >= SEEDS_REQUIRED)},
        {"name": "C3_lift_excluded_every_rung", "load_bearing": False,
         "measured": float(min_excluded), "threshold": float(SEEDS_REQUIRED),
         "description": "min over rungs of valid seeds with lift CI upper < %.2f "
                        "(FAIL-a: the manipulation genuinely does not work)" % DELTA_EQUIV,
         "passed": bool(min_excluded >= SEEDS_REQUIRED)},
    ]
    combination_rule = (
        "alpha_world=0.9 explicit. Per seed, per ON rung: paired row bootstrap (%d "
        "resamples, %.0f%% CI) on lift = d_act(ON) - d_act(OFF), d_act = (SSE_swap - "
        "SSE_true)/(SSE_swap + SSE_true) on a LIVE 512-row battery (reset on done; "
        "analytic null 0). A seed is valid iff rows are aligned across arms, encoders "
        "are equal, and the ridge positive control reads its action. PASS if some rung "
        "has lift CI lower > 0 AND ON d_act CI lower > 0 AND guard (A4 MSE(ON)/MSE(OFF) "
        "<= %.1f; 1075's A3 >= %.2f recorded only) on >= %d valid "
        "seeds; else FAIL-b if some rung has grows+reads with guard violated on >= %d; "
        "else FAIL-a (margin_does_not_raise_action_read) if EVERY rung has lift CI upper "
        "< %.2f on >= %d valid seeds (label margin_lowers_action_read if some rung's lift "
        "CI upper < 0 on >= 2 valid seeds, else margin_does_not_raise_action_read); else "
        "UNDETERMINED. Any precondition unmet -> "
        "substrate_not_ready_requeue. PASS label split by off_arm_premise. model_r2, "
        "persistence_r2, persistence_relative_skill, skill_vs_identity and the "
        "untrained-head d_act are RECORDED and never gate."
        % (N_BOOTSTRAP, 100 * CI_LEVEL, MSE_INFLATION_BAR, CONV_REL_DROP_BAR,
           SEEDS_REQUIRED, SEEDS_REQUIRED, DELTA_EQUIV, SEEDS_REQUIRED))

    lifts = [rr["lift"] for s in valid for rr in s["rungs"].values()
             if rr["lift"] is not None]
    widths = [rr["lift_ci_high"] - rr["lift_ci_low"] for s in valid
              for rr in s["rungs"].values()
              if rr["lift_ci_low"] is not None and rr["lift_ci_high"] is not None]
    non_degenerate = bool(all_pre_met and len(set(round(d, 12) for d in lifts)) > 1
                          and len(widths) > 0 and all(w > 0 for w in widths))

    def _f(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    flat: Dict[str, Any] = {
        "alpha_world": ALPHA_WORLD, "n_cells": len(rows), "n_valid_seeds": n_valid,
        "n_seeds_off_reads_action": n_off_reads,
        "all_preconditions_met": 1 if all_pre_met else 0,
        "best_rung_pass_seeds": best_pass, "best_rung_bought_seeds": best_bought,
        "min_rung_excluded_seeds": min_excluded,
        "verdict_pass": 1 if outcome == "PASS" else 0,
        "verdict_fail_a": 1 if label in ("margin_does_not_raise_action_read",
                                         "margin_lowers_action_read") else 0,
        "verdict_fail_a_lowers": 1 if label == "margin_lowers_action_read" else 0,
        "verdict_fail_b": 1 if label == "margin_read_bought_by_reconstruction" else 0,
        "verdict_undetermined": 1 if label == "margin_effect_undetermined" else 0,
    }
    for s in per_seed:
        for arm, rr in s["rungs"].items():
            tag = f"{arm.lower()}_seed{s['seed']}"
            flat[f"lift_{tag}"] = _f(rr["lift"])
            flat[f"lift_ci_low_{tag}"] = _f(rr["lift_ci_low"])
            flat[f"lift_ci_high_{tag}"] = _f(rr["lift_ci_high"])
    for r in rows:
        tag = f"{r['arm'].lower()}_seed{r['seed']}"
        for k in ("d_act", "pc_d_act", "d_act_untrained_head", "model_r2",
                  "persistence_r2", "persistence_relative_skill", "skill_vs_identity",
                  "conv_rel_drop", "rms_dz_per_dim_battery", "transition_l2_mean",
                  "head_action_separation_l2_mean"):
            flat[f"{k}_{tag}"] = _f(r.get(k))
    flat = {k: v for k, v in flat.items() if v is not None}

    manifest: Dict[str, Any] = {
        "queue_id": QUEUE_ID,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "bears_on": BEARS_ON,
        "autopsy": AUTOPSY,
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "alpha_world": ALPHA_WORLD,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate else
                              "preconditions unmet, or lifts identical / zero-width CIs"),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "readout": flat,
        "arm_results": rows,
        "per_seed_rung_analysis": per_seed,
        "rung_counts": rung_counts,
        "interpretation": {
            "label": label,
            "off_arm_premise": off_premise,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {c["name"]: non_degenerate for c in criteria},
        },
    }
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()

    _manifest, _t0 = run_experiment(dry_run=_args.dry_run)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run,
        config={"alpha_world": ALPHA_WORLD, "arms": [a for a, _ in ARMS], "seeds": SEEDS,
                "p0_steps": P0_STEPS, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
                "live_battery_n": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                "interventional_fraction": INTERVENTIONAL_FRACTION,
                "delta_equiv": DELTA_EQUIV, "conv_rel_drop_bar": CONV_REL_DROP_BAR,
                "mse_inflation_bar": MSE_INFLATION_BAR, "n_bootstrap": N_BOOTSTRAP,
                "ci_level": CI_LEVEL, "ridge_rel_lambda": RIDGE_REL_LAMBDA,
                "seeds_required": SEEDS_REQUIRED,
                "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                        "num_resources": N_RESOURCES, "use_proxy_fields": True}},
        seeds=SEEDS, script_path=Path(__file__), started_at=_t0,
        z_goal_stream_stats=_ZG.stats())

    print(f"[{EXPERIMENT_TYPE}] outcome={_manifest['outcome']} "
          f"label={_manifest['interpretation']['label']}", flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
