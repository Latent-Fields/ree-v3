"""V3-EXQ-1079 -- SD-PP-B10 test 1: the alpha_world operating-point probe.
Does the world_forward head's action read GROW when z_world is not EMA-damped?

PURPOSE: diagnostic (validates_substrate SD-PP-B10-zworld-encoder-action-displacement,
candidate test rank 1 -- "MAY DISSOLVE THE PREMISE -- run first"). Non-contributory to
governance confidence by design; claim_ids is EMPTY, bears_on names the claims.

SLEEP DRIVER: not applicable (no sleep machinery; P0 world_forward training only).

RED-TEAM (Step 4.5, fable): CONTESTED -> fixed. F1 GROWS label split on the 0.3 control;
F2 SHRINKS encoder attribution gated on the paired positive-control contrast; F3 DV-symmetry
claim corrected, head/PC ratio recorded; F4 positive-control gate expected to pass, kept.

WHY THIS RUN EXISTS
-------------------
V3-EXQ-1073 and V3-EXQ-1075 never set alpha_world, so both inherited the
REEConfig.from_dims default 0.3 (ree_core/utils/config.py:7562; the fixed EMA
z_world = a*z_new + (1-a)*z_prev at ree_core/latent/stack.py:1584). SD-008's stable
floor is >= 0.9. The governance-ratified validated-NEGATIVE verdict on SD-PP-B5 was
therefore measured with z_world's per-step displacement damped (GFLAG-0433). This run
re-runs V3-EXQ-1075's ARM_OFF (no interventional margin; world_forward as landed; the
SAME env, dims, budget, optimiser, rollout and seeds) at alpha_world 0.3 (the
reproduction control), 0.9 and 1.0, and asks whether the head's action read grows.

THE INSTRUMENT CHANGE, AND WHY (user decision rec-20260923-98a4da88, "Option A")
--------------------------------------------------------------------------------
The pre-run reconnaissance (ree-v3/.scratch/alphaworld_probe/DECISION.md, GFLAG-0434)
found that 1075's battery ratio cannot carry this contrast:

  * 1075 collects the ORIGINAL battery first, on a FRESH agent, so its rows 0-9 are
    the z_world EMA relaxing from its initial state (7-10x the steady-state per-step
    displacement at alpha 0.3). The counterfactual battery, collected second, has no
    such rows. That order effect -- not the inverted action map -- is what made the
    copy-the-input "blind null" 0.51-0.58 at alpha 0.3. Collect cf first: 2.26-2.41.
    Drop rows 0-9: 1.08-1.38.
  * At alpha 0.9/1.0 the blind null (1.08-1.34) sits ABOVE the random-init null
    (~1.0) in 6/6 cells, so the (blind, init) band is inverted exactly where this
    probe needs it. Routed on "ratio lead over the blind null", the recon would have
    concluded "does not grow -> encoder" on 4/6 cells from a moving denominator.

So, as decided:
  1. BURN-IN before recording either battery (BURN_IN_STEPS random steps on a fresh
     agent, identical at every alpha, for both batteries). (1-0.3)^30 ~ 2e-5, so no
     EMA warm-up row survives at the slowest alpha. The burn-in battery has
     BURNIN_BATTERY_N = 512 rows (coordinator decision E1, 2026-09-23, under the user's
     standing delegation; ree-v3/.scratch/alphaworld_probe/DECISION2.md). 1075's 64 rows
     cover only steps 31-94 after env.reset() against a 3600-step training trajectory, and
     that window changed the ANSWER in reconnaissance, not just the width of the interval:
     at 64 rows the head's 0.3 read was -0.075/+0.056/-0.032 and the positive control was
     live on 1/3 seeds; at 512 rows it was +0.18/+0.23/+0.24, live on 3/3. The
     V3-EXQ-1075-order batteries stay at 64 rows so the 0.3 reproduction control is exact.
  2. ROUTE on the within-row ACTION-SWAP statistic, on the SAME rows and targets:
         d_act = (SSE_swap - SSE_true) / (SSE_swap + SSE_true)
     where SSE_true sums the head's squared error at the true action and SSE_swap the
     per-row MEAN of its squared error over every OTHER action (4 of 5). A head that
     ignores its action scores EXACTLY 0 -- an analytic null, not a measured one, so
     it cannot drift across alphas or machine classes. Per cell: paired bootstrap CI
     over rows, built exactly like experiments/_lib/persistence_skill_gate.py.
  3. "GROWS" is read per SEED on the paired contrast d_act(0.9) - d_act(0.3). The
     burn-in batteries at the two alphas are the SAME env transitions (same seed,
     same action rng, env independent of alpha; asserted, not assumed), so the
     bootstrap resamples ROW INDICES JOINTLY across the two cells:
        CI lower > 0 -> grows    CI upper < 0 -> shrinks    else cannot_determine
  4. V3-EXQ-1075's ratio, its blind null (both collection orders), its random-init
     null, the burn-in ratio and nulls, and the persistence_skill_gate verdict are
     all RECORDED and NONE of them gates.

DECLARED ROUTES (three-way, both directions pre-registered per CLAUDE.md)
------------------------------------------------------------------------
  GROWS         >= 2/3 seeds grow at 0.9 vs 0.3. The 0.3 operating point attenuated the
                head's action read: SD-PP-B5's validated-negative was measured at a damped
                operating point, and the small-displacement premise must be re-stated
                before any SD-PP-B10 build (tests 2/3 do not proceed as written). The
                LABEL is split on the run's own 0.3 control (red-team F1): if the 0.3
                cell already reads its action (d_act CI lower > 0) on >= 2 of the growing
                seeds, the read was PRESENT at 0.3 and only larger at 0.9 -- label
                alpha_world_amplifies_present_action_read (the 1075 negative then also
                owes to its instrument, not only to alpha); otherwise the read was absent
                at 0.3 -- label alpha_world_damping_masked_action_read.
  SHRINKS       >= 2/3 seeds shrink at 0.9 vs 0.3. Removing the smoothing did not help
                the head read its action. The LABEL is split on the budget-free positive
                control (red-team F2): at 0.9 the head must fit a ~10x larger residual at
                the SAME fixed 3600-step budget, so a head shrink can be under-fit alone.
                If the paired POSITIVE-CONTROL contrast pc_d_act(0.9) - pc_d_act(0.3) has
                CI upper <= 0 on >= 2 of the shrinking seeds, no more linearly readable
                action information reached z_world at 0.9 either -- the loss is upstream
                of the head, the ENCODER leg (MECH-582; SD-PP-B10 test 2), not the EMA --
                label action_read_not_raised_by_alpha_encoder_leg. Otherwise the data
                gained readable action information the head did not use -- label
                action_read_shrinks_head_fit_confound (NOT an encoder attribution).
  UNDETERMINED  otherwise (CI straddles, or seeds split). NOT an encoder verdict and NOT
                a smoothing verdict: the rows could not separate the two operating
                points. label alpha_contrast_undetermined.
  NOT READY     a precondition unmet (below). label substrate_not_ready_requeue.

  outcome = PASS iff a DETERMINATE verdict (GROWS or SHRINKS) was reached with every
  precondition met; FAIL otherwise. The science is in the label, not the outcome word.
  The alpha 1.0 contrast (d_act(1.0) - d_act(0.3)) is computed identically and
  RECORDED; 0.9 (SD-008's floor) is the only gating contrast.

PRECONDITIONS
-------------
  battery_rows / battery_distinct_actions -- worst cell, both burn-in batteries.
  alpha_batteries_row_aligned -- seeds whose burn-in battery rows are the same env
      transitions at every alpha (the paired contrast is undefined otherwise).
  positive_control_live_seeds (READINESS, same statistic as the criterion) -- a
      linear action-aware predictor z0 + ridge([1, z0, a, z0 x a]) fitted on the
      cell's OWN P0 training buffer is scored with the SAME d_act on the SAME burn-in
      battery. It reads its action by construction wherever z_world carries linearly
      readable action information, so it certifies the instrument can move off its
      analytic null. A seed whose 0.3 OR 0.9 positive control has CI lower <= 0 is
      scoped out of routing (its contrast is cannot_determine: the instrument was not
      shown live there); the run needs >= 2 seeds with a live instrument or it
      self-routes substrate_not_ready_requeue. Per-seed scoping, never a whole-run AND,
      so one dead cell cannot vacate another seed's valid contrast. Expected to pass
      (red-team F4: a ridge with a z0 x a term reads its own one-hot input; recon CI
      lower >= 0.32 in every cell at 512 rows) -- its load-bearing use is the paired
      PC contrast that splits the SHRINKS label, above.
  The 0.3 reproduction of V3-EXQ-1075 (ARM_OFF ratio 0.8336/0.8790/0.7068) is RECORDED
  as a control readout, NOT a precondition (it is a continuous value that differs
  across machine classes by ~5e-4 -- 1075a stop handover sec. 3).

DV-SYMMETRY (Step 3.5, per arm): each arm's DV is d_act, a normalised difference of
two sums of squared errors on fixed rows. The manipulation (alpha_world) changes the
z_world trajectory and hence the trained head's parameters and the targets; it is not
a uniform additive constant on the predictions (MSE is not shift-invariant in the
residual), not a monotone rescaling applied to both SSEs (the SSEs are of different
predictions), and not a permutation of interchangeable units, so the contrast is not
invariant under the manipulation. d_act is normalised, so a pure rescaling of z_world
cannot move it -- BUT alpha is not a pure rescaling (red-team F3, corrected here): it
changes the INPUT (at 0.3, z0 is an EMA over ~10 past transitions; at 1.0 it is the
instantaneous encode) and removes an action-independent lag term from the target, so
every reader's d_act can move with alpha, including the fixed-form positive control
(smoke: pc_d_act rose +0.153 / +0.059 on seeds 42/123). GROWS therefore says the
head's read grows at the higher operating point, not that the head's use of the
available information improved. The statistic that separates those -- the head/PC
d_act ratio per cell (readability_ratio_head_over_pc_*) -- is RECORDED, never routed.

KNOWN LIMITS, INHERITED FROM 1075 AND STATED RATHER THAN FIXED
---------------------------------------------------------------
  * The encoder is random-init and FROZEN (only agent.e2 is optimised; z_world is
    detached into the buffer) -- identical parameters at every alpha (hash-checked in
    reconnaissance). "Retrain the encoder per alpha" is not a choice this lineage has.
  * At 1075's budget the head is persistence-dominated at every alpha (recon d vs
    persistence -0.51..-0.81). Recorded, not gated (user decision): a SHRINKS verdict
    is a statement about THIS head at THIS budget.
  * The env keeps stepping after `done` (1075 ignores it); inherited unchanged.
  * SD-007 reafference is OFF (reafference_action_dim = 0), so SD-PP-B10's
    MECH-583/584 open_tension is latent here.
  * Open substrate_queue entries on exercised paths: SD-PP-B5 (corrupting, on
    e2_fast.world_forward) is THE SUBJECT of this probe, not an unrelated defect;
    SD-ZWORLD-SENSE-PATH-PARITY / SD-018 / SD-106 (degrading, agent.sense / stack.py)
    are noted in the queue entry.

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): the decisive readout is d_act on a
burn-in battery at alpha_world 0.9 vs 0.3. No recorded manifest carries a within-row
action-swap statistic, and no 1073/1075-lineage run set alpha_world at all (both are
0.3-only). Not recoverable -> run.
"""

from __future__ import annotations

import argparse
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
from experiments._lib.action_sensitivity_gate import (
    format_verdict, identity_predictor_mse, readiness_verdict)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.persistence_skill_gate import (
    BOOTSTRAP_SEED, CI_LEVEL, N_BOOTSTRAP, per_row_squared_error, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1079_sdppb10_alphaworld_operating_point_probe"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1079"
# DELIBERATELY EMPTY: a diagnostic on a substrate premise. The claims it bears on are
# named in BEARS_ON and reach governance through GFLAG-0433/0434, not through scoring.
CLAIM_IDS: List[str] = []
BEARS_ON = ["MECH-573", "MECH-574", "INV-063", "MECH-582", "SD-008"]
VALIDATES_SUBSTRATE = "SD-PP-B10-zworld-encoder-action-displacement"
# The counting anchors (battery rows, distinct actions, row alignment across alphas) are
# reachable by construction under fixed-length uniform-random-action collection, and the
# positive-control anchor is computed by the SAME _d_act_ci code that computes the routed
# DV, on the same rows -- no second, narrower predicate exists. Reconnaissance at the
# shipped battery size (512 rows): positive control live 9/9 cells, d_act 0.35-0.62.
ANCHOR_REACHABILITY_EXEMPT = (
    "counting anchors reachable by construction (random-action fixed-length batteries); "
    "positive-control anchor uses the shipped _d_act_ci predicate itself; recon at "
    "512 rows: live 9/9 cells")
# The only load-bearing bar is a seed COUNT (>= 2 of 3) over three-valued per-seed CIs;
# the routed statistic's room to move is established per cell by the positive-control
# precondition (a known action-reading predictor on the same rows), and a CI straddle
# routes to cannot_determine rather than FAIL. The `len(set(deltas)) > 1` test is a
# degeneracy flag, not a criterion threshold.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "load-bearing bar is a seed count over three-valued bootstrap CIs; DV range certified "
    "per cell by the positive-control precondition; len(set(deltas))>1 is a degeneracy flag")
USER_DECISION = "rec-20260923-98a4da88"

# ---- operating point: COPIED from V3-EXQ-1075 (itself copied from 1073) ----------
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
BATTERY_N = 64            # V3-EXQ-1075-order batteries (reproduction control)
BURNIN_BATTERY_N = 512    # burn-in batteries (the routed DV) -- decision E1, see docstring

# ---- the manipulation ------------------------------------------------------------
ALPHA_CONTROL = 0.3       # the inherited from_dims default; reproduction control
ALPHA_PRIMARY = 0.9       # SD-008's stable floor; the ONLY gating contrast
ALPHA_NO_EMA = 1.0        # recorded contrast
ALPHAS = [ALPHA_CONTROL, ALPHA_PRIMARY, ALPHA_NO_EMA]

# ---- pre-registered constants (never derived from this run) ---------------------
BURN_IN_STEPS = 30        # (1-0.3)^30 ~ 2.3e-5 of the initial state survives
SEEDS_REQUIRED = 2        # of 3 (1075's count)
MIN_BATTERY_ROWS = 32     # persistence_skill_gate.MIN_ROWS: below this a paired bootstrap is not evidence
MIN_DISTINCT_ACTIONS = 2
RIDGE_REL_LAMBDA = 1e-3   # positive-control ridge: lambda = this x mean diag(Phi^T Phi)
# Reference values the 0.3 control is COMPARED to (recorded, never gated).
REF_1075_OFF_RATIO = {42: 0.8336, 123: 0.8790, 456: 0.7068}

_ZG = ZGoalStreamAccumulator()


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _make_env(seed: int, invert_action_map: bool = False) -> CausalGridWorldV2:
    env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
                            num_resources=N_RESOURCES, use_proxy_fields=True)
    if invert_action_map:
        am = env._action_map
        env._action_map = {0: am[1], 1: am[0], 2: am[3], 3: am[2],
                           **{k: v for k, v in am.items() if k > 3}}
    return env


def _config_slice(alpha_world: float) -> Dict[str, Any]:
    """Declares ONLY what the cell's computation reads."""
    return {
        "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES, "use_proxy_fields": True},
        "dims": {"self_dim": SELF_DIM, "world_dim": WORLD_DIM},
        "latent": {"alpha_world": float(alpha_world)},
        "schedule": {"p0_steps": P0_STEPS, "batch_k": BATCH_K, "lr": LR,
                     "min_buf": MIN_BUF_BEFORE_TRAIN, "max_grad_norm": MAX_GRAD_NORM,
                     "buf_cap": BUF_CAP, "steps_per_episode": STEPS_PER_EPISODE},
        "battery": {"n_1075_order": BATTERY_N, "n_burnin": BURNIN_BATTERY_N,
                    "burn_in": BURN_IN_STEPS, "min_distinct_actions": MIN_DISTINCT_ACTIONS},
        "positive_control": {"ridge_rel_lambda": RIDGE_REL_LAMBDA},
        "margin": {"use_world_interventional": False},
    }


def _build(seed: int, alpha_world: float) -> Tuple[CausalGridWorldV2, REEAgent]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        alpha_world=float(alpha_world))
    # MECH-307 guard: from_dims silently swallows unknown kwargs. Assert it landed.
    if float(getattr(cfg.latent, "alpha_world", -1.0)) != float(alpha_world):
        raise RuntimeError("from_dims did not thread alpha_world")
    if bool(getattr(cfg.e2, "use_world_interventional", False)):
        raise RuntimeError("ARM_OFF must not carry the interventional margin")
    return env, REEAgent(cfg)


def _sense_zworld(agent: REEAgent, obs: Dict[str, Any]) -> torch.Tensor:
    obs_harm = obs.get("harm_obs", None)
    return agent.sense(_to_b(obs["body_state"], agent.device),
                       _to_b(obs["world_state"], agent.device),
                       obs_harm=(_to_b(obs_harm, agent.device)
                                 if obs_harm is not None else None)
                       ).z_world.detach().reshape(-1).clone()


def _collect_battery(agent: REEAgent, env: CausalGridWorldV2, rng: random.Random,
                     n: int, burn_in: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """burn_in == 0 is V3-EXQ-1075's collector exactly (same rng consumption, same
    rows). burn_in > 0 steps the env and the agent's z_world EMA `burn_in` times
    with random actions before the first recorded transition."""
    obs = env.reset()
    obs = obs[-1] if isinstance(obs, tuple) else obs
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    steps = 0
    guard = 0
    while len(rows) < n and guard < (n + burn_in) * 8:
        guard += 1
        z = _sense_zworld(agent, obs)
        if prev is not None and steps > burn_in and bool(torch.isfinite(z).all()):
            rows.append((prev[0], prev[1], z))
        idx = rng.randrange(env.action_dim)
        a = torch.zeros(env.action_dim, dtype=torch.float32)
        a[idx] = 1.0
        prev = (z, a)
        out = env.step(a.unsqueeze(0).to(agent.device))
        obs = out[-1]
        steps += 1
    return (torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows]),
            torch.stack([r[2] for r in rows]))


def _battery_mse(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                 z1: torch.Tensor) -> float:
    with torch.no_grad():
        return float(((head(z0, acts) - z1) ** 2).mean().item())


def _swap_errors(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row squared error at the TRUE action, and the per-row MEAN squared error
    over every OTHER action. Same rows, same targets."""
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
    """Paired bootstrap over rows, the persistence_skill_gate construction with the
    swapped-action prediction in the baseline's place. Analytic null 0."""
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
    """CI on d_act(hi alpha) - d_act(lo alpha). Rows are the same env transitions at
    both alphas, so ROW INDICES are resampled JOINTLY (paired across cells)."""
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
    """POSITIVE CONTROL: z1_hat = z0 + [1, z0, a, z0 (x) a] @ W, fitted by ridge on the
    cell's own P0 buffer. Action-aware by construction."""
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


def _rms_per_dim(z0: torch.Tensor, z1: torch.Tensor) -> float:
    return float(((z1 - z0) ** 2).mean().sqrt().item())


def _run_cell(alpha_world: float, seed: int) -> Dict[str, Any]:
    arm = f"ALPHA_{alpha_world:.1f}"
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(alpha_world),
                  script_path=Path(__file__), config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(seed, alpha_world)
        # --- (1) V3-EXQ-1075-order batteries, on this fresh agent, BEFORE training --
        rng = random.Random(seed)
        z0, acts, z1 = _collect_battery(agent, env, rng, BATTERY_N)
        cf_batt = _collect_battery(agent, _make_env(seed, invert_action_map=True),
                                   random.Random(seed), BATTERY_N)
        id_orig = identity_predictor_mse(z0, z1)
        id_cf = identity_predictor_mse(cf_batt[0], cf_batt[2])
        init_orig = _battery_mse(agent.e2.world_forward, z0, acts, z1)
        init_cf = _battery_mse(agent.e2.world_forward, *cf_batt)

        # --- (2) P0 training: V3-EXQ-1075 ARM_OFF, unchanged ---------------------
        opt = torch.optim.Adam(agent.e2.parameters(), lr=LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_CAP)
        obs = env.reset()
        obs = obs[-1] if isinstance(obs, tuple) else obs
        prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for step in range(1, P0_STEPS + 1):
            if step % STEPS_PER_EPISODE == 0:
                ep = step // STEPS_PER_EPISODE
                print(f"  [train] {arm} seed={seed} ep {ep}/{EPISODES_PER_RUN} "
                      f"phase=P0", flush=True)
            z = _sense_zworld(agent, obs)
            if prev is not None and bool(torch.isfinite(z).all()):
                buf.append((prev[0], prev[1], z))
            idx = rng.randrange(env.action_dim)
            a = torch.zeros(env.action_dim, dtype=torch.float32)
            a[idx] = 1.0
            prev = (z, a)
            out = env.step(a.unsqueeze(0).to(agent.device))
            obs = out[-1]
            if len(buf) < MIN_BUF_BEFORE_TRAIN:
                continue
            pool = list(buf)
            batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
            b0 = torch.stack([t[0] for t in batch]).to(agent.device)
            ba = torch.stack([t[1] for t in batch]).to(agent.device)
            b1 = torch.stack([t[2] for t in batch]).to(agent.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(agent.e2.world_forward(b0, ba), b1)
            if not math.isfinite(float(loss.detach().item())):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()

        head = agent.e2.world_forward
        mse_final = _battery_mse(head, z0, acts, z1)
        v1075 = readiness_verdict(head, z0, acts, z1, counterfactual_battery=cf_batt,
                                  min_rows=16, min_distinct_actions=MIN_DISTINCT_ACTIONS,
                                  ratio_floor=1.0, skill_floor=float("-inf"))
        print(format_verdict(v1075, f"{arm} seed={seed} [1075-order, recorded]"), flush=True)

        # --- (3) auxiliary FRESH agents (same seed -> identical frozen encoder) ----
        # Built AFTER training so the main cell's rollout is byte-identical to 1075.
        _, ag_cf_first = _build(seed, alpha_world)
        cf_first = _collect_battery(ag_cf_first, _make_env(seed, invert_action_map=True),
                                    random.Random(seed), BATTERY_N)
        orig_second = _collect_battery(ag_cf_first, _make_env(seed),
                                       random.Random(seed), BATTERY_N)
        blind_cf_first = (identity_predictor_mse(cf_first[0], cf_first[2])
                          / identity_predictor_mse(orig_second[0], orig_second[2]))
        _, ag_bi = _build(seed, alpha_world)
        bz0, bacts, bz1 = _collect_battery(ag_bi, _make_env(seed), random.Random(seed),
                                           BURNIN_BATTERY_N, burn_in=BURN_IN_STEPS)
        _, ag_bi_cf = _build(seed, alpha_world)
        bcf = _collect_battery(ag_bi_cf, _make_env(seed, invert_action_map=True),
                               random.Random(seed), BURNIN_BATTERY_N, burn_in=BURN_IN_STEPS)
        untrained_head = ag_bi.e2.world_forward
        b_id = identity_predictor_mse(bz0, bz1)
        b_id_cf = identity_predictor_mse(bcf[0], bcf[2])

        # --- (4) THE ROUTED DV: d_act on the burn-in battery ---------------------
        et, es = _swap_errors(head, bz0, bacts, bz1)
        d_head = _d_act_ci(et, es, seed_offset=seed)
        # positive control (readiness, same statistic)
        pc_head = _ridge_head(list(buf))
        pt, ps = _swap_errors(pc_head, bz0, bacts, bz1)
        d_pc = _d_act_ci(pt, ps, seed_offset=seed + 7)
        # untrained head (recorded context)
        ut, us = _swap_errors(untrained_head, bz0, bacts, bz1)
        d_init = _d_from(ut, us)
        # persistence verdict (recorded, not gated)
        with torch.no_grad():
            pv = persistence_verdict(head(bz0, bacts), bz1, bz0)
        print(f"  [d_act] {arm} seed={seed} d_act={d_head['d_act']} CI=[{d_head['ci_low']}, "
              f"{d_head['ci_high']}] status={d_head['status']} | positive control "
              f"d_act={d_pc['d_act']} status={d_pc['status']} | persistence={pv.status}",
              flush=True)

        row: Dict[str, Any] = {
            "arm": arm, "alpha_world": float(alpha_world), "seed": seed,
            # routed DV
            "d_act": d_head["d_act"], "d_act_ci_low": d_head["ci_low"],
            "d_act_ci_high": d_head["ci_high"], "d_act_status": d_head["status"],
            # readiness positive control
            "pc_d_act": d_pc["d_act"], "pc_d_act_ci_low": d_pc["ci_low"],
            "pc_d_act_ci_high": d_pc["ci_high"], "pc_d_act_status": d_pc["status"],
            "pc_live": bool(d_pc["ci_low"] is not None and d_pc["ci_low"] > 0.0),
            "d_act_untrained_head": d_init,
            # displacement
            "rms_dz_per_dim_burnin": _rms_per_dim(bz0, bz1),
            "rms_dz_per_dim_1075_order": _rms_per_dim(z0, z1),
            # 1075 instrument, RECORDED
            "ratio_1075_order": v1075.ratio,
            "ratio_1075_ref": REF_1075_OFF_RATIO.get(seed),
            "gate_status_1075_order": v1075.status,
            "skill_vs_identity_1075_order": v1075.skill,
            "blind_null_1075_order": (id_cf / id_orig) if id_orig > 0 else None,
            "blind_null_cf_first": blind_cf_first,
            "init_null_1075_order": (init_cf / init_orig) if init_orig > 0 else None,
            "blind_null_burnin": (b_id_cf / b_id) if b_id > 0 else None,
            "ratio_burnin": (_battery_mse(head, *bcf) / _battery_mse(head, bz0, bacts, bz1)),
            "init_null_burnin": (_battery_mse(untrained_head, *bcf)
                                 / _battery_mse(untrained_head, bz0, bacts, bz1)),
            "identity_mse_1075_order": id_orig,
            "identity_mse_burnin": b_id,
            "battery_mse_init_1075_order": init_orig,
            "battery_mse_final_1075_order": mse_final,
            "conv_rel_drop_1075_order": (1.0 - mse_final / init_orig) if init_orig > 0 else None,
            # persistence, RECORDED
            "persistence_status": pv.status,
            "persistence_relative_skill": pv.relative_skill,
            "persistence_ci_low": pv.ci_low, "persistence_ci_high": pv.ci_high,
            "persistence_r2": pv.persistence_r2, "model_r2": pv.model_r2,
            # battery hygiene
            "n_rows_burnin": int(bz0.shape[0]), "n_rows_burnin_cf": int(bcf[0].shape[0]),
            "n_distinct_actions_burnin": int(torch.unique(bacts, dim=0).shape[0]),
            "burnin_action_sequence": [int(i) for i in bacts.argmax(-1).tolist()],
            "p0_buffer_rows": len(buf),
            # per-row errors, so every contrast is re-derivable post hoc
            "per_row_se_true": [float(x) for x in et.tolist()],
            "per_row_se_swap_mean": [float(x) for x in es.tolist()],
            "per_row_pc_se_true": [float(x) for x in pt.double().tolist()],
            "per_row_pc_se_swap_mean": [float(x) for x in ps.double().tolist()],
        }
        cell.stamp(row)
        _ZG.observe(agent)
    print(f"verdict: {'PASS' if d_head['status'] == 'reads_action' else 'FAIL'}", flush=True)
    return row


def _seed_contrast(rows: List[Dict[str, Any]], seed: int, hi: float) -> Dict[str, Any]:
    by = {r["alpha_world"]: r for r in rows if r["seed"] == seed}
    lo_r, hi_r = by.get(ALPHA_CONTROL), by.get(hi)
    res: Dict[str, Any] = {"seed": seed, "alpha_hi": hi, "alpha_lo": ALPHA_CONTROL}
    if lo_r is None or hi_r is None:
        res.update(status="cannot_determine", reason="cell missing")
        return res
    aligned = lo_r["burnin_action_sequence"] == hi_r["burnin_action_sequence"]
    res["rows_aligned"] = aligned
    res["instrument_live"] = bool(lo_r["pc_live"] and hi_r["pc_live"])
    c = _contrast_ci(torch.tensor(hi_r["per_row_se_true"], dtype=torch.float64),
                     torch.tensor(hi_r["per_row_se_swap_mean"], dtype=torch.float64),
                     torch.tensor(lo_r["per_row_se_true"], dtype=torch.float64),
                     torch.tensor(lo_r["per_row_se_swap_mean"], dtype=torch.float64),
                     seed_offset=seed + int(round(hi * 1000)))
    res.update({k: c[k] for k in ("delta", "ci_low", "ci_high", "n_rows", "n_bootstrap")})
    pc = _contrast_ci(torch.tensor(hi_r["per_row_pc_se_true"], dtype=torch.float64),
                      torch.tensor(hi_r["per_row_pc_se_swap_mean"], dtype=torch.float64),
                      torch.tensor(lo_r["per_row_pc_se_true"], dtype=torch.float64),
                      torch.tensor(lo_r["per_row_pc_se_swap_mean"], dtype=torch.float64),
                      seed_offset=seed + int(round(hi * 1000)) + 17)
    res.update(pc_delta=pc["delta"], pc_ci_low=pc["ci_low"], pc_ci_high=pc["ci_high"],
               pc_status=pc["status"],
               lo_cell_reads_action=bool(lo_r["d_act_status"] == "reads_action"))
    if not aligned:
        res.update(status="cannot_determine",
                   reason="burn-in battery rows are not the same transitions across alphas")
    elif not res["instrument_live"]:
        res.update(status="cannot_determine",
                   reason="positive control not live in one of the two cells")
    else:
        res.update(status=c["status"], reason="paired row bootstrap")
    return res


def _worst(rows: List[Dict[str, Any]], key: str) -> Tuple[float, str]:
    w = min(rows, key=lambda r: r[key])
    return float(w[key]), f"{w['arm']}/seed={w['seed']}"


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    # Dry run: 2 seeds x ALL 3 alphas at the FULL budget. 2 seeds keeps the
    # pre-registered SEEDS_REQUIRED=2 reachable; nothing is short-circuited.
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for aw in ALPHAS:
            rows.append(_run_cell(aw, seed))

    contrasts_09 = [_seed_contrast(rows, s, ALPHA_PRIMARY) for s in seeds]
    contrasts_10 = [_seed_contrast(rows, s, ALPHA_NO_EMA) for s in seeds]
    n_grows = sum(1 for c in contrasts_09 if c["status"] == "grows")
    n_shrinks = sum(1 for c in contrasts_09 if c["status"] == "shrinks")
    n_live = sum(1 for c in contrasts_09 if c.get("instrument_live"))
    n_aligned = sum(1 for c in contrasts_09 if c.get("rows_aligned"))
    for c in contrasts_09 + contrasts_10:
        print(f"  [contrast] seed={c['seed']} alpha {c['alpha_hi']} vs {c['alpha_lo']}: "
              f"delta={c.get('delta')} CI=[{c.get('ci_low')}, {c.get('ci_high')}] "
              f"status={c['status']} ({c.get('reason')})", flush=True)

    worst_rows, worst_rows_cell = _worst(
        [dict(r, _n=min(r["n_rows_burnin"], r["n_rows_burnin_cf"])) for r in rows], "_n")
    worst_distinct, worst_distinct_cell = _worst(rows, "n_distinct_actions_burnin")
    preconditions = [
        {"name": "battery_rows", "description":
         "worst cell's burn-in battery row count (either battery)",
         "measured": worst_rows, "threshold": float(MIN_BATTERY_ROWS), "direction": "lower",
         "offending_cell": worst_rows_cell,
         "control": "fixed-length random-action collection after burn-in",
         "met": worst_rows >= MIN_BATTERY_ROWS},
        {"name": "battery_distinct_actions", "description":
         "worst cell's distinct actions in the burn-in battery (a 1-action battery "
         "makes the swap undefined)",
         "measured": worst_distinct, "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cell,
         "control": "uniform random actions", "met": worst_distinct >= MIN_DISTINCT_ACTIONS},
        {"name": "alpha_batteries_row_aligned", "description":
         "seeds whose burn-in battery rows are the SAME env transitions at 0.3 and 0.9 "
         "(the paired contrast is undefined otherwise)",
         "measured": float(n_aligned), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "same seed, same action rng, alpha-independent env",
         "met": n_aligned >= SEEDS_REQUIRED},
        {"name": "positive_control_live_seeds", "description":
         "READINESS (same statistic as the criterion): seeds where a linear action-aware "
         "predictor fitted on the cell's own P0 buffer scores d_act CI lower > 0 on the "
         "same burn-in battery at BOTH 0.3 and 0.9. A seed failing this is scoped out "
         "of routing (cannot_determine), not failed",
         "measured": float(n_live), "threshold": float(SEEDS_REQUIRED), "direction": "lower",
         "control": "ridge z0 + [1, z0, a, z0 x a] @ W; reads its action by construction "
                    "wherever z_world carries linearly readable action information",
         "met": n_live >= SEEDS_REQUIRED},
    ]
    all_pre_met = all(p["met"] for p in preconditions)

    if not all_pre_met:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif n_grows >= SEEDS_REQUIRED:
        # F1: split on the run's own 0.3 control, among the growing seeds.
        present = sum(1 for c in contrasts_09
                      if c["status"] == "grows" and c.get("lo_cell_reads_action"))
        label = ("alpha_world_amplifies_present_action_read" if present >= SEEDS_REQUIRED
                 else "alpha_world_damping_masked_action_read")
        outcome = "PASS"
    elif n_shrinks >= SEEDS_REQUIRED:
        # F2: attribute to the encoder only if the budget-free positive control also
        # gained no readable action information at 0.9 on the shrinking seeds.
        pc_flat = sum(1 for c in contrasts_09
                      if c["status"] == "shrinks" and c.get("pc_ci_high") is not None
                      and c["pc_ci_high"] <= 0.0)
        label = ("action_read_not_raised_by_alpha_encoder_leg" if pc_flat >= SEEDS_REQUIRED
                 else "action_read_shrinks_head_fit_confound")
        outcome = "PASS"
    else:
        label, outcome = "alpha_contrast_undetermined", "FAIL"

    n_det = max(n_grows, n_shrinks)
    criteria = [
        {"name": "C1_determinate_alpha_contrast", "load_bearing": True,
         "measured": float(n_det), "threshold": float(SEEDS_REQUIRED),
         "description": "max(seeds growing, seeds shrinking) on the paired "
                        "d_act(0.9) - d_act(0.3) CI; >= 2 of 3 gives a determinate verdict",
         "passed": bool(all_pre_met and n_det >= SEEDS_REQUIRED)},
        {"name": "C1a_seeds_grow", "load_bearing": False,
         "measured": float(n_grows), "threshold": float(SEEDS_REQUIRED),
         "description": "seeds whose contrast CI lies entirely above 0 (GROWS route)",
         "passed": bool(n_grows >= SEEDS_REQUIRED)},
        {"name": "C1b_seeds_shrink", "load_bearing": False,
         "measured": float(n_shrinks), "threshold": float(SEEDS_REQUIRED),
         "description": "seeds whose contrast CI lies entirely below 0 (SHRINKS route)",
         "passed": bool(n_shrinks >= SEEDS_REQUIRED)},
    ]
    combination_rule = (
        "Per seed, a paired row bootstrap (%d resamples, %.0f%% CI) on d_act(0.9) - "
        "d_act(0.3), d_act = (SSE_swap - SSE_true)/(SSE_swap + SSE_true) on the burn-in "
        "battery (analytic null 0): CI lower > 0 -> grows, CI upper < 0 -> shrinks, else "
        "cannot_determine; a seed whose positive control is not live at either alpha, or "
        "whose rows are not aligned, is cannot_determine. GROWS if >= %d seeds grow; "
        "SHRINKS if >= %d seeds shrink; otherwise UNDETERMINED. Every precondition must be "
        "met or the run self-routes substrate_not_ready_requeue. outcome PASS iff GROWS or "
        "SHRINKS. LABEL SPLITS (red-team, interpretation only; counts unchanged): GROWS is "
        "alpha_world_amplifies_present_action_read if the 0.3 cell already reads its action "
        "(d_act CI lower > 0) on >= %d growing seeds, else alpha_world_damping_masked_"
        "action_read; SHRINKS is attributed to the encoder "
        "(action_read_not_raised_by_alpha_encoder_leg) only if the paired positive-control "
        "contrast pc_d_act(0.9) - pc_d_act(0.3) has CI upper <= 0 on >= %d shrinking seeds, "
        "else action_read_shrinks_head_fit_confound. The alpha 1.0 contrast, the "
        "V3-EXQ-1075 battery ratio and its nulls, the head/PC readability ratio, and the "
        "persistence verdict are RECORDED and do not gate."
        % (N_BOOTSTRAP, 100 * CI_LEVEL, SEEDS_REQUIRED, SEEDS_REQUIRED,
           SEEDS_REQUIRED, SEEDS_REQUIRED))

    deltas = [c["delta"] for c in contrasts_09 if c.get("delta") is not None]
    widths = [c["ci_high"] - c["ci_low"] for c in contrasts_09
              if c.get("ci_low") is not None and c.get("ci_high") is not None]
    c1_non_degenerate = bool(all_pre_met and len(set(round(d, 12) for d in deltas)) > 1
                             and all(w > 0 for w in widths) and len(widths) > 0)
    non_degenerate = c1_non_degenerate

    def _f(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    flat: Dict[str, Any] = {
        "n_cells": len(rows), "n_seeds_grow_09": n_grows, "n_seeds_shrink_09": n_shrinks,
        "n_seeds_instrument_live": n_live, "n_seeds_rows_aligned": n_aligned,
        "n_seeds_grow_10": sum(1 for c in contrasts_10 if c["status"] == "grows"),
        "n_seeds_shrink_10": sum(1 for c in contrasts_10 if c["status"] == "shrinks"),
        "all_preconditions_met": 1 if all_pre_met else 0,
        "verdict_grows": 1 if label in ("alpha_world_damping_masked_action_read",
                                        "alpha_world_amplifies_present_action_read") else 0,
        "verdict_grows_read_present_at_0p3":
            1 if label == "alpha_world_amplifies_present_action_read" else 0,
        "verdict_shrinks": 1 if label in ("action_read_not_raised_by_alpha_encoder_leg",
                                          "action_read_shrinks_head_fit_confound") else 0,
        "verdict_shrinks_encoder_attributed":
            1 if label == "action_read_not_raised_by_alpha_encoder_leg" else 0,
        "verdict_undetermined": 1 if label == "alpha_contrast_undetermined" else 0,
    }
    for c in contrasts_09:
        flat[f"delta_d_act_09_seed{c['seed']}"] = _f(c.get("delta"))
        flat[f"delta_d_act_09_ci_low_seed{c['seed']}"] = _f(c.get("ci_low"))
        flat[f"delta_d_act_09_ci_high_seed{c['seed']}"] = _f(c.get("ci_high"))
        flat[f"delta_pc_d_act_09_seed{c['seed']}"] = _f(c.get("pc_delta"))
        flat[f"delta_pc_d_act_09_ci_low_seed{c['seed']}"] = _f(c.get("pc_ci_low"))
        flat[f"delta_pc_d_act_09_ci_high_seed{c['seed']}"] = _f(c.get("pc_ci_high"))
    for c in contrasts_10:
        flat[f"delta_d_act_10_seed{c['seed']}"] = _f(c.get("delta"))
    for r in rows:
        tag = f"a{int(round(r['alpha_world'] * 10)):02d}_seed{r['seed']}"
        for k in ("d_act", "pc_d_act", "rms_dz_per_dim_burnin", "rms_dz_per_dim_1075_order",
                  "ratio_1075_order", "blind_null_1075_order", "blind_null_cf_first",
                  "init_null_1075_order", "blind_null_burnin", "ratio_burnin",
                  "init_null_burnin", "persistence_relative_skill"):
            flat[f"{k}_{tag}"] = _f(r.get(k))
        if r.get("pc_d_act"):
            flat[f"readability_ratio_head_over_pc_{tag}"] = _f(r["d_act"] / r["pc_d_act"])
    flat = {k: v for k, v in flat.items() if v is not None}

    repro = {r["seed"]: {"ratio": r["ratio_1075_order"], "ref": r["ratio_1075_ref"],
                         "abs_diff": (abs(r["ratio_1075_order"] - r["ratio_1075_ref"])
                                      if r["ratio_1075_order"] is not None
                                      and r["ratio_1075_ref"] is not None else None)}
             for r in rows if r["alpha_world"] == ALPHA_CONTROL}

    manifest: Dict[str, Any] = {
        "queue_id": QUEUE_ID,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "bears_on": BEARS_ON,
        "user_decision": USER_DECISION,
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": (None if non_degenerate else
                              "preconditions unmet, or contrast deltas identical / "
                              "zero-width CIs across seeds"),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "readout": flat,
        "arm_results": rows,
        "contrasts_alpha_0p9_vs_0p3": contrasts_09,
        "contrasts_alpha_1p0_vs_0p3_RECORDED": contrasts_10,
        "custom_information": {
            "reproduction_of_v3_exq_1075_off_ratio_RECORDED": repro,
            "note": "1075 ratio and nulls are recorded non-gating per " + USER_DECISION,
        },
        "interpretation": {
            "label": label,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                "C1_determinate_alpha_contrast": c1_non_degenerate,
                "C1a_seeds_grow": c1_non_degenerate,
                "C1b_seeds_shrink": c1_non_degenerate},
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
        config={"alphas": ALPHAS, "alpha_primary": ALPHA_PRIMARY, "seeds": SEEDS,
                "p0_steps": P0_STEPS, "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
                "battery_n": BATTERY_N, "burn_in_steps": BURN_IN_STEPS,
                "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                "ridge_rel_lambda": RIDGE_REL_LAMBDA, "seeds_required": SEEDS_REQUIRED,
                "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                        "num_resources": N_RESOURCES, "use_proxy_fields": True}},
        seeds=SEEDS, script_path=Path(__file__), started_at=_t0,
        z_goal_stream_stats=_ZG.stats())

    print(f"[{EXPERIMENT_TYPE}] outcome={_manifest['outcome']} "
          f"label={_manifest['interpretation']['label']}", flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
