"""
V3-EXQ-1093 -- MECH-428 parent-statistic ESS sweep: an INSTRUMENT / NON-DEGENERACY DIAGNOSTIC.

Sweeps SD-092's parent_goal_alpha x parent_goal_decay x N_STEPS (event count) and asks, per
operating point, whether the correctly-nulled parent-level statistic S separates from its own
re-constructed null AT ALL. It does not test MECH-428.

DECISION (user, 2026-09-24 ~17:20Z, rec-20260924-701216a5, on
chip-20260924-decision-exq1093-c1-pr3-identity): option (b) -- C1 ranges only over sub-ceiling
cells (ESS <= 0.5 x n_att); PR3 stays readiness; the default-cell replication is a REPORTED
readout (C2). Parked 2026-09-24 12:xx-17:20Z; full record REE_assembly/evidence/planning/
exq1093_mech428_parent_stat_redteam_blocking_20260924.md. Pilot raw data:
experiments/_scratch/v3_exq_1093_pilot_seeds101_102.json.

RED-TEAM PASS 2 (fable, one pass, foreground, 2026-09-24, after decision (b)): CONTESTED, no
  BLOCKING. C1 now discriminates (it can fail with PR3 met).
  F4 CONTESTED, FIXED: the C1-unmet label claimed "no achievable range at registered settings"
     although excluded above-cap cells are registered settings and may separate; that branch now
     splits (parent_statistic_separates_only_above_c1_ess_cap), and separating_real_cells lists
     ALL real cells with an in_c1_set flag.
  F5 CONTESTED-minor, RECORDED: C1 is a max, so it binds on the highest-ESS eligible cell (pilot
     (0.025, 0, N=1600), ESS 78.7); a PASS means the margin survives to ESS <= 0.5 x n_att, and
     the crossover itself is reported (C1_min_separating_ess_in_set, cell map). best_cell now
     breaks ties toward the binding (highest-ESS) cell. Scope stated in the manifest.

RED-TEAM PASS 1 (fable, one pass, foreground, 2026-09-24): BLOCKING.
  F1 BLOCKING, CONFIRMED against source + pilot; RESOLVED by the user's decision (b) above
     (not re-designed unilaterally): C1 was implied by PR3 by construction.
     The real cell (alpha=0.005, decay=0) has near-flat weights, i.e. it IS the uniform ceiling
     (pilot ESS 58.6 vs 59 at N=400; S 0.002414 vs 0.002422, p95 0.000744 vs 0.000749), and
     PR3 is evaluated before C1. So C1 never fails independently, and the one genuine negative
     ("no achievable range even at the ceiling") routes to substrate_not_ready_requeue, a
     harness-defect label that re-queues an identical run. The label
     parent_statistic_no_achievable_range_at_registered_settings is unreachable.
  F2 CONTESTED, accepted as CONSERVATIVE (biases the crossover toward non-separation, never the
     sign): the shift null's bump rule puts ~15 % of pseudo-credits on the arrival tick after a
     credited one, and bumped pseudo-credits lose their nearest control neighbour. Confirmer now
     recorded per seed x N (null_bump_frac): smoke 0.08-0.16, matching the reviewer's 14.5-15.3 %.
  F3 CONTESTED, FIXED: the reported-only unmatched null drew controls from a pool that still held
     its own pseudo-credited ticks (null-side self-overlap); controls now exclude them.

SCOPE (orchestrator decision Q-MECH428 -> A, relayed in campaign
science-20260924-mech428-alpha-diag; chip chip-20260917-mech428-parent-goal-alpha-sweep):
  * A PASS here establishes ONLY the third clause of MECH-428's what_would_answer NON-DEGENERACY
    PRECONDITION ("the parent-level statistic must have an achievable range that separates from a
    null that re-runs parent construction (parent_goal_alpha / event rate swept ...)"), plus the
    attainment-rate clause. It does NOT test the CONFIRMING clause: no live consumer is armed
    (E3Config.parent_goal_weight stays 0.0 -- the behavioural leg is EXP-0710 / GFLAG-0464), the
    policy is a scripted waypoint walk, and there is no NO-SUBGOAL arm and no 626b forced-seed arm.
  * claim_ids = [MECH-428], experiment_purpose = diagnostic, evidence_direction =
    non_contributory on EVERY branch. A PASS must not be routed as support.

AUTHORITY: REE_assembly/evidence/planning/exq884c_mech428_c1_ema_self_overlap_redteam_blocking_20260916.md
(ree-v3 bda53ea / 5cc777a). Carried forward, not re-derived:
  * pre-arrival crediting via the REAL substrate call
    agent.notify_subgoal_attainment(ttype, child_representation=prev_z_world)  (agent.py:10718)
  * alpha_world = 0.9 (the default 0.3 smears the attainment tick into transit; 884c sec 2a)
  * SD-094 env flags (subgoal_arrival_position_check, hazard_free_contamination_gate) ON
  * P0: SD-070 z_world warmup 20 x 50 steps, preservation_weight 1000, latent_stack transferred
    to a fresh eval agent (the 884c probe pattern).

THE STATISTIC, and why it does NOT have 884c F1's shape.
  F1: any statistic comparing the parent to a group the parent was BUILT FROM carries a
  self-overlap term that a fixed-parent permutation null cancels, so it is biased positive with
  zero signal (12/12 false positives on synthetic data). S never does that:
      S = 1 - cos(parent_credited, parent_control)
  parent_credited is the SD-092 EMA over the credited (pre-arrival) representations; parent_control
  is built with the IDENTICAL weights (same credit order, same alpha/decay) from a DISJOINT set of
  non-credited representations. Neither parent is compared to a set it contains, and the null
  RE-RUNS parent construction on both sides, so any construction-induced bias appears identically
  in observed and null.

  Replay is exact linear algebra: per-tick decay d (applied in GoalState.update, before the credit
  on the same tick) and per-credit pull a give
      parent_T = sum_k w_k z_k,   w_k = a (1-a)^(n-1-k) (1-d)^(T - t_k)
  so every parent is w @ Z. PR2 checks this against the REAL substrate primitive for EVERY cell
  (one real GoalState per (alpha, decay), driven with the agent's own update/credit sequence), and
  checks the agent's own GoalState against the reference-cell GoalState.

  CONTROL AND NULL -- corrections to the 884c probe form, each MEASURED in this session's pilots:
  * K_CTRL-averaged control (orchestrator decision): s_obs is the mean of S over K_CTRL control
    draws, and every null sample is the same K_CTRL-average, so both sides carry the same
    control-side averaging.
  * TIME-LOCAL controls. Recency weighting puts most of the parent on the LATEST credits, so a
    control drawn from anywhere in the run (the 884c form) differs from the credited parent by
    whatever z_world content persists in time -- e.g. the waypoint layout, constant within a
    sequence. Measured: under that unmatched form the RANDOM-CREDIT negative control reaches pct
    98.2-99.8 at N=400 on pilot seed 102 (it is kept as a reported-only readout). So each credit's
    controls come from its own neighbourhood: the ticks strictly between its previous and next
    credit (parameter-free -- set by the event structure; widened by one neighbour only if empty).
  * CIRCULAR TIME-SHIFT null. A null that draws its pseudo-credited tick uniformly inside the same
    windows is biased too: the credited tick sits in the window interior, so its controls are
    systematically farther away in time than a uniform pair's (measured: the negative control at
    pct 94-98 on pilot seed 102, N=1600). The null therefore shifts the WHOLE credited set by a
    random delta in [1, N-1] (circularly; a shifted tick landing on an attained or already-used tick
    is bumped forward), keeps credit order (w_k stays with position k), and draws controls by the
    same window rule around the shifted set. Observed and null then share event geometry, recency
    structure and control locality, and differ only in WHICH ticks are credited.
  * Every generator is seeded with zlib.crc32 of a string tag -- never hash() (salted per process
    by PYTHONHASHSEED) and never a tuple passed to random.Random (TypeError). Index draws are shared
    across all (alpha, decay) cells of a seed x N x arm (common random numbers).

SENSE PATH. The policy is scripted, so only z_world is needed and the rollout calls
agent.sense_flat() (the first statement of act(); ~9 ms/tick vs ~1.7 s for act() on a contended
2-core box). PR0 re-proves per seed that this is exact: a twin carrying the eval agent's FULL
state_dict runs the full act() for the first EQUIV_TICKS ticks of the same walk; its z_world must
match (pilot: max |diff| 0.0 on both seeds). The twin must copy the FULL state, not only
latent_stack: sense() also runs the agent-level world_obs_encoder / body_obs_encoder (random
Linear+ReLU pre-projections outside latent_stack) -- the open cosmetic substrate item
SD-ZWORLD-SENSE-PATH-PARITY (P0 trains split_encoder.world_encoder on RAW world_state; inference
reads it through that untrained pre-projection). This run measures through the production path,
as every 884-lineage probe did; that limitation applies to it unchanged.

ARMS / CELLS (all from ONE rollout per seed -- the parent does not feed back into behaviour:
parent_goal_weight = 0 and the policy is scripted, so every (alpha, decay) cell sees the identical
representation stream; N=400 is the tick-400 prefix of the 1600-tick walk):
  * REAL cells: alpha in ALPHAS x decay in DECAYS x N in N_LIST (24 cells).
  * UNIFORM ceiling (positive control of the SAME statistic S, SAME null): uniform weights over
    all credits up to N -- the alpha->0, decay=0 limit, the maximum ESS any SD-092 setting can
    reach with this many events. If S cannot separate here, no alpha can.
  * RANDOM-CREDIT negative control (884c F2 + F4 fixed): n_att ticks drawn at random from the
    NON-attained ticks (a real random draw -- no set() collapse), weighted at THEIR OWN tick
    positions, controls and null built by the same rules on its OWN credited partition, never the
    attainment partition.
  * UNMATCHED 884c form (REPORTED ONLY, never gating), true arm and negative control.

EVENT RATE via N_STEPS (orchestrator decision): N_LIST = [400, 1600]. The scripted walk's credit
rate per tick is fixed by the geometry; N raises the credit COUNT (~60 -> ~245), which is what
bounds ESS. decay is co-swept over {0.005 (GoalConfig default), 0.0}.

PILOT (held-out seeds 101/102, full budgets, this exact instrument; not part of the verdict):
  * ESS ceiling: uniform S 0.0024-0.0034 vs null p95 0.00016-0.00097, pct 100 on both seeds at
    both N.
  * Every REAL cell with ESS >= 14 separated on both seeds (pct 97.8-100), INCLUDING the default
    cell (0.05, 0.005, N=400): pct 100 / 100, ESS 23.9 / 24.9. The alpha = 0.2 cells (ESS 7.7-9.0)
    are the boundary: 4 of 8 seed-cells separate (pct 86.8-98.0).
  * Negative control: 0 of 52 cell x N combos separated on either seed (max pct 92.5).
  * PR0 0.0, PR2 min cos 0.99999988, credits by N=400: 59 / 66, by N=1600: 248 / 245.

PRE-REGISTERED PREDICTION for SEEDS (written before they run):
  (P1) C1 PASS: every sub-ceiling real cell with ESS >= 14 separates on >= 4/5 seeds (pilot C1
       set: N=400 -> ESS <= ~31, N=1600 -> ESS <= ~123; the default cell is in the set).
  (P2) The crossover lies in ESS [8, 15]: the alpha = 0.2 cells separate on fewer seeds than the
       alpha <= 0.1 cells at the same (decay, N).
  (P3) C2 PASSES: the default cell (0.05, 0.005, N=400) separates on a majority. If so, 884c's
       at-chance reading of S at the default operating point (pct 78.0 / 63.0 / 86.5, K=1) was
       a single-control-draw instrument limitation, NOT an ESS ceiling. That corrects this chip's
       and its pre-flight's premise ("headroom is bounded by event count") and is the finding to
       carry forward, whatever else the sweep shows.
  (P4) PR4 holds: the negative control separates on no cell's majority.

CRITERIA (thresholds fixed here, not derived from the run):
  PR0 sense path         worst seed max |z_world(sense_flat) - z_world(act() twin)| over the
                         first EQUIV_TICKS ticks <= EQUIV_TOL (direction: upper)
  PR1 attainment         min over seeds of credited events by N=400        >= MIN_CREDITS_400
  PR2 C0 fidelity        min over seeds x cells x N of cos(replay, real GoalState parent)
                         >= C0_COS_FLOOR, max relative norm error <= C0_NORM_TOL, and the agent's
                         own parent equals the reference-cell GoalState's parent
  PR3 ceiling control    fraction of seeds where UNIFORM S separates at N=max(N_LIST)
                         >= SEP_FRAC  (positive control, SAME statistic and null as C1;
                         reachability asserted at setup against the pilot reference cells)
  PR4 negative control   max over all (cell, N) of fraction of seeds where the RANDOM-CREDIT arm
                         separates <= NC_MAX_FRAC   (direction: upper)
  C1  LOAD-BEARING       max over SUB-CEILING real cells (ess_mean <= C1_ESS_FRAC_MAX x mean
                         n_att at that N) of fraction of seeds where S separates >= SEP_FRAC.
                         The near-flat cells are excluded: they ARE the ceiling (red-team F1).
  C2  reported only      default cell (0.05, 0.005, N=400) separation fraction >= SEP_FRAC
                         (does the 884c at-chance reading replicate? PREDICTED NOT, see P3)
  C3  reported only      Spearman rho(cell ESS, cell mean percentile) over REAL cells >= 0.5
  "separates" for one seed-cell = K-averaged s_obs > the 95th percentile of its own K-averaged
  null (N_PERM samples). SEP_FRAC = 0.6 = 3 of 5 seeds (the WWA's "majority of seeds").

VERDICT GRID (every branch: evidence_direction non_contributory):
  PR0, PR1 or PR2 unmet -> FAIL, substrate_not_ready_requeue (harness / replay defect)
  PR4 unmet        -> FAIL, instrument_negative_control_fired (non_degenerate false)
  PR3 unmet        -> FAIL, substrate_not_ready_requeue (the event budget cannot carry the content
                      even at the ESS ceiling -- the precondition is not satisfiable in this harness)
  C1 met           -> PASS, parent_statistic_achievable_range_exists (the WWA precondition's
                      achievable-range clause is satisfiable; readout names WHERE)
  C1 unmet, an excluded (above-cap) cell separates
                   -> FAIL, parent_statistic_separates_only_above_c1_ess_cap (red-team pass 2 F4)
  C1 unmet, nothing separates
                   -> FAIL, parent_statistic_no_achievable_range_at_registered_settings

Power: at the ESS ceiling the pilot's uniform S sits 3-17x above its null p95; at the default
cell 1.7-2.3x. With 5 seeds the 3/5 majority rule's chance rate per cell is ~1e-3 at p = 0.05 per
seed-cell, so a C1 PASS is not a multiple-comparison artefact over 24 cells, and PR4 has ~52
correlated chances to false-flag (bounded above by ~6 %).

Smoke: --dry-run (1 seed, tiny budgets).  Pilot: --pilot --pilot-out <path> (seeds 101/102, full
budgets, no manifest, no outcome).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

# works from experiments/ and from the parked experiments/_scratch/ location alike
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ree_core").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.goal import GoalState  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-1093"
EXPERIMENT_TYPE = "v3_exq_1093_mech428_parent_stat_ess_sweep"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-428"]
DIRECTION = "non_contributory"

# --- environment / encoder (884c operating point, carried forward) ---
GRID_SIZE = 12
NUM_WAYPOINTS = 3
STAY_ACTION = 4
WORLD_DIM = 32
ALPHA_WORLD = 0.9
P0_EPISODES = 20
P0_STEPS_PER_EPISODE = 50
P0_PRESERVATION_WEIGHT = 1000.0

# --- sweep ---
N_LIST = [400, 1600]
ALPHAS = [0.20, 0.10, 0.05, 0.025, 0.01, 0.005]
DECAYS = [0.005, 0.0]
REF_ALPHA, REF_DECAY = 0.05, 0.005          # GoalConfig defaults; the agent's own GoalState
DEFAULT_CELL = (0.05, 0.005, 400)           # 884c's operating point (C2 replication)

# --- statistic ---
K_CTRL = 32
N_PERM = 400
NULL_Q = 0.95

# --- thresholds (pre-registered) ---
MIN_CREDITS_400 = 30
C0_COS_FLOOR = 0.99999
C0_NORM_TOL = 1e-4
SEP_FRAC = 0.6
NC_MAX_FRAC = 0.4
C3_RHO_FLOOR = 0.5
C1_ESS_FRAC_MAX = 0.5      # decision (b): C1 ranges only over cells with ESS <= this x n_att

EQUIV_TICKS = 30                            # act() twin check on the sense path (PR0)
EQUIV_TOL = 1e-6

# PR3 reference: pilot seeds 101/102, UNIFORM cell at N=1600 (s_obs, null_p95), this instrument.
PR3_REFERENCE_CELLS = [{"s_obs": 0.002828, "null_p95": 0.000162},
                       {"s_obs": 0.003028, "null_p95": 0.000285}]

SEEDS = [42, 43, 44, 45, 46]
PILOT_SEEDS = [101, 102]

SMOKE = {"n_list": [60, 120], "p0_episodes": 4, "p0_steps": 20, "k_ctrl": 4, "n_perm": 40,
         "min_credits": 1}


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _gen(tag: str) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32(tag.encode("ascii")))


def build_env(seed: int, max_steps: int) -> CausalGridWorld:
    env = CausalGridWorld(size=GRID_SIZE, num_hazards=0, num_resources=0, subgoal_mode=True,
                          num_waypoints=NUM_WAYPOINTS, seed=seed, max_episode_steps=max_steps,
                          subgoal_arrival_position_check=True,
                          hazard_free_contamination_gate=True)
    assert env.subgoal_arrival_position_check is True
    assert float(env.contamination_spread) == 0.0
    return env


def build_agent(env: CausalGridWorld) -> REEAgent:
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, world_dim=WORLD_DIM,
                              z_goal_enabled=True, use_world_encoder_skip=True,
                              alpha_world=ALPHA_WORLD, use_hierarchical_goal_credit=True,
                              parent_goal_alpha=REF_ALPHA, parent_goal_decay=REF_DECAY,
                              parent_goal_weight=0.0)
    assert abs(float(cfg.latent.alpha_world) - ALPHA_WORLD) < 1e-12, "alpha_world swallowed"
    agent = REEAgent(cfg)
    gc = agent.goal_state.config
    assert gc.use_hierarchical_goal_credit is True, "use_hierarchical_goal_credit swallowed"
    assert abs(float(gc.parent_goal_alpha) - REF_ALPHA) < 1e-12, "parent_goal_alpha swallowed"
    assert abs(float(gc.parent_goal_decay) - REF_DECAY) < 1e-12, "parent_goal_decay swallowed"
    assert float(agent.config.e3.parent_goal_weight) == 0.0, "consumer must stay OFF"
    return agent


def scripted_action(env: CausalGridWorld) -> int:
    sub = env.get_subgoal_state()
    idx = sub["next_waypoint_idx"]
    if not env.waypoints or idx >= len(env.waypoints):
        return STAY_ACTION
    wx, wy = env.waypoints[idx]
    ax, ay = env.get_agent_position()
    if ax != wx:
        return 1 if wx > ax else 0
    if ay != wy:
        return 3 if wy > ay else 2
    return STAY_ACTION


# ----------------------------------------------------------------------------- statistic

def ema_weights(positions: List[int], t_end: int, alpha: float, decay: float) -> torch.Tensor:
    """Weights of parent_T = sum_k w_k z_k for credits at `positions` (ticks <= t_end)."""
    n = len(positions)
    pos = torch.tensor(positions, dtype=torch.float64)
    k = torch.arange(n, dtype=torch.float64)
    a = min(1.0, alpha)
    return (a * (1.0 - a) ** (n - 1 - k) * (1.0 - decay) ** (t_end - pos)).float()


def uniform_weights(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.float32)


def ess(w: torch.Tensor) -> float:
    w = w.double()
    return float(w.sum() ** 2 / (w * w).sum())


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def _pad(rows: List[List[int]]):
    """Pad per-position candidate index lists into [n, L] (+ lengths)."""
    L = max(len(c) for c in rows)
    t = torch.zeros((len(rows), L), dtype=torch.long)
    for k, c in enumerate(rows):
        t[k, :len(c)] = torch.tensor(c, dtype=torch.long)
    return t, torch.tensor([len(c) for c in rows], dtype=torch.long)


def _pick(cands: torch.Tensor, lens: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """k independent draws per position -> [k, n] tick indices."""
    u = torch.rand((k, cands.shape[0]), generator=gen)
    slot = (u * lens).long().clamp_max(lens - 1)
    return cands[torch.arange(cands.shape[0]), slot]


def local_windows(ticks: List[int], n_ticks: int, ctrl_ok) -> tuple:
    """Per credit tick (in credit order), the control candidates: ticks strictly between its
    previous and next neighbour in the credited set (by time), excluding every credited tick and
    every tick failing ctrl_ok. Parameter-free -- the window is set by the event structure. If
    empty, it widens by one more neighbour on each side (counted)."""
    srt = sorted(ticks)
    rank = {t: i for i, t in enumerate(srt)}
    tset = set(ticks)
    out, expansions = [], 0
    for t in ticks:
        i, e = rank[t], 1
        while True:
            lo = srt[i - e] if i - e >= 0 else -1
            hi = srt[i + e] if i + e < len(srt) else n_ticks
            win = [j for j in range(lo + 1, hi) if j not in tset and ctrl_ok(j)]
            if win or (lo == -1 and hi == n_ticks):
                break
            e += 1
            expansions += 1
        assert win, "no control candidate for a credited tick"
        out.append(win)
    return out, expansions


def shifted_ticks(ticks: List[int], delta: int, n_ticks: int, forbidden: set) -> List[int]:
    """Circular time-shift of the credited ticks by delta, keeping credit ORDER (so weight w_k
    stays with position k). A shifted tick landing on a forbidden (attained) or already-used
    tick is bumped forward to the next free one."""
    used, out = set(), []
    for t in ticks:
        s = (t + delta) % n_ticks
        while s in forbidden or s in used:
            s = (s + 1) % n_ticks
        used.add(s)
        out.append(s)
    return out


def arm_draws(true_ticks: List[int], n_ticks: int, attained: set, ctrl_ok, gen: torch.Generator,
              k_ctrl: int, n_perm: int, mode: str) -> dict:
    """Index draws for one arm, shared by every weight vector (common random numbers).

    mode "shift" (PRIMARY): observed controls are time-local (local_windows); each null sample
    circularly shifts the whole credited set by a random delta in [1, n_ticks-1] (off attained
    ticks) and draws its controls by the SAME window rule around the shifted set -- so observed
    and null have identical event geometry, recency structure and control locality, and differ
    only in WHICH ticks are credited.
    mode "unmatched" (REPORTED ONLY, the 884c probe form): any eligible tick of the run, at any
    position, for controls and for the null's pseudo-credited set.
    """
    n = len(true_ticks)
    if mode == "shift":
        wins, exp_ = local_windows(true_ticks, n_ticks, lambda j: ctrl_ok(j) and j not in attained)
        c, l = _pad(wins)
        obs_ctrl = _pick(c, l, k_ctrl, gen)
        null_true, null_ctrl, bumped = [], [], []
        for _ in range(n_perm):
            delta = int(torch.randint(1, n_ticks, (1,), generator=gen))
            st = shifted_ticks(true_ticks, delta, n_ticks, attained)
            # red-team F2 confirmer: fraction of pseudo-credits moved off an attained/used tick
            bumped.append(sum(1 for t, u in zip(true_ticks, st) if u != (t + delta) % n_ticks) / n)
            w2, _e = local_windows(st, n_ticks, lambda j: ctrl_ok(j) and j not in attained)
            c2, l2 = _pad(w2)
            null_true.append(torch.tensor(st, dtype=torch.long))
            null_ctrl.append(_pick(c2, l2, k_ctrl, gen))
        return {"true": torch.tensor(true_ticks, dtype=torch.long), "obs_ctrl": obs_ctrl,
                "null_true": torch.stack(null_true), "null_ctrl": torch.stack(null_ctrl),
                "window_expansions": exp_, "window_len_min": int(l.min()),
                "null_bump_frac_mean": sum(bumped) / len(bumped),
                "null_bump_frac_max": max(bumped)}
    tset = set(true_ticks)
    ctrl_pool = [j for j in range(n_ticks) if j not in tset and j not in attained and ctrl_ok(j)]
    null_pool = [j for j in range(n_ticks) if j not in attained and ctrl_ok(j)]
    cp = torch.tensor(ctrl_pool, dtype=torch.long)
    npl = torch.tensor(null_pool, dtype=torch.long)
    obs_ctrl = cp[torch.randint(len(ctrl_pool), (k_ctrl, n), generator=gen)]
    # red-team F3: each null sample's controls come from the pool MINUS its own pseudo-credited
    # ticks (as the observed controls exclude the credited set) -- otherwise the null parent and
    # control share members, the 884c F1 self-overlap shape on the null side only.
    null_true, null_ctrl = [], []
    for _ in range(n_perm):
        perm = torch.randperm(len(null_pool), generator=gen)
        rest = npl[perm[n:]]
        null_true.append(npl[perm[:n]])
        null_ctrl.append(rest[torch.randint(len(rest), (k_ctrl, n), generator=gen)])
    null_true, null_ctrl = torch.stack(null_true), torch.stack(null_ctrl)
    return {"true": torch.tensor(true_ticks, dtype=torch.long), "obs_ctrl": obs_ctrl,
            "null_true": null_true, "null_ctrl": null_ctrl,
            "window_expansions": 0, "window_len_min": len(ctrl_pool),
            "null_bump_frac_mean": 0.0, "null_bump_frac_max": 0.0}


def separates(s_obs: float, null_p95: float) -> bool:
    """THE per-seed-cell predicate: K-averaged s_obs above its own null's 95th percentile."""
    return bool(s_obs > null_p95)


def s_eval(Z: torch.Tensor, dr: dict, weights: Dict[str, torch.Tensor], chunk: int = 25) -> dict:
    """K-averaged S = 1 - cos(parent_credited, parent_control) for every weight vector, observed
    and null, from one set of draws."""
    out = {}
    p_true = {k: w @ Z[dr["true"]] for k, w in weights.items()}
    Zc = Z[dr["obs_ctrl"]]                                                 # [K, n, D]
    for k, w in weights.items():
        pc = torch.einsum("n,knd->kd", w, Zc)
        s_obs = float((1.0 - _cos_rows(p_true[k].expand_as(pc), pc)).mean())
        out[k] = {"s_obs": s_obs, "_null": []}
    P = dr["null_true"].shape[0]
    for s0 in range(0, P, chunk):
        Za = Z[dr["null_true"][s0:s0 + chunk]]                             # [r, n, D]
        Zb = Z[dr["null_ctrl"][s0:s0 + chunk]]                             # [r, K, n, D]
        for k, w in weights.items():
            pa = torch.einsum("n,rnd->rd", w, Za)
            pb = torch.einsum("n,rknd->rkd", w, Zb)
            s = (1.0 - _cos_rows(pa.unsqueeze(1).expand_as(pb), pb)).mean(dim=1)
            out[k]["_null"].extend(float(v) for v in s)
    for k in out:
        null = sorted(out[k].pop("_null"))
        p95 = null[min(len(null) - 1, int(NULL_Q * len(null)))]
        s_obs = out[k]["s_obs"]
        out[k].update({"null_med": null[len(null) // 2], "null_p95": p95,
                       "pct": 100.0 * sum(1 for v in null if v < s_obs) / len(null),
                       "separates": separates(s_obs, p95), "effect_over_p95": s_obs - p95})
    return out


# ----------------------------------------------------------------------------- rollout

def rollout(seed: int, n_list: List[int], p0_eps: int, p0_steps: int, zg) -> dict:
    n_max = max(n_list)
    warm_env = build_env(seed, n_max)
    master = build_agent(warm_env)
    p0 = run_zworld_p0(master, warm_env, seed=seed, episodes=p0_eps,
                       steps_per_episode=p0_steps, policy=RandomPolicy(seed),
                       label=f"exq1093_s{seed}", dry_run=False,
                       config=ZWorldP0Config(preservation_weight=P0_PRESERVATION_WEIGHT))
    trained = {k: v.detach().clone() for k, v in master.latent_stack.state_dict().items()}
    print(f"  seed={seed} P0 warmup done ({p0_eps} episodes)", flush=True)

    env = build_env(seed, n_max)
    agent = build_agent(env)
    agent.latent_stack.load_state_dict(trained)
    # One REAL GoalState per (alpha, decay) cell, driven with the same call sequence the agent
    # issues (update() once per tick with benefit 0, then credit on attainment ticks).
    real_gs: Dict[tuple, GoalState] = {}
    for a in ALPHAS:
        for d in DECAYS:
            gcfg = dataclasses.replace(agent.goal_state.config, parent_goal_alpha=a,
                                       parent_goal_decay=d, use_hierarchical_goal_credit=True)
            real_gs[(a, d)] = GoalState(gcfg, torch.device("cpu"))
    snap_real: Dict[int, Dict[tuple, torch.Tensor]] = {}
    snap_agent: Dict[int, torch.Tensor] = {}

    # SENSE PATH. The policy is scripted, so only z_world is needed: sense_flat() is the first
    # statement of act() and nothing after it in act() (E1 tick, trajectory generation, E3
    # selection, replay) writes the latent stack. PR0 re-proves that per seed: a twin agent with
    # the same weights runs the full act() for the first EQUIV_TICKS ticks of the same walk and
    # its z_world must match. act() costs ~1.7 s/tick on a contended 2-core box vs ~9 ms.
    twin_env = build_env(seed, n_max)
    twin = build_agent(twin_env)
    # FULL agent state, not only latent_stack: sense() also runs the agent-level
    # world_obs_encoder / body_obs_encoder (random Linear+ReLU pre-projections outside
    # latent_stack -- SD-ZWORLD-SENSE-PATH-PARITY), so a twin with its own init would differ.
    twin.load_state_dict(agent.state_dict())
    twin_obs, _ = twin_env.reset()
    twin.act(twin_obs)
    equiv_max = 0.0

    obs_flat, _ = env.reset()
    agent.sense_flat(obs_flat)
    equiv_max = max(equiv_max, float((agent._current_latent.z_world
                                      - twin._current_latent.z_world).abs().max()))
    prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()
    reprs, attained, n_credits_agent = [], [], 0
    steps = 0
    for _ in range(n_max):
        action = scripted_action(env)
        obs_flat, _h, done, info, _od = env.step(action)
        steps += 1
        agent.sense_flat(obs_flat)
        if steps <= EQUIV_TICKS:
            assert scripted_action(twin_env) == action, "twin walk diverged"
            twin_obs, _t1, _t2, _t3, _t4 = twin_env.step(action)
            twin.act(twin_obs)
            equiv_max = max(equiv_max, float((agent._current_latent.z_world
                                              - twin._current_latent.z_world).abs().max()))
        z_now = agent._current_latent.z_world.detach().clone()
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
        for gs in real_gs.values():
            gs.update(z_now, benefit_exposure=0.0, drive_level=0.0)
        ttype = info.get("transition_type", "none")
        res = agent.notify_subgoal_attainment(ttype, child_representation=prev_z)
        is_att = ttype in ("waypoint", "sequence_complete")
        if res:
            n_credits_agent = int(res["n_subgoal_credits"])
        if is_att:
            for gs in real_gs.values():
                gs.credit_subgoal_attainment(prev_z)
        reprs.append(prev_z)
        attained.append(is_att)
        if steps in n_list:
            snap_real[steps] = {c: gs.z_goal_parent.detach().clone().reshape(-1).float()
                                for c, gs in real_gs.items() if gs.z_goal_parent is not None}
            if agent.goal_state.z_goal_parent is not None:
                snap_agent[steps] = agent.goal_state.z_goal_parent.detach().clone().reshape(-1)
        prev_z = z_now.reshape(-1).float()
        if done:
            break
    zg.observe(agent)
    print(f"  seed={seed} rollout done steps={steps} "
          f"credits={n_credits_agent}", flush=True)
    del twin, twin_env
    return {"Z": torch.stack(reprs), "attained": torch.tensor(attained), "steps": steps,
            "equiv_max_abs_diff": equiv_max, "equiv_ticks": min(EQUIV_TICKS, steps),
            "n_credits_agent": n_credits_agent, "snap_real": snap_real,
            "snap_agent": snap_agent, "p0": {k: v for k, v in (p0 or {}).items()
                                              if isinstance(v, (int, float, str, bool))}}


def analyse_seed(seed: int, ro: dict, n_list: List[int], k_ctrl: int, n_perm: int) -> dict:
    Z, att = ro["Z"], ro["attained"]
    out = {"seed": seed, "steps": ro["steps"], "n_credits_agent": ro["n_credits_agent"],
           "equiv_max_abs_diff": ro["equiv_max_abs_diff"], "equiv_ticks": ro["equiv_ticks"],
           "cells": [], "uniform": {}, "c0": [], "nc": {}, "unmatched": {}, "n_att": {},
           "window_expansions": {}, "window_len_min": {}}
    for N in n_list:
        if N > ro["steps"]:
            continue
        att_l = att[:N].tolist()
        att_idx = [i for i in range(N) if att_l[i]]
        non_idx = [i for i in range(N) if not att_l[i]]
        attained = set(att_idx)
        n_a = len(att_idx)
        out["n_att"][str(N)] = n_a
        if n_a < 2 or len(non_idx) < 2 * n_a:
            continue
        # negative control: random non-attained ticks, scored on their OWN partition
        g_nc = _gen(f"exq1093:nc_pick:{seed}:{N}")
        pick = sorted(torch.tensor(non_idx)[torch.randperm(len(non_idx), generator=g_nc)[:n_a]]
                      .tolist())

        # agent vs reference-cell GoalState, and replay vs every real GoalState
        true_idx = torch.tensor(att_idx, dtype=torch.long)
        if N in ro["snap_agent"] and (REF_ALPHA, REF_DECAY) in ro["snap_real"].get(N, {}):
            ag = ro["snap_agent"][N]
            rf = ro["snap_real"][N][(REF_ALPHA, REF_DECAY)]
            out["c0"].append({"N": N, "cell": "agent_vs_ref_goalstate",
                              "cos": float(_cos_rows(ag, rf)),
                              "rel_norm_err": float((ag - rf).norm() / rf.norm().clamp_min(1e-12))})
        w_true, w_nc = {}, {}
        for a in ALPHAS:
            for d in DECAYS:
                key = f"a{a}_d{d}"
                w_true[key] = ema_weights([i + 1 for i in att_idx], N, a, d)
                w_nc[key] = ema_weights([i + 1 for i in pick], N, a, d)
                real = ro["snap_real"].get(N, {}).get((a, d))
                if real is not None:
                    rep = w_true[key] @ Z[true_idx]
                    out["c0"].append({"N": N, "cell": key, "cos": float(_cos_rows(rep, real)),
                                      "rel_norm_err": float((rep - real).norm()
                                                            / real.norm().clamp_min(1e-12))})
        w_true["uniform"] = uniform_weights(n_a)
        w_nc["uniform"] = uniform_weights(n_a)

        yes = lambda j: True  # noqa: E731
        dr_t = arm_draws(att_idx, N, attained, yes, _gen(f"exq1093:S:{seed}:{N}"), k_ctrl, n_perm,
                         "shift")
        dr_n = arm_draws(pick, N, attained, yes, _gen(f"exq1093:NC:{seed}:{N}"), k_ctrl, n_perm,
                         "shift")
        um_t = arm_draws(att_idx, N, attained, yes, _gen(f"exq1093:UM:{seed}:{N}"), k_ctrl, n_perm,
                         "unmatched")
        um_n = arm_draws(pick, N, attained, yes, _gen(f"exq1093:UMNC:{seed}:{N}"), k_ctrl, n_perm,
                         "unmatched")
        out["window_expansions"][str(N)] = dr_t["window_expansions"] + dr_n["window_expansions"]
        out["window_len_min"][str(N)] = min(dr_t["window_len_min"], dr_n["window_len_min"])
        out.setdefault("null_bump_frac", {})[str(N)] = {
            "true_mean": dr_t["null_bump_frac_mean"], "true_max": dr_t["null_bump_frac_max"],
            "nc_mean": dr_n["null_bump_frac_mean"], "nc_max": dr_n["null_bump_frac_max"]}
        r_t, r_n = s_eval(Z, dr_t, w_true), s_eval(Z, dr_n, w_nc)
        u_t, u_n = s_eval(Z, um_t, w_true), s_eval(Z, um_n, w_nc)
        for a in ALPHAS:
            for d in DECAYS:
                key = f"a{a}_d{d}"
                out["cells"].append({"alpha": a, "decay": d, "N": N, "ess": ess(w_true[key]),
                                     **r_t[key]})
                out["nc"][f"{key}_N{N}"] = {"ess": ess(w_nc[key]), **r_n[key]}
                out["unmatched"][f"{key}_N{N}"] = {"true": u_t[key], "nc": u_n[key]}
        out["uniform"][str(N)] = {"ess": float(n_a), **r_t["uniform"]}
        out["nc"][f"uniform_N{N}"] = {"ess": float(n_a), **r_n["uniform"]}
        out["unmatched"][f"uniform_N{N}"] = {"true": u_t["uniform"], "nc": u_n["uniform"]}
    return out


# ----------------------------------------------------------------------------- aggregation

def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else None


def _flat(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v if v == v and abs(v) != float("inf") else None
    return None


def build_manifest(per_seed: List[dict], n_list: List[int], smoke: bool, min_credits: int) -> dict:
    n_seeds = len(per_seed)
    n_max = max(n_list)
    # ---- per-cell aggregation over seeds
    cell_keys = [(a, d, N) for N in n_list for a in ALPHAS for d in DECAYS]
    cell_summary = []
    for (a, d, N) in cell_keys:
        rows = [c for s in per_seed for c in s["cells"] if (c["alpha"], c["decay"], c["N"]) == (a, d, N)]
        if not rows:
            continue
        cell_summary.append({
            "alpha": a, "decay": d, "N": N,
            "sep_frac": sum(r["separates"] for r in rows) / n_seeds,
            "n_seeds_scored": len(rows),
            "ess_mean": statistics.fmean(r["ess"] for r in rows),
            "pct_mean": statistics.fmean(r["pct"] for r in rows),
            "s_obs_mean": statistics.fmean(r["s_obs"] for r in rows),
            "null_p95_mean": statistics.fmean(r["null_p95"] for r in rows),
        })
    # Decision (b), chip-20260924-decision-exq1093-c1-pr3-identity: C1 ranges ONLY over real
    # cells whose ESS <= C1_ESS_FRAC_MAX x n_att (mean over seeds at that N). The near-flat cells
    # (alpha 0.005/0.01 at decay 0) ARE the uniform ceiling (red-team F1), so including them made
    # C1 a restatement of PR3; excluding them makes C1 ask whether S separates at a NON-ceiling
    # SD-092 setting, and PR3 stays the readiness control.
    n_att_mean = {N: statistics.fmean(s["n_att"].get(str(N), 0) for s in per_seed) for N in n_list}
    for c in cell_summary:
        c["c1_ess_cap"] = C1_ESS_FRAC_MAX * n_att_mean[c["N"]]
        c["c1_eligible"] = bool(c["ess_mean"] <= c["c1_ess_cap"])
    c1_cells = [c for c in cell_summary if c["c1_eligible"]]
    # red-team pass 2 F5: name the cell that BINDS -- ties broken toward the highest ESS
    best = (max(c1_cells, key=lambda c: (c["sep_frac"], c["pct_mean"], c["ess_mean"]))
            if c1_cells else None)
    c1_measured = best["sep_frac"] if best else 0.0
    c1_pass = bool(c1_cells) and c1_measured >= SEP_FRAC
    # red-team pass 2 F4: list separating cells over ALL real cells, flagged by C1 membership
    sep_cells = [{"cell": f"a{c['alpha']}_d{c['decay']}_N{c['N']}", "ess_mean": c["ess_mean"],
                  "sep_frac": c["sep_frac"], "in_c1_set": c["c1_eligible"]}
                 for c in cell_summary if c["sep_frac"] >= SEP_FRAC]
    excluded_separates = any(c["sep_frac"] >= SEP_FRAC for c in cell_summary
                             if not c["c1_eligible"])
    sep_in_set = sorted(c["ess_mean"] for c in c1_cells if c["sep_frac"] >= SEP_FRAC)

    # ---- preconditions
    n_att_400 = [s["n_att"].get(str(min(n_list)), 0) for s in per_seed]
    pr1_measured = min(n_att_400) if n_att_400 else 0
    pr1 = pr1_measured >= min_credits
    c0_rows = [dict(r, seed=s["seed"]) for s in per_seed for r in s["c0"]]
    c0_min = min((r["cos"] for r in c0_rows), default=0.0)
    c0_err = max((r["rel_norm_err"] for r in c0_rows), default=float("inf"))
    worst_c0 = min(c0_rows, key=lambda r: r["cos"]) if c0_rows else None
    n_c0_expected = n_seeds * len(n_list) * (len(ALPHAS) * len(DECAYS) + 1)
    pr2 = (c0_min >= C0_COS_FLOOR and c0_err <= C0_NORM_TOL and len(c0_rows) == n_c0_expected)
    uni = [s["uniform"].get(str(n_max)) for s in per_seed]
    pr3_measured = sum(1 for u in uni if u and u["separates"]) / n_seeds
    pr3 = pr3_measured >= SEP_FRAC
    nc_keys = sorted({k for s in per_seed for k in s["nc"]})
    nc_frac = {k: sum(1 for s in per_seed if s["nc"].get(k, {}).get("separates")) / n_seeds
               for k in nc_keys}
    worst_nc = max(nc_frac.items(), key=lambda kv: kv[1]) if nc_frac else ("none", 0.0)
    pr4 = worst_nc[1] <= NC_MAX_FRAC

    # reported only: the 884c UNMATCHED control/null form, true arm and its negative control
    um_keys = sorted({k for s in per_seed for k in s.get("unmatched", {})})
    um_true_frac = {k: sum(1 for s in per_seed if s["unmatched"].get(k, {}).get("true", {})
                           .get("separates")) / n_seeds for k in um_keys}
    um_nc_frac = {k: sum(1 for s in per_seed if s["unmatched"].get(k, {}).get("nc", {})
                         .get("separates")) / n_seeds for k in um_keys}

    default = next((c for c in cell_summary if (c["alpha"], c["decay"], c["N"]) == DEFAULT_CELL), None)
    c2_measured = default["sep_frac"] if default else None
    # REPORTED, not load-bearing: the 884c at-chance reading at its default operating point.
    # Pre-registered prediction P3: it does NOT replicate (separates on a majority).
    c2_pass = c2_measured is not None and c2_measured >= SEP_FRAC
    rho = _spearman([c["ess_mean"] for c in cell_summary], [c["pct_mean"] for c in cell_summary])
    c3_pass = rho is not None and rho >= C3_RHO_FLOOR

    eq_vals = [s["equiv_max_abs_diff"] for s in per_seed]
    pr0_measured = max(eq_vals) if eq_vals else float("inf")
    pr0 = pr0_measured <= EQUIV_TOL and all(s["equiv_ticks"] >= 1 for s in per_seed)

    if not (pr0 and pr1 and pr2):
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not pr4:
        outcome, label = "FAIL", "instrument_negative_control_fired"
    elif not pr3:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif c1_pass:
        outcome, label = "PASS", "parent_statistic_achievable_range_exists"
    elif excluded_separates:
        # red-team pass 2 F4: separation exists, but only above the C1 ESS cap
        outcome, label = "FAIL", "parent_statistic_separates_only_above_c1_ess_cap"
    else:
        outcome, label = "FAIL", "parent_statistic_no_achievable_range_at_registered_settings"
    ready = pr0 and pr1 and pr2 and pr3 and pr4
    c1_non_degenerate = bool(ready and c1_cells
                             and len({round(c["s_obs_mean"], 9) for c in cell_summary}) > 1
                             and all(c["null_p95_mean"] > 0.0 for c in cell_summary))

    criteria = [
        {"name": "C1_subceiling_real_cell_separates_majority", "load_bearing": True,
         "passed": c1_pass, "measured": c1_measured, "threshold": SEP_FRAC, "comparator": ">=",
         "cell_set": f"real cells with ess_mean <= {C1_ESS_FRAC_MAX} x mean n_att at that N",
         "n_cells_in_set": len(c1_cells),
         "best_cell": (f"a{best['alpha']}_d{best['decay']}_N{best['N']}" if best else None)},
        {"name": "C2_default_cell_separates_reported", "load_bearing": False, "passed": c2_pass,
         "measured": c2_measured, "threshold": SEP_FRAC, "comparator": ">=",
         "note": "884c at-chance replication check at (0.05, 0.005, N=400); predicted to separate "
                 "(P3). Reported only."},
        {"name": "C3_ess_orders_separation", "load_bearing": False, "passed": c3_pass,
         "measured": rho, "threshold": C3_RHO_FLOOR, "comparator": ">="},
    ]
    preconditions = [
        {"name": "PR0_sense_path_equals_act_path_max_abs_diff", "measured": pr0_measured,
         "threshold": EQUIV_TOL, "direction": "upper",
         "control": "worst seed: max |z_world(sense_flat) - z_world(act twin)| over the first "
                    f"{EQUIV_TICKS} ticks of the same scripted walk with the same weights "
                    "(the 884c probes used act(); this run uses sense_flat())", "met": pr0},
        {"name": "PR1_attainment_credits_by_N400", "measured": pr1_measured,
         "threshold": min_credits, "direction": "lower",
         "control": "worst seed's count of credited (pre-arrival) events by the first N; the "
                    "attainment-rate clause of the WWA precondition", "met": pr1},
        {"name": "PR2_c0_replay_fidelity_min_cos", "measured": c0_min, "threshold": C0_COS_FLOOR,
         "direction": "lower", "offending_cell": worst_c0,
         "control": "cos(w @ Z_credited, parent of a REAL GoalState driven with the agent's call "
                    "sequence), worst over seed x cell x N, plus agent-vs-reference GoalState; "
                    f"also requires max rel norm err <= {C0_NORM_TOL} (measured {c0_err:.3g}) and "
                    f"{n_c0_expected} checks present (got {len(c0_rows)})", "met": pr2},
        {"name": "PR3_uniform_ceiling_S_separates_frac", "measured": pr3_measured,
         "threshold": SEP_FRAC, "direction": "lower",
         "control": "positive control of the SAME statistic S and SAME re-constructed null, at "
                    "the maximum achievable ESS (uniform weights over all credits by N=%d)" % n_max,
         "met": pr3},
        {"name": "PR4_random_credit_negative_control_max_sep_frac", "measured": worst_nc[1],
         "threshold": NC_MAX_FRAC, "direction": "upper", "offending_cell": worst_nc[0],
         "control": "random non-attained ticks credited at their own positions, scored on their "
                    "OWN partition (884c F2/F4 fixed); must not separate on a majority anywhere",
         "met": pr4},
    ]

    readout = {
        "C1_max_sep_frac_real_cells": c1_measured, "C1_passed": c1_pass,
        "C2_default_cell_sep_frac": c2_measured, "C1_n_cells_in_set": len(c1_cells),
        "C1_min_separating_ess_in_set": sep_in_set[0] if sep_in_set else None,
        "any_excluded_cell_separates": excluded_separates, "C3_spearman_ess_pct": rho,
        "PR0_sense_act_max_abs_diff": pr0_measured,
        "PR1_min_credits_N400": pr1_measured, "PR2_c0_min_cos": c0_min,
        "PR2_c0_max_rel_norm_err": c0_err, "PR3_uniform_sep_frac": pr3_measured,
        "PR4_nc_max_sep_frac": worst_nc[1], "n_real_cells_separating": len(sep_cells),
        "ready": ready,
        "unmatched_nc_max_sep_frac": max(um_nc_frac.values()) if um_nc_frac else None,
        "unmatched_true_max_sep_frac": max(um_true_frac.values()) if um_true_frac else None,
        "window_expansions_total": sum(sum(s["window_expansions"].values()) for s in per_seed),
        "window_len_min": min((v for s in per_seed for v in s["window_len_min"].values()),
                              default=None),
    }
    for c in cell_summary:
        key = f"a{c['alpha']}_d{c['decay']}_N{c['N']}"
        readout[f"sep_frac_{key}"] = c["sep_frac"]
        readout[f"ess_{key}"] = c["ess_mean"]
        readout[f"pct_mean_{key}"] = c["pct_mean"]
    for N in n_list:
        us = [s["uniform"].get(str(N)) for s in per_seed]
        us = [u for u in us if u]
        if us:
            readout[f"uniform_sep_frac_N{N}"] = sum(u["separates"] for u in us) / n_seeds
            readout[f"uniform_ess_mean_N{N}"] = statistics.fmean(u["ess"] for u in us)
    readout = {k: _flat(v) for k, v in readout.items() if _flat(v) is not None}

    manifest = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_stamp(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": DIRECTION,
        "evidence_direction_per_claim": {"MECH-428": DIRECTION},
        "criteria": criteria,
        "combination_rule": "PASS iff PR0 AND PR1 AND PR2 AND PR3 AND PR4 AND C1 (C1 is the only "
                            "load-bearing criterion). C2 and C3 are reported, never gating. "
                            "Every branch is non_contributory for MECH-428.",
        "scope_note": "Instrument / non-degeneracy diagnostic ONLY. A PASS says the WWA "
                      "precondition's achievable-range clause is satisfiable at the named "
                      "(alpha, decay, N) cells. It does NOT test MECH-428's CONFIRMING clause: "
                      "no live consumer (parent_goal_weight = 0), scripted policy, no "
                      "NO-SUBGOAL or forced-seed arm. Behavioural leg: EXP-0710 / GFLAG-0464. "
                      "C1 is a MAX over the sub-ceiling set, so it binds on the highest-ESS "
                      "eligible cell (red-team pass 2 F5): a PASS says the separation margin "
                      "survives down to ESS <= 0.5 x n_att, not that every sub-ceiling cell "
                      "separates -- the crossover lives in the reported cell map "
                      "(readout C1_min_separating_ess_in_set, sep_frac_*) and C2/C3.",
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1": c1_non_degenerate},
            "separating_real_cells": sep_cells,
        },
        "readout": readout,
        "cell_summary": cell_summary,
        "negative_control_sep_frac": nc_frac,
        "unmatched_884c_form_reported_only": {
            "note": "the 884c probe's control/null drew any tick of the run at any position; "
                    "with recency weighting this is confounded by temporal structure in z_world "
                    "(waypoint layout persists within a sequence). Reported, never gating.",
            "true_sep_frac": um_true_frac, "nc_sep_frac": um_nc_frac},
        "per_seed": [{k: v for k, v in s.items() if k != "_row"} for s in per_seed],
        "arm_results": [s["_row"] for s in per_seed if "_row" in s],
        "thresholds": {"MIN_CREDITS_400": min_credits, "C0_COS_FLOOR": C0_COS_FLOOR,
                       "C0_NORM_TOL": C0_NORM_TOL, "SEP_FRAC": SEP_FRAC,
                       "NC_MAX_FRAC": NC_MAX_FRAC, "C3_RHO_FLOOR": C3_RHO_FLOOR,
                       "NULL_Q": NULL_Q},
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "smoke": smoke,
        "notes": "chip-20260917-mech428-parent-goal-alpha-sweep; authority "
                 "exq884c_mech428_c1_ema_self_overlap_redteam_blocking_20260916.md.",
    }
    if not c1_non_degenerate:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = ("preconditions unmet or C1 statistic flat across cells"
                                         if not ready else "C1 statistic flat across cells")
    return manifest


# ----------------------------------------------------------------------------- main

def main(mode: str, pilot_out: Optional[str]):
    t0 = time.perf_counter()
    if mode == "smoke":
        seeds, n_list = SEEDS[:1], SMOKE["n_list"]
        p0e, p0s, kc, npm, minc = (SMOKE["p0_episodes"], SMOKE["p0_steps"], SMOKE["k_ctrl"],
                                   SMOKE["n_perm"], SMOKE["min_credits"])
    else:
        seeds = PILOT_SEEDS if mode == "pilot" else SEEDS
        n_list, p0e, p0s, kc, npm, minc = (N_LIST, P0_EPISODES, P0_STEPS_PER_EPISODE, K_CTRL,
                                           N_PERM, MIN_CREDITS_400)
    full_config = {"grid_size": GRID_SIZE, "num_waypoints": NUM_WAYPOINTS, "world_dim": WORLD_DIM,
                   "alpha_world": ALPHA_WORLD, "p0_episodes": p0e, "p0_steps": p0s,
                   "p0_preservation_weight": P0_PRESERVATION_WEIGHT, "n_list": n_list,
                   "alphas": ALPHAS, "decays": DECAYS, "ref_cell": [REF_ALPHA, REF_DECAY],
                   "k_ctrl": kc, "n_perm": npm, "credit_representation": "pre_arrival_prev_z_world",
                   "parent_goal_weight": 0.0, "policy": "scripted_waypoint_walk"}
    anchor = assert_anchor_reachable(
        anchor_name="PR3_uniform_ceiling_S_separates_frac",
        reference_cells=PR3_REFERENCE_CELLS,
        score_fn=lambda c: separates(c["s_obs"], c["null_p95"]),
        threshold=SEP_FRAC,
        reference_source="--pilot seeds 101/102, uniform N=1600, 2026-09-24 (driver docstring PILOT)")
    zg = ZGoalStreamAccumulator()
    per_seed = []
    for seed in seeds:
        print(f"Seed {seed} Condition parent_stat_ess_sweep", flush=True)
        with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
            ro = rollout(seed, n_list, p0e, p0s, zg)
            res = analyse_seed(seed, ro, n_list, kc, npm)
            row = {"seed": seed, "steps": res["steps"], "n_credits_agent": res["n_credits_agent"],
                   "n_att": res["n_att"], "p0": ro["p0"]}
            cell.stamp(row)
        res["_row"] = row
        per_seed.append(res)
        ok = any(c["separates"] for c in res["cells"])
        print(f"verdict: {'PASS' if ok else 'FAIL'}", flush=True)

    if mode == "pilot":
        Path(pilot_out).parent.mkdir(parents=True, exist_ok=True)
        Path(pilot_out).write_text(json.dumps({"per_seed": [{k: v for k, v in s.items() if k != "_row"}
                                                            for s in per_seed]}, indent=1))
        print(f"pilot written to {pilot_out} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return None

    manifest = build_manifest(per_seed, n_list, smoke=(mode == "smoke"), min_credits=minc)
    manifest["interpretation"]["anchor_reachability"] = anchor
    print(f"=== {QUEUE_ID} {EXPERIMENT_TYPE} ===", flush=True)
    print(f"outcome: {manifest['outcome']} label={manifest['interpretation']['label']} "
          f"C1={manifest['criteria'][0]['measured']} vs {SEP_FRAC}", flush=True)
    for p in manifest["interpretation"]["preconditions"]:
        print(f"  {p['name']}: measured={p['measured']} threshold={p['threshold']} met={p['met']}",
              flush=True)
    out_path = write_flat_manifest(manifest, None, dry_run=(mode == "smoke"), config=full_config,
                                   seeds=list(seeds), script_path=Path(__file__), started_at=t0,
                                   z_goal_stream_stats=zg.stats())
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (tiny budgets).")
    parser.add_argument("--pilot", action="store_true",
                        help="held-out-seed pilot (101/102) at full budgets; JSON only.")
    parser.add_argument("--pilot-out", default=None)
    args = parser.parse_args()
    if args.pilot:
        assert args.pilot_out, "--pilot requires --pilot-out"
        main("pilot", args.pilot_out)
        sys.exit(0)
    _outcome, _out_path, _run_id = main("smoke" if args.dry_run else "full", None)
    emit_outcome(outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, run_id=_run_id, queue_id=QUEUE_ID,
                 exit_reason="ok" if _outcome == "PASS" else "fail", dry_run=args.dry_run)
    sys.exit(0)
