"""
V3-EXQ-1050 -- MECH-021: is the subjective now a horizon-integrating control surface?

SLEEP DRIVER: K=never (SleepLoopManager disabled during training; sleep called manually at eval)
RED-TEAM (Step 4.5, opus -- the Fable spawn hit a model spend limit and was re-spawned
once inheriting the session model, per the skill): BLOCKING on first pass; 2 BLOCKING +
6 CONTESTED findings, all verified against source and all fixed (see RED-TEAM DISPOSITIONS).

QUESTION (MECH-021 falsifier, claims.yaml what_would_answer, disposition 2026-09-16)
------------------------------------------------------------------------------------
Does E3 commitment behave as a control surface that INTEGRATES predictions across
temporal horizons, or does it merely REACT to the current sensory timestamp?

MECH-021 names two integration axes, and this driver measures both:
  * FUTURE horizon -- how far ahead the E3 harm term reads (E2 world_forward depth).
  * PAST window    -- what "now" is built from (ThetaBuffer.summary() over N z_world).

READOUT: ANTICIPATORY-RESTRAINT FRACTION (arf)
----------------------------------------------
At every E3 tick whose nearest-hazard Manhattan distance d lies in [2, 6] -- harm is
reachable in MORE than one step (so restraint can be anticipatory) and has NOT been
realised -- we:

  1. Classify each of the env's 5 discrete actions from the GRID, not from any learned
     head: APPROACH if it strictly reduces d, RECEDE if it strictly increases d. This
     ground truth is independent of harm_eval, so the readout is NOT circular.
  2. Roll z0 forward under each action with e2.world_forward and accumulate the harm
     term of J, m_D(a) = sum_{j=1..D} harm_eval(world_forward^j(z0, a)), for each
     depth prefix D. One rollout yields every depth, so the depth contrast is
     WITHIN-TICK and perfectly paired.
  3. arf(D) = fraction of in-band ticks at which mean(m_D | APPROACH) > mean(m_D | RECEDE),
     i.e. the pre-sampling score correctly penalises approaching the hazard.

This is read off the PRE-SAMPLING score, never the sampled action, so it is
COMMITMENT-FREE: it does not depend on the sustained action-commitment layer, and
therefore cannot merely re-derive the known F-dominance conversion ceiling
(substrate_queue f_dominance_conversion_ceiling, implementation_status wontfix).

ARMS (2; ONE warmed substrate per seed, shared -- claims.yaml: "one trained substrate
per seed"). Both knobs are EVAL-TIME reads, so training is done once and deep-copied:
  ARM_WIN1  theta_buffer_size = 1   -- "now" IS the instantaneous z_world
  ARM_WIN10 theta_buffer_size = 10  -- "now" integrates the last 10 z_world (default)

Within ARM_WIN10 the PAST-window contrast is also taken WITHIN-TICK: a depth-10 buffer
holds both summaries, so m_D is scored from summary_10 and from summary_1 off the same
tick. ARM_WIN1 cannot host that contrast (its buffer holds one entry, so the two
summaries are identical BY CONSTRUCTION) -- precondition P5 is therefore SCOPED OUT of
ARM_WIN1 via applies_to (disposition (a)), never failed by it.

PRE-REGISTERED CRITERIA (thresholds are module constants; none is derived post-hoc)
-----------------------------------------------------------------------------------
  C1 (load-bearing, FUTURE horizon) arf(D_FULL) - arf(D=1) >= ARF_MARGIN, pooled over
      in-band ticks, AND the per-seed delta is positive on every seed (>= MIN_SEEDS).
  C2 (load-bearing, PAST window) in ARM_WIN10, arf under summary_10 minus arf under
      summary_1 >= ARF_MARGIN at D_FULL, AND per-seed-positive on every seed.
  C3 (NOT load-bearing, behavioural corollary) hazard CONTACTS per 100 steps are lower
      in ARM_WIN10 than in ARM_WIN1 by > HARM_EPISODE_MARGIN.

COMBINATION RULE and why C3 is not load-bearing. MECH-021's CONFIRMING branch also asks
that realised harm fall ("the restraint is protective, not just earlier"). That clause
runs through score-to-committed-action CONVERSION, which is a separately tracked and
currently un-lifted ceiling. Routing the verdict through it would re-derive that ceiling
rather than test MECH-021, so C3 is measured and reported but does not gate the outcome:
  outcome PASS  iff C1 and C2                (the claim's substrate content)
  supports      iff C1 and C2 and C3         (the claim's full CONFIRMING branch)
  mixed         iff exactly one of C1/C2, or (C1 and C2 and not C3)
  weakens       iff neither C1 nor C2        (the claim's FALSIFYING branch)

MEASURABILITY IS A PRECONDITION OF SCORING, NOT A CRITERION. C2 lives in ARM_WIN10 alone,
so if every ARM_WIN10 cell fails its readiness gate there is no instrument for it -- and a
criterion that could not be measured is NEVER reported as failed. The run then routes
`substrate_not_ready_requeue` / `unknown`, and `interpretation.unmeasured_criteria` names
which one. non_degenerate requires both load-bearing criteria to be measurable AND
un-starved, not merely that some arm survived.

RED-TEAM DISPOSITIONS (Step 4.5; every finding verified against source before acting)
--------------------------------------------------------------------------------------
 F1 BLOCKING, CONFIRMED, FIXED. An un-instrumented C2 was written to disk as MECH-021's
    FALSIFYING branch: with all ARM_WIN10 cells red, c2_pass went False for want of an
    instrument while non_degenerate ("any arm green") stayed True on a surviving ARM_WIN1
    cell, routing to `weakens`. Fixed by the measurability rule above; additionally the
    red cells' failed preconditions are re-attached to interpretation.preconditions,
    because aggregate_arm_gates deliberately publishes GREEN-arm preconditions only and
    a wholly-red single-criterion arm would otherwise vanish from adjudication.
 F2 BLOCKING, CONFIRMED, FIXED. Under use_proxy_fields (CausalGridWorldV2 sets it True)
    `transition_type == "hazard_approach"` fires on PROXIMITY WITH NO CONTACT and still
    deducts health, so the old `"hazard" in tt` substring test counted loitering as
    realised harm -- and per EPISODE that count is monotone in episode LENGTH, making
    C3's "lower is better" satisfiable by dying sooner. Now counts contact transitions
    only, denominated per 100 STEPS; proximity steps and episode lengths are recorded
    separately so the survival difference is visible rather than hidden in the DV.
 F3 CONTESTED -- the in-band sample. Addressed by budget, and it is exactly what the
    inband_tick_count precondition gates: too few in-band ticks self-routes
    substrate_not_ready_requeue rather than reading as a null.
 F4 CONTESTED, CONFIRMED, DECLARED (not fixed -- it does not touch the load-bearing
    criteria). theta_buffer_size also sizes the deque behind ThetaBuffer.recent, which
    agent._do_replay consumes, so the arms differ in replay depth as well as in the E3
    summary. C1 and C2 are immune: both are WITHIN-TICK contrasts computed from a single
    tick's z0 via world_forward, and C2 never uses ARM_WIN1 at all. It bears only on C3
    (already non-load-bearing) and on which states get sampled, and is declared as a
    joint attribution there.
 F5 CONTESTED, CONFIRMED, FIXED. The readiness gate ranged over all 5 actions while
    _restraint reads only the APPROACH and RECEDE subsets -- which always exclude the
    stay action and every blocked move -- so the gate could be green while the contrast
    the criterion routes on was identically zero. The gate is now that contrast
    (approach_recede_gap); the 5-action range is kept as a diagnostic only.
 F6 CONTESTED, CONFIRMED, FIXED. The scaffold pins env_drift_interval only on its
    HAZARD-STAGE env, so this p0 eval env inherited the class default and hazards
    random-walked every 5 steps -- INSIDE the D_FULL=30 horizon whose ground-truth label
    they define, adding noise against C1's pre-registered direction. env_drift_prob is
    now pinned to 0.0 for eval. grid_model_match_rate was also diluted over every env
    step (~100x the ticks that matter) and is now denominated over IN-BAND ticks only.
 F7 CONTESTED, CONFIRMED, FIXED. arf is bounded above by 1.0, so C1's headroom is
    1 - arf(D=1) and a shallow fraction at ceiling makes the lift unreachable whatever
    the substrate does. Added the arf_shallow_headroom CEILING precondition
    (direction "upper"), so a capped arf(1) self-routes not-ready instead of reading as
    a refutation.
 F8 CONTESTED, CONFIRMED, FIXED. Sign-consistency was a set over cells pooled across
    both arms (">=3 seeds had at least one positive cell"); it is now each seed's own
    mean delta. Pooling is now in-band-tick-weighted, matching what the criterion detail
    string claims. Non-finite values are scrubbed from `criteria` before the write.

DV-SYMMETRY INVARIANCE (mandatory per-arm declaration)
-------------------------------------------------------
arf is a RANK/ORDER statistic (a mean comparison between two disjoint action subsets),
so it is invariant under any constant added uniformly across actions and under any
monotone rescaling of m.
  ARM_WIN1 / ARM_WIN10, C1 (depth): deepening D adds sum_{j} harm_eval(z_j(a)) where
    z_j(a) depends on a through world_forward. It is a uniform constant ONLY if
    world_forward collapses to an action-independent fixed point. It does not here --
    measured 2026-09-17, cross-action harm range GROWS with depth (0.0033 at D=1 to
    0.0257 at D=5) -- and P3 asserts exactly that statistic at D_FULL, so a collapsed
    substrate self-routes substrate_not_ready_requeue instead of reading as a null.
  ARM_WIN10, C2 (theta window): swapping summary_10 for summary_1 changes z0, which is
    shared across actions WITHIN a tick -- but m(a) is not affine in z0 (world_forward
    and harm_eval are both nonlinear), so it is not a uniform shift and cannot cancel in
    the rank. P5 asserts the two summaries actually diverge; P3 asserts the resulting
    m still has cross-action range.
  Neither manipulation is a permutation of interchangeable units, and the action set is
  fixed across arms, so the set-aggregate symmetry does not apply.

SUBSTRATE (all built; the arf readout is a driver-side instrument)
  ree_core/latent/theta_buffer.py       ThetaBuffer.summary (MECH-089)
  ree_core/predictors/e2_fast.py        world_forward
  ree_core/predictors/e3_selector.py    harm_eval / harm_eval_head (the harm term of J)
  experiments/_lib/infant_warmup.py     the 603n-style harm/survival warm-up + its own
                                        harm_pathway_discriminative readiness gate

GROUND-TRUTH SELF-CHECK. The grid model used for APPROACH/RECEDE is validated against
the env every step: the position it predicts for the COMMITTED action is compared with
the env's actual post-step position, and the match rate is precondition P7. A driver
whose env model has drifted fails the gate instead of silently mislabelling the DV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.infant_warmup import (  # noqa: E402
    HARM_EVAL_RANGE_FLOOR,
    build_warmed_agent,
    run_warmup,
    warmup_config_slice,
    warmup_scaffold_config,
)
from scaffolded_sd054_onboarding import (  # noqa: E402
    _build_env,
    _sense_with_optional_harm,
)

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1050_mech021_subjective_now_horizon_integration"
QUEUE_ID = "V3-EXQ-1050"
CLAIM_IDS = ["MECH-021"]
RED_TEAM_VERDICT = ("red-team (opus): BLOCKING on first pass -- 2 BLOCKING + 6 CONTESTED, "
                    "all verified against source and all fixed or declared; see the "
                    "RED-TEAM DISPOSITIONS block in the module docstring")

# --------------------------------------------------------------------------
# PRE-REGISTERED constants. None of these is derived from the run's own stats.
# --------------------------------------------------------------------------
SEEDS = [0, 1, 2]
MIN_SEEDS = 3                      # claims.yaml: "sign-consistent across >=3 seeds"

ARM_WIN1 = "ARM_WIN1"
ARM_WIN10 = "ARM_WIN10"
ARMS: Dict[str, int] = {ARM_WIN1: 1, ARM_WIN10: 10}

DEPTHS: Tuple[int, ...] = (1, 2, 5, 10, 20, 30)
D_SHALLOW = 1
D_FULL = 30                        # == REEConfig.e2.rollout_horizon default

BAND_LO, BAND_HI = 2, 6            # hazard reachable in >1 step, <= the read horizon

ARF_MARGIN = 0.05                  # C1 / C2 effect-size bar on a [0,1] fraction
HARM_EPISODE_MARGIN = 0.0          # C3: any strict reduction in realised harm

# Readiness floors.
WF_ACTION_SPREAD_FLOOR = 0.05      # V3-EXQ-649's calibrated GAP-A bar, same statistic
HARM_RANGE_FLOOR = 1e-4            # cross-action range of m at D_FULL (diagnostic only)
AR_GAP_FLOOR = 1e-5                # |mean(m|APPROACH) - mean(m|RECEDE)| -- the statistic
                                   # _restraint ACTUALLY routes on (red-team F5)
ARF_SHALLOW_CEILING = 1.0 - ARF_MARGIN   # C1 headroom: arf(D=1) must leave room for a lift
THETA_DIVERGENCE_FLOOR = 1e-3      # summary_10 vs summary_1 L2 (C2's manipulation)
MIN_INBAND_TICKS = 40              # per arm x seed cell
ZWORLD_NORM_CEILING = 1e6          # numerical-stability ceiling (direction: upper)
GRID_MODEL_MATCH_FLOOR = 0.95      # ground-truth self-check

# Budgets (full scale). --dry-run shrinks all of them.
WARM_STAGE0, WARM_STAGE0B, WARM_P0, WARM_HAZARD = 8, 4, 25, 20
STEPS_PER_EPISODE = 120
EVAL_EPISODES = 20
EVAL_HAZARDS = 3                   # warm-up trains at 1; raised for event rate (declared)
WARMUP_EPISODES_TOTAL = WARM_STAGE0 + WARM_STAGE0B + WARM_P0 + WARM_HAZARD

SLEEP_DRIVER_PATTERN = "K=never (SleepLoopManager disabled during training; sleep called manually at eval)"

# Dead-z_goal-stream detector. The agent is built INSIDE _run_cell, so a run-level
# list would keep every cell's agent alive to the end; the accumulator reads the
# counters at observe() time instead.
_ZG = ZGoalStreamAccumulator()

ETHICS_PREFLIGHT = {
    "involves_negative_valence": False,
    "involves_suffering_like_state": False,
    "involves_self_model": False,
    "involves_inescapability_or_helplessness": False,
    "involves_offline_replay_over_harm": False,
    "involves_social_mind_or_language": False,
    "involves_human_data_or_clinical_context": False,
    "decision": "allow",
}


# --------------------------------------------------------------------------
# Grid ground truth -- deliberately independent of every learned head.
# --------------------------------------------------------------------------
def _manhattan(env, ax: int, ay: int, hx: int, hy: int) -> int:
    dx, dy = abs(ax - hx), abs(ay - hy)
    if getattr(env, "toroidal", False):
        dx, dy = min(dx, env.size - dx), min(dy, env.size - dy)
    return int(dx + dy)


def _hazard_dist(env, ax: Optional[int] = None, ay: Optional[int] = None) -> Optional[int]:
    """Manhattan distance to the nearest hazard, or None when there is no hazard."""
    hz = list(getattr(env, "hazards", []) or [])
    if not hz:
        return None
    ax = env.agent_x if ax is None else ax
    ay = env.agent_y if ay is None else ay
    return min(_manhattan(env, ax, ay, int(h[0]), int(h[1])) for h in hz)


def _next_pos(env, a: int) -> Tuple[int, int]:
    """Where action `a` lands the agent, mirroring env.step()'s movement rules.

    Reads env._action_map (the EFFECTIVE map), never the class-level ACTIONS dict:
    a world-rule shift permutes the effective map, and a stale copy would silently
    mislabel every APPROACH/RECEDE decision.
    """
    ax, ay = int(env.agent_x), int(env.agent_y)
    amap = getattr(env, "_action_map", None) or env.ACTIONS
    if a not in amap:                      # e.g. the CONSUME action: a no-move
        return ax, ay
    dx, dy = amap[a]
    if getattr(env, "toroidal", False):
        return (ax + dx) % env.size, (ay + dy) % env.size
    nx, ny = ax + dx, ay + dy
    if not (0 <= nx < env.size and 0 <= ny < env.size):
        return ax, ay                      # off-grid -> blocked, agent stays put
    try:
        if env.grid[nx, ny] == env.ENTITY_TYPES["wall"]:
            return ax, ay
    except Exception:
        return ax, ay
    return nx, ny


def _classify_actions(env, d_now: int, n_actions: int) -> Tuple[List[int], List[int]]:
    """Split the action set into APPROACH / RECEDE by grid geometry alone."""
    approach, recede = [], []
    for a in range(n_actions):
        nx, ny = _next_pos(env, a)
        d_next = _hazard_dist(env, nx, ny)
        if d_next is None:
            continue
        if d_next < d_now:
            approach.append(a)
        elif d_next > d_now:
            recede.append(a)
    return approach, recede


# --------------------------------------------------------------------------
# The within-tick instrument.
# --------------------------------------------------------------------------
@torch.no_grad()
def _depth_profile(agent, z0: torch.Tensor, acts: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], float, float]:
    """m_D(a) for every D in DEPTHS, from ONE rollout per action.

    Returns (per-depth harm-cost vectors over actions, cross-action spread of
    z_world_1, max rolled-out z_world norm).
    """
    n = acts.shape[0]
    zk = z0.reshape(1, -1).expand(n, -1).contiguous()
    running = torch.zeros(n, dtype=zk.dtype)
    out: Dict[int, torch.Tensor] = {}
    wf_spread, max_norm = 0.0, 0.0
    for step in range(1, max(DEPTHS) + 1):
        zk = agent.e2.world_forward(zk, acts)
        if step == 1:
            wf_spread = float(torch.cdist(zk, zk).max().item())
        max_norm = max(max_norm, float(zk.norm(dim=-1).max().item()))
        running = running + agent.e3.harm_eval(zk).reshape(-1)
        if step in DEPTHS:
            out[step] = running.clone()
    return out, wf_spread, max_norm


def _restraint(m: torch.Tensor, approach: List[int], recede: List[int]) -> Optional[bool]:
    """Did the pre-sampling score rank APPROACH worse (higher harm cost) than RECEDE?"""
    if not approach or not recede:
        return None
    return bool(float(m[approach].mean()) > float(m[recede].mean()))


def _frac(hits: int, n: int) -> float:
    return float(hits) / float(n) if n else float("nan")


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = len(s) // 2
    return float(s[k]) if len(s) % 2 else float(0.5 * (s[k - 1] + s[k]))


# --------------------------------------------------------------------------
# Warm-up (once per seed) and evaluation (once per arm x seed cell).
# --------------------------------------------------------------------------
def _warm_one_seed(seed: int, budgets: Dict[str, int], verbose: bool = True):
    """Run the shared harm/survival warm-up. Returns (agent, scaffold_cfg, WarmupResult)."""
    scfg = warmup_scaffold_config(
        stage0_budget=budgets["stage0"],
        stage0b_budget=budgets["stage0b"],
        p0_budget=budgets["p0"],
        hazard_budget=budgets["hazard"],
        steps_per_episode=budgets["steps"],
        env_seed=seed,
    )
    torch.manual_seed(seed)
    agent = build_warmed_agent(scfg, sleep_loop_episodes_K=10 ** 9)  # K=never
    total = budgets["stage0"] + budgets["stage0b"] + budgets["p0"] + budgets["hazard"]
    if verbose:
        print(f"  [train] warmup seed={seed} ep 1/{total} (stage0 -> stage0b -> P0 -> stageH)",
              flush=True)
    res = run_warmup(agent, scfg, include_hazard_stage=True, verbose=False)
    if verbose:
        print(f"  [train] warmup seed={seed} ep {total}/{total} done; "
              f"harm_eval_range={res.harm_eval_range:.6f} "
              f"discriminative={res.harm_pathway_discriminative}", flush=True)
    return agent, scfg, res


@torch.no_grad()
def _eval_cell(agent, scfg, arm: str, seed: int, budgets: Dict[str, int]) -> Dict[str, Any]:
    """Evaluate one arm on this cell's own freshly warmed substrate.

    REEAgent does not support copy.deepcopy (verified 2026-09-17: it raises even on a
    FRESH agent), so the warm-up is re-run per cell rather than shared by copy. That is
    not a workaround with a cost: the warm-up is a pure function of
    (substrate, warm-up config, seed) -- infant_warmup.py's own REUSE note -- so the two
    arms of a seed receive an IDENTICAL substrate, and re-warming additionally removes
    the arm-ORDER confound that sequentially reusing one mutable agent would introduce
    (residue field, goal_state and hippocampal memory all accumulate across eval).
    """
    dev = torch.device("cpu")
    win = ARMS[arm]
    agent.config.heartbeat.theta_buffer_size = win
    agent.clock.theta_buffer_size = win
    agent.theta_buffer.buffer_size = win
    agent.theta_buffer._z_world_buffer = type(agent.theta_buffer._z_world_buffer)(
        list(agent.theta_buffer._z_world_buffer)[-win:], maxlen=win)
    agent.theta_buffer._z_self_buffer = type(agent.theta_buffer._z_self_buffer)(
        list(agent.theta_buffer._z_self_buffer)[-win:], maxlen=win)

    ecfg = warmup_scaffold_config(steps_per_episode=budgets["steps"], env_seed=seed + 7919)
    ecfg.scaffold_p0_num_hazards = EVAL_HAZARDS

    wd = agent.config.latent.world_dim
    hits = {d: 0 for d in DEPTHS}
    hits_win1 = {d: 0 for d in DEPTHS}
    n_inband = 0
    n_e3 = 0
    n_latched_ticks = 0
    wf_spreads: List[float] = []
    harm_ranges: List[float] = []
    ar_gaps: List[float] = []
    theta_divs: List[float] = []
    max_zw_norm = 0.0
    grid_ok, grid_tot = 0, 0           # over IN-BAND ticks only (red-team F6)
    n_contacts = 0                     # REALISED harm: contact only (red-team F2)
    n_approach_steps = 0               # proximity-only steps, recorded separately
    n_steps = 0
    n_eps = 0
    ep_len: List[int] = []
    ep_contacts: List[int] = []

    for ep in range(budgets["eval_eps"]):
        env = _build_env(ecfg, "p0", seed=seed * 1000 + ep)
        # red-team F6: the scaffold pins env_drift_interval only on its HAZARD-STAGE env,
        # so this p0 env would inherit the class default (drift every 5 steps, p=0.3) and
        # the hazards defining APPROACH/RECEDE would random-walk INSIDE the D_FULL=30
        # scoring horizon -- injecting label noise that runs directly against C1's
        # pre-registered direction. Pin the drift off so the ground-truth label is stable
        # across the horizon it is read over. Declared in the config slice.
        env.env_drift_prob = 0.0
        _, obs = env.reset()
        agent.reset()
        n_eps += 1
        ep_c = 0
        this_len = 0
        for step in range(budgets["steps"]):
            ob = obs["body_state"].to(dev)
            ow = obs["world_state"].to(dev)
            latent = _sense_with_optional_harm(agent, ob, ow, obs, dev, True)
            ticks = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, wd, device=dev))
            cands = agent.generate_trajectories(latent, e1_prior, ticks)

            _inband_now = False
            if ticks.get("e3_tick"):
                n_e3 += 1
                d_now = _hazard_dist(env)
                if d_now is not None and BAND_LO <= d_now <= BAND_HI:
                    approach, recede = _classify_actions(env, d_now, env.action_dim)
                    if approach and recede:
                        buf = list(agent.theta_buffer._z_world_buffer)
                        if buf:
                            acts = torch.eye(env.action_dim, dtype=buf[-1].dtype)
                            z_sum = agent.theta_buffer.summary().detach().reshape(-1)
                            prof, wfs, mx = _depth_profile(agent, z_sum, acts)
                            wf_spreads.append(wfs)
                            max_zw_norm = max(max_zw_norm, mx)
                            harm_ranges.append(
                                float(prof[D_FULL].max() - prof[D_FULL].min()))
                            # red-team F5: the 5-action RANGE is a SUPERSET of what
                            # _restraint reads (it always excludes the stay action and
                            # any blocked move), so it can be green while the
                            # approach-vs-recede contrast is identically zero. Record
                            # the gap the criterion actually routes on.
                            ar_gaps.append(abs(
                                float(prof[D_FULL][approach].mean())
                                - float(prof[D_FULL][recede].mean())))
                            grid_tot += 1
                            for d in DEPTHS:
                                r = _restraint(prof[d], approach, recede)
                                if r:
                                    hits[d] += 1
                            # PAST-window counterfactual: only ARM_WIN10 holds both.
                            if win > 1:
                                z_inst = buf[-1].detach().reshape(-1)
                                theta_divs.append(float((z_sum - z_inst).norm()))
                                prof1, _, mx1 = _depth_profile(agent, z_inst, acts)
                                max_zw_norm = max(max_zw_norm, mx1)
                                for d in DEPTHS:
                                    r1 = _restraint(prof1[d], approach, recede)
                                    if r1:
                                        hits_win1[d] += 1
                            n_inband += 1
                            _inband_now = True
            else:
                n_latched_ticks += 1

            act = agent.select_action(cands, ticks)
            a_idx = int(act.argmax(dim=-1).item())
            px, py = _next_pos(env, a_idx)
            was_inband = bool(_inband_now)
            _, harm_signal, done, info, obs = env.step(a_idx)
            n_steps += 1
            this_len += 1
            if was_inband:                       # F6: score the model where it MATTERS
                grid_ok += int((int(env.agent_x), int(env.agent_y)) == (px, py))
            tt = str(info.get("transition_type") or "")
            # red-team F2: under use_proxy_fields (CausalGridWorldV2 sets it True) the
            # branch `hazard_approach` fires on PROXIMITY WITH NO CONTACT and still
            # deducts health, so a substring test for "hazard" counts loitering near a
            # hazard as realised harm -- and its per-EPISODE rate is then monotone in
            # episode LENGTH, making "lower is better" satisfiable by dying sooner.
            # Count CONTACT only, and denominate in STEPS.
            if tt in ("agent_caused_hazard", "env_caused_hazard"):
                n_contacts += 1
                ep_c += 1
            elif tt == "hazard_approach":
                n_approach_steps += 1
            if done:
                break
        ep_len.append(this_len)
        ep_contacts.append(ep_c)
        print(f"  [eval] {arm} seed={seed} ep {ep + 1}/{budgets['eval_eps']} "
              f"inband={n_inband} contacts={n_contacts} approach_steps={n_approach_steps} "
              f"steps={n_steps}", flush=True)

    arf = {d: _frac(hits[d], n_inband) for d in DEPTHS}
    arf_win1 = ({d: _frac(hits_win1[d], n_inband) for d in DEPTHS} if win > 1 else None)
    return {
        "arm": arm,
        "seed": seed,
        "theta_buffer_size": win,
        "n_inband_ticks": n_inband,
        "n_e3_ticks": n_e3,
        "n_latched_ticks": n_latched_ticks,
        "arf_by_depth": {str(d): arf[d] for d in DEPTHS},
        "arf_win1_summary_by_depth": (
            {str(d): arf_win1[d] for d in DEPTHS} if arf_win1 else None),
        "arf_depth_delta": (arf[D_FULL] - arf[D_SHALLOW]),
        "arf_theta_delta": (
            (arf[D_FULL] - arf_win1[D_FULL]) if arf_win1 else None),
        "arf_shallow": arf[D_SHALLOW],
        "wf_action_spread_median": _median(wf_spreads),
        "harm_cost_range_full_depth_median": _median(harm_ranges),
        "approach_recede_gap_median": _median(ar_gaps),
        "theta_summary_divergence_median": _median(theta_divs) if win > 1 else None,
        "max_rolled_zworld_norm": max_zw_norm,
        "grid_model_match_rate": _frac(grid_ok, grid_tot),
        "hazard_drift_disabled": True,
        "hazard_contacts_total": n_contacts,
        "hazard_approach_steps_total": n_approach_steps,
        "eval_steps_total": n_steps,
        "episodes": n_eps,
        "mean_episode_length": (n_steps / n_eps) if n_eps else float("nan"),
        "contacts_per_100_steps": (100.0 * n_contacts / n_steps) if n_steps else float("nan"),
        "per_episode_length": ep_len,
        "per_episode_contacts": ep_contacts,
    }


def _run_cell(arm: str, seed: int, budgets: Dict[str, int]) -> Dict[str, Any]:
    """ONE (arm, seed) cell: warm a fresh substrate, then evaluate it.

    The `with arm_cell(...)` wrapping lives HERE, inside the per-cell function, not in
    the caller's loop -- so calling this function directly (for reproduction or
    spot-checking) still gets the complete RNG reset and the fingerprint stamp. The
    V3-EXQ-1048 figures were invalidated by exactly that bypass.
    """
    scfg_probe = warmup_scaffold_config(
        stage0_budget=budgets["stage0"], stage0b_budget=budgets["stage0b"],
        p0_budget=budgets["p0"], hazard_budget=budgets["hazard"],
        steps_per_episode=budgets["steps"], env_seed=seed)
    cslice = dict(warmup_config_slice(scfg_probe))
    cslice.update({
        "arm": arm,
        "theta_buffer_size": ARMS[arm],
        "eval_episodes": budgets["eval_eps"],
        "eval_hazards": EVAL_HAZARDS,
        "steps_per_episode": budgets["steps"],
        "depths": list(DEPTHS),
        "d_full": D_FULL,
        "d_shallow": D_SHALLOW,
        "band": [BAND_LO, BAND_HI],
        "arf_margin": ARF_MARGIN,
    })
    with arm_cell(
        seed,
        config_slice=cslice,
        script_path=Path(__file__),
        config_slice_declared=True,
        # MINT AS YOU GO: the whole cell (warm-up + eval) is a pure function of
        # (substrate, this slice, seed) -- nothing mutable is shared across cells --
        # so it is emitted reuse-ELIGIBLE, and with the driver EXCLUDED from the hash
        # so a later, different driver can match it.
        include_driver_script_in_hash=False,
    ) as cell:
        agent, scfg, wres = _warm_one_seed(seed, budgets)
        row = _eval_cell(agent, scfg, arm, seed, budgets)
        _ZG.observe(agent)          # AFTER stepping -- it reads the counters at call time
        row["warmup"] = wres.as_dict()
        row["_slice_ref"] = cslice
        cell.stamp(row)
    return row


# --------------------------------------------------------------------------
# Preconditions.
# --------------------------------------------------------------------------
def _specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="harm_pathway_discriminative",
            description="Stage-H trained the harm landscape: harm_eval output RANGE over "
                        "in-distribution z_world clears the 603n non-vacuity floor.",
            control="infant_warmup Stage-H's own harm-discriminativeness measurement",
            threshold=HARM_EVAL_RANGE_FLOOR, direction="lower", kind="readiness"),
        PreconditionSpec(
            name="worldforward_action_spread",
            description="e2.world_forward is action-discriminative: cross-action max "
                        "pairwise L2 of z_world_1 (the ARC-065 GAP-A collapse check).",
            control="the 5 discrete env actions from the live theta summary, in-band ticks",
            threshold=WF_ACTION_SPREAD_FLOOR, direction="lower", kind="readiness"),
        PreconditionSpec(
            name="approach_recede_gap",
            description="THE statistic C1/C2 route on: |mean(m_30 | APPROACH) - "
                        "mean(m_30 | RECEDE)|. _restraint compares exactly these two "
                        "subset means, and they EXCLUDE the stay action and every "
                        "blocked move -- so the 5-action range is a superset that can be "
                        "green while this contrast is identically zero. A collapsed gap "
                        "means the criterion is starved, not falsified.",
            control="median over in-band ticks",
            threshold=AR_GAP_FLOOR, direction="lower", kind="readiness"),
        PreconditionSpec(
            name="arf_shallow_headroom",
            description="C1 needs room to show a lift: arf is bounded above by 1.0, so a "
                        "shallow-depth fraction already at ceiling makes the >= ARF_MARGIN "
                        "lift unreachable regardless of the substrate.",
            control="arf at D_SHALLOW, this cell",
            threshold=ARF_SHALLOW_CEILING, direction="upper", kind="readiness"),
        PreconditionSpec(
            name="inband_tick_count",
            description="Enough hazard-reachable-but-not-realised E3 ticks to estimate arf.",
            control="E3 ticks with nearest-hazard distance in [2,6] and both classes non-empty",
            threshold=float(MIN_INBAND_TICKS), direction="lower", kind="readiness"),
        PreconditionSpec(
            name="theta_summary_divergence",
            description="C2's manipulation actually moves z0: L2 between summary_10 and "
                        "summary_1. Structurally 0 when the buffer holds one entry.",
            control="in-band ticks in the depth-10 arm",
            threshold=THETA_DIVERGENCE_FLOOR, direction="lower", kind="readiness",
            applies_to=lambda ctx: int(ctx.get("theta_buffer_size", 1)) > 1,
            applies_note="ARM_WIN1's buffer holds ONE entry, so summary_10 and summary_1 "
                         "are identical by construction; the PAST-window contrast is not "
                         "meaningful for this regime (disposition (a), scoped out)."),
        PreconditionSpec(
            name="rolled_out_zworld_bounded",
            description="Numerical stability: rolled-out z_world norm stayed below the "
                        "explosion ceiling.",
            control="max over every rollout step of every in-band tick",
            threshold=ZWORLD_NORM_CEILING, direction="upper", kind="readiness"),
        PreconditionSpec(
            name="grid_model_match_rate",
            description="The driver's env movement model (which defines APPROACH/RECEDE) "
                        "agrees with the env's actual post-step position.",
            control="committed action, every env step",
            threshold=GRID_MODEL_MATCH_FLOOR, direction="lower", kind="readiness"),
    ]


def _measured(cell: Dict[str, Any], warm: Dict[str, Any]) -> Dict[str, float]:
    m = {
        "harm_pathway_discriminative": float(warm["harm_eval_range"]),
        "worldforward_action_spread": float(cell["wf_action_spread_median"]),
        "approach_recede_gap": float(cell["approach_recede_gap_median"]),
        "arf_shallow_headroom": float(cell["arf_shallow"]) if not math.isnan(
            cell["arf_shallow"]) else 1.0,
        "inband_tick_count": float(cell["n_inband_ticks"]),
        "rolled_out_zworld_bounded": float(cell["max_rolled_zworld_norm"]),
        "grid_model_match_rate": float(cell["grid_model_match_rate"]),
    }
    if int(cell["theta_buffer_size"]) > 1:
        m["theta_summary_divergence"] = float(cell["theta_summary_divergence_median"] or 0.0)
    return m


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def run_experiment(budgets: Dict[str, int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    specs = _specs()

    # Design-time refusal BEFORE compute: a precondition no arm could ever satisfy.
    arm_ctxs = [{"arm": a, "theta_buffer_size": w} for a, w in ARMS.items()]
    assert_no_structurally_unsatisfiable_gate(specs, arm_ctxs)

    arm_results: List[Dict[str, Any]] = []
    arm_gates: List[Dict[str, Any]] = []
    warm_by_seed: Dict[int, Dict[str, Any]] = {}
    slice_ref: Dict[str, Any] = {}

    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed, budgets)
            warm_by_seed[seed] = row["warmup"]
            slice_ref = row.pop("_slice_ref")
            arm_results.append(row)
            gate = evaluate_arm_gate(
                f"{arm}_s{seed}",
                {"arm": arm, "theta_buffer_size": ARMS[arm]},
                specs,
                _measured(row, row["warmup"]),
            )
            arm_gates.append(gate)
            row["gate_green"] = bool(gate["gate_green"])
            print(f"verdict: {'PASS' if gate['gate_green'] else 'FAIL'}", flush=True)

    agg = aggregate_arm_gates(arm_gates)
    green_rows = [r for r in arm_results if r.get("gate_green")]

    # ---- criteria, computed over GREEN cells only -------------------------
    # red-team F8: weight each cell by the in-band ticks it actually contributed.
    # An unweighted mean over cells lets a 40-tick cell outvote a 400-tick one while
    # the criterion's own detail string says "pooled over in-band ticks".
    def _pool(rows, key):
        num = den = 0.0
        for r in rows:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            w = float(r.get("n_inband_ticks") or 0)
            if w <= 0:
                continue
            num += float(v) * w
            den += w
        return (num / den) if den else float("nan")

    def _seeds_positive(rows, key):
        """A seed counts only if its OWN mean delta is positive (F8).

        The previous form was a set comprehension over cells pooled across BOTH arms,
        so a seed counted as positive when EITHER of its two cells was -- which is
        "at least one positive cell", not the sign-consistency claims.yaml asks for.
        """
        by_seed: Dict[int, List[float]] = {}
        for r in rows:
            v = r.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            by_seed.setdefault(int(r["seed"]), []).append(float(v))
        return sorted(sd for sd, vs in by_seed.items() if vs and (sum(vs) / len(vs)) > 0.0)

    d_rows = green_rows
    t_rows = [r for r in green_rows if r["arm"] == ARM_WIN10]

    # red-team F1 (BLOCKING, confirmed): C2 lives in ARM_WIN10 ALONE. If every
    # ARM_WIN10 cell is red, `t_rows` is empty -- c2_pass then evaluates False for want
    # of an INSTRUMENT, while aggregate_arm_gates' non_degenerate ("any arm green") is
    # still True on a surviving ARM_WIN1 cell, so the run routed to `weakens` and wrote
    # MECH-021's FALSIFYING branch to disk having never measured the past-window
    # manipulation at all. Measurability is therefore a precondition of SCORING, not a
    # criterion: a criterion that could not be measured is never reported as failed.
    c1_measurable = bool(d_rows)
    c2_measurable = bool(t_rows)

    c1_measured = _pool(d_rows, "arf_depth_delta")
    c1_seed_pos = _seeds_positive(d_rows, "arf_depth_delta")
    c1_pass = bool(
        c1_measurable and not math.isnan(c1_measured) and c1_measured >= ARF_MARGIN
        and len(c1_seed_pos) >= MIN_SEEDS)

    c2_measured = _pool(t_rows, "arf_theta_delta")
    c2_seed_pos = _seeds_positive(t_rows, "arf_theta_delta")
    c2_pass = bool(
        c2_measurable and not math.isnan(c2_measured) and c2_measured >= ARF_MARGIN
        and len(c2_seed_pos) >= MIN_SEEDS)

    # red-team F2: contacts per 100 STEPS -- realised harm, on a denominator that does
    # not reward dying sooner.
    def _rate(rows):
        c = sum(r["hazard_contacts_total"] for r in rows)
        n = sum(r["eval_steps_total"] for r in rows)
        return (100.0 * c / n) if n else float("nan")

    r10_rows = [r for r in green_rows if r["arm"] == ARM_WIN10]
    r1_rows = [r for r in green_rows if r["arm"] == ARM_WIN1]
    h10, h1 = _rate(r10_rows), _rate(r1_rows)
    c3_measurable = bool(r10_rows and r1_rows)
    c3_measured = (h1 - h10) if c3_measurable and not (
        math.isnan(h1) or math.isnan(h10)) else float("nan")
    c3_pass = bool(c3_measurable and not math.isnan(c3_measured)
                   and c3_measured > HARM_EPISODE_MARGIN)

    # Non-degeneracy now requires that BOTH load-bearing criteria were measurable and
    # that neither is starved -- not merely that some arm survived its gate.
    gap_ok = bool(d_rows) and _pool(d_rows, "approach_recede_gap_median") > AR_GAP_FLOOR
    theta_ok = bool(t_rows) and _pool(
        t_rows, "theta_summary_divergence_median") > THETA_DIVERGENCE_FLOOR
    non_degenerate = bool(agg["non_degenerate"] and c1_measurable and c2_measurable
                          and gap_ok and theta_ok)
    overall_pass = bool(c1_pass and c2_pass and non_degenerate)

    unmeasured = [n for n, ok in (("C1_future_horizon_depth_lift", c1_measurable and gap_ok),
                                  ("C2_past_window_theta_lift", c2_measurable and theta_ok))
                  if not ok]
    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif c1_pass and c2_pass and c3_pass:
        label = "now_is_a_horizon_integrating_control_surface"
        direction = "supports"
    elif c1_pass and c2_pass:
        label = "score_level_integration_confirmed_behavioural_conversion_not"
        direction = "mixed"
    elif c1_pass or c2_pass:
        label = "one_integration_axis_only"
        direction = "mixed"
    else:
        label = "now_is_reactive_not_horizon_integrating"
        direction = "weakens"

    criteria = [
        {"name": "C1_future_horizon_depth_lift", "load_bearing": True, "passed": c1_pass,
         "measured": c1_measured, "threshold": ARF_MARGIN,
         "seeds_positive": len(c1_seed_pos), "seeds_required": MIN_SEEDS,
         "measurable": c1_measurable and gap_ok,
         "detail": "arf(D=30) - arf(D=1), in-band-tick-weighted mean over green cells"},
        {"name": "C2_past_window_theta_lift", "load_bearing": True, "passed": c2_pass,
         "measured": c2_measured, "threshold": ARF_MARGIN,
         "seeds_positive": len(c2_seed_pos), "seeds_required": MIN_SEEDS,
         "measurable": c2_measurable and theta_ok,
         "detail": "arf(summary_10) - arf(summary_1) at D=30, ARM_WIN10, within-tick paired"},
        {"name": "C3_realised_harm_falls", "load_bearing": False, "passed": c3_pass,
         "measured": c3_measured, "threshold": HARM_EPISODE_MARGIN,
         "measurable": c3_measurable,
         "detail": "hazard CONTACTS per 100 steps, ARM_WIN1 minus ARM_WIN10 (contact "
                   "only -- 'hazard_approach' is proximity-without-contact and its "
                   "per-episode rate rewards dying sooner). NOT load-bearing: it runs "
                   "through score-to-committed-action conversion, a separately tracked "
                   "un-lifted ceiling (f_dominance_conversion_ceiling)."},
    ]
    combination_rule = (
        "PASS = C1 AND C2 AND non_degenerate. evidence_direction: supports = C1 AND C2 AND C3; "
        "mixed = (C1 AND C2 AND NOT C3) or exactly one of C1/C2; weakens = NOT C1 AND NOT C2. "
        "C3 is measured and reported but never gates the outcome.")

    criteria_non_degenerate = {
        "C1_future_horizon_depth_lift": bool(gap_ok),
        "C2_past_window_theta_lift": bool(theta_ok),
        "C3_realised_harm_falls": bool(
            c3_measurable and not math.isnan(h1) and not math.isnan(h10)
            and (h1 > 0.0 or h10 > 0.0)),
    }

    # red-team F1, second half: aggregate_arm_gates publishes GREEN-ARM preconditions
    # only (precondition_gate.py, deliberately -- it stops a red arm vacating a green
    # one). That is right for a design where each arm carries its own criterion; here
    # C2 lives in ONE arm, so a wholly-red ARM_WIN10 would vanish from the list the
    # indexer adjudicates. Re-attach the red cells' failures so an unmeasured criterion
    # is visible to the pipeline rather than silently absent.
    adj_pre = list(agg["adjudication_preconditions"])
    for g in arm_gates:
        if not g["gate_green"]:
            adj_pre.extend(g.get("applied", []))

    def _fs(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    readout: Dict[str, float] = {}
    for k, v in {
        "arf_depth_delta_pooled": c1_measured,
        "arf_theta_delta_pooled": c2_measured,
        "contacts_per_100_steps_delta": c3_measured,
        "contacts_per_100_steps_win1": h1,
        "contacts_per_100_steps_win10": h10,
        "mean_episode_length_win1": _pool(r1_rows, "mean_episode_length"),
        "mean_episode_length_win10": _pool(r10_rows, "mean_episode_length"),
        "approach_recede_gap_pooled": _pool(green_rows, "approach_recede_gap_median"),
        "arf_shallow_pooled": _pool(d_rows, "arf_shallow"),
        "c1_measurable": int(c1_measurable and gap_ok),
        "c2_measurable": int(c2_measurable and theta_ok),
        "c1_seeds_positive": len(c1_seed_pos),
        "c2_seeds_positive": len(c2_seed_pos),
        "c1_pass": int(c1_pass), "c2_pass": int(c2_pass), "c3_pass": int(c3_pass),
        "non_degenerate": int(non_degenerate),
        "n_green_cells": len(green_rows), "n_cells": len(arm_results),
        "inband_ticks_total": sum(r["n_inband_ticks"] for r in arm_results),
        "arf_full_depth_pooled": _pool(d_rows, "arf_depth_delta"),
        "wf_action_spread_median_pooled": _pool(green_rows, "wf_action_spread_median"),
        "harm_cost_range_full_depth_pooled": _pool(green_rows, "harm_cost_range_full_depth_median"),
        "grid_model_match_rate_pooled": _pool(arm_results, "grid_model_match_rate"),
    }.items():
        f = _fs(v)
        if f is not None:
            readout[k] = f
    for d in DEPTHS:
        f = _fs(_pool([{"v": r["arf_by_depth"][str(d)]} for r in d_rows], "v"))
        if f is not None:
            readout[f"arf_depth_{d}_pooled"] = f

    # red-team F8: `readout` is NaN-filtered but `criteria` was not, so an unmeasured
    # criterion wrote a bare NaN token into the field an adjudicator reads first.
    for _c in criteria:
        for _k in ("measured", "threshold"):
            _v = _c.get(_k)
            if isinstance(_v, float) and not math.isfinite(_v):
                _c[_k] = None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_class": "experimental",
        "timestamp_utc": ts,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "ethics_preflight": ETHICS_PREFLIGHT,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": agg.get("degeneracy_reason", ""),
        "per_arm_gate": agg["per_arm_gate"],
        "arm_results": arm_results,
        "warmup_by_seed": {str(k): v for k, v in warm_by_seed.items()},
        "readout": readout,
        "interpretation": {
            "label": label,
            "preconditions": adj_pre,
            "unmeasured_criteria": unmeasured,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "red_team_verdict": RED_TEAM_VERDICT,
        "notes": (
            "arf is read off the PRE-SAMPLING E3 harm term, never the sampled action, so "
            "this probe is commitment-free. APPROACH/RECEDE is grid ground truth, "
            "independent of harm_eval, so the readout is not circular. Eval runs at "
            f"num_hazards={EVAL_HAZARDS} while the warm-up trains at 1 -- a declared, "
            "arm-symmetric distribution shift that preserves cross-arm comparability but "
            "not comparability to warm-up-config numbers."),
    }

    full_config = {
        "seeds": SEEDS, "arms": ARMS, "depths": list(DEPTHS),
        "band": [BAND_LO, BAND_HI], "budgets": budgets,
        "eval_hazards": EVAL_HAZARDS,
        "arf_margin": ARF_MARGIN, "min_seeds": MIN_SEEDS,
        "warmup_config_slice": slice_ref,
    }
    return {"manifest": manifest, "config": full_config, "started_at": t0,
            "outcome": manifest["outcome"]}


def main():
    """Run the experiment and write its manifest. Returns (manifest, out_path, dry_run)."""
    ap = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    ap.add_argument("--dry-run", action="store_true", help="tiny smoke budgets")
    args = ap.parse_args()

    budgets = {
        "stage0": WARM_STAGE0, "stage0b": WARM_STAGE0B, "p0": WARM_P0,
        "hazard": WARM_HAZARD, "steps": STEPS_PER_EPISODE, "eval_eps": EVAL_EPISODES,
    }
    if args.dry_run:
        budgets = {"stage0": 1, "stage0b": 1, "p0": 2, "hazard": 2,
                   "steps": 60, "eval_eps": 2}
        globals()["SEEDS"] = [0]

    res = run_experiment(budgets, args.dry_run)
    manifest = res["manifest"]
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=res["config"],
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=res["started_at"],
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome: {manifest['outcome']}  label: {manifest['interpretation']['label']}",
          flush=True)
    print(f"manifest: {out_path}", flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry_run = main()
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
