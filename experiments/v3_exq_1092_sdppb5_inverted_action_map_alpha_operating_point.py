"""V3-EXQ-1092 -- SD-PP-B5 inverted-action-map READINESS readout at alpha_world 0.3 vs 0.9.
Does e2.world_forward's action read clear an action-sensitivity bar on a LIVE battery, and
does it differ between SD-008's damped (0.3) and stable-floor (0.9) operating points?

PURPOSE: diagnostic (validates_substrate SD-PP-B5-z-world-per-step-displacement-range).
Non-contributory to governance confidence by design; claim_ids EMPTY, bears_on names the claims.

SLEEP DRIVER: not applicable (no sleep machinery; P0 world_forward training only).

RED-TEAM (Step 4.5): see the RED-TEAM line at the end of this docstring.

WHY THIS RUN EXISTS
-------------------
The confirmed failure_autopsy_V3-EXQ-1082_2026-09-24 (targets[0].fanout_recommendation,
suggested_probes[0], hypothesis H-operating-point) routes ONE cheap measurement probe: the
inverted-action-map readout is the `zworld_action_readability_lever` live_gate that has never
been measured on a LIVE battery. H-operating-point is CONFIRMED on d_act and
skill-vs-identity by 1082 (OFF d_act 0.3415/0.2160/0.2266, CI>0 3/3; skill +0.35/+0.22/+0.21)
but its live_gate readout was unmeasured, and the causal attribution to alpha was cross-run.
This run composes V3-EXQ-1079's inverted-map env with V3-EXQ-1082's live reset-on-done
collector -- exactly what the autopsy's C1-style self-check says is available and uncovered.

THE BAR -- USER DECISION (option C), and why the ratified "~1.0" needed one
--------------------------------------------------------------------------
The autopsy's sketch reads "inverted-action-map battery MSE / original MSE with a bar
excluding the degenerate ~1.0 ... reuse readiness_verdict". `readiness_verdict` has TWO forms
with DIFFERENT nulls:
  * action_shuffle_ratio (counterfactual_battery=None): SAME rows, actions permuted in place.
    Null IS exactly 1.0 -- intrinsic transition difficulty cancels by construction.
  * battery_pair_ratio (counterfactual_battery=cf): TWO DIFFERENT batteries, one from an
    inverted-action-map env. Null is NOT 1.0 -- the two row sets differ in intrinsic |dz|,
    which moves the ratio with no action-reading at all. V3-EXQ-1079 measures that nuisance
    per cell as blind_null_1075_order (an action-blind identity predictor over the same pair).
On 1079's landed cells the raw-1.0 bar and the blind-null bar disagree on 6 of 9 cells and
route OPPOSITELY at alpha 0.9 (raw: 2/3 seeds read, 1.2563/1.2755 -> PASS; blind-null: 1/3 ->
FAIL). Since a PASS here is what makes the V3-EXQ-1073 MECH-572 contradiction design
re-posable, the bar decides whether an expensive build is authorised.
Raised as GFLAG-0470 (contested_disposition) + decision chip
chip-20260924-sdppb5-invmap-ratio-bar-decision; analysis in
REE_assembly/evidence/planning/sdppb5_inverted_map_ratio_criterion_aliasing_staged_20260924.md
(origin/master e9e37e46c2). USER CHOSE OPTION C (2026-09-24, via orchestrate-20260924-b):
  LOAD-BEARING: the SAME-ROWS action_shuffle_ratio verdict (readiness_verdict with
    counterfactual_battery=None) at RATIO_FLOOR=1.0 -- the one form where 1.0 is the true null.
  RECORDED, NEVER GATING: the cross-battery inverted-map ratio with its raw >1.0 verdict (so
    the autopsy's literal declared null stays evaluable), AND that ratio against its
    action-blind null on the same battery pair.
This resolves GFLAG-0470's contested disposition as C per the user; /governance applies the flag.

DESIGN
------
2 arms x 3 seeds = 6 cells. The ONLY between-arm difference is alpha_world.
  ARM_ALPHA_0p3  alpha_world 0.3 (REEConfig.from_dims default; SD-008's damped point)
  ARM_ALPHA_0p9  alpha_world 0.9 (SD-008's stable floor; 1082's operating point)
alpha_world is set EXPLICITLY at from_dims and asserted threaded (MECH-307 guard: from_dims
silently swallows unknown kwargs). Both arms are OFF -- no interventional margin, asserted.
Env, dims, budget (3600 P0 steps), optimiser, batch sampling and seeds are 1082's ARM_OFF.

DONE-HANDLING: 1082's `_collect_live_battery` (itself ported from V3-EXQ-1073 :1054-1056) in
BOTH the P0 rollout and every battery collector: unpack `done` from env.step, on done ->
env.reset() and prev=None, so no transition spanning a death/reset is recorded; the collector
also skips the first POST_RESET_SKIP transitions after each reset and records per-row health.
The precondition `battery_rows_all_live` requires ZERO post-death rows -- the 1075/1079 defect
(their collectors never read `done`, so every row was post-death) is what made the existing
inverted-map ratios unusable and is why this cannot be answered by reanalysis.

THE READOUTS, per cell (head = the cell's OWN trained e2.world_forward)
  LOAD-BEARING  shuffle_status   readiness_verdict(head, z0, acts, z1,
                                   counterfactual_battery=None) -> "ready" requires
                                   shuffle ratio > 1.0 AND skill > 0.0 vs copy-the-input.
  RECORDED      cross_ratio      battery_pair_ratio over (orig battery, inverted-map battery)
  RECORDED      cross_blind_null identity_predictor_mse(cf)/identity_predictor_mse(orig) --
                                 the SAME pair's action-blind nuisance
  RECORDED      cross_reads_raw  cross_ratio > 1.0            (autopsy's literal bar)
  RECORDED      cross_reads_blind cross_ratio > cross_blind_null (nuisance-corrected)
  RECORDED      d_act + paired row bootstrap CI (1082 verbatim), skill_vs_identity,
                the ridge positive control, the untrained head, persistence/model_r2.
Both batteries are collected from FRESH agents built at the SAME alpha as the cell (the
encoder IS part of the arm here, unlike 1082 where alpha was constant and one battery agent
served every arm). `encoder_equal_to_battery_agent` records that the battery agent's frozen
encoder equals the trained cell's.

VERDICT GRID -- the autopsy's declared null is a DISJUNCTION, so PASS rejects BOTH clauses:
  N1 "at 0.9 the ratio stays <= ~1.0 despite d_act > 0"  -> rejected by C1
  N2 "the 0.3 and 0.9 arms do not differ"                -> rejected by C2
  PASS inverted_map_readout_confirms_action_read_at_operating_point -- C1 and C2.
  FAIL action_read_present_at_0p9_but_alpha_contrast_undetermined -- C1 only (N2 holds).
  FAIL action_read_absent_at_operating_point_despite_d_act -- C1 fails (N1 holds); this is
       the autopsy's own first disjunct and is a real finding, not a null run.
  FAIL inverted_map_readout_undetermined -- neither resolves.
  FAIL substrate_not_ready_requeue -- any precondition unmet.
outcome PASS iff C1 and C2. Every cross-battery number is RECORDED and never gates.

DV-SYMMETRY (Step 3.5), per arm -- BOTH arms, same statement:
The DV is a ratio of MSEs of the same head over the same rows under true vs permuted actions.
Symmetry group of that DV: it is INVARIANT under any uniform positive rescaling of z_world
(numerator and denominator both scale by c^2) and under any relabeling of action indices
applied consistently. The manipulation (alpha_world) is NOT invariant under it: alpha is an
EMA blend coefficient, i.e. a temporal low-pass filter on z_world, so it changes the
per-step displacement STRUCTURE and the action-to-dz coupling, not merely z_world's scale.
Measured, not argued: across 1079's alpha arms d_act moves 0.175 -> 0.498 (seed 42) and the
burn-in ratio 1.54 -> 2.96, so the DV demonstrably moves with alpha. `rms_dz_per_dim_battery`
and `transition_l2_mean` are recorded per cell so a reader can confirm the displacement scale
changed rather than taking this on trust. Corollary, stated because it bounds the readout: a
manipulation that acted on z_world as a PURE uniform rescaling would be invisible to this DV.

MULTI-ARM GATE (Step 3.5 / V3-EXQ-785): preconditions are whole-run worst-cell here, which is
correct rather than the 785 defect -- no precondition is structurally unsatisfiable for either
arm (both arms are ordinary OFF cells differing only in a blend coefficient), so no arm's
impossible gate can vacate the other's finding. C1 is read at ARM_ALPHA_0p9 by design (that is
the operating point under test); C2 needs both arms and is cannot_determine without them.

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): the decisive readout is the inverted-map ratio on
a LIVE battery. V3-EXQ-1079 (substrate_hash 17e203f335387880) HAS the ratio at both alphas but
every row is POST-DEATH; V3-EXQ-1082 (same hash) HAS the live battery but no inverted-map
readout -- its own autopsy states the readout "is still unmeasured at 0.9". Checked run_ids
v3_exq_1073_...20260922T182856Z_v3, v3_exq_1079_...20260923T172400Z_v3,
v3_exq_1082_...20260924T045004Z_v3. Deriving it post hoc is impossible: it requires a live
inverted-map battery, which requires a run. NOT recoverable -> run.

RE-DERIVE BRAKE (Step 2.5b): does not hold. claim_ids [], a new EXQ NUMBER (not a lettered
re-run), experiment_purpose diagnostic, and the confirmed 1082 autopsy is itself the producer
half that routes this probe to /queue-experiment. GOV-DIAG-1 counts the full SD-PP-B5 token at
2 (1079 + 1082), below N=3, and the autopsy notes the chain "is converging ..., not circling".

RED-TEAM (Step 4.5, fable): see queue entry note for the verdict of record.
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
from experiments._lib.action_sensitivity_gate import (
    MIN_DISTINCT_ACTIONS as GATE_MIN_DISTINCT,
    RATIO_FLOOR, SKILL_FLOOR, battery_pair_ratio, check_canary, format_verdict,
    identity_predictor_mse, readiness_verdict)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.persistence_skill_gate import (
    BOOTSTRAP_SEED, CI_LEVEL, N_BOOTSTRAP, per_row_squared_error, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1092_sdppb5_inverted_action_map_alpha_operating_point"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1092"

CLAIM_IDS: List[str] = []
# SD-PP-B5 + MECH-573 from the chip; SD-PP-B10 + SD-008 because the 0.3-vs-0.9 alpha contrast
# is that pair's subject and the 1082 autopsy's own bears_on lists both.
BEARS_ON = ["SD-PP-B5-z-world-per-step-displacement-range",
            "SD-PP-B10-zworld-encoder-action-displacement", "MECH-573", "SD-008"]
VALIDATES_SUBSTRATE = "SD-PP-B5-z-world-per-step-displacement-range"
AUTOPSY = "failure_autopsy_V3-EXQ-1082_2026-09-24"

# Counting anchors (rows, distinct actions, live rows, alpha read-back, acts alignment) are
# reachable by construction under fixed-length uniform-random collection with reset-on-done;
# the canary anchor is the gate's own pinned-value check; and the positive-control anchor runs
# the SHIPPED `_shuffle_verdict` predicate -- the very function the load-bearing criterion
# routes on -- against a ridge head that is action-aware BY CONSTRUCTION, so the bar it must
# clear is reachable by that control by construction rather than by a hand-written narrower
# predicate (the V3-EXQ-778d failure shape).
ANCHOR_REACHABILITY_EXEMPT = (
    "counting anchors reachable by construction (random-action fixed-length live batteries "
    "with reset-on-done); canary anchor is the gate's own pinned-value check; "
    "positive-control anchor runs the shipped _shuffle_verdict predicate itself against an "
    "action-aware-by-construction ridge head")
# Load-bearing bars are seed COUNTS over three-valued verdicts, and BOTH directions of
# starvation already self-report rather than masquerading as a FAIL: C1's statistic is
# certified to have room by the positive-control precondition (same predicate, same battery);
# C2's per-seed contrast returns `cannot_determine` -- not "differ" and not "does not differ"
# -- on a degenerate or zero-width bootstrap CI, and `non_degenerate` additionally requires
# every realised CI width > 0. So a starved C2 lands as undetermined + non_degenerate:false,
# which is the distinction this check exists to protect. Note also that "the arms do not
# differ" is one of the two disjuncts of the autopsy's OWN declared null, i.e. a legitimate
# informative outcome here, not only a failure-to-measure.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "load-bearing bars are seed counts over three-valued verdicts; C1's range is certified "
    "per seed by the positive-control precondition running the same predicate, and a starved "
    "C2 self-reports as cannot_determine + non_degenerate:false rather than as a FAIL")

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

# The two arms. alpha_world EXPLICIT in both; 0.3 is from_dims' default but is still passed
# explicitly so the MECH-307 assert covers it and the manifest records it per cell.
ARMS: List[Tuple[str, float]] = [("ARM_ALPHA_0p3", 0.3), ("ARM_ALPHA_0p9", 0.9)]
OPERATING_POINT_ARM = "ARM_ALPHA_0p9"       # C1 is read here (SD-008's stable floor)

LIVE_BATTERY_N = 512        # 1082's LIVE_BATTERY_N, per the lineage
POST_RESET_SKIP = 3         # transitions skipped after each reset (EMA residual 1e-3 at 0.9)
BATTERY_RNG_XOR = 0xB477E2  # 1082's, so the action draw sequence matches the lineage
SHUFFLE_GEN_SEED = 20260924 # the shuffle permutation is stochastic -> pin it
N_SHUFFLE_DRAWS = 16        # recorded draw-variance of the shuffle ratio (never gates)
MIN_BATTERY_ROWS = 32       # persistence_skill_gate.MIN_ROWS
MIN_DISTINCT_ACTIONS = GATE_MIN_DISTINCT
RIDGE_REL_LAMBDA = 1e-3
SEEDS_REQUIRED = 2          # of 3, as 1082

_ZG = ZGoalStreamAccumulator()


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _make_env(seed: int, invert_action_map: bool = False) -> CausalGridWorldV2:
    """V3-EXQ-1079 :258-267 verbatim. The inversion swaps the two axis pairs (0<->1, 2<->3)
    and preserves every index > 3 (index 4 is the stay action at action_dim 5)."""
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
                     "buf_cap": BUF_CAP, "reset_on_done": True,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "battery": {"n_live": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                    "rng_xor": BATTERY_RNG_XOR, "invert_action_map_pair": True},
        # The shuffle permutation is STOCHASTIC, so its seed and draw count change the
        # recorded shuffle_ratio / shuffle_draw_spread. They are declared here because this
        # driver emits CROSS-DRIVER-reusable fingerprints
        # (include_driver_script_in_hash=False): under-declaring a readout-affecting
        # constant is a false-cache-HIT, which corrupts a conclusion rather than merely
        # wasting compute (arm_reuse_fingerprint_plan.md 7b; V3-EXQ-798's SSL_BIN_EDGES).
        "shuffle": {"gen_seed": SHUFFLE_GEN_SEED, "n_draws": N_SHUFFLE_DRAWS,
                    "ratio_floor": RATIO_FLOOR, "skill_floor": SKILL_FLOOR},
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
        raise RuntimeError("OFF arm must not carry the interventional margin")
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
    """LIVE battery, V3-EXQ-1082 :292-333 verbatim. done-handling ported from V3-EXQ-1073
    (:1054-1056): unpack done, on done -> env.reset() and prev=None, so the dying transition
    is never recorded. Also skips the first POST_RESET_SKIP transitions of every episode."""
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
    if not rows:
        raise RuntimeError("battery collected zero rows")
    z0 = torch.stack([r[0] for r in rows])
    acts = torch.stack([r[1] for r in rows])
    z1 = torch.stack([r[2] for r in rows])
    n_dead = sum(1 for h0, h1 in health if h0 <= 0.0 or h1 <= 0.0)
    return {"z0": z0, "acts": acts, "z1": z1, "n_resets": n_resets, "causes": causes,
            "n_postdeath_rows": n_dead, "min_health": min(min(h) for h in health),
            "max_world_state_norm": max(ws_norm),
            "mean_world_state_norm": sum(ws_norm) / len(ws_norm)}


def _battery_mse(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                 z1: torch.Tensor) -> float:
    with torch.no_grad():
        return float(((head(z0, acts) - z1) ** 2).mean().item())


def _swap_errors(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row squared error at the TRUE action, and the per-row MEAN squared error over
    every OTHER action (V3-EXQ-1079/1082 verbatim). Same rows, same targets."""
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
    """Paired row bootstrap (V3-EXQ-1079/1082 verbatim). Analytic null 0."""
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
    out["status"] = _classify(out["ci_low"], out["ci_high"], "reads_action", "swap_better")
    return out


def _contrast_ci(et_hi: torch.Tensor, es_hi: torch.Tensor, et_lo: torch.Tensor,
                 es_lo: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """CI on d_act(0.9) - d_act(0.3). Row INDICES are resampled JOINTLY (V3-EXQ-1079
    construction), which is valid only because both arms' batteries are driven by the same
    rng from the same env seed, so row i is the same (state, action) under both encoders --
    asserted by the acts_hash alignment precondition."""
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
    th, sh = et_hi[idx].sum(1), es_hi[idx].sum(1)
    tl, sl = et_lo[idx].sum(1), es_lo[idx].sum(1)
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
    """POSITIVE CONTROL (V3-EXQ-1079/1082 verbatim): z1_hat = z0 + [1, z0, a, z0 (x) a] @ W,
    ridge-fitted on the cell's own P0 buffer. Action-aware BY CONSTRUCTION, so it is the
    right positive control for an action-sensitivity statistic."""
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


def _shuffle_verdict(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor,
                     seed_offset: int) -> Any:
    """THE LOAD-BEARING readout (user option C). Same-rows action_shuffle form, where
    RATIO_FLOOR=1.0 IS the true null. Gate defaults on both floors; the permutation is
    stochastic so the generator is pinned for reproducibility."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SHUFFLE_GEN_SEED + seed_offset)
    return readiness_verdict(head, z0, acts, z1, counterfactual_battery=None,
                             min_rows=MIN_BATTERY_ROWS,
                             min_distinct_actions=MIN_DISTINCT_ACTIONS,
                             ratio_floor=RATIO_FLOOR, skill_floor=SKILL_FLOOR,
                             generator=gen)


def _shuffle_draw_spread(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                         z1: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """RECORDED, never gates: the shuffle permutation is random, so report the spread over
    N_SHUFFLE_DRAWS independent draws. A load-bearing verdict taken from ONE draw whose
    spread straddles the floor is a verdict the reader must be able to see is fragile."""
    vals: List[float] = []
    for k in range(N_SHUFFLE_DRAWS):
        v = _shuffle_verdict(head, z0, acts, z1, seed_offset + 1000 * (k + 1))
        if v.ratio is not None and math.isfinite(v.ratio):
            vals.append(float(v.ratio))
    if not vals:
        return {"n_draws": 0, "min": None, "max": None, "mean": None,
                "frac_above_floor": None}
    return {"n_draws": len(vals), "min": min(vals), "max": max(vals),
            "mean": sum(vals) / len(vals),
            "frac_above_floor": sum(1 for v in vals if v > RATIO_FLOOR) / len(vals)}


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


def _run_cell(arm: str, alpha_world: float, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(alpha_world),
                  script_path=Path(__file__), config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(seed, alpha_world)
        rng = random.Random(seed)

        # --- P0 training, reset-on-done (1073/1082 pattern), OFF objective only ---------
        opt = torch.optim.Adam(agent.e2.parameters(), lr=LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_CAP)
        obs = _reset(env)
        prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        n_train_resets = 0
        train_causes: Dict[str, int] = {}
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
            if not math.isfinite(float(loss.detach().item())):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()

        # --- LIVE batteries from FRESH agents at THIS arm's alpha -----------------------
        # alpha changes the encoder, so unlike 1082 the battery agent is per-arm. Same seed
        # -> the frozen encoder is identical to the trained cell's initial encoder.
        _, ag_b = _build(seed, alpha_world)
        enc_eq = _encoder_equal(agent, ag_b)
        bat = _collect_live_battery(ag_b, _make_env(seed),
                                    random.Random(seed ^ BATTERY_RNG_XOR), LIVE_BATTERY_N)
        _, ag_cf = _build(seed, alpha_world)
        bat_cf = _collect_live_battery(ag_cf, _make_env(seed, invert_action_map=True),
                                       random.Random(seed ^ BATTERY_RNG_XOR),
                                       LIVE_BATTERY_N)
        bz0, bacts, bz1 = bat["z0"], bat["acts"], bat["z1"]
        cf = (bat_cf["z0"], bat_cf["acts"], bat_cf["z1"])
        head = agent.e2.world_forward
        untrained = ag_b.e2.world_forward

        # --- LOAD-BEARING: same-rows shuffle verdict, bar 1.0 (its true null) -----------
        v_shuf = _shuffle_verdict(head, bz0, bacts, bz1, seed_offset=seed)
        print(format_verdict(v_shuf, f"{arm} seed={seed} [LOAD-BEARING shuffle]"),
              flush=True)
        shuf_spread = _shuffle_draw_spread(head, bz0, bacts, bz1, seed_offset=seed)

        # --- RECORDED: cross-battery inverted-map ratio, raw bar AND blind-null bar -----
        v_cross = readiness_verdict(head, bz0, bacts, bz1, counterfactual_battery=cf,
                                    min_rows=MIN_BATTERY_ROWS,
                                    min_distinct_actions=MIN_DISTINCT_ACTIONS,
                                    ratio_floor=RATIO_FLOOR, skill_floor=SKILL_FLOOR)
        cross_ratio, cross_mse_o, cross_mse_c = battery_pair_ratio(
            head, (bz0, bacts, bz1), cf)
        id_orig = identity_predictor_mse(bz0, bz1)
        id_cf = identity_predictor_mse(cf[0], cf[2])
        blind_null = (id_cf / id_orig) if id_orig > 0 else None
        cross_reads_raw = bool(cross_ratio is not None and cross_ratio > RATIO_FLOOR)
        cross_reads_blind = bool(cross_ratio is not None and blind_null is not None
                                 and cross_ratio > blind_null)

        # --- RECORDED: d_act (1082 verbatim), positive control, untrained head ----------
        et, es = _swap_errors(head, bz0, bacts, bz1)
        d_head = _d_act_ci(et, es, seed_offset=seed)
        pc = _ridge_head(list(buf))
        pt, ps = _swap_errors(pc, bz0, bacts, bz1)
        d_pc = _d_act_ci(pt, ps, seed_offset=seed + 7)
        # READINESS positive control on the SAME statistic as the load-bearing criterion.
        v_pc_shuf = _shuffle_verdict(pc, bz0, bacts, bz1, seed_offset=seed + 7)
        ut, us = _swap_errors(untrained, bz0, bacts, bz1)
        mse_init = _battery_mse(untrained, bz0, bacts, bz1)
        mse_final = _battery_mse(head, bz0, bacts, bz1)
        with torch.no_grad():
            pv = persistence_verdict(head(bz0, bacts), bz1, bz0)

        print(f"  [readout] {arm} seed={seed} alpha={alpha_world} "
              f"shuffle_ratio={v_shuf.ratio} status={v_shuf.status} | "
              f"cross_ratio={cross_ratio} blind_null={blind_null} "
              f"raw>1={cross_reads_raw} vs_blind={cross_reads_blind} | "
              f"d_act={d_head['d_act']} CI=[{d_head['ci_low']}, {d_head['ci_high']}] | "
              f"pc_shuf={v_pc_shuf.status} | postdeath={bat['n_postdeath_rows']}/"
              f"{bat_cf['n_postdeath_rows']}", flush=True)

        row: Dict[str, Any] = {
            "arm": arm, "alpha_world_requested": float(alpha_world),
            "alpha_world": float(agent.config.latent.alpha_world), "seed": seed,
            # ---- LOAD-BEARING (same-rows shuffle form; null IS 1.0) ----
            "shuffle_status": v_shuf.status,
            "shuffle_ratio": v_shuf.ratio,
            "shuffle_ratio_floor": float(RATIO_FLOOR),
            "shuffle_skill": v_shuf.skill,
            "shuffle_skill_floor": float(SKILL_FLOOR),
            "shuffle_reason": v_shuf.reason,
            "shuffle_verdict_full": v_shuf.to_dict(),
            "shuffle_draw_spread": shuf_spread,
            # ---- RECORDED: cross-battery inverted-map ratio (autopsy's literal readout) --
            "cross_ratio": cross_ratio,
            "cross_mse_original": cross_mse_o,
            "cross_mse_inverted": cross_mse_c,
            "cross_blind_null": blind_null,
            "cross_ratio_minus_blind_null": (None if (cross_ratio is None
                                                      or blind_null is None)
                                             else cross_ratio - blind_null),
            "cross_reads_raw_bar_1p0": 1 if cross_reads_raw else 0,
            "cross_reads_vs_blind_null": 1 if cross_reads_blind else 0,
            "cross_status_raw_bar": v_cross.status,
            "cross_verdict_full": v_cross.to_dict(),
            "identity_predictor_mse_original": id_orig,
            "identity_predictor_mse_inverted": id_cf,
            # ---- RECORDED: d_act and friends ----
            "d_act": d_head["d_act"], "d_act_ci_low": d_head["ci_low"],
            "d_act_ci_high": d_head["ci_high"], "d_act_status": d_head["status"],
            "pc_d_act": d_pc["d_act"], "pc_d_act_ci_low": d_pc["ci_low"],
            "pc_d_act_ci_high": d_pc["ci_high"], "pc_d_act_status": d_pc["status"],
            "pc_live": bool(d_pc["ci_low"] is not None and d_pc["ci_low"] > 0.0),
            "pc_shuffle_status": v_pc_shuf.status,
            "pc_shuffle_ratio": v_pc_shuf.ratio,
            "pc_shuffle_reads": 1 if v_pc_shuf.status == "ready" else 0,
            "d_act_untrained_head": _d_from(ut, us),
            # ---- reconstruction / readability context (never gates) ----
            "battery_mse_init": mse_init, "battery_mse_final": mse_final,
            "conv_rel_drop": (1.0 - mse_final / mse_init) if mse_init > 0 else None,
            "identity_predictor_mse": id_orig,
            "skill_vs_identity": (1.0 - mse_final / id_orig) if id_orig > 0 else None,
            "model_r2": pv.model_r2, "persistence_r2": pv.persistence_r2,
            "persistence_relative_skill": pv.relative_skill,
            "persistence_status": pv.status,
            "persistence_ci_low": pv.ci_low, "persistence_ci_high": pv.ci_high,
            # ---- live-battery hygiene, BOTH batteries ----
            "encoder_equal_to_battery_agent": enc_eq,
            "battery_hash": _hash(bz0, bacts, bz1),
            "battery_acts_hash": _hash(bacts),
            "battery_cf_acts_hash": _hash(cf[1]),
            "n_rows_battery": int(bz0.shape[0]),
            "n_rows_battery_cf": int(cf[0].shape[0]),
            "n_distinct_actions_battery": int(torch.unique(bacts, dim=0).shape[0]),
            "n_distinct_actions_battery_cf": int(torch.unique(cf[1], dim=0).shape[0]),
            "battery_n_resets": bat["n_resets"], "battery_done_causes": bat["causes"],
            "battery_n_postdeath_rows": bat["n_postdeath_rows"],
            "battery_cf_n_resets": bat_cf["n_resets"],
            "battery_cf_done_causes": bat_cf["causes"],
            "battery_cf_n_postdeath_rows": bat_cf["n_postdeath_rows"],
            "battery_min_health": bat["min_health"],
            "battery_cf_min_health": bat_cf["min_health"],
            "battery_max_world_state_norm": bat["max_world_state_norm"],
            "battery_mean_world_state_norm": bat["mean_world_state_norm"],
            # ---- displacement scale: the DV-symmetry disclosure (see docstring) ----
            "rms_dz_per_dim_battery": float(((bz1 - bz0) ** 2).mean().sqrt().item()),
            "rms_dz_per_dim_battery_cf": float(((cf[2] - cf[0]) ** 2).mean().sqrt().item()),
            "transition_l2_mean": float((bz1 - bz0).norm(dim=-1).mean().item()),
            "transition_l2_mean_cf": float((cf[2] - cf[0]).norm(dim=-1).mean().item()),
            "head_action_separation_l2_mean": _action_sep(head, bz0, bacts),
            "train_n_resets": n_train_resets, "train_done_causes": train_causes,
            "p0_buffer_rows": len(buf),
            # ---- per-row errors so every contrast is re-derivable post hoc ----
            "per_row_se_true": [float(x) for x in et.tolist()],
            "per_row_se_swap_mean": [float(x) for x in es.tolist()],
        }
        cell.stamp(row)
        _ZG.observe(agent)
    print(f"verdict: {'PASS' if v_shuf.status == 'ready' else 'FAIL'}", flush=True)
    return row


def _t(r: Dict[str, Any], k: str) -> torch.Tensor:
    return torch.tensor(r[k], dtype=torch.float64)


def _seed_analysis(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    by = {r["arm"]: r for r in rows if r["seed"] == seed}
    res: Dict[str, Any] = {"seed": seed}
    if len(by) != len(ARMS):
        res.update(valid=False, reason="cell missing")
        return res
    hi = by[OPERATING_POINT_ARM]
    lo = by["ARM_ALPHA_0p3"]
    # Pairing validity: both arms' batteries are driven by the same rng from the same env
    # seed, so the ACTION sequence must be bit-identical for the joint bootstrap to pair
    # row i with row i. Encoders differ by construction (that IS the arm), so z differs.
    aligned = bool(hi["battery_acts_hash"] == lo["battery_acts_hash"]
                   and hi["n_rows_battery"] == lo["n_rows_battery"])
    res.update(rows_aligned=aligned,
               encoder_equal=bool(hi["encoder_equal_to_battery_agent"]
                                  and lo["encoder_equal_to_battery_agent"]),
               pc_shuffle_reads=bool(hi["pc_shuffle_reads"] and lo["pc_shuffle_reads"]),
               pc_live=bool(hi["pc_live"] and lo["pc_live"]))
    res["valid"] = bool(res["encoder_equal"] and res["pc_shuffle_reads"])
    # C1: the LOAD-BEARING readout at the operating point.
    res["reads_at_operating_point"] = bool(hi["shuffle_status"] == "ready")
    res["shuffle_status_hi"] = hi["shuffle_status"]
    res["shuffle_status_lo"] = lo["shuffle_status"]
    res["shuffle_ratio_hi"] = hi["shuffle_ratio"]
    res["shuffle_ratio_lo"] = lo["shuffle_ratio"]
    res["shuffle_ratio_delta"] = (
        None if (hi["shuffle_ratio"] is None or lo["shuffle_ratio"] is None)
        else hi["shuffle_ratio"] - lo["shuffle_ratio"])
    # C2: do the arms differ? Paired joint bootstrap on d_act(0.9) - d_act(0.3), the
    # lineage's own cross-alpha contrast statistic (V3-EXQ-1079).
    if aligned:
        c = _contrast_ci(_t(hi, "per_row_se_true"), _t(hi, "per_row_se_swap_mean"),
                         _t(lo, "per_row_se_true"), _t(lo, "per_row_se_swap_mean"),
                         seed_offset=seed + 57)
    else:
        c = {"delta": None, "ci_low": None, "ci_high": None,
             "status": "cannot_determine", "n_rows": 0, "n_bootstrap": 0}
    res["alpha_contrast"] = c
    res["arms_differ"] = bool(c["status"] in ("grows", "shrinks"))
    # RECORDED cross-battery readouts at the operating point.
    res["cross_ratio_hi"] = hi["cross_ratio"]
    res["cross_blind_null_hi"] = hi["cross_blind_null"]
    res["cross_reads_raw_hi"] = bool(hi["cross_reads_raw_bar_1p0"])
    res["cross_reads_vs_blind_hi"] = bool(hi["cross_reads_vs_blind_null"])
    res["bars_agree_hi"] = bool(res["cross_reads_raw_hi"] == res["cross_reads_vs_blind_hi"])
    return res


def _worst(rows: List[Dict[str, Any]], key: str, lowest: bool = True) -> Tuple[float, str]:
    w = (min if lowest else max)(rows, key=lambda r: r[key])
    return float(w[key]), f"{w['arm']}/seed={w['seed']}"


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    canary = check_canary()
    print(f"[gate-canary] ok={canary.get('ok')} n_seeds={canary.get('n_seeds')}", flush=True)
    # Dry run: 2 seeds x BOTH arms at the FULL budget (SEEDS_REQUIRED stays reachable).
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm, alpha in ARMS:
            rows.append(_run_cell(arm, alpha, seed))

    per_seed = [_seed_analysis(rows, s) for s in seeds]
    valid = [s for s in per_seed if s["valid"]]
    for s in per_seed:
        print(f"  [seed] {s['seed']} valid={s['valid']} reads@0.9={s.get('reads_at_operating_point')} "
              f"shuf_hi={s.get('shuffle_ratio_hi')} shuf_lo={s.get('shuffle_ratio_lo')} | "
              f"alpha_contrast={s.get('alpha_contrast', {}).get('status')} "
              f"delta={s.get('alpha_contrast', {}).get('delta')} | "
              f"cross_hi={s.get('cross_ratio_hi')} blind={s.get('cross_blind_null_hi')} "
              f"raw>1={s.get('cross_reads_raw_hi')} vs_blind={s.get('cross_reads_vs_blind_hi')} "
              f"bars_agree={s.get('bars_agree_hi')}", flush=True)

    n_valid = len(valid)
    n_reads = sum(1 for s in valid if s["reads_at_operating_point"])
    n_differ = sum(1 for s in valid if s["arms_differ"])
    n_aligned = sum(1 for s in per_seed if s.get("rows_aligned"))
    n_enc = sum(1 for s in per_seed if s.get("encoder_equal"))
    n_pc_shuf = sum(1 for s in per_seed if s.get("pc_shuffle_reads"))
    n_pc_dact = sum(1 for s in per_seed if s.get("pc_live"))

    worst_rows, worst_rows_cell = _worst(rows, "n_rows_battery")
    worst_rows_cf, worst_rows_cf_cell = _worst(rows, "n_rows_battery_cf")
    worst_distinct, worst_distinct_cell = _worst(rows, "n_distinct_actions_battery")
    worst_distinct_cf, worst_distinct_cf_cell = _worst(
        rows, "n_distinct_actions_battery_cf")
    worst_dead, worst_dead_cell = _worst(rows, "battery_n_postdeath_rows", lowest=False)
    worst_dead_cf, worst_dead_cf_cell = _worst(
        rows, "battery_cf_n_postdeath_rows", lowest=False)
    # Worst |read-back alpha - requested alpha| over all cells. Computed inline rather than
    # via _worst() because it is a derived quantity, not a stored per-cell field.
    _alpha_errs = [(abs(r["alpha_world"] - r["alpha_world_requested"]),
                    f"{r['arm']}/seed={r['seed']}") for r in rows]
    alpha_err, alpha_err_cell = max(_alpha_errs) if _alpha_errs else (0.0, "none")

    preconditions = [
        {"name": "alpha_world_threaded_both_arms", "description":
         "worst cell's |alpha_world read back from the built config - requested| (must be 0)",
         "measured": float(alpha_err), "threshold": 0.0, "direction": "upper",
         "offending_cell": alpha_err_cell,
         "control": "explicit from_dims kwarg + MECH-307 assert in _build",
         "met": alpha_err <= 0.0},
        {"name": "battery_rows_all_live", "description":
         "worst cell's count of ORIGINAL-battery rows at or after agent_health <= 0 "
         "(the V3-EXQ-1075/1079 defect; must be 0)",
         "measured": float(worst_dead), "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_dead_cell, "control": "reset-on-done (1073/1082 pattern)",
         "met": worst_dead <= 0},
        {"name": "battery_cf_rows_all_live", "description":
         "worst cell's count of INVERTED-MAP-battery rows at or after agent_health <= 0 "
         "(must be 0; the inverted map changes the trajectory, so this is not implied by "
         "the original battery being live)",
         "measured": float(worst_dead_cf), "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_dead_cf_cell, "control": "same reset-on-done collector",
         "met": worst_dead_cf <= 0},
        {"name": "battery_rows", "description": "worst cell's original live-battery rows",
         "measured": float(worst_rows), "threshold": float(MIN_BATTERY_ROWS),
         "direction": "lower", "offending_cell": worst_rows_cell,
         "control": "fixed-length collection", "met": worst_rows >= MIN_BATTERY_ROWS},
        {"name": "battery_cf_rows", "description":
         "worst cell's inverted-map live-battery rows",
         "measured": float(worst_rows_cf), "threshold": float(MIN_BATTERY_ROWS),
         "direction": "lower", "offending_cell": worst_rows_cf_cell,
         "control": "fixed-length collection", "met": worst_rows_cf >= MIN_BATTERY_ROWS},
        {"name": "battery_distinct_actions", "description":
         "worst cell's distinct actions in the ORIGINAL battery. THE MONOSTRATEGY TRAP: at "
         "1 distinct action a permutation is a no-op and the ratio returns exactly 1.0, "
         "which is not action-blindness but an untestable battery",
         "measured": float(worst_distinct), "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cell,
         "control": "uniform random actions", "met": worst_distinct >= MIN_DISTINCT_ACTIONS},
        {"name": "battery_cf_distinct_actions", "description":
         "worst cell's distinct actions in the INVERTED-MAP battery (same trap)",
         "measured": float(worst_distinct_cf), "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cf_cell,
         "control": "uniform random actions",
         "met": worst_distinct_cf >= MIN_DISTINCT_ACTIONS},
        {"name": "encoder_equal_seeds", "description":
         "seeds where BOTH arms' trained cell shares the frozen encoder of its own "
         "battery agent (alpha changes the encoder, so this is checked per arm)",
         "measured": float(n_enc), "threshold": float(SEEDS_REQUIRED), "direction": "lower",
         "control": "same seed -> same init; encoder untrained in P0",
         "met": n_enc >= SEEDS_REQUIRED},
        {"name": "positive_control_shuffle_reads_seeds", "description":
         "READINESS, SAME STATISTIC AS THE LOAD-BEARING CRITERION: seeds where a ridge "
         "action-aware predictor fitted on each arm's own P0 buffer returns shuffle-form "
         "status 'ready' (ratio > 1.0) on that arm's battery, in BOTH arms. If an "
         "action-aware-by-construction head cannot clear this bar on this battery, the "
         "battery cannot test the question and a below-bar head is not evidence",
         "measured": float(n_pc_shuf), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "ridge z0 + [1, z0, a, z0 x a] @ W",
         "met": n_pc_shuf >= SEEDS_REQUIRED},
        {"name": "gate_canary_reproduces", "description":
         "action_sensitivity_gate.check_canary() reproduces the pinned V3-EXQ-1073 values "
         "(the only check that catches a PARTIALLY broken gate)",
         "measured": 1.0 if canary.get("ok") else 0.0, "threshold": 1.0,
         "direction": "lower", "control": "CANARY_V3_EXQ_1073 synthetic battery",
         "met": bool(canary.get("ok"))},
    ]
    # rows_aligned gates ONLY the C2 contrast, not C1 -- scoped, not whole-run (V3-EXQ-785).
    c2_precondition = {
        "name": "rows_aligned_seeds_for_alpha_contrast", "description":
        "seeds whose two arms' battery ACTION sequences are bit-identical, which is what "
        "licenses the paired joint bootstrap for C2. APPLIES TO C2 ONLY: C1 is a "
        "within-cell statistic and is unaffected",
        "measured": float(n_aligned), "threshold": float(SEEDS_REQUIRED),
        "direction": "lower", "control": "same env seed + same battery rng -> same draws",
        "applies_to": "C2_alpha_arms_differ", "met": n_aligned >= SEEDS_REQUIRED}

    all_pre_met = all(p["met"] for p in preconditions) and n_valid >= SEEDS_REQUIRED
    c1_met = bool(all_pre_met and n_reads >= SEEDS_REQUIRED)
    c2_met = bool(all_pre_met and c2_precondition["met"] and n_differ >= SEEDS_REQUIRED)

    if not all_pre_met:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif c1_met and c2_met:
        label, outcome = "inverted_map_readout_confirms_action_read_at_operating_point", "PASS"
    elif c1_met:
        label, outcome = "action_read_present_at_0p9_but_alpha_contrast_undetermined", "FAIL"
    elif n_reads == 0 and n_valid >= SEEDS_REQUIRED:
        label, outcome = "action_read_absent_at_operating_point_despite_d_act", "FAIL"
    else:
        label, outcome = "inverted_map_readout_undetermined", "FAIL"

    criteria = [
        {"name": "C1_reads_action_at_operating_point", "load_bearing": True,
         "measured": float(n_reads), "threshold": float(SEEDS_REQUIRED),
         "description": "valid seeds whose ARM_ALPHA_0p9 cell returns same-rows "
                        "action_shuffle readiness status 'ready' (ratio > %.2f AND skill > "
                        "%.2f). Rejects the autopsy's null clause N1." % (RATIO_FLOOR,
                                                                         SKILL_FLOOR),
         "passed": c1_met},
        {"name": "C2_alpha_arms_differ", "load_bearing": True,
         "measured": float(n_differ), "threshold": float(SEEDS_REQUIRED),
         "description": "valid, row-aligned seeds whose paired joint bootstrap CI on "
                        "d_act(0.9) - d_act(0.3) excludes 0. Rejects null clause N2.",
         "passed": c2_met},
        {"name": "C3_cross_battery_bars_agree", "load_bearing": False,
         "measured": float(sum(1 for s in valid if s.get("bars_agree_hi"))),
         "threshold": float(SEEDS_REQUIRED),
         "description": "RECORDED, NEVER GATES (GFLAG-0470): valid seeds where the "
                        "cross-battery raw >1.0 bar and the blind-null bar agree at 0.9. "
                        "Disagreement is the measured size of the aliasing this run's "
                        "criterion was moved off.",
         "passed": bool(sum(1 for s in valid if s.get("bars_agree_hi")) >= SEEDS_REQUIRED)},
    ]

    combination_rule = (
        "USER DECISION option C (2026-09-24, resolves GFLAG-0470's contested disposition; "
        "/governance applies the flag). LOAD-BEARING readout is the SAME-ROWS "
        "action_shuffle_ratio verdict from readiness_verdict(counterfactual_battery=None) "
        "at ratio_floor=%.2f and skill_floor=%.2f -- the one form whose true null IS 1.0. "
        "The autopsy's declared null is a DISJUNCTION, so PASS must reject BOTH clauses: "
        "C1 (N1) = shuffle status 'ready' at ARM_ALPHA_0p9 on >= %d valid seeds; C2 (N2) = "
        "paired joint bootstrap (%d resamples, %.0f%% CI) on d_act(0.9) - d_act(0.3) "
        "excluding 0 on >= %d valid, row-aligned seeds. outcome PASS iff C1 AND C2; C1 only "
        "-> action_read_present_at_0p9_but_alpha_contrast_undetermined; C1 failing with 0 "
        "reading seeds -> action_read_absent_at_operating_point_despite_d_act (the "
        "autopsy's own N1, a real finding); else undetermined. Any precondition unmet -> "
        "substrate_not_ready_requeue. A seed is valid iff both arms' encoders match their "
        "battery agents AND the ridge positive control clears the SAME shuffle bar in both "
        "arms. rows_aligned is scoped to C2 alone (applies_to) so an unpairable contrast "
        "cannot vacate C1. RECORDED AND NEVER GATING: the cross-battery inverted-map ratio, "
        "its raw >1.0 verdict (so the autopsy's literal declared null stays evaluable), its "
        "action-blind null on the same battery pair, the shuffle draw spread over %d "
        "permutations, d_act, skill_vs_identity, model_r2, persistence and the untrained "
        "head."
        % (RATIO_FLOOR, SKILL_FLOOR, SEEDS_REQUIRED, N_BOOTSTRAP, 100 * CI_LEVEL,
           SEEDS_REQUIRED, N_SHUFFLE_DRAWS))

    ratios = [r["shuffle_ratio"] for r in rows if r["shuffle_ratio"] is not None]
    widths = [s["alpha_contrast"]["ci_high"] - s["alpha_contrast"]["ci_low"]
              for s in valid if s.get("alpha_contrast", {}).get("ci_low") is not None
              and s["alpha_contrast"].get("ci_high") is not None]
    non_degenerate = bool(
        all_pre_met and len(set(round(v, 12) for v in ratios)) > 1
        and len(widths) > 0 and all(w > 0 for w in widths))

    def _f(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    flat: Dict[str, Any] = {
        "n_cells": len(rows), "n_valid_seeds": n_valid,
        "n_seeds_reads_at_operating_point": n_reads,
        "n_seeds_arms_differ": n_differ,
        "n_seeds_rows_aligned": n_aligned,
        "n_seeds_pc_shuffle_reads": n_pc_shuf,
        "n_seeds_pc_d_act_live": n_pc_dact,
        "n_seeds_cross_reads_raw_bar": sum(1 for s in valid
                                           if s.get("cross_reads_raw_hi")),
        "n_seeds_cross_reads_vs_blind_null": sum(1 for s in valid
                                                  if s.get("cross_reads_vs_blind_hi")),
        "n_seeds_cross_bars_agree": sum(1 for s in valid if s.get("bars_agree_hi")),
        "all_preconditions_met": 1 if all_pre_met else 0,
        "gate_canary_ok": 1 if canary.get("ok") else 0,
        "c1_reads_at_operating_point": 1 if c1_met else 0,
        "c2_alpha_arms_differ": 1 if c2_met else 0,
        "verdict_pass": 1 if outcome == "PASS" else 0,
        "verdict_c1_only": 1 if label ==
        "action_read_present_at_0p9_but_alpha_contrast_undetermined" else 0,
        "verdict_read_absent": 1 if label ==
        "action_read_absent_at_operating_point_despite_d_act" else 0,
        "verdict_undetermined": 1 if label == "inverted_map_readout_undetermined" else 0,
        "verdict_not_ready": 1 if label == "substrate_not_ready_requeue" else 0,
        "shuffle_ratio_floor": float(RATIO_FLOOR),
    }
    for s in per_seed:
        tag = f"seed{s['seed']}"
        flat[f"shuffle_ratio_alpha09_{tag}"] = _f(s.get("shuffle_ratio_hi"))
        flat[f"shuffle_ratio_alpha03_{tag}"] = _f(s.get("shuffle_ratio_lo"))
        flat[f"shuffle_ratio_delta_{tag}"] = _f(s.get("shuffle_ratio_delta"))
        ac = s.get("alpha_contrast") or {}
        flat[f"d_act_delta_09_minus_03_{tag}"] = _f(ac.get("delta"))
        flat[f"d_act_delta_ci_low_{tag}"] = _f(ac.get("ci_low"))
        flat[f"d_act_delta_ci_high_{tag}"] = _f(ac.get("ci_high"))
        flat[f"cross_ratio_alpha09_{tag}"] = _f(s.get("cross_ratio_hi"))
        flat[f"cross_blind_null_alpha09_{tag}"] = _f(s.get("cross_blind_null_hi"))
    for r in rows:
        tag = f"{r['arm'].lower()}_seed{r['seed']}"
        for k in ("shuffle_ratio", "shuffle_skill", "cross_ratio", "cross_blind_null",
                  "cross_ratio_minus_blind_null", "d_act", "d_act_ci_low", "d_act_ci_high",
                  "pc_d_act", "pc_shuffle_ratio", "d_act_untrained_head", "model_r2",
                  "persistence_r2", "persistence_relative_skill", "skill_vs_identity",
                  "conv_rel_drop", "rms_dz_per_dim_battery", "rms_dz_per_dim_battery_cf",
                  "transition_l2_mean", "transition_l2_mean_cf",
                  "head_action_separation_l2_mean", "alpha_world"):
            flat[f"{k}_{tag}"] = _f(r.get(k))
        flat[f"cross_reads_raw_bar_1p0_{tag}"] = int(r["cross_reads_raw_bar_1p0"])
        flat[f"cross_reads_vs_blind_null_{tag}"] = int(r["cross_reads_vs_blind_null"])
        flat[f"pc_shuffle_reads_{tag}"] = int(r["pc_shuffle_reads"])
    flat = {k: v for k, v in flat.items() if v is not None}

    manifest: Dict[str, Any] = {
        "queue_id": QUEUE_ID,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "bears_on": BEARS_ON,
        "bears_on_provenance": (
            "SD-PP-B5 + MECH-573 from chip-20260924-sdppb5-inverted-map-probe; "
            "SD-PP-B10 + SD-008 because the 0.3-vs-0.9 alpha contrast is that pair's "
            "subject and the 1082 autopsy's own bears_on lists both."),
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
        "arms": [{"arm": a, "alpha_world": al} for a, al in ARMS],
        "operating_point_arm": OPERATING_POINT_ARM,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate else
            "preconditions unmet, or shuffle ratios identical across cells / zero-width "
            "alpha-contrast CIs"),
        "gate_canary": canary,
        "criterion_provenance": (
            "Bar chosen by the USER (option C, 2026-09-24) on decision chip "
            "chip-20260924-sdppb5-invmap-ratio-bar-decision, raised because the 1082 "
            "autopsy's ~1.0 bar is the null of the SAME-ROWS shuffle form, not of the "
            "cross-battery form its sketch specifies. Resolves GFLAG-0470's contested "
            "disposition as C; /governance applies the flag. Analysis: REE_assembly "
            "evidence/planning/"
            "sdppb5_inverted_map_ratio_criterion_aliasing_staged_20260924.md (e9e37e46c2)."),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "readout": flat,
        "arm_results": rows,
        "per_seed_analysis": per_seed,
        "interpretation": {
            "label": label,
            "combination_rule": combination_rule,
            "preconditions": preconditions + [c2_precondition],
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
        config={"arms": [{"arm": a, "alpha_world": al} for a, al in ARMS],
                "operating_point_arm": OPERATING_POINT_ARM, "seeds": SEEDS,
                "p0_steps": P0_STEPS, "steps_per_episode": STEPS_PER_EPISODE,
                "episodes_per_run": EPISODES_PER_RUN, "batch_k": BATCH_K, "lr": LR,
                "min_buf_before_train": MIN_BUF_BEFORE_TRAIN, "buf_cap": BUF_CAP,
                "max_grad_norm": MAX_GRAD_NORM,
                "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
                "live_battery_n": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                "battery_rng_xor": BATTERY_RNG_XOR,
                "shuffle_gen_seed": SHUFFLE_GEN_SEED,
                "n_shuffle_draws": N_SHUFFLE_DRAWS,
                "ratio_floor": RATIO_FLOOR, "skill_floor": SKILL_FLOOR,
                "min_battery_rows": MIN_BATTERY_ROWS,
                "min_distinct_actions": MIN_DISTINCT_ACTIONS,
                "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                "ridge_rel_lambda": RIDGE_REL_LAMBDA,
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
