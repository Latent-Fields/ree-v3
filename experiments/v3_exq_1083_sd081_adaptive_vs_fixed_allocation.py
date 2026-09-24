"""
V3-EXQ-1083 -- SD-081 adaptive-vs-fixed allocation falsifier, MODEL-UNCERTAINTY HALF ONLY:
does uncertainty-driven dual-system arbitration track E2 model uncertainty, REACH committed
selection, and BEAT a matched fixed mixture?

Claims: SD-081 (e3.dualsystem_uncertainty_arbitration)
FALSIFIER SCOPE: this run bears ONLY on the model-uncertainty half of SD-081's registered
falsifier. The familiarity half (u_habit) is NOT tested and stays OPEN -- see SCOPE below.
Red-team (see queue note): verdict recorded in the queue entry note and on the line below.
RED-TEAM (fable, 2026-09-24): CONTESTED -> fixed F1-F4 (orchestrator chose option A, then
  Q-1083b option A): F1 the familiarity axis proved untestable on this substrate (below), so it
  is DROPPED rather than repaired; P4 gated PER SEED (>= 2/3 seeds); F2 C3 scored only over
  cells passing the per-cell single-path separation bar, steps survived reported beside return;
  F3 C2-fail with complete reach routes reach_complete_dose_insufficient (mixed), not "does not
  reach" (weakens); F4 MODEL_PERTURB_REL_SIGMA = 2.0, pre-registered from an out-of-sample dose
  probe (seeds 100/101; see the constant's comment).

===========================================================================
WHY THIS EXPERIMENT EXISTS
===========================================================================
SD-081 was built 2026-07-22 (ree-v3 4472811) to make MECH-477 falsifiable. Its only
experimental lineage so far is V3-EXQ-811 / 811a (MECH-477 OFF-vs-ON, 811a PASS
2026-07-24). The 2026-09-21 falsifier-completion audit
(REE_assembly/evidence/planning/falsifier_completion_and_runnability_audit_20260921.json,
disposition "EXISTING SUBSTRATE, EXPERIMENT DOES NOT USE IT") registered a what_would_answer
for SD-081 ITSELF that 811a does not discharge:

  1. NO FIXED-MIXTURE OR SHUFFLED CONTROL. 811a's OFF arm is w == 1 (planned-only), which
     cannot distinguish "the weight tracks uncertainty" from "any habit admixture changes
     behaviour". The falsifier names a FIXED mixture at matched mean weight, a SHUFFLED weight
     series and single-path references.
  2. NO COMMITTED-SELECTION READ. 811a's recruitment DV came from a different scorer from the
     E3 score_trajectory the arbitrator blends. Here, per E3 tick: the habit argmin, the
     planned argmin and e3.last_selected_idx.
  3. NO INDEPENDENT MODEL-UNCERTAINTY MANIPULATION. 811a varied only familiarity.

SCOPE -- WHY THE FAMILIARITY AXIS IS DROPPED (orchestrator Q-1083b, option A). u_habit is
1 - familiarity_tracker.query(z_world) (agent.py SD-081 block); the SD-025 tracker is a
128-anchor FIFO written on every waking step. Measured 2026-09-24 on seeds outside SEEDS:
it turns over ~10x during practice (209/216/350/524 anchor allocations per practice layout),
so FROZEN at practice end it reads u_habit 0.989-1.000 on familiar and 0.979-1.000 on novel
layouts (seed 102), and LIVE it tracks recency (dry run: per-seed AUC novel>familiar 0.0/0.25,
inverted). Neither setting carries practice familiarity, so a familiarity contrast here would
be non_contributory by construction. This run therefore says NOTHING about whether the weight
tracks habit (familiarity) uncertainty; that half needs a practice-retaining familiarity read
first (raised to governance). The tracker is left live and identical across arms: u_habit is
still an input to w, it is just not manipulated.

===========================================================================
DESIGN
===========================================================================
One agent per seed is practised exactly as in the 811a lineage (canonical module
experiments/_lib/baselines/mech477_dualsystem_arbitration.py: 4 familiar layouts x 30
episodes x 120 steps, arbitration ON, curiosity_weight 0.05 so the familiarity tracker
the arbitrator reads exists). The practised agent is then deep-copied into FIVE arms, so
every arm has BIT-IDENTICAL learned weights -- "matching the two underlying scorer
pathways" as the falsifier requires. Arms differ ONLY in the allocation rule, set through
the registered E3Config knobs (gain 0 makes w = sigmoid(bias), a constant):

  adaptive      gain 4.0, bias 0.0 -- the registered SD-081 rule.
  fixed         gain 0, w = mean of THIS SEED's adaptive w over scored ticks.
  shuffled      gain 0, w per E3 tick = a seeded permutation of THIS SEED's adaptive w
                series -- same marginal distribution, decoupled from context.
  habit_only    gain 0, bias -12 (w ~ 6e-6): single-path reference.
  planned_only  gain 0, bias +12 (w ~ 1 - 6e-6): single-path reference.

Every arm resets all RNG to the seed at arm entry, so arms see matched stochasticity.

CONTEXT CELLS -- two, on the 4 PRACTISED layouts:
  model        intact vs perturbed E2 world_transition. u_planned is E3._running_variance,
               the EMA of || z_world_actual - trajectory.world_states[1] ||^2 where
               world_states come from E2 rollouts (e3_selector.post_action_update, reached
               via agent.update_residue). Fixed, seeded Gaussian noise is added to the E2
               world_transition weights (relative scale MODEL_PERTURB_REL_SIGMA = 2.0 of each
               tensor's mean |w|), applied per episode and restored bit-exactly afterwards.
               The dose was chosen BEFORE the run by a probe on seeds outside SEEDS (per-seed
               AUC u_planned perturbed>intact 0.72/0.70 at 0.5, 0.86/0.64 at 1.0, 0.91/0.92 at
               2.0; red-team F4). (Env hazard drift was measured during authoring NOT to raise
               u_planned, so it is not used.)

Probe schedule per arm: burn-in of one episode per cell (unscored; lets the arbitration
EMAs settle on the mixed schedule after they are reset to None at arm entry), then
PROBE_REPS reps x 4 layout indices x 2 cells, cell order rotated by (rep + layout) so each
cell appears in each ordinal position equally often (12 scored episodes per cell per arm).

===========================================================================
PRE-REGISTERED CRITERIA (all thresholds fixed here, never derived from the run)
===========================================================================
Readiness preconditions (the core gate -- C1..C3 are CROSS-ARM contrasts on one shared
practised agent per seed, so a red core gate vacates all of them):
  P1  pathways_differ_on_choices -- adaptive arm, fraction of scored ticks with
      argmin(habit) != argmin(planned), WORST seed >= 0.20. The falsifier's
      "differ on some choices".
  P1b habit_score_range_non_degenerate -- worst scored tick, cross-candidate range of the
      habit vector > 1e-6 (the 786a constant-vector defect, stated as its own statistic).
  P2  arbitration_live_on_familiarity -- adaptive arm, fraction of E3 ticks that produced a
      non-degenerate arbitration WITH u_habit source "familiarity", worst seed >= 0.95 (the
      arbitration path is live; it does not certify that u_habit carries familiarity).
  P4  model_manipulation_reaches_u_planned -- per seed, AUC(episode-mean raw u_planned,
      perturbed > intact), adaptive arm; >= 2/3 of the INTENDED seeds must reach 0.70 (per
      seed, matching C1's per-seed sign leg; an unmeasured seed fails).
  P7  seeds_with_reach_data -- seeds whose adaptive AND fixed arms each log >= 5
      habit/planned-disagreement ticks in EVERY cell; >= 6 of 8 required.
Benefit-leg precondition (vacates C3 ONLY):
  P5  single_path_outcome_separation -- max over the 2 cells of |mean over seeds of
      (return planned_only - return habit_only)| >= 2 * BENEFIT_FLOOR.

  C1 (load-bearing) allocation_tracks_model_uncertainty -- adaptive arm, per seed:
     dw_model = w(intact) - w(perturbed). Must clear mean - 1.0*SEM > 0.02 with >= 2/3 of
     seeds positive. Necessary, NOT sufficient: "the sigmoid formula alone is not
     supporting evidence".
  C2 (load-bearing) committed_recruitment_beyond_fixed -- PA = P(selected == argmin
     planned | argmin habit != argmin planned). Per seed, DID_model = [PA_ad(I) - PA_ad(P)]
     - [PA_fx(I) - PA_fx(P)]. Must clear mean - 1.0*SEM > 0.03 with >= 2/3 of seeds
     positive. The fixed arm subtracts whatever perturbation-driven change in candidate
     geometry would move PA with NO change in allocation.
  C3 (load-bearing, gated by P5) benefit_over_fixed -- per seed, d = mean over the C3-SCORED
     CELLS of [mean episode return (adaptive) - mean episode return (fixed)]. The scored
     cells are those where |mean over seeds (planned_only - habit_only return)| >=
     2 * BENEFIT_FLOOR -- the per-cell form of P5 (red-team F2: return is death-censored,
     so cells where the pure pathways do not separate cannot carry a benefit contrast).
     Must clear mean - 1.0*SEM > BENEFIT_FLOOR with >= 2/3 of seeds positive. Steps
     survived per arm and cell are recorded beside return.
  C4 (secondary) the same C2/C3 contrasts against the SHUFFLED arm. Reported, not routed.

  Combination rule: SUPPORTS (model-uncertainty half) iff core gate green AND C1 AND C2 AND
  P5 AND C3.

ROUTING (every direction applies to the MODEL-UNCERTAINTY HALF of SD-081's falsifier only):
  core gate red                          -> FAIL substrate_not_ready_requeue, non_contributory
  C1 fails                                -> FAIL allocation_does_not_track_model_uncertainty, weakens
  C1 passes, C2 fails, reach complete     -> FAIL reach_complete_dose_insufficient, mixed
                                             (reach = selected == argmin(arbitrated) on >= 0.95
                                             of adaptive ticks, worst seed; red-team F3)
  C1 passes, C2 fails, DID mean <= 0      -> FAIL allocation_does_not_reach_committed_selection, weakens
  C1 passes, C2 fails otherwise           -> FAIL reach_equivocal, mixed
  C1, C2 pass, P5 unmet                   -> FAIL reach_confirmed_benefit_untestable, non_contributory
  C1, C2 pass, P5 met, C3 passes          -> PASS adaptive_allocation_reaches_selection_and_beats_fixed_model_axis, supports
  C1, C2 pass, P5 met, C3 fails, mean d <= 0 -> FAIL adaptive_allocation_no_benefit_over_fixed, weakens
  C1, C2 pass, P5 met, C3 fails otherwise -> FAIL benefit_equivocal, mixed

NULL DECLARATIONS. A non_contributory here says the instrument or the model manipulation was
not ready -- it says nothing about SD-081. A C3-only null with P5 unmet says the two pure
pathways do not differ behaviourally on this substrate. NO outcome of this run says anything
about the familiarity (habit-uncertainty) half of SD-081's falsifier.

===========================================================================
DV-SYMMETRY DECLARATION (per arm)
===========================================================================
PA is an indicator over argmins: invariant to any monotone transform of the BLENDED score
vector, and to a uniform additive constant. The manipulations are (a) the weight w, which
re-weights two DIFFERENT standardised vectors -- (1-w) z_h + w z_p is not a monotone remap
of either unless the two rank identically, and P1 requires they do not on >= 20% of ticks;
(b) the model cells, which change the world states the candidates roll out through.
Neither is in the DV's symmetry group, for every arm. The weight w itself (C1) is a
strictly monotone function of (u_h_n - u_p_n); its cross-cell difference is not an identity
because u_p_n is an EMA-normalised reading of a raw signal the cells move (P4 certifies the
raw movement). Episode return (C3) is a sum over realised transitions -- no symmetry of it is
induced by any arm's allocation rule.

SLEEP: not used, so no SLEEP DRIVER line applies.
PHASED TRAINING: not required -- SD-081 has no learned parameters; the only training is
warmup_train's own schedule, identical to the 811a lineage. Nothing is trained during probes.
ETHICS PREFLIGHT (documentation habit, non-enforced): all involvement flags false,
decision: allow.
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402

from _lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from _lib.baselines import mech477_dualsystem_arbitration as LIN  # noqa: E402
from _lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from _lib.robustness_bars import robust_by_sem  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1083_sd081_adaptive_vs_fixed_allocation"
QUEUE_ID = "V3-EXQ-1083"
CLAIM_IDS = ["SD-081"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---------------------------------------------------------------------------
# Pre-registered constants.
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

ARM_ADAPTIVE = "adaptive"
ARM_FIXED = "fixed"
ARM_SHUFFLED = "shuffled"
ARM_HABIT = "habit_only"
ARM_PLANNED = "planned_only"
ARMS = (ARM_ADAPTIVE, ARM_FIXED, ARM_SHUFFLED, ARM_HABIT, ARM_PLANNED)
SINGLE_PATH_BIAS = 12.0          # |logit| for the single-path references (w ~ 6e-6 / 1-6e-6)

CELLS: Tuple[Tuple[str, str], ...] = (
    ("familiar", "intact"),
    ("familiar", "perturbed"),
)
# SCOPE (orchestrator Q-1083b option A): the familiarity axis is DROPPED. Probe
# (seeds 102, 100, 101, outside SEEDS): the SD-025 familiarity tracker's 128-anchor FIFO turns
# over ~10x during practice (209/216/350/524 allocations per practice layout), so at probe
# time u_habit ~= 1.0 on familiar AND novel layouts alike (0.989-1.000 vs 0.979-1.000);
# live, it tracks recency instead. No setting makes u_habit carry PRACTICE familiarity, so
# only the model-uncertainty axis is manipulated here, on the practised layouts.
PROBE_REPS = 3
# Noise sd as a fraction of each tensor's mean |w|. Pre-registered from an out-of-sample dose
# probe (red-team F4; seeds 100/101, full practice, adaptive arm, 1 rep x 4 layouts): per-seed
# AUC(u_planned perturbed > intact) was 0.72/0.70 at 0.5, 0.86/0.64 at 1.0, 0.91/0.92 at 2.0.
# 0.5 sits ON the 0.70 bar for both seeds; 2.0 is the only dose that clears it robustly.
MODEL_PERTURB_REL_SIGMA = 2.0

# Readiness floors.
PATHWAY_DISAGREE_FRAC_FLOOR = 0.20
SCORE_RANGE_FLOOR = LIN.SCORE_RANGE_FLOOR          # 1e-6
ARB_LIVE_FAMILIARITY_FRAC_FLOOR = 0.95
U_PLANNED_AUC_FLOOR = 0.70
MIN_DISAGREE_TICKS_PER_CELL = 5
MIN_SEEDS_WITH_REACH_DATA = 6
# F3 (red-team): committed selection counts as fully REACHED by the arbitrated vector when
# selected == argmin(arbitrated) on >= this fraction of non-degenerate ticks, worst seed.
REACH_COMPLETE_FLOOR = 0.95

# Criteria.
SEM_K = 1.0
SEM_MIN_N = 3
SEED_POSITIVE_FRACTION_FLOOR = 2.0 / 3.0
W_SHIFT_FLOOR = 0.02              # C1: two percentage points of reallocation
PA_DID_FLOOR = 0.03               # C2
# C3 absolute floor in episode-return units (return = sum of the env's signed
# harm_signal). 0.10 is one fifth of a full hazard contact in the unscaled env
# (hazard_harm 0.5 default; fishtank kwargs scale contact to 0.05 plus proximity
# harm 0.1 * field) and a third of a resource contact (0.3): the smallest per-episode
# change that corresponds to a whole behavioural event rather than field noise.
BENEFIT_FLOOR = 0.10
SINGLE_PATH_SEPARATION_MULT = 2.0

ARB_SAMPLE_CAP = 150


# ---------------------------------------------------------------------------
# Small statistics helpers.
# ---------------------------------------------------------------------------
def _finite(vals) -> List[float]:
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def _mean(vals) -> Optional[float]:
    xs = _finite(vals)
    return float(statistics.fmean(xs)) if xs else None


def _auc_greater(pos: List[float], neg: List[float]) -> Optional[float]:
    return LIN.auc_greater(_finite(pos), _finite(neg))


def _logit(w: float) -> float:
    w = min(max(float(w), 1e-6), 1.0 - 1e-6)
    return math.log(w / (1.0 - w))


def _seed_leg(vals: List[Optional[float]]) -> Dict[str, Any]:
    xs = _finite(vals)
    n_pos = sum(1 for v in xs if v > 0.0)
    frac = (float(n_pos) / float(len(xs))) if xs else 0.0
    return {"n": len(xs), "n_positive": n_pos, "positive_fraction": frac,
            "passes": bool(xs and frac >= SEED_POSITIVE_FRACTION_FLOOR)}


def _criterion_leg(vals: List[Optional[float]], margin: float) -> Dict[str, Any]:
    bar = robust_by_sem(_finite(vals), margin=margin, k=SEM_K, min_n=SEM_MIN_N)
    seed = _seed_leg(vals)
    xs = _finite(vals)
    return {
        "per_seed": vals,
        "mean": _mean(xs),
        "sem_bar": bar,
        "seed_leg": seed,
        "measured": (float(bar.get("mean", 0.0)) - SEM_K * float(bar.get("sem", 0.0)))
        if bar.get("mean") is not None and bar.get("sem") is not None else None,
        "threshold": margin,
        "passes": bool(bar.get("passes", False) and seed["passes"]),
    }


# ---------------------------------------------------------------------------
# Instrumentation: capture the habit vector and the arbitration on real E3 ticks.
# ---------------------------------------------------------------------------
class ArbTap:
    """Instance-level wrappers on ONE agent's E3 selector.

    _arbitrate_dual_system is wrapped to capture the planned (input) and arbitrated
    (output) score vectors; score_trajectory is wrapped to capture the habit vector
    EXACTLY as the substrate computes it (the calls made while _score_depth_limit is
    set, i.e. inside the habit pass), rather than reconstructing it algebraically.
    `current` is cleared by the driver immediately before every select_action and is
    only ever filled from inside a real select(), so a latched non-E3 tick records
    nothing.
    """

    def __init__(self, agent) -> None:
        self.e3 = agent.e3
        self.current: Optional[Dict[str, Any]] = None
        self._habit_buf: Optional[List[torch.Tensor]] = None
        orig_arb = self.e3._arbitrate_dual_system
        orig_score = self.e3.score_trajectory
        tap = self

        def score_wrapped(*a, **kw):
            out = orig_score(*a, **kw)
            if tap._habit_buf is not None and tap.e3._score_depth_limit is not None:
                tap._habit_buf.append(out.detach())
            return out

        def arb_wrapped(candidates, planned_scores, habit_uncertainty,
                        habit_uncertainty_source, **kw):
            tap._habit_buf = []
            try:
                out = orig_arb(candidates, planned_scores, habit_uncertainty,
                               habit_uncertainty_source, **kw)
            finally:
                habit_list = tap._habit_buf
                tap._habit_buf = None
            arb = dict(tap.e3.last_arbitration or {})
            row: Dict[str, Any] = {
                "degenerate": bool(arb.get("degenerate", True)),
                "degeneracy_reason": arb.get("degeneracy_reason", ""),
                "source": str(arb.get("habit_uncertainty_source") or "none"),
            }
            if not row["degenerate"] and habit_list and len(habit_list) == len(candidates):
                p = planned_scores.detach().flatten().double()
                a_ = out.detach().flatten().double()
                h = torch.stack(habit_list).mean(dim=-1).flatten().double()
                row.update({
                    "w": float(arb["w_planned"]),
                    "u_h": float(arb["u_habit_raw"]),
                    "u_p": float(arb["u_planned_raw"]),
                    "u_h_n": float(arb["u_habit_norm"]),
                    "u_p_n": float(arb["u_planned_norm"]),
                    "argmin_p": int(p.argmin().item()),
                    "argmin_h": int(h.argmin().item()),
                    "argmin_arb": int(a_.argmin().item()),
                    "habit_range": float((h.max() - h.min()).item()),
                    "planned_range": float((p.max() - p.min()).item()),
                })
            elif not row["degenerate"]:
                row["degenerate"] = True
                row["degeneracy_reason"] = "habit_vector_not_captured"
            tap.current = row
            return out

        self.e3.score_trajectory = score_wrapped
        self.e3._arbitrate_dual_system = arb_wrapped


def _set_allocation(agent, gain: float, bias: float) -> None:
    cfg = agent.e3.config
    cfg.dualsystem_arbitration_gain = float(gain)
    cfg.dualsystem_arbitration_bias = float(bias)


def _make_perturbation(agent, seed: int) -> List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]]:
    """Fixed, seeded noise for every E2 world_transition tensor: (param, original, noise)."""
    gen = torch.Generator().manual_seed(int(seed) * 7919 + 13)
    out = []
    for p in agent.e2.world_transition.parameters():
        orig = p.detach().clone()
        scale = MODEL_PERTURB_REL_SIGMA * float(orig.abs().mean().item())
        noise = torch.randn(orig.shape, generator=gen, dtype=orig.dtype) * scale
        out.append((p, orig, noise))
    return out


def _apply_model(pert, perturbed: bool) -> None:
    with torch.no_grad():
        for p, orig, noise in pert:
            p.copy_(orig + noise if perturbed else orig)


def _build_schedule(reps: int, n_layouts: int) -> List[Tuple[int, int, Tuple[str, str], bool]]:
    """(rep, layout_idx, cell, scored). Burn-in = one episode per cell at layout 0."""
    sched = [(-1, 0, c, False) for c in CELLS]
    for r in range(reps):
        for li in range(n_layouts):
            k = (r + li) % len(CELLS)
            for c in CELLS[k:] + CELLS[:k]:
                sched.append((r, li, c, True))
    return sched


def _run_episode(agent, tap: ArbTap, env_seed: int, env_kwargs: Dict[str, Any],
                 shuffled_ws: Optional[List[float]], shuffled_pos: List[int]) -> Dict[str, Any]:
    env = CausalGridWorldV2(seed=env_seed, **env_kwargs)
    _flat, obs_dict = env.reset()
    agent.reset()
    ticks_rows: List[Dict[str, Any]] = []
    n_e3 = 0
    n_latched = 0
    ret = 0.0
    n_harm = 0
    n_benefit = 0
    steps = 0
    done_cause = ""
    for _step in range(LIN.STEPS_PER_EPISODE):
        body = obs_dict.get("body_state")
        world = obs_dict.get("world_state")
        if body is None or world is None:
            break
        latent = agent.sense(
            obs_body=body, obs_world=world,
            obs_harm=obs_dict.get("harm_obs"), obs_harm_a=obs_dict.get("harm_obs_a"),
            obs_harm_history=obs_dict.get("harm_history"),
        )
        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, wdim, device=agent.device))
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        is_e3 = bool(ticks.get("e3_tick", False))
        if is_e3 and shuffled_ws:
            _set_allocation(agent, 0.0, _logit(shuffled_ws[shuffled_pos[0] % len(shuffled_ws)]))
            shuffled_pos[0] += 1
        # Latch discipline: clear every diagnostic read below IMMEDIATELY before the call.
        tap.current = None
        agent.e3.last_arbitration = None
        agent.e3.last_selected_idx = None
        action = agent.select_action(candidates, ticks)
        if is_e3:
            n_e3 += 1
            row = tap.current
            sel = agent.e3.last_selected_idx
            if row is None or sel is None:
                n_latched += 1
            else:
                row = dict(row)
                row["sel"] = int(sel)
                ticks_rows.append(row)
        if action is None or not torch.isfinite(action).all():
            act_idx = int(np.random.randint(0, int(env.action_dim)))
        else:
            act_idx = int(action[0].argmax().item())
        _flat, harm_signal, done, info, obs_dict = env.step(act_idx)
        hs = float(harm_signal)
        ret += hs
        n_harm += int(hs < 0.0)
        n_benefit += int(hs > 0.0)
        steps += 1
        with torch.no_grad():
            agent.update_residue(harm_signal=hs, world_delta=None,
                                 hypothesis_tag=False, owned=True)
        if done:
            done_cause = str(info.get("done_cause", "")) if isinstance(info, dict) else ""
            break
    return {"ticks": ticks_rows, "n_e3_ticks": n_e3, "n_latched_ticks": n_latched,
            "return": ret, "n_harm_steps": n_harm, "n_benefit_steps": n_benefit,
            "steps": steps, "done_cause": done_cause}


def _clone_agent(base_agent):
    """Deep-copy one practised agent into an arm with BIT-IDENTICAL weights.

    Two things plain copy.deepcopy cannot do on a practised REEAgent, handled here:
    (1) some attribute holds a Python module object -- pre-seeding the memo with every
    loaded module makes those shared references rather than copies; (2) after
    warmup_train the agent caches NON-LEAF tensors (autograd history attached), whose
    __deepcopy__ torch refuses. For the duration of THIS copy only, a non-leaf tensor
    is copied as detach().clone() -- value-identical, history dropped, which is
    harmless because nothing is trained during probes. torch's own __deepcopy__ is
    restored in `finally`, so no other code in the process ever sees the patch.
    """
    memo = {id(m): m for m in list(sys.modules.values()) if m is not None}
    orig_dc = torch.Tensor.__deepcopy__

    def _dc(self, memo_):
        if not self.is_leaf:
            out = self.detach().clone()
            memo_[id(self)] = out
            return out
        return orig_dc(self, memo_)

    torch.Tensor.__deepcopy__ = _dc
    try:
        return copy.deepcopy(base_agent, memo)
    finally:
        torch.Tensor.__deepcopy__ = orig_dc


def _run_arm(base_agent, seed: int, arm_id: str, reps: int, n_layouts: int, adaptive_ws: List[float],
             zg: ZGoalStreamAccumulator, progress: List[int], eps_total: int) -> Dict[str, Any]:
    agent = _clone_agent(base_agent)
    reset_all_rng(seed)                       # matched stochasticity across arms
    agent.e3._arb_ema_habit = None            # every arm starts its EMAs from nothing
    agent.e3._arb_ema_planned = None
    # The familiarity tracker is left LIVE (substrate default), identical in every arm: u_habit
    # is still the arbitrator's own habit-uncertainty input, but no experiment contrast is
    # built on it (see the CELLS scope note).
    tap = ArbTap(agent)
    pert = _make_perturbation(agent, seed)
    shuffled_ws: Optional[List[float]] = None
    fixed_w: Optional[float] = None
    if arm_id == ARM_ADAPTIVE:
        _set_allocation(agent, LIN.ARB_GAIN, LIN.ARB_BIAS)
    elif arm_id == ARM_FIXED:
        fixed_w = _mean(adaptive_ws)
        _set_allocation(agent, 0.0, _logit(fixed_w if fixed_w is not None else 0.5))
    elif arm_id == ARM_SHUFFLED:
        rng = np.random.RandomState(int(seed) + 4242)
        shuffled_ws = list(np.asarray(adaptive_ws, dtype=float)[rng.permutation(len(adaptive_ws))]) \
            if adaptive_ws else [0.5]
        _set_allocation(agent, 0.0, 0.0)
    elif arm_id == ARM_HABIT:
        _set_allocation(agent, 0.0, -SINGLE_PATH_BIAS)
    elif arm_id == ARM_PLANNED:
        _set_allocation(agent, 0.0, SINGLE_PATH_BIAS)
    else:
        raise ValueError(arm_id)

    episodes: List[Dict[str, Any]] = []
    shuffled_pos = [0]
    for rep, li, (fam, model), scored in _build_schedule(reps, n_layouts):
        env_seed = (LIN.FAMILIAR_ENV_SEEDS if fam == "familiar" else LIN.NOVEL_ENV_SEEDS)[li]
        env_kwargs = LIN.FAMILIAR_ENV_KWARGS if fam == "familiar" else LIN.NOVEL_ENV_KWARGS
        _apply_model(pert, model == "perturbed")
        ep = _run_episode(agent, tap, env_seed, env_kwargs, shuffled_ws, shuffled_pos)
        _apply_model(pert, False)
        ep.update({"rep": rep, "layout_idx": li, "familiarity": fam, "model": model,
                   "scored": scored, "env_seed": env_seed})
        episodes.append(ep)
        progress[0] += 1
        print(f"  [train] seed={seed} arm={arm_id} probe ep {progress[0]}/{eps_total}",
              flush=True)
    zg.observe(agent)
    return {"arm_id": arm_id, "episodes": episodes, "fixed_w": fixed_w}


def _cell_summary(episodes: List[Dict[str, Any]], fam: Optional[str], model: Optional[str]) -> Dict[str, Any]:
    eps = [e for e in episodes if e["scored"]
           and (fam is None or e["familiarity"] == fam)
           and (model is None or e["model"] == model)]
    rows = [t for e in eps for t in e["ticks"]]
    nd = [t for t in rows if not t["degenerate"]]
    disagree = [t for t in nd if t["argmin_h"] != t["argmin_p"]]
    pa = (float(sum(1 for t in disagree if t["sel"] == t["argmin_p"])) / len(disagree)) if disagree else None
    return {
        "n_episodes": len(eps),
        "n_e3_ticks": int(sum(e["n_e3_ticks"] for e in eps)),
        "n_latched_ticks": int(sum(e["n_latched_ticks"] for e in eps)),
        "n_ticks_recorded": len(rows),
        "n_nondegenerate": len(nd),
        "n_familiarity_source": sum(1 for t in nd if t["source"] == "familiarity"),
        "n_disagree": len(disagree),
        "disagree_frac": (float(len(disagree)) / len(nd)) if nd else None,
        "planned_agreement": pa,
        "sel_equals_arbitrated_argmin_frac": (
            float(sum(1 for t in nd if t["sel"] == t["argmin_arb"])) / len(nd)) if nd else None,
        "w_mean": _mean(t["w"] for t in nd),
        "u_h_mean": _mean(t["u_h"] for t in nd),
        "u_p_mean": _mean(t["u_p"] for t in nd),
        "min_habit_range": (min(t["habit_range"] for t in nd) if nd else None),
        "return_mean": _mean(e["return"] for e in eps),
        "steps_mean": _mean(e["steps"] for e in eps),
        "harm_steps_mean": _mean(e["n_harm_steps"] for e in eps),
        "benefit_steps_mean": _mean(e["n_benefit_steps"] for e in eps),
    }


def _episode_means(episodes, key_axis: str, key: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for e in episodes:
        if not e["scored"]:
            continue
        vals = [t[key] for t in e["ticks"] if not t["degenerate"]]
        if vals:
            out.setdefault(e[key_axis], []).append(float(np.mean(vals)))
    return out


def run_seed(seed: int, practice: int, reps: int, n_layouts: int,
             zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    n_probe_eps = len(_build_schedule(reps, n_layouts))
    eps_total = practice * len(LIN.FAMILIAR_ENV_SEEDS) + n_probe_eps * len(ARMS)
    print(f"Seed {seed} Condition all_arms", flush=True)
    agent, cfg = LIN.build_arm_agent(LIN.ARM_ON)
    for env_seed in LIN.FAMILIAR_ENV_SEEDS:
        env = CausalGridWorldV2(seed=env_seed, **LIN.FAMILIAR_ENV_KWARGS)
        warmup_train(agent, env, num_episodes=practice, steps_per_episode=LIN.STEPS_PER_EPISODE,
                     label=f"seed{seed}_practice_layout{env_seed}",
                     progress_total_episodes=eps_total)
    progress = [practice * len(LIN.FAMILIAR_ENV_SEEDS)]

    arm_runs: Dict[str, Dict[str, Any]] = {}
    arm_runs[ARM_ADAPTIVE] = _run_arm(agent, seed, ARM_ADAPTIVE, reps, n_layouts, [], zg, progress, eps_total)
    adaptive_ws = [t["w"] for e in arm_runs[ARM_ADAPTIVE]["episodes"] if e["scored"]
                   for t in e["ticks"] if not t["degenerate"]]
    for arm_id in ARMS[1:]:
        arm_runs[arm_id] = _run_arm(agent, seed, arm_id, reps, n_layouts, adaptive_ws, zg, progress, eps_total)

    per_arm: Dict[str, Any] = {}
    for arm_id, run in arm_runs.items():
        eps = run["episodes"]
        cells = {f"{f}_{m}": _cell_summary(eps, f, m) for f, m in CELLS}
        per_arm[arm_id] = {
            "fixed_w": run["fixed_w"],
            "cells": cells,
            "by_model": {m: _cell_summary(eps, None, m) for m in ("intact", "perturbed")},
            "all": _cell_summary(eps, None, None),
            "episodes": [{k: v for k, v in e.items() if k != "ticks"} for e in eps],
        }
    ad_eps = arm_runs[ARM_ADAPTIVE]["episodes"]
    up = _episode_means(ad_eps, "model", "u_p")
    ad_ticks = [t for e in ad_eps if e["scored"] for t in e["ticks"]]
    return {
        "seed": seed,
        "per_arm": per_arm,
        "u_planned_auc_perturbed_gt_intact": _auc_greater(up.get("perturbed", []), up.get("intact", [])),
        "adaptive_w_series_len": len(adaptive_ws),
        "adaptive_arb_sample": [
            {k: t.get(k) for k in ("w", "u_h", "u_p", "u_h_n", "u_p_n", "argmin_p", "argmin_h",
                                   "argmin_arb", "sel", "source", "degenerate")}
            for t in ad_ticks[:ARB_SAMPLE_CAP]
        ],
        "episodes_per_seed": eps_total,
    }


def _pa(seed_row, arm, level):
    return seed_row["per_arm"][arm]["by_model"][level]["planned_agreement"]


def _diff(a, b):
    return (a - b) if (a is not None and b is not None) else None


def analyse(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # ---- readiness -----------------------------------------------------------
    disagree_fracs, habit_ranges, live_fracs = [], [], []
    seeds_with_reach = 0
    for r in seed_rows:
        ad = r["per_arm"][ARM_ADAPTIVE]["all"]
        disagree_fracs.append(ad["disagree_frac"])
        habit_ranges.append(ad["min_habit_range"])
        live_fracs.append((float(ad["n_familiarity_source"]) / ad["n_e3_ticks"]) if ad["n_e3_ticks"] else 0.0)
        ok = all(
            (r["per_arm"][arm]["cells"][f"{f}_{m}"]["n_disagree"] >= MIN_DISAGREE_TICKS_PER_CELL)
            for arm in (ARM_ADAPTIVE, ARM_FIXED) for f, m in CELLS
        )
        seeds_with_reach += int(ok)
    up_aucs = [r["u_planned_auc_perturbed_gt_intact"] for r in seed_rows]

    def _seed_frac_at_least(vals, floor):
        # F1 (red-team): gate per seed, matching C1's per-seed sign leg. An unmeasured
        # (None) seed counts as a failing seed; the denominator is the INTENDED n.
        return (float(sum(1 for v in vals if v is not None and math.isfinite(float(v)) and float(v) >= floor))
                / float(len(vals))) if vals else 0.0

    def _worst(vals, lo=True):
        xs = _finite(vals)
        if len(xs) < len(vals):          # an unmeasured seed is the worst seed
            return 0.0 if lo else float("inf")
        return (min(xs) if lo else max(xs)) if xs else (0.0 if lo else float("inf"))

    pre = []

    def _add(name, desc, measured, threshold, control, direction="lower", scope="core"):
        m = measured if measured is not None else 0.0
        met = bool(m >= threshold) if direction == "lower" else bool(m <= threshold)
        pre.append({"name": name, "description": desc, "kind": "readiness", "scope": scope,
                    "measured": float(m), "threshold": float(threshold), "direction": direction,
                    "control": control, "met": met})

    _add("pathways_differ_on_choices",
         "Adaptive arm: fraction of scored E3 ticks whose habit argmin differs from the planned argmin, WORST seed.",
         _worst(disagree_fracs), PATHWAY_DISAGREE_FRAC_FLOOR,
         "real E3 ticks after practice; habit vector captured from the substrate's own depth-limited pass")
    _add("habit_score_range_non_degenerate",
         "Cross-candidate range of the habit vector, WORST scored tick over all seeds (the 786a constant-vector defect).",
         _worst(habit_ranges), SCORE_RANGE_FLOOR * 1.000001,
         "real E3 ticks after practice")
    _add("arbitration_live_on_familiarity",
         "Adaptive arm: fraction of E3 ticks producing a non-degenerate arbitration whose u_habit came from the familiarity tracker, WORST seed.",
         _worst(live_fracs), ARB_LIVE_FAMILIARITY_FRAC_FLOOR,
         "curiosity_weight 0.05 builds the tracker; the E1-novelty fallback is measured-constant on this config")
    _add("model_manipulation_reaches_u_planned",
         f"Fraction of seeds (intended n) whose AUC(episode-mean raw u_planned: perturbed > intact), adaptive arm, "
         f">= {U_PLANNED_AUC_FLOOR}.",
         _seed_frac_at_least(up_aucs, U_PLANNED_AUC_FLOOR), SEED_POSITIVE_FRACTION_FLOOR,
         "seeded Gaussian noise on E2 world_transition, applied per episode (sigma pre-registered by dose probe)")
    _add("seeds_with_reach_data",
         f"Seeds whose adaptive AND fixed arms each log >= {MIN_DISAGREE_TICKS_PER_CELL} disagreement ticks in every cell.",
         float(seeds_with_reach), float(MIN_SEEDS_WITH_REACH_DATA) if len(seed_rows) >= MIN_SEEDS_WITH_REACH_DATA else float(len(seed_rows)),
         "count, not a rate")
    core_green = all(p["met"] for p in pre)

    # ---- single-path separation (benefit leg precondition) -------------------
    sep_by_cell = {}
    for f, m in CELLS:
        k = f"{f}_{m}"
        diffs = [_diff(r["per_arm"][ARM_PLANNED]["cells"][k]["return_mean"],
                       r["per_arm"][ARM_HABIT]["cells"][k]["return_mean"]) for r in seed_rows]
        sep_by_cell[k] = _mean(diffs)
    sep_abs = [abs(v) for v in sep_by_cell.values() if v is not None]
    sep_max = max(sep_abs) if sep_abs else 0.0
    # F2 (red-team): C3 is scored ONLY over the cells where the two pure pathways separate
    # by >= 2 * BENEFIT_FLOOR -- the same per-cell bar P5 certifies -- so a benefit null
    # cannot come from floor-pinned (death-censored) cells P5 never certified.
    c3_cells = [k for k, v in sep_by_cell.items()
                if v is not None and abs(v) >= SINGLE_PATH_SEPARATION_MULT * BENEFIT_FLOOR]
    _add("single_path_outcome_separation",
         "max over cells of |mean over seeds (return planned_only - return habit_only)|. Gates C3 ONLY.",
         sep_max, SINGLE_PATH_SEPARATION_MULT * BENEFIT_FLOOR,
         "the two pure pathways on identical weights", scope="benefit_leg")
    p5_met = bool(pre[-1]["met"])

    # ---- C1 (model axis only) -----------------------------------------------
    dw_model = []
    for r in seed_rows:
        a = r["per_arm"][ARM_ADAPTIVE]
        dw_model.append(_diff(a["by_model"]["intact"]["w_mean"], a["by_model"]["perturbed"]["w_mean"]))
    c1_model = _criterion_leg(dw_model, W_SHIFT_FLOOR)
    c1 = bool(c1_model["passes"])

    # ---- C2 / C4-reach (model axis only) -------------------------------------
    def _did(ref_arm):
        return [_diff(_diff(_pa(r, ARM_ADAPTIVE, "intact"), _pa(r, ARM_ADAPTIVE, "perturbed")),
                      _diff(_pa(r, ref_arm, "intact"), _pa(r, ref_arm, "perturbed"))) for r in seed_rows]
    did_mod = _did(ARM_FIXED)
    c2_mod = _criterion_leg(did_mod, PA_DID_FLOOR)
    c2 = bool(c2_mod["passes"])
    sdid_mod = _did(ARM_SHUFFLED)
    c4_mod = _criterion_leg(sdid_mod, PA_DID_FLOOR)

    # ---- C3 / C4-benefit -----------------------------------------------------
    def _d_over_cells(r, ref_arm):
        ds = [_diff(r["per_arm"][ARM_ADAPTIVE]["cells"][k]["return_mean"],
                    r["per_arm"][ref_arm]["cells"][k]["return_mean"]) for k in c3_cells]
        return _mean(ds)
    d_fix = [_d_over_cells(r, ARM_FIXED) for r in seed_rows]
    d_shuf = [_d_over_cells(r, ARM_SHUFFLED) for r in seed_rows]

    # F3 (red-team): reach statistic -- does the arbitrated vector's argmin BECOME the
    # committed choice? Worst seed over the adaptive arm's non-degenerate scored ticks.
    reach_vals = [r["per_arm"][ARM_ADAPTIVE]["all"]["sel_equals_arbitrated_argmin_frac"] for r in seed_rows]
    reach_worst = _worst(reach_vals)
    reach_complete = bool(reach_worst >= REACH_COMPLETE_FLOOR)
    c3_leg = _criterion_leg(d_fix, BENEFIT_FLOOR)
    c3 = bool(c3_leg["passes"])
    c4_ben = _criterion_leg(d_shuf, BENEFIT_FLOOR)

    # ---- routing -----------------------------------------------------------
    did_means = [c2_mod["mean"]]
    if not core_green:
        outcome, label, direction = "FAIL", "substrate_not_ready_requeue", "non_contributory"
    elif not c1:
        outcome, label, direction = "FAIL", "allocation_does_not_track_model_uncertainty", "weakens"
    elif not c2:
        if reach_complete:
            # F3: the blend DOES determine the committed choice; C2 failing then means the
            # measured w shifts are too small to move PA past the floor (dose), not that
            # allocation fails to reach selection.
            outcome, label, direction = "FAIL", "reach_complete_dose_insufficient", "mixed"
        elif all(m is not None and m <= 0.0 for m in did_means):
            outcome, label, direction = "FAIL", "allocation_does_not_reach_committed_selection", "weakens"
        else:
            outcome, label, direction = "FAIL", "reach_equivocal", "mixed"
    elif not p5_met:
        outcome, label, direction = "FAIL", "reach_confirmed_benefit_untestable", "non_contributory"
    elif c3:
        outcome, label, direction = "PASS", "adaptive_allocation_reaches_selection_and_beats_fixed_model_axis", "supports"
    elif c3_leg["mean"] is not None and c3_leg["mean"] <= 0.0:
        outcome, label, direction = "FAIL", "adaptive_allocation_no_benefit_over_fixed", "weakens"
    else:
        outcome, label, direction = "FAIL", "benefit_equivocal", "mixed"

    criteria = [
        {"name": "C1_allocation_tracks_model_uncertainty", "load_bearing": True, "passed": c1,
         "measured": c1_model["measured"], "threshold": W_SHIFT_FLOOR,
         "seeds_positive": c1_model["seed_leg"]["n_positive"],
         "seed_positive_fraction_required": SEED_POSITIVE_FRACTION_FLOOR},
        {"name": "C2_committed_recruitment_beyond_fixed", "load_bearing": True, "passed": c2,
         "measured": c2_mod["measured"], "threshold": PA_DID_FLOOR,
         "seeds_positive": c2_mod["seed_leg"]["n_positive"],
         "seed_positive_fraction_required": SEED_POSITIVE_FRACTION_FLOOR},
        {"name": "C3_benefit_over_fixed", "load_bearing": True, "passed": c3,
         "measured": c3_leg["measured"], "threshold": BENEFIT_FLOOR,
         "scored_cells": c3_cells,
         "seeds_positive": c3_leg["seed_leg"]["n_positive"],
         "seed_positive_fraction_required": SEED_POSITIVE_FRACTION_FLOOR,
         "gated_by": "single_path_outcome_separation"},
        {"name": "C4_reach_beyond_shuffled", "load_bearing": False,
         "passed": bool(c4_mod["passes"]),
         "measured": c4_mod["measured"],
         "threshold": PA_DID_FLOOR},
        {"name": "C4_benefit_over_shuffled", "load_bearing": False, "passed": bool(c4_ben["passes"]),
         "measured": c4_ben["measured"], "threshold": BENEFIT_FLOOR},
    ]
    criteria_nd = {
        "C1_allocation_tracks_model_uncertainty": core_green,
        "C2_committed_recruitment_beyond_fixed": core_green,
        "C3_benefit_over_fixed": bool(core_green and p5_met),
        "C4_reach_beyond_shuffled": core_green,
        "C4_benefit_over_shuffled": bool(core_green and p5_met),
    }
    return {
        "outcome": outcome, "label": label, "direction": direction,
        "core_green": core_green, "p5_met": p5_met,
        "preconditions": pre, "criteria": criteria, "criteria_nd": criteria_nd,
        "legs": {"C1_model": c1_model,
                 "C2_model": c2_mod, "C3": c3_leg,
                 "C4_reach_model": c4_mod, "C4_benefit": c4_ben},
        "single_path_separation_by_cell": sep_by_cell,
        "c3_scored_cells": c3_cells,
        "adaptive_reach_frac_per_seed": reach_vals,
        "adaptive_reach_frac_worst": reach_worst,
        "reach_complete": reach_complete,
        "u_planned_aucs": up_aucs,
        "seeds_with_reach_data": seeds_with_reach,
    }


def _flat_scalar(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def config_slice() -> Dict[str, Any]:
    return {
        "lineage_base": LIN.LINEAGE,
        "base_arm_config": LIN.arm_config_slice(LIN.ARM_ON),
        "arms": list(ARMS), "single_path_bias": SINGLE_PATH_BIAS,
        "cells": [list(c) for c in CELLS], "probe_reps": PROBE_REPS,
        "probe_layouts": len(LIN.FAMILIAR_ENV_SEEDS),
        "model_perturb_rel_sigma": MODEL_PERTURB_REL_SIGMA,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    dims = LIN.assert_dims_match()
    seeds = SEEDS[:2] if dry_run else SEEDS
    practice = 2 if dry_run else LIN.PRACTICE_EPISODES_PER_LAYOUT
    reps = 1 if dry_run else PROBE_REPS
    # Dry-run exercises every arm, every cell and the full analysis on ONE layout index.
    n_layouts = 1 if dry_run else len(LIN.FAMILIAR_ENV_SEEDS)
    zg = ZGoalStreamAccumulator()

    seed_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        with arm_cell(
            seed, config_slice=config_slice(), script_path=Path(__file__),
            config_slice_declared=True,
            # One cell = one seed: five arms deep-copied from ONE practised agent, with
            # the fixed/shuffled arms derived from the adaptive arm's own w series.
            # Not reusable as an independent baseline cell by construction.
            extra_ineligible_reasons=["arms_share_one_practised_agent_and_adaptive_derived_controls"],
        ) as cell:
            row = run_seed(seed, practice, reps, n_layouts, zg)
            cell.stamp(row)
        a = row["per_arm"]
        d = _diff(a[ARM_ADAPTIVE]["all"]["return_mean"], a[ARM_FIXED]["all"]["return_mean"])
        print(f"verdict: {'PASS' if (d is not None and d > BENEFIT_FLOOR) else 'FAIL'}", flush=True)
        seed_rows.append(row)

    res = analyse(seed_rows)
    L = res["legs"]
    readout = {
        "core_gate_green": res["core_green"],
        "single_path_separation_met": res["p5_met"],
        "c1_w_shift_model_mean": L["C1_model"]["mean"],
        "c2_pa_did_model_mean": L["C2_model"]["mean"],
        "c3_return_adaptive_minus_fixed_mean": L["C3"]["mean"],
        "c4_pa_did_model_vs_shuffled_mean": L["C4_reach_model"]["mean"],
        "c4_return_adaptive_minus_shuffled_mean": L["C4_benefit"]["mean"],
        "u_planned_auc_mean": _mean(res["u_planned_aucs"]),
        "seeds_with_reach_data": res["seeds_with_reach_data"],
        "adaptive_reach_frac_worst": res["adaptive_reach_frac_worst"],
        "reach_complete": res["reach_complete"],
        "c3_n_scored_cells": len(res["c3_scored_cells"]),
        "single_path_separation_max": max([abs(v) for v in res["single_path_separation_by_cell"].values()
                                           if v is not None] or [0.0]),
    }
    for p in res["preconditions"]:
        readout[f"pre_{p['name']}_measured"] = p["measured"]
        readout[f"pre_{p['name']}_met"] = p["met"]
    for arm in ARMS:
        allv = [r["per_arm"][arm]["all"] for r in seed_rows]
        readout[f"{arm}_return_mean"] = _mean(x["return_mean"] for x in allv)
        readout[f"{arm}_planned_agreement_mean"] = _mean(x["planned_agreement"] for x in allv)
        readout[f"{arm}_w_mean"] = _mean(x["w_mean"] for x in allv)
        # F2 (red-team): steps survived is reported beside return, so a death-censored
        # return (sum of harm_signal, episode ends at health 0) is readable as such.
        readout[f"{arm}_steps_mean"] = _mean(x["steps_mean"] for x in allv)
        for f, m in CELLS:
            readout[f"{arm}_{f}_{m}_steps_mean"] = _mean(
                r["per_arm"][arm]["cells"][f"{f}_{m}"]["steps_mean"] for r in seed_rows)
            readout[f"{arm}_{f}_{m}_return_mean"] = _mean(
                r["per_arm"][arm]["cells"][f"{f}_{m}"]["return_mean"] for r in seed_rows)
    readout = {k: _flat_scalar(v) for k, v in readout.items()}
    readout = {k: v for k, v in readout.items() if v is not None}

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "outcome": res["outcome"],
        "evidence_direction": res["direction"],
        "evidence_direction_per_claim": {"SD-081": res["direction"]},
        "falsifier_scope": "model_uncertainty_half_only",
        "arm_results": seed_rows,
        "readout": readout,
        "criteria": res["criteria"],
        "combination_rule": (
            "SUPPORTS iff core readiness gate green AND C1 AND C2 AND "
            "single_path_outcome_separation AND C3 (all load-bearing). C3 is scored only "
            "over the cells passing the per-cell single-path separation bar, and is vacated "
            "(non-degenerate false) when no cell passes it. A C2 failure with complete reach "
            "(selected == argmin(arbitrated) >= 0.95, worst seed) routes "
            "reach_complete_dose_insufficient (mixed). C4 is secondary and never routes."
        ),
        "criteria_legs": res["legs"],
        "single_path_separation_by_cell": res["single_path_separation_by_cell"],
        "c3_scored_cells": res["c3_scored_cells"],
        "adaptive_reach_frac_per_seed": res["adaptive_reach_frac_per_seed"],
        "u_planned_auc_per_seed": res["u_planned_aucs"],
        "interpretation": {
            "label": res["label"],
            "preconditions": res["preconditions"],
            "criteria_non_degenerate": res["criteria_nd"],
        },
        "non_degenerate": bool(res["core_green"]),
        "non_degenerate_per_claim": {"SD-081": bool(res["core_green"])},
        "degeneracy_reason": "" if res["core_green"] else (
            "core readiness gate red: "
            + ", ".join(p["name"] for p in res["preconditions"] if p["scope"] == "core" and not p["met"])
        ),
        "observation_dims": dims,
        "pre_registered": {
            "seeds": seeds, "arms": list(ARMS), "cells": [list(c) for c in CELLS],
            "probe_reps": reps, "probe_layouts": n_layouts,
            "practice_episodes_per_layout": practice,
            "model_perturb_rel_sigma": MODEL_PERTURB_REL_SIGMA,
            "single_path_bias": SINGLE_PATH_BIAS,
            "pathway_disagree_frac_floor": PATHWAY_DISAGREE_FRAC_FLOOR,
            "score_range_floor": SCORE_RANGE_FLOOR,
            "arb_live_familiarity_frac_floor": ARB_LIVE_FAMILIARITY_FRAC_FLOOR,
            "u_planned_auc_floor": U_PLANNED_AUC_FLOOR,
            "min_disagree_ticks_per_cell": MIN_DISAGREE_TICKS_PER_CELL,
            "min_seeds_with_reach_data": MIN_SEEDS_WITH_REACH_DATA,
            "sem_k": SEM_K, "sem_min_n": SEM_MIN_N,
            "seed_positive_fraction_floor": SEED_POSITIVE_FRACTION_FLOOR,
            "w_shift_floor": W_SHIFT_FLOOR, "pa_did_floor": PA_DID_FLOOR,
            "benefit_floor": BENEFIT_FLOOR,
            "single_path_separation_mult": SINGLE_PATH_SEPARATION_MULT,
            "reach_complete_floor": REACH_COMPLETE_FLOOR,
            "familiarity_axis_dropped": True,
            "falsifier_scope": "model_uncertainty_half_only",
            "p3_p4_gated_per_seed": True,
            "c3_scored_on_separating_cells_only": True,
            "arbitration_gain": LIN.ARB_GAIN, "arbitration_bias": LIN.ARB_BIAS,
            "arbitration_ema_alpha": LIN.ARB_UNCERTAINTY_EMA_ALPHA,
            "habit_depth": LIN.HABIT_DEPTH, "curiosity_weight": LIN.CURIOSITY_WEIGHT,
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "scope_note": (
            "Tests SD-081 (the arbitration DESIGN) against the MODEL-UNCERTAINTY HALF ONLY of "
            "its 2026-09-21 what_would_answer: adaptive vs matched fixed mixture, shuffled and "
            "single-path references, an E2 world_transition perturbation manipulating u_planned, "
            "committed-selection reach and episode-return benefit. The familiarity "
            "(habit-uncertainty) half is NOT tested and stays open: the SD-025 familiarity "
            "tracker does not carry practice familiarity at probe time (128-anchor FIFO turnover), "
            "so no familiarity contrast is built. Does not adjudicate V3-EXQ-811a or "
            "MECH-477/MECH-163, and says nothing about harm modulation of the weight (MECH-235)."
        ),
    }
    manifest["_full_config"] = {
        "config_slice": config_slice(),
        "familiar_env_kwargs": LIN.FAMILIAR_ENV_KWARGS,
        "familiar_env_seeds": LIN.FAMILIAR_ENV_SEEDS,
        "steps_per_episode": LIN.STEPS_PER_EPISODE,
        "practice_episodes_per_layout": practice, "probe_reps": reps,
        "probe_layouts": n_layouts,
    }
    manifest["_seeds"] = seeds
    manifest["_started_at"] = t0
    manifest["_zg"] = zg.stats()
    return manifest


def _out_dir() -> Path:
    return (_ROOT.parent / "REE_assembly" / "evidence" / "experiments").resolve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)
    full_config = manifest.pop("_full_config")
    seeds_used = manifest.pop("_seeds")
    started_at = manifest.pop("_started_at")
    zg_stats = manifest.pop("_zg")
    out_path = write_flat_manifest(
        manifest, _out_dir(), dry_run=args.dry_run, config=full_config, seeds=seeds_used,
        script_path=Path(__file__), started_at=started_at, z_goal_stream_stats=zg_stats,
    )
    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"evidence_direction: {manifest.get('evidence_direction')}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
