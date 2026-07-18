"""V3-EXQ-784: SD-074 probe-warmup de-saturation budget sweep (DIAGNOSTIC).

QUESTION (one sentence): does the SD-074 probe_warmup bring the E3 pre-commit
action-value distribution into a measurable dynamic range, and at what warmup budget?

WHAT THIS IS NOT
----------------
This is NOT a test of MECH-063 sub-claim (i), and it is NOT a V3-EXQ-777b.
`claim_ids` is EMPTY and `EXPERIMENT_PURPOSE` is "diagnostic": nothing here weights any
claim's confidence. The re-derive brake fired on the 777 lineage
(failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18, user-confirmed "Fire on 777
lineage, exempt 779") and REFUSES a further lettered iteration of the
orthogonal-control-axes probe against an untrained agent. This run does not attempt one:
it measures ONLY the substrate precondition -- the dynamic range of the distribution --
and says nothing about whether the two control axes are orthogonal. The MECH-063
experiment becomes authorable only after this run establishes an adequate budget.

WHY IT EXISTS
-------------
V3-EXQ-777a measured D_action_mass (pre-commit softmax mass on non-noop candidates)
pinned at ceiling on 7 of 14 seeds and at floor on 2 more, i.e. informative-seed yield
4 of 14 = 28.6%, with corr(distance of D from saturation, norm_v_score) = 0.884. A
regulator that MODULATES a distribution is unobservable when that distribution has no
dynamic range, so that yield caps achievable power at ~51 informative seeds needed =>
~177 raw seeds, ~31 h. It is NOT a sampling defect: 777a gave every cell 250 fresh E3
selections (zero starved cells) and the saturation rate barely moved from its starved
predecessor. SD-074 (probe.trained_enough_agent_warmup, IMPLEMENTED 2026-07-18) supplies
a warmup; this run measures whether it works and how much is enough.

DESIGN: incremental warmup with checkpoint reads (NOT independent arms)
-----------------------------------------------------------------------
Per seed, ONE agent is trained incrementally and D_action_mass is read at each budget
checkpoint [0, 4, 10, 25]. This is deliberate and better than four independent arms for
two reasons:
  (1) It is ~1.6x cheaper (25 episodes per seed, not 0+4+10+25 = 39).
  (2) It removes a confound: an independent-arms design would compare budgets ACROSS
      differently-initialised agents, so a budget effect and a seed-init effect would be
      entangled. Here every checkpoint is the SAME agent further along one trajectory,
      so the within-seed budget contrast is clean.

BUDGETS ARE SIZED FROM A MEASURED COST. warmup_train was timed on this substrate at 780 s
for 5 episodes x 300 steps = 0.52 s per env step. A first-drafted sweep of
[0,20,60,150] x 300 steps x 14 seeds would have been ~91 h of fleet time; these numbers
are what fits. Training uses 100 steps/episode (a pure cost lever -- 777a did no training,
so nothing is confounded by shortening it), while the READ is held at 777a's 300
steps/episode because the read is the measurement being compared to that run.
The checkpoint read is safe precisely because SD-074's measure_action_mass is
NON-DESTRUCTIVE: it snapshots and restores both the full state_dict and the declared
non-buffer E3 scalars, so reading at episode 4 leaves the agent bit-identical for the
continued training to episode 10 (verified max|dw| = 0.000e+00). Without that property
this design would be invalid -- the act of measuring would perturb the training run.

CONSEQUENCE FOR ARM REUSE: cells within a seed SHARE a mutable agent, so they are NOT
independent and every cell is stamped reuse-INELIGIBLE via extra_ineligible_reasons.
That is the correctness guard, not an oversight -- a future consumer must never reuse a
checkpoint cell as if it were a freshly-initialised arm.

THE BUDGET-0 CHECKPOINT IS THE POSITIVE CONTROL. It is an untrained agent read with the
same instrument, so it must REPRODUCE 777a's saturation. If budget 0 came back mostly
unsaturated, the instrument or the env would differ from 777a and the whole comparison
would be void -- so that is a readiness PRECONDITION, not a result.

ACCEPTANCE (pre-registered, not derived from this run's own statistics)
----------------------------------------------------------------------
C1 (LOAD-BEARING): at some swept budget > 0, informative-seed yield (fraction of seeds
    with D_action_mass_mean STRICTLY inside (0.05, 0.95)) exceeds 0.5 -- a majority,
    matching the target condition the autopsy set.
C2: that yield strictly exceeds V3-EXQ-777a's HEADROOM fraction 5/14 = 0.357 by a margin.
C3: the budget-0 control reproduces heavy saturation (>= 0.5 of seeds saturated),
    confirming the instrument sees what 777a saw.

ON THE C2 COMPARATOR -- this is a correction worth stating explicitly. The autopsy's
headline figure is "informative yield 4 of 14 (28.6%)", but that is NOT the right
comparator here. 777a's "informative" required BOTH non-saturation AND an authority test
(a norm_v_score effect floor): it recorded 5 of 14 seeds in HEADROOM, of which only 4 also
cleared authority. THIS run measures saturation ONLY and computes no authority quantity at
all, so its like-for-like comparator is the HEADROOM fraction 0.357, not 0.286. Comparing
against 0.286 would flatter this run by 0.071 on a metric it never measured.

Env and seeds are IDENTICAL to V3-EXQ-777a (size 8, 2 hazards, 3 resources; the same 14
seeds), so either figure is drawn from the same cells.

SELF-ROUTE: if the instrument cannot read D at all (cells starved of fresh E3
selections), this self-routes to substrate_not_ready_requeue -- NEVER to a substrate
verdict. A starved instrument cannot falsify anything.

MECH-094: N/A -- waking-only gradient training, no simulation, no replay, no memory
write.

ASCII-only output (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.probe_warmup import (  # noqa: E402
    D_SAT_HIGH,
    D_SAT_LOW,
    WarmupRecipe,
    measure_action_mass,
    reapply_candidate_capture,
    saturation_regime,
)
from experiments._lib.goal_pipeline_tier1 import warmup_train  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_784_sd074_probe_warmup_desaturation_budget_sweep"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []  # DELIBERATELY EMPTY -- see "WHAT THIS IS NOT" above.

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----

# Identical to V3-EXQ-777a so the baseline is drawn from the same cells.
SEEDS = [11, 17, 23, 29, 37, 3, 5, 13, 19, 41, 53, 61, 71, 83]
ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3

# Budget checkpoints, cumulative along ONE training trajectory per seed.
#
# SIZED FROM A MEASURED COST, not guessed. warmup_train was timed on this substrate at
# 780 s for 5 episodes x 300 steps = 0.52 s per env step (4 optimisers, batch 32, CPU).
# The originally-drafted sweep ([0,20,60,150] x 300 steps x 14 seeds) would have been
# ~91 h of fleet time -- infeasible, and it is why these numbers are small.
#
# The budgets are ALSO informed by the SD-074 smoke reading: an untrained agent read
# D = 0.9949 (ceiling-pinned) while the SAME agent after only 3 episodes x 40 steps read
# 0.51-0.67 (headroom). That is n=1 at a toy budget and is NOT evidence -- but it does
# say the interesting range is small, so a sweep concentrated at 4-25 episodes is far
# better targeted than one reaching to 150.
BUDGET_CHECKPOINTS = [0, 4, 10, 25]

# TRAIN cost lever. Reduced from 777a's 300 purely to fit the sweep in feasible compute;
# 777a did NO training, so there is no training regime to match and nothing is confounded
# by shortening it.
TRAIN_STEPS_PER_EPISODE = 100

# READ conditions. Held at 777a's STEPS_PER_EPISODE = 300 DELIBERATELY: the de-saturation
# read is the measurement being compared against 777a's headroom baseline, so its rollout
# conditions must match that run even though the training regime does not.
READ_STEPS_PER_EPISODE = 300

# De-saturation read at each checkpoint (read-only, non-destructive).
PROBE_SELECTS = 150
PROBE_MAX_ENV_STEPS = 4000
# max_episodes DERIVED from max_env_steps, not set independently. The smoke run tripped
# sample_driven_rollout's EpisodeCapWarning on an earlier value of 400: with
# max_episodes=400 < max_env_steps=4000, the EPISODE cap binds first for any seed whose
# episodes are shorter than 10 steps, so the read silently spends a fraction of its step
# budget. That is exactly the V3-EXQ-779a seed-23 defect (835 of 2400 steps at ~7
# steps/episode) -- and seed 23 is in THIS seed list and was floor-pinned with very short
# episodes in 777a, so the hazard is live here, not hypothetical. Setting them equal makes
# the STEP cap the binding one for every seed.
PROBE_MAX_EPISODES = PROBE_MAX_ENV_STEPS

# Verdict thresholds.
YIELD_MAJORITY = 0.5          # C1: strictly greater than -- "a majority", per the autopsy

# LIKE-FOR-LIKE BASELINE. The autopsy's headline "informative yield 4 of 14 (28.6%)" is
# NOT the right comparator for this run, and using it would flatter the result by 0.071.
# 777a's "informative" required BOTH non-saturation AND an authority test (a norm_v_score
# effect floor); it measured 5 of 14 seeds in HEADROOM but only 4 of those also cleared
# authority. THIS run measures saturation ONLY -- it computes no authority quantity at all
# -- so its like-for-like comparator is 777a's HEADROOM fraction, 5/14 = 0.357.
BASELINE_HEADROOM_777A = 5.0 / 14.0        # 0.3571 -- the comparator C2 actually uses
BASELINE_INFORMATIVE_777A = 4.0 / 14.0     # 0.2857 -- recorded for context, NOT compared to
YIELD_MARGIN = 0.10           # C2: must beat the like-for-like baseline by this margin

# C3 / control precondition, expressed as a FLOOR on the SATURATED fraction rather than a
# ceiling on yield. Same statement, but a floor is what assert_anchor_reachable can guard,
# so the gate is proven reachable by 777a's own recorded cells before the run starts.
CONTROL_MIN_SATURATED_FRAC = 0.5

# 777a's recorded per-seed D_seed_mean, frozen as a literal from
# v3_exq_777a_..._20260718T101635Z_v3.json `per_seed[].D_seed_mean`. This is the
# known-degenerate positive control the readiness anchor scores with the SHIPPED predicate.
# Under saturation_regime() these give 9 of 14 saturated = 0.643, which clears the 0.5 gate.
_777A_REFERENCE_D_MEANS = [
    0.99834, 0.497272, 0.009944, 0.959463, 0.882364, 0.90876, 0.000301,
    0.998541, 1.0, 0.995469, 0.148769, 1.0, 0.939666, 0.969998,
]

# Readiness floor: a cell must actually collect selections for its read to mean anything.
MIN_SELECTS_FOR_READ = 50
MIN_CELLS_READABLE_FRAC = 0.8

PROGRESS_TICKS = 10

DRY_RUN_SEEDS = [11, 17]
DRY_RUN_BUDGETS = [0, 2]
DRY_RUN_TRAIN_STEPS_PER_EPISODE = 30
DRY_RUN_READ_STEPS_PER_EPISODE = 40
DRY_RUN_SELECTS = 15


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE, num_hazards=ENV_HAZARDS, num_resources=ENV_RESOURCES
    )


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    """Shared operating config. Matches V3-EXQ-777a's SHARED settings exactly.

    Note both regulators stay OFF: this run measures the DISTRIBUTION's dynamic range,
    not any regulator's effect on it. Turning them on would confound the de-saturation
    reading with the very modulation a later experiment wants to measure.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_tonic_vigor = False
    cfg.use_noise_floor = False
    return cfg


def _config_slice(train_spe: int, read_spe: int, budgets: List[int], selects: int) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + the sweep schedule."""
    return {
        "env": {
            "size": ENV_SIZE,
            "num_hazards": ENV_HAZARDS,
            "num_resources": ENV_RESOURCES,
        },
        "shared": {
            "use_control_vector_logging": True,
            "use_action_class_scaffold_candidates": True,
            "use_tonic_vigor": False,
            "use_noise_floor": False,
        },
        "schedule": {
            "train_steps_per_episode": train_spe,
            "read_steps_per_episode": read_spe,
            "budget_checkpoints": list(budgets),
            "probe_selects": selects,
        },
    }


def _run_seed(
    seed: int,
    budgets: List[int],
    train_spe: int,
    read_spe: int,
    selects: int,
    config_slice: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """One seed: train ONE agent incrementally, reading D at each budget checkpoint."""
    rows: List[Dict[str, Any]] = []
    total_budget = max(budgets)
    trained_so_far = 0

    # The whole seed trajectory is ONE fingerprinted cell-group. RNG is reset once at
    # entry; the checkpoints share the resulting agent, which is exactly why every row
    # is stamped reuse-ineligible below.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,
        extra_ineligible_reasons=[
            "incremental_warmup_shared_agent_across_budget_checkpoints",
        ],
    ) as cell:
        env = _mk_env()
        cfg = _build_config(env)
        agent = REEAgent(cfg)
        captured = reapply_candidate_capture(agent)

        for budget in budgets:
            print(f"Seed {seed} Condition budget_{budget}", flush=True)
            delta = budget - trained_so_far
            if delta > 0:
                # Continue the SAME trajectory rather than restarting.
                warmup_train(
                    agent,
                    env,
                    num_episodes=delta,
                    steps_per_episode=train_spe,
                    label=f"seed={seed} budget={budget}",
                    progress_total_episodes=total_budget,
                )
                trained_so_far = budget

            # Emit the loop-bound progress line the runner parses. The denominator is
            # the loop bound (total_budget), never a hardcoded constant.
            print(
                f"  [train] seed={seed} budget={budget} "
                f"ep {max(trained_so_far, 1)}/{max(total_budget, 1)}",
                flush=True,
            )

            read = measure_action_mass(
                agent,
                env,
                seed=seed,
                n_selections=selects,
                max_env_steps=PROBE_MAX_ENV_STEPS,
                max_episodes=PROBE_MAX_EPISODES,
                steps_per_episode=read_spe,
                captured=captured,
                label=f"desat seed={seed} budget={budget}",
            )
            d_mean = read["d_action_mass_mean"]
            regime = saturation_regime(d_mean)
            readable = int(read["n_selections"]) >= (
                MIN_SELECTS_FOR_READ if selects >= MIN_SELECTS_FOR_READ else 1
            )
            row = {
                "seed": seed,
                "budget": budget,
                "d_action_mass_mean": d_mean,
                "d_action_mass_std": read["d_action_mass_std"],
                "saturation_regime": regime,
                "informative": bool(regime == "headroom"),
                "n_selections": int(read["n_selections"]),
                "readable": bool(readable),
                "probe_stop_reason": read["stop_reason"],
                "warmup_episodes_cumulative": trained_so_far,
            }
            cell.stamp(row)
            rows.append(row)
            print(
                f"  budget={budget} D={('None' if d_mean is None else '%.4f' % d_mean)} "
                f"regime={regime} n={row['n_selections']} stop={row['probe_stop_reason']}",
                flush=True,
            )
            # One verdict line per (seed x condition) unit, as the runner expects.
            print(f"verdict: {'PASS' if row['informative'] else 'FAIL'}", flush=True)

    return rows


def _yield_at(rows: List[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    at = [r for r in rows if r["budget"] == budget]
    n = len(at)
    inf = [r for r in at if r["informative"]]
    regimes = {"ceiling": 0, "headroom": 0, "floor": 0, "unmeasured": 0}
    for r in at:
        regimes[r["saturation_regime"]] = regimes.get(r["saturation_regime"], 0) + 1
    return {
        "budget": budget,
        "n_seeds": n,
        "n_informative": len(inf),
        "informative_yield": (len(inf) / n) if n else 0.0,
        "informative_seeds": sorted(r["seed"] for r in inf),
        "regimes": regimes,
        "d_means": {r["seed"]: r["d_action_mass_mean"] for r in at},
    }


def _cell_is_saturated(d_mean: Optional[float]) -> bool:
    """THE SHIPPED PREDICATE. Used both to score live cells and to score the frozen
    V3-EXQ-777a reference in the readiness anchor -- deliberately the same callable, so
    the anchor cannot drift from what the run actually measures."""
    return saturation_regime(d_mean) != "headroom"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # READINESS ANCHOR (setup-time, fails loudly). Prove the control gate is REACHABLE by
    # a bit-perfect replication of the known-degenerate control before spending fleet
    # hours: score V3-EXQ-777a's own recorded per-seed D values with the shipped
    # predicate and confirm they clear CONTROL_MIN_SATURATED_FRAC. Without this, a
    # hand-written gate narrower than the state it anchors to would report met=false on
    # every run forever and mislabel an instrument-specification gap as a substrate
    # verdict (the V3-EXQ-778d failure mode).
    anchor = assert_anchor_reachable(
        anchor_name="budget0_control_reproduces_777a_saturation",
        reference_cells=_777A_REFERENCE_D_MEANS,
        score_fn=_cell_is_saturated,
        threshold=CONTROL_MIN_SATURATED_FRAC,
        reference_source=(
            "v3_exq_777a_mech063_orthogonal_control_axes_dissociation_"
            "20260718T101635Z_v3.json per_seed[].D_seed_mean (14 cells)"
        ),
    )
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    budgets = DRY_RUN_BUDGETS if dry_run else BUDGET_CHECKPOINTS
    train_spe = DRY_RUN_TRAIN_STEPS_PER_EPISODE if dry_run else TRAIN_STEPS_PER_EPISODE
    read_spe = DRY_RUN_READ_STEPS_PER_EPISODE if dry_run else READ_STEPS_PER_EPISODE
    selects = DRY_RUN_SELECTS if dry_run else PROBE_SELECTS
    cslice = _config_slice(train_spe, read_spe, budgets, selects)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.extend(_run_seed(seed, budgets, train_spe, read_spe, selects, cslice))

    by_budget = [_yield_at(rows, b) for b in budgets]
    control = next((b for b in by_budget if b["budget"] == 0), None)
    swept = [b for b in by_budget if b["budget"] > 0]
    best = max(swept, key=lambda b: b["informative_yield"]) if swept else None

    # ---- Readiness precondition -------------------------------------------------
    # SAME STATISTIC RULE: the load-bearing criterion C1 routes on informative YIELD, a
    # fraction of seeds whose D read is inside the band. The thing that can starve it is
    # a cell that collected too few fresh E3 selections to have a meaningful D at all --
    # so the readiness check asserts the FRACTION OF CELLS THAT ARE READABLE, the same
    # kind of quantity (a fraction over the same cell population), not a magnitude proxy.
    n_cells = len(rows)
    n_readable = sum(1 for r in rows if r["readable"])
    readable_frac = (n_readable / n_cells) if n_cells else 0.0
    instrument_ready = readable_frac >= MIN_CELLS_READABLE_FRAC

    # The budget-0 control must reproduce 777a's saturation, else the instrument or env
    # differs from the run this whole comparison is anchored on. Expressed as a FLOOR on
    # the saturated fraction so it is the same predicate the readiness anchor proved
    # reachable against 777a's recorded cells.
    control_yield = control["informative_yield"] if control else None
    control_saturated_frac = (1.0 - control_yield) if control_yield is not None else None
    control_reproduces = (
        control_saturated_frac is not None
        and control_saturated_frac >= CONTROL_MIN_SATURATED_FRAC
    )

    preconditions = [
        {
            "name": "desaturation_read_cells_readable_frac",
            "description": (
                "fraction of (seed x budget) cells whose de-saturation read collected at "
                "least MIN_SELECTS_FOR_READ fresh E3 selections -- the same fraction-over-"
                "cells statistic the load-bearing yield criterion routes on"
            ),
            "control": "all cells, including the untrained budget-0 control",
            "measured": round(readable_frac, 4),
            "threshold": MIN_CELLS_READABLE_FRAC,
            "direction": "lower",  # FLOOR: met when measured >= threshold
            "met": bool(instrument_ready),
        },
        {
            "name": "budget0_control_reproduces_777a_saturation",
            "description": (
                "the untrained (budget 0) checkpoint must show a MAJORITY of seeds "
                "saturated, reproducing V3-EXQ-777a's 9/14 = 0.643 saturated fraction on "
                "the same env and the same 14 seeds; if the untrained agent were already "
                "unsaturated, the instrument or env would differ from 777a and the whole "
                "comparison would be void. Proven reachable at setup by scoring 777a's "
                "own recorded per-seed D values with the shipped predicate (see "
                "readiness_anchor below)."
            ),
            "control": "budget-0 checkpoint = untrained agent, same env and seeds as 777a",
            "measured": (
                None if control_saturated_frac is None else round(control_saturated_frac, 4)
            ),
            "threshold": CONTROL_MIN_SATURATED_FRAC,
            "direction": "lower",  # FLOOR: met when the saturated fraction >= threshold
            "met": bool(control_reproduces),
        },
    ]

    # ---- Criteria ----------------------------------------------------------------
    c1 = bool(best is not None and best["informative_yield"] > YIELD_MAJORITY)
    c2 = bool(
        best is not None
        and best["informative_yield"] > (BASELINE_HEADROOM_777A + YIELD_MARGIN)
    )
    c3 = bool(control_reproduces)

    yields = [b["informative_yield"] for b in by_budget]
    c1_nondegenerate = bool(len(set(round(y, 6) for y in yields)) > 1)
    c2_nondegenerate = c1_nondegenerate
    c3_nondegenerate = bool(control is not None and control["n_seeds"] > 0)

    if not instrument_ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif not control_reproduces:
        # The anchor itself failed; this is an instrument/env mismatch, not a verdict on
        # the warmup. Route to requeue rather than to any substrate verdict.
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif c1 and c2:
        label = "warmup_desaturates_landscape"
        outcome = "PASS"
    else:
        label = "warmup_insufficient_at_swept_budgets"
        outcome = "FAIL"

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "substrate_under_test": "SD-074",
        "routed_by": "failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18",
        "baseline_comparator": {
            "run_id": "v3_exq_777a_mech063_orthogonal_control_axes_dissociation_20260718T101635Z_v3",
            "headroom_yield": BASELINE_HEADROOM_777A,
            "informative_yield_reported_by_autopsy": BASELINE_INFORMATIVE_777A,
            "compared_against": "headroom_yield",
            "note": (
                "777a recorded 5 of 14 seeds in HEADROOM (0.357) but only 4 of 14 "
                "INFORMATIVE (0.286), because its informative test required BOTH "
                "non-saturation AND an authority (norm_v_score effect floor) check. This "
                "run measures saturation ONLY and computes no authority quantity, so the "
                "like-for-like comparator is the 0.357 headroom fraction. Comparing "
                "against 0.286 would flatter this run by 0.071. Same env and same 14 "
                "seeds either way."
            ),
        },
        "readiness_anchor": anchor,
        "d_sat_low": D_SAT_LOW,
        "d_sat_high": D_SAT_HIGH,
        "budget_checkpoints": budgets,
        "per_cell": rows,
        "per_budget": by_budget,
        "best_swept_budget": best,
        "control_budget0": control,
        "criteria": [
            {
                "name": "C1_majority_informative_at_some_budget",
                "load_bearing": True,
                "passed": c1,
                "threshold": YIELD_MAJORITY,
                "measured": None if best is None else best["informative_yield"],
            },
            {
                "name": "C2_beats_777a_headroom_baseline_by_margin",
                "load_bearing": False,
                "passed": c2,
                "threshold": BASELINE_HEADROOM_777A + YIELD_MARGIN,
                "measured": None if best is None else best["informative_yield"],
            },
            {
                "name": "C3_control_reproduces_saturation",
                "load_bearing": False,
                "passed": c3,
                "threshold": CONTROL_MIN_SATURATED_FRAC,
                "measured": control_saturated_frac,
            },
        ],
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1": c1_nondegenerate,
                "C2": c2_nondegenerate,
                "C3": c3_nondegenerate,
            },
            "null_meaning": (
                "A FAIL at label warmup_insufficient_at_swept_budgets means the swept "
                "budgets (max %d episodes) did not de-saturate a majority of seeds. It "
                "does NOT mean warmup cannot work -- the honest next question is whether a "
                "larger budget, or a different training signal, moves the yield. It says "
                "NOTHING about MECH-063: no control-axis quantity is measured here."
                % max(budgets)
            ),
        },
        "arm_results": rows,
        "sd074_note": (
            "Cells within a seed share ONE incrementally-trained agent, so every cell is "
            "stamped reuse-INELIGIBLE (incremental_warmup_shared_agent_across_budget_"
            "checkpoints). The checkpoint design is only valid because SD-074's "
            "measure_action_mass is non-destructive (restores state_dict + non-buffer E3 "
            "scalars), so reading at one checkpoint leaves the agent bit-identical for "
            "continued training."
        ),
    }

    # Non-degeneracy net (applies on any purpose): if every cell read the same D, the
    # sweep discriminated nothing.
    d_vals = [r["d_action_mass_mean"] for r in rows if r["d_action_mass_mean"] is not None]
    if len(d_vals) > 1 and statistics.pstdev(d_vals) <= 1e-9:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "every cell returned an identical D_action_mass_mean; the budget sweep "
            "discriminated nothing"
        )

    full_config = {
        "env": {"size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES},
        "budget_checkpoints": budgets,
        "train_steps_per_episode": train_spe,
        "read_steps_per_episode": read_spe,
        "probe_selects": selects,
        "probe_max_env_steps": PROBE_MAX_ENV_STEPS,
        "probe_max_episodes": PROBE_MAX_EPISODES,
        "thresholds": {
            "D_SAT_LOW": D_SAT_LOW,
            "D_SAT_HIGH": D_SAT_HIGH,
            "YIELD_MAJORITY": YIELD_MAJORITY,
            "BASELINE_HEADROOM_777A": BASELINE_HEADROOM_777A,
            "BASELINE_INFORMATIVE_777A": BASELINE_INFORMATIVE_777A,
            "YIELD_MARGIN": YIELD_MARGIN,
            "CONTROL_MIN_SATURATED_FRAC": CONTROL_MIN_SATURATED_FRAC,
            "MIN_SELECTS_FOR_READ": MIN_SELECTS_FOR_READ,
            "MIN_CELLS_READABLE_FRAC": MIN_CELLS_READABLE_FRAC,
        },
        "warmup_recipe_reference": WarmupRecipe(
            num_episodes=max(budgets), steps_per_episode=train_spe, probe_selections=selects
        ).as_dict(),
    }

    # Multi-arm: stamp AFTER arm_results is assembled so substrate_hash HOISTS from the
    # per-cell fingerprints rather than being recomputed driver-inclusive.
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=_THIS,
        started_at=t0,
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    # stamp=False: run_experiment already called stamp_recording_core AFTER arm_results
    # was assembled, so substrate_hash is hoisted from the per-cell fingerprints. Letting
    # the writer stamp again would recompute it driver-inclusive and mismatch the cells.
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        stamp=False,
    )

    print("")
    print("=" * 62)
    print(f"outcome: {result['outcome']}")
    print(f"label:   {result['interpretation']['label']}")
    for b in result["per_budget"]:
        print(
            "  budget %-4d yield %.3f (%d/%d) regimes=%s"
            % (
                b["budget"],
                b["informative_yield"],
                b["n_informative"],
                b["n_seeds"],
                b["regimes"],
            )
        )
    print(f"  777a headroom baseline: {BASELINE_HEADROOM_777A:.3f} (like-for-like)")
    print(f"manifest: {out_path}")
    print("=" * 62)

    _raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )

