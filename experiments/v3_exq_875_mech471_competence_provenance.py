#!/opt/local/bin/python3
"""V3-EXQ-875 -- MECH-471 LOCAL-UPDATE INTERFERENCE TEST (EXP-0399).

WHY THIS EXISTS. MECH-392/INV-080/MECH-401 give REE's memory-CONSOLIDATION path a bounded,
provenance-carrying, rollback-capable editing discipline. Nothing equivalent is registered
for behavioural COMPETENCE: policy / affordance / action-object learning currently has no
rollback, no provenance record, and no gate against a single successful update propagating
widely. MECH-471 is the generalisation claim that the competence path needs the same
discipline. The registered FALSIFIER (claims.yaml) is a LOCAL-UPDATE INTERFERENCE TEST: train
a targeted competence improvement, then measure whether it degrades UNRELATED, previously-
acquired competence. NOTE THE POLARITY (stated in claims.yaml and mirrored exactly here):
FAIL (unrelated competence degrades) is what SUPPORTS MECH-471 and motivates building the
discipline; PASS (no degradation) WEAKENS it -- competence learning is already sufficiently
local. This experiment does NOT build or exercise any provenance/rollback machinery itself
(that machinery is the thing MECH-471 would motivate, and is NOT what is being tested); it
tests only whether the underlying phenomenon (cross-competence interference) exists on the
substrate that is actually live today.

SUBSTRATE USED (TESTABLE NOW; complicated (buildable), not complex (probe-gated)). The all-ON
REE stack's CURRENT action-learning mechanism is exactly the "thin bias_head REINFORCE
readout" MECH-457 describes: SD-056 e2 world-forward warmup (frozen through P1, A0 recipe)
feeding a two-head REINFORCE update (lateral-PFC action bias + OFC devaluation bias) against
an episodic-return baseline. This is a SHARED, single parameterised policy object -- exactly
the kind of thing a local update could interfere across if trained on two different tasks in
sequence. Reused verbatim (no new architecture): experiments._lib.allon_training.
_train_all_on_agent (the 724-A0 / 734 / 742 recipe) and experiments._lib.capability_eval
(claim-agnostic behavioural yardstick). MECH-392/401/ARC-092 (the machinery this claim
generalises FROM) are implementation_phase=v4/v3_pending and are NOT needed here -- this
experiment is purely behavioural/measurement, on substrate that already exists.

WHY SURVIVAL, NOT FORAGING. The obvious axis for "separable competence" in this env family is
resource-foraging under different layouts, but foraging_competence is exactly the metric the
MECH-457 GOV-FANOUT-1 campaign (719a/724/728/734/742/780/788, 74% of that campaign's autopsies
routing to substrate_ceiling) has repeatedly found pinned near/at floor for the all-ON stack --
using it here would very likely just reproduce that ceiling and self-route
substrate_not_ready_requeue with no information gained about MECH-471. V3-EXQ-728 separately
established that the SAME all-ON stack DOES survive competently ("+6.07" normalized survival
against the random-walk floor) even where it does not forage. So the two separable
competences here are SURVIVAL/AVOIDANCE strategies under two qualitatively different hazard
GEOMETRIES (not two different environments-writ-large, and not two resource layouts of the
same geometry, which the agent's egocentric 5x5 local view would generalise across for free
and so would not be separable at all):
  * competence A ("corner")  -- 3 reef safe-zone patches at fixed grid corners (legacy
                                _place_reef_patches), plus the same scattered point hazards.
                                Requires generic local hazard-gradient avoidance around
                                scattered corner safe-zones.
  * competence B ("banded")  -- the SAME 3 reef patches, SAME scattered point hazards, but
                                placed as a persistent BANDED corridor along one grid edge
                                (SD-054 _place_reef_patches_bipartite, reef_bipartite_
                                layout=True). Requires learning to stay clear of (or
                                navigate across) a spatially EXTENDED, persistent hazard
                                feature -- a qualitatively different avoidance strategy
                                from dodging scattered point hazards.
Both envs set hazard_food_attraction=0.0 so foraging behaviour is decoupled from the DV, and
BOTH hold reef_enabled=True AND n_reef_patches=3 fixed and IDENTICAL -- required so the SAME
shared agent's encoder (sized once, from env A's dims) can process both envs; only
reef_bipartite_layout differs, which changes reef PLACEMENT geometry, not observation
dimensionality (see _env_kwargs docstring for the property-vs-actual env quirk this avoids).
The DV is capability_eval's survival_horizon (mean ticks survived/episode) and death_rate,
read off real env transitions, exactly as for the foraging yardstick.

DESIGN. Local-update interference test, per (seed x perturbation strength) cell:
  Phase 1 (ACQUIRE BOTH, verified independently). One shared agent: P0 SD-056 e2 warmup on
    env A, then P1 REINFORCE (e2 frozen, A0 recipe) on env A to acquire competence A; then,
    SAME agent (no re-warmup), P1 REINFORCE on env B to acquire competence B. Evaluate BOTH
    (env A, env B) immediately after acquisition -> a_baseline, b_baseline. These are the
    "SET of separable competences trained to above-floor performance, verified independently"
    the proposal's design calls for.
  Phase 2 (TARGETED IMPROVEMENT, dose-response). SAME agent continues P1 REINFORCE on env B
    ONLY, for `strength` additional episodes. strength in STRENGTH_LEVELS, strength=0 is the
    CONTROL (no additional training at all -- the no-perturbation reference, used to estimate
    pure eval-noise SD for the interference margin).
  Phase 3 (RE-MEASURE UNRELATED). Evaluate env A again -> a_after. interference_delta =
    a_after.survival_horizon - a_baseline.survival_horizon (negative = degradation). Also
    re-measure env B -> b_after, to confirm the targeted update actually MOVED B (else nothing
    was perturbed -- the acceptance_checks non-degeneracy guard).

Each (seed, strength) cell independently replays Phase 1 from a fresh env+agent (the standard
arm_cell convention: full RNG reset per cell) -- a_baseline/b_baseline should therefore be
BIT-IDENTICAL across strength levels for a fixed seed, since Phase 1 is deterministic and
strength only affects Phase 2. This is recorded and checked as an internal determinism sanity
check, not just assumed.

ACCEPTANCE (mirrors EXP-0399 / claims.yaml verbatim, including the polarity):
  * FAIL-DIRECTION-IS-THE-MOTIVATING-RESULT (outcome=FAIL, evidence_direction=supports):
    interference_delta on env A becomes reliably negative and grows with strength -- at the
    two highest strength levels the mean delta clears (more negative than) a pre-registered
    margin relative to the strength=0 control, and the highest-strength delta is negative.
    Label: local_update_interference_confirmed.
  * PASS-AS-NOT-NEEDED (outcome=PASS, evidence_direction=weakens): interference_delta stays
    within the noise margin across the FULL strength range, including the highest level.
    Label: local_update_interference_not_detected.
  * Ambiguous dose-response (clears the margin only at the top, or non-monotonic beyond
    noise): outcome=FAIL, evidence_direction=mixed, label=interference_dose_response_ambiguous.
  * NON-DEGENERACY (outcome=FAIL, evidence_direction=unknown, non_degenerate=False,
    label=substrate_not_ready_requeue): EITHER competence A or B fails to clear its floor
    above the random-walk anchor by the declared margin on the worst seed (else there was no
    installed competence to interfere with), OR there is no live cross-seed variance in the
    acquired competences (a single pinned value cannot support a dose-response reading), OR
    the targeted update does not measurably MOVE competence B at the highest strength (else
    nothing was perturbed and any "no interference" reading would be vacuous).

evidence_direction_per_claim is NOT populated -- claim_ids has exactly one entry (MECH-471).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.allon_training import _train_all_on_agent  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    RandomPolicy,
    REEForwardPolicy,
    evaluate_seed,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_875_mech471_competence_provenance"
QUEUE_ID = "V3-EXQ-875"
CLAIM_IDS: List[str] = ["MECH-471"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Budget (real run) ----------------------------------------------------------------
SEEDS: List[int] = [42, 43, 45]   # 44 skipped per repo convention (recurring early-death
                                  # instability on this env family; substitute 45).
P0_WARMUP_EPISODES = 120          # SD-056 e2 forward-model warmup, run ONCE on env A only
                                  # (A0 recipe: e2 frozen through every subsequent P1 call).
P1_ACQUIRE_EPISODES = 50          # REINFORCE episodes to acquire EACH of competence A and B.
STRENGTH_LEVELS: List[int] = [0, 25, 50, 150]   # additional TARGETED REINFORCE episodes on B
                                                 # ONLY. 0 = no-perturbation control (used to
                                                 # estimate the pure eval-noise margin).
STEPS_PER_EPISODE = 150
EVAL_EPISODES = 12

# --- Dry-run budget ---------------------------------------------------------------------
DRY_SEEDS: List[int] = [42]
DRY_P0 = 2
DRY_P1_ACQUIRE = 2
DRY_STRENGTH_LEVELS: List[int] = [0, 2]
DRY_STEPS = 15
DRY_EVAL = 2

# --- Pre-registered thresholds (declared as fractions of STEPS_PER_EPISODE; never derived
# from the run's own statistics) ----------------------------------------------------------
SURVIVAL_FLOOR_FRAC = 0.6          # acquired competence must survive >= 60% of the episode
RANDOM_MARGIN_FRAC = 0.15          # ... AND beat the random-walk anchor by >= 15% of the
                                   # episode length (achievability / non-vacuity check)
TARGETED_MOVE_FLOOR_FRAC = 0.05    # the targeted update must move competence B by >= 5% of
                                   # the episode length at the highest strength, or nothing
                                   # was perturbed
INTERFERENCE_ABS_FLOOR_FRAC = 0.05  # absolute floor component of the interference margin
INTERFERENCE_SD_MULT = 2.0          # scale-on-noise component (feedback_effect_size_pass_
                                     # gate_margin: scale on SD of the delta, plus an
                                     # absolute floor), noise SD estimated from the
                                     # strength=0 control arm's per-seed deltas
CONFIRM_STRONG_FRAC = 0.5           # the two highest strengths must both clear >= this
                                     # fraction of the margin (guards against a single
                                     # anomalous top point reading as a confirmed dose-response)


def _env_kwargs(variant: str) -> Dict[str, Any]:
    """Both competences share the 724 base env kwargs; hazard_food_attraction is zeroed in
    BOTH so foraging is decoupled from the survival DV. Variant "A" ("corner" -- reef patches
    at fixed grid corners, the legacy _place_reef_patches layout) vs "B" ("banded" -- the SAME
    3 reef patches instead placed along one edge as a bipartite band, SD-054
    _place_reef_patches_bipartite) -- a qualitatively different avoidance geometry (scattered
    corner safe-zones vs a persistent extended corridor), not merely a different resource
    layout (which the agent's egocentric 5x5 local view would generalise across for free).

    reef_enabled=True AND n_reef_patches=3 (>0) are held FIXED and IDENTICAL in BOTH variants
    -- this is load-bearing for a shared agent, not a stylistic choice. CausalGridWorldV2.
    world_obs_dim (the PROPERTY used to size the agent's encoder) adds +25 whenever
    reef_enabled is True, UNCONDITIONALLY on n_reef_patches. But the ACTUAL observation
    construction (_get_observation_dict) only appends that +25 reef-scent channel to
    world_state when self._reef_cells is non-empty -- which requires n_reef_patches > 0 (see
    _place_reef_patches / _place_reef_patches_bipartite, both of which early-return an EMPTY
    _reef_cells when n_reef_patches<=0). So reef_enabled=True with n_reef_patches=0 is a real
    property-vs-actual MISMATCH in the env (sized for 275, actually emits 250) -- confirmed by
    running this driver, which crashed on exactly that shape mismatch before this was fixed.
    n_reef_patches=3 in BOTH variants keeps _reef_cells non-empty (and hence obs dims at 275)
    on both sides; reef_bipartite_layout is the ONLY lever toggled, and it changes placement
    geometry, not dimensionality."""
    kw = dict(x724.ENV_KWARGS)
    kw["hazard_food_attraction"] = 0.0
    kw["reef_enabled"] = True       # FIXED across both variants -- see docstring above.
    kw["n_reef_patches"] = 3        # FIXED across both variants -- see docstring above.
    if variant == "A":
        kw["reef_bipartite_layout"] = False   # legacy corner-pattern reef placement
    elif variant == "B":
        kw.update(
            reef_bipartite_layout=True,
            reef_bipartite_axis="horizontal", reef_bipartite_agent_band_radius=1,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return kw


def _make_env(seed: int, env_kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The all-ON REE stack (724/734/742 recipe): the two-head REINFORCE bias substrate
    (lateral-PFC action bias + OFC devaluation bias) over the SD-056 e2 world-forward encoder
    -- the CURRENT default action-learning mechanism MECH-457 describes as a "thin bias_head
    REINFORCE readout". No new architecture; reused verbatim."""
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


def _eval_agent(agent: REEAgent, env_kwargs: Dict[str, Any], seed: int, eval_eps: int,
                steps: int, policy_name: str) -> Dict[str, Any]:
    env = _make_env(seed, env_kwargs)
    return evaluate_seed(REEForwardPolicy(agent, name=policy_name), env, eval_eps, steps)


def _eval_random(env_kwargs: Dict[str, Any], seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    env = _make_env(seed, env_kwargs)
    return evaluate_seed(RandomPolicy(seed), env, eval_eps, steps)


def _run_cell(
    seed: int, strength: int, p0: int, p1_acquire: int, eval_eps: int, steps: int,
    kw_a: Dict[str, Any], kw_b: Dict[str, Any], zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """One (seed, strength) cell: acquire A then B on ONE shared agent, apply the targeted
    perturbation to B only at `strength`, re-measure both."""
    denom = p0 + p1_acquire + p1_acquire + max(strength, 1)

    # Phase 1a: acquire competence A (P0 e2 warmup + P1 REINFORCE, e2 frozen thereafter).
    env_a_train = _make_env(seed, kw_a)
    agent = _make_agent(env_a_train)
    _train_all_on_agent(
        agent, env_a_train, seed, p0_episodes=p0, p1_episodes=p1_acquire,
        steps_per_episode=steps, rung_id=f"A_acquire_strength{strength}",
        total_denominator=denom,
    )
    a_baseline = _eval_agent(agent, kw_a, seed, eval_eps, steps, "acquired_A_pre_perturb")

    # Phase 1b: acquire competence B on the SAME agent (no re-warmup: p0_episodes=0).
    env_b_train = _make_env(seed, kw_b)
    _train_all_on_agent(
        agent, env_b_train, seed, p0_episodes=0, p1_episodes=p1_acquire,
        steps_per_episode=steps, rung_id=f"B_acquire_strength{strength}",
        total_denominator=denom,
    )
    b_baseline = _eval_agent(agent, kw_b, seed, eval_eps, steps, "acquired_B_pre_perturb")

    # Phase 2: TARGETED improvement on B only, `strength` additional REINFORCE episodes.
    # strength=0 is a genuine no-op (range(0) loop body never runs) -- the control cell.
    _train_all_on_agent(
        agent, env_b_train, seed, p0_episodes=0, p1_episodes=strength,
        steps_per_episode=steps, rung_id=f"B_perturb_strength{strength}",
        total_denominator=denom,
    )

    # Phase 3: re-measure both. A is UNRELATED to the Phase-2 manipulation; B is the target.
    a_after = _eval_agent(agent, kw_a, seed, eval_eps, steps, "post_perturb_A")
    b_after = _eval_agent(agent, kw_b, seed, eval_eps, steps, "post_perturb_B")
    zg.observe(agent)

    interference_delta = round(
        float(a_after["survival_horizon"]) - float(a_baseline["survival_horizon"]), 6
    )
    targeted_delta = round(
        float(b_after["survival_horizon"]) - float(b_baseline["survival_horizon"]), 6
    )
    return {
        "seed": int(seed),
        "strength": int(strength),
        "a_baseline": a_baseline,
        "b_baseline": b_baseline,
        "a_after": a_after,
        "b_after": b_after,
        "interference_delta_survival": interference_delta,
        "targeted_delta_survival": targeted_delta,
    }


def run_experiment(
    seeds: List[int], strengths: List[int], p0: int, p1_acquire: int, eval_eps: int,
    steps: int,
) -> Dict[str, Any]:
    kw_a = _env_kwargs("A")
    kw_b = _env_kwargs("B")
    steps_f = float(steps)
    survival_floor = SURVIVAL_FLOOR_FRAC * steps_f
    random_margin = RANDOM_MARGIN_FRAC * steps_f
    move_floor = TARGETED_MOVE_FLOOR_FRAC * steps_f
    abs_interference_floor = INTERFERENCE_ABS_FLOOR_FRAC * steps_f

    print(
        f"MECH-471 local-update interference test ({len(strengths)} strengths x "
        f"{len(seeds)} seeds; P0={p0} P1_acquire={p1_acquire} steps={steps} eval={eval_eps})",
        flush=True,
    )

    # Random-walk anchors: cheap, computed once per (env, seed) -- readiness/achievability
    # reference only, not part of the boot training grid.
    random_a = {s: _eval_random(kw_a, s, eval_eps, steps) for s in seeds}
    random_b = {s: _eval_random(kw_b, s, eval_eps, steps) for s in seeds}

    zg = ZGoalStreamAccumulator()
    all_cells: List[Dict[str, Any]] = []
    per_strength: Dict[int, List[Dict[str, Any]]] = {s: [] for s in strengths}

    for strength in strengths:
        for seed in seeds:
            print(f"Seed {seed} Condition strength_{strength}", flush=True)
            slice_cfg = {
                "kind": "mech471_local_update_interference",
                "env_kwargs_a": dict(kw_a),
                "env_kwargs_b": dict(kw_b),
                "p0_warmup_episodes": int(p0),
                "p1_acquire_episodes": int(p1_acquire),
                "strength": int(strength),
                "steps_per_episode": int(steps),
                "eval_episodes": int(eval_eps),
            }
            with arm_cell(
                seed, config_slice=slice_cfg, script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=(strength != 0),
            ) as cell:
                row = _run_cell(seed, strength, p0, p1_acquire, eval_eps, steps, kw_a, kw_b, zg)
                cell.stamp(row)
            all_cells.append(row)
            per_strength[strength].append(row)
            acquired_ok = (
                float(row["a_baseline"]["survival_horizon"]) >= survival_floor
                and float(row["b_baseline"]["survival_horizon"]) >= survival_floor
            )
            print(
                f"verdict: {'PASS' if acquired_ok else 'FAIL'} "
                f"(strength={strength} seed={seed} "
                f"a_baseline={row['a_baseline']['survival_horizon']} "
                f"b_baseline={row['b_baseline']['survival_horizon']} "
                f"interference_delta={row['interference_delta_survival']})",
                flush=True,
            )

    control_rows = per_strength[strengths[0]]  # strength=0 is always first in the list
    assert strengths[0] == 0, "STRENGTH_LEVELS[0] must be 0 (the control arm)"

    # ---- readiness / non-degeneracy ------------------------------------------------------
    def _worst(rows: List[Dict[str, Any]], key_path: Tuple[str, str]) -> float:
        outer, inner = key_path
        return min(float(r[outer][inner]) for r in rows) if rows else 0.0

    a_baseline_worst = _worst(control_rows, ("a_baseline", "survival_horizon"))
    b_baseline_worst = _worst(control_rows, ("b_baseline", "survival_horizon"))
    a_random_worst_margin = min(
        float(r["a_baseline"]["survival_horizon"]) - float(random_a[r["seed"]]["survival_horizon"])
        for r in control_rows
    ) if control_rows else 0.0
    b_random_worst_margin = min(
        float(r["b_baseline"]["survival_horizon"]) - float(random_b[r["seed"]]["survival_horizon"])
        for r in control_rows
    ) if control_rows else 0.0

    a_baseline_values = [float(r["a_baseline"]["survival_horizon"]) for r in control_rows]
    b_baseline_values = [float(r["b_baseline"]["survival_horizon"]) for r in control_rows]
    a_baseline_variance = (
        len(set(round(v, 3) for v in a_baseline_values)) > 1 if len(a_baseline_values) > 1 else False
    )
    b_baseline_variance = (
        len(set(round(v, 3) for v in b_baseline_values)) > 1 if len(b_baseline_values) > 1 else False
    )

    max_strength = strengths[-1]
    max_rows = per_strength[max_strength]
    targeted_move_worst = (
        min(abs(float(r["targeted_delta_survival"])) for r in max_rows) if max_rows else 0.0
    )

    acquisition_ok = bool(
        a_baseline_worst >= survival_floor and b_baseline_worst >= survival_floor
        and a_random_worst_margin >= random_margin and b_random_worst_margin >= random_margin
    )
    cross_seed_variance_ok = bool(a_baseline_variance and b_baseline_variance)
    targeted_moved = bool(targeted_move_worst >= move_floor)

    # ---- dose-response readout ------------------------------------------------------------
    per_strength_mean_delta: Dict[int, float] = {}
    per_strength_sd_delta: Dict[int, float] = {}
    for s in strengths:
        deltas = [float(r["interference_delta_survival"]) for r in per_strength[s]]
        per_strength_mean_delta[s] = round(statistics.fmean(deltas), 6) if deltas else 0.0
        per_strength_sd_delta[s] = (
            round(statistics.pstdev(deltas), 6) if len(deltas) > 1 else 0.0
        )

    control_deltas = [float(r["interference_delta_survival"]) for r in control_rows]
    control_sd = statistics.pstdev(control_deltas) if len(control_deltas) > 1 else 0.0
    interference_margin = round(max(abs_interference_floor, INTERFERENCE_SD_MULT * control_sd), 6)

    top_two = strengths[-2:] if len(strengths) >= 2 else strengths[-1:]
    top_two_deltas = [per_strength_mean_delta[s] for s in top_two]
    dose_response_confirmed = bool(
        per_strength_mean_delta[max_strength] < 0.0
        and (per_strength_mean_delta[max_strength] <= per_strength_mean_delta[0] - interference_margin)
        and all(d <= -(CONFIRM_STRONG_FRAC * interference_margin) for d in top_two_deltas)
    )
    interference_not_detected = bool(
        all(
            per_strength_mean_delta[s] >= -interference_margin
            for s in strengths
        )
    )

    if not acquisition_ok or not cross_seed_variance_ok:
        outcome, direction, label = "FAIL", "unknown", "substrate_not_ready_requeue"
    elif not targeted_moved:
        outcome, direction, label = "FAIL", "unknown", "substrate_not_ready_requeue"
    elif dose_response_confirmed:
        outcome, direction, label = "FAIL", "supports", "local_update_interference_confirmed"
    elif interference_not_detected:
        outcome, direction, label = "PASS", "weakens", "local_update_interference_not_detected"
    else:
        outcome, direction, label = "FAIL", "mixed", "interference_dose_response_ambiguous"

    degeneracy = check_degeneracy({
        "a_baseline_survival_horizon": {"values": a_baseline_values, "floor": survival_floor},
        "b_baseline_survival_horizon": {"values": b_baseline_values, "floor": survival_floor},
        "targeted_move_at_max_strength": {
            "values": [abs(float(r["targeted_delta_survival"])) for r in max_rows],
            "floor": move_floor,
        },
    })
    non_degenerate = bool(degeneracy["non_degenerate"] and acquisition_ok and cross_seed_variance_ok)
    degeneracy_reason = degeneracy["degeneracy_reason"]
    if not acquisition_ok and not degeneracy_reason:
        degeneracy_reason = "acquired competence(s) below floor or below random-walk margin"
    if not cross_seed_variance_ok and not degeneracy_reason:
        degeneracy_reason = "no cross-seed variance in acquired baseline competence"

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "readiness": {
            "acquisition_ok": acquisition_ok,
            "cross_seed_variance_ok": cross_seed_variance_ok,
            "targeted_moved": targeted_moved,
            "a_baseline_worst_seed": round(a_baseline_worst, 6),
            "b_baseline_worst_seed": round(b_baseline_worst, 6),
            "a_random_worst_margin": round(a_random_worst_margin, 6),
            "b_random_worst_margin": round(b_random_worst_margin, 6),
            "targeted_move_worst_seed_abs": round(targeted_move_worst, 6),
            "survival_floor_ticks": round(survival_floor, 6),
            "random_margin_ticks": round(random_margin, 6),
            "targeted_move_floor_ticks": round(move_floor, 6),
        },
        "dose_response": {
            "per_strength_mean_interference_delta": per_strength_mean_delta,
            "per_strength_sd_interference_delta": per_strength_sd_delta,
            "interference_margin_ticks": interference_margin,
            "control_arm_delta_sd": round(control_sd, 6),
            "dose_response_confirmed": dose_response_confirmed,
            "interference_not_detected": interference_not_detected,
            "strengths": list(strengths),
        },
        "headline": {
            "max_strength": max_strength,
            "mean_interference_delta_at_max_strength": per_strength_mean_delta[max_strength],
            "mean_interference_delta_at_control": per_strength_mean_delta[0],
        },
        "random_walk_anchors": {
            "env_a_survival_horizon_per_seed": {
                str(s): round(float(random_a[s]["survival_horizon"]), 6) for s in seeds
            },
            "env_b_survival_horizon_per_seed": {
                str(s): round(float(random_b[s]["survival_horizon"]), 6) for s in seeds
            },
        },
        "arm_results": all_cells,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "z_goal_stream_stats": zg.stats(),
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation_label": result["interpretation_label"],
        "readiness": result["readiness"],
        "dose_response": result["dose_response"],
        "headline": result["headline"],
        "random_walk_anchors": result["random_walk_anchors"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": ["strength_0"],
            "reuse_eligible": True,
            "note": (
                "strength_0 (no targeted perturbation, the dual-competence-acquired control) "
                "is emitted reuse-ELIGIBLE with include_driver_script_in_hash=False as a "
                "mint-as-you-go baseline for any future MECH-471 cluster-E successor -- no "
                "separate baseline-only mint job."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "interference_delta_survival = env-A survival_horizon (mean ticks survived/"
            "episode) measured AFTER the Phase-2 targeted REINFORCE update on env B, minus "
            "the SAME statistic measured immediately after Phase-1 acquisition (env A is "
            "never touched by Phase 2). Reported per targeted-update STRENGTH (additional "
            "REINFORCE episodes on env B only), against a strength=0 no-perturbation control "
            "used to estimate the pure eval-noise margin. FAIL (interference confirmed, grows "
            "with strength beyond the margin) SUPPORTS MECH-471; PASS (no interference across "
            "the full strength range) WEAKENS it -- polarity is deliberate, see claims.yaml."
        ),
        "notes": (
            "MECH-471 EXP-0399 local-update interference test. Two separable competences "
            "(survival/avoidance under two reef-placement GEOMETRIES -- corner safe-zones vs "
            "a persistent banded reef corridor, same n_reef_patches -- on the SAME shared "
            "all-ON REE bias_head/"
            "OFC-devaluation REINFORCE policy substrate, e2 frozen throughout per the 724-A0 "
            "recipe). Survival (not foraging) was chosen as the DV specifically to avoid the "
            "well-documented MECH-457 foraging-competence ceiling (719a/724/728/734/742/780/"
            "788), which V3-EXQ-728 showed the same all-ON stack does NOT share for survival. "
            "Testable on existing substrate; no new architecture. Does not build or exercise "
            "any provenance/rollback machinery (that is what MECH-471 would motivate building, "
            "not what is tested here)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-875 MECH-471 local-update interference test (EXP-0399)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--strengths", type=int, nargs="*", default=None)
    parser.add_argument("--p0", type=int, default=None)
    parser.add_argument("--p1-acquire", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(DRY_SEEDS)
        strengths = list(DRY_STRENGTH_LEVELS)
        p0, p1_acquire = DRY_P0, DRY_P1_ACQUIRE
        eval_eps, steps = DRY_EVAL, DRY_STEPS
    else:
        seeds = list(SEEDS)
        strengths = list(STRENGTH_LEVELS)
        p0, p1_acquire = P0_WARMUP_EPISODES, P1_ACQUIRE_EPISODES
        eval_eps, steps = EVAL_EPISODES, STEPS_PER_EPISODE

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.strengths is not None:
        strengths = [int(s) for s in args.strengths]
    if args.p0 is not None:
        p0 = int(args.p0)
    if args.p1_acquire is not None:
        p1_acquire = int(args.p1_acquire)
    if args.eval_episodes is not None:
        eval_eps = int(args.eval_episodes)
    if args.steps is not None:
        steps = int(args.steps)

    if 0 not in strengths:
        raise ValueError("strengths must include 0 (the no-perturbation control)")
    strengths = sorted(strengths)

    result = run_experiment(
        seeds=seeds, strengths=strengths, p0=p0, p1_acquire=p1_acquire,
        eval_eps=eval_eps, steps=steps,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds,
        "strengths": strengths,
        "p0_warmup_episodes": p0,
        "p1_acquire_episodes": p1_acquire,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "env_kwargs_a": _env_kwargs("A"),
        "env_kwargs_b": _env_kwargs("B"),
        "survival_floor_frac": SURVIVAL_FLOOR_FRAC,
        "random_margin_frac": RANDOM_MARGIN_FRAC,
        "targeted_move_floor_frac": TARGETED_MOVE_FLOOR_FRAC,
        "interference_abs_floor_frac": INTERFERENCE_ABS_FLOOR_FRAC,
        "interference_sd_mult": INTERFERENCE_SD_MULT,
        "confirm_strong_frac": CONFIRM_STRONG_FRAC,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    stamp_recording_core(
        manifest, config=cfg, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        stamp=False,  # already stamped above (need z_goal_stream_stats folded in first)
        z_goal_stream_stats=result.get("z_goal_stream_stats"),  # so the --dry-run smoke
                                                                # print reports wired=True
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"direction={result['evidence_direction']} "
        f"acquisition_ok={result['readiness']['acquisition_ok']} "
        f"targeted_moved={result['readiness']['targeted_moved']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(
        f"  per_strength_mean_interference_delta="
        f"{result['dose_response']['per_strength_mean_interference_delta']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
