"""
V3-EXQ-591g -- LIVE (closed-loop) validation of the 591f crossing-count Phase 0->1 gate.

SLEEP DRIVER: K=never (SleepLoopManager instantiated with K > total episodes via the
diversity-armed builder; never fires during this readiness probe).

red-team (fable): see RED-TEAM VERDICT section below.

QUESTION
--------
V3-EXQ-591f (PASS, 2026-06-15) established that a CROSSING-COUNT criterion (advance only
once the cumulative post-ep_min spike-bar crossings reach 3) ADMITS the genuine explorers
(seeds 42/43/44) and REJECTS the seed-45 false advancer -- where the single-episode SPIKE
gate, K-of-N, and EMA-of-level all failed to discriminate (591d/591e).

But 591f validated that criterion **OFFLINE**. It ran the live scheduler under the LEGACY
spike gate and then replayed the candidate criteria over the resulting per-episode h_pos
sequence. The criterion has since landed in InfantCurriculumScheduler as an OPT-IN knob
(`phase_0to1_use_crossing_count`, default OFF) and -- as of this experiment -- NO experiment
script has ever set it. It has never run in closed loop.

WHY THE OFFLINE REPLAY CAN BE WRONG (the mechanism this probe tests)
--------------------------------------------------------------------
The scheduler's phase feeds BACK into the agent: `config_overrides()` raises
`novelty_bonus_weight` from 0.5 (Phase 0) to 0.7 (Phase 1), and the 591 readiness lineage
applies that override every episode. So the gate's own decision changes the exploration
drive that generates the statistic the gate reads.

591f replayed telemetry produced by a run that advanced at the SPIKE point (first crossing
after ep 100). Every episode after that point was generated at novelty_bonus_weight=0.7.
A TRUE crossing-count run is still in Phase 0 -- at 0.5 -- while it waits for its 3rd
crossing, and the crossing-count decision window extends PAST the spike advance point.
The replay therefore counts crossings off telemetry biased UPWARD relative to the run the
criterion would actually produce.

Directional prediction: the live crossing counts should be <= the replayed ones for seeds
that advanced early under the spike gate. The failure mode that matters is a SELF-DEFEATING
GATE -- a genuine explorer held in Phase 0 at the lower novelty weight never reaches 3
crossings, so the criterion that discriminated on paper starves in closed loop.

This is exactly the ARC-019 NON-DEGENERACY PRECONDITION ("the phase-advance gate criteria
must be reachable"), which ARC-019 requires to be independently verified BEFORE any
staged-vs-unstaged outcome comparison counts as evidence either way.

DESIGN -- discriminative pair (EVB-1189 dispatch_mode=discriminative_pair)
--------------------------------------------------------------------------
Two arms, matched seeds, identical in every respect except the Phase 0->1 gate RULE:

  ARM_SPIKE    (control / ablation) -- legacy single-episode spike gate (scheduler default).
                 Bit-identical to the 591c/591d/591e/591f readiness runner; it reproduces
                 the telemetry basis 591f replayed over.
  ARM_CROSSING (primary)            -- `phase_0to1_use_crossing_count=True`, min 3 crossings,
                 wired LIVE into the scheduler so the gate's decision feeds back through
                 `novelty_bonus_weight`.

Seeds 42-46 (5 >= EVB-1189 min_shared_seeds=2), 160 episodes, 200 steps/episode,
diversity-armed agent build (MECH-313 noise floor + MECH-314 structured curiosity, SP-CEM
main-path default) -- all faithful to 591b/591c/591d/591e/591f.

SEED-44 NOTE: the skill's standing caution to substitute seed 45 for seed 44 is a
reef-config env hazard. This env is CausalGridWorldV2 with all infant features OFF, and
seed 44 is one of the lineage's three ORACLE genuine explorers (36 post-ep_min crossings in
591f). Changing the seed set would break comparability with the entire 591 lineage, so
seeds 42-46 are retained deliberately.

THE ORACLE (independent of the criterion under test -- NOT the criterion itself)
--------------------------------------------------------------------------------
A criterion cannot be its own ground truth. 591d/591f's genuine-explorer oracle combined a
LEVEL statistic (h_pos_mean floor) with a CROSSING COUNT (>=2) -- and the crossing-count
half is nearly the criterion being tested, so reusing it whole would let the gate certify
its own subject.

This probe therefore uses the LEVEL half ONLY: a seed is an ORACLE genuine explorer iff its
full-run mean per-episode h_pos, measured on ARM_SPIKE (the control), clears
ORACLE_H_POS_MEAN_FLOOR = 0.20. Dropping the crossing-count half is what removes the
circularity; measuring on the CONTROL arm is what keeps the label independent of the
crossing-count rule under test.

Why measuring on ARM_SPIKE is not itself circular: a seed that advances under the spike gate
spends its remaining episodes at the higher Phase-1 novelty_bonus_weight, which inflates its
full-run mean. That inflation does NOT manufacture genuine labels -- seed 45 ALSO advanced
under the spike gate (ep 142) and still records h_pos_mean 0.1404, well below the floor.
The recorded 591f partition is clean with margin on both sides:
    genuine     seed 42 = 0.5621, seed 43 = 0.3226, seed 44 = 0.8424
    non-genuine seed 45 = 0.1404, seed 46 = 0.0375
This is the SAME statistic 591d/591f recorded and the 591d autopsy user-adjudicated, so the
floor is calibrated against recorded data rather than guessed -- and REFERENCE_591F below
freezes those cells so `assert_anchor_reachable` can prove, at setup, that every readiness
predicate this script ships can actually clear its own gate on the known-positive control.

ACCEPTANCE
----------
LOAD-BEARING C_live_gate_discriminates: in ARM_CROSSING, for EVERY seed, the live
phase-advance decision agrees with the ORACLE label -- every oracle-genuine explorer
reaches Phase 1, and every oracle-non-explorer does not.

NON-DEGENERACY C_arms_differ (not load-bearing): the manipulation must reach the DV -- the
two arms must differ on at least one seed's advance EPISODE or final phase. If the two
gates produce bit-identical advance decisions on every seed, the pair discriminates nothing
and no verdict is licensed.

READINESS PRECONDITIONS (below-floor -> substrate_not_ready_requeue, never a verdict label)
-------------------------------------------------------------------------------------------
All four are measured on ARM_SPIKE, the positive control (591c: 4/5 seeds advanced), and
each asserts the SAME statistic class its dependent criterion routes on:

  P1 spike_arm_reproduces_advance      -- COUNT of ARM_SPIKE seeds reaching Phase 1 >= 2.
       (The lineage reproduced at all. Below -> the 591c basis did not reproduce.)
  P2 crossing_counts_reach_gate_minimum -- MAX over seeds of post-ep_min spike-bar crossing
       count >= PHASE_01_CROSSING_COUNT_MIN. This is the SAME COUNT statistic the
       crossing-count gate routes on. Below -> the gate could not admit ANY seed under any
       outcome, so C_live_gate_discriminates would be STARVED, not falsified.
  P3 oracle_genuine_explorers_present  -- COUNT of oracle-genuine seeds >= 1.
  P4 oracle_non_explorers_present      -- COUNT of oracle-non-genuine seeds >= 1.
       (P3+P4 together: the oracle must actually SEPARATE the seed draw. If every seed
       carries the same label, "discriminates" is untestable on this draw.)

DV-SYMMETRY INVARIANCE (mandatory declaration, per arm)
-------------------------------------------------------
DV (both arms): per-seed Phase 0->1 advance EPISODE and final phase.
Manipulation: the gate RULE -- the cumulative post-ep_min crossing count at which advance
fires (1 for ARM_SPIKE, 3 for ARM_CROSSING).
Symmetry group of the DV: temporal/ordinal position in the episode sequence.
The manipulation is NOT invariant under it. The DV is the episode index at which a
cumulative crossing count FIRST reaches the threshold; that index is a strictly
non-decreasing function of the threshold and strictly increases whenever a seed records
>= 2 crossings. It is not a broadcast additive constant (it changes an ordinal threshold,
not a level), not a monotone rescaling of the DV (it moves the crossing at which the DV
fires, which argmax/rank DVs would be blind to but an EPISODE INDEX is not), and not a
permutation of interchangeable units (episode order is load-bearing, not exchangeable).
The one regime where the manipulation IS invisible -- a seed recording < 2 crossings, where
both rules give the same (non-)advance -- is precisely what C_arms_differ tests for and P2
guards against as a whole-run condition.

SCOPE LIMITATION (recorded, not silently inherited)
----------------------------------------------------
`sched.env_kwargs()` is called informationally and NOT spread into the env constructor, and
`config_overrides()` supplies `novelty_bonus_weight` only. This is faithful to 591b-591f
(591b/591d compute env_kwargs and discard it; 591e/591f annotate the call "informational").
Only the ORIGINAL V3-EXQ-591 curriculum-vs-flat run applied env_kwargs.
Consequence: the ENVIRONMENTAL staging of the infant curriculum (harm gradient, transient
benefit, microhabitat) is NOT exercised here, and neither are the `residue_scale_factor` /
`offline_integration_frequency` overrides. That is deliberate and does not affect the
load-bearing DV -- every pre-advance episode is Phase 0, whose env_kwargs are all-False and
equal to the CausalGridWorldV2 constructor defaults, and phases never retreat, so no
post-advance state can feed back into a Phase 0->1 decision. Applying env_kwargs here would
add a second difference from the 591f replay basis and confound the gate comparison.
It DOES mean this probe validates the Phase 0->1 GATE, not the full staged curriculum;
ARC-019's staged-vs-unstaged outcome comparison remains out of scope and substrate-gated.

RED-TEAM VERDICT
----------------
red-team (fable): recorded in the queue entry note for V3-EXQ-591g.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from infant_curriculum import (  # noqa: E402
    InfantCurriculumScheduler,
    H_POS_FRAC_OF_MAX,
    PHASE_01_CROSSING_COUNT_MIN,
    PHASE_EP_MIN,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Reuse the canonical 591 helpers + constants (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _extract_obs,
    BODY_OBS_DIM,
    WORLD_OBS_DIM,
    GRID_SIZE,
    ACTION_DIM,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

QUEUE_ID = "V3-EXQ-591g"
EXPERIMENT_TYPE = "v3_exq_591g_isef005_phase01_gate_live"
EXPERIMENT_PURPOSE = "diagnostic"

# ARC-019's stated NON-DEGENERACY PRECONDITION is exactly what this probe establishes or
# refutes. It does NOT test ARC-019's CONFIRMING/FALSIFYING legs (staged-vs-unstaged outcome
# quality at matched budget), which remain substrate-gated -- hence non_contributory.
CLAIM_IDS: List[str] = ["ARC-019"]
EVIDENCE_DIRECTION = "non_contributory"

# --- pre-registered constants (NOT derived from this run's own statistics) ---------------
SEEDS = [42, 43, 44, 45, 46]     # the lineage seed draw (591b/591c/591d/591e/591f)
N_EPISODES = 160                 # faithful to 591b-591f
STEPS_PER_EPISODE = 200          # faithful to 591b-591f

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # the spike bar, ~0.9940 at grid 12

# Readiness precondition thresholds (all pre-registered).
P1_MIN_SPIKE_ADVANCERS = 2
P2_MIN_MAX_CROSSINGS = PHASE_01_CROSSING_COUNT_MIN   # 3 -- SAME statistic the gate routes on
P3_MIN_ORACLE_GENUINE = 1
P4_MIN_ORACLE_NON_GENUINE = 1
P5_MIN_LIVE_FALSE_ADVANCERS = 1

# ORACLE: level-only (the crossing-count half of the 591d/591f oracle is dropped -- it is
# nearly the criterion under test). Floor unchanged from GENUINE_EXPLORATION_H_POS_MEAN_FLOOR.
ORACLE_H_POS_MEAN_FLOOR = 0.20
ORACLE_WINDOW_END = PHASE_EP_MIN[1]   # 100 -- pre-ep_min slice kept as INFORMATIONAL only

# Frozen positive-control reference: the recorded per-seed cells of
# v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3 (per_seed_criteria).
# Used ONLY by assert_anchor_reachable at setup, to prove each shipped readiness predicate
# can clear its own gate on a known-good control. Never used as a live label.
REFERENCE_591F = [
    {"seed": 42, "h_pos_mean": 0.5621, "n_eligible_ge_threshold": 7,  "spike_advanced": True},
    {"seed": 43, "h_pos_mean": 0.3226, "n_eligible_ge_threshold": 6,  "spike_advanced": True},
    {"seed": 44, "h_pos_mean": 0.8424, "n_eligible_ge_threshold": 36, "spike_advanced": True},
    {"seed": 45, "h_pos_mean": 0.1404, "n_eligible_ge_threshold": 2,  "spike_advanced": True},
    {"seed": 46, "h_pos_mean": 0.0375, "n_eligible_ge_threshold": 0,  "spike_advanced": False},
]
REFERENCE_591F_SOURCE = (
    "v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3.per_seed_criteria")

# --- THE SHIPPED PREDICATES (the same callables the live cells are scored with) ----------
def _is_oracle_genuine(cell) -> bool:
    return float(cell["h_pos_mean"]) >= ORACLE_H_POS_MEAN_FLOOR

def _is_oracle_non_genuine(cell) -> bool:
    return not _is_oracle_genuine(cell)

def _meets_crossing_minimum(cell) -> bool:
    return int(cell["n_eligible_ge_threshold"]) >= P2_MIN_MAX_CROSSINGS

def _spike_advanced(cell) -> bool:
    return bool(cell["spike_advanced"])

def _is_live_false_advancer(cell) -> bool:
    """CONJUNCTIVE, as 591f's own `false_advancer` was: oracle-non-genuine AND admitted by
    the legacy spike gate. Splitting the 591d/591f oracle dropped this conjunction, which
    would let the gate's REJECT leg go unchallenged while still minting the strongest PASS
    label (a seed that never spikes is rejected by BOTH gates, so its rejection is not
    attributable to the crossing rule)."""
    return _is_oracle_non_genuine(cell) and _spike_advanced(cell)

ARM_SPIKE = "ARM_SPIKE"
ARM_CROSSING = "ARM_CROSSING"
ARMS = [ARM_SPIKE, ARM_CROSSING]

_ZG = ZGoalStreamAccumulator()


def _agent_config_kwargs() -> Dict[str, Any]:
    """The 591c diversity-armed build, as a declared dict (also the fingerprint slice)."""
    return {
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM,
        "z_goal_enabled": True,
        "drive_weight": 2.0,
        "novelty_bonus_weight": 0.5,
        "use_sleep_loop": True,
        "sleep_loop_episodes_K": N_EPISODES + 1,   # K=never
        "use_noise_floor": True,                   # MECH-313
        "use_structured_curiosity": True,          # MECH-314
    }


def _build_diversity_agent() -> REEAgent:
    """Bit-identical to the 591c/591d/591e/591f diversity-armed agent build."""
    cfg = REEConfig.from_dims(**_agent_config_kwargs())
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _config_slice(arm: str) -> Dict[str, Any]:
    """Everything the cell computation reads. The gate rule is INCLUDED -- it is what
    distinguishes the two arms, so omitting it would collapse their fingerprints."""
    return {
        "agent": _agent_config_kwargs(),
        "latent_alpha_world": 0.9,
        "sws_enabled": True,
        "rem_enabled": True,
        "env": {
            "size": GRID_SIZE,
            "resource_respawn_on_consume": True,
            "pos_telemetry_enabled": True,
            "traj_telemetry_enabled": True,
        },
        "schedule": {
            "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "gate": {
            "phase_0to1_use_crossing_count": arm == ARM_CROSSING,
            "phase_0to1_crossing_count_min": PHASE_01_CROSSING_COUNT_MIN,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
        },
    }


def _run_cell(*, arm: str, seed: int, n_episodes: int,
              steps_per_episode: int = STEPS_PER_EPISODE) -> Dict[str, Any]:
    """One (arm x seed) cell. Mirrors the 591c seed runner; only the gate rule differs."""
    print(f"Seed {seed} Condition {arm}", flush=True)

    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,   # cross-driver reusable mint (both arms)
    ) as cell:
        torch.manual_seed(seed)
        agent = _build_diversity_agent()
        sched = InfantCurriculumScheduler(
            grid_size=GRID_SIZE,
            phase_0to1_use_crossing_count=(arm == ARM_CROSSING),
            phase_0to1_crossing_count_min=PHASE_01_CROSSING_COUNT_MIN,
        )

        h_pos_window: deque = deque(maxlen=100)   # rolling (informational only)
        per_ep_h_pos: List[float] = []
        phase_01_at: Optional[int] = None
        phase_12_at: Optional[int] = None

        for ep in range(n_episodes):
            # Informational only -- NOT spread into the env. See SCOPE LIMITATION.
            sched.env_kwargs()
            agent.config.e3.novelty_bonus_weight = float(
                sched.config_overrides().get("novelty_bonus_weight", 0.5))

            env = CausalGridWorldV2(
                size=GRID_SIZE,
                seed=seed * n_episodes + ep,
                resource_respawn_on_consume=True,
                pos_telemetry_enabled=True,
                traj_telemetry_enabled=True,
            )
            _flat, obs_dict = env.reset()
            ob, ow = _extract_obs(obs_dict)

            ep_h_pos = -1.0
            ep_benefit_contacts = 0

            for _step in range(steps_per_episode):
                with torch.no_grad():
                    action = agent.act_with_split_obs(obs_body=ob, obs_world=ow)
                ai = int(action.argmax().item()) % ACTION_DIM
                _o, harm_signal, done, info, obs_dict = env.step(ai)
                agent.update_residue(float(harm_signal))
                ob, ow = _extract_obs(obs_dict)
                benefit = float(ob[11].item()) if ob.shape[0] > 11 else 0.0
                energy = float(ob[3].item()) if ob.shape[0] > 3 else 0.5
                drive = max(0.0, min(1.0, 1.0 - energy))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                ep_h_pos = float(info.get("pos_entropy", -1.0))
                ep_benefit_contacts += int(
                    float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0)
                if done:
                    _flat, obs_dict = env.reset()
                    ob, ow = _extract_obs(obs_dict)

            z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
            cov = float(agent.residue_field.get_coverage_telemetry()["residue_coverage_pct"])

            per_ep_h_pos.append(ep_h_pos)
            h_pos_window.append(ep_h_pos)

            prev_phase = sched.current_phase
            sched.update(
                ep,
                h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
                z_goal_norm=z_norm,
                benefit_contacts=ep_benefit_contacts,
                residue_coverage_pct=cov,
            )
            if prev_phase == 0 and sched.current_phase >= 1 and phase_01_at is None:
                phase_01_at = ep
            if prev_phase <= 1 and sched.current_phase >= 2 and phase_12_at is None:
                phase_12_at = ep

            if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
                print(
                    f"  [train] {arm} seed={seed} ep {ep + 1}/{n_episodes}"
                    f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}"
                    f" crossings={sched.phase_summary().get('phase01_crossing_count', 0)}",
                    flush=True,
                )

        _ZG.observe(agent)

        valid = [h for h in per_ep_h_pos if h >= 0.0]
        pre = [h for h in per_ep_h_pos[:ORACLE_WINDOW_END] if h >= 0.0]
        post = [h for h in per_ep_h_pos[ORACLE_WINDOW_END:] if h >= 0.0]

        row: Dict[str, Any] = {
            "arm_id": arm,
            "seed": seed,
            "final_phase": sched.current_phase,
            "reached_phase1": bool(sched.current_phase >= 1),
            "phase_01_at": phase_01_at,
            "phase_12_at": phase_12_at,
            # the COUNT statistic the crossing-count gate routes on
            "post_ep_min_crossings": sum(1 for h in post if h >= PHASE_01_THRESHOLD),
            "scheduler_crossing_count": int(
                sched.phase_summary().get("phase01_crossing_count", 0)),
            # ORACLE input is h_pos_mean_full_run (below); this pre-ep_min slice is
            # INFORMATIONAL ONLY and governs nothing load-bearing.
            "h_pos_mean_pre_ep_min": (sum(pre) / len(pre)) if pre else -1.0,
            # informational -- lets a successor recalibrate without re-running
            "h_pos_mean_full_run": (sum(valid) / len(valid)) if valid else -1.0,
            "h_pos_max_full_run": max(valid) if valid else -1.0,
            "n_pre_ep_min_crossings": sum(1 for h in pre if h >= PHASE_01_THRESHOLD),
            "per_episode_h_pos": per_ep_h_pos,
            "final_z_goal_norm": float(z_norm),
            "final_residue_coverage_pct": float(cov),
            "phase_summary": sched.phase_summary(),
        }
        cell.stamp(row)

    verdict = "PASS" if row["reached_phase1"] else "FAIL"
    print(f"verdict: {verdict}", flush=True)
    return row


def _assert_readiness_anchors_reachable() -> List[Dict[str, Any]]:
    """Prove at SETUP that every shipped readiness predicate can clear its own gate on the
    frozen 591f positive control. A predicate narrower than the state it anchors to is a
    guaranteed false negative -- it would report met=False on every run forever and mislabel
    an instrument-specification gap as a substrate verdict. Raises AnchorUnreachable."""
    n = len(REFERENCE_591F)
    return [
        assert_anchor_reachable(
            anchor_name="spike_arm_reproduces_advance",
            reference_cells=REFERENCE_591F, score_fn=_spike_advanced,
            threshold=P1_MIN_SPIKE_ADVANCERS / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="crossing_counts_reach_gate_minimum",
            reference_cells=REFERENCE_591F, score_fn=_meets_crossing_minimum,
            threshold=1.0 / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="oracle_genuine_explorers_present",
            reference_cells=REFERENCE_591F, score_fn=_is_oracle_genuine,
            threshold=P3_MIN_ORACLE_GENUINE / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="oracle_non_explorers_present",
            reference_cells=REFERENCE_591F, score_fn=_is_oracle_non_genuine,
            threshold=P4_MIN_ORACLE_NON_GENUINE / n, reference_source=REFERENCE_591F_SOURCE),
        assert_anchor_reachable(
            anchor_name="live_false_advancer_present",
            reference_cells=REFERENCE_591F, score_fn=_is_live_false_advancer,
            threshold=P5_MIN_LIVE_FALSE_ADVANCERS / n,
            reference_source=REFERENCE_591F_SOURCE),
    ]


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Setup-time guard -- runs on --dry-run too, before any compute is spent.
    anchor_payloads = _assert_readiness_anchors_reachable()
    print(f"[setup] readiness anchors reachable on {REFERENCE_591F_SOURCE}: "
          f"{[a['anchor_name'] for a in anchor_payloads]}", flush=True)
    seeds = [SEEDS[0], SEEDS[3]] if dry_run else SEEDS   # 42 (explorer) + 45 (false advancer)
    n_episodes = 3 if dry_run else N_EPISODES
    steps_per_episode = 25 if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            arm_results.append(_run_cell(arm=arm, seed=seed, n_episodes=n_episodes,
                                         steps_per_episode=steps_per_episode))

    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {
        a: {r["seed"]: r for r in arm_results if r["arm_id"] == a} for a in ARMS
    }
    spike = by_arm[ARM_SPIKE]
    cross = by_arm[ARM_CROSSING]

    # --- ORACLE labels: level-only, pre-ep_min window, identical in both arms ------------
    # THE SHIPPED PREDICATE, applied to the control arm's cells (same callable the
    # reference is scored with in the setup anchors below).
    oracle: Dict[int, bool] = {
        s: _is_oracle_genuine({"h_pos_mean": spike[s]["h_pos_mean_full_run"]}) for s in seeds
    }
    n_oracle_genuine = sum(1 for v in oracle.values() if v)
    n_oracle_non_genuine = sum(1 for v in oracle.values() if not v)
    live_false_advancers = [
        s_ for s_ in seeds if (not oracle[s_]) and spike[s_]["reached_phase1"]]
    n_live_false_advancers = len(live_false_advancers)

    # --- READINESS PRECONDITIONS (measured on ARM_SPIKE, the positive control) -----------
    n_spike_advancers = sum(1 for s in seeds if spike[s]["reached_phase1"])
    max_crossings = max((spike[s]["post_ep_min_crossings"] for s in seeds), default=0)
    worst_crossing_seed = max(seeds, key=lambda s: spike[s]["post_ep_min_crossings"]) \
        if seeds else None

    preconditions = [
        {
            "name": "spike_arm_reproduces_advance",
            "description": (
                "COUNT of ARM_SPIKE seeds reaching Phase 1. The legacy-gate control is the "
                "positive control (591c advanced 4/5 seeds); below floor means the lineage "
                "basis did not reproduce, so nothing here is a verdict."),
            "measured": n_spike_advancers,
            "threshold": P1_MIN_SPIKE_ADVANCERS,
            "direction": "lower",
            "control": "ARM_SPIKE = the legacy single-episode spike gate (591c reproduction)",
            "met": bool(n_spike_advancers >= P1_MIN_SPIKE_ADVANCERS),
        },
        {
            "name": "crossing_counts_reach_gate_minimum",
            "description": (
                "MAX over seeds of post-ep_min spike-bar crossing count in ARM_SPIKE. This is "
                "the SAME COUNT statistic the crossing-count gate routes on. Below floor, the "
                "gate could admit NO seed under any outcome, so C_live_gate_discriminates "
                "would be STARVED rather than falsified."),
            "measured": max_crossings,
            "threshold": P2_MIN_MAX_CROSSINGS,
            "direction": "lower",
            "control": f"best-crossing seed in ARM_SPIKE (seed {worst_crossing_seed})",
            "offending_cell": f"seed {worst_crossing_seed}",
            "met": bool(max_crossings >= P2_MIN_MAX_CROSSINGS),
        },
        {
            "name": "oracle_genuine_explorers_present",
            "description": (
                "COUNT of seeds the level oracle (full-run h_pos_mean on ARM_SPIKE) labels genuine. "
                "Below floor -> no positive class to admit."),
            "measured": n_oracle_genuine,
            "threshold": P3_MIN_ORACLE_GENUINE,
            "direction": "lower",
            "control": "full-run h_pos_mean measured on ARM_SPIKE (the control arm)",
            "met": bool(n_oracle_genuine >= P3_MIN_ORACLE_GENUINE),
        },
        {
            "name": "oracle_non_explorers_present",
            "description": (
                "COUNT of seeds the oracle labels NON-genuine. Below floor -> no negative "
                "class to reject, so 'discriminates' is untestable on this seed draw."),
            "measured": n_oracle_non_genuine,
            "threshold": P4_MIN_ORACLE_NON_GENUINE,
            "direction": "lower",
            "control": "full-run h_pos_mean measured on ARM_SPIKE (the control arm)",
            "met": bool(n_oracle_non_genuine >= P4_MIN_ORACLE_NON_GENUINE),
        },
        {
            "name": "live_false_advancer_present",
            "description": (
                "COUNT of seeds that are oracle-non-genuine AND were admitted by the legacy "
                "spike gate in ARM_SPIKE -- i.e. a live false advancer for the crossing gate "
                "to REJECT. Below floor, the reject leg is never challenged: a seed that "
                "never spikes is rejected by BOTH gates, so its rejection cannot be "
                "attributed to the crossing rule, and a PASS would overstate the result. "
                "591f's own false_advancer definition was conjunctive in exactly this way."),
            "measured": n_live_false_advancers,
            "threshold": P5_MIN_LIVE_FALSE_ADVANCERS,
            "direction": "lower",
            "control": "ARM_SPIKE advance decisions x the oracle label (591f: seed 45)",
            "offending_cell": f"live false advancers: {live_false_advancers}",
            "met": bool(n_live_false_advancers >= P5_MIN_LIVE_FALSE_ADVANCERS),
        },
    ]
    gate_green = all(p["met"] for p in preconditions)

    # --- CRITERIA -------------------------------------------------------------------------
    per_seed_agreement = {
        s: bool(cross[s]["reached_phase1"] == oracle[s]) for s in seeds
    }
    c_discriminates = bool(
        gate_green and not dropout_bypassed and all(per_seed_agreement.values()))

    # F4 GUARD: infant_curriculum._try_phase_0_to_1 advances Phase 0->1 unconditionally on
    # a missing-telemetry episode (`h_pos is None`) at ep >= PHASE_EP_MIN[1], under BOTH
    # criteria. Such an advance bypasses the crossing count the gate is supposed to apply,
    # so a resulting oracle disagreement is a telemetry-dropout artifact, NOT a refutation
    # of the crossing rule. Detect it and refuse to score the criterion if it fired.
    dropout_bypassed = [
        s_ for s_ in seeds
        if cross[s_]["reached_phase1"]
        and cross[s_]["scheduler_crossing_count"] < PHASE_01_CROSSING_COUNT_MIN
    ]
    n_missing_telemetry_episodes = {
        str(s_): sum(1 for h in cross[s_]["per_episode_h_pos"] if h < 0.0) for s_ in seeds
    }

    arms_differ = any(
        (spike[s]["reached_phase1"] != cross[s]["reached_phase1"])
        or (spike[s]["phase_01_at"] != cross[s]["phase_01_at"])
        for s in seeds
    )

    criteria = [
        {"name": "C_live_gate_discriminates", "load_bearing": True, "passed": c_discriminates},
        {"name": "C_arms_differ", "load_bearing": False, "passed": bool(arms_differ)},
    ]
    criteria_non_degenerate = {
        # The load-bearing criterion is only meaningful if readiness held AND the oracle
        # separated the draw AND the manipulation reached the DV AND no ARM_CROSSING seed
        # advanced via the telemetry-dropout bypass rather than via the crossing count.
        "C_live_gate_discriminates": bool(
            gate_green and n_oracle_genuine >= 1 and n_oracle_non_genuine >= 1
            and arms_differ and not dropout_bypassed),
        "C_arms_differ": bool(gate_green),
    }

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif dropout_bypassed:
        # A gate-bypassing advance happened; no verdict on the crossing rule is licensed.
        label = "substrate_not_ready_requeue"
    elif not arms_differ:
        label = "substrate_not_ready_requeue"
    elif c_discriminates:
        label = "crossing_count_gate_discriminates_live_closed_loop"
    else:
        disagreeing = [s for s in seeds if not per_seed_agreement[s]]
        held_back = [s for s in disagreeing if oracle[s] and not cross[s]["reached_phase1"]]
        label = ("crossing_count_gate_self_defeating_holds_back_genuine_explorers"
                 if held_back else
                 "crossing_count_gate_discrimination_lost_in_closed_loop")

    outcome = "PASS" if c_discriminates else "FAIL"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "status": outcome,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "simulation",
        "evidence_direction": EVIDENCE_DIRECTION,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "K=never",
        "dry_run": bool(dry_run),
        "arm_results": arm_results,
        "oracle_labels": {str(s): oracle[s] for s in seeds},
        "per_seed_agreement": {str(s): per_seed_agreement[s] for s in seeds},
        "combination_rule": (
            "C_live_gate_discriminates = readiness_gate_green AND "
            "for-every-seed(ARM_CROSSING.reached_phase1 == oracle_genuine_explorer). "
            "Plain AND over seeds; C_arms_differ is a non-degeneracy check, not a PASS input."),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "readiness_anchors": anchor_payloads,
        },
        "metrics": {
            "n_spike_advancers": n_spike_advancers,
            "n_crossing_advancers": sum(1 for s in seeds if cross[s]["reached_phase1"]),
            "max_post_ep_min_crossings_spike": max_crossings,
            "n_oracle_genuine": n_oracle_genuine,
            "n_oracle_non_genuine": n_oracle_non_genuine,
            "n_seeds_agreeing": sum(1 for v in per_seed_agreement.values() if v),
            "arms_differ": bool(arms_differ),
            "readiness_gate_green": bool(gate_green),
            "n_live_false_advancers": n_live_false_advancers,
            "dropout_bypassed_seeds": dropout_bypassed,
            "n_missing_telemetry_episodes_per_seed": n_missing_telemetry_episodes,
        },
        "informational_only": {"ORACLE_WINDOW_END": ORACLE_WINDOW_END},
        "registered_thresholds": {
            "PHASE_01_THRESHOLD": PHASE_01_THRESHOLD,
            "PHASE_01_CROSSING_COUNT_MIN": PHASE_01_CROSSING_COUNT_MIN,
            "ORACLE_H_POS_MEAN_FLOOR": ORACLE_H_POS_MEAN_FLOOR,
            "P1_MIN_SPIKE_ADVANCERS": P1_MIN_SPIKE_ADVANCERS,
            "P2_MIN_MAX_CROSSINGS": P2_MIN_MAX_CROSSINGS,
            "P3_MIN_ORACLE_GENUINE": P3_MIN_ORACLE_GENUINE,
            "P4_MIN_ORACLE_NON_GENUINE": P4_MIN_ORACLE_NON_GENUINE,
            "P5_MIN_LIVE_FALSE_ADVANCERS": P5_MIN_LIVE_FALSE_ADVANCERS,
        },
        "summary": {
            "scenario": (
                "Live closed-loop validation of the V3-EXQ-591f crossing-count Phase 0->1 "
                "curriculum gate: crossing-count gate (ARM_CROSSING) vs legacy spike gate "
                "(ARM_SPIKE), matched seeds, diversity-armed 591c build."),
            "interpretation": label,
            "pairwise_deltas": {
                str(s): {
                    "spike_phase_01_at": spike[s]["phase_01_at"],
                    "crossing_phase_01_at": cross[s]["phase_01_at"],
                    "spike_final_phase": spike[s]["final_phase"],
                    "crossing_final_phase": cross[s]["final_phase"],
                    "spike_post_ep_min_crossings": spike[s]["post_ep_min_crossings"],
                    "crossing_post_ep_min_crossings": cross[s]["post_ep_min_crossings"],
                    "oracle_genuine": oracle[s],
                    "agrees_with_oracle": per_seed_agreement[s],
                } for s in seeds
            },
        },
        "verdict_routing_note": (
            "outcome=FAIL is recorded both when readiness failed and when the gate was "
            "genuinely refuted; the label alone distinguishes them and NOTHING in the "
            "runner or coordinator reads the label string. The machine-readable distinction "
            "lives in interpretation.preconditions[] instead: every entry carries numeric "
            "measured+threshold+direction, so build_experiment_indexes._precondition_unmet "
            "recomputes them and emits adjudication=precondition_unmet, which "
            "generate_pending_review.py surfaces under 'Diagnostic adjudication required'. "
            "There is no auto-requeue anywhere -- 'requeue' in the label is a recommendation "
            "to the adjudicating /failure-autopsy, not a mechanism (lineage-wide, 591b-591f)."),
        "scope_limitation": (
            "env_kwargs() is called informationally and NOT applied (faithful to 591b-591f); "
            "config_overrides() supplies novelty_bonus_weight only. Validates the Phase 0->1 "
            "GATE, not the full staged curriculum. ARC-019's staged-vs-unstaged outcome "
            "comparison is out of scope and remains substrate-gated."),
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "agent": _agent_config_kwargs(),
            "env": _config_slice(ARM_SPIKE)["env"],
            "schedule": {"n_episodes": n_episodes, "steps_per_episode": steps_per_episode},
            "arms": {a: _config_slice(a)["gate"] for a in ARMS},
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    return {"outcome": outcome, "manifest_path": out_path, "label": label}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = result["manifest_path"]
    print(f"outcome={result['outcome']} label={result['label']}")
    print(f"manifest: {out_path}")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
