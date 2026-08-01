#!/opt/local/bin/python3
"""V3-EXQ-865 -- Q-081 cheap reach pre-flight SCAN for the REFRAMED measured
pair: does the landmark-manipulation lever give reach to z_goal (PRIMARY), or
-- falling back -- to z_harm_a (FALLBACK), or does the pair need to escalate
past a config-lever probe entirely?

Follow-on of the 2026-08-01 /claim-synthesis Q-081 measured-pair reframe
(REE_assembly/evidence/planning/claim_synthesis_Q-081_2026-08-01.md,
elastic-merkle-e0cca8 -> chip chip-20260801-q081-zgoal-probe), itself the
direct successor of V3-EXQ-849 (the same discipline applied to the now-
retired RV(z_world, operating_mode) pair). 824/824a/838/849 confirmed ZERO
reach from four structurally different manipulation families to
(z_world, operating_mode); the claim-synthesis traced WHY (operating_mode's
precursor set has no wired edge from z_world/landmark structure at all) and
found TWO genuinely different, already-coded, single-hop paths from the SAME
landmark-manipulation lever into cross-stream targets: z_goal (via E1's
gate_stream call, agent.py ~4895) and z_harm_a (via E2_harm_a's forward-model
gate_stream call, agent.py ~8332). Per the SAME discipline: build a cheap
pre-flight reach probe for the SPECIFIC new pair(s) BEFORE any full
multi-seed recording run, and gate the run on it.

WHAT THIS SCRIPT DOES (and does not)
-------------------------------------
Uses `experiments/_lib/q081_pair_reach_check_stream.py`
(the generalisation of Q081-REACH-CHECK-PAIR-SPECIFIC's
`q081_pair_reach_check.py`, built for this task -- see that module's
docstring for the full mechanism). Scans z_goal first (2 levers x 5 seeds);
if it finds no informative reach, falls back to scanning z_harm_a the same
way. `experiment_purpose="diagnostic"`, no learning, no backprop
(forward-pass rollouts only, matched-arm construction via deepcopy + full RNG
reset -- see q081_pair_reach_check_stream.py for the determinism discipline).

This probe does NOT adjudicate Q-081 Outcome A vs B, and does NOT itself
queue or write any substrate change -- per the claim-synthesis doc's own
escalation note, if BOTH targets come back genuinely zero-reach (not merely
"couldn't test"), the next move is a governance decision about a direct
substrate write path, which is out of scope for a `/queue-experiment` probe.

MANUAL VALIDATION AT AUTHORING TIME (measured 2026-08-01, informs the
settings below -- same discipline q081_pair_reach_check.py's own authoring
used before V3-EXQ-849 queued it)
--------------------------------------------------------------------------
z_goal's activation precondition (`GoalState.is_active()`, requires
`agent.update_z_goal(benefit_exposure=...)` to have fired at least once above
`goal.benefit_threshold`, default 0.1) turned out to be the harder-to-clear
precondition of the two targets -- the OPPOSITE of the claim-synthesis doc's
prediction that z_goal would have "lower precondition surface" than z_harm_a.
A single-agent scan (no scrambler, no second arm -- cheaper than the full
probe) across seeds 0-4 at up to 30 episodes x 300 steps (env_size=6, the
q081_profile_kwargs() + use_anchor_sets + use_per_region_vs config this
module shares with 824a/838/849) found:

  seed 0: never activated within 303 cumulative ticks (short episode
          survival under this untrained agent's own action policy)
  seed 1: never activated within 810 cumulative ticks
  seed 2: activated at cumulative tick 630
  seed 3: activated at cumulative tick 1156
  seed 4: activated at cumulative tick 730

So z_goal activation is real and reachable within an affordable multi-episode
budget for at least some seeds, but needs MANY episode restarts (short
per-episode survival dominates, not a longer per-episode step cap) -- hence
N_EPISODES=30 / STEPS_PER_EPISODE=300 below, rather than the 3x400 budget
V3-EXQ-849 used for its (always-populated) salience-precursor probe. The full
matched-arm probe (`run_pair_specific_stream_reach_probe`, both arms +
LandmarkScrambler) was directly run at target="z_goal", seed=2,
n_episodes=1..15, steps_per_episode=150..400 and stayed degenerate throughout
that smaller range (0 active ticks) -- consistent with the single-agent scan
above, which needed cumulative tick 630 to activate at that same seed.

z_harm_a's activation precondition (`agent._harm_a_prev` non-None, requires
only `use_affective_harm_stream=True` -- already in q081_profile_kwargs() --
plus `use_e2_harm_a=True`, which this script's module sets explicitly for
this target) is populated from the FIRST tick onward, every tick, regardless
of environmental events -- confirmed at seed=2, n_episodes=1,
steps_per_episode=150: `n_active_ticks_intact=150/150`,
`is_degenerate=False`. At that same cheap setting, `has_pair_specific_reach`
was False for ALL THREE manipulation levers tried manually
(iei_permute, jitter, suppress -- suppress being the maximal/lesion lever, a
sensitivity positive control per q081_pair_reach_check.py's own convention).
That result is carried forward as informative context below, not assumed --
this script's own (larger, multi-seed) z_harm_a leg is what actually
adjudicates it.

SLEEP DRIVER: K=never (SleepLoopManager disabled; use_sleep_loop=False,
matches 824/824a/838/849/q081_pair_reach_check(_stream).py -- scope is
waking-stream organisation only).

Design record: REE_assembly/evidence/planning/claim_synthesis_Q-081_2026-08-01.md
section 3-5; REE_assembly/docs/claims/claims.yaml Q-081 "MEASURED-PAIR REFRAME"
note; experiments/_lib/q081_pair_reach_check_stream.py module docstring.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.q081_pair_reach_check_stream import (  # noqa: E402
    run_pair_specific_stream_reach_probe,
)

EXPERIMENT_TYPE = "v3_exq_865_q081_zgoal_reach_preflight_scan"
QUEUE_ID = "V3-EXQ-865"
CLAIM_IDS: List[str] = ["Q-081"]

# ANCHOR REACHABILITY (validate_experiments.py advisory): reachable by
# construction -- both targets' activation preconditions and the boundary-
# event guard were validated manually at (a superset of) these exact settings
# during this script's own authoring (see module docstring above): z_harm_a
# cleared both guards at seed=2/n_episodes=1/steps=150 (a smaller budget than
# this script uses); z_goal's SINGLE-AGENT activation scan (no scrambler
# overhead) confirmed activation is reachable within a 30-episode/300-step
# cumulative budget at seeds 2/3/4. This script's own --dry-run smoke also
# cleared both guards for z_harm_a at tiny scale.
ANCHOR_REACHABILITY_EXEMPT = (
    "reachable by construction -- z_harm_a validated manually at cheaper "
    "settings than this script uses (seed 2, n_episodes=1, steps=150, "
    "q081_pair_reach_check_stream.py's own authoring); z_goal's activation "
    "precondition validated reachable within this script's budget via a "
    "separate single-agent scan (seeds 2/3/4 of 0-4) -- see module docstring"
)
EXPERIMENT_PURPOSE = "diagnostic"

# ------------------------------------------------------------------ config
SEEDS = [0, 1, 2, 3, 4]     # Q-081 standing seed convention (824/824a/838/849)
ENV_SIZE = 6
MIN_BOUNDARY_EVENTS = 1
MIN_ACTIVE_TICKS = 1

# Two levers, matching V3-EXQ-849's own scan: the best-known reach lever from
# the retired pair's history, plus the one genuinely untried donor-free lever
# already built into q081_landmark_removal.MODES.
LEVER_KNOWN = "iei_permute"
LEVER_UNTRIED = "jitter"
LEVERS = (LEVER_KNOWN, LEVER_UNTRIED)

# z_goal: activation is the hard-to-clear precondition (see module docstring)
# -- needs many episode RESTARTS, not a longer per-episode step cap, per the
# manual single-agent scan (short per-episode survival under this untrained
# agent's own policy dominates).
ZGOAL_N_EPISODES = 30
ZGOAL_STEPS_PER_EPISODE = 300

# z_harm_a: active from tick 1 onward once configured -- no restart budget
# needed. Kept modest; this is the fallback leg and only runs if z_goal's
# leg does not establish reach.
ZHARM_A_N_EPISODES = 3
ZHARM_A_STEPS_PER_EPISODE = 200


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_cell(target: str, mode: str, seed: int, n_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    """One (target, mode, seed) reach-probe cell. Never raises -- strict=False
    so a degenerate or no-reach cell is recorded, not aborted; the scan needs
    all cells to compute its aggregate verdict."""
    report = run_pair_specific_stream_reach_probe(
        target=target,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        env_size=ENV_SIZE,
        strict=False,
        min_boundary_events=MIN_BOUNDARY_EVENTS,
        min_active_ticks=MIN_ACTIVE_TICKS,
        mode=mode,
    )
    report["mode"] = mode
    return report


def run_target_leg(
    target: str, n_episodes: int, steps_per_episode: int, seeds: List[int],
) -> Dict[str, Any]:
    """Run the full (lever x seed) grid for one target stream and summarise
    it. Never raises."""
    cells: List[Dict[str, Any]] = []
    for mode in LEVERS:
        for i, seed in enumerate(seeds):
            cell = run_cell(target, mode, seed, n_episodes, steps_per_episode)
            cells.append(cell)
            print(f"  [{EXPERIMENT_TYPE}] target={target} mode={mode} seed={seed} "
                  f"is_degenerate={cell['is_degenerate']} "
                  f"has_reach={cell['has_pair_specific_reach']} "
                  f"n_boundaries={cell['n_boundaries_true_total']} "
                  f"n_active_ticks={cell['n_active_ticks_intact']} "
                  f"({i + 1}/{len(seeds)})", flush=True)
    non_degenerate_cells = [c for c in cells if not c["is_degenerate"]]
    reach_cells = [c for c in non_degenerate_cells if c["has_pair_specific_reach"]]
    return {
        "target": target,
        "n_episodes": n_episodes,
        "steps_per_episode": steps_per_episode,
        "cells": cells,
        "n_non_degenerate": len(non_degenerate_cells),
        "n_reach_cells": len(reach_cells),
        "reach_found_at": sorted({(c["mode"], c["seed"]) for c in reach_cells}),
    }


def adjudicate(zgoal_leg: Dict[str, Any], zharm_a_leg: Dict[str, Any] | None) -> Dict[str, Any]:
    """Combine the z_goal leg (always run) and the z_harm_a fallback leg
    (run only if z_goal did not establish reach) into one verdict.

    Preconditions: per-target non-degeneracy is INFORMATIONAL (a target
    coming back fully degenerate does not fail the whole run -- the other
    target may still be informative), but at least ONE target across the
    two legs actually run must have cleared its non-degeneracy guard for
    this scan to say anything at all -- that combined precondition IS
    load-bearing (all_preconditions_met).
    """
    legs = [zgoal_leg] + ([zharm_a_leg] if zharm_a_leg is not None else [])
    preconditions = []
    for leg in legs:
        preconditions.append({
            "name": f"{leg['target']}_at_least_one_non_degenerate_cell",
            "description": (
                f"For target={leg['target']!r}: at least one (mode, seed) "
                "cell must clear BOTH non-degeneracy guards (boundary events "
                "AND target-stream activation -- see "
                "q081_pair_reach_check_stream.py module docstring) for that "
                "target's reading to be informative rather than vacuous."
            ),
            "measured": float(leg["n_non_degenerate"]),
            "threshold": 1.0,
            "direction": "lower",
            "met": bool(leg["n_non_degenerate"] >= 1),
        })
    total_non_degenerate = sum(leg["n_non_degenerate"] for leg in legs)
    preconditions.append({
        "name": "at_least_one_target_has_a_non_degenerate_cell",
        "description": (
            "Combined across whichever target leg(s) actually ran: if BOTH "
            "z_goal and z_harm_a came back fully degenerate (activation "
            "precondition never cleared, or no boundary events fired), this "
            "scan obtained no informative reading at all and must self-route "
            "substrate_not_ready_requeue rather than any reach verdict."
        ),
        "measured": float(total_non_degenerate),
        "threshold": 1.0,
        "direction": "lower",
        "met": bool(total_non_degenerate >= 1),
    })
    all_preconditions_met = preconditions[-1]["met"]

    criteria = [{
        "name": "C1_any_target_shows_precursor_reach",
        "load_bearing": True,
        "passed": bool(zgoal_leg["n_reach_cells"] > 0
                        or (zharm_a_leg is not None and zharm_a_leg["n_reach_cells"] > 0)),
        "measured": float(zgoal_leg["n_reach_cells"]
                           + (zharm_a_leg["n_reach_cells"] if zharm_a_leg is not None else 0)),
        "threshold": 1.0,
        "note": (
            "Count of non-degenerate cells (across whichever target leg(s) "
            "ran) where at least one raw or gated dimension diverged between "
            "INTACT and the manipulated arm. >=1 -> a full multi-seed "
            "recording run on RV(z_world, <that target>) is now justified; "
            "0 (across all non-degenerate cells of both legs run) -> neither "
            "target shows precursor-level reach from this lever family."
        ),
    }]
    criteria_non_degenerate = {"C1_any_target_shows_precursor_reach": bool(all_preconditions_met)}

    if not all_preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            "Every (target, mode, seed) cell run this scan came back "
            "degenerate -- neither z_goal's activation precondition nor "
            "z_harm_a's (nor, for z_goal, the shared boundary-event guard) "
            "cleared at any cell. Re-queue with a larger episode/step budget "
            "or a wider seed set; do not read this as evidence against reach "
            "for either target."
        )
        summary = (
            f"Q-081 z_goal/z_harm_a reach pre-flight SCAN inconclusive: all "
            f"cells across the leg(s) run were degenerate. See "
            "interpretation.preconditions."
        )
    elif zgoal_leg["n_reach_cells"] > 0:
        label = "q081_zgoal_reach_preflight_lever_found"
        outcome = "PASS"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"z_goal shows precursor-level reach: {zgoal_leg['n_reach_cells']} "
            f"of {zgoal_leg['n_non_degenerate']} non-degenerate cell(s) -- "
            f"(mode, seed) = {zgoal_leg['reach_found_at']}. Do NOT conclude "
            "this settles Q-081 Outcome A/B (this probe does not adjudicate "
            "the claim); it justifies a full multi-seed recording run on "
            "RV(z_world, z_goal) using the lever(s) that showed reach. "
            "z_harm_a fallback leg was not needed and was not run."
        )
    elif zharm_a_leg is not None and zharm_a_leg["n_reach_cells"] > 0:
        label = "q081_zharm_a_fallback_reach_preflight_lever_found"
        outcome = "PASS"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"z_goal showed no precursor-level reach ({zgoal_leg['n_non_degenerate']} "
            "non-degenerate cell(s), 0 reach) -- fell back to z_harm_a per "
            "the claim-synthesis routing. z_harm_a DOES show reach: "
            f"{zharm_a_leg['n_reach_cells']} of {zharm_a_leg['n_non_degenerate']} "
            f"non-degenerate cell(s) -- (mode, seed) = "
            f"{zharm_a_leg['reach_found_at']}. Do NOT conclude this settles "
            "Q-081 Outcome A/B; it justifies a full multi-seed recording run "
            "on RV(z_world, z_harm_a) using the lever(s) that showed reach."
        )
    else:
        label = "q081_zgoal_and_zharm_a_no_lever_found"
        outcome = "PASS"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = None
        zg_note = (
            f"{zgoal_leg['n_non_degenerate']} non-degenerate cell(s), 0 reach"
            if zgoal_leg["n_non_degenerate"] > 0
            else "0 non-degenerate cells (activation precondition never cleared)"
        )
        zh_note = (
            "not run (z_goal leg was fully degenerate, so the fallback leg "
            "was skipped)" if zharm_a_leg is None
            else (
                f"{zharm_a_leg['n_non_degenerate']} non-degenerate cell(s), 0 reach"
                if zharm_a_leg["n_non_degenerate"] > 0
                else "0 non-degenerate cells"
            )
        )
        summary = (
            f"Neither target shows precursor-level reach from this lever "
            f"family. z_goal: {zg_note}. z_harm_a: {zh_note}. Combined with "
            "824/824a/838/849's exhaustive null on the retired "
            "(z_world, operating_mode) pair, this is now evidence that this "
            "specific landmark-manipulation lever family does not reach ANY "
            "of the three cross-stream targets tried so far at the precursor "
            "level. Per the claim-synthesis doc's own escalation note: do "
            "NOT queue another config-lever probe on this lever family for "
            "Q-081 -- the next move (a direct write-path substrate build, or "
            "a different manipulation lever entirely) is a governance "
            "decision, out of scope for this probe."
        )

    return {
        "label": label, "outcome": outcome, "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "summary": summary, "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
    }


def main(dry_run: bool = False) -> Any:
    global ZGOAL_N_EPISODES, ZGOAL_STEPS_PER_EPISODE
    global ZHARM_A_N_EPISODES, ZHARM_A_STEPS_PER_EPISODE
    seeds = [SEEDS[0]] if dry_run else SEEDS
    if dry_run:
        # Deliberately tiny -- the real scan's budget (validated at authoring
        # time, see module docstring) is far larger; the smoke only needs to
        # confirm the pipeline runs end to end without crashing.
        ZGOAL_N_EPISODES, ZGOAL_STEPS_PER_EPISODE = 1, 20
        ZHARM_A_N_EPISODES, ZHARM_A_STEPS_PER_EPISODE = 1, 20

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds} levers={LEVERS} "
          f"zgoal_n_episodes={ZGOAL_N_EPISODES} zgoal_steps={ZGOAL_STEPS_PER_EPISODE} "
          f"zharm_a_n_episodes={ZHARM_A_N_EPISODES} zharm_a_steps={ZHARM_A_STEPS_PER_EPISODE}",
          flush=True)
    t0 = time.time()

    zgoal_leg = run_target_leg("z_goal", ZGOAL_N_EPISODES, ZGOAL_STEPS_PER_EPISODE, seeds)

    zharm_a_leg = None
    if zgoal_leg["n_reach_cells"] == 0:
        print(f"[{EXPERIMENT_TYPE}] z_goal leg found no reach "
              f"(n_non_degenerate={zgoal_leg['n_non_degenerate']}) -- "
              "falling back to z_harm_a leg", flush=True)
        zharm_a_leg = run_target_leg(
            "z_harm_a", ZHARM_A_N_EPISODES, ZHARM_A_STEPS_PER_EPISODE, seeds,
        )
    else:
        print(f"[{EXPERIMENT_TYPE}] z_goal leg found reach -- "
              "z_harm_a fallback leg not needed, skipping", flush=True)

    verdict = adjudicate(zgoal_leg, zharm_a_leg)
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] label={verdict['label']} outcome={verdict['outcome']} "
          f"elapsed={elapsed:.1f}s", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": verdict["outcome"],
        "evidence_direction": verdict["evidence_direction"],
        "non_degenerate": verdict["non_degenerate"],
        "degeneracy_reason": verdict["degeneracy_reason"],
        "interpretation": {
            "label": verdict["label"],
            "summary": verdict["summary"],
            "preconditions": verdict["preconditions"],
            "criteria_non_degenerate": verdict["criteria_non_degenerate"],
            "does_not_adjudicate_note": (
                "This probe answers only whether a lever gives the "
                "manipulation precursor-level reach to z_goal and/or "
                "z_harm_a. It does not adjudicate Q-081 Outcome A vs B."
            ),
        },
        "criteria": verdict["criteria"],
        "elapsed_seconds": elapsed,
        "levers_tested": list(LEVERS),
        "zgoal_leg": zgoal_leg,
        "zharm_a_leg": zharm_a_leg,
        "primary_pair": ["z_world", "z_goal"],
        "fallback_pair": ["z_world", "z_harm_a"],
        "retired_pair": ["z_world", "operating_mode"],
        "prior_runs": {
            "V3-EXQ-824": "measurement_test_design_defect -- retired pair, use_invalidation_trigger alone, no reach",
            "V3-EXQ-824a": "measurement_test_design_defect -- retired pair, use_anchor_sets, reach-check MET, RV bit-identical",
            "V3-EXQ-838": "measurement_test_design_defect -- retired pair, + use_per_region_vs, 2 manipulation families, RV bit-identical",
            "V3-EXQ-849": "PASS (diagnostic, non_contributory) -- retired pair, precursor-level reach scan, 2 levers, zero reach -> reframe recommendation",
        },
        "reframe_source": "claim_synthesis_Q-081_2026-08-01.md",
        "hypothesis_registry_qid": "q081-cross-stream-shared-organisation",
    }

    out_path = write_flat_manifest(
        manifest, dry_run=False,
        config={
            "seeds": seeds, "env_size": ENV_SIZE, "levers": list(LEVERS),
            "min_boundary_events": MIN_BOUNDARY_EVENTS,
            "min_active_ticks": MIN_ACTIVE_TICKS,
            "zgoal_n_episodes": ZGOAL_N_EPISODES,
            "zgoal_steps_per_episode": ZGOAL_STEPS_PER_EPISODE,
            "zharm_a_n_episodes": ZHARM_A_N_EPISODES,
            "zharm_a_steps_per_episode": ZHARM_A_STEPS_PER_EPISODE,
            "primary_pair": ["z_world", "z_goal"],
            "fallback_pair": ["z_world", "z_harm_a"],
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return verdict["outcome"], out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V3-EXQ-865 Q-081 z_goal/z_harm_a reach pre-flight scan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        emit_outcome(outcome="PASS", manifest_path=None, dry_run=True)
        sys.exit(0)
    outcome, out_path = result
    emit_outcome(
        outcome=outcome, manifest_path=out_path, run_id=None, queue_id=QUEUE_ID,
        dry_run=bool(args.dry_run),
    )
