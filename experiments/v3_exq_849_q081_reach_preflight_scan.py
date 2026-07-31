#!/opt/local/bin/python3
"""V3-EXQ-849 -- Q-081 cheap reach pre-flight SCAN: does ANY available lever
give the cross-stream recording manipulation reach to operating_mode, or does
Q-081's measured pair need to be reframed?

Follow-on of failure_autopsy_v3-exq-838_2026-07-29 (REE_assembly 4cff1b73e8),
routed via GOV-FANOUT-1 (queue-experiment SKILL.md sec 2.5b). 838 is the THIRD
consecutive bit-identical RV(z_world, operating_mode) result (824, 824a, 838)
and refuted the 824a-recommended fix (use_per_region_vs=True). The autopsy's
fanout_recommendation registers two live hypotheses on
`q081-cross-stream-shared-organisation` (hypothesis_space_registry.v1.json):

  H1 (axis=substrate): operating_mode CAN be made reachable via a mechanism
     not yet tried (use_per_region_vs already refuted).
  H2 (axis=measurement): operating_mode is not downstream of these
     manipulations at all; RV(z_world, operating_mode) is the wrong DV and
     Q-081 must reframe onto a confirmed-reachable pair.

The autopsy explicitly refuses "another blind full re-run of the same pair"
and routes to a CHEAP pre-flight BEFORE any full multi-seed run. Per
GOV-FANOUT-1, a single reach-scan discriminates H1 from H2 (a lever that
moves operating_mode -> H1 build; none does -> H2 reframe), so this is one
script, not two full legs.

WHAT THIS SCRIPT DOES (and does not)
-------------------------------------
Uses the substrate built for exactly this purpose --
`experiments/_lib/q081_pair_reach_check.py` (Q081-REACH-CHECK-PAIR-SPECIFIC,
substrate_queue.json, IMPLEMENTED 2026-07-31) -- which inspects the
PRECURSOR of operating_mode (`agent.salience._input_signals`, the closed set
of named values `SalienceCoordinator.tick()` computes the softmax from)
rather than waiting for an RV statistic on the softmax OUTPUT across a full
20-warmup + 20-recording x 5-seed x 4-arm pipeline (which is what 838 did, at
~83 minutes, to re-establish a bit-identical result). operating_mode is a
DETERMINISTIC function of exactly those named signals plus static config: if
none of them ever diverges between INTACT and a manipulated arm, operating_mode
provably cannot differ either -- no further computation needed. This makes the
precursor check strictly cheaper AND, for a null result, strictly stronger
than the RV-level check (it rules out reach at the wiring level, not just in
one particular downstream statistic).

That module's own manual validation (recorded in substrate_queue.json,
implemented_utc 2026-07-31) already re-ran the SAME lever 838 used
(use_anchor_sets + use_per_region_vs, mode="iei_permute") at the precursor
level and reproduced the known null (has_pair_specific_reach=False at 2 clean
seeds). This script does two things that validation did NOT:

  1. Formalises that replication as a governance-visible, queued, recorded
     diagnostic (the manual run is dev-validation, not evidence; MECH-450-
     style "reviewed != discussed" discipline requires an actual manifest).
  2. Extends the SCAN to a genuinely UNTRIED lever: mode="jitter"
     (`q081_landmark_removal.MODES`). jitter has never been run against this
     pair at ANY level -- 824/824a/838 covered off/iei_permute/circular_shift/
     suppress only. jitter is a causal, donor-free, per-event delay/smear
     (distinct mechanism from the yoked permutation/shift modes): if the
     wiring bottleneck is specific to how iei_permute/circular_shift disturb
     the boundary train, jitter could show reach where they did not.

This probe does NOT adjudicate Q-081 Outcome A vs B (per
q081_pair_reach_check.py's own docstring: "THIS PROBE DOES NOT ADJUDICATE
Q-081"). It answers the narrower, PRIOR question the autopsy asked for: would
a full run on ANY of the reach levers tried here have even a chance of
producing a non-bit-identical result. `experiment_purpose="diagnostic"`, no
learning, no backprop (forward-pass rollouts only, matched-arm construction
via deepcopy + full RNG reset -- see q081_pair_reach_check.py for the
determinism discipline this reuses verbatim).

WHAT A non_contributory / "no lever found" RESULT WOULD AND WOULD NOT MEAN
(GOV-FANOUT-1 null declaration)
----------------------------------------------------------------------------
WOULD mean: neither of the two levers tested here (the best-known 838 lever,
re-verified cheaply; and jitter, the one untried donor-free lever already
built into the library) shows precursor-level reach on this exact pair. That
is real, cumulative evidence toward H2 -- three structurally different
manipulation families (yoked permutation, circular shift, lesion-suppress,
now also jitter) plus two reach-consumer flags have all failed to move any
named salience-signal precursor of operating_mode.
WOULD NOT mean: that NO lever anywhere could ever reach operating_mode (H1 is
not falsified outright -- "direct operating_mode write path", the autopsy's
other H1 sketch item, is a substrate BUILD, not a config-level lever, and is
out of scope for a cheap probe); nor does it mean Q-081 itself is settled --
only that this specific DV pair needs reframing (H2), which is the autopsy's
own routing target.
A reach FOUND on either lever would mean H1 is live again and a full
multi-seed recording run using that lever is now justified -- route to
`/implement-substrate` only if the reach needs a NEW capability, otherwise
straight to a lettered `/queue-experiment` full run citing this manifest.

SLEEP DRIVER: K=never (SleepLoopManager disabled; use_sleep_loop=False,
matches 824/824a/838/q081_pair_reach_check.py -- scope is waking-stream
organisation only).

Design record: REE_assembly/evidence/planning/failure_autopsy_v3-exq-838_2026-07-29.md
section 7 (routing) and its `fanout_recommendation`; hypothesis registry entry
`q081-cross-stream-shared-organisation` in
REE_assembly/evidence/planning/hypothesis_space_registry.v1.json.
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
from experiments._lib.q081_pair_reach_check import (  # noqa: E402
    SALIENCE_SIGNAL_NAMES,
    run_pair_specific_reach_probe,
)

EXPERIMENT_TYPE = "v3_exq_849_q081_reach_preflight_scan"
QUEUE_ID = "V3-EXQ-849"
CLAIM_IDS: List[str] = ["Q-081"]

# ANCHOR REACHABILITY (validate_experiments.py advisory): the
# `at_least_one_non_degenerate_cell` precondition is reachable by construction,
# not a hand-narrowed gate -- q081_pair_reach_check.py's own manual validation
# (substrate_queue.json Q081-REACH-CHECK-PAIR-SPECIFIC, implemented_utc
# 2026-07-31) already cleared it at seeds 0/1 with these exact n_episodes=3/
# steps_per_episode=400/env_size=6 defaults, and this script's own --dry-run
# smoke (seed 0, mode="jitter", tiny scale) also cleared it. With 5 seeds x 2
# levers = 10 cells at the validated non-degenerate scale, at least one
# non-degenerate cell is the expected, not the marginal, outcome.
ANCHOR_REACHABILITY_EXEMPT = (
    "reachable by construction -- validated manually at these exact settings "
    "during q081_pair_reach_check.py's own authoring (seeds 0/1 non-degenerate) "
    "and reconfirmed by this script's own --dry-run smoke"
)
EXPERIMENT_PURPOSE = "diagnostic"

# ------------------------------------------------------------------ config
SEEDS = [0, 1, 2, 3, 4]     # Q-081 standing seed convention (824/824a/838)
N_EPISODES = 3              # matches q081_pair_reach_check's own validated,
STEPS_PER_EPISODE = 400     # non-degenerate defaults (manual validation:
ENV_SIZE = 6                # seeds 0/1 cleared the guard at these settings)
MIN_BOUNDARY_EVENTS = 1

# Two levers. "iei_permute" replicates 838's own best-known (already-refuted
# at the RV level) lever at the cheaper precursor level. "jitter" is the one
# genuinely untried lever already present in q081_landmark_removal.MODES.
LEVER_KNOWN = "iei_permute"
LEVER_UNTRIED = "jitter"
LEVERS = (LEVER_KNOWN, LEVER_UNTRIED)


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_cell(mode: str, seed: int) -> Dict[str, Any]:
    """One (mode, seed) reach-probe cell. Never raises -- strict=False so a
    degenerate or no-reach cell is recorded, not aborted; the scan needs all
    cells to compute its aggregate verdict."""
    report = run_pair_specific_reach_probe(
        n_episodes=N_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        seed=seed,
        env_size=ENV_SIZE,
        strict=False,
        min_boundary_events=MIN_BOUNDARY_EVENTS,
        mode=mode,
    )
    report["mode"] = mode
    return report


def adjudicate(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    non_degenerate_cells = [c for c in cells if not c["is_degenerate"]]
    n_non_degenerate = len(non_degenerate_cells)
    reach_cells = [c for c in non_degenerate_cells if c["has_pair_specific_reach"]]

    preconditions = [
        {
            "name": "at_least_one_non_degenerate_cell",
            "description": (
                "The untrained event segmenter fires boundary events sparsely "
                "with high seed-to-seed variance; a cell with zero true "
                "boundary events gives its manipulation nothing to scramble "
                "and its 'no reach' reading is vacuous, not informative "
                "(q081_pair_reach_check.py's own non-degeneracy guard). At "
                "least one of the "
                f"{len(cells)} (mode x seed) cells run here must clear that "
                "guard for this scan to say anything at all."
            ),
            "measured": float(n_non_degenerate),
            "threshold": 1.0,
            "direction": "lower",
            "control": (
                "q081_pair_reach_check.py's own manual validation (substrate_"
                "queue.json Q081-REACH-CHECK-PAIR-SPECIFIC, implemented_utc "
                "2026-07-31) cleared this guard at seeds 0/1 with these exact "
                "n_episodes/steps_per_episode/env_size defaults."
            ),
            "met": bool(n_non_degenerate >= 1),
        },
    ]
    all_preconditions_met = all(p["met"] for p in preconditions)

    criteria = [
        {
            "name": "C1_any_lever_shows_precursor_reach",
            "load_bearing": True,
            "passed": bool(len(reach_cells) > 0),
            "measured": float(len(reach_cells)),
            "threshold": 1.0,
            "note": (
                "Count of non-degenerate cells where at least one named "
                "salience input signal diverged between INTACT and the "
                "manipulated arm at any compared tick. >=1 -> H1 (build "
                "reach) is live again; 0 (across all non-degenerate cells) "
                "-> H2 (reframe the measured pair) is favoured."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C1_any_lever_shows_precursor_reach": bool(all_preconditions_met),
    }

    if not all_preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        degeneracy_reason = (
            "every (mode, seed) cell came back degenerate (< "
            f"{MIN_BOUNDARY_EVENTS} true boundary event(s)) -- no cell gave "
            "either manipulation lever anything to scramble, so this scan "
            "obtained no informative reach reading at all. Re-queue with a "
            "larger steps_per_episode/n_episodes or a wider seed set; do not "
            "read this as evidence for H2."
        )
        summary = (
            "Q-081 reach pre-flight SCAN inconclusive: all "
            f"{len(cells)} (mode x seed) cells were degenerate. See "
            "interpretation.preconditions."
        )
    elif reach_cells:
        label = "q081_reach_preflight_lever_found"
        outcome = "PASS"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = None
        found = sorted({(c["mode"], c["seed"]) for c in reach_cells})
        summary = (
            f"H1 (build reach) is LIVE: {len(reach_cells)} of "
            f"{n_non_degenerate} non-degenerate cell(s) show precursor-level "
            f"reach to operating_mode -- (mode, seed) = {found}. Do NOT "
            "conclude this settles Q-081 Outcome A/B (this probe does not "
            "adjudicate the claim); it justifies a full multi-seed recording "
            "run using the lever(s) that showed reach, or /implement-"
            "substrate if the reach needs a capability not yet present."
        )
    else:
        label = "q081_reach_preflight_no_lever_found"
        outcome = "PASS"
        evidence_direction = "non_contributory"
        non_degenerate = True
        degeneracy_reason = None
        summary = (
            f"H2 (reframe pair) is favoured: {n_non_degenerate} non-"
            f"degenerate cell(s) across levers {list(LEVERS)} show NO "
            "precursor-level reach to operating_mode -- no named salience "
            "input signal ever diverged between INTACT and either "
            "manipulated arm. Combined with 824/824a/838's RV-level nulls "
            "across off/iei_permute/circular_shift/suppress, this is now "
            "four structurally different manipulation families (plus the "
            "two REACH_CONSUMERS flags) with zero reach to this pair. Does "
            "NOT prove no lever anywhere could reach operating_mode -- a "
            "direct operating_mode write path (a substrate build, not a "
            "config lever) remains untested and out of scope here. "
            "Recommend: reframe Q-081's measured pair onto a confirmed-"
            "reachable one (H2) rather than another full run on "
            "(z_world, operating_mode)."
        )

    return {
        "label": label, "outcome": outcome, "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "summary": summary, "preconditions": preconditions, "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "n_non_degenerate": n_non_degenerate, "n_reach_cells": len(reach_cells),
    }


def main(dry_run: bool = False) -> Any:
    global N_EPISODES, STEPS_PER_EPISODE
    seeds = [SEEDS[0]] if dry_run else SEEDS
    if dry_run:
        N_EPISODES, STEPS_PER_EPISODE = 2, 15

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds} levers={LEVERS} "
          f"n_episodes={N_EPISODES} steps_per_episode={STEPS_PER_EPISODE}", flush=True)
    t0 = time.time()

    cells: List[Dict[str, Any]] = []
    for mode in LEVERS:
        for i, seed in enumerate(seeds):
            cell = run_cell(mode, seed)
            cells.append(cell)
            print(f"  [{EXPERIMENT_TYPE}] mode={mode} seed={seed} "
                  f"is_degenerate={cell['is_degenerate']} "
                  f"has_reach={cell['has_pair_specific_reach']} "
                  f"n_boundaries={cell['n_boundaries_true_total']} "
                  f"({i + 1}/{len(seeds)})", flush=True)

    verdict = adjudicate(cells)
    elapsed = time.time() - t0
    print(f"[{EXPERIMENT_TYPE}] label={verdict['label']} outcome={verdict['outcome']} "
          f"n_non_degenerate={verdict['n_non_degenerate']} "
          f"n_reach_cells={verdict['n_reach_cells']} elapsed={elapsed:.1f}s", flush=True)

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
                "manipulation precursor-level reach to operating_mode. It "
                "does not adjudicate Q-081 Outcome A vs B (per "
                "q081_pair_reach_check.py's own docstring)."
            ),
        },
        "criteria": verdict["criteria"],
        "elapsed_seconds": elapsed,
        "levers_tested": list(LEVERS),
        "salience_signal_names_checked": sorted(SALIENCE_SIGNAL_NAMES),
        "reach_scan_results": cells,
        "primary_pair": ["z_world", "operating_mode"],
        "prior_runs": {
            "V3-EXQ-824": "measurement_test_design_defect -- use_invalidation_trigger alone, no reach",
            "V3-EXQ-824a": "measurement_test_design_defect -- use_anchor_sets, reach-check MET, RV bit-identical",
            "V3-EXQ-838": "measurement_test_design_defect -- + use_per_region_vs, 2 manipulation families, RV bit-identical",
        },
        "fanout_source": "failure_autopsy_v3-exq-838_2026-07-29.json",
        "hypothesis_registry_qid": "q081-cross-stream-shared-organisation",
    }

    out_path = write_flat_manifest(
        manifest, dry_run=False,
        config={
            "seeds": seeds, "env_size": ENV_SIZE, "n_episodes": N_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE, "levers": list(LEVERS),
            "min_boundary_events": MIN_BOUNDARY_EVENTS,
            "primary_pair": ["z_world", "operating_mode"],
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return verdict["outcome"], out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="V3-EXQ-849 Q-081 cheap reach pre-flight scan (H1/H2 discriminator)")
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
