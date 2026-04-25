#!/usr/bin/env python3
"""
V3-EXQ-476 -- MECH-269 V_s validation entropy probe (cascade gate).

Purpose (diagnostic)
--------------------
Gate for the V_s-enabled validation cascade (V3-EXQ-445d / V3-EXQ-449c /
V3-EXQ-455a). The agent currently shows a monostrategy lock
(action_class_entropy = 0.0 across all seed/condition cells). MECH-269
Phase 1 + Phase 2 and MECH-287 / MECH-288 give us per-region / per-stream
V_s signals as observables, but the Phase 3 consumer chain -- the
MECH-284 staleness accumulator that VARIES ACTION SELECTION based on V_s
-- is NOT yet landed. So naively enabling the V_s flags on a baseline
agent may or may not produce non-zero action entropy.

This probe runs the simplest possible 2-arm comparison to answer that:

Arms
----
    OFF:  baseline agent, all V_s runtime flags OFF
    ON:   baseline agent, V_s runtime flags ON:
            use_per_stream_vs=True
            use_per_region_vs=True
            use_event_segmenter=True
            use_invalidation_trigger=True
            use_anchor_sets=True

Metric
------
    action_class_entropy per arm, averaged over 2 seeds.

Pass / fail rule
----------------
    PASS = ON entropy > OFF entropy by >= 0.1 in >= 2/2 seeds
         => cascade may proceed; V3-EXQ-445d / 449c / 455a are unblocked.

    FAIL / INCONCLUSIVE = ON entropy does not clear OFF entropy by >= 0.1
         => MECH-284 Phase 3 consumer (staleness accumulator that feeds
            back into action selection) must land before any of the
            downstream cascade can be run. V_s flags alone do not break
            the monostrategy lock.

Supersedes
----------
    None (new gate).

Depends on
----------
    Nothing (this IS the gate).

Status
------
    This file is a PLANNING STUB. Full implementation deferred pending
    the MECH-284 Phase 3 landing and the runtime-flag plumbing in
    REEConfig.from_dims() for use_per_region_vs / use_invalidation_trigger
    / use_anchor_sets / use_event_segmenter. The --dry-run path prints
    the plan and exits 0 so validate_queue.py can see the script exists
    with valid entry points.

Full-implementation TODO
------------------------
    - Build 2 REEConfig.from_dims() configs (OFF and ON arm) using the
      flags listed above.
    - Construct CausalGridWorldV2 with the same seed / hazard density
      settings that EXQ-445a uses.
    - Run 2 seeds x 2 arms x N episodes (~60 eps at 200 steps).
    - Accumulate action_class_entropy per arm per seed.
    - Write flat JSON manifest to evidence/experiments/<dir>/record.json
      with run_id ending _v3 and architecture_epoch
      "ree_hybrid_guardrails_v1".
    - Print "Result written to: <path>" (required by validate_queue.py
      RE_SAVED_TO_IN_SCRIPT).
"""

from __future__ import annotations

import argparse
import sys


def _print_plan() -> None:
    print("V3-EXQ-476 -- MECH-269 V_s validation entropy probe (cascade gate)", flush=True)
    print("Arms: OFF (all V_s flags off) vs ON (per-stream / per-region / anchor /", flush=True)
    print("      event-segmenter / invalidation-trigger all on)", flush=True)
    print("Metric: action_class_entropy per arm per seed (2 seeds x 2 arms)", flush=True)
    print("PASS = ON - OFF >= 0.1 in >=2/2 seeds -> unblock cascade", flush=True)
    print("FAIL / INCONCLUSIVE -> MECH-284 Phase 3 consumer must land first", flush=True)
    print("experiment_purpose=diagnostic", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3-EXQ-476 MECH-269 V_s entropy probe")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    raise NotImplementedError(
        "V3-EXQ-476 superseded by V3-EXQ-476a/476b (both FAIL: V_s flags "
        "wired-but-inert -- accumulator integrates, MECH-269 anchor resets "
        "fire, but action_class_entropy unchanged; catatonic-lock at policy "
        "layer). MECH-284 Phase 3 substrate landed 2026-04-24. The remaining "
        "blocker is SD-037 (broadcast override regulator, orexin-analog) -- "
        "next /implement-substrate target per substrate_queue.json. Do not "
        "re-run this stub; await SD-037 substrate then re-queue a new EXQ."
    )


if __name__ == "__main__":
    sys.exit(main())
