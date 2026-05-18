#!/usr/bin/env python3
"""
V3-EXQ-455a -- SD-032a salience coordinator behavioural re-run with
V_s runtime flags enabled.

Purpose (evidence)
------------------
Clone of V3-EXQ-455 with the MECH-269 / MECH-287 / MECH-288 V_s
invalidation runtime ENABLED. The original EXQ-455 ran before the
cascade gate V3-EXQ-476 could confirm that V_s flags break the
background monostrategy lock. With the gate in place, EXQ-455a
re-measures the SD-032a salience-coordinator behavioural effect under
a regime where the baseline is actually diverse, not entropy-zero.

Arms
----
    COORD_OFF: use_salience_coordinator=False,
               V_s runtime flags ON.
    COORD_ON:  use_salience_coordinator=True,
               salience_apply_to_dacc_bias=True,
               V_s runtime flags ON.

V_s runtime flags on both arms:
    use_per_stream_vs=True
    use_per_region_vs=True
    use_event_segmenter=True
    use_invalidation_trigger=True
    use_anchor_sets=True

Other shared config matches EXQ-455 (3 seeds, 2 conditions, same
episode / step counts, use_dacc=True on both arms so the coordinator
has a bundle to consume).

Pass criteria
-------------
    C1: action_class_entropy >= 0.5 in COORD_ON arm in >= 2/3 seeds.
    C2: entropy_COORD_ON > entropy_COORD_OFF in >= 2/3 seeds.
    PASS = C1 AND C2.

Note on C2 relative to V3-EXQ-476:
    EXQ-476 establishes that V_s flags alone raise entropy above OFF
    baseline. C2 here asks the stronger question: does the salience
    coordinator ADD coherent policy-level structure on top of V_s
    diversity, or is V_s sufficient on its own? C2 PASS would mean the
    coordinator is doing real work; C2 FAIL with C1 PASS would suggest
    V_s is carrying the diversity and the coordinator is a no-op at
    this stage of the substrate stack.

Tags / claims
-------------
    SD-032a, MECH-259, MECH-261.

Supersedes
----------
    V3-EXQ-455.

Depends on
----------
    V3-EXQ-476 (cascade gate). Do NOT run until V3-EXQ-476 PASSes.

Status
------
    PLANNING STUB. Full implementation deferred pending V3-EXQ-476
    PASS and the MECH-284 Phase 3 consumer landing (required for V_s
    flags to actually modulate action selection).

Full-implementation TODO
------------------------
    - Clone v3_exq_455_sd032a_salience_behavioural.py harness (when
      it lands, or its equivalent -- may need to be written from
      scratch if EXQ-455 was script-level stub only).
    - Add V_s runtime flags to both REEConfig.from_dims() calls.
    - Run 3 seeds x 2 conditions x N episodes; log
      action_class_entropy per arm per seed.
    - Write flat JSON manifest to evidence/experiments/.
    - Print "Result written to: <path>" (validator requirement).
"""

from __future__ import annotations

import argparse
import sys


def _print_plan() -> None:
    print("V3-EXQ-455a -- SD-032a salience coordinator, V_s-enabled re-run", flush=True)
    print("Arms: COORD_OFF vs COORD_ON, both with V_s flags on and dACC on", flush=True)
    print("V_s flags: per_stream / per_region / event_segmenter /", flush=True)
    print("           invalidation_trigger / anchor_sets", flush=True)
    print("C1 action_class_entropy >= 0.5 in COORD_ON arm (>=2/3 seeds)", flush=True)
    print("C2 entropy_COORD_ON > entropy_COORD_OFF (>=2/3 seeds)", flush=True)
    print("Supersedes V3-EXQ-455. Depends on V3-EXQ-476.", flush=True)
    print("experiment_purpose=evidence", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3-EXQ-455a SD-032a salience + V_s re-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    raise NotImplementedError(
        "V3-EXQ-455a gated. MECH-284 Phase 3 substrate landed 2026-04-24, "
        "but cascade gate V3-EXQ-476a/476b ran FAIL (V_s wired-but-inert: "
        "anchor resets fire but action selection unchanged -- catatonic-"
        "lock at policy layer). Remaining blocker is SD-037 (broadcast "
        "override regulator, orexin-analog) -- next /implement-substrate "
        "target per substrate_queue.json. Do not run until SD-037 has "
        "landed and a new cascade gate has confirmed V_s flags break the "
        "baseline monostrategy lock."
    )


if __name__ == "__main__":
    sys.exit(main())
