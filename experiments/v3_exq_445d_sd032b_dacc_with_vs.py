#!/usr/bin/env python3
"""
V3-EXQ-445d -- SD-032b dACC full pipeline, V_s-enabled re-run with
OFF-arm entropy precondition check.

Purpose (evidence)
------------------
Clone of V3-EXQ-445a with the full MECH-269 / MECH-287 / MECH-288 V_s
invalidation runtime ENABLED on top of the dACC+E2_harm_a pipeline.
Supersedes V3-EXQ-445c (which could not distinguish dACC effect from
background monostrategy lock).

Arms
----
    OFF:  dACC OFF (baseline E3 policy, no score_bias),
          V_s runtime flags ON
    ON_INDEPENDENT:
          use_dacc=True, use_e2_harm_a=True, use_shared_harm_trunk=False
          (ARC-033 independent per-stream forward models),
          V_s runtime flags ON

V_s runtime flags enabled on BOTH arms:
    use_per_stream_vs=True
    use_per_region_vs=True
    use_event_segmenter=True
    use_invalidation_trigger=True
    use_anchor_sets=True

Other shared config matches EXQ-445a (phased training P0 / P1 / P2,
~280 episodes per arm, 120 steps/ep, 3 seeds).

OFF-entropy precondition
------------------------
    Before scoring C1 / C2 / C3, check OFF-arm action_class_entropy:
        if OFF-arm entropy == 0.0 (or < 0.05) in >= 2/3 seeds:
            manifest records evidence_direction = "inconclusive"
            with evidence_direction_note = "V_s flags failed to break
            monostrategy lock; dACC effect indistinguishable from
            background."
            Exit PASS/FAIL scoring. The cascade gate V3-EXQ-476 should
            have caught this, but EXQ-445d re-checks because dACC could
            have secondary monostrategy effects.

Pass criteria (only evaluated if OFF-entropy precondition passes)
-----------------------------------------------------------------
    C1: harm_a_forward_r2 >= 0.3 in >= 2/3 seeds (E2_harm_a trained)
    C2: | entropy_ON - entropy_OFF | >= 0.1 in >= 2/3 seeds
        (dACC produces a distinguishable shift)
    C3: entropy_ON >= entropy_OFF in >= 2/3 seeds
        (dACC either preserves or increases diversity, never collapses)
    PASS = C1 AND C2 AND C3.

Supersedes
----------
    V3-EXQ-445c

Depends on
----------
    V3-EXQ-476 (cascade gate). Do NOT run until V3-EXQ-476 PASSes.

Status
------
    PLANNING STUB. Full implementation deferred pending
    (a) V3-EXQ-476 PASS, and
    (b) MECH-284 Phase 3 consumer landed so V_s actually modulates
        action selection.

Full-implementation TODO
------------------------
    - Clone v3_exq_445a_sd032b_dacc_full_pipeline.py harness
      (phased training loop P0 encoder warmup -> P1 frozen-encoder
      E2_harm_a training -> P2 pure-policy eval).
    - Add V_s runtime flags to both REEConfig.from_dims() calls
      (OFF arm and ON_INDEPENDENT arm).
    - After eval loop, compute OFF-arm action_class_entropy. If below
      0.05 in >=2/3 seeds, write manifest with
      evidence_direction="inconclusive" and exit without PASS/FAIL
      adjudication.
    - Otherwise compute C1/C2/C3 against the EXQ-445a thresholds.
    - Write flat JSON manifest to evidence/experiments/.
    - Print "Result written to: <path>" (required by validator).
"""

from __future__ import annotations

import argparse
import sys


def _print_plan() -> None:
    print("V3-EXQ-445d -- SD-032b dACC full pipeline, V_s-enabled re-run", flush=True)
    print("Arms: OFF (dACC off, V_s on) vs ON_INDEPENDENT (dACC on, V_s on)", flush=True)
    print("V_s flags on both arms: per_stream / per_region / event_segmenter /", flush=True)
    print("                        invalidation_trigger / anchor_sets", flush=True)
    print("OFF-entropy precondition: if OFF entropy < 0.05 in >=2/3 seeds ->", flush=True)
    print("                          evidence_direction = inconclusive.", flush=True)
    print("Else evaluate C1 harm_a_forward_r2 >= 0.3, C2 |d_entropy| >= 0.1,", flush=True)
    print("              C3 entropy_ON >= entropy_OFF (all in >=2/3 seeds).", flush=True)
    print("Supersedes V3-EXQ-445c. Depends on V3-EXQ-476.", flush=True)
    print("experiment_purpose=evidence", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3-EXQ-445d SD-032b dACC + V_s re-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    raise NotImplementedError(
        "V3-EXQ-445d gated. MECH-284 Phase 3 substrate landed 2026-04-24, "
        "but cascade gate V3-EXQ-476a/476b ran FAIL (V_s wired-but-inert: "
        "accumulator integrates and anchor resets fire, but action_class_"
        "entropy is unchanged -- catatonic-lock at policy layer). The "
        "remaining blocker is SD-037 (broadcast override regulator, "
        "orexin-analog) -- next /implement-substrate target per "
        "substrate_queue.json. Do not run until SD-037 has landed and a "
        "new cascade gate has confirmed V_s flags break the monostrategy "
        "lock on the baseline agent."
    )


if __name__ == "__main__":
    sys.exit(main())
