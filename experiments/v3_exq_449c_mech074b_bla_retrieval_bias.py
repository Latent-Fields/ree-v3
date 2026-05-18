#!/usr/bin/env python3
"""
V3-EXQ-449c -- MECH-074b BLA retrieval bias on action selection,
V_s-gated.

Purpose (evidence)
------------------
Measures whether BLAAnalog retrieval-bias ON vs OFF increases action
diversity and reduces harm_rate once the V_s cascade has already broken
the baseline monostrategy lock. MECH-074b retrieval_bias is the
content-selective per-trace weight vector
    w_i = 1 + alpha * arousal_tag_i
over hippocampal retrieval traces (LaBar & Cabeza 2006). It is NOT a
scalar gain and is orthogonal to MECH-074a encoding_gain.

Arms
----
    OFF:  use_amygdala_analog=True, use_bla_analog=True, but
          bla_retrieval_bias_alpha = 0.0 (retrieval bias path wired
          but producing no weight modulation).
          V_s runtime flags ON (per cascade precondition).
    ON:   use_amygdala_analog=True, use_bla_analog=True,
          bla_retrieval_bias_alpha = 0.5 (per SD-035 default).
          V_s runtime flags ON.

Metrics
-------
    action_class_entropy per arm per seed.
    harm_rate per arm per seed (episodic harm_exposure / episodes).
    retrieval_bias_weight_variance per arm (diagnostic, expected 0
    on OFF arm and > 0 on ON arm).

Pass criteria
-------------
    C1: entropy_ON - entropy_OFF >= 0.1 in >= 2/3 seeds
        (retrieval bias produces diversity over baseline-with-V_s).
    C2: harm_rate_ON <= harm_rate_OFF in >= 2/3 seeds
        (bias does not make the agent bolder in a harmful way; ideally
        strictly lower).
    C3: retrieval_bias_weight_variance_ON > 0 in >= 2/3 seeds
        (wiring sanity; if this is 0 the BLA consumer chain isn't
        actually consuming the bias).
    PASS = C1 AND C2 AND C3.

    NOTE: BLA retrieval-bias hippocampal consumer wiring (write-gain
    multiplication, retrieval reweighting, remap handoff) is recorded
    in the SD-035 landing entry as DEFERRED: "the module emits the
    signals but the HippocampalModule does not yet read them."
    Running this experiment requires that consumer wiring be landed
    first. If the consumer is still absent at run time, C3 will fail
    and the experiment should be reported as BLOCKED ON SD-035
    CONSUMER, not FAIL.

Tags / claims
-------------
    MECH-074b only. (NOT MECH-074a encoding_gain, NOT MECH-074d
    remap_signal -- those are separate experiments.)

Supersedes
----------
    None (first V_s-gated BLA retrieval bias behavioural run).

Depends on
----------
    V3-EXQ-445d (SD-032b dACC with V_s). If EXQ-445d fails the
    OFF-entropy precondition, EXQ-449c cannot distinguish retrieval-bias
    effect from the background lock either.

Status
------
    PLANNING STUB. Full implementation deferred pending:
    (a) V3-EXQ-476 PASS,
    (b) V3-EXQ-445d PASS,
    (c) MECH-074b hippocampal consumer wiring landed on SD-035.

Full-implementation TODO
------------------------
    - Build OFF / ON REEConfig.from_dims() pair with master amygdala
      switch on and only bla_retrieval_bias_alpha differing.
    - Standard CausalGridWorldV2 harm_rate + action_class_entropy
      harness (match EXQ-449-family episode / step counts).
    - Expose retrieval_bias_weight_variance diagnostic from BLAAnalog
      cached outputs (per-tick std of the emitted weight vector).
    - Write flat JSON manifest to evidence/experiments/.
    - Print "Result written to: <path>" (validator requirement).
"""

from __future__ import annotations

import argparse
import sys


def _print_plan() -> None:
    print("V3-EXQ-449c -- MECH-074b BLA retrieval bias, V_s-gated", flush=True)
    print("Arms: OFF (alpha=0.0) vs ON (alpha=0.5), both with V_s flags on", flush=True)
    print("Metrics: action_class_entropy, harm_rate, retrieval_bias_weight_variance", flush=True)
    print("C1 d_entropy >= 0.1; C2 harm_rate_ON <= harm_rate_OFF;", flush=True)
    print("C3 retrieval_bias_weight_variance_ON > 0 (wiring sanity).", flush=True)
    print("All criteria in >=2/3 seeds.", flush=True)
    print("Depends on V3-EXQ-445d PASS. experiment_purpose=evidence", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V3-EXQ-449c MECH-074b BLA retrieval bias")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit 0; do not execute.")
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return 0

    raise NotImplementedError(
        "V3-EXQ-449c gated. MECH-284 Phase 3 substrate landed 2026-04-24 "
        "but cascade gate V3-EXQ-476a/476b ran FAIL (V_s wired-but-inert "
        "-- catatonic-lock at policy layer). Remaining blockers: (a) SD-037 "
        "(broadcast override regulator, orexin-analog) -- next "
        "/implement-substrate target per substrate_queue.json -- to break "
        "the monostrategy lock; (b) MECH-074b hippocampal consumer wiring "
        "landed on SD-035 (retrieval-bias-aware replay path must be added "
        "before the bias signal can influence behaviour). Do not run until "
        "both have landed; re-queue under a new EXQ at that point."
    )


if __name__ == "__main__":
    sys.exit(main())
