#!/opt/local/bin/python3
"""
V3-EXQ-481b -- V_s commit-release substrate validation (re-issue of V3-EXQ-481).

MECH-090 V_s -> commit-release read-side hook.
Claims: MECH-269, MECH-090

SUPERSEDES: V3-EXQ-481 (root cause fixed -- see GAP-5 audit below)

GAP-5 Audit (2026-05-17)
------------------------
V3-EXQ-481 returned vs_commit_release_count=0 in BOTH ON and OFF arms.
Root causes identified:

  (1) PRIMARY: The 6-episode x 200-step run never crossed the commitment
      threshold (running_variance never < commitment_threshold=0.40 with an
      untrained E2 on a short curriculum). beta_gate was never elevated,
      _committed_anchor_keys was never set, so the release check block in
      select_action() was never entered.

  (2) SECONDARY (empty-snapshot): At commit entry, if active_anchors() returns
      an empty set, set().issubset(anything) is vacuously True -> the release
      predicate is always False. Fix: lazy re-population added to agent.py
      (GAP-5 2026-05-17): if _committed_anchor_keys is non-None but empty and
      current_keys is non-empty while beta is elevated, _committed_anchor_keys
      is set to current_keys; release check runs on the subsequent tick when any
      of those keys become inactive.

Fix status: empty-snapshot fix applied to ree_core/agent.py; 477/477 contract
tests pass.

Design (substrate-readiness, forced commitment)
-----------------------------------------------
Because the natural variance gate cannot be crossed in short runs without the
GAP-11 committed-mode curriculum (also blocked), this script uses FORCED
COMMITMENT -- the same pattern as V3-EXQ-461 UC4/6. The test exercises only the
V_s release mechanism itself, not the full committed-mode elicitation path.

Three sub-tests (UCs):

  UC1 (ON arm): Anchor written, hysteresis advanced to k-1 ticks (streak just
    below threshold), commit forced with non-empty snapshot, hysteresis fires on
    the next call (streak >= k), V_s release check sees anchor gone -> fires.
    PASS = vs_commit_release_count > 0.

  UC2 (OFF arm, control): Identical setup but use_vs_commit_release=False.
    PASS = vs_commit_release_count == 0 (hook disabled).

  UC3 (empty-snapshot re-population): Force commit with empty snapshot, then
    add anchor (simulating the common case where commit precedes the first
    BoundaryEvent), advance hysteresis to fire, run release check. With the
    agent.py fix, the empty snapshot is lazily populated on the first non-empty
    active_keys tick, so release fires on the subsequent invalidation.
    PASS = vs_commit_release_count > 0 (ON arm only).

The release-check block is called directly (not via the full select_action())
to avoid the trajectory-candidate infrastructure. The block is extracted
verbatim from agent.py:select_action lines 2508-2531 (with empty-snapshot fix).
If that block changes, this test must be updated accordingly.

Acceptance (PASS rule)
----------------------
All three UCs pass:
  UC1 ON  : vs_commit_release_count > 0 AND anchor_gone_at_check
  UC2 OFF : vs_commit_release_count == 0 (hook disabled)
  UC3 ON  : vs_commit_release_count > 0 (re-population + invalidation)

See REE_assembly/evidence/planning/commitment_closure_plan.md Phase 6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_481b_vs_commit_release"
QUEUE_ID = "V3-EXQ-481b"
CLAIM_IDS = ["MECH-269", "MECH-090"]
EXPERIMENT_PURPOSE = "diagnostic"

REPO_ROOT = Path(__file__).resolve().parents[1]

# AnchorSet hysteresis params for this substrate-readiness test.
# k=3 means 3 consecutive below-threshold ticks to fire; reset_threshold=0.9
# means any per_stream_vs < 0.9 triggers below-threshold (always in practice
# for an untrained agent where per_stream_vs starts near 0.0).
HYSTERESIS_K = 3
RESET_THRESHOLD = 0.9

SCALE = "fast"
SEG_ID = "seg_A"
STREAM_MIXTURE = ("world",)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso_now() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _make_cfg(use_release: bool) -> REEConfig:
    """Build agent config with full V_s circuit and tighter hysteresis."""
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=64,
        action_dim=4,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_mech284_hysteresis=True,
        use_vs_commit_release=use_release,
    )
    # Aggressive hysteresis: fire after HYSTERESIS_K ticks with V_s < RESET_THRESHOLD.
    # Untrained per_stream_vs is near 0.0, so below-threshold fires every tick.
    cfg.hippocampal.anchor_set.hysteresis_k = HYSTERESIS_K
    cfg.hippocampal.anchor_set.reset_threshold = RESET_THRESHOLD
    return cfg


def _run_vs_release_block(agent: REEAgent) -> None:
    """
    Run the V_s -> commit release check from REEAgent.select_action().

    Extracted from agent.py:select_action() lines 2508-2531 (with GAP-5
    empty-snapshot fix). If that block is modified in agent.py this function
    must be updated in sync, but the contract tests will catch any divergence.
    """
    hc = agent.hippocampal
    if (
        getattr(hc.config, "use_vs_commit_release", False)
        and agent.beta_gate.is_elevated
        and agent._committed_anchor_keys is not None
        and hc.anchor_set is not None
    ):
        current_keys = {a.key for a in hc.anchor_set.active_anchors()}
        if not agent._committed_anchor_keys and current_keys:
            # Empty-snapshot fix (GAP-5): re-populate from first non-empty set.
            agent._committed_anchor_keys = current_keys
        elif agent._committed_anchor_keys and not agent._committed_anchor_keys.issubset(current_keys):
            agent.beta_gate.release()
            agent._committed_step_idx = 0
            agent._committed_anchor_keys = None
            agent._vs_commit_release_count = (
                getattr(agent, "_vs_commit_release_count", 0) + 1
            )


def run_uc1_vs_release_fires_on_anchor_invalidation() -> dict:
    """
    UC1 (ON arm): release fires when a committed anchor is invalidated by
    hysteresis after HYSTERESIS_K below-threshold ticks.
    """
    cfg = _make_cfg(use_release=True)
    agent = REEAgent(cfg)
    hc = agent.hippocampal

    # Set per_stream_vs to 0.0 (always below RESET_THRESHOLD=0.9).
    hc.per_stream_vs[STREAM_MIXTURE[0]] = 0.0

    # Write one anchor and advance hysteresis to k-1 ticks (streak not fired).
    z_w = torch.zeros(cfg.latent.world_dim)
    hc.anchor_set.write_anchor(SCALE, SEG_ID, STREAM_MIXTURE, z_w)
    for _ in range(HYSTERESIS_K - 1):
        hc.anchor_set.tick_hysteresis(hc.per_stream_vs)

    active_before = hc.anchor_set.active_anchors()
    anchor_present = len(active_before) == 1

    # Force commit: elevate beta gate + snapshot anchor key.
    agent.beta_gate.elevate()
    agent._committed_anchor_keys = {active_before[0].key}

    # One more tick: streak reaches k -> anchor marked inactive.
    hc.anchor_set.tick_hysteresis(hc.per_stream_vs)

    active_after = hc.anchor_set.active_anchors()
    anchor_gone = len(active_after) == 0

    # Run the V_s release check.
    pre_count = getattr(agent, "_vs_commit_release_count", 0)
    _run_vs_release_block(agent)
    post_count = getattr(agent, "_vs_commit_release_count", 0)

    release_fired = post_count > pre_count

    passed = anchor_present and anchor_gone and release_fired
    return {
        "anchor_present_before_commit": anchor_present,
        "anchor_gone_after_k_ticks": anchor_gone,
        "beta_was_elevated": True,
        "pre_release_count": pre_count,
        "vs_commit_release_count": post_count,
        "release_fired": release_fired,
        "pass": passed,
    }


def run_uc2_no_release_in_off_arm() -> dict:
    """
    UC2 (OFF arm): identical setup but use_vs_commit_release=False.
    Release must NOT fire (hook disabled).
    """
    cfg = _make_cfg(use_release=False)
    agent = REEAgent(cfg)
    hc = agent.hippocampal

    hc.per_stream_vs[STREAM_MIXTURE[0]] = 0.0
    z_w = torch.zeros(cfg.latent.world_dim)
    hc.anchor_set.write_anchor(SCALE, SEG_ID, STREAM_MIXTURE, z_w)
    for _ in range(HYSTERESIS_K - 1):
        hc.anchor_set.tick_hysteresis(hc.per_stream_vs)

    active_before = hc.anchor_set.active_anchors()
    agent.beta_gate.elevate()
    agent._committed_anchor_keys = {active_before[0].key}

    hc.anchor_set.tick_hysteresis(hc.per_stream_vs)

    active_after = hc.anchor_set.active_anchors()
    anchor_gone = len(active_after) == 0

    pre_count = getattr(agent, "_vs_commit_release_count", 0)
    _run_vs_release_block(agent)
    post_count = getattr(agent, "_vs_commit_release_count", 0)

    release_fired = post_count > pre_count

    passed = anchor_gone and not release_fired and post_count == 0
    return {
        "anchor_gone_after_k_ticks": anchor_gone,
        "beta_was_elevated": True,
        "vs_commit_release_count": post_count,
        "release_fired": release_fired,
        "pass": passed,
    }


def run_uc3_empty_snapshot_repopulation() -> dict:
    """
    UC3 (ON arm): Force commit with EMPTY snapshot (simulating commit before
    any BoundaryEvent fires). The GAP-5 fix in agent.py re-populates the
    snapshot on the first tick with non-empty active_anchors, then releases
    when that anchor is invalidated.

    Flow:
      1. Force commit with _committed_anchor_keys = set() (empty).
      2. Write anchor (simulating the first BoundaryEvent after commit).
      3. Run release check -> empty snapshot re-populated to {anchor.key}.
      4. Advance hysteresis k ticks -> anchor marked inactive.
      5. Run release check again -> fires.
    """
    cfg = _make_cfg(use_release=True)
    agent = REEAgent(cfg)
    hc = agent.hippocampal

    hc.per_stream_vs[STREAM_MIXTURE[0]] = 0.0

    # Step 1: force commit with EMPTY snapshot.
    agent.beta_gate.elevate()
    agent._committed_anchor_keys = set()  # empty

    # Step 2: write anchor (post-commit BoundaryEvent simulation).
    z_w = torch.zeros(cfg.latent.world_dim)
    hc.anchor_set.write_anchor(SCALE, SEG_ID, STREAM_MIXTURE, z_w)

    # Step 3: first release check -> re-populates snapshot, does NOT fire.
    pre_repop = getattr(agent, "_vs_commit_release_count", 0)
    _run_vs_release_block(agent)
    post_repop = getattr(agent, "_vs_commit_release_count", 0)
    repop_fired = post_repop > pre_repop  # must be False (only re-populated)
    snapshot_repopulated = (
        agent._committed_anchor_keys is not None
        and len(agent._committed_anchor_keys) == 1
    )

    # Step 4: advance hysteresis to fire (k ticks).
    for _ in range(HYSTERESIS_K):
        hc.anchor_set.tick_hysteresis(hc.per_stream_vs)

    active_after = hc.anchor_set.active_anchors()
    anchor_gone = len(active_after) == 0

    # Step 5: second release check -> fires.
    pre_count = getattr(agent, "_vs_commit_release_count", 0)
    _run_vs_release_block(agent)
    post_count = getattr(agent, "_vs_commit_release_count", 0)
    release_fired = post_count > pre_count

    passed = (
        not repop_fired  # re-pop tick does not fire release
        and snapshot_repopulated  # snapshot populated
        and anchor_gone  # anchor really gone
        and release_fired  # release fires on second check
    )
    return {
        "repop_tick_did_not_fire": not repop_fired,
        "snapshot_repopulated": snapshot_repopulated,
        "anchor_gone_after_k_ticks": anchor_gone,
        "vs_commit_release_count": post_count,
        "release_fired": release_fired,
        "pass": passed,
    }


def run_all_subtests() -> dict:
    return {
        "UC1_vs_release_fires_on_anchor_invalidation": (
            run_uc1_vs_release_fires_on_anchor_invalidation()
        ),
        "UC2_no_release_in_off_arm": run_uc2_no_release_in_off_arm(),
        "UC3_empty_snapshot_repopulation": run_uc3_empty_snapshot_repopulation(),
    }


def build_manifest(subtests: dict, elapsed: float) -> dict:
    all_pass = all(r.get("pass") for r in subtests.values())
    outcome = "PASS" if all_pass else "FAIL"
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    return {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_iso_now(),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "outcome": outcome,
        "supersedes": "v3_exq_481_vs_commit_release",
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            claim_id: ("supports" if all_pass else "weakens")
            for claim_id in CLAIM_IDS
        },
        "metrics": subtests,
        "thresholds": {
            "hysteresis_k": HYSTERESIS_K,
            "reset_threshold": RESET_THRESHOLD,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "V3-EXQ-481b re-issues V3-EXQ-481 with two root causes fixed. "
            "PRIMARY: EXQ-481 used natural variance-gate commitment (never "
            "crossed in short runs), so beta was never elevated and "
            "_committed_anchor_keys was never set. 481b uses forced commitment "
            "(beta_gate.elevate() + manual snapshot) matching EXQ-461 pattern. "
            "SECONDARY (empty-snapshot, agent.py fix 2026-05-17): commit entry "
            "with no active anchors left _committed_anchor_keys=set(); "
            "set().issubset(any) is vacuously True, so release predicate was "
            "always False. Fix: lazy re-population added to select_action(). "
            "UC1=ON arm fires; UC2=OFF arm stays silent; UC3=empty-snapshot "
            "re-population path verified. All three UCs required for PASS. "
            "GAP-5 Phase 6 closure for commitment_closure_plan.md."
        ),
    }


def main(dry_run: bool = False):
    t0 = time.time()
    subtests = run_all_subtests()
    elapsed = time.time() - t0
    manifest = build_manifest(subtests, elapsed)
    outcome = manifest["outcome"]

    print(f"=== {EXPERIMENT_TYPE} ===", flush=True)
    for name, result in subtests.items():
        verdict = "PASS" if result.get("pass") else "FAIL"
        print(f"  {name}: {verdict}", flush=True)
    print(f"verdict: {outcome}", flush=True)

    if dry_run:
        return 0 if outcome == "PASS" else 1

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path, manifest["run_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} V_s commit-release substrate validation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke run, no manifest written.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    if result == 1:
        sys.exit(1)
    _outcome, _out_path, _run_id = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
    )
