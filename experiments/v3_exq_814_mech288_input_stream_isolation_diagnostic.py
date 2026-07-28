#!/opt/local/bin/python3
"""
V3-EXQ-814 -- MECH-288 input_stream isolation diagnostic (substrate readiness).

This is Step 8 of /implement-substrate for the MECH-288 `input_stream`
extension landed in ree_core/hippocampal/event_segmenter.py (ree-v3
commit dea87fc), the substrate prerequisite ARC-070/MECH-321
(policy_decomposition_on_prediction_failure) needs per the 2026-05-10
ARC-070 lit-pull R2 verdict (LOAD-BEARING, conf 0.74). ARC-070/MECH-321
itself is NOT built yet, so nothing in the live agent loop calls the
rollout-stream path -- this script exercises it directly to validate the
extension before anything depends on it.

EXPERIMENT_PURPOSE = "diagnostic": this is substrate-readiness validation,
not a claim-scoring run. claim_ids intentionally empty -- see the
docstring section below on why this does not tag MECH-288/MECH-321.

DESIGN CHOICE (departs from a full REEAgent+env harness -- stated
explicitly rather than silently): this diagnostic drives EventSegmenter
directly with a deterministic, seeded, structured synthetic latent
trajectory (slow drift + periodic jumps -- real events for the detectors
to find, not pure noise), the same style the module's own contract tests
(tests/contracts/test_mech_288_event_segmenter.py) use. The property under
test -- that the observation and rollout streams cannot share state -- is
a property of EventSegmenter itself, not of any particular latent source.
Standing up a full REEAgent + CausalGridWorldV2 harness would add a large
surface of config/attribute-path risk (exactly what this repo's code-review
checklist warns about) without changing what this diagnostic can prove.
Agent-level integration correctness for the (unchanged) observation path is
already covered by the 61/61 passing downstream-consumer contracts
(test_mech_288_event_segmenter, test_mech_269_anchor_set,
test_mech_269_per_region_vs, test_mech_287_invalidation_trigger,
test_q081_landmark_removal) run on the cloud fleet before this landed.

Mechanism under test
---------------------
EventSegmenter.step()/force_boundary()/boundary_on() now take an
input_stream ("observation" | "rollout") and keep fully separate detector
instances + outer/inner/last-fire counters per stream. A rollout-stream
tick must never read or write the observation stream's state.

Arms (2 conditions x SEEDS)
----------------------------
  ARM_A_observation_only: tick the segmenter on the SAME synthetic
    latent trajectory using ONLY input_stream="observation", exactly the
    pre-extension single-stream call pattern.
  ARM_B_observation_plus_rollout_queries: tick the SAME trajectory on
    input_stream="observation" (identical calls to A), interleaved every
    tick with a synthetic boundary_on(stream="rollout", ...) query built
    from an independently-seeded noised copy of that tick's latents.

Both arms consume the IDENTICAL pre-generated observation-latent sequence
per seed (generated once, before either arm cell runs, from a local
seeded torch.Generator never touched by the cells' own RNG resets) so a
divergence in the observation-stream trajectory between A and B can only
be caused by the rollout activity leaking into observation-stream state,
not by different input data.

Criteria
--------
  C1 (load-bearing, isolation): for every seed, ARM_A's and ARM_B's
     observation-stream segment_id trajectory and per-tick BoundaryEvent
     list are IDENTICAL. This is the falsifiable claim the whole
     extension rests on.
  C2 (non-degenerate rollout path): across all seeds' ARM_B runs,
     boundary_on(stream="rollout", ...) fires at least once with
     posterior > 0, and the set of firing posteriors is not a single
     constant value -- proves the rollout path is a real graded signal,
     not dead code (which would pass C1 vacuously).
  C3 (reset isolation): after the tick loop, EventSegmenter.reset()
     brings BOTH streams' current_segment_id() back to "0.0".

PASS requires C1 AND C2 AND C3 to hold in every seed (small N, diagnostic
purpose -- this is a substrate-correctness check, not a statistical claim).

No SLEEP DRIVER (no sleep-related config touched). No phased training
(EventSegmenter has no learned/trainable component). No claims tagged
(see below).

Why claim_ids is empty
-----------------------
This validates the SUBSTRATE MECHANISM (isolation + a working rollout
query path) that MECH-321 will need, not the MECH-288 or MECH-321 claims
themselves. MECH-288's own promote-to-active gate (Phase 2 ii/iii/iv
integration + its falsifiable predictions) is untouched by this run, and
MECH-321 cannot be meaningfully tested until ARC-070 itself is built (its
other two dependencies -- ARC-070 substrate and live MECH-094-gated usage
-- remain unbuilt). Tagging either claim here would misrepresent what a
PASS/FAIL on this run actually supports.

Supersedes
----------
None (first validation of this extension).

Depends on
----------
ree_core/hippocampal/event_segmenter.py input_stream extension (landed).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

from ree_core.hippocampal.event_segmenter import (  # noqa: E402
    EventSegmenter,
    Scale,
    BoundaryEvent,
)

EXPERIMENT_TYPE = "v3_exq_814_mech288_input_stream_isolation_diagnostic"
QUEUE_ID = "V3-EXQ-814"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS_DEFAULT = [0, 1, 2]
N_TICKS_DEFAULT = 80
LATENT_DIM = 8

ARM_A = "ARM_A_observation_only"
ARM_B = "ARM_B_observation_plus_rollout_queries"

CANONICAL_SCALES = [
    Scale(name="fast", streams=("z_world", "z_self"), algorithm="pe_threshold",
          tau=2, min_segment_length=2, pe_threshold=0.65, window_length=200),
    Scale(name="slow", streams=("z_goal",), algorithm="bocpd_gaussian",
          tau=40, min_segment_length=15, hazard=1 / 40, posterior_threshold=0.5,
          top_k=20, prior_var=1.0),
]


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_segmenter() -> EventSegmenter:
    return EventSegmenter(
        scales=list(CANONICAL_SCALES),
        emit_to=["mech_287_broadcast", "mech_269_anchor_set"],
        scale_id_format="{outer}.{inner}",
        slow_scale_name="slow",
    )


def _synthetic_latent_stream(seed: int, n_ticks: int) -> List[Dict[str, torch.Tensor]]:
    """Deterministic, structured latent trajectory: slow drift + periodic jumps.

    Uses a local torch.Generator seeded from `seed` alone -- independent of
    global RNG state, so this sequence is identical however many times it
    is (re)generated and is never perturbed by an arm_cell's global RNG
    reset. Jumps every 13 (z_world/z_self) and 27 (z_goal) ticks give the
    fast and slow detectors real events to find, not pure white noise.
    """
    gen = torch.Generator().manual_seed(seed)
    latents: List[Dict[str, torch.Tensor]] = []
    z_world = torch.zeros(LATENT_DIM)
    z_self = torch.zeros(LATENT_DIM)
    z_goal = torch.zeros(LATENT_DIM)
    for t in range(n_ticks):
        z_world = z_world + 0.05 * torch.randn(LATENT_DIM, generator=gen)
        z_self = z_self + 0.05 * torch.randn(LATENT_DIM, generator=gen)
        z_goal = z_goal + 0.02 * torch.randn(LATENT_DIM, generator=gen)
        if t > 0 and t % 13 == 0:
            z_world = z_world + 3.0 * torch.randn(LATENT_DIM, generator=gen)
        if t > 0 and t % 27 == 0:
            z_goal = z_goal + 5.0 * torch.randn(LATENT_DIM, generator=gen)
        latents.append({
            "z_world": z_world.clone(),
            "z_self": z_self.clone(),
            "z_goal": z_goal.clone(),
        })
    return latents


def _events_to_rows(events: List[BoundaryEvent]) -> List[Dict[str, Any]]:
    return [
        {
            "segment_id_old": e.segment_id_old,
            "segment_id_new": e.segment_id_new,
            "scale": e.scale,
            "posterior": e.posterior,
            "t": e.t,
            "input_stream": e.input_stream,
        }
        for e in events
    ]


def _run_arm_a(latents: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """ARM_A: observation-stream ticks only, exactly the pre-extension call shape."""
    seg = _make_segmenter()
    segment_id_trajectory: List[str] = []
    all_events: List[Dict[str, Any]] = []
    for t, latent in enumerate(latents):
        events = seg.step(latent_dict=latent, pe_dict=None, t=t, input_stream="observation")
        segment_id_trajectory.append(seg.current_segment_id("observation"))
        all_events.extend(_events_to_rows(events))
    return {
        "segment_id_trajectory": segment_id_trajectory,
        "events": all_events,
        "final_segment_id": seg.current_segment_id("observation"),
    }


def _run_arm_b(
    latents: List[Dict[str, torch.Tensor]], seed: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ARM_B: observation ticks identical to A, PLUS a rollout boundary_on() query
    every tick using an independently-seeded noised copy of the same latent.

    Returns (observation_side, rollout_side) so the caller can check C1
    against observation_side and C2 against rollout_side without conflating
    the two streams' results.
    """
    seg = _make_segmenter()
    # Independent generator: seed offset well clear of any seed range this
    # experiment uses, so it can never coincide with an observation-stream seed.
    rollout_gen = torch.Generator().manual_seed(seed + 1_000_000)

    segment_id_trajectory: List[str] = []
    all_events: List[Dict[str, Any]] = []
    rollout_fired_posteriors: List[float] = []

    for t, latent in enumerate(latents):
        events = seg.step(latent_dict=latent, pe_dict=None, t=t, input_stream="observation")
        segment_id_trajectory.append(seg.current_segment_id("observation"))
        all_events.extend(_events_to_rows(events))

        rollout_latent = {
            k: v + 0.3 * torch.randn(LATENT_DIM, generator=rollout_gen)
            for k, v in latent.items()
        }
        result = seg.boundary_on(stream="rollout", latent=rollout_latent, pe=None)
        if result.fired:
            rollout_fired_posteriors.append(result.posterior)

    observation_side = {
        "segment_id_trajectory": segment_id_trajectory,
        "events": all_events,
        "final_segment_id": seg.current_segment_id("observation"),
    }
    rollout_side = {
        "fired_posteriors": rollout_fired_posteriors,
        "n_fired": len(rollout_fired_posteriors),
        "final_rollout_segment_id": seg.current_segment_id("rollout"),
    }
    return observation_side, rollout_side


def _run_seed(seed: int, n_ticks: int) -> Dict[str, Any]:
    latents = _synthetic_latent_stream(seed, n_ticks)
    config_slice = {
        "n_ticks": n_ticks,
        "latent_dim": LATENT_DIM,
        "scales": [
            {"name": sc.name, "streams": sc.streams, "algorithm": sc.algorithm,
             "pe_threshold": sc.pe_threshold, "hazard": sc.hazard,
             "posterior_threshold": sc.posterior_threshold}
            for sc in CANONICAL_SCALES
        ],
    }

    with arm_cell(seed, config_slice={**config_slice, "arm": ARM_A},
                  script_path=Path(__file__)) as cell_a:
        row_a = _run_arm_a(latents)
        cell_a.stamp(row_a)   # attaches row_a["arm_fingerprint"] in place

    with arm_cell(seed, config_slice={**config_slice, "arm": ARM_B},
                  script_path=Path(__file__)) as cell_b:
        obs_b, rollout_b = _run_arm_b(latents, seed)
        row_b = dict(obs_b)
        row_b["rollout"] = rollout_b
        cell_b.stamp(row_b)   # attaches row_b["arm_fingerprint"] in place -- row_b
                               # is the single source of truth returned below, not
                               # obs_b/rollout_b separately (those never get stamped)

    # C1: observation-stream trajectory + events identical between A and B.
    c1_trajectory_match = row_a["segment_id_trajectory"] == row_b["segment_id_trajectory"]
    c1_events_match = row_a["events"] == row_b["events"]
    c1_pass = bool(c1_trajectory_match and c1_events_match)

    # C3: reset brings both streams back to "0.0". Use a fresh segmenter run
    # through the same tick loop (B's shape, since it exercises both streams),
    # then reset and check.
    seg_reset_check = _make_segmenter()
    for t, latent in enumerate(latents):
        seg_reset_check.step(latent_dict=latent, pe_dict=None, t=t, input_stream="observation")
        seg_reset_check.boundary_on(stream="rollout", latent=latent, pe=None)
    seg_reset_check.reset()
    c3_pass = (
        seg_reset_check.current_segment_id("observation") == "0.0"
        and seg_reset_check.current_segment_id("rollout") == "0.0"
    )

    return {
        "seed": seed,
        "arm_a": row_a,
        "arm_b": row_b,
        "c1_trajectory_match": c1_trajectory_match,
        "c1_events_match": c1_events_match,
        "c1_pass": c1_pass,
        "c3_pass": c3_pass,
    }


def main(dry_run: bool = False) -> Any:
    seeds = [SEEDS_DEFAULT[0]] if dry_run else SEEDS_DEFAULT
    n_ticks = 20 if dry_run else N_TICKS_DEFAULT

    print(f"[{EXPERIMENT_TYPE}] dry_run={dry_run} seeds={seeds} n_ticks={n_ticks}", flush=True)
    t0 = time.time()
    per_seed_rows: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        row = _run_seed(seed, n_ticks)
        per_seed_rows.append(row)
        print(
            f"  [seed] {seed} c1_pass={row['c1_pass']} c3_pass={row['c3_pass']} "
            f"rollout_n_fired={row['arm_b']['rollout']['n_fired']}",
            flush=True,
        )
        print(f"  [progress] ep {i + 1}/{len(seeds)}", flush=True)

    # C2: across ALL seeds' rollout results -- fires at least once, posteriors not
    # a single constant value.
    all_rollout_posteriors: List[float] = []
    for row in per_seed_rows:
        all_rollout_posteriors.extend(row["arm_b"]["rollout"]["fired_posteriors"])
    c2_fired_at_least_once = len(all_rollout_posteriors) >= 1
    c2_not_constant = len(set(round(p, 9) for p in all_rollout_posteriors)) > 1
    c2_pass = bool(c2_fired_at_least_once and c2_not_constant)

    c1_pass_all = all(row["c1_pass"] for row in per_seed_rows)
    c3_pass_all = all(row["c3_pass"] for row in per_seed_rows)

    overall_pass = bool(c1_pass_all and c2_pass and c3_pass_all)
    outcome = "PASS" if overall_pass else "FAIL"
    elapsed = time.time() - t0

    interpretation = {
        "label": "isolation_and_rollout_path_verified" if overall_pass else "isolation_or_rollout_path_defect",
        "criteria_non_degenerate": {
            "C1": c1_pass_all,
            "C2": c2_pass,
            "C3": c3_pass_all,
        },
    }
    criteria = [
        {"name": "C1_observation_stream_isolation", "load_bearing": True, "passed": c1_pass_all},
        {"name": "C2_rollout_path_non_degenerate", "load_bearing": True, "passed": c2_pass},
        {"name": "C3_reset_isolates_both_streams", "load_bearing": False, "passed": c3_pass_all},
    ]

    print(
        f"[{EXPERIMENT_TYPE}] C1={c1_pass_all} C2={c2_pass} C3={c3_pass_all} "
        f"outcome={outcome} elapsed={elapsed:.2f}s",
        flush=True,
    )
    print(f"verdict: {outcome}", flush=True)

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; no manifest.", flush=True)
        return 0

    run_id = f"{EXPERIMENT_TYPE}_{_utc_compact()}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": _utc_compact(),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "non_degenerate": bool(c2_pass),
        "degeneracy_reason": None if c2_pass else "rollout path never fired or fired at a constant posterior",
        "interpretation": interpretation,
        "criteria": criteria,
        "elapsed_seconds": elapsed,
        "per_seed_results": per_seed_rows,
        "arm_results": (
            [{"arm_id": ARM_A, "seed": row["seed"], **row["arm_a"]} for row in per_seed_rows]
            + [{"arm_id": ARM_B, "seed": row["seed"], **row["arm_b"]} for row in per_seed_rows]
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config={"seeds": seeds, "n_ticks": n_ticks, "latent_dim": LATENT_DIM},
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-814 MECH-288 input_stream isolation diagnostic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if result == 0:
        emit_outcome(outcome="PASS", manifest_path=None, dry_run=True)
        sys.exit(0)
    outcome, out_path = result
    # main() returns 0 before writing a manifest under --dry-run, so this call is
    # unreachable in a smoke run. Threaded anyway so the guarantee is LOCAL: it
    # survives a future edit that lets the dry-run path fall through to the
    # writer. Inert on the evidence path -- False whenever --dry-run is absent.
    emit_outcome(outcome=outcome, manifest_path=out_path,
                 dry_run=bool(args.dry_run))
    sys.exit(0)
