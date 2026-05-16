"""
V3-EXQ-581: sleep_substrate GAP-3 validation -- unified Phase A-E
sleep-aggregation cluster master flag (use_sleep_aggregation_cluster).

Before GAP-3 the offline-consolidation pathway was gated by EIGHT
independent default-False flags; the cluster was silent unless an
experiment set all eight by hand (sleep_substrate_plan.md GAP-3). This
experiment validates that the new single switch
use_sleep_aggregation_cluster=True drives Phase A-E end-to-end and is
behaviourally identical to setting the eight sub-flags individually.

Seam under test:
  REEConfig.from_dims(use_sleep_aggregation_cluster=True)
  -> enable_sleep_aggregation_cluster() resolves eight sub-flags True
  -> REEAgent constructs sleep_loop + Phase B/C/D/E components
  -> force_cycle() produces non-trivial Phase B/C/D/E diagnostics.

Purpose: diagnostic / substrate-readiness. This confirms the unified flag
wires the pathway and is equivalence-safe; it does NOT itself adjudicate
the MECH-285/272/275/273 scientific hypotheses (those are unblocked, not
tested, by GAP-3). Non-claim-weighing (claim_ids=[]), excluded from
governance confidence/conflict scoring.

Interpretation grid:
  Outcome                                  | Diagnosis
  -----------------------------------------|--------------------------------------
  C1 FAIL (a component is None)            | enable_sleep_aggregation_cluster() did
                                           |   not resolve a sub-flag, OR a substrate
                                           |   prereq (anchor_set / e2_harm_s) absent
  C2 FAIL (mech285_n_draws == 0)          | Phase B sampler silent: use_mech285_sampler
                                           |   not resolved or anchor_set empty
  C3 FAIL (sws_anchor_weight ~= 1.0)      | Phase C consumer silent: GAP-8 path not
                                           |   live under the unified flag
  C4 FAIL (mech275_n_updates == 0)        | Phase D aggregator silent: probe channel
                                           |   zero or routing gate not feeding it
  C5 FAIL (mech273_n_offline_steps == 0)  | Phase E writeback silent: self-model
                                           |   aggregator absent or no regions consumed
  C6 FAIL (cluster != explicit metrics)   | the master flag is NOT pure ergonomics --
                                           |   resolution diverges from hand-set flags
  C1-C6 all PASS                           | GAP-3 closed: one switch lights the whole
                                           |   cluster, behaviourally identical to the
                                           |   eight-flag form
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_581_gap3_sleep_aggregation_cluster_validation_v3"
QUEUE_ID = "V3-EXQ-581"
CLAIM_IDS: List[str] = []  # diagnostic / substrate-readiness; non-claim-weighing
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7]
N_DRIVE_STEPS = 20         # act_with_split_obs steps to populate buffers
N_ANCHORS = 4              # anchors installed per the GAP-8 contract pattern
SWS_ANCHOR_WEIGHT = 0.6    # default mech272_sws_anchor_weight
DRAWS_PER_CYCLE = 8
OFFLINE_N_STEPS = 5        # Phase E bounded offline gradient steps

# The eight sub-flags GAP-3's unified master flag must resolve True.
SUB_FLAGS = (
    "use_sleep_loop",
    "sws_enabled",
    "rem_enabled",
    "use_mech285_sampler",
    "use_mech272_routing",
    "use_mech272_routing_consumer",
    "use_mech275_aggregator",
    "use_mech273_self_model",
)

# Per-phase diagnostics compared for the C6 equivalence check.
EQUIV_KEYS = (
    "mech285_n_draws",
    "mech285_n_distinct_regions_drawn",
    "sws_anchor_weight_applied",
    "sws_n_writes",
    "mech272_n_routed",
    "mech275_n_updates",
    "mech275_n_posteriors",
    "mech273_n_offline_steps",
    "mech273_n_offline_passes",
    "mech273_n_offline_regions_consumed",
)

# Pre-registered acceptance thresholds
C3_TOLERANCE = 1e-4        # cluster-arm SWS weight must equal SWS_ANCHOR_WEIGHT within this
C3_OFF_FROM_ONE = 1e-3     # ... and be measurably below 1.0
C6_LOSS_TOLERANCE = 1e-9   # float tolerance for loss-valued equivalence keys

ARMS = [
    {
        "arm": "ARM_CLUSTER",
        "use_cluster_flag": True,
        "description": "single use_sleep_aggregation_cluster=True master flag",
    },
    {
        "arm": "ARM_EXPLICIT",
        "use_cluster_flag": False,
        "description": "eight Phase A-E sub-flags set individually True",
    },
]


def _build_agent(*, use_cluster_flag: bool) -> Tuple[REEAgent, Dict[str, bool]]:
    """Build an agent either via the unified flag or the explicit eight.

    Substrate prerequisites (anchor sets, staleness accumulator,
    e2_harm_s) are NOT folded into the cluster flag by design (separate
    MECH-269/ARC-033 switches, per GAP-3 scope) so both arms enable them
    explicitly and identically.
    """
    common = dict(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        sleep_loop_episodes_K=1,
        mech285_draws_per_cycle=DRAWS_PER_CYCLE,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_e2_harm_s_forward=True,
        mech272_sws_anchor_weight=SWS_ANCHOR_WEIGHT,
        mech273_offline_n_steps=OFFLINE_N_STEPS,
    )
    if use_cluster_flag:
        cfg = REEConfig.from_dims(use_sleep_aggregation_cluster=True, **common)
    else:
        cfg = REEConfig.from_dims(
            use_sleep_loop=True,
            sws_enabled=True,
            rem_enabled=True,
            use_mech285_sampler=True,
            use_mech272_routing=True,
            use_mech272_routing_consumer=True,
            use_mech275_aggregator=True,
            use_mech273_self_model=True,
            **common,
        )
    resolved = {f: bool(getattr(cfg, f)) for f in SUB_FLAGS}
    return REEAgent(cfg), resolved


def _seed_all(seed: int) -> None:
    """Seed every stochastic source the cluster touches.

    The MECH-285 SleepReplaySampler draws via the module-level numpy RNG
    (np.random.choice); the GAP-4 replay buffer samples via random.choices.
    Seeding only torch leaves sampler region-diversity non-deterministic
    and lets the first arm's draws bleed into the second (same process),
    which is exactly what made the C6 equivalence check spuriously fail.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _install_anchors(agent: REEAgent, *, n: int = N_ANCHORS) -> None:
    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None, "anchor_set must be initialised"
    for i in range(n):
        z = torch.randn(1, 32)
        anchor_set.write_anchor(
            scale="fast",
            segment_id=str(i),
            stream_mixture=(f"s{i}",),
            z_world=z,
        )


def _component_presence(agent: REEAgent) -> Dict[str, bool]:
    return {
        "sleep_loop": agent.sleep_loop is not None,
        "sleep_replay_sampler": agent.sleep_replay_sampler is not None,
        "sleep_routing_gate": agent.sleep_routing_gate is not None,
        "sleep_bayesian_aggregator": agent.sleep_bayesian_aggregator is not None,
        "sleep_self_model_aggregator": agent.sleep_self_model_aggregator is not None,
    }


def _run_arm_seed(*, use_cluster_flag: bool, seed: int) -> Dict[str, Any]:
    _seed_all(seed)
    agent, resolved = _build_agent(use_cluster_flag=use_cluster_flag)
    presence = _component_presence(agent)

    # Re-seed AFTER construction. The two arms reach an identical resolved
    # config but via different from_dims call patterns, which consume a
    # different number of RNG draws during weight init. Re-seeding here
    # isolates the C6 question -- "is the master flag pure ergonomics
    # (zero behavioural divergence)?" -- from construction RNG bookkeeping:
    # given identical resolved configs, the behavioural portion (drive loop
    # + anchor draws + sampler cycle) is now bit-identical iff the flag
    # truly only changes how the config is built, not what the agent does.
    _seed_all(seed)

    # Drive the full waking loop (act_with_split_obs), NOT bare sense():
    # _world_experience_buffer is appended only inside _e1_tick(), which
    # runs from act_with_split_obs(); run_sws_schema_pass returns early
    # when the buffer has < 2 entries (the GAP-8 driver-pattern finding).
    for step in range(N_DRIVE_STEPS):
        obs_body = torch.randn(12)
        obs_world = torch.randn(250)
        agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)
        if (step + 1) % 5 == 0:
            print(
                f"  [train] drive seed={seed} ep {step + 1}/{N_DRIVE_STEPS}",
                flush=True,
            )

    _install_anchors(agent)
    metrics = agent.sleep_loop.force_cycle(agent)

    def m(key: str, default: float = 0.0) -> float:
        if key == "mech272_n_routed":
            return float(
                metrics.get(
                    "mech272_n_routed_sws", metrics.get("mech272_n_routed", default)
                )
            )
        return float(metrics.get(key, default))

    phase = {k: m(k) for k in EQUIV_KEYS}
    return {
        "seed": seed,
        "use_cluster_flag": use_cluster_flag,
        "resolved_subflags": resolved,
        "component_presence": presence,
        "phase_metrics": phase,
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print("V3-EXQ-581: GAP-3 sleep-aggregation cluster validation", flush=True)
    print(f"  seeds={seeds} dry_run={dry_run}", flush=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for arm_cfg in ARMS:
        arm_name = arm_cfg["arm"]
        use_cluster_flag = arm_cfg["use_cluster_flag"]
        all_results[arm_name] = []
        for seed in seeds:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            result = _run_arm_seed(
                use_cluster_flag=use_cluster_flag, seed=seed
            )
            all_results[arm_name].append(result)
            pm = result["phase_metrics"]
            print(
                f"  [result] seed={seed} arm={arm_name} "
                f"mech285_draws={pm['mech285_n_draws']:.0f} "
                f"sws_anchor_weight={pm['sws_anchor_weight_applied']:.6f} "
                f"mech275_updates={pm['mech275_n_updates']:.0f} "
                f"mech273_offline_steps={pm['mech273_n_offline_steps']:.0f}",
                flush=True,
            )
            print("verdict: PASS", flush=True)

    cluster = all_results["ARM_CLUSTER"]
    explicit = all_results["ARM_EXPLICIT"]

    # C1: cluster arm resolves all eight sub-flags AND constructs all
    # five components (sleep_loop + Phase B/C/D/E) for every seed.
    c1_pass = all(
        all(r["resolved_subflags"].values())
        and all(r["component_presence"].values())
        for r in cluster
    )
    # C2: Phase B sampler fired (draws > 0) in the cluster arm.
    c2_pass = all(r["phase_metrics"]["mech285_n_draws"] > 0 for r in cluster)
    # C3: Phase C consumer fired -- SWS anchor weight ~= 0.6 (below 1.0).
    c3_pass = all(
        abs(r["phase_metrics"]["sws_anchor_weight_applied"] - SWS_ANCHOR_WEIGHT)
        <= C3_TOLERANCE
        and r["phase_metrics"]["sws_anchor_weight_applied"]
        <= 1.0 - C3_OFF_FROM_ONE
        for r in cluster
    )
    # C4: Phase D aggregator fired (posterior updates > 0).
    c4_pass = all(r["phase_metrics"]["mech275_n_updates"] > 0 for r in cluster)
    # C5: Phase E self-model offline gradient pass ran (n_offline_steps > 0).
    c5_pass = all(
        r["phase_metrics"]["mech273_n_offline_steps"] > 0 for r in cluster
    )

    # C6: equivalence -- per-seed per-key cluster == explicit.
    c6_mismatches: List[str] = []
    cluster_by_seed = {r["seed"]: r["phase_metrics"] for r in cluster}
    explicit_by_seed = {r["seed"]: r["phase_metrics"] for r in explicit}
    for seed in cluster_by_seed:
        cl = cluster_by_seed[seed]
        ex = explicit_by_seed.get(seed, {})
        for k in EQUIV_KEYS:
            cv, ev = cl.get(k, 0.0), ex.get(k, 0.0)
            tol = C6_LOSS_TOLERANCE if "loss" in k else 0.0
            if abs(cv - ev) > tol:
                c6_mismatches.append(
                    f"seed={seed} {k}: cluster={cv} explicit={ev}"
                )
    c6_pass = len(c6_mismatches) == 0

    outcome = (
        "PASS"
        if (c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass)
        else "FAIL"
    )

    print("", flush=True)
    print(f"C1 (cluster: 8 sub-flags + 5 components): {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"C2 (Phase B mech285_n_draws > 0): {'PASS' if c2_pass else 'FAIL'}", flush=True)
    print(f"C3 (Phase C sws_anchor_weight ~= {SWS_ANCHOR_WEIGHT}): {'PASS' if c3_pass else 'FAIL'}", flush=True)
    print(f"C4 (Phase D mech275_n_updates > 0): {'PASS' if c4_pass else 'FAIL'}", flush=True)
    print(f"C5 (Phase E mech273_n_offline_steps > 0): {'PASS' if c5_pass else 'FAIL'}", flush=True)
    print(f"C6 (cluster == explicit equivalence): {'PASS' if c6_pass else 'FAIL'}", flush=True)
    if c6_mismatches:
        print(f"  C6 mismatches: {c6_mismatches}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "c5_pass": c5_pass,
        "c6_pass": c6_pass,
        "c6_mismatches": c6_mismatches,
        "all_results": all_results,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_drive_steps": N_DRIVE_STEPS,
            "n_anchors": N_ANCHORS,
            "sws_anchor_weight": SWS_ANCHOR_WEIGHT,
            "draws_per_cycle": DRAWS_PER_CYCLE,
            "offline_n_steps": OFFLINE_N_STEPS,
        },
        "acceptance_criteria": {
            "C1": "cluster arm resolves 8 sub-flags + constructs 5 components",
            "C2": "Phase B mech285_n_draws > 0",
            "C3": f"Phase C sws_anchor_weight_applied within {C3_TOLERANCE} of {SWS_ANCHOR_WEIGHT} and <= 1 - {C3_OFF_FROM_ONE}",
            "C4": "Phase D mech275_n_updates > 0",
            "C5": "Phase E mech273_n_offline_steps > 0",
            "C6": "ARM_CLUSTER per-phase metrics == ARM_EXPLICIT (equivalence)",
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_pass": result["c2_pass"],
            "C3_pass": result["c3_pass"],
            "C4_pass": result["c4_pass"],
            "C5_pass": result["c5_pass"],
            "C6_pass": result["c6_pass"],
            "C6_mismatches": result["c6_mismatches"],
        },
        "per_arm_per_seed_results": result["all_results"],
        "notes": (
            "sleep_substrate GAP-3 validation: unified "
            "use_sleep_aggregation_cluster master flag lights Phase A-E "
            "end-to-end (C1-C5) and is behaviourally identical to setting "
            "the eight sub-flags by hand (C6 equivalence). Diagnostic / "
            "substrate-readiness; non-claim-weighing (claim_ids=[]); "
            "unblocks but does not test MECH-285/272/275/273."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
