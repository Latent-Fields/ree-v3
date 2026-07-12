"""V3-EXQ-658 -- MECH-045 token-instance object-file buffer: persistence vs ablation.

LINEAGE / ROUTING
-----------------
- Originating proposal : EVB-0293 / EXP-0117 (claim MECH-045). A /queue-experiment
  Step-2.5 readiness gate marked it status=blocked_substrate on 2026-06-09 because
  the token-instance object-file buffer was DORMANT (design-only). A persistence-
  vs-ablation probe against the two LIVE adjacent stores (SD-057 IncentiveTokenBank
  = TYPE-keyed; SD-039 ghost-goal bank = ANCHOR-keyed) would be vacuous -- neither
  can re-identify a moved entity, so neither can fail the test.
- Substrate landed (this session, /implement-substrate): the TOKEN store --
  ree_core/entities/object_file_buffer.py (ObjectFileBuffer), behind no-op-default
  use_object_file_buffer, driven on the waking stream via
  REEAgent.update_object_file_buffer(). The missing third store of the ARC-080
  type/token/anchor triad.
- THIS experiment is the discriminative persistence-vs-ablation test pre-registered
  in the design memo (REE_assembly/docs/architecture/mech_045_object_file_buffer.md
  Section 6): can the buffer assign the SAME token id to an entity BEFORE and AFTER
  it moves to a new cell, with a same-TYPE distractor present so continuity (not
  type, not nearest-cell) must carry identity?

CLAIM HANDLING
--------------
claim_ids = ["MECH-045"]  (ARC-006 bears-on but is a substrate_coherence
commitment exempt from experiment gating -- NOT co-tagged, per the claim_ids
accuracy rule). evidence_direction is set per the memo Section 6.3 grid:
  readiness (G0-G2) unmet  -> non_contributory, label substrate_not_ready_requeue
  readiness met + C1+C2+C3 -> supports,         label token_buffer_reidentifies_cross_motion
  readiness met + C1/C2 fail -> weakens,        label identity_not_continuity_carried
  readiness met + only C3 fail -> mixed
This experiment does NOT itself promote/weaken MECH-045 -- governance applies that.

SUBSTRATE DETAIL (memo Section 5.1)
-----------------------------------
world_dim = 128 (the E2WorldForward floor; the dim=32 z_world granularity ceiling
would make the feature term degenerate -> a vacuous motion-only pass). The buffer
is feature-source agnostic; this controlled probe supplies each entity a distinct
world_dim-dim unit "appearance" (same TYPE = same resource_tag, different
appearance) so the feature term is genuinely discriminable. G1 verifies the
appearances are non-degenerate (d_cos(target, distractor) >= 0.1) BEFORE scoring --
the SAME statistic the re-identification criterion depends on -- so a degenerate
feature space self-routes substrate_not_ready_requeue rather than masquerading as
a refuted claim.

DESIGN (3 arms x 3 seeds [42,43,44]; matched seeds; controlled appearances)
---------------------------------------------------------------------------
  ARM_INTACT        use_object_file_buffer=True; token id = data-association.
  ARM_ABLATION_OFF  use_object_file_buffer=False; no buffer -> no token (reid=0 floor).
  ARM_ABLATION_SHUFFLE  buffer ON, but token ids permuted each tick before readout
                    -> kills identity CONTINUITY while keeping the buffer "on"; the
                    load-bearing control showing IDENTITY (not buffer presence)
                    carries persistence.
Per trial: a target + a same-type distractor are observed at t0; the target MOVES
to a new cell at t1 (within the continuity radius); the buffer is read; a brief
occlusion of the target then tests feature persistence (C3).

This experiment trains NOTHING (the buffer is non-trainable; appearances are
constructed) -> no phased training. The agent is built only to exercise the real
ree_core wiring (REEAgent.update_object_file_buffer).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.entities.object_file_buffer import EntityObservation  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_658_mech045_object_file_persistence"
QUEUE_ID = "V3-EXQ-658"
BACKLOG_ID = "EVB-0293"
CLAIM_IDS: List[str] = ["MECH-045"]
EXPERIMENT_PURPOSE = "evidence"  # discriminative_pair; weights MECH-045 when readiness met

SEEDS = [42, 43, 44]
N_TRIALS = 40
DRY_RUN_SEEDS = [42]
DRY_RUN_TRIALS = 6

WORLD_DIM = 128          # memo Section 5.1 -- the E2WorldForward / discriminative floor
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

CONTINUITY_RADIUS = 2.0  # obf_continuity_radius (cells)
OCCLUDE_TICKS = 3        # < obf_persist_ttl (8) so the token survives the occlusion
APPEAR_NOISE = 0.02      # per-observation appearance noise
# Several SAME-TYPE distractors (the MOT "identical objects" structure): more
# same-type entities make continuity carry identity and drop the SHUFFLE chance-
# re-identification floor toward ~1/(k+1).
N_DISTRACTORS = 3
DISTRACTOR_POSITIONS = [(3.0, 7.0), (7.0, 3.0), (7.0, 7.0), (1.0, 9.0)]

# Pre-registered thresholds (memo Section 6.2 / 6.3).
G1_DCOS_FLOOR = 0.1      # feature-degeneracy guard (the dim=32 ceiling guard)
C1_REID_FLOOR = 0.6      # PRIMARY: reid_frac_INTACT >= 0.6
C2_SHUFFLE_MARGIN = 0.3  # reid_frac_INTACT > reid_frac_SHUFFLE + 0.3
MIN_SEEDS_2OF3 = 2

ARM_INTACT = "ARM_INTACT"
ARM_OFF = "ARM_ABLATION_OFF"
ARM_SHUFFLE = "ARM_ABLATION_SHUFFLE"
ARMS = [ARM_INTACT, ARM_OFF, ARM_SHUFFLE]


def _unit(dim: int, gen: torch.Generator) -> torch.Tensor:
    v = torch.randn(dim, generator=gen)
    return v / (v.norm() + 1e-8)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item()) / (float(a.norm()) * float(b.norm()) + 1e-8)


def _build_agent(use_buffer: bool) -> REEAgent:
    # arm_cell.__enter__ already reset RNG for this seed.
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=32,
        world_dim=WORLD_DIM,
        use_object_file_buffer=use_buffer,
        obf_continuity_radius=CONTINUITY_RADIUS,
        obf_max_tokens=N_DISTRACTORS + 2,  # 1 target + k distractors, with headroom
    )
    return REEAgent(cfg)


def _shuffle_ids(assignment: Dict[int, int], gen: torch.Generator) -> Dict[int, int]:
    """ABLATION-SHUFFLE: permute the token-id VALUES of an assignment before
    readout, killing identity continuity while keeping the buffer 'on'."""
    if not assignment:
        return {}
    ids = sorted(set(assignment.values()))
    if len(ids) <= 1:
        # one id -> a permutation is a no-op; remap to a fresh sentinel each tick
        # so the readout id is unstable across ticks (the whole point).
        sentinel = int(torch.randint(10_000, 1_000_000, (1,), generator=gen).item())
        return {k: sentinel for k in assignment}
    perm = torch.randperm(len(ids), generator=gen).tolist()
    remap = {ids[i]: ids[perm[i]] for i in range(len(ids))}
    return {k: remap[v] for k, v in assignment.items()}


def _run_trial(agent: REEAgent, arm: str, gen: torch.Generator) -> Dict[str, Any]:
    """One persistence trial. Returns per-trial metrics."""
    buf = agent.object_file_buffer
    if buf is not None:
        buf.reset()

    vT = _unit(WORLD_DIM, gen)   # target appearance (instance 1)
    # k SAME-TYPE distractor appearances (distinct instances, same resource_tag).
    vDs = [_unit(WORLD_DIM, gen) for _ in range(N_DISTRACTORS)]
    d_cos_td = min(1.0 - _cos(vT, vd) for vd in vDs)  # worst-case (min) separation

    # grid cells: target at A=(3,3) moves to A'=(4,4) (new cell, within radius);
    # distractors held at distinct far cells.
    posA = torch.tensor([3.0, 3.0])
    posA2 = torch.tensor([4.0, 4.0])
    posDs = [torch.tensor(list(DISTRACTOR_POSITIONS[i])) for i in range(N_DISTRACTORS)]
    moved = (int(posA[0]) != int(posA2[0])) or (int(posA[1]) != int(posA2[1]))

    def obs(pos, appear):
        noise = torch.randn(WORLD_DIM, generator=gen) * APPEAR_NOISE
        return EntityObservation(pos=pos.clone(), z=(appear + noise),
                                  salience=1.0, precision=1.0, resource_tag=1)

    def step(observations):
        a = agent.update_object_file_buffer(observations)
        if arm == ARM_SHUFFLE:
            a = _shuffle_ids(a, gen)
        return a

    def scene(target_pos, include_target=True):
        obs_list = []
        if include_target:
            obs_list.append(obs(target_pos, vT))   # target is index 0 when present
        for i in range(N_DISTRACTORS):
            obs_list.append(obs(posDs[i], vDs[i]))
        return obs_list

    # t0: target + all distractors present (target at index 0).
    a0 = step(scene(posA))
    tok_target0 = a0.get(0, None)
    distractor_toks0 = {a0.get(1 + i, None) for i in range(N_DISTRACTORS)}
    # t1: target MOVED to a new cell; distractors held.
    a1 = step(scene(posA2))
    tok_target1 = a1.get(0, None)

    # G0: INTACT actually maintained a live token for the target across the move.
    target_maintained = (tok_target0 is not None) and (tok_target1 is not None)
    # C1/C2: same token AND it is the target's (not any distractor's).
    reid = bool(
        tok_target1 is not None
        and tok_target0 is not None
        and tok_target1 == tok_target0
        and tok_target1 not in distractor_toks0
    )

    # C3: persistence-through-absence -- occlude the target (only distractors
    # visible), then check the token's bound features stay closer to the TARGET
    # appearance than to any distractor appearance.
    persisted_match = False
    if arm == ARM_INTACT and buf is not None and tok_target1 is not None:
        for _ in range(OCCLUDE_TICKS):
            step(scene(posA2, include_target=False))
        tok = buf.query(tok_target1)
        if tok is not None:
            sim_target = _cos(tok.z_features, vT)
            persisted_match = all(sim_target > _cos(tok.z_features, vd) for vd in vDs)

    return {
        "d_cos_td": d_cos_td,
        "moved": moved,
        "target_maintained": target_maintained,
        "reid": reid,
        "persisted_match": persisted_match,
    }


def _run_cell(arm: str, seed: int, n_trials: int) -> Dict[str, Any]:
    use_buffer = arm in (ARM_INTACT, ARM_SHUFFLE)
    agent = _build_agent(use_buffer)
    gen = torch.Generator().manual_seed(seed * 1000 + hash(arm) % 997)
    rows = []
    for i in range(n_trials):
        rows.append(_run_trial(agent, arm, gen))
        if (i + 1) % 10 == 0 or (i + 1) == n_trials:
            print(f"  [eval] {arm} seed={seed} ep {i + 1}/{n_trials}", flush=True)
    n = len(rows)
    reid_frac = sum(r["reid"] for r in rows) / n
    maintained_frac = sum(r["target_maintained"] for r in rows) / n
    moved_frac = sum(r["moved"] for r in rows) / n
    d_cos_mean = sum(r["d_cos_td"] for r in rows) / n
    persisted_frac = sum(r["persisted_match"] for r in rows) / n
    return {
        "arm": arm,
        "seed": seed,
        "n_trials": n,
        "reid_frac": reid_frac,
        "maintained_frac": maintained_frac,
        "moved_frac": moved_frac,
        "d_cos_td_mean": d_cos_mean,
        "persisted_match_frac": persisted_frac,
    }


def _frac_seeds(pred_by_seed: Dict[int, bool]) -> int:
    return sum(1 for v in pred_by_seed.values() if v)


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    n_trials = DRY_RUN_TRIALS if dry_run else N_TRIALS

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice={
                    "arm": arm, "world_dim": WORLD_DIM,
                    "use_object_file_buffer": arm in (ARM_INTACT, ARM_SHUFFLE),
                    "continuity_radius": CONTINUITY_RADIUS, "n_trials": n_trials,
                },
                script_path=Path(__file__),
                extra_ineligible_reasons=["controlled_synthetic_probe"],
            ) as cell:
                row = _run_cell(arm, seed, n_trials)
                cell.stamp(row)
            rows.append(row)
            # per-cell progress verdict: did the cell meet its arm-local expectation?
            if arm == ARM_INTACT:
                cell_ok = row["reid_frac"] >= C1_REID_FLOOR
            else:
                cell_ok = row["reid_frac"] < C1_REID_FLOOR  # ablations should NOT re-identify
            print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    by = {(r["arm"], r["seed"]): r for r in rows}

    # ---- readiness / non-vacuity gates (memo Section 6.2) -- ALL before scoring.
    g0_by_seed = {s: by[(ARM_INTACT, s)]["maintained_frac"] >= 0.999 for s in seeds}
    g1_by_seed = {s: by[(ARM_INTACT, s)]["d_cos_td_mean"] >= G1_DCOS_FLOOR for s in seeds}
    g2_by_seed = {s: by[(ARM_INTACT, s)]["moved_frac"] >= 0.999 for s in seeds}
    g0 = _frac_seeds(g0_by_seed) >= MIN_SEEDS_2OF3
    g1 = _frac_seeds(g1_by_seed) >= MIN_SEEDS_2OF3
    g2 = _frac_seeds(g2_by_seed) >= len(seeds)  # every scored trial must have moved
    readiness_met = bool(g0 and g1 and g2)

    g1_dcos_min = min(by[(ARM_INTACT, s)]["d_cos_td_mean"] for s in seeds)

    # ---- discriminators (scored only if readiness met) -- memo Section 6.3.
    c1_by_seed = {s: by[(ARM_INTACT, s)]["reid_frac"] >= C1_REID_FLOOR for s in seeds}
    c2_by_seed = {
        s: (by[(ARM_INTACT, s)]["reid_frac"] > by[(ARM_SHUFFLE, s)]["reid_frac"] + C2_SHUFFLE_MARGIN
            and by[(ARM_INTACT, s)]["reid_frac"] > by[(ARM_OFF, s)]["reid_frac"])
        for s in seeds
    }
    c3_by_seed = {s: by[(ARM_INTACT, s)]["persisted_match_frac"] >= 0.999 for s in seeds}
    c1 = _frac_seeds(c1_by_seed) >= MIN_SEEDS_2OF3
    c2 = _frac_seeds(c2_by_seed) >= MIN_SEEDS_2OF3
    c3 = _frac_seeds(c3_by_seed) >= MIN_SEEDS_2OF3

    reid_intact_mean = sum(by[(ARM_INTACT, s)]["reid_frac"] for s in seeds) / len(seeds)
    reid_shuffle_mean = sum(by[(ARM_SHUFFLE, s)]["reid_frac"] for s in seeds) / len(seeds)
    reid_off_mean = sum(by[(ARM_OFF, s)]["reid_frac"] for s in seeds) / len(seeds)

    # ---- adjudicate (memo Section 6.3 grid).
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
    elif c1 and c2 and c3:
        outcome = "PASS"
        label = "token_buffer_reidentifies_cross_motion"
        evidence_direction = "supports"
    elif (not c1) or (not c2):
        outcome = "FAIL"
        label = "identity_not_continuity_carried"
        evidence_direction = "weakens"
    else:  # only C3 failed
        outcome = "FAIL"
        label = "feature_persistence_failed"
        evidence_direction = "mixed"

    summary = {
        "label": label,
        "evidence_direction": evidence_direction,
        "readiness_met": readiness_met,
        "gate_pass": {"G0_buffer_populated": g0, "G1_feature_nondegenerate": g1,
                      "G2_entity_moved": g2},
        "discriminators": {"C1_reid_primary": c1, "C2_ablation_contrast": c2,
                           "C3_feature_persistence": c3},
        "reid_frac_mean": {"INTACT": reid_intact_mean, "SHUFFLE": reid_shuffle_mean,
                           "OFF": reid_off_mean},
        "g1_dcos_min": g1_dcos_min,
        "overall_pass": outcome == "PASS",
    }

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "G0_intact_maintains_token", "kind": "readiness",
             "description": "INTACT keeps >=1 live token for the target across the move on >=2/3 seeds",
             "measured": _frac_seeds(g0_by_seed), "threshold": MIN_SEEDS_2OF3,
             "control": "INTACT arm, target observed at t0", "met": g0},
            {"name": "G1_feature_term_nondegenerate", "kind": "readiness",
             "description": "min over seeds of mean d_cos(target,distractor) -- the SAME feature "
                            "statistic the re-identification criterion depends on (dim=32 guard)",
             "measured": g1_dcos_min, "threshold": G1_DCOS_FLOOR,
             "control": "two same-type instances with distinct constructed appearances", "met": g1},
            {"name": "G2_entity_moved", "kind": "readiness",
             "description": "target cell at re-observation differs from first observation on every scored trial",
             "measured": _frac_seeds(g2_by_seed), "threshold": len(seeds), "met": g2},
        ],
        "criteria": [
            {"name": "C1_reid_frac_intact_ge_0.6", "load_bearing": True,
             "passed": bool(c1)},
            {"name": "C2_ablation_contrast", "load_bearing": True, "passed": bool(c2)},
            {"name": "C3_feature_persistence", "load_bearing": False, "passed": bool(c3)},
        ],
        "criteria_non_degenerate": {
            "C1": reid_intact_mean != reid_off_mean,
            "C2": (reid_shuffle_mean is not None) and (reid_off_mean is not None),
            "C3": True,
        },
    }

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "config": {
            "seeds": seeds, "n_trials": n_trials, "world_dim": WORLD_DIM,
            "continuity_radius": CONTINUITY_RADIUS, "occlude_ticks": OCCLUDE_TICKS,
            "arms": ARMS,
            "thresholds": {
                "G1_dcos_floor": G1_DCOS_FLOOR, "C1_reid_floor": C1_REID_FLOOR,
                "C2_shuffle_margin": C2_SHUFFLE_MARGIN, "min_seeds_2of3": MIN_SEEDS_2OF3,
            },
        },
        "acceptance_criteria": {
            **summary["gate_pass"], **summary["discriminators"],
            "overall_pass": summary["overall_pass"],
        },
        "interpretation": interpretation,
        "summary": summary,
        "arm_results": rows,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={label}, evidence={evidence_direction})", flush=True)
    print(f"  readiness G0/G1/G2: {summary['gate_pass']} (g1_dcos_min={g1_dcos_min:.3f})", flush=True)
    print(f"  reid_frac mean: {summary['reid_frac_mean']}", flush=True)
    print(f"  discriminators: {summary['discriminators']}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-658 MECH-045 object-file buffer persistence vs ablation"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
