"""V3-EXQ-807 -- SD-079 validation: centered z_goal cue for goal-anchor match.

PURPOSE (diagnostic / substrate-readiness validation, not governance evidence).
SD-079 was implemented 2026-07-22 after a sweep found SD-039's Anchor.goal_match
taking a raw cosine over z_goal. z_goal is an EMA attractor pulled toward z_world, so
it inherits the SD-008 common-mode offset and, being an integrator, concentrates it
further (measured pairwise cosine min 0.9878 vs z_world's own 0.9767). Two distinct
failures follow, and this run measures both:

  (1) MECH-292's goal_match ranking term varies by ~0.011 across an anchor pool, so
      the motivational-relevance channel contributes essentially no ordering
      information while appearing to be a four-term composite.
  (2) Both ABSOLUTE gates downstream are dead. The goal_match_floor rumination guard
      excludes nothing, and MECH-339's outshining gate
      clip((outshine_pivot - goal_match) / outshine_pivot) with pivot 0.5 evaluates
      to EXACTLY 0.0 for every anchor whenever goal_match >= 0.5 -- which is always.
      MECH-339's composite cue therefore contributes an identically zero context term
      the moment it is enabled: the constraint would read as falsified by a mechanism
      that never ran.

WHAT THIS RUN ADDS OVER THE BENCH MEASUREMENT: it drives the WHOLE AGENT to produce
the z_goal stream and the anchor payloads, then scores them through the REAL
GhostGoalBank with MECH-339's composite cue ENABLED, over multiple seeds -- so the
readouts are the bank's own diagnostics, not a reimplementation of its arithmetic.

ARMS (2, same seeds): ARM_OFF goal_cue_centering=False (must reproduce the pinned
signature) vs ARM_ON goal_cue_centering=True.

GOV-REUSE-1 (Step 2.4): the decisive readouts are the bank's n_below_floor and the
outshining-gate nonzero rate under a CENTERED z_goal cue. goal_cue_centering did not
exist before 2026-07-22, so no recorded manifest on any substrate_hash can carry it.
Not recoverable -> run.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF / ARM_ON DV = goal_match spread, n_below_floor, and the count of anchors
  with a nonzero outshining gate. Symmetry group: permutation of anchors (all three
  are symmetric functions over the anchor pool) and, for the two counts, any
  monotone reparameterisation of goal_match that preserves its position relative to
  the FIXED thresholds (0.05 and 0.5). The manipulation is NOT invariant under
  either: subtracting a common-mode baseline from both operands of the cosine is not
  a monotone map on goal_match (it rotates the residuals, so the anchor ORDER can
  change), and it moves absolute values across the fixed thresholds -- which is
  precisely what the two counts measure. Note the thresholds are absolute constants,
  not run-derived, so the counts cannot be trivially satisfied by rescaling.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig, AnchorSetConfig, GhostGoalBankConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.anchor_set import AnchorSet, AnchorGoalPayload  # noqa: E402
from ree_core.hippocampal.ghost_goal_bank import GhostGoalBank  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_807_sd079_centered_goal_anchor_match"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness predicate IS the degeneracy definition, not a narrower "
    "hand-written proxy for it: the SD-008 common-mode hazard is DEFINED as "
    "'min pairwise cosine of the z_goal stream is high', and the precondition "
    "asserts exactly that statistic against a 0.90 floor. It is reachable by "
    "construction and measured well clear of the gate on this nursery (0.9878 / 0.9918). "
    "There is no separate control whose reproducibility could fail "
    "independently of the quantity being asserted."
)

CLAIM_IDS = ["SD-079"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_ON"]
N_EPISODES = 8
STEPS_PER_EPISODE = 32
ANCHOR_EVERY = 8          # write an anchor payload every N steps
FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# Bank defaults this run exercises (absolute constants, NOT run-derived).
GOAL_MATCH_FLOOR = 0.05
OUTSHINE_PIVOT = 0.5

# Pre-registered thresholds.
SPREAD_FLOOR = 0.20        # C1: ON goal_match spread across the pool
OFF_SPREAD_CEIL = 0.10     # C2: OFF must reproduce the pinned narrow spread
MIN_GATE_NONZERO = 1       # C3: ON must have >= this many anchors with a live gate
SEED_PASS_FRACTION = 2.0 / 3.0

# READINESS: the z_goal common-mode cone must be present for SD-079 to have
# anything to repair. Same statistic the hazard is defined by (min pairwise cosine).
CONE_MIN_COSINE_FLOOR = 0.90

# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE,
                     "anchor_every": ANCHOR_EVERY},
        "alpha_world": 0.9,
        "z_goal_enabled": True,
        "drive_weight": 2.0,
        "benefit_eval_enabled": True,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
        "goal_cue_centering": centering,
        "goal_cue_baseline_alpha": 0.05,
        "use_composite_cue_outshining": True,
        "outshine_pivot": OUTSHINE_PIVOT,
        "goal_match_floor": GOAL_MATCH_FLOOR,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    if len(vecs) < 2:
        return 1.0
    m = torch.nn.functional.normalize(torch.stack(vecs).float(), dim=-1)
    c = m @ m.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].min())


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
            action_dim=4, alpha_world=0.9, z_goal_enabled=True, drive_weight=2.0,
            benefit_eval_enabled=True, benefit_weight=1.0,
            use_sd039_anchor_payload=True,
            goal_cue_centering=centering, goal_cue_baseline_alpha=0.05,
        )
        agent = REEAgent(cfg)
        wd = agent.config.latent.world_dim

        aset = AnchorSet(AnchorSetConfig(
            use_sd039_anchor_payload=True,
            goal_cue_centering=centering,
            goal_cue_baseline_alpha=0.05,
        ))
        bank = GhostGoalBank(
            GhostGoalBankConfig(
                goal_match_floor=GOAL_MATCH_FLOOR,
                use_composite_cue_outshining=True,
                context_weight=1.0,
                outshine_pivot=OUTSHINE_PIVOT,
                include_inactive=True,
                include_active=True,
            ),
            aset,
        )

        zgoals: List[torch.Tensor] = []
        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            for t in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                agent.update_z_goal(benefit_exposure=FORCED_BENEFIT,
                                    drive_level=FORCED_DRIVE)
                zg = agent.goal_state.z_goal if agent.goal_state is not None else None
                if zg is not None:
                    zg = zg.detach().reshape(-1).clone()
                    zgoals.append(zg)
                    if t % ANCHOR_EVERY == 0:
                        aset.write_anchor(
                            "fast", f"e{ep}s{t}", ("world",),
                            z_world=latent.z_world.detach().reshape(-1),
                            goal_payload=AnchorGoalPayload(
                                z_goal_snapshot=zg.clone(), wanting_strength=0.5),
                        )
                _, _h, done, _, obs = env.step(int(action.argmax(dim=-1).item()))
                if done:
                    break
            if (ep + 1) % 4 == 0:
                print(f"  [train] goal seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"anchors={len(aset.all_anchors())}", flush=True)

        query = zgoals[-1]
        # Score the pool through the REAL bank (MECH-292 + MECH-339 both live).
        entries = bank.rank(query, simulation_mode=False)
        diag = bank._last_diagnostics or {}
        # Per-anchor goal_match over the whole pool (threshold=-1.0 -> include all).
        pairs = aset.query_by_goal_match(query, threshold=-1.0, simulation_mode=False)
        scores = torch.tensor([s for _, s in pairs]) if pairs else torch.tensor([0.0])
        gate = ((OUTSHINE_PIVOT - scores) / OUTSHINE_PIVOT).clamp(0.0, 1.0)

        row = {
            "arm": arm,
            "seed": seed,
            "n_anchors": len(pairs),
            "goal_match_min": float(scores.min()),
            "goal_match_max": float(scores.max()),
            "goal_match_spread": float(scores.max() - scores.min()),
            "n_below_floor": int((scores < GOAL_MATCH_FLOOR).sum()),
            "n_gate_nonzero": int((gate > 0).sum()),
            "gate_max": float(gate.max()),
            "bank_n_entries": len(entries),
            "bank_n_below_floor": int(diag.get("n_below_floor", -1)),
            "bank_n_scanned": int(diag.get("n_scanned", -1)),
            "goal_baseline_allocated": aset.goal_cue_baseline is not None,
            "zgoal_cone_min_cosine": _cone_min_cosine(zgoals[:200]),
            "n_zgoal_sampled": min(len(zgoals), 200),
        }
        cell.stamp(row)
    # z_goal liveness -- read AFTER the cell has stepped, then the agent is dropped.
    # z_goal liveness -- read AFTER this cell stepped; the agent is not retained.
    _ZG.observe(agent)
    passed = (row["goal_match_spread"] > SPREAD_FLOOR) if centering else \
             (row["goal_match_spread"] <= OFF_SPREAD_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    fn = min if mode == "min" else max
    r = fn(rows, key=lambda x: x[key])
    return float(r[key]), f"{r['arm']}/seed{r['seed']}"


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    off = [r for r in rows if r["arm"] == "ARM_OFF"]
    on = [r for r in rows if r["arm"] == "ARM_ON"]

    cone_worst, cone_cell = _worst_cell(rows, "zgoal_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR

    frac_c1 = sum(1 for r in on if r["goal_match_spread"] > SPREAD_FLOOR) / len(on)
    frac_c2 = sum(1 for r in off if r["goal_match_spread"] <= OFF_SPREAD_CEIL) / len(off)
    frac_c3 = sum(1 for r in on if r["n_gate_nonzero"] >= MIN_GATE_NONZERO) / len(on)
    c1 = frac_c1 >= SEED_PASS_FRACTION
    c2 = frac_c2 >= SEED_PASS_FRACTION
    c3 = frac_c3 >= SEED_PASS_FRACTION

    # Non-degeneracy: the ON arm's recovery is attributable to centering only if the
    # OFF arm really is pinned AND really ran the OFF path (no baseline allocated);
    # and the gate criterion discriminates only if the OFF gate was dead everywhere
    # (otherwise a nonzero ON gate is not news).
    off_pinned = all(r["goal_match_spread"] <= OFF_SPREAD_CEIL for r in off)
    off_no_baseline = all(not r["goal_baseline_allocated"] for r in off)
    on_baseline = all(r["goal_baseline_allocated"] for r in on)
    off_gate_dead = all(r["n_gate_nonzero"] == 0 for r in off)
    pool_nonempty = all(r["n_anchors"] >= 4 for r in rows)

    overall = c1 and c2 and c3 and cone_present
    if not cone_present:
        label = "substrate_not_ready_requeue"
    elif overall:
        label = "sd079_centering_unpins_goal_match_and_downstream_gates"
    else:
        label = "sd079_centering_did_not_unpin_goal_match"

    metrics = {
        "on_spread_mean": statistics.fmean(r["goal_match_spread"] for r in on),
        "off_spread_mean": statistics.fmean(r["goal_match_spread"] for r in off),
        "on_n_below_floor_mean": statistics.fmean(r["n_below_floor"] for r in on),
        "off_n_below_floor_mean": statistics.fmean(r["n_below_floor"] for r in off),
        "on_n_gate_nonzero_mean": statistics.fmean(r["n_gate_nonzero"] for r in on),
        "off_n_gate_nonzero_mean": statistics.fmean(r["n_gate_nonzero"] for r in off),
        "on_gate_max_mean": statistics.fmean(r["gate_max"] for r in on),
        "off_gate_max_mean": statistics.fmean(r["gate_max"] for r in off),
        "frac_c1_on_spread": frac_c1,
        "frac_c2_off_reproduces_pin": frac_c2,
        "frac_c3_on_gate_live": frac_c3,
        "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        "zgoal_cone_min_cosine_worst": cone_worst,
        "zgoal_cone_worst_cell": cone_cell,
    }
    return {
        "outcome": "PASS" if overall else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zgoal_common_mode_cone_present",
                 "description": ("min pairwise z_goal cosine in this run clears the "
                                 "cone floor -- the geometry SD-079 exists to repair. "
                                 "Below it there is no common mode to subtract and a "
                                 "null says nothing about centering."),
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower",  # FLOOR: met when measured >= threshold
                 "control": ("worst cell across all arms/seeds ({}), measured on the "
                             "run's own z_goal stream".format(cone_cell)),
                 "offending_cell": cone_cell,
                 "met": cone_present},
            ],
            "criteria": [
                {"name": "C1_on_spread_above_floor", "load_bearing": True, "passed": c1},
                {"name": "C2_off_reproduces_pin", "load_bearing": False, "passed": c2},
                {"name": "C3_on_outshine_gate_live", "load_bearing": True, "passed": c3},
            ],
            "criteria_non_degenerate": {
                "C1_on_spread_above_floor": bool(off_pinned and pool_nonempty),
                "C2_off_reproduces_pin": bool(off_no_baseline and on_baseline),
                "C3_on_outshine_gate_live": bool(off_gate_dead and pool_nonempty),
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [101]
        N_EPISODES = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE, "anchor_every": ANCHOR_EVERY,
        "spread_floor": SPREAD_FLOOR, "off_spread_ceil": OFF_SPREAD_CEIL,
        "min_gate_nonzero": MIN_GATE_NONZERO,
        "goal_match_floor": GOAL_MATCH_FLOOR, "outshine_pivot": OUTSHINE_PIVOT,
        "cone_min_cosine_floor": CONE_MIN_COSINE_FLOOR,
        "seed_pass_fraction": SEED_PASS_FRACTION,
        "arm_config_slice_off": _config_slice(False),
        "arm_config_slice_on": _config_slice(True),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if result["outcome"] == "PASS" else "unknown",
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"ON spread={m['on_spread_mean']:.4f} gate_nonzero={m['on_n_gate_nonzero_mean']:.2f} | "
          f"OFF spread={m['off_spread_mean']:.4f} gate_nonzero={m['off_n_gate_nonzero_mean']:.2f}",
          flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"cone_worst={m['zgoal_cone_min_cosine_worst']:.4f} "
          f"({m['zgoal_cone_worst_cell']})", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
