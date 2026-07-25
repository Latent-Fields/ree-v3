"""V3-EXQ-806 -- SD-078 validation: centered CandidateRuleField context key.

PURPOSE (diagnostic / substrate-readiness validation, not governance evidence).
SD-078 was implemented 2026-07-22 after a sweep found the ARC-063 CandidateRuleField
keying mint-block, recurrence bucketing and gate retrieval on a raw z_world context.
Under SD-008 z_world under-differentiation every context sits in a ~0.98-cosine
common-mode cone, so the mint-block fires against the FIRST minted rule for every
later context at any threshold the config can express: the pool is structurally
capped at ONE rule and crf_max_pairwise_rule_dist == 0.0 is a TAUTOLOGY (the metric
needs two concurrent rules) rather than the retire-churn symptom the V3-EXQ-654b /
654d autopsies read it as.

WHAT THIS RUN ADDS OVER THE BENCH MEASUREMENT. The bench measurement that motivated
SD-078 drove the CandidateRuleField module directly over a recorded context stream.
This runs the WHOLE AGENT with the field wired through agent.py's own config build
and the lateral-PFC consumer attached, over multiple seeds, so the readouts are the
ones the ARC-063 GAP-B readiness gate actually reads.

FAILURE RECORD THIS MUST MOVE (the acceptance criteria are taken from it):
  V3-EXQ-654  / 654a / 654b / 654d -- crf_max_pairwise_rule_dist == 0.0 in every
  ARM_ON cell, crf_frac_active ~0.12-0.13 against a 0.30 floor, cumulative
  n_minted 131-408 (mint -> brief life -> retire -> re-mint churn).
A working SD-078 must move crf_max_pairwise_rule_dist off zero and hold >= 2
concurrent rules. It is NOT expected to clear the 0.30 frac_active floor -- that is
the separate maintenance/gate-dynamics question the 654b/654d machinery addresses,
and this run deliberately does not claim it.

ARMS (2, same seeds): ARM_OFF crf_cue_centering=False (must reproduce the pinned
signature exactly) vs ARM_ON crf_cue_centering=True.

NOTE ON THE TWO EXISTING MITIGATIONS. Both 654b-amend mitigations are measured
ineffective and are left at their current values in BOTH arms, so this run isolates
centering alone: mature_mint_block_threshold=0.8 cannot clear a 0.9426 floor, and
crf_context_from_e2_world_forward routes to a context carrying the same offset.
Neither is the manipulation here.

GOV-REUSE-1 (Step 2.4): the decisive readout is crf_max_pairwise_rule_dist under a
CENTERED context key. crf_cue_centering did not exist before 2026-07-22, so no
recorded manifest on any substrate_hash can carry it -- checked the 654/654a/654b/654d
manifests, which carry the readout but only on the raw key. Not recoverable -> run.

DV-SYMMETRY (Step 3 mandatory declaration, per arm):
  ARM_OFF / ARM_ON DV = crf_max_pairwise_rule_dist and max concurrent live rules.
  Symmetry group of that DV: permutation of rule slots (the distance is a symmetric
  function over the rule set) and any relabelling of slot indices. The manipulation
  (subtracting a per-tick common-mode baseline before each cue comparison) is NOT
  invariant under it: centering changes WHICH contexts pass the mint-block, hence the
  CARDINALITY of the rule set, and cardinality is not preserved by any slot
  permutation. It is also not a broadcast constant on a rank readout -- the baseline
  is subtracted from both operands of a cosine, which is not an order-preserving map
  on cosine (it rotates the residuals), so the mint-block's accept/reject partition
  genuinely changes.
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

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_806_sd078_centered_rule_field_context_key"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "The readiness predicate IS the degeneracy definition, not a narrower "
    "hand-written proxy for it: the SD-008 common-mode hazard is DEFINED as "
    "'min pairwise cosine of the z_world stream is high', and the precondition "
    "asserts exactly that statistic against a 0.90 floor. It is reachable by "
    "construction and measured well clear of the gate on this nursery (0.9767 / 0.9732). "
    "There is no separate control whose reproducibility could fail "
    "independently of the quantity being asserted."
)

CLAIM_IDS = ["SD-078"]

SEEDS = [101, 202, 303]
ARMS = ["ARM_OFF", "ARM_ON"]
N_EPISODES = 12
STEPS_PER_EPISODE = 32

# Pre-registered thresholds (defined HERE, never inferred post-hoc).
DIST_FLOOR = 0.05          # C1: ON arm crf_max_pairwise_rule_dist must exceed this
MIN_LIVE_RULES = 2         # C2: ON arm must hold >= this many concurrent rules
OFF_DIST_CEIL = 1e-9       # C3: OFF arm must reproduce the pinned dist == 0.0
SEED_PASS_FRACTION = 2.0 / 3.0

# READINESS: the common-mode cone must actually be present in this run's z_world.
# If a future encoder change removes it, SD-078 has nothing to repair here and the
# run must self-route substrate_not_ready_requeue rather than report a ceiling.
CONE_MIN_COSINE_FLOOR = 0.90


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                             use_proxy_fields=True, seed=seed)


def _build_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,  # SD-008: z_world fidelity
        use_candidate_rule_field=True,
        use_lateral_pfc_analog=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_cue_centering=centering,
        crf_cue_baseline_alpha=0.02,
    ))


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": {"size": 8, "num_hazards": 0, "num_resources": 6,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "use_candidate_rule_field": True,
        "use_lateral_pfc_analog": True,
        "crf_persist_rules_across_episode_reset": True,
        "crf_mature_pool_dynamics": True,
        "crf_cue_centering": centering,
        "crf_cue_baseline_alpha": 0.02,
    }


def _cone_min_cosine(vecs: List[torch.Tensor]) -> float:
    """Min pairwise cosine of the collected z_world stream (the cone measurement)."""
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
        agent = _build_agent(env, centering)
        crf = agent.candidate_rule_field
        wd = agent.config.latent.world_dim
        max_live = 0
        zworlds: List[torch.Tensor] = []
        active_ticks = 0
        total_ticks = 0
        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            for _ in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                cands = agent.generate_trajectories(latent, e1, ticks)
                action = agent.select_action(cands, ticks)
                if len(zworlds) < 200:
                    zworlds.append(latent.z_world.detach().reshape(-1).clone())
                max_live = max(max_live, len(crf._rules))
                total_ticks += 1
                if crf.n_active_rules() > 0:
                    active_ticks += 1
                _, _h, done, _, obs = env.step(int(action.argmax(dim=-1).item()))
                if done:
                    break
            if (ep + 1) % 4 == 0:
                print(f"  [train] crf seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                      f"live={len(crf._rules)} minted={crf._n_minted}", flush=True)
        row = {
            "arm": arm,
            "seed": seed,
            "crf_max_pairwise_rule_dist": float(crf.max_pairwise_rule_distance()),
            "crf_max_live_rules": int(max_live),
            "crf_live_rules_final": int(len(crf._rules)),
            "crf_n_minted": int(crf._n_minted),
            "crf_n_retired": int(crf._n_retired),
            "crf_frac_active": (float(active_ticks) / total_ticks) if total_ticks else 0.0,
            "crf_baseline_allocated": crf._baseline is not None,
            "zworld_cone_min_cosine": _cone_min_cosine(zworlds),
            "n_zworld_sampled": len(zworlds),
        }
        cell.stamp(row)
    passed = (row["crf_max_pairwise_rule_dist"] > DIST_FLOOR) if centering else \
             (row["crf_max_pairwise_rule_dist"] <= OFF_DIST_CEIL)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    """Extremum plus the offending cell id -- never a mean (recomputability rule)."""
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

    # READINESS precondition: the cone this SD repairs must be present. The statistic
    # is the SAME one the hazard is defined by (min pairwise cosine), measured on the
    # run's own z_world stream -- not a magnitude proxy. Worst cell, with its id.
    cone_worst, cone_cell = _worst_cell(rows, "zworld_cone_min_cosine", mode="min")
    cone_present = cone_worst >= CONE_MIN_COSINE_FLOOR

    frac_c1 = sum(1 for r in on if r["crf_max_pairwise_rule_dist"] > DIST_FLOOR) / len(on)
    frac_c2 = sum(1 for r in on if r["crf_max_live_rules"] >= MIN_LIVE_RULES) / len(on)
    frac_c3 = sum(1 for r in off if r["crf_max_pairwise_rule_dist"] <= OFF_DIST_CEIL) / len(off)
    c1 = frac_c1 >= SEED_PASS_FRACTION
    c2 = frac_c2 >= SEED_PASS_FRACTION
    c3 = frac_c3 >= SEED_PASS_FRACTION

    # Non-degeneracy: C1/C2 discriminate only if the OFF arm is genuinely pinned
    # (otherwise ON "recovering" range is not attributable to centering), and C3
    # only if the OFF arm allocated no baseline (the OFF path really was OFF).
    off_pinned = all(r["crf_max_pairwise_rule_dist"] <= OFF_DIST_CEIL for r in off)
    off_no_baseline = all(not r["crf_baseline_allocated"] for r in off)
    on_baseline = all(r["crf_baseline_allocated"] for r in on)

    overall = c1 and c2 and c3 and cone_present
    if not cone_present:
        label = "substrate_not_ready_requeue"
    elif overall:
        label = "sd078_centering_recovers_differentiated_rule_pool"
    else:
        label = "sd078_centering_did_not_recover_rule_pool"

    metrics = {
        "on_dist_mean": statistics.fmean(r["crf_max_pairwise_rule_dist"] for r in on),
        "off_dist_mean": statistics.fmean(r["crf_max_pairwise_rule_dist"] for r in off),
        "on_max_live_mean": statistics.fmean(r["crf_max_live_rules"] for r in on),
        "off_max_live_mean": statistics.fmean(r["crf_max_live_rules"] for r in off),
        "on_frac_active_mean": statistics.fmean(r["crf_frac_active"] for r in on),
        "off_frac_active_mean": statistics.fmean(r["crf_frac_active"] for r in off),
        "frac_c1_dist_above_floor": frac_c1,
        "frac_c2_min_live_rules": frac_c2,
        "frac_c3_off_reproduces_pin": frac_c3,
        "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        "zworld_cone_min_cosine_worst": cone_worst,
        "zworld_cone_worst_cell": cone_cell,
    }
    return {
        "outcome": "PASS" if overall else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "zworld_common_mode_cone_present",
                 "description": ("min pairwise z_world cosine in this run clears the "
                                 "cone floor -- the geometry SD-078 exists to repair. "
                                 "Below it, there is no common mode to subtract and a "
                                 "null says nothing about centering."),
                 "measured": cone_worst, "threshold": CONE_MIN_COSINE_FLOOR,
                 "direction": "lower",  # FLOOR: met when measured >= threshold
                 "control": ("worst cell across all arms/seeds ({}), measured on the "
                             "run's own z_world stream".format(cone_cell)),
                 "offending_cell": cone_cell,
                 "met": cone_present},
            ],
            "criteria": [
                {"name": "C1_on_dist_above_floor", "load_bearing": True, "passed": c1},
                {"name": "C2_on_holds_two_rules", "load_bearing": True, "passed": c2},
                {"name": "C3_off_reproduces_pin", "load_bearing": False, "passed": c3},
            ],
            "criteria_non_degenerate": {
                "C1_on_dist_above_floor": bool(off_pinned),
                "C2_on_holds_two_rules": bool(off_pinned),
                "C3_off_reproduces_pin": bool(off_no_baseline and on_baseline),
            },
            "note": ("Does NOT claim the 0.30 crf_frac_active floor from the 654b/654d "
                     "readiness gate -- that is the separate maintenance/gate-dynamics "
                     "question. frac_active is reported for both arms as context only."),
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
        "steps_per_episode": STEPS_PER_EPISODE,
        "dist_floor": DIST_FLOOR, "min_live_rules": MIN_LIVE_RULES,
        "off_dist_ceil": OFF_DIST_CEIL,
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
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"ON dist={m['on_dist_mean']:.4f} live={m['on_max_live_mean']:.2f} | "
          f"OFF dist={m['off_dist_mean']:.4f} live={m['off_max_live_mean']:.2f}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"cone_worst={m['zworld_cone_min_cosine_worst']:.4f} "
          f"({m['zworld_cone_worst_cell']})", flush=True)
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
