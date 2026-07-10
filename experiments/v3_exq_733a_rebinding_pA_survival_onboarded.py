#!/opt/local/bin/python3
"""
V3-EXQ-733a-a -- Leg P-A of the MECH-456 functional-rebinding redesign portfolio.
TRAINING-REGIME AXIS: run the V3-EXQ-733 functional test on a SURVIVAL-ONBOARDED
agent so P1 episodes survive long enough to accrue the pre-registered overtake
quorum (>= MIN_OVERTAKES_P1 per seed on >= MIN_SEEDS_FOR_PASS of 6 seeds).

supersedes: V3-EXQ-733 (same scientific question -> letter suffix; GOV-FANOUT-1
2-leg portfolio with the sibling P-B directed-traversal leg
v3_exq_733a_rebinding_pB_directed_traversal.py).

WHY 733 FAILED (not a rebinding verdict). V3-EXQ-733 self-routed
substrate_not_ready_requeue: only seed 42 reached the 20-overtake floor (min 2 vs
20). Root cause (failure_autopsy_V3-EXQ-733_2026-07-10): a COLD, un-onboarded agent
dies to hazards in ~10-40 steps in the lethal SD-054 reef-bipartite env
(done=health<=0; episodes capped at 120<500), so within-episode ground-truth region
overtakes are starved. The binder itself is fine (DV1 6/6, DV2 5/6; seed 42 strong
on both). So the block is a TRAINING-REGIME / test-bed gap, NOT a rebinding failure.

THE P-A FIX (this leg). Before running the functional test, ONBOARD the agent for
SURVIVAL using the validated `scaffolded_sd054_onboarding` curriculum (the exact
V3-EXQ-603m/603n recipe that reached P1 survival 3/3). The curriculum trains
E1/E2/harm-valuation/avoidance across Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1;
it does NOT touch the cross_stream_binder (confirmed: zero references in the
scheduler), so the binder is still trained fresh during the functional test's own
P0 curriculum -- exactly as in 733. The ONLY change vs 733 is the survival
onboarding, so this is a clean single-axis manipulation.

NULL (declared): even survival-onboarded, if overtakes stay < MIN_OVERTAKES_P1/seed
on >= MIN_SEEDS_FOR_PASS seeds -> self-routes substrate_not_ready_requeue
(non_contributory), meaning survival is NOT the (only) lever (points to the P-B
test-bed axis). A non_contributory result is INFORMATIVE here, not wasted.

WHAT THIS TESTS (MECH-456 what_would_answer, via the shared 733 harness):
  DV1: rebinding tracks the TRUE current region above a region-label-shuffle
       control (alignment_real >= alignment_shuffle + ALIGN_MARGIN).
  DV2: rebinding-ON re-acquires the correct binding FASTER than a rebinding-FROZEN
       arm on a graded, non-saturating re-acquisition-latency metric
       (mean_lat_frozen - mean_lat_on >= LATENCY_MARGIN).
RUN PASS = DV1 AND DV2 on >= MIN_SEEDS_FOR_PASS seeds with readiness met. On
readiness-unmet -> substrate_not_ready_requeue / non_contributory (NOT a verdict).
MECH-456 stays candidate / v3_pending regardless (V3-pending gate).

NO ree_core change -- harness-level measurement over the trained-then-frozen learned
cross_stream_binder. Env / agent / measurement primitives are shared with 733 via
experiments/_lib/rebinding_functional_harness.py.

Claims: [MECH-456] (experiment_purpose=evidence; claim-tagged functional test).
Bears on (cited, NOT tagged): MECH-269, MECH-270, ARC-006, MECH-045, INV-002.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._metrics import check_degeneracy
from experiments._lib import rebinding_functional_harness as H
from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env as build_scaffold_env,
)
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_733a_rebinding_pA_survival_onboarded"
QUEUE_ID = "V3-EXQ-733a-a"
CLAIM_IDS: List[str] = ["MECH-456"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-733"

# Converged learned cross_stream_binder activation (carried from V3-EXQ-733 / 725a;
# NOT tuned to a pass here).
CROSS_STREAM_BINDING_ENABLED = True
CROSS_STREAM_BINDING_LEARNED = True
CROSS_STREAM_BINDING_STRENGTH = 0.5
CROSS_STREAM_BINDING_TEMPERATURE = 0.2
CROSS_STREAM_BINDING_CONV_FRAC = 0.85

# Ground-truth lattice (same G as 733 -- single-variable change = survival onboarding).
G_PARTITION = 2
K_REGIONS = G_PARTITION * G_PARTITION

SEEDS = [42, 43, 44, 45, 46, 47]

# Functional-test schedule (identical to V3-EXQ-733; SEPARATE from the onboarding
# curriculum that precedes it). P0 trains the binder + builds prototypes; P1 measures.
P0_WARMUP_EPISODES = 40
P1_MEASUREMENT_EPISODES = 25
FUNC_STEPS_PER_EPISODE = 120

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 3
DRY_RUN_P1 = 3
DRY_RUN_STEPS = 40

# --- Survival-onboarding recipe (VERBATIM V3-EXQ-603m/603n; reached P1 3/3) ------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
CUE_RECALL_GAIN = 0.2
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

ONBOARD_STAGE0_BUDGET = 20
ONBOARD_STAGE0B_BUDGET = 10
ONBOARD_P0_BUDGET = 100
ONBOARD_HAZARD_BUDGET = 40
ONBOARD_P1_BUDGET = 50
ONBOARD_P2_BUDGET = 15
ONBOARD_TRAIN_STEPS = 200

DRY_ONBOARD_STAGE0 = 2
DRY_ONBOARD_STAGE0B = 1
DRY_ONBOARD_P0 = 2
DRY_ONBOARD_HAZARD = 2
DRY_ONBOARD_P1 = 2
DRY_ONBOARD_P2 = 1
DRY_ONBOARD_STEPS = 30


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The exact V3-EXQ-603m/603n curriculum config (P1 survival 3/3)."""
    steps = DRY_ONBOARD_STEPS if dry_run else ONBOARD_TRAIN_STEPS
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=(DRY_ONBOARD_STAGE0 if dry_run else ONBOARD_STAGE0_BUDGET),
        scaffold_p0_episode_budget=(DRY_ONBOARD_P0 if dry_run else ONBOARD_P0_BUDGET),
        scaffold_p1_episode_budget=(DRY_ONBOARD_P1 if dry_run else ONBOARD_P1_BUDGET),
        scaffold_p2_episode_budget=(DRY_ONBOARD_P2 if dry_run else ONBOARD_P2_BUDGET),
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=1,
        scaffold_p1_anneal_hold_fraction=0.3,
        scaffold_p2_hazard_food_attraction_guard=0.3,
        # developmental-window / consolidation
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=(DRY_ONBOARD_STAGE0B if dry_run else ONBOARD_STAGE0B_BUDGET),
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        # 634c seeding calibration
        scaffold_z_goal_seeding_gain=1.5,
        scaffold_benefit_threshold=0.02,
        scaffold_drive_floor=0.9,
        # foraging-competence residual
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=0.4,
        # SD-057 cue-recall bridge
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=3,
        scaffold_stage0_bind_incentive_token=True,
        # curriculum-decomposition: isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=(DRY_ONBOARD_HAZARD if dry_run else ONBOARD_HAZARD_BUDGET),
        scaffold_hazard_stage_num_hazards=4,
        scaffold_hazard_stage_num_resources=2,
        scaffold_hazard_stage_hazard_food_attraction=0.0,
        scaffold_hazard_stage_proximity_harm_scale=0.1,
        scaffold_hazard_stage_spawn_in_reef_half=True,
        scaffold_hazard_stage_survival_gate_steps=75,
        scaffold_hazard_stage_stability_window=10,
        # SD-058 / MECH-357 avoidance-learning driver
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=0.8,
        scaffold_avoidance_scaffold_floor_end=0.0,
        # feed the env harm stream so z_harm / z_harm_a populate
        scaffold_feed_harm_stream=True,
        # harm-pathway training (the 2026-06-09 amend that made survival learnable)
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=1e-3,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_agent_config(env) -> REEConfig:
    """Union of the V3-EXQ-603m/603n onboarding stack AND the V3-EXQ-733 converged
    learned cross_stream_binder. The curriculum needs the onboarding substrate; the
    functional test needs the binder. The curriculum never touches the binder, so
    both coexist cleanly (dims fixed by the onboarding env)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # z_harm sensory (SD-010) + affective (SD-011)
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # E2_harm_s forward model (ARC-033) so harm-pathway term 4 engages.
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # V3-EXQ-733 converged learned cross_stream_binder (the falsifier substrate).
        cross_stream_binding_enabled=CROSS_STREAM_BINDING_ENABLED,
        cross_stream_binding_learned=CROSS_STREAM_BINDING_LEARNED,
        cross_stream_binding_strength=CROSS_STREAM_BINDING_STRENGTH,
        cross_stream_binding_temperature=CROSS_STREAM_BINDING_TEMPERATURE,
        cross_stream_binding_conv_frac=CROSS_STREAM_BINDING_CONV_FRAC,
    )
    cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _onboard_agent(agent: REEAgent, scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                   device: torch.device) -> Dict[str, Any]:
    """Run the validated survival curriculum on `agent` (Stage-0 -> Stage-0b -> P0
    -> Stage-H -> P1). Returns a survival-readout dict. Aborts short-circuit to a
    survival_onboarded=False readout (the functional test still runs -- readiness
    then reports the honest overtake count)."""
    sched = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    diag: Dict[str, Any] = {"survival_onboarded": False, "aborted_stage": None}
    s0 = sched.run_stage0_nursery(agent, device)
    if s0.aborted:
        diag["aborted_stage"] = "stage0"
        return diag
    s0b = sched.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    if s0b.aborted:
        diag["aborted_stage"] = "stage0b"
        return diag
    p0 = sched.run_p0(agent, device)
    if p0.aborted:
        diag["aborted_stage"] = "p0"
        return diag
    hz = sched.run_hazard_avoidance(agent, device)
    if hz.aborted:
        diag["aborted_stage"] = "hazard"
        return diag
    p1 = sched.run_p1(agent, device)
    diag.update({
        "survival_onboarded": bool(p1.survival_gate_passed),
        "onboard_p1_survival_gate_passed": bool(p1.survival_gate_passed),
        "onboard_p1_median_last_window_len": float(p1.median_last_window_episode_length),
        "onboard_hazard_survival_gate_passed": bool(hz.survival_gate_passed),
        "onboard_p0_mean_episode_length": float(p0.mean_episode_length),
    })
    return diag


def _run_seed(seed: int, p0_episodes: int, p1_episodes: int, steps_per_episode: int,
              dry_run: bool) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    scaffold_cfg = _make_scaffold_cfg(dry_run)

    # Build the agent from the onboarding p2-env dims (the functional-test env is the
    # SAME builder + phase, so dims match by construction).
    probe_env = build_scaffold_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_agent_config(probe_env)).to(device)

    # 1) SURVIVAL ONBOARDING (the P-A manipulation).
    onboard = _onboard_agent(agent, scaffold_cfg, device)

    # 2) FUNCTIONAL TEST on a dim-matched env (agent is now survival-competent, so
    #    P1 episodes should survive long enough to accrue overtakes naturally).
    def env_factory(s: int):
        env = build_scaffold_env(scaffold_cfg, "p2", seed=s)
        # Defensive parity guard: functional-test env dims MUST match the agent's.
        if (int(env.body_obs_dim) != int(probe_env.body_obs_dim)
                or int(env.world_obs_dim) != int(probe_env.world_obs_dim)):
            raise RuntimeError(
                f"env dim parity FAILED seed={s}: "
                f"func(body={env.body_obs_dim},world={env.world_obs_dim}) != "
                f"probe(body={probe_env.body_obs_dim},world={probe_env.world_obs_dim})"
            )
        return env

    row = H.run_functional_test_seed(
        seed=seed,
        agent=agent,
        env_factory=env_factory,
        g_partition=G_PARTITION,
        p0_episodes=p0_episodes,
        p1_episodes=p1_episodes,
        steps_per_episode=steps_per_episode,
        spawn_director=None,   # P-A: natural traversal by the survival-competent agent
        step_director=None,
    )
    row.update(onboard)
    return row


def run_experiment(seeds: List[int], p0_episodes: int, p1_episodes: int,
                   steps_per_episode: int, dry_run: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    first = True
    for s in seeds:
        print(f"Seed {s} Condition rebinding_pA_onboarded", flush=True)
        if first:
            print(
                f"  (onboard curriculum then func-test P0={p0_episodes} ep, "
                f"P1={p1_episodes} ep, steps={steps_per_episode}, K_regions={K_REGIONS}, "
                f"align_margin={H.ALIGN_MARGIN}, latency_margin={H.LATENCY_MARGIN}, "
                f"dry_run={dry_run})",
                flush=True,
            )
            first = False
        row = _run_seed(s, p0_episodes, p1_episodes, steps_per_episode, dry_run)
        rows.append(row)
        verdict = "PASS" if row["error_note"] is None else "FAIL"
        print(f"verdict: {verdict}", flush=True)

    interp = H.interpret(rows, conv_frac=CROSS_STREAM_BINDING_CONV_FRAC)
    ok = [r for r in rows if r["error_note"] is None]
    n_completed = len(ok)

    passed = bool(
        n_completed >= H.MIN_SEEDS_COMPLETED
        and interp["label"] == "functional_rebinding_supported"
    )

    degen = check_degeneracy({
        "alignment_margin": [r["alignment_margin_obs"] for r in ok] or [0.0, 0.0],
        "latency_gap": [r["latency_gap_obs"] for r in ok] or [0.0, 0.0],
    })

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": list(seeds),
        "n_completed": int(n_completed),
        "n_total_runs": int(len(seeds)),
        "min_seeds_completed": int(H.MIN_SEEDS_COMPLETED),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "k_regions": int(K_REGIONS),
        "g_partition": int(G_PARTITION),
        "n_seeds_survival_onboarded": int(sum(1 for r in ok if r.get("survival_onboarded"))),
        "acceptance_thresholds": {
            "align_margin": float(H.ALIGN_MARGIN),
            "latency_margin": float(H.LATENCY_MARGIN),
            "reacq_window": int(H.REACQ_WINDOW),
            "n_shuffle_perms": int(H.N_SHUFFLE_PERMS),
            "min_seeds_for_pass": int(H.MIN_SEEDS_FOR_PASS),
            "min_region_visits_p0": int(H.MIN_REGION_VISITS_P0),
            "min_overtakes_p1": int(H.MIN_OVERTAKES_P1),
        },
        "binder_config": {
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
            "cross_stream_binding_strength": float(CROSS_STREAM_BINDING_STRENGTH),
            "cross_stream_binding_temperature": float(CROSS_STREAM_BINDING_TEMPERATURE),
            "cross_stream_binding_conv_frac": float(CROSS_STREAM_BINDING_CONV_FRAC),
        },
        "per_seed_results": rows,
        "interpretation": interp,
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
        "degenerate_metrics": degen["degenerate_metrics"],
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    interp = result.get("interpretation", {})
    label = interp.get("label", "")
    direction, direction_note = H.evidence_direction(label)

    review_caveats = [
        "Leg P-A (training-regime axis) of the V3-EXQ-733a portfolio. The ONLY "
        "change vs V3-EXQ-733 is that the agent is survival-onboarded (validated "
        "scaffolded_sd054_onboarding curriculum, the 603m/603n P1-survival-3/3 "
        "recipe) BEFORE the functional test, so P1 episodes survive long enough to "
        "accrue overtakes. The curriculum does NOT touch the cross_stream_binder; "
        "the binder is trained fresh in the functional test's own P0 (as in 733).",
        "GROUND TRUTH is the agent's G x G lattice region read off env.agent_x/"
        "agent_y -- binder-independent. DV1 (alignment_real vs alignment_shuffle) "
        "tests whether the live-anchor argmax binding-affinity tracks the true "
        "region above a region-label-shuffle chance baseline; DV2 is a graded, "
        "non-saturating re-acquisition latency (REBIND-ON vs REBIND-FROZEN over ONE "
        "shared trajectory per seed).",
        "NULL declared: even survival-onboarded, if min overtakes/seed stays < "
        f"{H.MIN_OVERTAKES_P1} on >= {H.MIN_SEEDS_FOR_PASS} seeds the run self-routes "
        "substrate_not_ready_requeue (non_contributory) -- survival is NOT the (only) "
        "lever; the sibling P-B (test-bed axis) leg then carries the discrimination.",
        "UNSUPPORTED routing: rebinding_inert_off_equals_on (DV1 holds, DV2 fails = "
        "MECH-269(b) inert signature) and rebinding_not_tracking_truth (DV1 fails) "
        "both WEAKEN MECH-456; substrate_not_ready_requeue is non_contributory.",
    ]
    if not interp.get("all_ready", False):
        review_caveats.insert(
            0,
            "WARNING readiness gate UNMET on >=1 completed seed "
            f"(min_overtakes={interp.get('min_overtakes')} vs {H.MIN_OVERTAKES_P1}; "
            f"min_region_visits_p0={interp.get('min_region_visits_p0')} vs "
            f"{H.MIN_REGION_VISITS_P0}). Label self-routes substrate_not_ready_requeue "
            "(non_contributory); do NOT read as a rebinding verdict.",
        )

    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": SUPERSEDES,
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": direction,
        "evidence_direction_note": direction_note,
        "evidence_direction_per_claim": {"MECH-456": direction},
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "bears_on_not_tagged": [
            "MECH-269", "MECH-270", "ARC-006", "MECH-045", "INV-002",
        ],
        "review_caveats": review_caveats,
        "dry_run": bool(dry_run),
        "config_summary": {
            "leg": "P-A_survival_onboarded",
            "portfolio": "V3-EXQ-733a (GOV-FANOUT-1)",
            "onboarding_recipe": "scaffolded_sd054_onboarding (V3-EXQ-603m/603n)",
            "g_partition": int(G_PARTITION),
            "harness_level_measurement": True,
            "ree_core_modified": False,
            "cross_stream_binding_enabled": bool(CROSS_STREAM_BINDING_ENABLED),
            "cross_stream_binding_learned": bool(CROSS_STREAM_BINDING_LEARNED),
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke-test (1 seed, tiny onboarding + func test).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output dir (default: REE_assembly evidence/experiments).")
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, steps = P0_WARMUP_EPISODES, P1_MEASUREMENT_EPISODES, FUNC_STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0_episodes=p0, p1_episodes=p1,
                            steps_per_episode=steps, dry_run=bool(args.dry_run))

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"DV1={result['interpretation']['n_DV1']}/{result['interpretation']['n_completed']} "
        f"DV2={result['interpretation']['n_DV2']}/{result['interpretation']['n_completed']} "
        f"(need {H.MIN_SEEDS_FOR_PASS}) onboarded="
        f"{result['n_seeds_survival_onboarded']}/{result['n_completed']} "
        f"direction={manifest['evidence_direction']}",
        flush=True,
    )

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
    sys.exit(0)
