#!/opt/local/bin/python3
"""
V3-EXQ-551 -- Pipeline entropy diagnostic: WHERE does monostrategy originate?

Claims: none (diagnostic; experiment_purpose=diagnostic; no claim weighting)

Purpose
-------
V3-EXQ-550 (z_goal monostrategy falsifier) FAILed with
action_class_entropy = 0.0 in BOTH arms (z_goal_enabled=False and
z_goal_enabled=True), bit-identical per seed. The substrate-level
monostrategy reading is sustained at no-training depth, but the
observation only tells us WHERE the agent ends up (single action class),
not WHERE in the action-selection pipeline diversity collapses.

This experiment is a no-training localization probe. It instruments
four pipeline stages per tick to identify the cliff:

  Stage 1 -- CEM candidate proposal: pairwise L2 distance variance
    across the K candidate action sequences (first-step actions)
    BEFORE selection. Tests whether the proposer produces a diverse
    candidate set or already-identical actions.

  Stage 2 -- E2 rollout divergence: variance of per-candidate
    final-step world_states across the K candidates. Tests whether
    distinct candidate action sequences produce distinct predicted
    trajectories through E2's world_forward, or whether E2 collapses
    diverse inputs to similar outputs.

  Stage 3 -- E3 evaluation spread: variance of the K per-candidate
    E3 scores (read from e3.last_scores immediately after select()).
    Tests whether E3 evaluations actually discriminate between
    rollouts, or return near-identical values that make selection
    effectively arbitrary.

  Stage 4 -- Final action class entropy: standard Shannon entropy
    over the run's executed action class histogram. Replicates the
    EXQ-550 metric for cross-comparability.

No training. No goal arm. Single arm (V_s circuit ON, matches
EXQ-550 ARM_OFF wiring -- the configuration in which EXQ-550 already
showed entropy=0.0). The experiment isolates the locus of the cliff;
the fix depends on which stage collapses, and is routed via the
pre-registered interpretation grid below.

Pre-registered interpretation grid (baked into evidence_direction_note)
----------------------------------------------------------------------
  Stage 1 variance ~0  -> proposer collapse: next move = structural
    CEM seeding (terrain prior / e1_prior already differentiated;
    CEM noise insufficient at init). Successor: V3-EXQ-553+.

  Stage 1 OK, Stage 2 ~0  -> E2 rollout collapse: distinct candidates
    converge under E2's world_forward. Next move = E2 conditioning
    diagnostic.

  Stages 1+2 OK, Stage 3 ~0  -> evaluator flatness: rollouts diverse
    but E3 scores them identically. Next move = forced-exploration
    warmup (V3-EXQ-552, queued in parallel session).

  Stages 1+2+3 OK, Stage 4 collapses  -> selection-temperature /
    policy-head bug: scores discriminate but selection picks the
    same action class. Next move = z_beta -> CEM-temperature wiring.

  All four stages diverse  -> baseline diagnostic OK; the
    monostrategy observed in EXQ-550 is training-induced, not a
    no-training pipeline property.

experiment_purpose=diagnostic. The result drives the next experiment,
not a verdict on a claim. PASS is per-stage metrics present + finite.

Implementation notes
--------------------
Stage 1, 2, 3 are read DIRECTLY off the existing API surface:
  - Stage 1: Trajectory.actions [batch, horizon, action_dim] across
    the K candidates returned by agent.generate_trajectories(); take
    first-step actions [K, action_dim] and compute pairwise L2 variance.
  - Stage 2: Trajectory.world_states is a List of [batch, world_dim]
    of length horizon+1; take the final state per candidate, stack to
    [K, world_dim], compute mean elementwise variance across K.
  - Stage 3: agent.e3.last_scores after agent.select_action(). The
    E3Selector caches last_scores on every select(); we read it back
    on the diagnostic side.

No monkey-patch; no ree_core modification. This is purely a
no-training observer.

Caveat (honest scoping)
-----------------------
- At no-training depth, the substrate exhibits the same monostrategy
  signature EXQ-550 surfaced. The four-stage decomposition is
  informative about the pipeline-level cause, not about what would
  happen after training.
- Generate_trajectories may return cached candidates between e3_ticks
  (multi-rate clock). We compute Stage 1/2 metrics only on ticks
  where new candidates were generated; ticks holding cached
  candidates are flagged in diagnostics and excluded from the
  variance aggregate.

See ree-v3/CLAUDE.md MECH-269 / SD-029 sections.
See REE_assembly/evidence/planning/goal_pipeline_plan.md for the
parent ARC-066 / MECH-269 V_s monostrategy line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_551_pipeline_entropy_diagnostic"
QUEUE_ID = "V3-EXQ-551"
CLAIM_IDS: List[str] = []  # purely diagnostic; no claim under test
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 7, 17]
TICKS_PER_SEED = 400  # ~400 ticks/seed per parent spec
EPISODE_STEPS = 200   # matches EXQ-550 per-episode budget; ep boundaries
                      # reset agent state but ticks accumulate across eps.


# Acceptance thresholds for "essentially zero" variance (Stage 1/2/3).
# These are FLAG thresholds, not pass-gate thresholds. The experiment
# PASSes if all four metrics are finite (diagnostic readiness gate).
# The flags below drive the interpretation-grid attribution downstream.
ZERO_VARIANCE_FLAG = 1e-6


def _make_env(seed: int) -> CausalGridWorldV2:
    """Mirror V3-EXQ-550 env config exactly (SD-029 scheduled hazards ON)."""
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        proximity_benefit_scale=0.18,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        harm_history_len=10,
        scheduled_external_hazard_enabled=True,
        scheduled_external_hazard_interval=50,
        scheduled_external_hazard_prob=0.5,
        scheduled_external_hazard_adjacent_only=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Mirror V3-EXQ-550 ARM_OFF agent wiring (single arm; V_s circuit ON,
    z_goal OFF). This is the configuration in which EXQ-550 already
    showed entropy=0.0, so the pipeline decomposition is run against the
    exact substrate state under test.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        harm_history_len=10,
        z_harm_a_dim=16,
        # No goal arm. Single arm matches EXQ-550 ARM_OFF.
        z_goal_enabled=False,
        drive_weight=0.0,
        e1_goal_conditioned=False,
        goal_weight=0.0,
        # V_s invalidation circuit ON (matches EXQ-550 ARM_OFF).
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    return REEAgent(cfg)


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _stage1_action_variance(candidates) -> Tuple[float, float, int]:
    """Stage 1: variance of FIRST-STEP actions across K candidates.

    Returns (mean_pairwise_l2, elementwise_variance, n_candidates).
    Higher = more diverse candidate set. ~0 = proposer collapse.
    """
    if not candidates:
        return (0.0, 0.0, 0)
    # actions: [batch=1, horizon, action_dim] per trajectory
    firsts = []
    for t in candidates:
        a = t.actions[:, 0, :]  # [1, action_dim]
        firsts.append(a.squeeze(0).detach())  # [action_dim]
    stacked = torch.stack(firsts, dim=0)  # [K, action_dim]
    K = stacked.shape[0]
    # Elementwise variance across the K candidates, mean over dims
    elem_var = float(stacked.var(dim=0, unbiased=False).mean().item())
    # Mean pairwise L2 distance
    if K >= 2:
        diffs = stacked.unsqueeze(0) - stacked.unsqueeze(1)  # [K,K,d]
        l2 = diffs.norm(dim=-1)  # [K,K]
        # Upper triangle excluding diagonal
        iu = torch.triu_indices(K, K, offset=1)
        pairwise = float(l2[iu[0], iu[1]].mean().item())
    else:
        pairwise = 0.0
    return (pairwise, elem_var, K)


def _stage2_rollout_variance(candidates) -> Tuple[float, int]:
    """Stage 2: variance of FINAL z_world states across K candidates.

    Returns (mean_elementwise_variance, n_candidates_with_world_states).
    Higher = E2 rollouts diverge under distinct candidate actions.
    ~0 = E2 collapses diverse inputs to similar predicted trajectories.
    """
    finals = []
    for t in candidates:
        if t.world_states is None or len(t.world_states) == 0:
            continue
        final = t.world_states[-1]  # [1, world_dim]
        finals.append(final.squeeze(0).detach())
    if len(finals) < 2:
        return (0.0, len(finals))
    stacked = torch.stack(finals, dim=0)  # [K, world_dim]
    elem_var = float(stacked.var(dim=0, unbiased=False).mean().item())
    return (elem_var, stacked.shape[0])


def _stage3_score_variance(agent) -> Tuple[float, float, int]:
    """Stage 3: variance of E3 per-candidate scores.

    Reads agent.e3.last_scores immediately after select_action().
    Returns (variance, score_range, K).
    Higher = E3 discriminates between candidates.
    ~0 = E3 returns near-identical values across rollouts (evaluator flatness).
    """
    e3 = agent.e3
    last_scores = getattr(e3, "last_scores", None)
    if last_scores is None:
        return (0.0, 0.0, 0)
    s = last_scores.detach()
    if s.numel() == 0:
        return (0.0, 0.0, 0)
    var = float(s.var(unbiased=False).item()) if s.numel() >= 2 else 0.0
    rng = float((s.max() - s.min()).item()) if s.numel() >= 2 else 0.0
    return (var, rng, int(s.numel()))


def _run_seed(seed: int) -> Dict:
    """Run one seed of the diagnostic. ~TICKS_PER_SEED ticks total,
    split across episodes (episode boundaries reset agent state, but
    per-tick instrumentation accumulates across the seed).
    """
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env)

    # Per-tick instrumentation lists
    stage1_pairwise: List[float] = []
    stage1_elem_var: List[float] = []
    stage2_world_var: List[float] = []
    stage3_score_var: List[float] = []
    stage3_score_range: List[float] = []

    action_counts: Dict[int, int] = {}
    n_ticks = 0
    n_ticks_with_new_candidates = 0
    n_ticks_with_world_states = 0
    n_ticks_with_scores = 0
    n_nans = 0
    error_note: Optional[str] = None

    # Number of episodes needed to cover TICKS_PER_SEED ticks.
    n_episodes = max(1, (TICKS_PER_SEED + EPISODE_STEPS - 1) // EPISODE_STEPS)
    prev_candidates_id: Optional[int] = None

    for ep in range(n_episodes):
        if n_ticks >= TICKS_PER_SEED:
            break
        _, obs_dict = env.reset()
        agent.reset()
        prev_candidates_id = None
        for _step in range(EPISODE_STEPS):
            if n_ticks >= TICKS_PER_SEED:
                break
            body = obs_dict["body_state"].float().unsqueeze(0)
            world = obs_dict["world_state"].float().unsqueeze(0)
            harm = obs_dict.get("harm_obs")
            if harm is not None:
                harm = harm.float().unsqueeze(0)
            harm_a = obs_dict.get("harm_obs_a")
            if harm_a is not None:
                harm_a = harm_a.float().unsqueeze(0)
            harm_hist = obs_dict.get("harm_history")
            if harm_hist is not None:
                harm_hist = harm_hist.float().unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=harm, obs_harm_a=harm_a, obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # Detect whether candidates are freshly generated this tick
            # (Stage 1/2 are only meaningful on new-candidate ticks; cached
            # candidates between e3_ticks reproduce the previous-tick set).
            cur_id = id(candidates)
            fresh_candidates = (cur_id != prev_candidates_id)
            prev_candidates_id = cur_id

            if fresh_candidates and candidates:
                n_ticks_with_new_candidates += 1
                # Stage 1
                pw, ev, K1 = _stage1_action_variance(candidates)
                stage1_pairwise.append(pw)
                stage1_elem_var.append(ev)
                # Stage 2
                wv, K2 = _stage2_rollout_variance(candidates)
                if K2 >= 2:
                    n_ticks_with_world_states += 1
                    stage2_world_var.append(wv)

            action = agent.select_action(candidates, ticks)

            if not torch.isfinite(action).all():
                n_nans += 1
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} ep={ep} "
                        f"step={_step}"
                    )
                break

            # Stage 3 -- read last_scores AFTER select_action(). The E3
            # selector caches the per-candidate score tensor.
            if fresh_candidates:
                sv, sr, K3 = _stage3_score_variance(agent)
                if K3 >= 2:
                    n_ticks_with_scores += 1
                    stage3_score_var.append(sv)
                    stage3_score_range.append(sr)

            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            _, _harm_signal, done, info, obs_dict = env.step(action)
            n_ticks += 1

            if done:
                break

        if error_note is not None:
            break

    # Aggregate Stage 1/2/3 metrics
    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ss = sorted(xs)
        m = len(ss)
        return float(ss[m // 2]) if m % 2 == 1 else float(0.5 * (ss[m//2 - 1] + ss[m//2]))

    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "n_ticks_with_new_candidates": n_ticks_with_new_candidates,
        "n_ticks_with_world_states": n_ticks_with_world_states,
        "n_ticks_with_scores": n_ticks_with_scores,
        "n_nans": n_nans,
        "error_note": error_note,
        # Stage 1
        "stage1_pairwise_l2_mean": _mean(stage1_pairwise),
        "stage1_pairwise_l2_median": _median(stage1_pairwise),
        "stage1_elem_var_mean": _mean(stage1_elem_var),
        # Stage 2
        "stage2_world_var_mean": _mean(stage2_world_var),
        "stage2_world_var_median": _median(stage2_world_var),
        # Stage 3
        "stage3_score_var_mean": _mean(stage3_score_var),
        "stage3_score_var_median": _median(stage3_score_var),
        "stage3_score_range_mean": _mean(stage3_score_range),
        # Stage 4
        "stage4_action_class_entropy": _shannon_entropy(action_counts),
        "stage4_action_class_counts": action_counts,
        "stage4_n_actions": sum(action_counts.values()),
    }


def _interpret_seed(r: Dict) -> str:
    """Per-seed interpretation per the pre-registered grid."""
    s1 = r["stage1_pairwise_l2_mean"]
    s2 = r["stage2_world_var_mean"]
    s3 = r["stage3_score_var_mean"]
    s4 = r["stage4_action_class_entropy"]
    z = ZERO_VARIANCE_FLAG
    if s1 < z:
        return "proposer_collapse"
    if s2 < z:
        return "e2_rollout_collapse"
    if s3 < z:
        return "evaluator_flatness"
    if s4 < z:
        return "selection_temperature_or_policy_head_bug"
    return "all_stages_diverse_training_induced"


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- pipeline entropy localization", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Ticks per seed: {TICKS_PER_SEED} "
          f"(episode_steps={EPISODE_STEPS})", flush=True)
    print(f"Env: CausalGridWorldV2 + SD-029 scheduled hazards ON "
          f"(mirrors EXQ-550)", flush=True)
    print(f"Single arm. No training. V_s circuit ON. No goal arm.",
          flush=True)
    print("Metrics: stage1 (CEM action var), stage2 (E2 rollout var), "
          "stage3 (E3 score var), stage4 (action class entropy)",
          flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)
    print(f"Zero-variance flag threshold: {ZERO_VARIANCE_FLAG}",
          flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} pipeline entropy diagnostic"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x ~40 ticks smoke test (no manifest written).",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        global SEEDS, TICKS_PER_SEED, EPISODE_STEPS
        SEEDS = [42]
        TICKS_PER_SEED = 40
        EPISODE_STEPS = 40
        print("SMOKE MODE: 1 seed x 40 ticks; no manifest write",
              flush=True)
        print(f"Seed {SEEDS[0]} Condition single_arm", flush=True)
        r = _run_seed(SEEDS[0])
        print(
            f"  [train] label seed={SEEDS[0]} ep 1/1 "
            f"n_ticks={r['n_ticks']} "
            f"s1_pw={r['stage1_pairwise_l2_mean']:.4e} "
            f"s2_wv={r['stage2_world_var_mean']:.4e} "
            f"s3_sv={r['stage3_score_var_mean']:.4e} "
            f"s4_entropy={r['stage4_action_class_entropy']:.4f}",
            flush=True,
        )
        cliff = _interpret_seed(r)
        print(f"  interpretation: {cliff}", flush=True)
        verdict = "FAIL" if r["error_note"] is not None else "PASS"
        print(f"verdict: {verdict}", flush=True)
        print("SMOKE OK", flush=True)
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    any_error = False
    for seed in SEEDS:
        print(f"Seed {seed} Condition single_arm", flush=True)
        r = _run_seed(seed)
        cliff = _interpret_seed(r)
        print(
            f"  [train] label seed={seed} ep 1/1 "
            f"n_ticks={r['n_ticks']} "
            f"new_cand_ticks={r['n_ticks_with_new_candidates']} "
            f"s1_pw_mean={r['stage1_pairwise_l2_mean']:.4e} "
            f"s1_elem_var={r['stage1_elem_var_mean']:.4e} "
            f"s2_world_var={r['stage2_world_var_mean']:.4e} "
            f"s3_score_var={r['stage3_score_var_mean']:.4e} "
            f"s3_score_range={r['stage3_score_range_mean']:.4e} "
            f"s4_entropy={r['stage4_action_class_entropy']:.4f}",
            flush=True,
        )
        print(f"  interpretation: {cliff}", flush=True)
        if r["error_note"] is not None:
            any_error = True
            print(f"  ERROR: {r['error_note']}", flush=True)
            print(f"verdict: FAIL", flush=True)
        else:
            print(f"verdict: PASS", flush=True)
        r["interpretation"] = cliff
        all_results.append(r)

    # Diagnostic outcome: PASS if all 3 seeds produced finite metrics
    # with no agent errors. The interpretation labels drive the next
    # experiment, not the pass verdict.
    def _finite(x: float) -> bool:
        return math.isfinite(x)

    all_finite = all(
        _finite(r["stage1_pairwise_l2_mean"])
        and _finite(r["stage2_world_var_mean"])
        and _finite(r["stage3_score_var_mean"])
        and _finite(r["stage4_action_class_entropy"])
        for r in all_results
    )
    outcome = "FAIL" if (any_error or not all_finite) else "PASS"

    # Aggregate per-stage diagnoses
    cliff_counts: Dict[str, int] = {}
    for r in all_results:
        cliff_counts[r["interpretation"]] = cliff_counts.get(
            r["interpretation"], 0
        ) + 1

    summary = {
        "gate_rule": (
            "Diagnostic PASS = all per-stage metrics finite across "
            "all seeds with no agent ERROR. Interpretation drives "
            "the next experiment, not the verdict."
        ),
        "cliff_distribution": cliff_counts,
        "n_seeds": len(SEEDS),
        "zero_variance_flag_threshold": ZERO_VARIANCE_FLAG,
        "any_error_seed": any_error,
        "all_metrics_finite": all_finite,
    }

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Cliff distribution across seeds: {cliff_counts}", flush=True)

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "unknown",  # diagnostic; no claim direction
        "evidence_direction_note": (
            "Diagnostic localization of WHERE the action-class-entropy=0.0 "
            "monostrategy cliff (observed in V3-EXQ-550 ARM_OFF and ARM_ON) "
            "originates in the action-selection pipeline. "
            "Pre-registered interpretation grid: "
            "(a) Stage 1 pairwise-L2 < 1e-6 -> proposer collapse; next "
            "move = structural CEM seeding (V3-EXQ-553+). "
            "(b) Stage 1 OK + Stage 2 world_var < 1e-6 -> E2 rollout "
            "collapse; next move = E2 conditioning diagnostic. "
            "(c) Stages 1+2 OK + Stage 3 score_var < 1e-6 -> evaluator "
            "flatness; next move = forced-exploration warmup (V3-EXQ-552, "
            "queued in parallel session). "
            "(d) Stages 1+2+3 OK + Stage 4 entropy < 1e-6 -> selection-"
            "temperature / policy-head bug; next move = z_beta -> CEM "
            "temperature wiring. "
            "(e) All four stages diverse -> baseline diagnostic OK; the "
            "EXQ-550 monostrategy signature is training-induced rather "
            "than a no-training pipeline property. "
            "experiment_purpose=diagnostic: PASS gates only that all "
            "metrics are finite; the cliff label drives the next "
            "experiment, not a claim-direction call."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "seeds": SEEDS,
            "ticks_per_seed": TICKS_PER_SEED,
            "episode_steps": EPISODE_STEPS,
            "zero_variance_flag_threshold": ZERO_VARIANCE_FLAG,
            "single_arm": True,
            "z_goal_enabled": False,
            "v_s_circuit_on": True,
            "sd029_scheduled_hazards": True,
            "no_training": True,
            "supersedes": None,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
