#!/opt/local/bin/python3
"""
V3-EXQ-553 -- Orthogonal CEM-candidate seeding (proposer-fix arm A).

Claims: none (proposer-fix probe; experiment_purpose=evidence; claim_ids=[]).

Purpose
-------
V3-EXQ-551 / 551a / 552 confirmed: at no-training depth, CEM candidates
differ by ~2-3e-4 pairwise L2 across all 3 seeds, and action_class_entropy
collapses to 0.0. This matches the ARC-062 Phase 3 candidate-
distinguishability finding (~1e-4 at init). The proposer is collapsed at
the trajectory-generation stage.

Brainstorm idea #2 offered two candidate fixes:

  A. Orthogonal seeding (lower-disturbance, this experiment): replace
     the iid Gaussian noise that perturbs CEM candidates with an
     orthogonal basis over the (horizon * action_object_dim) space, so
     each CEM iteration starts with maximally-distinct proposals.
     Isolates "noise structure" as the variable.

  B. Discrete action-class partition (higher-disturbance, deferred to
     V3-EXQ-554 successor if A FAILs): force CEM to sample from N
     pre-declared behavioural classes, with within-class refinement.

This experiment tests arm A. If arm A lifts entropy, we know the cliff
is sensitive to seed-structure alone. If arm A FAILs to lift entropy
while pairwise L2 lifts, downstream cliffs (V3-EXQ-551a stages 2/3) are
independent blockers. If arm A FAILs to lift either, the proposer is
not the cliff at all and the evaluator/E2 fix is the primary route.

Substrate
---------
HippocampalConfig.use_orthogonal_cem_seeding (default False; bit-
identical OFF) routes the CEM inner-loop noise through an orthogonal
basis (QR-decomposed) instead of iid Gaussian per-candidate. When
n_candidates <= flatten_dim (= horizon * action_object_dim), the n
noise vectors are the columns of an orthonormal basis; when
n_candidates > flatten_dim, the basis is rank-deficient and the
implementation falls back to iid per-candidate for the surplus.

Substrate-side flag was preferred over experiment-side monkey-patch
because the change is small (~5 lines in 2 files) and makes future
experiments trivial. 309/309 contracts + 7/7 preflight PASS with master
OFF (bit-identical guarantee).

Arms
----
  ARM_IID:   current behavior, use_orthogonal_cem_seeding=False (baseline).
  ARM_ORTHO: use_orthogonal_cem_seeding=True; otherwise identical.

3 seeds [42, 7, 17] x 2 arms = 6 cells. No training (P0=20 ep at 30
ticks/ep; cliff exists at depth 0). ~30 min on Mac.

Env: CausalGridWorldV2 + SD-029 scheduled hazards, identical to
V3-EXQ-550 / 551 / 551a / 552 (cross-comparable metric scales).

Metrics
-------
Per seed x arm:
  stage1_pairwise_l2_mean -- mean pairwise L2 of first-step CEM
    candidate actions; orthogonal seeding's primary lever.
  stage1_min_pairwise_l2  -- worst-case proposal distinctness; the
    architectural guarantee orthogonal seeding provides.
  action_class_entropy    -- Shannon entropy over executed action
    classes; the V3-EXQ-550 monostrategy metric.
  action_class_counts     -- raw histogram for inspection.
  diagnostic_use_orthogonal_seeding -- echo of the flag (sanity).

Pre-registered acceptance (baked into evidence_direction_note)
--------------------------------------------------------------
  PASS = ARM_ORTHO stage1_pairwise_l2_mean > 1e-2
       AND ARM_ORTHO action_class_entropy > 0.30 in >=2/3 seeds
       AND ARM_ORTHO entropy > ARM_IID entropy + 0.10 in >=2/3 seeds.

  PARTIAL PASS = ARM_ORTHO stage1_pairwise_l2_mean > 1e-2
       AND ARM_ORTHO action_class_entropy ~= 0
       -> downstream cliffs (V3-EXQ-551a stages 2/3) are independent
       blockers; proposer fix necessary but not sufficient.
       Routes to V3-EXQ-551a stages 2/3 result review for
       evaluator / E2 fix prioritisation.

  FAIL all = ARM_ORTHO stage1_pairwise_l2_mean > 1e-2
       AND no entropy lift in either ARM_ORTHO or ARM_IID
       -> proposer is NOT the cliff; substrate collapses regardless
       of candidate distinguishability. Routes to evaluator / E2 fix
       as primary, NOT secondary.

  Five-row interpretation grid (also baked into manifest):
    R1 ARM_ORTHO stage1>1e-2 + entropy>0.3 in 2/3 seeds AND lift over
       ARM_IID -> PASS. Confirms proposer seed-structure is the
       cliff variable. Next: arm B (discrete partition) optional;
       behavioural validation downstream of stage1 fix.
    R2 ARM_ORTHO stage1>1e-2 but entropy ~= 0 -> PARTIAL PASS. The
       proposer is now diverse but downstream stages still collapse.
       Next: review V3-EXQ-551a stage 2/3 results to prioritise
       evaluator or E2 fix.
    R3 ARM_ORTHO stage1 < 1e-2 -> the substrate did not lift pairwise
       L2 in practice (decoder collapse downstream of noise; CEM
       elite-refit collapses the basis structure). Next: diagnose
       the action-object decoder or CEM elite-refit step.
    R4 ARM_IID matches ARM_ORTHO entropy lift -> bit-identical OFF
       broken or arms not differentiated. Investigate substrate
       wiring before any further routing.
    R5 Both arms produce non-zero entropy with no significant lift
       gap -> the V3-EXQ-550 / 551 / 552 monostrategy reproduction is
       seed-sensitive; rerun at higher episode count or different env
       config before drawing architectural conclusions.

experiment_purpose=evidence (testable hypothesis with clear PASS/FAIL).
claim_ids=[] intentionally: this is a proposer-fix probe, not a
single-claim test; assignment to ARC-062 / MECH-269 happens downstream
after PARTIAL/FAIL routes are walked.

See ree-v3/CLAUDE.md MECH-269 / SD-029 sections.
See REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
for the cluster context.
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


EXPERIMENT_TYPE = "v3_exq_553_orthogonal_cem_seeding"
QUEUE_ID = "V3-EXQ-553"
CLAIM_IDS: List[str] = []  # proposer-fix probe; no single-claim attribution
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 17]
ARMS = ("ARM_IID", "ARM_ORTHO")
P0_EPISODES = 20            # minimal-training depth -- cliff is at depth 0
EPISODE_STEPS = 30          # per task spec ~30-50 ticks/ep


PAIRWISE_L2_LIFT_THRESHOLD = 1e-2
ENTROPY_HIGH_THRESHOLD = 0.30
ENTROPY_GAP_THRESHOLD = 0.10


def _make_env(seed: int) -> CausalGridWorldV2:
    """Mirror V3-EXQ-551 env config exactly (SD-029 scheduled hazards ON)."""
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


def _make_agent(env: CausalGridWorldV2, use_orthogonal: bool) -> REEAgent:
    """Mirror V3-EXQ-551 ARM_OFF agent wiring; only manipulated variable is
    use_orthogonal_cem_seeding."""
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
        z_goal_enabled=False,
        drive_weight=0.0,
        e1_goal_conditioned=False,
        goal_weight=0.0,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # V3-EXQ-553 manipulated variable
        use_orthogonal_cem_seeding=use_orthogonal,
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


def _stage1_metrics(candidates) -> Tuple[float, float, int]:
    """Return (mean_pairwise_l2, min_pairwise_l2, K) of first-step CEM
    candidate actions."""
    if not candidates:
        return (0.0, 0.0, 0)
    firsts = []
    for t in candidates:
        a = t.actions[:, 0, :]  # [1, action_dim]
        firsts.append(a.squeeze(0).detach())
    stacked = torch.stack(firsts, dim=0)
    K = stacked.shape[0]
    if K < 2:
        return (0.0, 0.0, K)
    diffs = stacked.unsqueeze(0) - stacked.unsqueeze(1)
    l2 = diffs.norm(dim=-1)
    iu = torch.triu_indices(K, K, offset=1)
    pairs = l2[iu[0], iu[1]]
    return (float(pairs.mean().item()), float(pairs.min().item()), K)


def _run_cell(seed: int, arm: str) -> Dict:
    use_orthogonal = (arm == "ARM_ORTHO")
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(env, use_orthogonal=use_orthogonal)

    pairwise_l2_per_tick: List[float] = []
    min_pairwise_l2_per_tick: List[float] = []
    n_ticks_with_new_candidates = 0
    n_ticks = 0
    n_nans = 0
    error_note: Optional[str] = None
    action_counts: Dict[int, int] = {}

    prev_candidates_id: Optional[int] = None
    ortho_diag_last: Dict = {}

    for ep in range(P0_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        prev_candidates_id = None
        for _step in range(EPISODE_STEPS):
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

            cur_id = id(candidates)
            fresh = (cur_id != prev_candidates_id)
            prev_candidates_id = cur_id

            if fresh and candidates:
                n_ticks_with_new_candidates += 1
                pw_mean, pw_min, _K = _stage1_metrics(candidates)
                pairwise_l2_per_tick.append(pw_mean)
                min_pairwise_l2_per_tick.append(pw_min)
                # Read substrate diagnostic for sanity (only on fresh-candidate
                # ticks; the diagnostic dict is overwritten each propose call).
                hippocampal = getattr(agent, "hippocampal", None)
                if hippocampal is not None:
                    ortho_diag_last = dict(
                        getattr(hippocampal, "_last_propose_diagnostics", {})
                    )

            action = agent.select_action(candidates, ticks)
            if not torch.isfinite(action).all():
                n_nans += 1
                if error_note is None:
                    error_note = (
                        f"non-finite action at seed={seed} arm={arm} "
                        f"ep={ep} step={_step}"
                    )
                break

            a_idx = int(action[0].argmax().item())
            action_counts[a_idx] = action_counts.get(a_idx, 0) + 1
            _, _harm_signal, done, info, obs_dict = env.step(action)
            n_ticks += 1
            if done:
                break

        if error_note is not None:
            break

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ss = sorted(xs)
        m = len(ss)
        return float(ss[m // 2]) if m % 2 == 1 else float(0.5 * (ss[m // 2 - 1] + ss[m // 2]))

    return {
        "seed": seed,
        "arm": arm,
        "use_orthogonal_cem_seeding": use_orthogonal,
        "n_ticks": n_ticks,
        "n_ticks_with_new_candidates": n_ticks_with_new_candidates,
        "n_nans": n_nans,
        "error_note": error_note,
        "stage1_pairwise_l2_mean": _mean(pairwise_l2_per_tick),
        "stage1_pairwise_l2_median": _median(pairwise_l2_per_tick),
        "stage1_min_pairwise_l2_mean": _mean(min_pairwise_l2_per_tick),
        "action_class_entropy": _shannon_entropy(action_counts),
        "action_class_counts": action_counts,
        "n_actions": sum(action_counts.values()),
        "ortho_diag_last": ortho_diag_last,
    }


def _interpret(per_arm: Dict[str, List[Dict]]) -> Tuple[str, str]:
    """Return (outcome, interpretation_label) per the pre-registered grid."""
    iid_seeds = per_arm["ARM_IID"]
    ortho_seeds = per_arm["ARM_ORTHO"]

    if any(r.get("error_note") for r in iid_seeds + ortho_seeds):
        return ("FAIL", "agent_error")

    # ARM_ORTHO stage1 lift threshold
    ortho_pw_means = [r["stage1_pairwise_l2_mean"] for r in ortho_seeds]
    ortho_pw_mean = sum(ortho_pw_means) / len(ortho_pw_means)
    stage1_lifted = ortho_pw_mean > PAIRWISE_L2_LIFT_THRESHOLD

    # Entropy criteria
    ortho_ent = [r["action_class_entropy"] for r in ortho_seeds]
    iid_ent = [r["action_class_entropy"] for r in iid_seeds]
    n_seeds = len(ortho_ent)
    n_high = sum(1 for e in ortho_ent if e > ENTROPY_HIGH_THRESHOLD)
    n_lift = 0
    # Match seeds positionally (both lists in SEEDS order).
    for o_e, i_e in zip(ortho_ent, iid_ent):
        if (o_e - i_e) > ENTROPY_GAP_THRESHOLD:
            n_lift += 1

    high_2of3 = n_high >= 2
    lift_2of3 = n_lift >= 2

    if stage1_lifted and high_2of3 and lift_2of3:
        return ("PASS", "R1_proposer_seed_structure_is_cliff_variable")
    if stage1_lifted and not high_2of3 and not lift_2of3:
        return ("FAIL", "R2_partial_pass_downstream_cliff_independent_blocker")
    if not stage1_lifted:
        return ("FAIL", "R3_stage1_did_not_lift_decoder_or_elite_refit_collapse")
    # Edge: stage1 lifted, entropy mixed
    return ("FAIL", "R5_mixed_inconclusive")


def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- orthogonal CEM seeding proposer-fix", flush=True)
    print(f"Queue ID: {QUEUE_ID}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Arms: {ARMS}", flush=True)
    print(f"P0 episodes: {P0_EPISODES} x {EPISODE_STEPS} steps/ep "
          f"-> ~{P0_EPISODES * EPISODE_STEPS} ticks/cell, "
          f"6 cells (3 seeds x 2 arms)", flush=True)
    print(f"Env: CausalGridWorldV2 + SD-029 scheduled hazards "
          f"(mirrors EXQ-551)", flush=True)
    print(f"Manipulated variable: use_orthogonal_cem_seeding "
          f"(substrate flag, default False; bit-identical OFF)", flush=True)
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)
    print(f"PASS: ARM_ORTHO stage1_pw > {PAIRWISE_L2_LIFT_THRESHOLD} "
          f"AND entropy > {ENTROPY_HIGH_THRESHOLD} in 2/3 seeds "
          f"AND entropy lift over ARM_IID > {ENTROPY_GAP_THRESHOLD} "
          f"in 2/3 seeds", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} orthogonal CEM-candidate seeding"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit 0; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x 1 ep x 10 ticks per arm smoke; no manifest written.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        global SEEDS, P0_EPISODES, EPISODE_STEPS
        SEEDS = [42]
        P0_EPISODES = 1
        EPISODE_STEPS = 10
        print("SMOKE MODE: 1 seed x 1 ep x 10 ticks per arm; no manifest",
              flush=True)
        for arm in ARMS:
            r = _run_cell(SEEDS[0], arm)
            print(
                f"  seed={SEEDS[0]} arm={arm} "
                f"n_ticks={r['n_ticks']} "
                f"s1_pw_mean={r['stage1_pairwise_l2_mean']:.4e} "
                f"s1_min_pw={r['stage1_min_pairwise_l2_mean']:.4e} "
                f"entropy={r['action_class_entropy']:.4f} "
                f"actions={r['action_class_counts']} "
                f"ortho_diag={r['ortho_diag_last']}",
                flush=True,
            )
        print("SMOKE OK", flush=True)
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm: Dict[str, List[Dict]] = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            r = _run_cell(seed, arm)
            print(
                f"  [train] label seed={seed} arm={arm} "
                f"n_ticks={r['n_ticks']} "
                f"new_cand_ticks={r['n_ticks_with_new_candidates']} "
                f"s1_pw_mean={r['stage1_pairwise_l2_mean']:.4e} "
                f"s1_min_pw={r['stage1_min_pairwise_l2_mean']:.4e} "
                f"entropy={r['action_class_entropy']:.4f} "
                f"actions={r['action_class_counts']}",
                flush=True,
            )
            if r["error_note"] is not None:
                print(f"  ERROR: {r['error_note']}", flush=True)
            per_arm[arm].append(r)

    outcome, label = _interpret(per_arm)
    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation: {label}", flush=True)

    summary = {
        "gate_rule": (
            "PASS = ARM_ORTHO stage1_pairwise_l2_mean > 1e-2 AND "
            "action_class_entropy > 0.30 in >=2/3 seeds AND "
            "entropy > ARM_IID entropy + 0.10 in >=2/3 seeds."
        ),
        "pairwise_l2_lift_threshold": PAIRWISE_L2_LIFT_THRESHOLD,
        "entropy_high_threshold": ENTROPY_HIGH_THRESHOLD,
        "entropy_gap_threshold": ENTROPY_GAP_THRESHOLD,
        "interpretation_label": label,
        "n_seeds": len(SEEDS),
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_note": (
            "Proposer-fix probe (arm A: orthogonal-basis CEM seeding) for the "
            "action-class-entropy=0.0 monostrategy cliff confirmed by "
            "V3-EXQ-550 / 551 / 551a / 552. Substrate flag "
            "HippocampalConfig.use_orthogonal_cem_seeding replaces iid "
            "Gaussian per-candidate noise with QR-decomposed orthogonal "
            "basis over (horizon x action_object_dim). Five-row "
            "interpretation grid: "
            "R1 PASS (ARM_ORTHO stage1>1e-2 + entropy>0.3 in 2/3 + "
            "lift>0.10 over ARM_IID in 2/3) -> proposer seed-structure IS "
            "the cliff variable; arm B (discrete partition) optional. "
            "R2 PARTIAL_PASS (stage1 lifts but entropy ~= 0) -> downstream "
            "cliffs (V3-EXQ-551a stages 2/3) are independent blockers; "
            "review evaluator / E2 fix prioritisation. "
            "R3 stage1<1e-2 -> decoder collapse or CEM elite-refit "
            "collapses the orthogonal structure downstream of the noise; "
            "diagnose action-object decoder or elite-refit step. "
            "R4 ARM_IID matches ARM_ORTHO -> bit-identical-OFF broken; "
            "investigate substrate wiring. "
            "R5 mixed -> rerun at higher episode depth before architectural "
            "conclusions. "
            "experiment_purpose=evidence with claim_ids=[] intentional: "
            "this is a proposer-fix probe; claim assignment downstream of "
            "PARTIAL/FAIL routing."
        ),
        "pass_criteria_summary": summary,
        "per_arm_results": per_arm,
        "config": {
            "seeds": SEEDS,
            "arms": list(ARMS),
            "p0_episodes": P0_EPISODES,
            "episode_steps": EPISODE_STEPS,
            "manipulated_variable": "use_orthogonal_cem_seeding",
            "env_kwargs_match": "V3-EXQ-551",
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
