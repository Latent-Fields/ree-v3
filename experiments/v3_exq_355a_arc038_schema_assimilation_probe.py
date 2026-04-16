#!/opt/local/bin/python3
"""
V3-EXQ-355a: ARC-038 Schema Assimilation Discriminative Pair (Optimizer Isolation Fix)

MECHANISM UNDER TEST: ARC-038 (hippocampus.waking_consolidation_mode)

EXPERIMENT PURPOSE: evidence

CHANGE vs EXQ-355 (optimizer isolation bug fixed):
  EXQ-355 used two optimizers that both covered ALL E3 parameters:
    e3_online_opt = optim.Adam(agent.e3.parameters(), ...)
    e3_consolidation_opt = optim.Adam(agent.e3.parameters(), ...)
  Online training (many harm events per episode) overwhelmed the
  consolidation signal (only 3 replay steps per quiescent cycle).
  Both optimizers racing to update the same harm_eval_head meant online
  training erased consolidation updates, making CONSOLIDATION_ENABLED
  and CONSOLIDATION_ABLATED functionally identical.

  Fix:
    e3_online_opt: updates all E3 EXCEPT harm_eval_head
    e3_consolidation_opt: updates ONLY harm_eval_head
  This ensures online E3 training cannot overwrite consolidation updates to
  harm_eval_head, allowing the waking consolidation signal to accumulate.
  The gradient clip in _consolidation_step() is narrowed to harm_eval_head
  parameters only.

SCIENTIFIC QUESTION (unchanged from EXQ-355):
  Does waking consolidation replay (MECH-092) training E3.harm_eval via
  residue field supervision produce better harm avoidance vs ablation?

DESIGN (unchanged from EXQ-355):
  Two conditions, 3 matched seeds:
    CONSOLIDATION_ENABLED: quiescent E3 cycles trigger replay -> residue-labeled
      E3.harm_eval training (map-geometry update from recent experience).
    CONSOLIDATION_ABLATED: identical except replay-based E3 training skipped.

ACCEPTANCE CRITERIA (unchanged from EXQ-355):
  C1: ENABLED Phase1 harm_rate < Phase0 harm_rate in >= 2/3 seeds
  C2: ENABLED Phase1 harm_rate < ABLATED Phase1 harm_rate by >=15% relative in >= 2/3 seeds
  C3: ENABLED active_centers >= ABLATED by >=10% relative in >= 2/3 seeds (secondary)
  C4: Both conditions, all seeds: total_residue > 0 (data quality)

PASS: C1 AND C2 AND C4.

claim_ids: ["ARC-038"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-355
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ===== EXPERIMENT CONSTANTS =====
EXPERIMENT_TYPE    = "v3_exq_355a_arc038_schema_assimilation_probe"
SUPERSEDES_ID      = "V3-EXQ-355"
CLAIM_IDS          = ["ARC-038"]
EXPERIMENT_PURPOSE = "evidence"

WARMUP_EPISODES   = 50
EVAL_EPISODES     = 50
TOTAL_EPISODES    = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE = 200
NUM_SEEDS         = 3

REPLAY_STEPS_PER_QUIESCENT = 3
CONSOL_LR         = 5e-4
CONSOL_GRAD_CLIP  = 1.0

HARM_RATE_IMPROVEMENT_REQ = 0.15
MIN_CENTER_IMPROVEMENT    = 0.10

ENV_PARAMS = dict(
    size=10,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    contaminated_harm=0.04,
    resource_benefit=0.3,
    use_proxy_fields=True,
    resource_respawn_on_consume=True,
    proximity_harm_scale=0.02,
    proximity_benefit_scale=0.02,
)

ENCODER_LR = 1e-4
E3_HARM_LR = 5e-4

OUTPUT_REL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "REE_assembly", "evidence", "experiments",
)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_PARAMS)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Standard V3 agent with harm stream and world encoder."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
    )
    return REEAgent(cfg)


def _get_residue_stats(agent: REEAgent) -> Dict[str, float]:
    """Extract viability map statistics from the residue field."""
    stats = agent.get_residue_statistics()
    return {
        "active_centers": float(stats.get("active_centers", torch.tensor(0)).item()),
        "total_residue":  float(stats.get("total_residue",  torch.tensor(0.0)).item()),
        "num_harm_events": float(stats.get("num_harm_events", torch.tensor(0)).item()),
        "mean_weight":    float(stats.get("mean_weight",    torch.tensor(0.0)).item()),
    }


def _consolidation_step(
    agent: REEAgent,
    e3_consolidation_opt: optim.Optimizer,
) -> float:
    """
    Run one waking consolidation step on quiescent E3 cycle.

    Fix vs EXQ-355: gradient clip now applies only to harm_eval_head parameters
    (matching the isolated optimizer). Previously clipped all e3.parameters()
    which produced misleadingly wide clip scope given the narrow optimizer target.

    ARC-038 mechanism:
      - Generate replay trajectories from recent theta_buffer.
      - Extract z_world states from replay trajectories.
      - Evaluate residue field (read-only, no accumulation per MECH-094).
      - Train E3.harm_eval_head on residue-labeled z_world states.

    Returns: consolidation loss (0.0 if no replay content).
    """
    recent = agent.theta_buffer.recent
    if recent is None or recent.shape[0] == 0:
        return 0.0

    device = agent.device

    replay_trajs = agent.hippocampal.replay(
        recent, num_replay_steps=REPLAY_STEPS_PER_QUIESCENT, drive_state=None
    )

    if not replay_trajs:
        return 0.0

    all_z_world_states = []
    for traj in replay_trajs:
        if traj.world_states is not None and len(traj.world_states) > 1:
            for zw in traj.world_states[1:]:
                all_z_world_states.append(zw.detach())

    if not all_z_world_states:
        return 0.0

    z_world_batch = torch.cat(all_z_world_states, dim=0)

    with torch.no_grad():
        residue_vals = agent.residue_field.evaluate(z_world_batch)
        harm_targets = torch.tanh(residue_vals).unsqueeze(1)

    z_world_detached = z_world_batch.detach()

    e3_consolidation_opt.zero_grad()
    harm_pred = agent.e3.harm_eval(z_world_detached)
    loss = F.mse_loss(harm_pred, harm_targets)
    loss.backward()
    # Fix: clip only harm_eval_head parameters (matches isolated optimizer target)
    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), CONSOL_GRAD_CLIP)
    e3_consolidation_opt.step()

    return float(loss.item())


def run_condition(
    condition_name: str,
    seed: int,
    consolidation_enabled: bool,
    dry_run: bool = False,
) -> Dict:
    """
    Run one condition x seed.

    Fix vs EXQ-355: e3_online_opt excludes harm_eval parameters;
    e3_consolidation_opt covers ONLY harm_eval_head parameters.
    This prevents online training from overwriting consolidation updates.
    """
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device

    encoder_opt = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=ENCODER_LR,
    )

    # Fix vs EXQ-355: exclude harm_eval parameters from online optimizer
    e3_online_opt = optim.Adam(
        [p for n, p in agent.e3.named_parameters() if 'harm_eval' not in n],
        lr=E3_HARM_LR,
    )

    # Fix vs EXQ-355: consolidation optimizer targets ONLY harm_eval_head
    e3_consolidation_opt = optim.Adam(
        agent.e3.harm_eval_head.parameters(),
        lr=CONSOL_LR,
    )

    phase0_harm_rates: List[float] = []
    phase1_harm_rates: List[float] = []
    consol_losses: List[float] = []

    n_episodes = 2 if dry_run else TOTAL_EPISODES
    n_steps    = 5 if dry_run else STEPS_PER_EPISODE
    n_warmup   = 1 if dry_run else WARMUP_EPISODES

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        ep_harm        = 0.0
        ep_steps       = 0
        ep_consol_loss  = 0.0
        ep_consol_count = 0

        for step in range(n_steps):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm  = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = obs_harm.to(device)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            e1_prior   = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)

            if consolidation_enabled and ticks.get("e3_quiescent", False):
                c_loss = _consolidation_step(agent, e3_consolidation_opt)
                if c_loss > 0:
                    ep_consol_loss  += c_loss
                    ep_consol_count += 1

            _, harm_signal, done, info, obs_dict = env.step(action)
            ep_harm += max(0.0, float(-harm_signal))

            if harm_signal < 0:
                z_world_now = agent._current_latent.z_world if agent._current_latent is not None else None
                if z_world_now is not None:
                    agent.residue_field.accumulate(
                        z_world=z_world_now.detach(),
                        harm_magnitude=float(-harm_signal),
                        hypothesis_tag=False,
                    )

            encoder_opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
                    max_norm=1.0,
                )
                encoder_opt.step()

            # Online E3 harm_eval training: uses e3_online_opt (excludes harm_eval)
            # harm_eval_head is trained ONLY by consolidation optimizer
            if harm_signal < 0 and agent._current_latent is not None:
                z_world_now = agent._current_latent.z_world.detach()
                harm_target = torch.tensor([[float(-harm_signal)]], device=device)
                e3_online_opt.zero_grad()
                harm_pred = agent.e3.harm_eval(z_world_now)
                e3_loss   = F.mse_loss(harm_pred, harm_target)
                e3_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in agent.e3.named_parameters() if 'harm_eval' not in n],
                    1.0,
                )
                e3_online_opt.step()

            ep_steps += 1
            if done:
                break

        harm_rate  = ep_harm / max(1, ep_steps)
        mean_consol = ep_consol_loss / max(1, ep_consol_count) if ep_consol_count > 0 else 0.0

        if ep < n_warmup:
            phase0_harm_rates.append(harm_rate)
        else:
            phase1_harm_rates.append(harm_rate)
            consol_losses.append(mean_consol)

        if (ep + 1) % 10 == 0 or dry_run:
            phase_label = "P0" if ep < n_warmup else "P1"
            print(
                f"  [train] {condition_name} seed={seed} ep {ep+1}/{n_episodes}"
                f" [{phase_label}] harm_rate={harm_rate:.4f}"
                f" consol_loss={mean_consol:.5f}",
                flush=True,
            )

    residue_stats = _get_residue_stats(agent)

    mean_phase0   = float(sum(phase0_harm_rates) / max(1, len(phase0_harm_rates)))
    mean_phase1   = float(sum(phase1_harm_rates) / max(1, len(phase1_harm_rates)))
    mean_consol_all = float(sum(consol_losses)   / max(1, len(consol_losses)))

    per_seed_verdict = "PASS" if mean_phase1 < mean_phase0 else "FAIL"
    print(
        f"  verdict: [{condition_name} seed={seed}] P0={mean_phase0:.4f}"
        f" P1={mean_phase1:.4f} centers={residue_stats['active_centers']:.0f}"
        f" consol_loss={mean_consol_all:.5f} {per_seed_verdict}",
        flush=True,
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "consolidation_enabled": consolidation_enabled,
        "mean_phase0_harm_rate": mean_phase0,
        "mean_phase1_harm_rate": mean_phase1,
        "mean_consolidation_loss": mean_consol_all,
        "residue_active_centers": residue_stats["active_centers"],
        "residue_total": residue_stats["total_residue"],
        "residue_num_harm_events": residue_stats["num_harm_events"],
        "residue_mean_weight": residue_stats["mean_weight"],
        "phase0_harm_rates": phase0_harm_rates,
        "phase1_harm_rates": phase1_harm_rates,
    }


def main(dry_run: bool = False):
    print("[EXQ-355a] ARC-038 Schema Assimilation Discriminative Pair (Optimizer Isolation Fix)")
    print("=" * 70)
    print("Conditions: CONSOLIDATION_ENABLED vs CONSOLIDATION_ABLATED")
    print(f"Seeds: {NUM_SEEDS}, Episodes: {TOTAL_EPISODES}, Steps/ep: {STEPS_PER_EPISODE}")
    print("Fix vs EXQ-355: e3_online_opt excludes harm_eval; e3_consolidation_opt")
    print("  covers ONLY harm_eval_head -- prevents online training erasing consolidation.")
    print("=" * 70)

    enabled_results: List[Dict] = []
    ablated_results: List[Dict] = []

    seeds = [42 + i * 13 for i in range(NUM_SEEDS)]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print("  [CONSOLIDATION_ENABLED]")
        res_en = run_condition("CONSOLIDATION_ENABLED", seed,
                               consolidation_enabled=True, dry_run=dry_run)
        enabled_results.append(res_en)

        print("  [CONSOLIDATION_ABLATED]")
        res_ab = run_condition("CONSOLIDATION_ABLATED", seed,
                               consolidation_enabled=False, dry_run=dry_run)
        ablated_results.append(res_ab)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    c1_wins  = 0
    c2_wins  = 0
    c3_wins  = 0
    c4_ok_all = True

    seed_details = []
    for en_r, ab_r in zip(enabled_results, ablated_results):
        seed   = en_r["seed"]
        en_p0  = en_r["mean_phase0_harm_rate"]
        en_p1  = en_r["mean_phase1_harm_rate"]
        ab_p1  = ab_r["mean_phase1_harm_rate"]
        en_ac  = en_r["residue_active_centers"]
        ab_ac  = ab_r["residue_active_centers"]

        c1_seed = en_p1 < en_p0
        if c1_seed:
            c1_wins += 1

        if ab_p1 > 0:
            rel_improvement = (ab_p1 - en_p1) / ab_p1
        else:
            rel_improvement = 0.0
        c2_seed = rel_improvement >= HARM_RATE_IMPROVEMENT_REQ
        if c2_seed:
            c2_wins += 1

        if ab_ac > 0:
            rel_centers = (en_ac - ab_ac) / ab_ac
        else:
            rel_centers = float(en_ac > 0)
        c3_seed = rel_centers >= MIN_CENTER_IMPROVEMENT
        if c3_seed:
            c3_wins += 1

        if en_r["residue_total"] <= 0 or ab_r["residue_total"] <= 0:
            c4_ok_all = False

        print(
            f"Seed {seed}:"
            f" EN p0={en_p0:.4f} p1={en_p1:.4f}"
            f" AB p1={ab_p1:.4f}"
            f" rel_improv={rel_improvement:.3f}"
            f" consol_loss={en_r['mean_consolidation_loss']:.5f}"
            f" C1={c1_seed} C2={c2_seed} C3={c3_seed}",
            flush=True,
        )

        seed_details.append({
            "seed": seed,
            "enabled_phase0_harm_rate": en_p0,
            "enabled_phase1_harm_rate": en_p1,
            "ablated_phase1_harm_rate": ab_p1,
            "relative_improvement": rel_improvement,
            "enabled_active_centers": en_ac,
            "ablated_active_centers": ab_ac,
            "center_rel_improvement": rel_centers,
            "enabled_mean_consolidation_loss": en_r["mean_consolidation_loss"],
            "c1_pass": c1_seed,
            "c2_pass": c2_seed,
            "c3_pass": c3_seed,
        })

    def _mean_list(lst, key):
        vals = [r[key] for r in lst]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg_en = {
        "mean_phase0_harm_rate": _mean_list(enabled_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(enabled_results, "mean_phase1_harm_rate"),
        "mean_active_centers":   _mean_list(enabled_results, "residue_active_centers"),
        "mean_total_residue":    _mean_list(enabled_results, "residue_total"),
        "mean_consolidation_loss": _mean_list(enabled_results, "mean_consolidation_loss"),
    }
    agg_ab = {
        "mean_phase0_harm_rate": _mean_list(ablated_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(ablated_results, "mean_phase1_harm_rate"),
        "mean_active_centers":   _mean_list(ablated_results, "residue_active_centers"),
        "mean_total_residue":    _mean_list(ablated_results, "residue_total"),
    }

    c1_pass = c1_wins >= 2
    c2_pass = c2_wins >= 2
    c3_pass = c3_wins >= 2
    c4_pass = c4_ok_all

    outcome = "PASS" if (c1_pass and c2_pass and c4_pass) else "FAIL"

    print(f"\nC1 (enabled improves over warmup, >=2/3 seeds): {c1_pass} ({c1_wins}/3)", flush=True)
    print(f"C2 (enabled beats ablated >=15%, >=2/3 seeds): {c2_pass} ({c2_wins}/3)", flush=True)
    print(f"C3 (enabled has more centers >=10%, >=2/3 seeds): {c3_pass} ({c3_wins}/3)", flush=True)
    print(f"C4 (data quality: residue>0 all conditions): {c4_pass}", flush=True)
    print(f"verdict: {outcome}", flush=True)
    print(
        f"  ENABLED agg: p0={agg_en['mean_phase0_harm_rate']:.4f}"
        f" p1={agg_en['mean_phase1_harm_rate']:.4f}"
        f" centers={agg_en['mean_active_centers']:.1f}"
        f" consol_loss={agg_en['mean_consolidation_loss']:.5f}",
        flush=True,
    )
    print(
        f"  ABLATED agg: p0={agg_ab['mean_phase0_harm_rate']:.4f}"
        f" p1={agg_ab['mean_phase1_harm_rate']:.4f}"
        f" centers={agg_ab['mean_active_centers']:.1f}",
        flush=True,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "dispatch_mode": "discriminative_pair",
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_class": "waking_consolidation_necessity",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "supersedes": SUPERSEDES_ID,
        "experiment_version": "a",
        "fix_description": (
            "Optimizer isolation fix. EXQ-355 both optimizers covered all e3.parameters(). "
            "Online harm events (many/ep) overwhelmed consolidation (3 replay steps/quiescent). "
            "Fix: e3_online_opt excludes harm_eval parameters; e3_consolidation_opt covers "
            "ONLY harm_eval_head. Gradient clip in _consolidation_step() narrowed to "
            "harm_eval_head.parameters()."
        ),
        "summary": (
            f"ARC-038 discriminative pair with optimizer isolation fix. "
            f"ENABLED Phase1 harm={agg_en['mean_phase1_harm_rate']:.4f}, "
            f"ABLATED Phase1 harm={agg_ab['mean_phase1_harm_rate']:.4f}. "
            f"C1 (improvement over warmup): {c1_wins}/3 seeds. "
            f"C2 (>={int(HARM_RATE_IMPROVEMENT_REQ*100)}% relative advantage): {c2_wins}/3 seeds. "
            f"Consolidation loss={agg_en['mean_consolidation_loss']:.5f}. "
            f"Outcome: {outcome}."
        ),
        "pre_registered_thresholds": {
            "C1": "Phase1 harm_rate < Phase0 in >=2/3 enabled seeds",
            "C2": (f"Enabled Phase1 harm < Ablated Phase1 by "
                   f">={int(HARM_RATE_IMPROVEMENT_REQ*100)}% relative in >=2/3 seeds"),
            "C3": (f"Enabled active_centers >= Ablated by "
                   f">={int(MIN_CENTER_IMPROVEMENT*100)}% in >=2/3 seeds"),
            "C4": "total_residue > 0 in all conditions (data quality)",
        },
        "acceptance_checks": {
            "C1_enabled_improves_over_warmup": c1_pass,
            "C1_wins": c1_wins,
            "C2_enabled_beats_ablated_by_threshold": c2_pass,
            "C2_wins": c2_wins,
            "C3_enabled_more_centers": c3_pass,
            "C3_wins": c3_wins,
            "C4_data_quality_residue_nonzero": c4_pass,
        },
        "aggregated": {
            "CONSOLIDATION_ENABLED": agg_en,
            "CONSOLIDATION_ABLATED": agg_ab,
        },
        "seed_details": seed_details,
        "per_seed_results": {
            "CONSOLIDATION_ENABLED": [
                {k: v for k, v in r.items()
                 if k not in ("phase0_harm_rates", "phase1_harm_rates")}
                for r in enabled_results
            ],
            "CONSOLIDATION_ABLATED": [
                {k: v for k, v in r.items()
                 if k not in ("phase0_harm_rates", "phase1_harm_rates")}
                for r in ablated_results
            ],
        },
        "params": {
            "total_episodes": TOTAL_EPISODES,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_seeds": NUM_SEEDS,
            "replay_steps_per_quiescent": REPLAY_STEPS_PER_QUIESCENT,
            "consol_lr": CONSOL_LR,
            "encoder_lr": ENCODER_LR,
            "e3_harm_lr": E3_HARM_LR,
            "env": ENV_PARAMS,
            "harm_rate_improvement_req": HARM_RATE_IMPROVEMENT_REQ,
            "min_center_improvement": MIN_CENTER_IMPROVEMENT,
        },
        "design_note": (
            "Optimizer isolation fix vs EXQ-355 (non_contributory). "
            "Both conditions produced identical results in EXQ-355 because "
            "two Adam optimizers both covered all e3.parameters() -- online "
            "harm training erased consolidation updates every episode. "
            "Fix isolates e3_consolidation_opt to harm_eval_head parameters only, "
            "so consolidation signal accumulates between online updates. "
            "MECH-094 maintained: residue.accumulate() not called on replay states."
        ),
    }

    if dry_run:
        print("\n[DRY RUN] Skipping file write.", flush=True)
        print(f"[DRY RUN] Outcome would be: {outcome}", flush=True)
        return output

    out_path = os.path.join(OUTPUT_REL, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {out_path}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-355a ARC-038 Schema Assimilation (Optimizer Isolation Fix)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal wiring check (2 episodes per condition, 5 steps each)",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
