"""
V3-EXQ-355: ARC-038 Schema Assimilation Discriminative Pair (Support Attempt)

MECHANISM UNDER TEST: ARC-038 (hippocampus.waking_consolidation_mode)
  "During waking immobility, hippocampal replay switches between task-focused
   forward sweeps (planning mode) and local consolidation replay (integration
   mode) based on task demand; this waking consolidation mode is architecturally
   necessary for viability map integration during experience."

EXPERIMENT PURPOSE: evidence

SCIENTIFIC QUESTION:
  Does waking consolidation replay (MECH-092) that uses residue field supervision
  to train E3.harm_eval -- i.e., the map-geometry update from recent trajectory
  experience without a z_goal target -- produce better harm avoidance than an
  ablated agent that receives no such consolidation training?

  This is a support-attempt discriminative pair for ARC-038. EXQ-267 produced a
  valid FAIL (does_not_support) with a design where replay trajectories were
  generated but not used for any training. The key insight: if consolidation replay
  is to update "map geometry based on recent trajectory experience", the replay
  trajectories must be used to train the harm evaluation head (E3.harm_eval) using
  the residue field as supervision. Without this coupling, enabled vs ablated are
  functionally identical.

DESIGN:
  Two conditions, 3 matched seeds each:

    CONSOLIDATION_ENABLED: Standard V3 agent with waking consolidation replay active.
      During quiescent E3 cycles (e3_quiescent=True from clock):
        (1) Generate replay trajectories from recent theta_buffer content.
        (2) For each replay trajectory, extract z_world states (hypothesis_tag=True).
        (3) Evaluate residue field at each z_world to get harm labels (no gradient).
        (4) Train E3.harm_eval head: MSE loss against residue values as supervision.
      This is the "map-geometry update from recent trajectory experience" that
      ARC-038 specifies: harm avoidance knowledge becomes generalised across
      trajectory states, not just states where real harm occurred.

    CONSOLIDATION_ABLATED: Identical agent, quiescent replay training skipped.
      No replay, no E3 consolidation training. Residue field accumulates normally.
      E3.harm_eval is only trained on real harm events during online training.

  Both conditions:
    - Phase 0 (ep 0-49): Encoder warmup + online E1/E3 harm_eval training.
      Residue accumulates on harm events. Both conditions identical in Phase 0.
    - Phase 1 (ep 50-99): Same training but report harm_rate separately.
      CONSOLIDATION_ENABLED additionally runs replay consolidation on quiescent cycles.
      Primary metric: Phase 1 harm_rate (lower = better map generalisation).

  Key distinction from EXQ-267: E3.harm_eval is trained on replay-trajectory
  z_world states labeled by the residue field. This makes consolidation mechanistically
  active (map geometry updates E3 harm prediction) rather than a no-op (replay
  trajectories generated and discarded as in EXQ-267).

  Environment: CausalGridWorldV2 with reachable harm_rate target ~0.05-0.10.
    hazard_harm=0.05, num_hazards=4, size=10, use_proxy_fields=True

DESIGN NOTE ON MECH-094:
  MECH-094 requires hypothesis_tag=True for replay to block residue accumulation.
  In this design, hypothesis_tag=True is maintained -- residue.accumulate() is
  NOT called on replay states. We use residue.evaluate() (read-only) to generate
  harm labels for E3 training. This is consistent with MECH-094: replay cannot
  WRITE to residue, but CAN read from it to inform E3 learning.

PRE-REGISTERED THRESHOLDS:
  C1 (primary): CONSOLIDATION_ENABLED Phase1 harm_rate < Phase0 harm_rate
      in >= 2/3 seeds (consolidation improves map generalisation over training).
  C2 (primary): CONSOLIDATION_ENABLED Phase1 harm_rate < CONSOLIDATION_ABLATED
      Phase1 harm_rate by >=15% relative in >= 2/3 seeds
      (consolidation necessary: ablation degrades generalisation).
  C3 (secondary): CONSOLIDATION_ENABLED residue_active_centers >= ABLATED
      by >=10% relative in >= 2/3 seeds at Phase 1 end.
  C4 (data quality): Both conditions, all seeds: total_residue > 0
      (harm events occurred and residue accumulated -- experiment is informative).

PASS: C1 AND C2 AND C4
  C3 is secondary (map coverage proxy).

Supersedes: EXQ-267 (same mechanism, but EXQ-267 replay was a no-op --
  trajectories generated but not used for E3 training).

claim_ids: ["ARC-038"]
experiment_purpose: "evidence"
dispatch_mode: "discriminative_pair"
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
EXPERIMENT_TYPE = "v3_exq_355_arc038_schema_assimilation_probe"
CLAIM_IDS = ["ARC-038"]
EXPERIMENT_PURPOSE = "evidence"

WARMUP_EPISODES = 50          # Phase 0 (encoder warmup + online harm training)
EVAL_EPISODES = 50            # Phase 1 (evaluation + consolidation active)
TOTAL_EPISODES = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE = 200       # Steps per episode
NUM_SEEDS = 3

# Consolidation replay parameters
REPLAY_STEPS_PER_QUIESCENT = 3   # Replay trajectories per quiescent cycle
CONSOL_LR = 5e-4                  # E3 consolidation optimizer learning rate
CONSOL_GRAD_CLIP = 1.0            # Gradient clipping for consolidation step

# PRE-REGISTERED THRESHOLDS
HARM_RATE_IMPROVEMENT_REQ = 0.15   # C2: ENABLED must beat ABLATED by >=15% relative
MIN_CENTER_IMPROVEMENT = 0.10       # C3: ENABLED active_centers >= ABLATED by >=10%

# Environment parameters (target harm_rate ~0.05-0.10)
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

# Online training: E1 + latent_stack (encoder warmup)
ENCODER_LR = 1e-4

# E3 online training: harm_eval from real harm events
E3_HARM_LR = 5e-4

# Output path (relative to REE_assembly/evidence/experiments)
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
        alpha_world=0.9,    # SD-008: event-responsive world encoding
        alpha_self=0.3,
        use_harm_stream=True,   # SD-010: z_harm nociceptive stream
    )
    return REEAgent(cfg)


def _get_residue_stats(agent: REEAgent) -> Dict[str, float]:
    """Extract viability map statistics from the residue field."""
    stats = agent.get_residue_statistics()
    return {
        "active_centers": float(stats.get("active_centers", torch.tensor(0)).item()),
        "total_residue": float(stats.get("total_residue", torch.tensor(0.0)).item()),
        "num_harm_events": float(stats.get("num_harm_events", torch.tensor(0)).item()),
        "mean_weight": float(stats.get("mean_weight", torch.tensor(0.0)).item()),
    }


def _consolidation_step(
    agent: REEAgent,
    e3_consolidation_opt: optim.Optimizer,
) -> float:
    """
    Run one waking consolidation step on quiescent E3 cycle.

    ARC-038 mechanism:
      - Generate replay trajectories from recent theta_buffer (no z_goal target).
      - Extract z_world states from replay trajectories.
      - Evaluate residue field at those z_world states (read-only, no accumulation).
      - Train E3.harm_eval to predict residue values (map geometry update).

    MECH-094 maintained: residue.accumulate() is NOT called on replay states.
    Only residue.evaluate() (read-only) is used to label replay z_world states.

    Returns: consolidation loss value (0.0 if no replay content available).
    """
    recent = agent.theta_buffer.recent
    if recent is None or recent.shape[0] == 0:
        return 0.0

    device = agent.device

    # Generate replay trajectories (goal-free: no z_goal, only z_world from buffer)
    # Uses hippocampal.replay() -- internally calls e2.rollout_with_world()
    # which produces Trajectory with world_states
    replay_trajs = agent.hippocampal.replay(
        recent, num_replay_steps=REPLAY_STEPS_PER_QUIESCENT, drive_state=None
    )

    if not replay_trajs:
        return 0.0

    # Collect z_world states from replay trajectories for E3 training
    all_z_world_states = []
    for traj in replay_trajs:
        if traj.world_states is not None and len(traj.world_states) > 1:
            # Skip first state (initial, already in residue field)
            for zw in traj.world_states[1:]:
                all_z_world_states.append(zw.detach())

    if not all_z_world_states:
        return 0.0

    # Stack: [N, batch, world_dim] -> flatten to [N*batch, world_dim]
    # Each state is [batch=1, world_dim]
    z_world_batch = torch.cat(all_z_world_states, dim=0)  # [N, world_dim]

    # Evaluate residue field at replay z_world states (read-only, no hypothesis_tag issue)
    # residue.evaluate() returns values in [0, inf); normalize to [0, 1] target
    with torch.no_grad():
        residue_vals = agent.residue_field.evaluate(z_world_batch)  # [N]
        # Normalise: soft clamp to [0, 1] using tanh scaling
        harm_targets = torch.tanh(residue_vals).unsqueeze(1)  # [N, 1]

    # Train E3.harm_eval on replay z_world -> residue-labeled harm targets
    # Detach z_world: we are training E3 head only, not the encoder
    z_world_detached = z_world_batch.detach()

    e3_consolidation_opt.zero_grad()
    harm_pred = agent.e3.harm_eval(z_world_detached)  # [N, 1]
    loss = F.mse_loss(harm_pred, harm_targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e3.parameters(), CONSOL_GRAD_CLIP)
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

    consolidation_enabled=True: waking consolidation replay active
      On quiescent E3 cycles, generate replay from theta_buffer, label
      z_world states with residue field, train E3.harm_eval on them.

    consolidation_enabled=False: ablation -- replay-based E3 training skipped.
      E3.harm_eval only trained on real harm events (online training).
    """
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device

    # Encoder warmup optimizer: E1 + latent_stack
    encoder_opt = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=ENCODER_LR,
    )

    # E3 online harm optimizer: trains harm_eval from real harm events
    e3_online_opt = optim.Adam(agent.e3.parameters(), lr=E3_HARM_LR)

    # E3 consolidation optimizer (used only in CONSOLIDATION_ENABLED)
    e3_consolidation_opt = optim.Adam(agent.e3.parameters(), lr=CONSOL_LR)

    phase0_harm_rates: List[float] = []
    phase1_harm_rates: List[float] = []
    consol_losses: List[float] = []

    n_episodes = 2 if dry_run else TOTAL_EPISODES
    n_steps = 5 if dry_run else STEPS_PER_EPISODE
    n_warmup = 1 if dry_run else WARMUP_EPISODES

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        ep_harm = 0.0
        ep_steps = 0
        ep_consol_loss = 0.0
        ep_consol_count = 0

        for step in range(n_steps):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = obs_harm.to(device)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()

            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else \
                torch.zeros(1, agent.config.latent.world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # ARC-038: waking consolidation on quiescent E3 cycles.
            # CONSOLIDATION_ENABLED: replay-based E3.harm_eval training.
            # CONSOLIDATION_ABLATED: skip entirely.
            if consolidation_enabled and ticks.get("e3_quiescent", False):
                c_loss = _consolidation_step(agent, e3_consolidation_opt)
                if c_loss > 0:
                    ep_consol_loss += c_loss
                    ep_consol_count += 1

            _, harm_signal, done, info, obs_dict = env.step(action)
            ep_harm += max(0.0, float(-harm_signal))

            # Accumulate residue on harm events (both conditions identical)
            if harm_signal < 0:
                z_world_now = agent._current_latent.z_world if agent._current_latent is not None else None
                if z_world_now is not None:
                    agent.residue_field.accumulate(
                        z_world=z_world_now.detach(),
                        harm_magnitude=float(-harm_signal),
                        hypothesis_tag=False,
                    )

            # Online E1 encoder training (both conditions identical)
            encoder_opt.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
                    max_norm=1.0,
                )
                encoder_opt.step()

            # Online E3 harm_eval training from real harm events (both conditions)
            if harm_signal < 0 and agent._current_latent is not None:
                z_world_now = agent._current_latent.z_world.detach()
                harm_target = torch.tensor([[float(-harm_signal)]], device=device)
                e3_online_opt.zero_grad()
                harm_pred = agent.e3.harm_eval(z_world_now)
                e3_loss = F.mse_loss(harm_pred, harm_target)
                e3_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e3.parameters(), 1.0)
                e3_online_opt.step()

            ep_steps += 1
            if done:
                break

        harm_rate = ep_harm / max(1, ep_steps)
        mean_consol = ep_consol_loss / max(1, ep_consol_count) if ep_consol_count > 0 else 0.0

        if ep < n_warmup:
            phase0_harm_rates.append(harm_rate)
        else:
            phase1_harm_rates.append(harm_rate)
            consol_losses.append(mean_consol)

        if (ep + 1) % 10 == 0 or dry_run:
            phase_label = "P0" if ep < n_warmup else "P1"
            print(f"  [train] {condition_name} seed={seed} ep {ep+1}/{n_episodes} "
                  f"[{phase_label}] harm_rate={harm_rate:.4f} "
                  f"consol_loss={mean_consol:.5f}")

    # Final residue field statistics
    residue_stats = _get_residue_stats(agent)

    mean_phase0 = float(sum(phase0_harm_rates) / max(1, len(phase0_harm_rates)))
    mean_phase1 = float(sum(phase1_harm_rates) / max(1, len(phase1_harm_rates)))
    mean_consol_all = float(sum(consol_losses) / max(1, len(consol_losses)))

    per_seed_verdict = "PASS" if mean_phase1 < mean_phase0 else "FAIL"
    print(f"  verdict: [{condition_name} seed={seed}] P0={mean_phase0:.4f}"
          f" P1={mean_phase1:.4f} centers={residue_stats['active_centers']:.0f}"
          f" consol_loss={mean_consol_all:.5f} {per_seed_verdict}")

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
    print("[EXQ-355] ARC-038 Schema Assimilation Discriminative Pair")
    print("=" * 65)
    print("Conditions: CONSOLIDATION_ENABLED vs CONSOLIDATION_ABLATED")
    print(f"Seeds: {NUM_SEEDS}, Episodes: {TOTAL_EPISODES}, Steps/ep: {STEPS_PER_EPISODE}")
    print(f"Harm target: hazard_harm={ENV_PARAMS['hazard_harm']}"
          f" (target harm_rate ~0.05-0.10)")
    print("Key change vs EXQ-267: replay z_world states used to train E3.harm_eval")
    print(f"  via residue field supervision (map-geometry consolidation, ARC-038)")
    print("=" * 65)

    enabled_results: List[Dict] = []
    ablated_results: List[Dict] = []

    seeds = [42 + i * 13 for i in range(NUM_SEEDS)]

    # Run matched seed pairs
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        print(f"  [CONSOLIDATION_ENABLED]")
        res_en = run_condition("CONSOLIDATION_ENABLED", seed,
                               consolidation_enabled=True, dry_run=dry_run)
        enabled_results.append(res_en)

        print(f"  [CONSOLIDATION_ABLATED]")
        res_ab = run_condition("CONSOLIDATION_ABLATED", seed,
                               consolidation_enabled=False, dry_run=dry_run)
        ablated_results.append(res_ab)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    # Per-seed pairwise comparisons
    c1_wins = 0   # enabled: Phase1 < Phase0
    c2_wins = 0   # enabled Phase1 < ablated Phase1 by >=15% relative
    c3_wins = 0   # enabled active_centers >= ablated by >=10% relative
    c4_ok_all = True  # all residue > 0

    seed_details = []
    for en_r, ab_r in zip(enabled_results, ablated_results):
        seed = en_r["seed"]
        en_p0 = en_r["mean_phase0_harm_rate"]
        en_p1 = en_r["mean_phase1_harm_rate"]
        ab_p1 = ab_r["mean_phase1_harm_rate"]
        en_ac = en_r["residue_active_centers"]
        ab_ac = ab_r["residue_active_centers"]

        # C1: enabled improves over warmup
        c1_seed = en_p1 < en_p0
        if c1_seed:
            c1_wins += 1

        # C2: enabled beats ablated by >=15% relative
        if ab_p1 > 0:
            rel_improvement = (ab_p1 - en_p1) / ab_p1
        else:
            rel_improvement = 0.0
        c2_seed = rel_improvement >= HARM_RATE_IMPROVEMENT_REQ
        if c2_seed:
            c2_wins += 1

        # C3: enabled has more active_centers
        if ab_ac > 0:
            rel_centers = (en_ac - ab_ac) / ab_ac
        else:
            rel_centers = float(en_ac > 0)
        c3_seed = rel_centers >= MIN_CENTER_IMPROVEMENT
        if c3_seed:
            c3_wins += 1

        # C4: data quality
        if en_r["residue_total"] <= 0 or ab_r["residue_total"] <= 0:
            c4_ok_all = False

        print(f"Seed {seed}:"
              f" EN p0={en_p0:.4f} p1={en_p1:.4f}"
              f" AB p1={ab_p1:.4f}"
              f" rel_improv={rel_improvement:.3f}"
              f" consol_loss={en_r['mean_consolidation_loss']:.5f}"
              f" C1={c1_seed} C2={c2_seed} C3={c3_seed}")

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

    # Aggregate
    def _mean_list(lst, key):
        vals = [r[key] for r in lst]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg_en = {
        "mean_phase0_harm_rate": _mean_list(enabled_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(enabled_results, "mean_phase1_harm_rate"),
        "mean_active_centers": _mean_list(enabled_results, "residue_active_centers"),
        "mean_total_residue": _mean_list(enabled_results, "residue_total"),
        "mean_consolidation_loss": _mean_list(enabled_results, "mean_consolidation_loss"),
    }
    agg_ab = {
        "mean_phase0_harm_rate": _mean_list(ablated_results, "mean_phase0_harm_rate"),
        "mean_phase1_harm_rate": _mean_list(ablated_results, "mean_phase1_harm_rate"),
        "mean_active_centers": _mean_list(ablated_results, "residue_active_centers"),
        "mean_total_residue": _mean_list(ablated_results, "residue_total"),
    }

    # Acceptance checks
    c1_pass = c1_wins >= 2   # enabled improves in >=2/3 seeds
    c2_pass = c2_wins >= 2   # enabled beats ablated by >=15% in >=2/3 seeds
    c3_pass = c3_wins >= 2   # enabled has more centers in >=2/3 seeds
    c4_pass = c4_ok_all

    outcome = "PASS" if (c1_pass and c2_pass and c4_pass) else "FAIL"

    print(f"\nC1 (enabled improves over warmup, >=2/3 seeds): {c1_pass} ({c1_wins}/3)")
    print(f"C2 (enabled beats ablated >=15%, >=2/3 seeds): {c2_pass} ({c2_wins}/3)")
    print(f"C3 (enabled has more centers >=10%, >=2/3 seeds): {c3_pass} ({c3_wins}/3)")
    print(f"C4 (data quality: residue>0 all conditions): {c4_pass}")
    print(f"verdict: {outcome}")
    print(f"  ENABLED agg: p0={agg_en['mean_phase0_harm_rate']:.4f}"
          f" p1={agg_en['mean_phase1_harm_rate']:.4f}"
          f" centers={agg_en['mean_active_centers']:.1f}"
          f" consol_loss={agg_en['mean_consolidation_loss']:.5f}")
    print(f"  ABLATED agg: p0={agg_ab['mean_phase0_harm_rate']:.4f}"
          f" p1={agg_ab['mean_phase1_harm_rate']:.4f}"
          f" centers={agg_ab['mean_active_centers']:.1f}")

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
        "summary": (
            f"ARC-038 discriminative pair: waking consolidation replay with "
            f"residue-supervised E3.harm_eval training vs ablation. "
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
            "Support attempt vs EXQ-267 FAIL (does_not_support). Key design change: "
            "replay trajectories are used to train E3.harm_eval via residue-field "
            "supervision -- map-geometry update from recent trajectory experience "
            "without a z_goal target (ARC-038 consolidation mode). In EXQ-267, "
            "replay trajectories were generated but discarded (no-op); enabled vs "
            "ablated were functionally identical. MECH-094 maintained: residue.accumulate() "
            "not called on replay states; only residue.evaluate() used (read-only labeling)."
        ),
    }

    if dry_run:
        print("\n[DRY RUN] Skipping file write.")
        print(f"[DRY RUN] Outcome would be: {outcome}")
        return output

    out_path = os.path.join(OUTPUT_REL, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-355 ARC-038 Schema Assimilation Discriminative Pair"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal wiring check (2 episodes per condition, 5 steps each)"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
