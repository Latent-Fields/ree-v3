"""
V3-EXQ-266a: Q-020 Discriminative Pair -- Hippocampal Valence Geometry vs Value Computation
(Fixes EXQ-266 ablation bug: residue.accumulate() was unconditional, so TERRAIN_FLAT == TERRAIN_SHAPED)

MECHANISM UNDER TEST: Q-020 (hippocampal.arc007_valence_constraint_survival)
  Q-020 asks: Does ARC-007 no-value-computation constraint survive MECH-073
  (valence intrinsic to hippocampal map geometry)?

  Resolution A (adopted 2026-04-02 via literature): ARC-007 and MECH-073 are co-true.
  Hippocampus embodies value-shaped geometry via external writes (E3 harm eval ->
  ResidueField -> terrain that HippocampalModule navigates), but does NOT compute value
  itself (no RPE, utility, or outcome evaluation inside HippocampalModule).

EXPERIMENTAL DESIGN (discriminative pair):
  Condition TERRAIN_SHAPED:
    - ResidueField accumulates harm events (residue.accumulate() called on harm)
    - HippocampalModule navigates residue-shaped terrain
    - Terrain geometry becomes value-shaped via EXTERNAL writes by E3 harm evaluator
    - HippocampalModule itself does NOT evaluate harm -- it just navigates terrain
    - Prediction: lower harm_rate (terrain guidance steers away from high-residue zones)
    - Prediction: terrain_harm_corr > 0 (map geometry correlates with harm outcomes)

  Condition TERRAIN_FLAT:
    - ResidueField accumulate() NOT called (no harm residue builds up)
    - HippocampalModule navigates flat terrain (no harm geometry signal)
    - Prediction: higher harm_rate (no terrain signal for navigation)
    - Prediction: terrain_harm_corr ~ 0 (flat terrain has no predictive signal)

BUG FIXED (vs EXQ-266):
  EXQ-266 set cfg.residue.valence_enabled=False for TERRAIN_FLAT.
  However, valence_enabled only controls the 4-component SD-014 valence VECTOR recording
  (wanting/liking/harm_discriminative/surprise); it does NOT gate the basic harm scalar
  accumulation via accumulate(). evaluate() returns the plain RBF harm scalar, which was
  identical in both conditions (both accumulated equally). TERRAIN_SHAPED == TERRAIN_FLAT.
  Fix: gate the residue_field.accumulate() call by terrain_shaped flag directly.
  Both conditions still get cfg.residue.valence_enabled=True (no change needed there).

KEY DISCRIMINATION (Q-020):
  If TERRAIN_SHAPED outperforms TERRAIN_FLAT on harm_rate:
    - Confirmed: hippocampal MAP GEOMETRY (shaped by external residue writes) aids behavior
    - The improvement comes from TERRAIN NAVIGATION, not from hippocampus computing harm
    - This is exactly Resolution A: value-shaped geometry from external writes, not internal computation
  If terrain_harm_corr (TERRAIN_SHAPED) > terrain_harm_corr (TERRAIN_FLAT):
    - Confirmed: residue field writes value information INTO the map geometry
    - Hippocampus passively navigates this value-shaped geometry

PRE-REGISTERED THRESHOLDS:
  C1 (harm reduction): harm_rate_TERRAIN_FLAT >= harm_rate_TERRAIN_SHAPED * 1.15
      TERRAIN_SHAPED must reduce harm rate by at least 15% vs TERRAIN_FLAT.
  C2 (terrain correlation): terrain_harm_corr_TERRAIN_SHAPED >= 0.1
      Residue terrain scores correlate with subsequent harm outcomes in TERRAIN_SHAPED.
  C3 (consistency): both seeds show TERRAIN_SHAPED harm_rate < TERRAIN_FLAT harm_rate.
  C4 (data quality): n_harm_events_min >= 20 per (seed, condition) for eval.

EXPERIMENT_PURPOSE: evidence (Q-020 is a resolved open question; this provides
  experimental confirmation of Resolution A via behavioral discrimination)
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_266a_q020_valence_geometry_pair_fixed"
CLAIM_IDS = ["Q-020"]

# Pre-registered thresholds (unchanged from EXQ-266)
THRESH_C1_HARM_REDUCTION_FRAC = 0.15   # >= 15% harm reduction
THRESH_C2_TERRAIN_CORR = 0.10          # terrain-harm correlation >= 0.10
THRESH_C4_MIN_HARM_EVENTS = 20         # data quality gate

# Experiment parameters
SEEDS = [42, 123]
WARMUP_EPISODES = 300
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200

GRID_SIZE = 6
NUM_HAZARDS = 4
NUM_RESOURCES = 3
HARM_SCALE = 0.02
PROXIMITY_HARM_SCALE = 0.05
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
SELF_DIM = 32
WORLD_DIM = 32
LR = 1e-3


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    terrain_shaped: bool,
    dry_run: bool = False,
) -> Dict:
    """Run one (seed, condition) cell.

    terrain_shaped=True  -> TERRAIN_SHAPED: residue accumulates normally
    terrain_shaped=False -> TERRAIN_FLAT: residue.accumulate() NOT called (flat terrain)
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "TERRAIN_SHAPED" if terrain_shaped else "TERRAIN_FLAT"

    env = CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HARM_SCALE,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=PROXIMITY_HARM_SCALE,
        proximity_benefit_scale=PROXIMITY_HARM_SCALE * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=n_actions,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=0,  # SD-007 disabled -- isolate Q-020 terrain effect
    )
    cfg.latent.unified_latent_mode = False  # SD-005: z_self != z_world
    # valence_enabled=True in both conditions (SD-014 valence vectors unrelated to the ablation)

    agent = REEAgent(cfg)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    actual_warmup = min(3, WARMUP_EPISODES) if dry_run else WARMUP_EPISODES
    actual_eval = min(2, EVAL_EPISODES) if dry_run else EVAL_EPISODES

    harm_events_train = 0
    total_steps_train = 0

    # --- WARMUP (training phase) ---
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(
                1, cfg.latent.world_dim, device=agent.device
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            agent._last_action = action
            _, harm_signal, done, info, obs_dict = env.step(action)
            total_steps_train += 1

            z_world_curr = latent.z_world.detach()

            # Record transition for E2
            z_self_next_approx = latent.z_self.detach()
            agent.record_transition(latent.z_self.detach(), action.detach(), z_self_next_approx)

            # FIX (vs EXQ-266): only accumulate residue in TERRAIN_SHAPED condition.
            # In TERRAIN_FLAT, residue.accumulate() is NOT called -> terrain stays flat.
            # EXQ-266 used valence_enabled=False which only gates SD-014 valence vectors,
            # NOT the basic harm scalar accumulation. evaluate() returns the plain RBF
            # scalar, so conditions were identical in EXQ-266.
            if float(harm_signal) < 0:
                harm_events_train += 1
                if terrain_shaped:
                    agent.residue_field.accumulate(
                        z_world_curr,
                        harm_magnitude=abs(float(harm_signal)),
                    )

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == actual_warmup - 1:
            rate_so_far = harm_events_train / max(1, total_steps_train)
            print(
                f"  [train] {cond_label} seed={seed}"
                f" ep {ep+1}/{actual_warmup}"
                f" harm_rate={rate_so_far:.4f}",
                flush=True,
            )

    # --- EVAL phase ---
    agent.eval()

    harm_events_eval = 0
    total_steps_eval = 0
    terrain_scores_at_harm: List[float] = []
    terrain_scores_at_safe: List[float] = []
    trajectory_residue_scores: List[float] = []

    for ep in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world

                # Terrain score at current z_world position
                try:
                    terrain_score_now = float(
                        agent.residue_field.evaluate(z_world_curr).mean().item()
                    )
                except Exception:
                    terrain_score_now = 0.0

                # Hippocampal proposals
                try:
                    candidates = agent.hippocampal.propose_trajectories(
                        z_world_curr.detach(),
                        z_self=latent.z_self.detach(),
                        num_candidates=4,
                    )
                    if candidates:
                        best_traj = candidates[0]
                        world_seq = best_traj.get_world_state_sequence()
                        if world_seq is not None:
                            try:
                                traj_residue = float(
                                    agent.residue_field.evaluate_trajectory(world_seq).mean().item()
                                )
                                trajectory_residue_scores.append(traj_residue)
                            except Exception:
                                pass

                        ao_seq = best_traj.get_action_object_sequence()
                        if ao_seq is not None and ao_seq.shape[1] > 0:
                            first_ao = ao_seq[:, 0, :]
                            raw_logits = agent.hippocampal.action_object_decoder(first_ao)
                            action_idx = int(torch.argmax(raw_logits, dim=-1).item())
                        else:
                            action_idx = random.randint(0, n_actions - 1)
                    else:
                        action_idx = random.randint(0, n_actions - 1)
                except Exception:
                    action_idx = random.randint(0, n_actions - 1)

            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action
            _, harm_signal, done, info, obs_dict = env.step(action)
            total_steps_eval += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_events_eval += 1
                terrain_scores_at_harm.append(terrain_score_now)
            else:
                terrain_scores_at_safe.append(terrain_score_now)

            if done:
                break

    harm_rate_eval = harm_events_eval / max(1, total_steps_eval)

    terrain_harm_corr = 0.0
    if terrain_scores_at_harm and terrain_scores_at_safe:
        mean_at_harm = float(np.mean(terrain_scores_at_harm))
        mean_at_safe = float(np.mean(terrain_scores_at_safe))
        all_scores = terrain_scores_at_harm + terrain_scores_at_safe
        std_all = float(np.std(all_scores)) if len(all_scores) > 1 else 1.0
        if std_all > 1e-8:
            terrain_harm_corr = (mean_at_harm - mean_at_safe) / std_all
        else:
            terrain_harm_corr = 0.0
    else:
        mean_at_harm = 0.0
        mean_at_safe = 0.0

    mean_terrain_at_harm = float(np.mean(terrain_scores_at_harm)) if terrain_scores_at_harm else 0.0
    mean_terrain_at_safe = float(np.mean(terrain_scores_at_safe)) if terrain_scores_at_safe else 0.0
    mean_traj_residue = float(np.mean(trajectory_residue_scores)) if trajectory_residue_scores else 0.0

    residue_field_total = float(agent.residue_field.total_residue.item())
    residue_field_harm_events = int(agent.residue_field.num_harm_events.item())

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" harm_rate={harm_rate_eval:.4f}"
        f" harm_events={harm_events_eval}/{total_steps_eval}"
        f" terrain_corr={terrain_harm_corr:.4f}"
        f" residue_total={residue_field_total:.4f}"
        f" traj_residue={mean_traj_residue:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "terrain_shaped": terrain_shaped,
        "harm_events_train": int(harm_events_train),
        "total_steps_train": int(total_steps_train),
        "harm_rate_train": float(harm_events_train / max(1, total_steps_train)),
        "harm_events_eval": int(harm_events_eval),
        "total_steps_eval": int(total_steps_eval),
        "harm_rate_eval": float(harm_rate_eval),
        "terrain_harm_corr": float(terrain_harm_corr),
        "mean_terrain_at_harm": mean_terrain_at_harm,
        "mean_terrain_at_safe": mean_terrain_at_safe,
        "mean_traj_residue": float(mean_traj_residue),
        "n_traj_residue_samples": int(len(trajectory_residue_scores)),
        "residue_field_total": residue_field_total,
        "residue_field_harm_events": residue_field_harm_events,
    }


def main(dry_run=False):
    if dry_run:
        print("DRY-RUN: Q-020 valence geometry discriminative pair (EXQ-266a fix)", flush=True)
        print("Fix: accumulate() gated by terrain_shaped (was unconditional in EXQ-266)", flush=True)

        env = CausalGridWorldV2(
            size=GRID_SIZE,
            num_hazards=NUM_HAZARDS,
            num_resources=NUM_RESOURCES,
            hazard_harm=HARM_SCALE,
            use_proxy_fields=True,
        )
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=SELF_DIM,
            world_dim=WORLD_DIM,
            alpha_world=ALPHA_WORLD,
            alpha_self=ALPHA_SELF,
        )
        cfg.latent.unified_latent_mode = False

        agent_shaped = REEAgent(cfg)
        agent_flat = REEAgent(cfg)

        assert hasattr(agent_shaped.residue_field, "rbf_field"), "rbf_field missing"
        assert hasattr(agent_shaped.hippocampal, "propose_trajectories"), "hippocampal.propose_trajectories missing"
        assert hasattr(agent_shaped.hippocampal, "action_object_decoder"), "action_object_decoder missing"

        # Verify that accumulate() affects evaluate() (the fix depends on this)
        flat_obs, obs_dict = env.reset()
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent_shaped.sense(obs_body, obs_world)
        z_world = latent.z_world
        eval_before = float(agent_shaped.residue_field.evaluate(z_world).mean().item())
        agent_shaped.residue_field.accumulate(z_world, harm_magnitude=1.0)
        eval_after = float(agent_shaped.residue_field.evaluate(z_world).mean().item())
        print(f"  residue evaluate() before accumulate: {eval_before:.6f}", flush=True)
        print(f"  residue evaluate() after accumulate:  {eval_after:.6f}", flush=True)
        assert eval_after > eval_before or eval_after != eval_before, (
            "evaluate() not changed by accumulate() -- ablation fix would be ineffective"
        )

        # Verify agent_flat has zero residue (no accumulate called)
        eval_flat = float(agent_flat.residue_field.evaluate(z_world).mean().item())
        print(f"  flat agent residue (no accumulate): {eval_flat:.6f}", flush=True)

        print("DRY-RUN: single condition run ...", flush=True)
        result = _run_single(seed=42, terrain_shaped=True, dry_run=True)
        print(f"  dry-run residue_field_total={result['residue_field_total']:.4f}", flush=True)

        result_flat = _run_single(seed=42, terrain_shaped=False, dry_run=True)
        print(f"  dry-run FLAT residue_field_total={result_flat['residue_field_total']:.4f}", flush=True)

        assert result_flat["residue_field_total"] == 0.0, (
            f"TERRAIN_FLAT residue_field_total should be 0.0, got {result_flat['residue_field_total']}"
        )
        print(f"  ablation check PASS: TERRAIN_FLAT residue_field_total=0.0", flush=True)
        print("DRY-RUN PASS: fix verified, ablation effective", flush=True)
        return {"dry_run": True, "status": "PASS"}

    # Full run: 2 conditions x 2 seeds
    results_shaped: List[Dict] = []
    results_flat: List[Dict] = []

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---", flush=True)
        for terrain_shaped in [True, False]:
            cond = "TERRAIN_SHAPED" if terrain_shaped else "TERRAIN_FLAT"
            print(
                f"\n[V3-EXQ-266a] {cond} seed={seed}"
                f" warmup={WARMUP_EPISODES} eval={EVAL_EPISODES}",
                flush=True,
            )
            r = _run_single(seed=seed, terrain_shaped=terrain_shaped, dry_run=False)
            if terrain_shaped:
                results_shaped.append(r)
            else:
                results_flat.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if isinstance(r[key], (int, float))
                and not (isinstance(r[key], float) and r[key] != r[key])]
        return float(np.mean(vals)) if vals else 0.0

    harm_rate_shaped = _avg(results_shaped, "harm_rate_eval")
    harm_rate_flat = _avg(results_flat, "harm_rate_eval")
    terrain_corr_shaped = _avg(results_shaped, "terrain_harm_corr")
    terrain_corr_flat = _avg(results_flat, "terrain_harm_corr")

    n_harm_min = min(
        r["harm_events_eval"]
        for r in results_shaped + results_flat
    )

    # C1: harm reduction fraction
    if harm_rate_flat > 1e-6:
        harm_reduction_frac = (harm_rate_flat - harm_rate_shaped) / harm_rate_flat
    else:
        harm_reduction_frac = 0.0

    seed_pass_harm = sum(
        1 for s, f in zip(results_shaped, results_flat)
        if s["harm_rate_eval"] < f["harm_rate_eval"]
    )

    c1_pass = harm_reduction_frac >= THRESH_C1_HARM_REDUCTION_FRAC
    c2_pass = terrain_corr_shaped >= THRESH_C2_TERRAIN_CORR
    c3_pass = seed_pass_harm >= len(SEEDS)
    c4_pass = n_harm_min >= THRESH_C4_MIN_HARM_EVENTS

    criteria_passed = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    outcome = "PASS" if criteria_passed >= 3 else "FAIL"

    print(f"\n[V3-EXQ-266a] Results:", flush=True)
    print(
        f"  harm_rate: SHAPED={harm_rate_shaped:.4f}"
        f" FLAT={harm_rate_flat:.4f}"
        f" reduction={harm_reduction_frac:+.4f}",
        flush=True,
    )
    print(
        f"  terrain_corr: SHAPED={terrain_corr_shaped:.4f}"
        f" FLAT={terrain_corr_flat:.4f}",
        flush=True,
    )
    print(
        f"  C1 (harm_reduction >= {THRESH_C1_HARM_REDUCTION_FRAC}): {c1_pass}"
        f"  C2 (terrain_corr >= {THRESH_C2_TERRAIN_CORR}): {c2_pass}",
        flush=True,
    )
    print(
        f"  C3 (consistent seeds {seed_pass_harm}/{len(SEEDS)}): {c3_pass}"
        f"  C4 (n_harm_min={n_harm_min} >= {THRESH_C4_MIN_HARM_EVENTS}): {c4_pass}",
        flush=True,
    )
    print(f"  verdict: {outcome} ({criteria_passed}/4 criteria)", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "dispatch_mode": "discriminative_pair",
        "supersedes": "V3-EXQ-266",
        "conditions": ["TERRAIN_SHAPED", "TERRAIN_FLAT"],
        "seeds": SEEDS,
        "registered_thresholds": {
            "C1_harm_reduction_frac": THRESH_C1_HARM_REDUCTION_FRAC,
            "C2_terrain_harm_corr": THRESH_C2_TERRAIN_CORR,
            "C3_seed_pair_pass": len(SEEDS),
            "C4_n_harm_min": THRESH_C4_MIN_HARM_EVENTS,
        },
        "metrics": {
            "harm_rate_TERRAIN_SHAPED": float(harm_rate_shaped),
            "harm_rate_TERRAIN_FLAT": float(harm_rate_flat),
            "harm_reduction_frac": float(harm_reduction_frac),
            "terrain_harm_corr_TERRAIN_SHAPED": float(terrain_corr_shaped),
            "terrain_harm_corr_TERRAIN_FLAT": float(terrain_corr_flat),
            "seed_pair_pass_harm": int(seed_pass_harm),
            "n_harm_min": int(n_harm_min),
        },
        "acceptance_checks": {
            "C1_harm_reduction_frac_ge_0.15": c1_pass,
            "C2_terrain_corr_SHAPED_ge_0.10": c2_pass,
            "C3_consistent_across_seeds": c3_pass,
            "C4_data_quality_n_harm_min": c4_pass,
        },
        "criteria_passed": int(criteria_passed),
        "criteria_total": 4,
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "params": {
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "harm_scale": HARM_SCALE,
            "proximity_harm_scale": PROXIMITY_HARM_SCALE,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "alpha_world": ALPHA_WORLD,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
        },
        "bug_fix_note": (
            "EXQ-266 ablation bug: cfg.residue.valence_enabled=False only gates SD-014 "
            "valence vectors (wanting/liking/etc.), NOT the basic harm scalar accumulation. "
            "evaluate() returns the plain RBF harm scalar, so both EXQ-266 conditions were "
            "identical. Fix: gate residue_field.accumulate() by terrain_shaped flag directly."
        ),
        "summary": (
            f"Discriminative pair for Q-020 Resolution A (co-true: ARC-007 + MECH-073). "
            f"TERRAIN_SHAPED: residue.accumulate() called on harm events; "
            f"TERRAIN_FLAT: residue.accumulate() NOT called (flat terrain). "
            f"harm_rate: SHAPED={harm_rate_shaped:.4f} vs FLAT={harm_rate_flat:.4f} "
            f"(reduction={harm_reduction_frac:+.4f}). "
            f"terrain_harm_corr SHAPED={terrain_corr_shaped:.4f} vs FLAT={terrain_corr_flat:.4f}. "
            f"Outcome: {outcome} ({criteria_passed}/4)."
        ),
        "per_seed_results": {
            "TERRAIN_SHAPED": results_shaped,
            "TERRAIN_FLAT": results_flat,
        },
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}", flush=True)
    print(f"Outcome: {outcome} ({criteria_passed}/4 criteria)", flush=True)
    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
