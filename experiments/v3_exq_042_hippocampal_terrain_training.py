"""
V3-EXQ-042 — Hippocampal Terrain Training (SD-004, ARC-007)

Claims: SD-004, ARC-007, MECH-089

Prerequisite: EXQ-041 PASS (ThetaBuffer stabilises E3 calibration).

EXQ-041 proved E3.harm_eval is stable on theta-averaged z_world but left
terrain_prior frozen at random init. In the full pipeline, HippocampalModule
proposes trajectories via CEM guided by terrain_prior — but if terrain_prior
is random, proposals are uninformed (equivalent to random candidates).

This experiment adds terrain_prior training via E3 behavioral cloning:
  After E3 selects a trajectory, terrain_prior should predict that trajectory's
  action-object sequence from the current (z_world_theta, e1_prior, residue).
  Loss: MSE(terrain_prior_ao_mean, selected_trajectory_ao_sequence.detach())

If terrain_prior learns, hippocampal proposals should:
  a) Score lower under E3 (better trajectories selected on average)
  b) Score lower under the residue field than random proposals
     (terrain-following: avoiding residue-heavy regions)

Hippo quality gap (eval metric):
  hippo_quality_gap = mean_residue_random - mean_residue_hippo
  Positive = hippocampal proposals navigate to lower-residue regions than random.

PASS criteria:
  C1: terrain_loss_final < terrain_loss_initial * 0.7 (terrain_prior learned)
  C2: hippo_quality_gap > 0 (hippocampal proposals better than random by residue)
  C3: calibration_gap_approach > 0.03 (E3 still calibrated; terrain training didn't break it)
  C4: n_approach_eval >= 30
  C5: world_forward_r2 > 0.05

Claims tested:
  SD-004: action-object space navigation produces terrain-informed proposals
  ARC-007: proposals are value-flat (scored by residue field, no independent value head)
  MECH-089: E3 receiving theta-averaged z_world remains stable during terrain training
"""

import math
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_042_hippocampal_terrain_training"
CLAIM_IDS = ["SD-004", "ARC-007", "MECH-089"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES  = {"agent_caused_hazard", "env_caused_hazard"}
N_RANDOM_COMPARE = 8   # random candidates for quality comparison
CANDIDATE_HORIZON = 5


def run(
    seed: int = 0,
    warmup_episodes: int = 600,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    # ── Optimizers ──────────────────────────────────────────────────────────
    wf_param_ids = set(
        id(p) for p in
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    terrain_param_ids = set(
        id(p) for p in
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters())
    )
    main_params = [
        p for p in agent.parameters()
        if id(p) not in wf_param_ids and id(p) not in terrain_param_ids
    ]
    optimizer = optim.Adam(main_params, lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    # Terrain_prior trains via E3 behavioral cloning (separate optimizer)
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )

    # ── Buffers ─────────────────────────────────────────────────────────────
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    # Track terrain_loss over time (to measure learning)
    terrain_losses_early: List[float] = []   # first 100 eps
    terrain_losses_late:  List[float] = []   # last 100 eps

    print(
        f"[V3-EXQ-042] Training {warmup_episodes} eps — full pipeline + terrain_prior training\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}",
        flush=True,
    )

    agent.train()
    e3_tick_total = 0

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # ── Full pipeline ─────────────────────────────────────────────
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z    = agent.theta_buffer.summary()  # theta-averaged z_world (detached)
            z_world_curr = latent.z_world.detach()

            # ── E3 selection (get SelectionResult for terrain training) ───
            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

                # ── Terrain_prior behavioral cloning ───────────────────
                # Teach terrain_prior to predict selected trajectory's AOs.
                # Gradient flows through terrain_prior's own weights;
                # theta_z is detached (from ThetaBuffer) so no graph into encoder.
                selected_ao = result.selected_trajectory.get_action_object_sequence()
                if selected_ao is not None:
                    ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                        theta_z, e1_prior=e1_prior.detach()
                    )
                    terrain_loss = F.mse_loss(ao_mean_pred, selected_ao.detach())
                    terrain_optimizer.zero_grad()
                    terrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.hippocampal.terrain_prior.parameters()) +
                        list(agent.hippocampal.action_object_decoder.parameters()),
                        1.0,
                    )
                    terrain_optimizer.step()

                    t_loss_val = float(terrain_loss.item())
                    if ep < 100:
                        terrain_losses_early.append(t_loss_val)
                    if ep >= warmup_episodes - 100:
                        terrain_losses_late.append(t_loss_val)
            else:
                action = agent._last_action
                if action is None:
                    action_idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

            # ── Step environment ──────────────────────────────────────────
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_pos = ttype in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0))
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(z_world_curr - z_pred)

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()

            # ── E3.harm_eval training (MECH-089: theta z_world) ──────────
            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k       = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z   = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z   = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels  = torch.cat([torch.ones(k, 1), torch.zeros(k, 1)], dim=0).to(agent.device)
                harm_loss  = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                e1_loss    = agent.compute_prediction_loss()
                total_loss = harm_loss + e1_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    optimizer.step()

            # ── E2.world_forward training (MECH-069) ─────────────────────
            if len(wf_buf) >= 16 and step % 4 == 0:
                k    = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_t  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_t   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0
                    )
                    wf_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            t_early = sum(terrain_losses_early) / max(1, len(terrain_losses_early))
            t_late  = sum(terrain_losses_late)  / max(1, len(terrain_losses_late))
            print(
                f"  ep {ep+1}/{warmup_episodes}  e3_ticks={e3_tick_total}  "
                f"terrain_loss_early={t_early:.4f}  terrain_loss_late={t_late:.4f}  "
                f"var={agent.e3._running_variance:.4f}",
                flush=True,
            )

    # ── Eval phase ─────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-042] Eval: {eval_episodes} eps", flush=True)
    agent.eval()

    scores_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }
    hippo_residue_scores: List[float] = []
    random_residue_scores: List[float] = []

    with torch.no_grad():
        for ep in range(eval_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                latent   = agent.sense(obs_body, obs_world)
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                theta_z = agent.theta_buffer.summary()

                # Harm_eval calibration metric
                harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                flat_obs, harm_signal, done, info, obs_dict = env.step(
                    agent.select_action(
                        agent.generate_trajectories(latent, e1_prior, ticks), ticks
                    )
                )
                ttype = info.get("transition_type", "none")
                if ttype in APPROACH_TTYPES:
                    scores_by_ttype["approach"].append(harm_score)
                elif ttype in CONTACT_TTYPES:
                    scores_by_ttype["contact"].append(harm_score)
                else:
                    scores_by_ttype["none"].append(harm_score)

                # ── Quality comparison: hippo vs random ───────────────────
                # Sample hippocampal proposals and random proposals every 10 steps.
                # Score both via residue field. Positive gap = hippo is better.
                if step % 10 == 0:
                    z_self = latent.z_self.detach()
                    # Hippocampal proposals (terrain-informed)
                    hippo_trajs = agent.hippocampal.propose_trajectories(
                        z_world=theta_z, z_self=z_self,
                        num_candidates=N_RANDOM_COMPARE, e1_prior=e1_prior.detach(),
                    )
                    # Random proposals (for baseline)
                    random_trajs = agent.e2.generate_candidates_random(
                        initial_z_self=z_self,
                        initial_z_world=theta_z,
                        num_candidates=N_RANDOM_COMPARE,
                        horizon=CANDIDATE_HORIZON,
                        compute_action_objects=False,
                    )

                    def mean_residue(trajs) -> float:
                        scores = []
                        for t in trajs:
                            ws = t.get_world_state_sequence()
                            if ws is not None and not torch.isnan(ws).any():
                                val = float(agent.residue_field.evaluate_trajectory(ws).mean().item())
                                if not math.isnan(val):
                                    scores.append(val)
                        return float(sum(scores) / len(scores)) if scores else 0.0

                    hippo_residue_scores.append(mean_residue(hippo_trajs))
                    random_residue_scores.append(mean_residue(random_trajs))

                if done:
                    break

    # ── Compute metrics ─────────────────────────────────────────────────────
    def mean_safe(lst: list, default: float = 0.0) -> float:
        return float(sum(lst) / len(lst)) if lst else default

    mean_approach = mean_safe(scores_by_ttype["approach"])
    mean_none     = mean_safe(scores_by_ttype["none"])
    mean_contact  = mean_safe(scores_by_ttype["contact"])
    n_approach    = len(scores_by_ttype["approach"])
    cal_gap_approach = mean_approach - mean_none
    cal_gap_contact  = mean_contact  - mean_none

    mean_hippo_residue  = mean_safe(hippo_residue_scores)
    mean_random_residue = mean_safe(random_residue_scores)
    hippo_quality_gap   = mean_random_residue - mean_hippo_residue  # positive = hippo better

    terrain_loss_initial = mean_safe(terrain_losses_early)
    terrain_loss_final   = mean_safe(terrain_losses_late)
    terrain_learned = (
        terrain_loss_final < terrain_loss_initial * 0.7
        if terrain_loss_initial > 0 else False
    )

    wf_r2 = 0.0
    if len(wf_buf) >= 32:
        with torch.no_grad():
            k    = min(200, len(wf_buf))
            idxs = torch.randperm(len(wf_buf))[:k].tolist()
            zw_t  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
            a_t   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
            zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
            pred  = agent.e2.world_forward(zw_t, a_t)
            ss_res = (zw_t1 - pred).pow(2).sum().item()
            ss_tot = (zw_t1 - zw_t1.mean(0, keepdim=True)).pow(2).sum().item()
            wf_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))

    # ── PASS / FAIL ─────────────────────────────────────────────────────────
    c1 = terrain_learned
    c2 = hippo_quality_gap > 0.0
    c3 = cal_gap_approach > 0.03
    c4 = n_approach >= 30
    c5 = wf_r2 > 0.05

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: terrain_loss not sufficiently reduced "
            f"(initial={terrain_loss_initial:.4f}  final={terrain_loss_final:.4f}, "
            f"threshold=70% reduction). terrain_prior not learning E3's preferences."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: hippo_quality_gap={hippo_quality_gap:.6f} <= 0 "
            f"(hippo={mean_hippo_residue:.6f}  random={mean_random_residue:.6f}). "
            f"Hippocampal proposals not terrain-informed."
        )
    if not c3:
        failure_notes.append(f"C3 FAIL: cal_gap_approach={cal_gap_approach:.4f} <= 0.03")
    if not c4:
        failure_notes.append(f"C4 FAIL: n_approach_eval={n_approach} < 30")
    if not c5:
        failure_notes.append(f"C5 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")

    print(f"\nV3-EXQ-042 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  terrain_loss: initial={terrain_loss_initial:.4f}  final={terrain_loss_final:.4f}\n"
        f"  hippo_residue={mean_hippo_residue:.6f}  random_residue={mean_random_residue:.6f}  "
        f"quality_gap={hippo_quality_gap:.6f}\n"
        f"  cal_gap_approach={cal_gap_approach:.4f}  n_approach={n_approach}  wf_r2={wf_r2:.4f}",
        flush=True,
    )

    metrics = {
        "terrain_loss_initial":      float(terrain_loss_initial),
        "terrain_loss_final":        float(terrain_loss_final),
        "terrain_learned":           1.0 if terrain_learned else 0.0,
        "hippo_mean_residue":        float(mean_hippo_residue),
        "random_mean_residue":       float(mean_random_residue),
        "hippo_quality_gap":         float(hippo_quality_gap),
        "calibration_gap_approach":  float(cal_gap_approach),
        "calibration_gap_contact":   float(cal_gap_contact),
        "mean_harm_eval_approach":   float(mean_approach),
        "mean_harm_eval_none":       float(mean_none),
        "n_approach_eval":           float(n_approach),
        "world_forward_r2":          float(wf_r2),
        "e3_tick_total":             float(e3_tick_total),
        "crit1_pass": 1.0 if c1 else 0.0,
        "crit2_pass": 1.0 if c2 else 0.0,
        "crit3_pass": 1.0 if c3 else 0.0,
        "crit4_pass": 1.0 if c4 else 0.0,
        "crit5_pass": 1.0 if c5 else 0.0,
        "criteria_met": float(n_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-042 — Hippocampal Terrain Training

**Status:** {status}
**Claims:** SD-004, ARC-007, MECH-089
**World:** CausalGridWorldV2 (body={env.body_obs_dim}, world={env.world_obs_dim})
**alpha_world:** {alpha_world} (SD-008)  |  **Seed:** {seed}
**Prerequisite:** EXQ-041 PASS (ThetaBuffer stabilises E3 calibration)

## Motivation

EXQ-041 left terrain_prior frozen. In the full pipeline, HippocampalModule
proposes via CEM guided by terrain_prior, but an untrained terrain_prior produces
uninformed proposals (equivalent to random sampling).

This experiment adds terrain_prior training via E3 behavioral cloning. After E3
selects a trajectory, we teach terrain_prior to predict that trajectory's
action-object sequence, conditioned on (theta_z_world, e1_prior, residue_val).

terrain_prior should learn: from this world state + residue terrain, E3 prefers
these kinds of action objects. Over episodes, proposals become terrain-informed.

ARC-007 STRICT (Q-020): proposals scored by residue field only — no independent
hippocampal value head. The terrain_prior guides CEM sampling; E3 introduces all
weighting via J(ζ) = F(ζ) + λM(ζ) + ρΦ_R(ζ).

## Results

### Terrain_prior Learning
| Phase | terrain_loss |
|---|---|
| Initial (first 100 eps) | {terrain_loss_initial:.4f} |
| Final (last 100 eps) | {terrain_loss_final:.4f} |
| Reduction | {(1 - terrain_loss_final / max(terrain_loss_initial, 1e-8)) * 100:.1f}% |

### Hippocampal vs Random Proposal Quality
| Proposal type | mean residue score |
|---|---|
| Hippocampal (terrain-guided) | {mean_hippo_residue:.6f} |
| Random | {mean_random_residue:.6f} |
| Quality gap (random - hippo) | {hippo_quality_gap:.6f} |

Positive quality gap = hippocampal proposals navigate to lower-residue regions.

### E3 Calibration (MECH-089 stability check)
| Type | mean harm_eval | n |
|---|---|---|
| none | {mean_safe(scores_by_ttype["none"]):.4f} | {len(scores_by_ttype["none"])} |
| approach | {mean_approach:.4f} | {n_approach} |
| contact | {mean_contact:.4f} | {len(scores_by_ttype["contact"])} |
**calibration_gap_approach:** {cal_gap_approach:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: terrain_loss reduced >= 30% | {"PASS" if c1 else "FAIL"} | {terrain_loss_initial:.4f}→{terrain_loss_final:.4f} |
| C2: hippo_quality_gap > 0 | {"PASS" if c2 else "FAIL"} | {hippo_quality_gap:.6f} |
| C3: cal_gap_approach > 0.03 | {"PASS" if c3 else "FAIL"} | {cal_gap_approach:.4f} |
| C4: n_approach_eval >= 30 | {"PASS" if c4 else "FAIL"} | {n_approach} |
| C5: world_forward R² > 0.05 | {"PASS" if c5 else "FAIL"} | {wf_r2:.4f} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--warmup-episodes",  type=int,   default=600)
    parser.add_argument("--eval-episodes",    type=int,   default=50)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]         = CLAIM_IDS[0]
    result["verdict"]       = result["status"]
    result["run_id"]        = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
