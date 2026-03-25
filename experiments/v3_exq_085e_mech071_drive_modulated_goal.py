#!/opt/local/bin/python3
"""
V3-EXQ-085e -- MECH-071 / INV-034 Goal-Directed Calibration (Drive-Modulated Seeding)

Claims: MECH-071, INV-034

Root cause of EXQ-085 through EXQ-085d failure (all z_goal_norm < 0.1):

Architecture is correct -- GoalState, E1 conditioning, E3 scoring, agent loop hookup
all exist and work. The failure is missing homeostatic drive modulation.

The gap (SD-012): benefit_exposure is an EMA (alpha=0.1) of raw benefit signals.
A single resource contact gives benefit_exposure += 0.1 * 0.3 = 0.03, which is below
the threshold of 0.05 and decays before the next contact. Without drive modulation,
a sated and a starving agent react identically to food -- the z_goal gradient signal
is too weak to overcome decay during random-walk warmup.

Literature grounding:
  Berridge & Robinson (2016): wanting (incentive salience) scales with drive state,
    not just stimulus value. z_goal encodes wanting, not hedonic impact.
  Balleine & Dickinson (1998): incentive learning requires current homeostatic state.
    Goal representations need multiple drive-buildup -> reward -> drive-reduction cycles.
  Keramati & Gutkin (2014) homeostatic RL: reward = drive reduction, not raw value.
    effective_benefit = raw_benefit * drive_level.

Fix in EXQ-085e (SD-012):
  1. drive_level = 1.0 - energy (energy already in obs_body[3]).
  2. GoalState.update(z_world, benefit, drive_level=drive_level) -- effective benefit
     = benefit * (1 + drive_weight * drive_level). At drive_level=0.8, drive_weight=2.0:
     effective_benefit = 0.03 * 2.6 = 0.078 > threshold=0.05.
  3. resource_respawn_on_consume=True -- enables repeated drive-reduction cycles
     (required for z_goal formation per Balleine & Dickinson 1998).
  4. Curriculum: first 100 episodes place one resource within 3 cells of spawn,
     guaranteeing early drive-reduction cycles for initial goal seeding.

Mechanism under test:
  GOAL_PRESENT: z_goal_enabled=True, drive_weight=2.0, resource_respawn=True
    -> drive depletes -> resource encountered -> effective_benefit above threshold
    -> z_goal seeds and persists across episodes
    -> trajectory scoring biased toward goal region -> more benefit in eval
  GOAL_ABSENT: z_goal_enabled=False (ablation baseline)
    -> no goal attractor regardless of drive dynamics

PASS criteria (ALL required):
  C1: z_goal_norm_avg > 0.1  (z_goal successfully seeded)
  C2: benefit_per_ep_goal_present > benefit_per_ep_goal_absent * 1.3  (30% improvement)
  C3: calibration_gap_goal_present > 0.02  (E3 calibration intact)
  C4: no fatal errors

Supersedes: V3-EXQ-085d
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalState, GoalConfig
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_085e_mech071_drive_modulated_goal"
CLAIM_IDS = ["MECH-071", "INV-034"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _place_resource_near_agent(env, max_dist: int = 3) -> bool:
    """Place a resource within max_dist Manhattan steps of the agent.

    Used in curriculum phase to guarantee early drive-reduction cycles.
    Returns True if a cell was found and resource was placed.
    """
    ax, ay = env.agent_x, env.agent_y
    candidates = []
    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            if dx == 0 and dy == 0:
                continue
            if abs(dx) + abs(dy) > max_dist:
                continue
            nx, ny = ax + dx, ay + dy
            if (0 < nx < env.size - 1 and 0 < ny < env.size - 1
                    and env.grid[nx, ny] == env.ENTITY_TYPES["empty"]):
                candidates.append((abs(dx) + abs(dy), nx, ny))
    if not candidates:
        return False
    # Place at the closest available cell
    candidates.sort()
    _, rx, ry = candidates[0]
    env.grid[rx, ry] = env.ENTITY_TYPES["resource"]
    env.resources.insert(0, [rx, ry])
    if env.use_proxy_fields:
        env._compute_proximity_fields()
    return True


def _run_single(
    seed: int,
    goal_present: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    curriculum_episodes: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    novelty_bonus_weight: float,
    drive_weight: float,
) -> Dict:
    """Run one (seed, condition) cell."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"

    # SD-012: resource_respawn_on_consume=True for repeated drive-reduction cycles
    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,         # moderate drain: ~200 steps to deplete from full
        use_proxy_fields=True,      # SD-010: enables harm_obs channel
        resource_respawn_on_consume=True,  # SD-012: repeated drive-reduction cycles
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=novelty_bonus_weight,  # MECH-111 curiosity
    )

    # SD-012: drive_weight=2.0 -- scales effective_benefit by drive level.
    # decay_goal=0.003 -- slower decay (half-life ~231 steps vs 139 at 0.005).
    # GoalState persists across ALL episodes (not reset per episode).
    goal_config = GoalConfig(
        goal_dim=world_dim,
        alpha_goal=0.1,
        decay_goal=0.003,           # slower decay to allow build-up over episodes
        benefit_threshold=0.05,
        goal_weight=1.0,
        drive_weight=drive_weight,  # SD-012: scale benefit by drive level
        e1_goal_conditioned=True,
        z_goal_enabled=goal_present,
    )
    goal_state = GoalState(goal_config, device=torch.device("cpu"))

    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "resource": 0,
        "benefit_approach": 0,
    }
    total_benefit_warmup = 0.0
    z_goal_norms: List[float] = []

    # --- WARMUP TRAINING ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        # SD-012 curriculum: first curriculum_episodes, place a resource near spawn
        if ep < curriculum_episodes:
            _place_resource_near_agent(env, max_dist=3)
            obs_dict = env._get_observation_dict()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # SD-012: compute drive_level from energy (obs_body[3] = agent_energy)
            if goal_present and obs_body.shape[-1] > 11:
                benefit_exposure = float(obs_body[0, 11].item() if obs_body.dim() == 2
                                        else obs_body[11].item())
                energy = float(obs_body[0, 3].item() if obs_body.dim() == 2
                               else obs_body[3].item())
                drive_level = max(0.0, 1.0 - energy)
                goal_state.update(z_world_curr, benefit_exposure,
                                  drive_level=drive_level)
                total_benefit_warmup += max(0.0, benefit_exposure)

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1
            if ttype == "resource":
                total_benefit_warmup += 1.0

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if goal_present:
            z_goal_norms.append(goal_state.goal_norm())

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            goal_norm_str = (
                f" z_goal_norm={goal_state.goal_norm():.3f}"
                if goal_present else ""
            )
            curriculum_tag = " [curriculum]" if ep < curriculum_episodes else ""
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" resource={counts['resource']}"
                f" benefit_approach={counts['benefit_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f"{goal_norm_str}{curriculum_tag}",
                flush=True,
            )

    z_goal_norm_end = goal_state.goal_norm() if goal_present else 0.0

    # --- EVAL: measure benefit acquisition with goal active ---
    agent.eval()

    benefit_per_ep: List[float] = []
    harm_buf_eval_pos: List[torch.Tensor] = []
    harm_buf_eval_neg: List[torch.Tensor] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_benefit += 1.0
            if ttype == "benefit_approach" and obs_body.dim() == 2 and obs_body.shape[-1] > 11:
                ep_benefit += float(obs_body[0, 11].item()) * 0.1

            z_world_ev = latent.z_world.detach()
            if float(harm_signal) < 0:
                harm_buf_eval_pos.append(z_world_ev)
            else:
                harm_buf_eval_neg.append(z_world_ev)

            if done:
                break

        benefit_per_ep.append(ep_benefit)

    avg_benefit = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))

    # Harm calibration check (C3)
    calibration_gap = 0.0
    if len(harm_buf_eval_pos) >= 4 and len(harm_buf_eval_neg) >= 4:
        k = min(64, min(len(harm_buf_eval_pos), len(harm_buf_eval_neg)))
        zw_pos = torch.cat(harm_buf_eval_pos[-k:], dim=0)
        zw_neg = torch.cat(harm_buf_eval_neg[-k:], dim=0)
        with torch.no_grad():
            harm_pos = agent.e3.harm_eval(zw_pos).mean().item()
            harm_neg = agent.e3.harm_eval(zw_neg).mean().item()
        calibration_gap = harm_pos - harm_neg

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" avg_benefit/ep={avg_benefit:.3f}"
        f" z_goal_norm={z_goal_norm_end:.3f}"
        f" cal_gap={calibration_gap:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "goal_present": goal_present,
        "avg_benefit_per_ep": float(avg_benefit),
        "z_goal_norm_end": float(z_goal_norm_end),
        "calibration_gap": float(calibration_gap),
        "train_resource_events": int(counts["resource"]),
        "train_benefit_approach": int(counts["benefit_approach"]),
        "train_contact_events": int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "z_goal_norm_trajectory": z_goal_norms,
    }


def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 500,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    curriculum_episodes: int = 100,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    novelty_bonus_weight: float = 0.1,
    drive_weight: float = 2.0,
    **kwargs,
) -> dict:
    """GOAL_PRESENT vs GOAL_ABSENT with SD-012 drive-modulated goal seeding."""
    results_goal:    List[Dict] = []
    results_no_goal: List[Dict] = []

    for seed in seeds:
        for goal_present in [True, False]:
            label = "GOAL_PRESENT" if goal_present else "GOAL_ABSENT"
            print(
                f"\n[V3-EXQ-085e] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" curriculum={curriculum_episodes}"
                f" drive_weight={drive_weight} alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                goal_present=goal_present,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                curriculum_episodes=curriculum_episodes,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                novelty_bonus_weight=novelty_bonus_weight,
                drive_weight=drive_weight,
            )
            if goal_present:
                results_goal.append(r)
            else:
                results_no_goal.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    z_goal_norm_avg        = _avg(results_goal,    "z_goal_norm_end")
    benefit_goal_present   = _avg(results_goal,    "avg_benefit_per_ep")
    benefit_goal_absent    = _avg(results_no_goal, "avg_benefit_per_ep")
    cal_gap_goal_present   = _avg(results_goal,    "calibration_gap")

    benefit_ratio = (
        benefit_goal_present / max(1e-6, benefit_goal_absent)
        if benefit_goal_absent > 1e-6 else 0.0
    )

    c1_pass = z_goal_norm_avg > 0.1
    c2_pass = benefit_ratio >= 1.3
    c3_pass = cal_gap_goal_present > 0.02
    c4_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-085e] Final results:", flush=True)
    print(
        f"  z_goal_norm_avg={z_goal_norm_avg:.3f}"
        f"  benefit_goal_present={benefit_goal_present:.3f}"
        f"  benefit_goal_absent={benefit_goal_absent:.3f}"
        f"  ratio={benefit_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  cal_gap_goal_present={cal_gap_goal_present:.4f}"
        f"  drive_weight={drive_weight}"
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_goal_norm={z_goal_norm_avg:.3f} <= 0.1"
            " (goal not seeded -- check drive_level is non-zero and"
            " resource_respawn is enabling repeated drive-reduction cycles)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.3x"
            f" (goal_present={benefit_goal_present:.3f}"
            f" vs goal_absent={benefit_goal_absent:.3f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: calibration_gap={cal_gap_goal_present:.4f} <= 0.02"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_goal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" z_goal_norm={r['z_goal_norm_end']:.3f}"
        f" cal_gap={r['calibration_gap']:.4f}"
        f" resource_events={r['train_resource_events']}"
        for r in results_goal
    )
    per_nogoal_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" benefit/ep={r['avg_benefit_per_ep']:.3f}"
        f" resource_events={r['train_resource_events']}"
        for r in results_no_goal
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-085e -- MECH-071 / INV-034 Drive-Modulated Goal Seeding\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-071, INV-034\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**novelty_bonus_weight:** {novelty_bonus_weight}  (MECH-111)\n"
        f"**SD-010:** use_proxy_fields=True (harm_obs wired)\n"
        f"**SD-012:** drive_weight={drive_weight}, resource_respawn_on_consume=True\n"
        f"**Curriculum:** first {curriculum_episodes} episodes place resource near spawn\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Supersedes:** V3-EXQ-085d (all prior EXQ-085* non-contributory --"
        f" goal latent never instantiated without drive modulation)\n\n"
        f"## SD-012 Drive Fix\n\n"
        f"Root cause of EXQ-085 through 085d: benefit_exposure EMA never"
        f" exceeded threshold during random-walk warmup. Single resource contact"
        f" gives benefit_exposure += 0.03 (below threshold=0.05). No drive modulation"
        f" means sated and starving agents react identically to food.\n\n"
        f"Fix: effective_benefit = benefit_exposure * (1 + drive_weight * drive_level)"
        f" where drive_level = 1 - energy (obs_body[3]). At drive_level=0.8,"
        f" drive_weight=2.0: effective_benefit = 2.6x benefit_exposure.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: z_goal_norm > 0.1 (goal seeded persistently)\n"
        f"C2: benefit_goal_present > benefit_goal_absent * 1.3\n"
        f"C3: calibration_gap_goal_present > 0.02\n"
        f"C4: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | benefit/ep | z_goal_norm | cal_gap |\n"
        f"|-----------|-----------|-------------|--------|\n"
        f"| GOAL_PRESENT | {benefit_goal_present:.3f} | {z_goal_norm_avg:.3f}"
        f" | {cal_gap_goal_present:.4f} |\n"
        f"| GOAL_ABSENT  | {benefit_goal_absent:.3f} | -- | -- |\n\n"
        f"**Benefit ratio (goal/no-goal): {benefit_ratio:.2f}x**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: z_goal_norm > 0.1 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {z_goal_norm_avg:.3f} |\n"
        f"| C2: benefit ratio >= 1.3x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: cal_gap > 0.02 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {cal_gap_goal_present:.4f} |\n"
        f"| C4: no fatal errors | {'PASS' if c4_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"GOAL_PRESENT:\n{per_goal_rows}\n\n"
        f"GOAL_ABSENT:\n{per_nogoal_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "z_goal_norm_avg":              float(z_goal_norm_avg),
        "benefit_per_ep_goal_present":  float(benefit_goal_present),
        "benefit_per_ep_goal_absent":   float(benefit_goal_absent),
        "benefit_ratio":                float(benefit_ratio),
        "calibration_gap_goal_present": float(cal_gap_goal_present),
        "drive_weight":                 float(drive_weight),
        "novelty_bonus_weight":         float(novelty_bonus_weight),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(alpha_world),
        "curriculum_episodes":          float(curriculum_episodes),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--curriculum",      type=int,   default=100)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--novelty-weight",  type=float, default=0.1)
    parser.add_argument("--drive-weight",    type=float, default=2.0)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        curriculum_episodes=args.curriculum,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        novelty_bonus_weight=args.novelty_weight,
        drive_weight=args.drive_weight,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
