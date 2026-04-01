#!/opt/local/bin/python3
"""
V3-EXQ-190 -- MECH-022 Hypothesis Injection Probe

Claims: MECH-022
Proposal: EXP-0042 / EVB-0042
Dispatch mode: targeted_probe

MECH-022 asserts:
  "Hippocampal systems inject hypotheses gated by control plane."

This is the first genuine V3 probe for MECH-022. It tests whether
hippocampal hypothesis injection (trajectory proposals injected into
E3 evaluation via control-plane gating) produces measurable behavioral
improvement over a system without injection.

Conditions
----------
HYPOTHESIS_INJECTION_ON:
  - HippocampalModule.propose_trajectories() generates terrain-guided
    trajectory hypotheses via CEM in action-object space (SD-004).
  - E3 evaluates these proposals and selects the best via score_trajectory().
  - The full REE pipeline: Hippocampal -> E3 evaluation -> action selection.

HYPOTHESIS_INJECTION_OFF:
  - HippocampalModule bypassed. E3 evaluates only random trajectories
    (uniform random action sequences rolled out through E2).
  - Same E3 evaluation, same training, but no terrain-informed proposals.

Design rationale
----------------
If MECH-022 is correct, the hippocampal hypothesis injection should
produce measurably better trajectory quality: lower cumulative harm
exposure and/or better residue-aware navigation. The control plane
(E3) provides the same evaluation in both conditions -- the only
difference is the quality of proposals it receives.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorld size=10, 5 hazards, 5 resources, nav_bias=0.25
Warmup: 150 episodes x 200 steps
Eval:   50 episodes x 200 steps
Estimated runtime: ~80 min any machine

Pre-registered thresholds
--------------------------
C1: harm_gap = harm_rate_OFF - harm_rate_ON >= THRESH_HARM_GAP (both seeds)
    (injection improves harm avoidance)
C2: residue_gap = mean_residue_OFF - mean_residue_ON >= 0 (both seeds)
    (injection enables lower-residue navigation; directional, no threshold)
C3: trajectory_quality_gap = traj_score_ON < traj_score_OFF (both seeds)
    (E3-scored trajectory quality is better with injection; lower = better)
C4: n_harm_min >= THRESH_N_HARM both conditions both seeds (data quality)
C5: hippo_proposals_nondegenerate: variance of hippocampal proposal scores
    in INJECTION_ON must exceed THRESH_SCORE_VAR (proposals are not all
    identical; CEM is actually refining)

Interpretation:
  C1+C2+C3+C4+C5 PASS: MECH-022 SUPPORTED. Hippocampal hypothesis injection
    produces measurable behavioral improvement; terrain-guided CEM proposals
    are better than random proposals under E3 evaluation.
  C1 fail, C2+C3 pass: harm avoidance gap below threshold but trajectory
    quality improves; may need more training or stronger harm supervision.
  C3 fail: E3 does not rate hippocampal proposals higher; proposals may not
    be meaningfully terrain-informed at current training scale.
  C5 fail: CEM degenerate; hippocampal proposals collapse to uniform.
"""

import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig
from ree_core.predictors.e2_fast import Trajectory


EXPERIMENT_TYPE = "v3_exq_190_mech022_hypothesis_injection_probe"
CLAIM_IDS = ["MECH-022"]

# Pre-registered thresholds
# C1: harm rate gap (OFF - ON) must be >= this both seeds
THRESH_HARM_GAP = 0.005
# C2: residue gap (OFF - ON) directional >= 0 (no hard threshold)
# C3: trajectory quality gap (ON < OFF) directional
# C4: minimum harm contacts both conditions both seeds (data quality)
THRESH_N_HARM = 10
# C5: variance of hippocampal proposal scores must exceed this
THRESH_SCORE_VAR = 1e-6

# Env / training configuration
BODY_OBS_DIM = 10
WORLD_OBS_DIM = 200  # CausalGridWorld size=10, use_proxy_fields=False
ACTION_DIM = 4


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=5,
        use_proxy_fields=False,
        seed=seed,
    )


def _generate_random_trajectories(
    agent: REEAgent,
    z_self: torch.Tensor,
    z_world: torch.Tensor,
    num_candidates: int,
    horizon: int,
) -> List[Trajectory]:
    """Generate random trajectory proposals (no hippocampal guidance).

    Produces uniform random action sequences and rolls them out through E2,
    bypassing HippocampalModule entirely. E3 still evaluates these -- the
    only difference is the proposal source.
    """
    device = z_world.device
    batch_size = z_world.shape[0]
    trajectories: List[Trajectory] = []

    for _ in range(num_candidates):
        # Random actions: one-hot encoded, uniform over ACTION_DIM
        action_indices = torch.randint(0, ACTION_DIM, (batch_size, horizon))
        actions = torch.zeros(batch_size, horizon, ACTION_DIM, device=device)
        for t in range(horizon):
            actions[:, t, :] = F.one_hot(
                action_indices[:, t], num_classes=ACTION_DIM
            ).float().to(device)

        traj = agent.e2.rollout_with_world(
            z_self, z_world, actions, compute_action_objects=True,
        )
        trajectories.append(traj)

    return trajectories


def _run_single(
    seed: int,
    injection_enabled: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell.

    Returns per-seed metrics for the paired comparison.
    HYPOTHESIS_INJECTION_ON: HippocampalModule proposes trajectories.
    HYPOTHESIS_INJECTION_OFF: random proposals, same E3 evaluation.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "INJECTION_ON" if injection_enabled else "INJECTION_OFF"
    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters())
        + list(agent.latent_stack.parameters())
        + list(agent.hippocampal.parameters()),
        lr=lr,
    )

    num_candidates = config.hippocampal.num_candidates
    horizon = config.hippocampal.horizon

    if dry_run:
        warmup_episodes = 3
        eval_episodes = 2

    print(
        f"\n[V3-EXQ-190] TRAIN {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" nav_bias={nav_bias}",
        flush=True,
    )
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )

            # Generate trajectory candidates -- this is the experimental manipulation
            if injection_enabled:
                # Full pipeline: HippocampalModule -> E3 evaluation
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            else:
                # Bypass hippocampal: random proposals -> E3 evaluation
                if ticks["e3_tick"] or agent._committed_candidates is None:
                    candidates = _generate_random_trajectories(
                        agent, latent.z_self, latent.z_world,
                        num_candidates, horizon,
                    )
                    agent._committed_candidates = candidates
                else:
                    candidates = agent._committed_candidates

            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            # E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            # Nav bias: with probability nav_bias, override to random action
            if random.random() < nav_bias:
                action = torch.randint(0, ACTION_DIM, (1, ACTION_DIM), dtype=torch.float32)

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    harm_events = 0
    total_steps = 0
    visited_cells: Set[tuple] = set()
    residue_vals: List[float] = []
    traj_scores_all: List[float] = []
    hippo_score_vars: List[float] = []  # per-E3-tick variance of proposal scores

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                    1, world_dim, device=agent.device
                )

                if injection_enabled:
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                else:
                    if ticks["e3_tick"] or agent._committed_candidates is None:
                        candidates = _generate_random_trajectories(
                            agent, latent.z_self, latent.z_world,
                            num_candidates, horizon,
                        )
                        agent._committed_candidates = candidates
                    else:
                        candidates = agent._committed_candidates

                # Score all candidates for metric collection
                if ticks["e3_tick"] and len(candidates) > 0:
                    scores = []
                    for cand in candidates:
                        s = agent.e3.score_trajectory(cand)
                        scores.append(float(s.item()))
                    traj_scores_all.extend(scores)
                    if len(scores) > 1:
                        score_mean = sum(scores) / len(scores)
                        score_var = sum((s - score_mean) ** 2 for s in scores) / len(scores)
                        hippo_score_vars.append(score_var)

                action = agent.select_action(candidates, ticks, temperature=0.5)

                # Collect residue at current state
                residue_val = float(
                    agent.residue_field.evaluate(latent.z_world).item()
                )
                residue_vals.append(residue_val)

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            pos_x = int(obs_dict["body_state"][0] * 10)
            pos_y = int(obs_dict["body_state"][1] * 10)
            visited_cells.add((pos_x, pos_y))
            total_steps += 1

            if done:
                break

    harm_rate = harm_events / max(1, total_steps)
    mean_residue = sum(residue_vals) / max(1, len(residue_vals))
    mean_traj_score = sum(traj_scores_all) / max(1, len(traj_scores_all))
    novel_cell_visits = len(visited_cells)
    mean_score_var = (
        sum(hippo_score_vars) / max(1, len(hippo_score_vars))
        if hippo_score_vars else 0.0
    )

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" harm_rate={harm_rate:.4f}"
        f" harm_events={harm_events}"
        f" mean_residue={mean_residue:.4f}"
        f" mean_traj_score={mean_traj_score:.4f}"
        f" cells={novel_cell_visits}"
        f" mean_score_var={mean_score_var:.6f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "injection_enabled": injection_enabled,
        "harm_rate": harm_rate,
        "harm_events": harm_events,
        "mean_residue": mean_residue,
        "mean_traj_score": mean_traj_score,
        "novel_cell_visits": novel_cell_visits,
        "mean_score_var": mean_score_var,
        "total_steps": total_steps,
    }


def run(
    seeds: Tuple[int, ...] = (42, 123),
    warmup_episodes: int = 150,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.25,
    dry_run: bool = False,
) -> dict:
    """MECH-022 hypothesis injection probe: INJECTION_ON vs INJECTION_OFF.

    Paired design: each seed runs both conditions (same env, same init).
    INJECTION_ON uses HippocampalModule CEM proposals; INJECTION_OFF uses
    random proposals. If MECH-022 is correct, hippocampal injection produces
    better trajectory quality and lower harm under E3 evaluation.
    """
    print(
        f"\n[V3-EXQ-190] MECH-022 Hypothesis Injection Probe"
        f" seeds={list(seeds)}",
        flush=True,
    )

    results_on: List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for injection_on in [True, False]:
            r = _run_single(
                seed=seed,
                injection_enabled=injection_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if injection_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    # Per-seed gaps (paired comparison)
    per_seed_harm_gap: List[float] = []      # OFF - ON (positive = injection helps)
    per_seed_residue_gap: List[float] = []   # OFF - ON (positive = injection helps)
    per_seed_traj_gap: List[float] = []      # ON - OFF (negative = injection better)

    for r_on in results_on:
        matching = [r for r in results_off if r["seed"] == r_on["seed"]]
        if matching:
            r_off = matching[0]
            per_seed_harm_gap.append(r_off["harm_rate"] - r_on["harm_rate"])
            per_seed_residue_gap.append(r_off["mean_residue"] - r_on["mean_residue"])
            per_seed_traj_gap.append(r_on["mean_traj_score"] - r_off["mean_traj_score"])

    # Aggregate means
    mean_harm_on = _avg(results_on, "harm_rate")
    mean_harm_off = _avg(results_off, "harm_rate")
    mean_residue_on = _avg(results_on, "mean_residue")
    mean_residue_off = _avg(results_off, "mean_residue")
    mean_traj_on = _avg(results_on, "mean_traj_score")
    mean_traj_off = _avg(results_off, "mean_traj_score")

    n_harm_min = min(r["harm_events"] for r in results_on + results_off)

    # Pre-registered PASS criteria
    # C1: harm gap (OFF - ON) >= THRESH_HARM_GAP both seeds
    c1_pass = (
        len(per_seed_harm_gap) > 0
        and all(g >= THRESH_HARM_GAP for g in per_seed_harm_gap)
    )
    # C2: residue gap (OFF - ON) >= 0 both seeds (directional)
    c2_pass = (
        len(per_seed_residue_gap) > 0
        and all(g >= 0.0 for g in per_seed_residue_gap)
    )
    # C3: trajectory quality gap (ON < OFF) both seeds (lower = better)
    c3_pass = (
        len(per_seed_traj_gap) > 0
        and all(g < 0.0 for g in per_seed_traj_gap)
    )
    # C4: data quality -- sufficient harm events both conditions both seeds
    c4_pass = n_harm_min >= THRESH_N_HARM
    # C5: proposal score variance non-degenerate in INJECTION_ON
    c5_pass = all(r["mean_score_var"] > THRESH_SCORE_VAR for r in results_on)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif criteria_met >= 3 and c4_pass:
        decision = "hybridize"
    elif not c4_pass:
        decision = "inconclusive"
    else:
        decision = "retire_ree_claim"

    print(
        f"\n[V3-EXQ-190] Results:"
        f" harm ON={mean_harm_on:.4f} OFF={mean_harm_off:.4f}"
        f" residue ON={mean_residue_on:.4f} OFF={mean_residue_off:.4f}"
        f" traj_score ON={mean_traj_on:.4f} OFF={mean_traj_off:.4f}",
        flush=True,
    )
    print(
        f"  per_seed_harm_gap={[round(g, 5) for g in per_seed_harm_gap]}"
        f" per_seed_residue_gap={[round(g, 5) for g in per_seed_residue_gap]}"
        f" per_seed_traj_gap={[round(g, 5) for g in per_seed_traj_gap]}"
        f" n_harm_min={n_harm_min}"
        f" decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: per-seed harm_gap (OFF-ON) {[round(g, 5) for g in per_seed_harm_gap]}"
            f" < {THRESH_HARM_GAP}"
            " -- hippocampal injection does not reduce harm rate vs random baseline"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed residue_gap (OFF-ON) {[round(g, 5) for g in per_seed_residue_gap]}"
            " -- injection does not reduce accumulated residue"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per-seed traj_gap (ON-OFF) {[round(g, 5) for g in per_seed_traj_gap]}"
            " -- E3 does not rate hippocampal proposals higher than random"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_harm_min={n_harm_min} < {THRESH_N_HARM}"
            " -- insufficient harm contacts; increase nav_bias or eval episodes"
        )
    if not c5_pass:
        failing_seeds = [r["seed"] for r in results_on
                         if r["mean_score_var"] <= THRESH_SCORE_VAR]
        failure_notes.append(
            f"C5 FAIL: proposal score variance <= {THRESH_SCORE_VAR} in seeds {failing_seeds}"
            " -- CEM proposals degenerate; hippocampal output is uniform"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            f"MECH-022 SUPPORTED: hippocampal hypothesis injection produces measurable"
            f" behavioral improvement over random proposals."
            f" INJECTION_ON: harm={mean_harm_on:.4f} residue={mean_residue_on:.4f}"
            f" traj_score={mean_traj_on:.4f}."
            f" INJECTION_OFF: harm={mean_harm_off:.4f} residue={mean_residue_off:.4f}"
            f" traj_score={mean_traj_off:.4f}."
            f" per-seed harm_gap={[round(g, 5) for g in per_seed_harm_gap]}"
            f" residue_gap={[round(g, 5) for g in per_seed_residue_gap]}"
            f" traj_gap={[round(g, 5) for g in per_seed_traj_gap]}."
            f" Terrain-guided CEM proposals are rated higher by E3 and produce"
            f" lower harm exposure, confirming hypothesis injection via control"
            f" plane gating is functional."
        )
    elif criteria_met >= 3 and c4_pass:
        interpretation = (
            f"Partial support for MECH-022: directional improvement observed on"
            f" {criteria_met}/5 criteria. Hippocampal injection shows some"
            f" benefit but below pre-registered threshold on"
            f" {5 - criteria_met} criterion/criteria."
            f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}."
            f" Consider longer training or stronger residue supervision."
        )
    elif not c4_pass:
        interpretation = (
            f"INCONCLUSIVE: insufficient harm contacts (n_harm_min={n_harm_min}"
            f" < {THRESH_N_HARM}). Cannot evaluate MECH-022. Increase nav_bias"
            f" or eval episodes."
        )
    else:
        interpretation = (
            f"MECH-022 NOT SUPPORTED: hippocampal injection does not produce"
            f" measurable improvement over random proposals at this training"
            f" scale. E3 scores, harm rate, and residue accumulation show no"
            f" consistent injection benefit across seeds."
            f" Criteria: C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}"
            f" C5={c5_pass}."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate']:.4f}"
        f" residue={r['mean_residue']:.4f}"
        f" traj_score={r['mean_traj_score']:.4f}"
        f" cells={r['novel_cell_visits']}"
        f" score_var={r['mean_score_var']:.6f}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate']:.4f}"
        f" residue={r['mean_residue']:.4f}"
        f" traj_score={r['mean_traj_score']:.4f}"
        f" cells={r['novel_cell_visits']}"
        for r in results_off
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-190 -- MECH-022 Hypothesis Injection Probe\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-022\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** HYPOTHESIS_INJECTION_ON vs HYPOTHESIS_INJECTION_OFF\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorld size=10, 5 hazards, 5 resources"
        f" nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"MECH-022 asserts hippocampal systems inject hypotheses gated by"
        f" the control plane. This experiment compares INJECTION_ON (full"
        f" HippocampalModule CEM proposals) against INJECTION_OFF (random"
        f" trajectory proposals, same E3 evaluation) across {len(seeds)}"
        f" matched seeds. Key test: does terrain-guided hypothesis injection"
        f" produce measurably better trajectories and lower harm?\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: per-seed harm_gap (OFF-ON) >= {THRESH_HARM_GAP} (both seeds)\n"
        f"C2: per-seed residue_gap (OFF-ON) >= 0 (both seeds, directional)\n"
        f"C3: per-seed traj_gap (ON-OFF) < 0 (both seeds, lower=better)\n"
        f"C4: n_harm_min >= {THRESH_N_HARM} both conditions (data quality)\n"
        f"C5: proposal score_var > {THRESH_SCORE_VAR} in INJECTION_ON (non-degenerate)\n\n"
        f"## Results\n\n"
        f"| Condition | harm_rate | mean_residue | mean_traj_score |\n"
        f"|-----------|-----------|--------------|----------------|\n"
        f"| INJECTION_ON  | {mean_harm_on:.4f}"
        f" | {mean_residue_on:.4f} | {mean_traj_on:.4f} |\n"
        f"| INJECTION_OFF | {mean_harm_off:.4f}"
        f" | {mean_residue_off:.4f} | {mean_traj_off:.4f} |\n\n"
        f"**per-seed harm_gap (OFF-ON): {[round(g, 5) for g in per_seed_harm_gap]}**\n"
        f"**per-seed residue_gap (OFF-ON): {[round(g, 5) for g in per_seed_residue_gap]}**\n"
        f"**per-seed traj_gap (ON-OFF): {[round(g, 5) for g in per_seed_traj_gap]}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: harm_gap >= {THRESH_HARM_GAP} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {[round(g, 5) for g in per_seed_harm_gap]} |\n"
        f"| C2: residue_gap >= 0 (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(g, 5) for g in per_seed_residue_gap]} |\n"
        f"| C3: traj_gap < 0 (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {[round(g, 5) for g in per_seed_traj_gap]} |\n"
        f"| C4: n_harm_min >= {THRESH_N_HARM}"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_harm_min} |\n"
        f"| C5: score_var > {THRESH_SCORE_VAR} (INJECTION_ON)"
        f" | {'PASS' if c5_pass else 'FAIL'}"
        f" | {[round(r['mean_score_var'], 8) for r in results_on]} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed Detail\n\n"
        f"INJECTION_ON:\n{per_on_rows}\n\n"
        f"INJECTION_OFF:\n{per_off_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "mean_harm_rate_on":         float(mean_harm_on),
        "mean_harm_rate_off":        float(mean_harm_off),
        "mean_harm_gap":             float(mean_harm_off - mean_harm_on),
        "mean_residue_on":           float(mean_residue_on),
        "mean_residue_off":          float(mean_residue_off),
        "mean_residue_gap":          float(mean_residue_off - mean_residue_on),
        "mean_traj_score_on":        float(mean_traj_on),
        "mean_traj_score_off":       float(mean_traj_off),
        "mean_traj_gap":             float(mean_traj_on - mean_traj_off),
        "per_seed_harm_gap_min":     float(min(per_seed_harm_gap)) if per_seed_harm_gap else 0.0,
        "per_seed_residue_gap_min":  float(min(per_seed_residue_gap)) if per_seed_residue_gap else 0.0,
        "per_seed_traj_gap_max":     float(max(per_seed_traj_gap)) if per_seed_traj_gap else 0.0,
        "n_harm_min":                float(n_harm_min),
        "score_var_min":             float(min(r["mean_score_var"] for r in results_on)),
        "n_seeds":                   float(len(seeds)),
        "nav_bias":                  float(nav_bias),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",      type=int,   default=150)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self",  type=float, default=0.3)
    parser.add_argument("--nav-bias",    type=float, default=0.25)
    parser.add_argument("--dry-run",     action="store_true",
                        help="3 warmup + 2 eval eps per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_per_seed_harm_gap":      THRESH_HARM_GAP,
        "C2_per_seed_residue_gap":   0.0,
        "C3_per_seed_traj_gap":      0.0,
        "C4_n_harm_min":             THRESH_N_HARM,
        "C5_score_var_min":          THRESH_SCORE_VAR,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["HYPOTHESIS_INJECTION_ON", "HYPOTHESIS_INJECTION_OFF"]
    result["dispatch_mode"] = "targeted_probe"
    result["backlog_id"] = "EVB-0042"
    result["evidence_class"] = "targeted_probe"
    result["claim_ids_tested"] = CLAIM_IDS

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

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
