#!/opt/local/bin/python3
"""
V3-EXQ-190a -- MECH-022 Hypothesis Injection Probe (well-powered successor)

Claims: MECH-022
Supersedes: V3-EXQ-190 (2026-04-01, FAIL/inconclusive, 3/5 criteria met)

MECH-022 asserts:
  "Hippocampal systems inject hypotheses gated by control plane."

V3-EXQ-190 tested the same hypothesis with the same design used here (paired
INJECTION_ON vs INJECTION_OFF comparison, same conditions, same criteria) but
was underpowered: C4 (data-quality floor, n_harm_min >= 10) FAILed at
n_harm_min=0 -- zero harm contacts were recorded in the INJECTION_ON eval
across both seeds, and only 6-17 in INJECTION_OFF. C1/C2 (harm_gap,
residue_gap) PASSed on the thin data available, but C3 (E3 rates ON
proposals better than OFF, the load-bearing test of whether hippocampal
injection actually improves the E3 candidate pool) FAILed inconsistently
across seeds (traj_gap [-716.4, +3.8] -- one seed strongly favoured ON, the
other barely favoured OFF), which is exactly the kind of noisy read a
data-starved probe produces.

WHY THE ORIGINAL WAS UNDERPOWERED (calibration, this redesign): a pure
random-policy probe against the SAME env config (num_hazards=5, size=10)
measures harm-encounter rate ~0.03-0.04/step -- comfortably above the n>=10
floor over even 50 eval episodes. The original's near-zero counts therefore
did not come from an insufficiently hazardous environment; they came from
the trained policy (both conditions, but ON especially, since injected
proposals are terrain-guided and increasingly avoid mapped hazards as
training accumulates) actively steering away from hazards at nav_bias=0.25
(only 25% of actions are forced-random; 75% are policy-selected and
increasingly avoidant). The fix is therefore NOT "more hazards" (a
calibration sweep at num_hazards in {5, 9, 12} found num_hazards=5 gives the
HIGHEST harm-encounter rate under random policy of the three, likely because
denser hazard fields truncate episodes earlier via faster episode-ending
events, reducing total exposure) -- it is more forced-random exploration
(nav_bias) plus more eval steps, so that C4's floor is cleared by
environmental exposure rather than by the very avoidance behaviour MECH-022
predicts. nav_bias is raised only moderately (0.25 -> 0.40) so condition-
dependent (policy-selected) behaviour still makes up the majority of each
episode -- the C1/C3 test is still a test of the policy, not swamped by a
forced-random floor.

Design changes vs V3-EXQ-190 (env/training/env exposure unchanged otherwise):
  - nav_bias:        0.25 -> 0.40  (more forced-random exploration -> more
                      harm contacts recorded regardless of policy avoidance)
  - eval_episodes:   50   -> 100   (more eval exposure -> more harm contacts,
                      more E3 trajectory-score samples for C3's per-seed test)
  - warmup_episodes: 150  -> 220   (more harm-supervision signal for E3's
                      harm_eval head before eval, since warmup previously
                      also starved of harm contacts under nav_bias=0.25)
  - seeds:           (42, 123) -> (42, 123, 7)  (3 seeds instead of 2, for a
                      more reliable read of "consistent across ALL seeds",
                      which is what C1/C2/C3 require)
  - num_hazards, num_resources, steps_per_episode, self_dim, world_dim, lr,
    alpha_world, alpha_self: UNCHANGED from V3-EXQ-190 (calibration found
    hazard density was not the bottleneck; changing it would also change
    the env this claim is being tested against, without evidence it helps).

Per-chip guidance (chip chip-20260808-mech022-v3exq190-successor) also asked
whether to additionally vary control-plane state directly rather than a
binary injection ON/OFF, to more directly test the "gated by control plane"
half of MECH-022. NOT done here: the chip's own guidance treats a clean,
well-powered C3 retest as a legitimate, sufficient first redesign, and
adding a third design axis here would roughly double runtime and design
complexity for a claim that has not yet cleared even the binary test with
adequate power. If this run passes or produces a clean, well-powered FAIL,
a control-plane-state axis is the natural V3-EXQ-190b/191 follow-up.

Conditions (unchanged from V3-EXQ-190)
----------
HYPOTHESIS_INJECTION_ON:
  - HippocampalModule.propose_trajectories() generates terrain-guided
    trajectory hypotheses via CEM in action-object space (SD-004).
  - E3 evaluates these proposals and selects the best via score_trajectory().
HYPOTHESIS_INJECTION_OFF:
  - HippocampalModule bypassed. E3 evaluates only random trajectories
    (uniform random action sequences rolled out through E2).

Seeds: [42, 123, 7] (matched -- same env per seed across conditions)
Env:   CausalGridWorld size=10, 5 hazards, 5 resources, nav_bias=0.40
Warmup: 220 episodes x 200 steps
Eval:   100 episodes x 200 steps
Estimated runtime: ~3.5-4.5 hrs any cloud machine (6 seed x condition cells,
  each ~1.75x the per-cell cost of V3-EXQ-190's ~20 min cells)

Pre-registered thresholds (UNCHANGED from V3-EXQ-190 -- same claim test)
--------------------------
C1: harm_gap = harm_rate_OFF - harm_rate_ON >= THRESH_HARM_GAP (all seeds)
C2: residue_gap = mean_residue_OFF - mean_residue_ON >= 0 (all seeds)
C3: trajectory_quality_gap = traj_score_ON < traj_score_OFF (all seeds)
C4: n_harm_min >= THRESH_N_HARM both conditions all seeds (data quality)
C5: hippo_proposals_nondegenerate: variance of hippocampal proposal scores
    in INJECTION_ON must exceed THRESH_SCORE_VAR

Interpretation:
  C1+C2+C3+C4+C5 PASS: MECH-022 SUPPORTED.
  C3 fail, C4 pass: well-powered null on the load-bearing criterion -- E3
    genuinely does not rate hippocampal proposals higher than random, not
    an artifact of insufficient data (unlike V3-EXQ-190's inconclusive read).
  C4 fail: still underpowered even at these settings -- escalate nav_bias
    or eval_episodes further in a V3-EXQ-190b.
"""

import sys
import random
import math
import time
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
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._metrics import check_degeneracy
from experiments.pack_writer import write_flat_manifest


EXPERIMENT_TYPE = "v3_exq_190a_mech022_hypothesis_injection_probe_wellpowered"
CLAIM_IDS = ["MECH-022"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds (unchanged from V3-EXQ-190 -- same claim test)
THRESH_HARM_GAP = 0.005
THRESH_N_HARM = 10
THRESH_SCORE_VAR = 1e-6

# Env / training configuration
BODY_OBS_DIM = 10
WORLD_OBS_DIM = 200  # CausalGridWorld size=10, use_proxy_fields=False
ACTION_DIM = 4


def _make_env(seed: int, num_hazards: int, num_resources: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=num_resources,
        num_hazards=num_hazards,
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
    """Generate random trajectory proposals (no hippocampal guidance)."""
    device = z_world.device
    batch_size = z_world.shape[0]
    trajectories: List[Trajectory] = []

    for _ in range(num_candidates):
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
    num_hazards: int,
    num_resources: int,
    zg_acc: ZGoalStreamAccumulator,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell.

    HYPOTHESIS_INJECTION_ON: HippocampalModule proposes trajectories.
    HYPOTHESIS_INJECTION_OFF: random proposals, same E3 evaluation.
    """
    cond_label = "INJECTION_ON" if injection_enabled else "INJECTION_OFF"
    config_slice = {
        "seed": seed,
        "injection_enabled": injection_enabled,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "self_dim": self_dim,
        "world_dim": world_dim,
        "lr": lr,
        "alpha_world": alpha_world,
        "alpha_self": alpha_self,
        "nav_bias": nav_bias,
        "num_hazards": num_hazards,
        "num_resources": num_resources,
        "env": "CausalGridWorld",
        "env_size": 10,
        "action_dim": ACTION_DIM,
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
    }

    print(f"\nSeed {seed} Condition {cond_label}", flush=True)

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        random.seed(seed)

        env = _make_env(seed, num_hazards, num_resources)

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

        _warmup_episodes = warmup_episodes
        _eval_episodes = eval_episodes
        if dry_run:
            _warmup_episodes = 3
            _eval_episodes = 2

        print(
            f"\n[V3-EXQ-190a] TRAIN {cond_label} seed={seed}"
            f" warmup={_warmup_episodes} eval={_eval_episodes}"
            f" nav_bias={nav_bias} num_hazards={num_hazards}",
            flush=True,
        )
        agent.train()

        for ep in range(_warmup_episodes):
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

                action = agent.select_action(candidates, ticks, temperature=1.0)

                if z_self_t is not None:
                    agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    e1_opt.step()

                e2_loss = agent.compute_e2_loss()
                if e2_loss.requires_grad:
                    e2_opt.zero_grad()
                    e2_loss.backward()
                    e2_opt.step()

                if random.random() < nav_bias:
                    action = torch.randint(0, ACTION_DIM, (1, ACTION_DIM), dtype=torch.float32)

                _, reward, done, info, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0
                ep_harm += abs(harm_signal)

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
                    f"  [train] {cond_label} seed={seed} ep {ep+1}/{_warmup_episodes}"
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
        hippo_score_vars: List[float] = []

        for ep in range(_eval_episodes):
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

                    # nav_bias also applies in eval -- this is the deliberate C4 fix:
                    # forced-random exploration generates harm contacts regardless of
                    # the trained policy's own avoidance behaviour.
                    if random.random() < nav_bias:
                        action = torch.randint(0, ACTION_DIM, (1, ACTION_DIM), dtype=torch.float32)

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

        zg_acc.observe(agent)

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

        row = {
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
        cell.stamp(row)

    # Per-cell completion signal for the runner's progress bar (runs_done / ETA) --
    # NOT a scientific verdict. The scientific PASS/FAIL is only meaningful once
    # all seeds x conditions are aggregated in run()'s C1-C5 criteria below.
    print("verdict: PASS", flush=True)
    return row


def run(
    seeds: Tuple[int, ...] = (42, 123, 7),
    warmup_episodes: int = 220,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.40,
    num_hazards: int = 5,
    num_resources: int = 5,
    dry_run: bool = False,
) -> dict:
    """MECH-022 hypothesis injection probe: INJECTION_ON vs INJECTION_OFF (well-powered).

    Paired design: each seed runs both conditions (same env, same init).
    See module docstring for the redesign rationale vs V3-EXQ-190.
    """
    t0 = time.perf_counter()
    print(
        f"\n[V3-EXQ-190a] MECH-022 Hypothesis Injection Probe (well-powered)"
        f" seeds={list(seeds)}",
        flush=True,
    )

    results_on: List[Dict] = []
    results_off: List[Dict] = []
    arm_results: List[Dict] = []
    zg_acc = ZGoalStreamAccumulator()

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
                num_hazards=num_hazards,
                num_resources=num_resources,
                zg_acc=zg_acc,
                dry_run=dry_run,
            )
            arm_results.append(r)
            if injection_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    per_seed_harm_gap: List[float] = []
    per_seed_residue_gap: List[float] = []
    per_seed_traj_gap: List[float] = []

    for r_on in results_on:
        matching = [r for r in results_off if r["seed"] == r_on["seed"]]
        if matching:
            r_off = matching[0]
            per_seed_harm_gap.append(r_off["harm_rate"] - r_on["harm_rate"])
            per_seed_residue_gap.append(r_off["mean_residue"] - r_on["mean_residue"])
            per_seed_traj_gap.append(r_on["mean_traj_score"] - r_off["mean_traj_score"])

    mean_harm_on = _avg(results_on, "harm_rate")
    mean_harm_off = _avg(results_off, "harm_rate")
    mean_residue_on = _avg(results_on, "mean_residue")
    mean_residue_off = _avg(results_off, "mean_residue")
    mean_traj_on = _avg(results_on, "mean_traj_score")
    mean_traj_off = _avg(results_off, "mean_traj_score")

    n_harm_min = min(r["harm_events"] for r in results_on + results_off)

    c1_pass = (
        len(per_seed_harm_gap) > 0
        and all(g >= THRESH_HARM_GAP for g in per_seed_harm_gap)
    )
    c2_pass = (
        len(per_seed_residue_gap) > 0
        and all(g >= 0.0 for g in per_seed_residue_gap)
    )
    c3_pass = (
        len(per_seed_traj_gap) > 0
        and all(g < 0.0 for g in per_seed_traj_gap)
    )
    c4_pass = n_harm_min >= THRESH_N_HARM
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
        f"\n[V3-EXQ-190a] Results:"
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
            " -- insufficient harm contacts even at nav_bias=0.40/eval=100;"
            " escalate further in a V3-EXQ-190b"
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
            f"MECH-022 SUPPORTED (well-powered): hippocampal hypothesis injection"
            f" produces measurable behavioral improvement over random proposals,"
            f" with a well-powered C3 read (n_harm_min={n_harm_min} across"
            f" {len(seeds)} seeds, up from n_harm_min=0 in V3-EXQ-190)."
            f" INJECTION_ON: harm={mean_harm_on:.4f} residue={mean_residue_on:.4f}"
            f" traj_score={mean_traj_on:.4f}."
            f" INJECTION_OFF: harm={mean_harm_off:.4f} residue={mean_residue_off:.4f}"
            f" traj_score={mean_traj_off:.4f}."
            f" per-seed harm_gap={[round(g, 5) for g in per_seed_harm_gap]}"
            f" residue_gap={[round(g, 5) for g in per_seed_residue_gap]}"
            f" traj_gap={[round(g, 5) for g in per_seed_traj_gap]}."
        )
    elif criteria_met >= 3 and c4_pass:
        interpretation = (
            f"Partial support for MECH-022, well-powered (n_harm_min={n_harm_min}"
            f" >= {THRESH_N_HARM}): directional improvement observed on"
            f" {criteria_met}/5 criteria."
            f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}."
        )
    elif not c4_pass:
        interpretation = (
            f"STILL INCONCLUSIVE despite the power increase: n_harm_min={n_harm_min}"
            f" < {THRESH_N_HARM} even at nav_bias=0.40, eval_episodes=100."
            f" The trained policy avoids hazards more effectively than the"
            f" nav_bias floor injects random exposure. Escalate nav_bias or"
            f" eval_episodes further in a V3-EXQ-190b."
        )
    else:
        interpretation = (
            f"MECH-022 NOT SUPPORTED, well-powered read (n_harm_min={n_harm_min}"
            f" >= {THRESH_N_HARM} across {len(seeds)} seeds): hippocampal"
            f" injection does not produce measurable improvement over random"
            f" proposals at this training scale, and this is no longer a"
            f" data-quality artifact as it was in V3-EXQ-190."
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
        f"# V3-EXQ-190a -- MECH-022 Hypothesis Injection Probe (well-powered)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-022\n"
        f"**Supersedes:** V3-EXQ-190\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** HYPOTHESIS_INJECTION_ON vs HYPOTHESIS_INJECTION_OFF\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorld size=10, {num_hazards} hazards, {num_resources} resources"
        f" nav_bias={nav_bias}\n\n"
        f"## Design\n\n"
        f"Well-powered successor to V3-EXQ-190 (C4 FAILed at n_harm_min=0)."
        f" Same claim test (paired INJECTION_ON vs INJECTION_OFF, same"
        f" thresholds), with nav_bias raised 0.25->{nav_bias}, eval_episodes"
        f" raised 50->{eval_episodes}, warmup raised 150->{warmup_episodes},"
        f" and seeds extended from 2 to {len(seeds)}. See module docstring for"
        f" the calibration that ruled out hazard density as the bottleneck.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: per-seed harm_gap (OFF-ON) >= {THRESH_HARM_GAP} (all seeds)\n"
        f"C2: per-seed residue_gap (OFF-ON) >= 0 (all seeds, directional)\n"
        f"C3: per-seed traj_gap (ON-OFF) < 0 (all seeds, lower=better)\n"
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
        f"| C1: harm_gap >= {THRESH_HARM_GAP} (all seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {[round(g, 5) for g in per_seed_harm_gap]} |\n"
        f"| C2: residue_gap >= 0 (all seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(g, 5) for g in per_seed_residue_gap]} |\n"
        f"| C3: traj_gap < 0 (all seeds)"
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
        "num_hazards":               float(num_hazards),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    result = {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "fatal_error_count": 0,
        "supersedes": "v3_exq_190_mech022_hypothesis_injection_probe",
        "arm_results": arm_results,
    }

    full_config = {
        "seeds": list(seeds),
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "self_dim": self_dim,
        "world_dim": world_dim,
        "lr": lr,
        "alpha_world": alpha_world,
        "alpha_self": alpha_self,
        "nav_bias": nav_bias,
        "num_hazards": num_hazards,
        "num_resources": num_resources,
        "conditions": ["HYPOTHESIS_INJECTION_ON", "HYPOTHESIS_INJECTION_OFF"],
    }
    stamp_recording_core(
        result,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_acc.stats(),
    )

    # Non-degeneracy self-report: the two load-bearing discriminative metrics
    # are per-seed harm_gap (C1) and per-seed traj_gap (C3, the load-bearing
    # criterion this redesign exists to power properly). A run where either
    # is pinned across all seeds (e.g. all-zero harm_gap because n_harm_min=0
    # again) should scoring-exclude rather than silently count as a genuine
    # weakens/mixed read.
    result.update(check_degeneracy({
        "per_seed_harm_gap": per_seed_harm_gap,
        "per_seed_traj_gap": per_seed_traj_gap,
    }))

    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int,   nargs="+", default=[42, 123, 7])
    parser.add_argument("--warmup",      type=int,   default=220)
    parser.add_argument("--eval-eps",    type=int,   default=100)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self",  type=float, default=0.3)
    parser.add_argument("--nav-bias",    type=float, default=0.40)
    parser.add_argument("--num-hazards", type=int,   default=5)
    parser.add_argument("--num-resources", type=int, default=5)
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
        num_hazards=args.num_hazards,
        num_resources=args.num_resources,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["outcome"] = result["status"]
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

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        script_path=Path(__file__),
        stamp=False,  # already stamped (stamp_recording_core called in run())
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
