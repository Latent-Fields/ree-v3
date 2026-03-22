"""
V3-EXQ-046 — ARC-007 Path Memory Ablation (Proper)

Claims: ARC-007, SD-004

Prerequisite: EXQ-042 PASS (terrain prior training works; hippo_quality_gap > 0).

Motivation (2026-03-19):
  V2 ARC-007 FAIL used a crude z_gamma observation-slice proxy for the hippocampal
  map. EXQ-042 PASS showed that the full HippocampalModule with terrain prior training
  achieves hippo_quality_gap > 0 (proposals navigate to lower-residue terrain).

  This experiment now tests the ABLATION variant of ARC-007:
    If the residue field (path memory) is zeroed after training, hippocampal
    proposals should degrade toward random quality because the terrain map loses
    its harm-location geometry.

  ARC-007 (STRICT / Q-020 decision): proposals are value-flat (scored only by
  residue field, no independent hippocampal value head). If ablating the residue
  field collapses hippo_quality_gap to ≤ 0, this confirms the hippocampal module
  itself has NO independent value computation — it relies entirely on residue geometry.

Protocol:
  Phase 1 (train, 600 eps): Full pipeline + terrain prior training (same as EXQ-042).
  Phase 2 (eval A, 50 eps): Measure hippo_quality_gap with intact residue field.
  Phase 3 (ablation): Zero the residue field output by monkeypatching evaluate() and
    evaluate_trajectory() to return zeros. This simulates complete path memory loss.
  Phase 4 (eval B, 50 eps): Measure hippo_quality_gap with ablated residue field.
  Phase 5 (restore): Remove monkeypatch, restore original evaluate methods.
  Phase 6 (eval C, 30 eps): Verify quality gap recovers toward Phase 2 value.

PASS criteria (ALL must hold):
  C1: hippo_quality_gap_intact > 0         (terrain-guided proposals are better)
  C2: hippo_quality_gap_ablated <= hippo_quality_gap_intact * 0.5
                                             (ablation degrades by at least 50%)
  C3: hippo_quality_gap_restored >= hippo_quality_gap_ablated
                                             (restoration doesn't make things worse)
  C4: calibration_gap_approach > 0.03      (E3 calibrated throughout)
  C5: No fatal errors

Informational:
  ablation_degradation = hippo_quality_gap_intact - hippo_quality_gap_ablated
  Larger = path memory more important for navigation quality.
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


EXPERIMENT_TYPE = "v3_exq_046_arc007_path_memory_ablation"
CLAIM_IDS = ["ARC-007", "SD-004"]

APPROACH_TTYPES = {"hazard_approach"}
N_RANDOM_COMPARE = 8
CANDIDATE_HORIZON = 5


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _ablate_residue_field(agent: REEAgent) -> Tuple:
    """Zero residue field output by monkeypatching evaluate methods. Returns originals."""
    orig_eval = agent.residue_field.evaluate
    orig_eval_traj = agent.residue_field.evaluate_trajectory

    def zeroed_eval(z_world: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z_world.shape[0], 1, device=z_world.device)

    def zeroed_eval_traj(z_world_seq: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z_world_seq.shape[0], device=z_world_seq.device)

    agent.residue_field.evaluate = zeroed_eval
    agent.residue_field.evaluate_trajectory = zeroed_eval_traj
    return orig_eval, orig_eval_traj


def _restore_residue_field(agent: REEAgent, orig_eval, orig_eval_traj) -> None:
    """Restore original residue field evaluation methods."""
    agent.residue_field.evaluate = orig_eval
    agent.residue_field.evaluate_trajectory = orig_eval_traj


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    terrain_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000
    terrain_losses_early: List[float] = []
    terrain_losses_late:  List[float] = []
    total_harm = 0
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z    = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

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
                    if ep >= num_episodes - 100:
                        terrain_losses_late.append(t_loss_val)
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # World-forward training buffer
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            # World-forward loss
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b  = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                zw_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(zw_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            # Harm eval (balanced)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(theta_z.expand(zw_b.shape[0], -1))
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {
        "terrain_losses_early": terrain_losses_early,
        "terrain_losses_late":  terrain_losses_late,
        "total_harm": total_harm,
        "e3_tick_total": e3_tick_total,
        "wf_buf": wf_buf,
    }


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
        return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_hippo_quality(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    label: str,
) -> Dict:
    """Measure hippo_quality_gap and calibration_gap_approach."""
    agent.eval()
    hippo_residue_scores:  List[float] = []
    random_residue_scores: List[float] = []
    approach_scores: List[float] = []
    none_scores:     List[float] = []
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent   = agent.sense(obs_body, obs_world)
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                theta_z = agent.theta_buffer.summary()
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            if ticks.get("e3_tick", False) and candidates:
                try:
                    with torch.no_grad():
                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action

                        # Generate random candidates for comparison
                        random_trajs = agent.e2.generate_candidates_random(
                            latent.z_self, latent.z_world,
                            num_candidates=N_RANDOM_COMPARE, horizon=CANDIDATE_HORIZON,
                        )
                        hippo_trajs = candidates[:N_RANDOM_COMPARE]

                        def _mean_residue(trajs) -> float:
                            scores = []
                            for t in trajs:
                                if t is None:
                                    continue
                                ws = t.get_world_state_sequence()
                                if ws is not None and ws.numel() > 0:
                                    val = float(
                                        agent.residue_field.evaluate_trajectory(ws).mean().item()
                                    )
                                    scores.append(val)
                            return _mean_safe(scores)

                        hr = _mean_residue(hippo_trajs)
                        rr = _mean_residue(random_trajs)
                        if hr != 0.0 or rr != 0.0:
                            hippo_residue_scores.append(hr)
                            random_residue_scores.append(rr)
                except Exception:
                    fatal += 1
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            # E3 calibration probe
            try:
                with torch.no_grad():
                    theta_z = agent.theta_buffer.summary()
                    score = float(agent.e3.harm_eval(theta_z).item())
                    if ttype in APPROACH_TTYPES:
                        approach_scores.append(score)
                    elif ttype == "none":
                        none_scores.append(score)
            except Exception:
                pass

            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()
            if done:
                break

    mean_hippo  = _mean_safe(hippo_residue_scores)
    mean_random = _mean_safe(random_residue_scores)
    quality_gap = mean_random - mean_hippo  # positive = hippo better (avoids residue)
    cal_gap = _mean_safe(approach_scores) - _mean_safe(none_scores)

    print(
        f"  [{label}] hippo_quality_gap={quality_gap:.6f}"
        f"  (hippo={mean_hippo:.6f}  random={mean_random:.6f})"
        f"  cal_gap_approach={cal_gap:.4f}  n_hippo={len(hippo_residue_scores)}",
        flush=True,
    )

    return {
        "hippo_quality_gap":  quality_gap,
        "mean_hippo_residue": mean_hippo,
        "mean_random_residue": mean_random,
        "calibration_gap_approach": cal_gap,
        "n_e3_evals": len(hippo_residue_scores),
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 600,
    eval_episodes: int = 50,
    eval_restore_episodes: int = 30,
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
        and "harm_eval_head" not in [n for n, pp in agent.named_parameters() if id(pp) == id(p)]
    ]
    optimizer       = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer    = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    print(
        f"[V3-EXQ-046] ARC-007 Path Memory Ablation\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  restore={eval_restore_episodes}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}",
        flush=True,
    )

    # Phase 1: Train
    train_out = _train(
        agent, env, optimizer, wf_optimizer, terrain_optimizer,
        harm_eval_optimizer, warmup_episodes, steps_per_episode, world_dim,
    )

    terrain_loss_initial = _mean_safe(train_out["terrain_losses_early"])
    terrain_loss_final   = _mean_safe(train_out["terrain_losses_late"])
    world_forward_r2     = _compute_world_forward_r2(agent, train_out["wf_buf"])

    print(
        f"  terrain_loss: {terrain_loss_initial:.4f} → {terrain_loss_final:.4f}"
        f"  world_forward_r2={world_forward_r2:.4f}",
        flush=True,
    )

    # Phase 2: Eval A — intact residue field
    print("\n[V3-EXQ-046] Eval A (intact residue field)...", flush=True)
    eval_a = _eval_hippo_quality(agent, env, eval_episodes, steps_per_episode, world_dim, "intact")

    # Phase 3: Ablate residue field
    print("\n[V3-EXQ-046] Ablating residue field...", flush=True)
    orig_eval, orig_eval_traj = _ablate_residue_field(agent)

    # Phase 4: Eval B — ablated residue field
    print("[V3-EXQ-046] Eval B (ablated residue field)...", flush=True)
    eval_b = _eval_hippo_quality(agent, env, eval_episodes, steps_per_episode, world_dim, "ablated")

    # Phase 5: Restore residue field
    _restore_residue_field(agent, orig_eval, orig_eval_traj)
    print("[V3-EXQ-046] Residue field restored.", flush=True)

    # Phase 6: Eval C — restored
    print("[V3-EXQ-046] Eval C (restored residue field)...", flush=True)
    eval_c = _eval_hippo_quality(
        agent, env, eval_restore_episodes, steps_per_episode, world_dim, "restored"
    )

    # PASS / FAIL
    c1_pass = eval_a["hippo_quality_gap"] > 0
    c2_pass = eval_b["hippo_quality_gap"] <= eval_a["hippo_quality_gap"] * 0.5
    c3_pass = eval_c["hippo_quality_gap"] >= eval_b["hippo_quality_gap"]
    c4_pass = eval_a["calibration_gap_approach"] > 0.03
    c5_pass = (eval_a["fatal_errors"] + eval_b["fatal_errors"] + eval_c["fatal_errors"]) == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    ablation_degradation = eval_a["hippo_quality_gap"] - eval_b["hippo_quality_gap"]

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: hippo_quality_gap_intact={eval_a['hippo_quality_gap']:.6f} <= 0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: hippo_quality_gap_ablated={eval_b['hippo_quality_gap']:.6f}"
            f" > {eval_a['hippo_quality_gap'] * 0.5:.6f} (50% of intact)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: quality_restored={eval_c['hippo_quality_gap']:.6f}"
            f" < ablated={eval_b['hippo_quality_gap']:.6f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: calibration_gap_approach={eval_a['calibration_gap_approach']:.4f} <= 0.03"
        )
    if not c5_pass:
        failure_notes.append("C5 FAIL: fatal errors occurred")

    print(f"\nV3-EXQ-046 verdict: {status}  ({criteria_met}/5)", flush=True)
    print(f"  ablation_degradation: {ablation_degradation:.6f}", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "terrain_loss_initial":           float(terrain_loss_initial),
        "terrain_loss_final":             float(terrain_loss_final),
        "world_forward_r2":               float(world_forward_r2),
        "hippo_quality_gap_intact":       float(eval_a["hippo_quality_gap"]),
        "hippo_quality_gap_ablated":      float(eval_b["hippo_quality_gap"]),
        "hippo_quality_gap_restored":     float(eval_c["hippo_quality_gap"]),
        "ablation_degradation":           float(ablation_degradation),
        "calibration_gap_approach":       float(eval_a["calibration_gap_approach"]),
        "mean_hippo_residue_intact":      float(eval_a["mean_hippo_residue"]),
        "mean_random_residue_intact":     float(eval_a["mean_random_residue"]),
        "mean_hippo_residue_ablated":     float(eval_b["mean_hippo_residue"]),
        "mean_random_residue_ablated":    float(eval_b["mean_random_residue"]),
        "e3_tick_total":                  float(train_out["e3_tick_total"]),
        "total_harm_train":               float(train_out["total_harm"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-046 — ARC-007 Path Memory Ablation (Proper)

**Status:** {status}
**Claims:** ARC-007, SD-004
**Prerequisite:** EXQ-042 PASS (terrain prior training validates hippo_quality_gap > 0)
**alpha_world:** {alpha_world}  (SD-008)
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps intact/ablated + {eval_restore_episodes} restored
**Seed:** {seed}

## Motivation

V2 ARC-007 FAIL used z_gamma proxy; EXQ-042 validated terrain prior on V3 substrate.
This experiment ablates the residue field (path memory) to test whether hippocampal
navigation quality degrades without terrain geometry. ARC-007 STRICT (Q-020): if
ablation causes quality collapse, hippocampal module has no independent value head.

## Navigation Quality: Intact vs Ablated vs Restored

| Phase | hippo_quality_gap | calibration_gap_approach |
|-------|------------------|--------------------------|
| Intact (trained) | {eval_a['hippo_quality_gap']:.6f} | {eval_a['calibration_gap_approach']:.4f} |
| Ablated (residue=0) | {eval_b['hippo_quality_gap']:.6f} | {eval_b['calibration_gap_approach']:.4f} |
| Restored | {eval_c['hippo_quality_gap']:.6f} | {eval_c['calibration_gap_approach']:.4f} |

- **ablation_degradation**: {ablation_degradation:.6f}
- world_forward_r2: {world_forward_r2:.4f}
- terrain_loss: {terrain_loss_initial:.4f} → {terrain_loss_final:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: hippo_quality_gap_intact > 0 | {"PASS" if c1_pass else "FAIL"} | {eval_a['hippo_quality_gap']:.6f} |
| C2: ablated <= intact x 0.5 (ablation degrades ≥50%) | {"PASS" if c2_pass else "FAIL"} | {eval_b['hippo_quality_gap']:.6f} vs {eval_a['hippo_quality_gap'] * 0.5:.6f} |
| C3: restored >= ablated (no further degradation) | {"PASS" if c3_pass else "FAIL"} | {eval_c['hippo_quality_gap']:.6f} vs {eval_b['hippo_quality_gap']:.6f} |
| C4: calibration_gap_approach > 0.03 | {"PASS" if c4_pass else "FAIL"} | {eval_a['calibration_gap_approach']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | 0 |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

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
        "fatal_error_count": eval_a["fatal_errors"] + eval_b["fatal_errors"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--warmup",        type=int,   default=600)
    parser.add_argument("--eval-eps",      type=int,   default=50)
    parser.add_argument("--eval-restore",  type=int,   default=30)
    parser.add_argument("--steps",         type=int,   default=200)
    parser.add_argument("--alpha-world",   type=float, default=0.9)
    parser.add_argument("--harm-scale",    type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        eval_restore_episodes=args.eval_restore,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
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
