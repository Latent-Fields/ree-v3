"""
V3-EXQ-044 -- SD-003 Trajectory Attribution: Fixed Sequential Training

Claims: SD-003, MECH-102

Root cause of EXQ-043 FAIL: simultaneous terrain_prior + E3 harm_eval training.
As terrain_prior improves, the agent spends more time at approach states
(navigating toward objectives through hazard gradients). harm_buf_pos fills
with approach-heavy states; harm_buf_neg sees fewer genuine "none" states.
E3 learns harm=1 for almost everything -> collapses to none=0.90.
With E3 uniform-high, all trajectory J(ζ) scores are nearly equal →
causal_sig stays at noise floor (~0.0004) regardless of true landscape.

Fix: three separated training phases with no cross-contamination:

  Phase 1 -- terrain training (400 eps):
    Train terrain_prior via E3 behavioral cloning.
    E2.world_forward trained simultaneously (wf data accumulates naturally).
    E3.harm_eval NOT updated -- weights frozen for this phase.
    Goal: terrain_prior learns z_world -> action_object mapping.

  Phase 2 -- E3 calibration (200 eps):
    Freeze terrain_prior. Random policy ensures balanced approach/none distribution.
    Train E3.harm_eval on theta-averaged z_world (MECH-089).
    Goal: clean calibrated E3 with stable data -- no terrain-induced imbalance.

  Phase 3 -- attribution eval (100 eps):
    Freeze ALL weights. Full pipeline with E3 selection.
    Generate N=16 hippocampal candidates per E3 tick.
    causal_sig = mean(E3_score(cf_trajs)) - E3_score(selected_traj)
    Measure by transition_type.

C6 (calibration gate): mean_harm_eval_none < 0.1
  If E3 collapses again, result is immediately flagged invalid.
  This prevents misreading a calibration failure as an attribution failure.

PASS criteria (ALL must hold):
  C1: causal_sig_approach > causal_sig_none
  C2: causal_sig_contact > causal_sig_approach   (MECH-102 escalation)
  C3: causal_sig_approach > 0.001                (above noise floor)
  C4: cal_gap_approach > 0.03                    (E3 calibrated)
  C5: n_approach_eval >= 30
  C6: mean_harm_eval_none < 0.1                  (no calibration collapse)

Claims tested:
  SD-003: trajectory attribution via hippocampal counterfactuals
  MECH-102: energy escalation ladder (contact > approach > none)
"""

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


EXPERIMENT_TYPE = "v3_exq_044_sd003_attribution_fixed"
CLAIM_IDS = ["SD-003", "MECH-102"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES  = {"agent_caused_hazard", "env_caused_hazard"}
N_CANDIDATES    = 16
CANDIDATE_HORIZON = 5


def run(
    seed: int = 0,
    terrain_episodes: int = 400,
    calibration_episodes: int = 200,
    eval_episodes: int = 100,
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

    # ── Separate optimizers per training phase ───────────────────────────────
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
    harm_eval_param_ids = set(
        id(p) for p in agent.e3.harm_eval_head.parameters()
    )

    # Phase 1: everything EXCEPT harm_eval
    main_params_p1 = [
        p for p in agent.parameters()
        if id(p) not in wf_param_ids
        and id(p) not in terrain_param_ids
        and id(p) not in harm_eval_param_ids
    ]
    optimizer_p1 = optim.Adam(main_params_p1, lr=lr)

    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )
    # Phase 2: harm_eval only
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    # ── Shared buffers ───────────────────────────────────────────────────────
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    # Terrain loss tracking (Phase 1)
    terrain_losses_early: List[float] = []
    terrain_losses_late:  List[float] = []

    total_episodes = terrain_episodes + calibration_episodes
    print(
        f"[V3-EXQ-044] Sequential training: {terrain_episodes} terrain eps + "
        f"{calibration_episodes} E3-calibration eps + {eval_episodes} eval eps\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  N_candidates={N_CANDIDATES}",
        flush=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 1 -- Terrain training (E3.harm_eval frozen)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 1] Terrain training ({terrain_episodes} eps) -- E3.harm_eval frozen",
          flush=True)
    agent.train()
    e3_tick_total = 0

    for ep in range(terrain_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates   = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z      = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

                # Terrain_prior behavioral cloning (Phase 1 only)
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
                    if ep >= terrain_episodes - 100:
                        terrain_losses_late.append(t_loss_val)
            else:
                action = agent._last_action
                if action is None:
                    action_idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

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

            # E1 + main (no harm_eval)
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer_p1.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params_p1, 1.0)
                optimizer_p1.step()

            # E2 world_forward
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

        if (ep + 1) % 100 == 0 or ep == terrain_episodes - 1:
            t_early = sum(terrain_losses_early) / max(1, len(terrain_losses_early))
            t_late  = sum(terrain_losses_late)  / max(1, len(terrain_losses_late))
            print(
                f"  [P1] ep {ep+1}/{terrain_episodes}  e3_ticks={e3_tick_total}  "
                f"terrain_early={t_early:.4f}  terrain_late={t_late:.4f}",
                flush=True,
            )

    terrain_loss_initial = sum(terrain_losses_early) / max(1, len(terrain_losses_early))
    terrain_loss_final   = sum(terrain_losses_late)  / max(1, len(terrain_losses_late))

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 2 -- E3 calibration (terrain_prior frozen, random policy)
    # ────────────────────────────────────────────────────────────────────────
    print(
        f"\n[Phase 2] E3 calibration ({calibration_episodes} eps) -- "
        f"terrain_prior frozen, random policy",
        flush=True,
    )

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000

    for ep in range(calibration_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            theta_z      = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            # Random policy -- ensures balanced ttype distribution
            action_idx = random.randint(0, env.action_dim - 1)
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, action_idx] = 1.0
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_pos = ttype in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0).detach())
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0).detach())
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]

            z_world_prev = z_world_curr
            action_prev  = action.detach()

            # E3.harm_eval training on balanced theta_z batches
            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k       = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z   = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z   = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels  = torch.cat([
                    torch.ones(k, 1), torch.zeros(k, 1)
                ], dim=0).to(agent.device)
                harm_loss = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e3.harm_eval_head.parameters()), 1.0
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == calibration_episodes - 1:
            # Quick calibration check
            with torch.no_grad():
                if harm_buf_pos and harm_buf_neg:
                    pos_sample = torch.stack(harm_buf_pos[-32:]).to(agent.device)
                    neg_sample = torch.stack(harm_buf_neg[-32:]).to(agent.device)
                    pos_score = float(agent.e3.harm_eval(pos_sample).mean().item())
                    neg_score = float(agent.e3.harm_eval(neg_sample).mean().item())
                    print(
                        f"  [P2] ep {ep+1}/{calibration_episodes}  "
                        f"harm_pos={pos_score:.3f}  harm_neg={neg_score:.3f}  "
                        f"gap={pos_score - neg_score:.3f}",
                        flush=True,
                    )

    # ────────────────────────────────────────────────────────────────────────
    # PHASE 3 -- Attribution eval (all weights frozen)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n[Phase 3] Attribution eval ({eval_episodes} eps) -- all weights frozen",
          flush=True)
    agent.eval()

    causal_sig_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }
    harm_eval_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }

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
                z_self  = latent.z_self.detach()

                # SD-003 trajectory attribution
                hippo_trajs = agent.hippocampal.propose_trajectories(
                    z_world=theta_z, z_self=z_self,
                    num_candidates=N_CANDIDATES, e1_prior=e1_prior.detach(),
                )

                if hippo_trajs:
                    scores = torch.stack([
                        agent.e3.score_trajectory(t).mean() for t in hippo_trajs
                    ])
                    selected_idx   = int(scores.argmin().item())
                    selected_score = float(scores[selected_idx].item())
                    cf_scores = [
                        float(scores[i].item())
                        for i in range(len(scores))
                        if i != selected_idx
                    ]
                    mean_cf_score = sum(cf_scores) / len(cf_scores) if cf_scores else selected_score
                    causal_sig    = mean_cf_score - selected_score
                    action        = hippo_trajs[selected_idx].actions[:, 0, :].detach()
                else:
                    causal_sig = 0.0
                    action_idx = random.randint(0, env.action_dim - 1)
                    action     = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

                harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                key = (
                    "approach" if ttype in APPROACH_TTYPES else
                    "contact"  if ttype in CONTACT_TTYPES  else
                    "none"
                )
                causal_sig_by_ttype[key].append(causal_sig)
                harm_eval_by_ttype[key].append(harm_score)

                if done:
                    break

    # ── Metrics ─────────────────────────────────────────────────────────────
    def mean_safe(lst: list, default: float = 0.0) -> float:
        return float(sum(lst) / len(lst)) if lst else default

    causal_sig_approach = mean_safe(causal_sig_by_ttype["approach"])
    causal_sig_contact  = mean_safe(causal_sig_by_ttype["contact"])
    causal_sig_none     = mean_safe(causal_sig_by_ttype["none"])
    mean_harm_approach  = mean_safe(harm_eval_by_ttype["approach"])
    mean_harm_none      = mean_safe(harm_eval_by_ttype["none"])
    cal_gap_approach    = mean_harm_approach - mean_harm_none
    n_approach = len(causal_sig_by_ttype["approach"])
    n_contact  = len(causal_sig_by_ttype["contact"])
    n_none     = len(causal_sig_by_ttype["none"])

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

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1 = causal_sig_approach > causal_sig_none
    c2 = causal_sig_contact  > causal_sig_approach
    c3 = causal_sig_approach > 0.001
    c4 = cal_gap_approach    > 0.03
    c5 = n_approach          >= 30
    c6 = mean_harm_none      < 0.1   # calibration gate

    all_pass = c1 and c2 and c3 and c4 and c5 and c6
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5, c6])

    failure_notes = []
    if not c6:
        failure_notes.append(
            f"C6 FAIL (calibration gate): mean_harm_eval_none={mean_harm_none:.4f} >= 0.1. "
            f"E3 collapsed again -- trajectory attribution results are INVALID. "
            f"Check Phase 2 calibration balance."
        )
    if not c1:
        failure_notes.append(
            f"C1 FAIL: causal_sig_approach={causal_sig_approach:.4f} <= "
            f"causal_sig_none={causal_sig_none:.4f}"
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: causal_sig_contact={causal_sig_contact:.4f} <= "
            f"causal_sig_approach={causal_sig_approach:.4f} (MECH-102 escalation absent)"
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: causal_sig_approach={causal_sig_approach:.4f} <= 0.001 "
            f"(trajectory scores don't diverge near hazards)"
        )
    if not c4:
        failure_notes.append(
            f"C4 FAIL: cal_gap_approach={cal_gap_approach:.4f} <= 0.03 (E3 not calibrated)"
        )
    if not c5:
        failure_notes.append(f"C5 FAIL: n_approach_eval={n_approach} < 30")

    print(f"\nV3-EXQ-044 verdict: {status}  ({n_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  causal_sig: none={causal_sig_none:.4f}  "
        f"approach={causal_sig_approach:.4f}  contact={causal_sig_contact:.4f}\n"
        f"  cal_gap_approach={cal_gap_approach:.4f}  "
        f"none={mean_harm_none:.4f}  approach={mean_harm_approach:.4f}\n"
        f"  n_approach={n_approach}  n_contact={n_contact}  n_none={n_none}\n"
        f"  terrain_loss: initial={terrain_loss_initial:.4f}  final={terrain_loss_final:.4f}\n"
        f"  wf_r2={wf_r2:.4f}  e3_ticks={e3_tick_total}",
        flush=True,
    )

    metrics = {
        "causal_sig_none":           float(causal_sig_none),
        "causal_sig_approach":       float(causal_sig_approach),
        "causal_sig_contact":        float(causal_sig_contact),
        "causal_sig_gap_approach":   float(causal_sig_approach - causal_sig_none),
        "causal_sig_gap_contact":    float(causal_sig_contact - causal_sig_approach),
        "calibration_gap_approach":  float(cal_gap_approach),
        "mean_harm_eval_approach":   float(mean_harm_approach),
        "mean_harm_eval_none":       float(mean_harm_none),
        "n_approach_eval":           float(n_approach),
        "n_contact_eval":            float(n_contact),
        "n_none_eval":               float(n_none),
        "terrain_loss_initial":      float(terrain_loss_initial),
        "terrain_loss_final":        float(terrain_loss_final),
        "world_forward_r2":          float(wf_r2),
        "e3_tick_total":             float(e3_tick_total),
        "alpha_world":               float(alpha_world),
        "crit1_pass": 1.0 if c1 else 0.0,
        "crit2_pass": 1.0 if c2 else 0.0,
        "crit3_pass": 1.0 if c3 else 0.0,
        "crit4_pass": 1.0 if c4 else 0.0,
        "crit5_pass": 1.0 if c5 else 0.0,
        "crit6_pass": 1.0 if c6 else 0.0,
        "criteria_met": float(n_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-044 -- SD-003 Trajectory Attribution (Fixed Sequential Training)

**Status:** {status}
**Claims:** SD-003, MECH-102
**World:** CausalGridWorldV2 (body={env.body_obs_dim}, world={env.world_obs_dim})
**alpha_world:** {alpha_world} (SD-008)  |  **Seed:** {seed}
**Predecessors:** EXQ-041 PASS (ThetaBuffer), EXQ-042 PASS (terrain training)
**Fixes:** EXQ-043 FAIL (E3 calibration collapse from simultaneous training)

## Design: Sequential Training Phases

**EXQ-043 failure mechanism:** Terrain_prior training shifts data distribution seen by
E3.harm_eval -- agent navigates toward objectives through hazard gradients, making
approach states dominate harm_buf_pos. E3 collapses to uniform-high output.

**Fix:** Three sequential phases with no cross-contamination:

| Phase | Episodes | What trains | Policy |
|---|---|---|---|
| P1: Terrain | {terrain_episodes} | terrain_prior + E2.world_forward + E1 | E3-guided |
| P2: E3 calib | {calibration_episodes} | E3.harm_eval only | Random (balanced) |
| P3: Eval | {eval_episodes} | Frozen | E3-guided |

## Terrain Training (Phase 1)

| Phase | terrain_loss |
|---|---|
| Initial (first 100 eps) | {terrain_loss_initial:.4f} |
| Final (last 100 eps) | {terrain_loss_final:.4f} |

## Attribution Results (Phase 3)

| Transition Type | causal_sig | n |
|---|---|---|
| none | {causal_sig_none:.4f} | {n_none} |
| approach | {causal_sig_approach:.4f} | {n_approach} |
| contact | {causal_sig_contact:.4f} | {n_contact} |

approach − none gap: {causal_sig_approach - causal_sig_none:.4f}
contact − approach gap: {causal_sig_contact - causal_sig_approach:.4f}

## E3 Calibration (Phase 3)

| Type | mean harm_eval |
|---|---|
| none | {mean_harm_none:.4f} |
| approach | {mean_harm_approach:.4f} |
calibration_gap_approach: {cal_gap_approach:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: causal_sig_approach > causal_sig_none | {"PASS" if c1 else "FAIL"} | {causal_sig_approach:.4f} vs {causal_sig_none:.4f} |
| C2: causal_sig_contact > causal_sig_approach (MECH-102) | {"PASS" if c2 else "FAIL"} | {causal_sig_contact:.4f} vs {causal_sig_approach:.4f} |
| C3: causal_sig_approach > 0.001 | {"PASS" if c3 else "FAIL"} | {causal_sig_approach:.4f} |
| C4: cal_gap_approach > 0.03 (E3 calibrated) | {"PASS" if c4 else "FAIL"} | {cal_gap_approach:.4f} |
| C5: n_approach_eval >= 30 | {"PASS" if c5 else "FAIL"} | {n_approach} |
| C6: mean_harm_eval_none < 0.1 (calibration gate) | {"PASS" if c6 else "FAIL"} | {mean_harm_none:.4f} |

Criteria met: {n_met}/6 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if n_met >= 4 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",                   type=int,   default=0)
    parser.add_argument("--terrain-episodes",       type=int,   default=400)
    parser.add_argument("--calibration-episodes",   type=int,   default=200)
    parser.add_argument("--eval-episodes",          type=int,   default=100)
    parser.add_argument("--steps",                  type=int,   default=200)
    parser.add_argument("--alpha-world",            type=float, default=0.9)
    parser.add_argument("--harm-scale",             type=float, default=0.02)
    parser.add_argument("--proximity-scale",        type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        terrain_episodes=args.terrain_episodes,
        calibration_episodes=args.calibration_episodes,
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
