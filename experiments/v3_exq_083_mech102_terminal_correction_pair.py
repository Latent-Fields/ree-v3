"""
V3-EXQ-083 -- MECH-102 Terminal Error-Correction Discriminative Pair

Claims: MECH-102
Proposal: EXP-0016 / EVB-0013

MECH-102 asserts: "Violence is a terminal error-correction mechanism triggered
only when all other channels fail." In CausalGridWorld terms:
  - Low-energy coordination pathways = approach-gradient avoidance (early correction)
  - Terminal error correction = contact events (maximum-energy intervention)

MECH-102 predicts: when low-energy pathways are BLOCKED (approach gradient
unavailable), the agent is forced to correct errors at the terminal stage
(contact), producing more contact events than when pathways are available.

Existing evidence (single-arm):
  EXQ-032b PASS (2026-03-19): Energy escalation ladder confirmed. Causal signature
    increases: none < approach < contact (wf_r2=0.948). Terminal events have the
    highest causal signature of any transition type.
  EXQ-059c PASS (2026-03-21): Contact rate ~10x lower when SD-010 harm stream
    active. Approach-gradient avoidance produces measurable early correction.

This discriminative pair directly tests the MECH-102 causal claim:

  COORDINATION_ENABLED  -- proximity_harm_scale=0.05: approach gradient available
                           E3 learns to score approach z_worlds as harmful.
                           Greedy harm-minimization policy uses early correction.
  COORDINATION_BLOCKED  -- proximity_harm_scale=0.0:  approach gradient absent.
                           E3 trained only on contact events (terminal signal).
                           Greedy policy cannot correct until contact.

Both conditions: same seeds, same training budget, E3-guided greedy eval policy
(argmin E3.harm_eval(E2.world_forward(z_world, a)) over all actions), NO
reafference (reafference_action_dim=0).

Mechanism under test:
  In COORDINATION_ENABLED, approach transitions produce harm_signal < 0, so
  E3 is trained to score approach z_world states as harmful. During eval, the
  greedy policy steers away from hazards at the approach stage (low-energy early
  correction). In COORDINATION_BLOCKED, only contact events produce harm_signal
  < 0. E3 cannot distinguish approach from safe locomotion. The greedy policy
  cannot act until contact -- all corrections are terminal.

Pre-registered primary discriminator (threshold >= 0.03):

  delta_contact_rate = contact_rate_BLOCKED - contact_rate_ENABLED
  contact_rate = contacts per eval episode (agent_caused + env_caused)

PASS criteria (ALL required):
  C1: contact_rate_BLOCKED  > contact_rate_ENABLED + 0.03
      (blocking pathways escalates terminal events)
  C2: approach_avoidance_ENABLED > approach_avoidance_BLOCKED + 0.05
      (gradient enables early correction; blocked condition cannot avoid at approach)
  C3: world_forward_r2_ENABLED > 0.10
      (E2.world_forward is trained sufficiently for greedy policy to be meaningful)
  C4: both seeds individually: contact_rate_BLOCKED > contact_rate_ENABLED (per-seed)

Decision scoring:
  retain_ree:       C1+C2+C3+C4 all pass
  hybridize:        C1 passes but C2 marginal (delta_avoidance 0.02-0.05)
  retire_ree_claim: C1 fails (blocking pathways does not increase terminal events)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_083_mech102_terminal_correction_pair"
CLAIM_IDS = ["MECH-102"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    proximity_harm_scale: float,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
) -> Dict:
    """
    Run one (seed, condition) cell.

    Training: random policy, E1+E2+E3 trained, E2.world_forward trained.
    Eval: greedy harm-minimization policy using E3.harm_eval(E2.world_forward(z, a)).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "COORDINATION_ENABLED" if proximity_harm_scale > 0 else "COORDINATION_BLOCKED"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=3,
        env_drift_prob=0.3,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6 if proximity_harm_scale > 0 else 0.0,
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
        reafference_action_dim=0,
    )
    agent = REEAgent(config)
    num_actions = env.action_dim

    # Separate optimizers: standard, harm_eval head, E2.world_forward
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    world_fwd_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)
    world_fwd_optimizer = optim.Adam(world_fwd_params, lr=1e-3)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf_zw: List[torch.Tensor] = []
    wf_buf_a: List[torch.Tensor] = []
    wf_buf_target: List[torch.Tensor] = []
    MAX_BUF = 2000

    train_counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    # --- TRAIN ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        zw_prev = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in train_counts:
                train_counts[ttype] += 1

            # E2.world_forward training buffer: (z_world_t, a_t) -> z_world_t+1
            if zw_prev is not None:
                wf_buf_zw.append(zw_prev)
                wf_buf_a.append(action.detach())
                wf_buf_target.append(z_world_curr)
                if len(wf_buf_zw) > MAX_BUF:
                    wf_buf_zw = wf_buf_zw[-MAX_BUF:]
                    wf_buf_a = wf_buf_a[-MAX_BUF:]
                    wf_buf_target = wf_buf_target[-MAX_BUF:]

            # E3 harm_eval balanced buffers
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E2.world_forward training (16-sample mini-batch)
            if len(wf_buf_zw) >= 16:
                k = min(32, len(wf_buf_zw))
                idx = torch.randperm(len(wf_buf_zw))[:k].tolist()
                zw_b = torch.cat([wf_buf_zw[i] for i in idx], dim=0)
                a_b  = torch.cat([wf_buf_a[i]  for i in idx], dim=0)
                t_b  = torch.cat([wf_buf_target[i] for i in idx], dim=0)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, t_b)
                if wf_loss.requires_grad:
                    world_fwd_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_fwd_optimizer.step()

            # E3 harm_eval balanced training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
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

            zw_prev = z_world_curr
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={train_counts['hazard_approach']}"
                f" agent_contact={train_counts['agent_caused_hazard']}"
                f" env_contact={train_counts['env_caused_hazard']}"
                f" buf_harm_pos={len(harm_buf_pos)}"
                f" wf_buf={len(wf_buf_zw)}",
                flush=True,
            )

    # world_forward R2 (validity check)
    world_forward_r2 = 0.0
    if len(wf_buf_target) >= 20:
        with torch.no_grad():
            n_test = min(200, len(wf_buf_zw))
            idx_t = torch.randperm(len(wf_buf_zw))[:n_test].tolist()
            zw_t = torch.cat([wf_buf_zw[i] for i in idx_t], dim=0)
            a_t  = torch.cat([wf_buf_a[i]  for i in idx_t], dim=0)
            tgt_t = torch.cat([wf_buf_target[i] for i in idx_t], dim=0)
            pred_t = agent.e2.world_forward(zw_t, a_t)
            ss_res = float(((tgt_t - pred_t) ** 2).sum().item())
            ss_tot = float(((tgt_t - tgt_t.mean(0)) ** 2).sum().item())
            world_forward_r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    print(
        f"  [world_fwd] seed={seed} cond={cond_label}"
        f" world_forward_r2={world_forward_r2:.4f}",
        flush=True,
    )

    # --- EVAL: greedy harm-minimization policy ---
    # At each step, enumerate all actions, predict z_world_next via E2.world_forward,
    # score with E3.harm_eval, choose action with lowest predicted harm.
    # This tests: does E3 steer away from hazards EARLY (approach stage) in ENABLED,
    # but fail to avoid until contact in BLOCKED?
    agent.eval()

    ep_contacts: List[float] = []
    ep_approach_steps: List[int] = []
    ep_approach_avoidances: List[int] = []
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_contact = 0
        ep_approach = 0
        ep_avoidance = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world

            # Greedy harm-minimization: pick action with lowest E3.harm_eval(E2(z, a))
            try:
                with torch.no_grad():
                    harm_scores = []
                    for a_idx in range(num_actions):
                        a_oh = _action_to_onehot(a_idx, num_actions, agent.device)
                        z_next = agent.e2.world_forward(z_world_curr, a_oh)
                        h = float(agent.e3.harm_eval(z_next).item())
                        harm_scores.append(h)
                best_action_idx = int(np.argmin(harm_scores))
            except Exception:
                best_action_idx = random.randint(0, num_actions - 1)
                n_fatal += 1

            action = _action_to_onehot(best_action_idx, num_actions, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_contact = ttype in ("agent_caused_hazard", "env_caused_hazard")
            is_approach = ttype == "hazard_approach"

            if is_contact:
                ep_contact += 1
            if is_approach:
                ep_approach += 1
                # "Avoidance" = agent did NOT choose the action that would have led
                # toward the hazard. Approximate: greedy chose best_action != argmax_harm.
                # Simple proxy: the action chosen had below-mean harm score.
                if harm_scores and harm_scores[best_action_idx] < float(np.mean(harm_scores)):
                    ep_avoidance += 1

            if done:
                break

        ep_contacts.append(float(ep_contact))
        ep_approach_steps.append(ep_approach)
        ep_approach_avoidances.append(ep_avoidance)

    contact_rate = float(np.mean(ep_contacts))
    n_approach_total = max(1, sum(ep_approach_steps))
    approach_avoidance_rate = float(sum(ep_approach_avoidances)) / n_approach_total

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" contact_rate={contact_rate:.4f}"
        f" approach_avoidance={approach_avoidance_rate:.4f}"
        f" n_approach_total={n_approach_total}"
        f" n_fatal={n_fatal}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "proximity_harm_scale": float(proximity_harm_scale),
        "contact_rate": float(contact_rate),
        "approach_avoidance_rate": float(approach_avoidance_rate),
        "world_forward_r2": float(world_forward_r2),
        "n_approach_total": int(n_approach_total),
        "mean_ep_contacts": float(contact_rate),
        "train_approach": int(train_counts["hazard_approach"]),
        "train_agent_contact": int(train_counts["agent_caused_hazard"]),
        "train_env_contact": int(train_counts["env_caused_hazard"]),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 80,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale_enabled: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: COORDINATION_ENABLED vs COORDINATION_BLOCKED."""
    results_enabled: List[Dict] = []
    results_blocked: List[Dict] = []

    for seed in seeds:
        for scale in [proximity_harm_scale_enabled, 0.0]:
            label = "COORDINATION_ENABLED" if scale > 0 else "COORDINATION_BLOCKED"
            print(
                f"\n[V3-EXQ-083] {label} seed={seed}"
                f" proximity_harm_scale={scale}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                proximity_harm_scale=scale,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
            )
            if scale > 0:
                results_enabled.append(r)
            else:
                results_blocked.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    contact_rate_enabled = _avg(results_enabled, "contact_rate")
    contact_rate_blocked = _avg(results_blocked, "contact_rate")
    delta_contact_rate   = contact_rate_blocked - contact_rate_enabled

    avoidance_enabled = _avg(results_enabled, "approach_avoidance_rate")
    avoidance_blocked = _avg(results_blocked, "approach_avoidance_rate")
    delta_avoidance   = avoidance_enabled - avoidance_blocked

    wf_r2_enabled = _avg(results_enabled, "world_forward_r2")

    # Per-seed directionality check (C4)
    c4_per_seed = [
        r_blocked["contact_rate"] > r_enabled["contact_rate"]
        for r_blocked, r_enabled in zip(results_blocked, results_enabled)
    ]
    c4_pass = all(c4_per_seed)

    # Pre-registered PASS criteria
    c1_pass = delta_contact_rate > 0.03
    c2_pass = delta_avoidance   > 0.05
    c3_pass = wf_r2_enabled     > 0.10
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and delta_contact_rate >= 0:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-083] Final results:", flush=True)
    print(
        f"  contact_rate_BLOCKED={contact_rate_blocked:.4f}"
        f"  contact_rate_ENABLED={contact_rate_enabled:.4f}",
        flush=True,
    )
    print(
        f"  delta_contact_rate={delta_contact_rate:+.4f}"
        f"  avoidance_ENABLED={avoidance_enabled:.4f}"
        f"  avoidance_BLOCKED={avoidance_blocked:.4f}",
        flush=True,
    )
    print(
        f"  delta_avoidance={delta_avoidance:+.4f}"
        f"  world_forward_r2_ENABLED={wf_r2_enabled:.4f}",
        flush=True,
    )
    print(f"  decision={decision}  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_contact_rate={delta_contact_rate:.4f} <= 0.03"
            " (blocking low-energy pathways did not increase terminal events)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: delta_avoidance={delta_avoidance:.4f} <= 0.05"
            " (gradient did not enable measurable early correction)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: world_forward_r2={wf_r2_enabled:.4f} <= 0.10"
            " (E2.world_forward insufficiently trained -- greedy policy unreliable)"
        )
    if not c4_pass:
        failure_notes.append(
            "C4 FAIL: not all seeds show BLOCKED > ENABLED contact_rate -- inconsistent"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "contact_rate_enabled":    float(contact_rate_enabled),
        "contact_rate_blocked":    float(contact_rate_blocked),
        "delta_contact_rate":      float(delta_contact_rate),
        "avoidance_enabled":       float(avoidance_enabled),
        "avoidance_blocked":       float(avoidance_blocked),
        "delta_avoidance":         float(delta_avoidance),
        "world_forward_r2_enabled": float(wf_r2_enabled),
        "world_forward_r2_blocked": float(_avg(results_blocked, "world_forward_r2")),
        "n_approach_enabled":      float(_avg(results_enabled, "n_approach_total")),
        "n_approach_blocked":      float(_avg(results_blocked, "n_approach_total")),
        "n_seeds":                 float(len(seeds)),
        "alpha_world":             float(alpha_world),
        "proximity_harm_scale":    float(proximity_harm_scale_enabled),
        "crit1_pass":              1.0 if c1_pass else 0.0,
        "crit2_pass":              1.0 if c2_pass else 0.0,
        "crit3_pass":              1.0 if c3_pass else 0.0,
        "crit4_pass":              1.0 if c4_pass else 0.0,
        "criteria_met":            float(criteria_met),
    }

    if all_pass:
        interpretation = (
            "MECH-102 SUPPORTED: blocking low-energy coordination pathways (approach"
            " gradient) forces the agent to escalate to terminal error correction"
            f" (contact). COORDINATION_BLOCKED contact_rate={contact_rate_blocked:.4f}"
            f" vs COORDINATION_ENABLED {contact_rate_enabled:.4f}"
            f" (delta={delta_contact_rate:+.4f}). E3-guided greedy policy achieves"
            f" approach avoidance in ENABLED condition (avoidance_rate={avoidance_enabled:.4f})"
            " but is forced to act terminally in BLOCKED. Violence (maximal-energy"
            " intervention) is genuinely terminal: it only appears when low-energy"
            " pathways are unavailable."
        )
    elif c1_pass and delta_contact_rate >= 0:
        interpretation = (
            "Weak positive: COORDINATION_BLOCKED shows higher contact rate but"
            " discriminative margin is below threshold. Early correction advantage"
            " is marginal. May indicate E2.world_forward noise limits greedy policy"
            " effectiveness, or approach gradient signal is insufficient for clear"
            " harm avoidance."
        )
    else:
        interpretation = (
            "MECH-102 NOT supported by this pair: blocking approach gradient does"
            f" not increase terminal events (delta_contact_rate={delta_contact_rate:.4f})."
            " Possible explanations: E2.world_forward noise makes greedy policy"
            " equivalent to random; or E3.harm_eval does not score approach states"
            " as harmful even in COORDINATION_ENABLED condition (ARC-024 prerequisite"
            " not met); or contact events are too rare to measure the difference."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_enabled_rows = "\n".join(
        f"  seed={r['seed']}: contact_rate={r['contact_rate']:.4f}"
        f" avoidance={r['approach_avoidance_rate']:.4f}"
        f" wf_r2={r['world_forward_r2']:.4f}"
        f" n_approach={r['n_approach_total']}"
        for r in results_enabled
    )
    per_blocked_rows = "\n".join(
        f"  seed={r['seed']}: contact_rate={r['contact_rate']:.4f}"
        f" avoidance={r['approach_avoidance_rate']:.4f}"
        f" wf_r2={r['world_forward_r2']:.4f}"
        for r in results_blocked
    )

    summary_markdown = (
        f"# V3-EXQ-083 -- MECH-102 Terminal Error-Correction Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-102\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Eval policy:** E3-guided greedy (argmin E3.harm_eval(E2.world_fwd(z, a)))\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_contact_rate           > 0.03\n"
        f"C2: delta_avoidance (enabled-blocked) > 0.05\n"
        f"C3: world_forward_r2_ENABLED     > 0.10\n"
        f"C4: per-seed BLOCKED > ENABLED contact_rate\n\n"
        f"## Results\n\n"
        f"| Condition | contact_rate | avoidance_rate | wf_r2 |\n"
        f"|-----------|-------------|----------------|-------|\n"
        f"| COORDINATION_ENABLED  | {contact_rate_enabled:.4f} | {avoidance_enabled:.4f}"
        f" | {wf_r2_enabled:.4f} |\n"
        f"| COORDINATION_BLOCKED  | {contact_rate_blocked:.4f} | {avoidance_blocked:.4f}"
        f" | {_avg(results_blocked, 'world_forward_r2'):.4f} |\n\n"
        f"**delta_contact_rate (BLOCKED - ENABLED): {delta_contact_rate:+.4f}**\n"
        f"**delta_avoidance (ENABLED - BLOCKED): {delta_avoidance:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: delta_contact_rate > 0.03              | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_contact_rate:.4f} |\n"
        f"| C2: delta_avoidance > 0.05                 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {delta_avoidance:.4f} |\n"
        f"| C3: world_forward_r2_ENABLED > 0.10        | {'PASS' if c3_pass else 'FAIL'}"
        f" | {wf_r2_enabled:.4f} |\n"
        f"| C4: per-seed BLOCKED > ENABLED contact_rate | {'PASS' if c4_pass else 'FAIL'}"
        f" | {c4_per_seed} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"COORDINATION_ENABLED:\n{per_enabled_rows}\n\n"
        f"COORDINATION_BLOCKED:\n{per_blocked_rows}\n"
        f"{failure_section}\n"
    )

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
        "fatal_error_count": sum(r["n_fatal"] for r in results_enabled + results_blocked),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",             type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",            type=int,   default=400)
    parser.add_argument("--eval-eps",          type=int,   default=80)
    parser.add_argument("--steps",             type=int,   default=200)
    parser.add_argument("--alpha-world",       type=float, default=0.9)
    parser.add_argument("--alpha-self",        type=float, default=0.3)
    parser.add_argument("--harm-scale",        type=float, default=0.02)
    parser.add_argument("--proximity-scale",   type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale_enabled=args.proximity_scale,
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
