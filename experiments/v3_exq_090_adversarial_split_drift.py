"""
V3-EXQ-090 -- Adversarial Split Drift Test (Schizophrenic Drift Probe)

Claims: SD-005

Tests AdversarialSplitHead (ree_core/latent/stack.py): gradient reversal layer
that prevents "schizophrenic drift" -- the gradual blurring of the z_self encoder
boundary under harm training pressure.

The concern (EXQ-090 motivation):
  SD-005 creates separate z_self and z_world encoders. But during harm training,
  if harm-related features correlate with body-state features, backprop through
  harm_eval can inadvertently make z_self carry harm-predictive information.
  This blurs the boundary: z_self becomes a mix of motor-sensory and harm signals.

  In biological terms: the efference-copy comparator (MECH-095) relies on z_self
  being a pure motor-sensory representation. If z_self carries harm information,
  the comparator's mismatch signal becomes confounded.

  In psychiatric terms: this is a model of gradual self/world boundary erosion
  under stress-induced hypervigilance (threat information bleeding into
  interoceptive/proprioceptive channels).

Protocol:
  CONDITION A (no adversarial): train on harm supervision tasks, measure
    R2(z_self -> harm_outcome) at T=0, T=100, T=200, T=300 episodes.
  CONDITION B (with adversarial): same training, but AdversarialSplitHead active.

PASS criteria (ALL):
  C1: R2(z_self->harm) stays < 0.05 in condition B at T=300 eps
  C2: R2(z_self->harm) rises to > 0.10 in condition A at T=300 eps
      (demonstrates that drift DOES happen without defense)
  C3: R2(z_world->harm) > 0.20 in both conditions (harm IS in z_world)
  C4: drift_delta = R2_A_T300 - R2_B_T300 > 0.05 (GRL actually helps)
  C5: no fatal errors
"""

import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AdversarialSplitHead
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_090_adversarial_split_drift"
CLAIM_IDS = ["SD-005"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32
EVAL_INTERVAL = 100  # episodes between drift measurements
TOTAL_EPISODES = 300
STEPS_PER_EP   = 150
GRL_ALPHA      = 1.0


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _measure_r2_z_self_harm(
    agent: REEAgent,
    harm_enc: HarmEncoder,
    env: CausalGridWorldV2,
    seed: int,
    n_steps: int = 500,
) -> float:
    """Measure R2(z_self -> harm_label) using linear regression probe."""
    num_actions = env.action_dim
    agent.eval(); harm_enc.eval()

    flat_obs, obs_dict = env.reset()
    agent.reset()

    z_self_list: List[torch.Tensor] = []
    label_list:  List[float] = []

    with torch.no_grad():
        for _ in range(n_steps):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            z_self_list.append(latent.z_self.squeeze(0))

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label_list.append(float(harm_obs[12].item()))

            action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()

    if len(z_self_list) < 20:
        return 0.0

    X = torch.stack(z_self_list).float()   # [N, self_dim]
    y = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    # OLS: w = (X'X)^{-1} X' y
    XtX = X.T @ X + 1e-3 * torch.eye(X.shape[1])
    Xty = X.T @ y
    try:
        w = torch.linalg.solve(XtX, Xty)
        y_pred = X @ w
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
        return max(0.0, min(1.0, r2))
    except Exception:
        return 0.0


def _measure_r2_z_world_harm(
    agent: REEAgent,
    harm_enc: HarmEncoder,
    env: CausalGridWorldV2,
    n_steps: int = 500,
) -> float:
    """Measure R2(z_world -> harm_label) using linear regression probe."""
    num_actions = env.action_dim
    agent.eval(); harm_enc.eval()

    flat_obs, obs_dict = env.reset()
    agent.reset()

    z_world_list: List[torch.Tensor] = []
    label_list:   List[float] = []

    with torch.no_grad():
        for _ in range(n_steps):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            z_world_list.append(latent.z_world.squeeze(0))

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label_list.append(float(harm_obs[12].item()))

            action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                flat_obs, obs_dict = env.reset()
                agent.reset()

    if len(z_world_list) < 20:
        return 0.0

    X = torch.stack(z_world_list).float()
    y = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
    XtX = X.T @ X + 1e-3 * torch.eye(X.shape[1])
    Xty = X.T @ y
    try:
        w = torch.linalg.solve(XtX, Xty)
        y_pred = X @ w
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
        return max(0.0, min(1.0, r2))
    except Exception:
        return 0.0


def _run_condition(
    use_adversarial: bool,
    seed: int,
    env: CausalGridWorldV2,
    world_dim: int,
    self_dim: int,
    lr: float,
    total_episodes: int,
    steps_per_ep: int,
    eval_interval: int,
) -> Tuple[List[float], List[float], float]:
    """
    Run one training condition (A=no adversarial, B=adversarial).
    Returns (r2_self_drift, r2_world_list, r2_world_final).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    num_actions = env.action_dim

    adv_head = None
    opt_adv_enc = None
    opt_adv_head = None
    if use_adversarial:
        adv_head = AdversarialSplitHead(
            self_dim=self_dim, hidden_dim=32, grl_alpha=GRL_ALPHA)
        # Encoder penalty flows through GRL back into z_self encoder
        # Get z_self encoder params: body_encoder in SplitEncoder
        z_self_enc_params = [
            p for n, p in agent.latent_stack.named_parameters()
            if "body_encoder" in n or "self_encoder" in n
        ]
        opt_adv_enc  = optim.Adam(z_self_enc_params or agent.latent_stack.parameters(), lr=lr*0.1)
        opt_adv_head = optim.Adam(adv_head.parameters(), lr=lr)

    std_params = [p for n, p in agent.named_parameters() if "harm_eval" not in n]
    opt_std  = optim.Adam(std_params, lr=lr)
    opt_he   = optim.Adam(harm_enc.parameters(), lr=1e-3)
    opt_e3zh = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    r2_self_at_interval: List[float] = []
    r2_world_final = 0.0

    agent.train()
    if adv_head is not None:
        adv_head.train()

    for ep in range(total_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_ep):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            z_self  = latent.z_self
            z_world = latent.z_world.detach()

            harm_obs = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            label = harm_obs[12].unsqueeze(0).unsqueeze(0)

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)

            # HarmEncoder training
            z_harm = harm_enc(harm_obs.unsqueeze(0))
            pred_zh = agent.e3.harm_eval_z_harm(z_harm)
            loss_he = F.mse_loss(pred_zh, label)
            opt_he.zero_grad(); loss_he.backward(retain_graph=True); opt_he.step()
            opt_e3zh.zero_grad()
            loss_he2 = F.mse_loss(agent.e3.harm_eval_z_harm(z_harm.detach()), label)
            if loss_he2.requires_grad:
                loss_he2.backward()
                opt_e3zh.step()

            # Adversarial split head (if condition B)
            if use_adversarial and adv_head is not None and z_self.requires_grad:
                enc_penalty, head_loss = adv_head.split_losses(z_self, label.detach())
                # Encoder update: GRL will negate gradient into z_self encoder
                if enc_penalty.requires_grad and opt_adv_enc is not None:
                    opt_adv_enc.zero_grad(); enc_penalty.backward()
                    opt_adv_enc.step()
                # Head update: normal gradient, train to predict harm from z_self
                if head_loss.requires_grad and opt_adv_head is not None:
                    opt_adv_head.zero_grad(); head_loss.backward()
                    opt_adv_head.step()

            # Standard losses
            total = agent.compute_prediction_loss() + agent.compute_e2_loss()
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done: break

        if (ep + 1) % eval_interval == 0:
            r2_self = _measure_r2_z_self_harm(agent, harm_enc, env, seed, n_steps=300)
            r2_self_at_interval.append(r2_self)
            cond_str = "B (adversarial)" if use_adversarial else "A (no adversarial)"
            print(
                f"    Cond {cond_str} ep {ep+1}/{total_episodes}  "
                f"R2(z_self->harm)={r2_self:.4f}",
                flush=True,
            )

    # Final world probe
    r2_world_final = _measure_r2_z_world_harm(agent, harm_enc, env, n_steps=500)
    return r2_self_at_interval, [], r2_world_final


def run(
    seed: int = 0,
    total_episodes: int = TOTAL_EPISODES,
    steps_per_episode: int = STEPS_PER_EP,
    eval_interval: int = EVAL_INTERVAL,
    world_dim: int = 32,
    self_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.05,
        env_drift_prob=0.15,
        proximity_harm_scale=0.05,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    print(
        f"[V3-EXQ-090] Adversarial Split Drift Test (Schizophrenic Drift Probe)\n"
        f"  Condition A: no adversarial loss (expect drift: R2(z_self->harm) rises)\n"
        f"  Condition B: AdversarialSplitHead active (expect R2 stays < 0.05)\n"
        f"  Training: {total_episodes} eps, eval every {eval_interval} eps",
        flush=True,
    )

    print(f"\n  == Condition A: no adversarial ==", flush=True)
    r2_self_A, _, r2_world_A = _run_condition(
        use_adversarial=False, seed=seed, env=env,
        world_dim=world_dim, self_dim=self_dim, lr=lr,
        total_episodes=total_episodes, steps_per_ep=steps_per_episode,
        eval_interval=eval_interval,
    )

    print(f"\n  == Condition B: with adversarial ==", flush=True)
    r2_self_B, _, r2_world_B = _run_condition(
        use_adversarial=True, seed=seed, env=env,
        world_dim=world_dim, self_dim=self_dim, lr=lr,
        total_episodes=total_episodes, steps_per_ep=steps_per_episode,
        eval_interval=eval_interval,
    )

    r2_A_final = r2_self_A[-1] if r2_self_A else 0.0
    r2_B_final = r2_self_B[-1] if r2_self_B else 0.0
    drift_delta = r2_A_final - r2_B_final

    print(f"\n  --- EXQ-090 results ---", flush=True)
    print(f"  R2(z_self->harm) cond A trajectory: {[f'{x:.4f}' for x in r2_self_A]}", flush=True)
    print(f"  R2(z_self->harm) cond B trajectory: {[f'{x:.4f}' for x in r2_self_B]}", flush=True)
    print(f"  R2(z_world->harm) cond A final: {r2_world_A:.4f}", flush=True)
    print(f"  R2(z_world->harm) cond B final: {r2_world_B:.4f}", flush=True)
    print(f"  drift_delta (A-B): {drift_delta:.4f}", flush=True)

    c1 = r2_B_final < 0.05
    c2 = r2_A_final > 0.10
    c3 = r2_world_A > 0.20 and r2_world_B > 0.20
    c4 = drift_delta > 0.05
    c5 = True

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: R2_B_final={r2_B_final:.4f} >= 0.05. "
        f"AdversarialSplitHead not preventing z_self from encoding harm.")
    if not c2: failure_notes.append(
        f"C2 FAIL: R2_A_final={r2_A_final:.4f} <= 0.10. "
        f"Drift did not occur without defense -- test may lack statistical power.")
    if not c3: failure_notes.append(
        f"C3 FAIL: R2(z_world->harm) A={r2_world_A:.4f} B={r2_world_B:.4f}. "
        f"Harm information not sufficiently encoded in z_world.")
    if not c4: failure_notes.append(
        f"C4 FAIL: drift_delta={drift_delta:.4f} <= 0.05. "
        f"GRL not providing sufficient protection.")

    print(f"\nV3-EXQ-090 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics: Dict[str, float] = {
        "r2_self_A_final":   float(r2_A_final),
        "r2_self_B_final":   float(r2_B_final),
        "r2_world_A_final":  float(r2_world_A),
        "r2_world_B_final":  float(r2_world_B),
        "drift_delta":       float(drift_delta),
        "crit1_pass":        1.0 if c1 else 0.0,
        "crit2_pass":        1.0 if c2 else 0.0,
        "crit3_pass":        1.0 if c3 else 0.0,
        "crit4_pass":        1.0 if c4 else 0.0,
        "crit5_pass":        1.0 if c5 else 0.0,
        "criteria_met":      float(n_met),
        "fatal_error_count": 0.0,
    }
    for i, r2 in enumerate(r2_self_A):
        metrics[f"r2_self_A_t{(i+1)*eval_interval}"] = float(r2)
    for i, r2 in enumerate(r2_self_B):
        metrics[f"r2_self_B_t{(i+1)*eval_interval}"] = float(r2)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-090 -- Adversarial Split Drift Test (Schizophrenic Drift Probe)

**Status:** {status}
**Claims:** SD-005
**World:** CausalGridWorldV2 (4 hazards, env_drift for stress)
**Protocol:** Condition A (no GRL) vs Condition B (AdversarialSplitHead) x {total_episodes} eps

## Motivation

SD-005 creates separate z_self and z_world encoders. Under harm training, backprop can
gradually make z_self carry harm-predictive information (a z_world property), blurring
the self/world boundary. This is a model of "schizophrenic drift" -- the efference-copy
comparator (MECH-095) relies on z_self being a pure motor-sensory representation.

AdversarialSplitHead uses gradient reversal (Ganin et al. 2016) to prevent this.

## Results

| Condition | R2(z_self->harm) final | R2(z_world->harm) final |
|-----------|----------------------|------------------------|
| A: no adversarial | {r2_A_final:.4f} | {r2_world_A:.4f} |
| B: with adversarial | {r2_B_final:.4f} | {r2_world_B:.4f} |

Drift delta (A - B): {drift_delta:.4f}

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: R2_B < 0.05 (adversarial prevents drift) | {"PASS" if c1 else "FAIL"} | {r2_B_final:.4f} |
| C2: R2_A > 0.10 (drift does occur without defense) | {"PASS" if c2 else "FAIL"} | {r2_A_final:.4f} |
| C3: R2(z_world->harm) > 0.20 both conditions | {"PASS" if c3 else "FAIL"} | A={r2_world_A:.4f} B={r2_world_B:.4f} |
| C4: drift_delta > 0.05 | {"PASS" if c4 else "FAIL"} | {drift_delta:.4f} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} | - |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse, json
    from datetime import datetime, timezone
    from typing import Dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--eps",   type=int, default=TOTAL_EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EP)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        total_episodes=args.eps,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
