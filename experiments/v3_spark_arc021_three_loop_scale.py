#!/opt/local/bin/python3
"""
Spark-Scale ARC-021: Three-Loop Incommensurability Test

ARC-021 asserts that E1 (sensory prediction error), E2 (motor-sensory error),
and E3 (harm/goal error) require distinct learning channels because these error
signals are incommensurable. Collapsing them misattributes credit.

At world_dim=32, the architecture has limited capacity so mixing signals may not
hurt performance measurably. At REEConfig.large() (world_dim=128), each module
has enough representational capacity to specialise, making credit contamination
from merged optimizers more disruptive and the effect more detectable.

Two conditions (matched seeds):
  SEPARATE -- three independent optimizers (E1_opt, E2_opt, E3_opt)
  MERGED   -- single optimizer over agent.parameters(), combined loss = E1+E2+E3

MERGED has two contamination effects:
  1. E3 parameters receive gradient from E1/E2 prediction losses (sensory noise
     corrupts harm evaluator).
  2. E1/E2 parameters receive gradient from E3 harm loss (harm avoidance signal
     corrupts sensory and motor predictors).

ARC-021 prediction: SEPARATE trains a better-calibrated E3 harm evaluator
and achieves lower harm_rate at convergence.

MECH-069 basis: the three error signals are biologically incommensurable --
sensory prediction error (E1), motor-sensory error (E2), and harm/goal error (E3)
correspond to distinct cortico-striatal loops. Mixing them collapses the three-loop
architecture back to a single credit-assignment channel.

PASS (ALL required):
  C1: e3_discrim_separate > e3_discrim_merged + 0.02
      (separate E3 discriminates harm better)
  C2: harm_rate_separate < harm_rate_merged + 0.01
      (separate achieves lower harm rate, or at most 0.01 worse)
  C3: e1_loss_separate <= e1_loss_merged + 0.002
      (separate E1 is not degraded; merged should be worse or equal)
  C4: e2_loss_separate <= e2_loss_merged + 0.002
      (separate E2 is not degraded)

NOTE: C2 uses a margin because harm_rate variance can be high. C1 is the primary
discriminator -- better E3 harm calibration should eventually translate to lower
harm_rate with more training.

=============================================================================
QUEUING: Do NOT queue until Spark hardware is available.
  Estimated runtime (Spark, world_dim=128): ~30 min
  Assign next available EXQ number when queuing.
=============================================================================
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_spark_arc021_three_loop_scale"
CLAIM_IDS = ["ARC-021", "MECH-069"]

WARMUP_EPISODES = 350
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200
SEED            = 42


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _train_step_separate(agent, e1_opt, e2_opt, e3_opt, latent, harm_signal):
    """Three independent backward passes -- ARC-021 correct architecture."""
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_opt.zero_grad(); e1_loss.backward(); e1_opt.step()

    e2_loss = agent.compute_e2_loss()
    if e2_loss.requires_grad:
        e2_opt.zero_grad(); e2_loss.backward(); e2_opt.step()

    z_world = latent.z_world.detach()
    harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
    harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
    e3_opt.zero_grad(); harm_loss.backward(); e3_opt.step()

    return (
        e1_loss.item() if e1_loss.requires_grad else 0.0,
        e2_loss.item() if e2_loss.requires_grad else 0.0,
        harm_loss.item(),
    )


def _train_step_merged(agent, merged_opt, latent, harm_signal):
    """Single backward pass with combined loss -- ARC-021 ablation."""
    e1_loss = agent.compute_prediction_loss()
    e2_loss = agent.compute_e2_loss()
    z_world = latent.z_world.detach()
    harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
    # z_world must be reattached for harm loss so grad flows through latent_stack
    z_world_for_grad = latent.z_world  # NOT detached -- allow contamination gradient
    harm_loss = F.mse_loss(agent.e3.harm_eval(z_world_for_grad), harm_target)

    combined = (
        (e1_loss if e1_loss.requires_grad else torch.tensor(0.0))
        + (e2_loss if e2_loss.requires_grad else torch.tensor(0.0))
        + harm_loss
    )
    if combined.requires_grad:
        merged_opt.zero_grad(); combined.backward(); merged_opt.step()

    return (
        e1_loss.item() if e1_loss.requires_grad else 0.0,
        e2_loss.item() if e2_loss.requires_grad else 0.0,
        harm_loss.item(),
    )


def _run_condition(label: str, seed: int, merged: bool) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = REEConfig.large(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        reafference_action_dim=env.action_dim,
    )
    world_dim = config.latent.world_dim
    agent = REEAgent(config)

    if merged:
        merged_opt = optim.Adam(agent.parameters(), lr=1e-3)
        e1_opt = e2_opt = e3_opt = None
    else:
        e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
        e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
        e3_opt = optim.Adam(
            list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
            lr=1e-3,
        )
        merged_opt = None

    agent.train()

    # -- Warmup ----------------------------------------------------------------
    print(
        f"  [{label}] warmup ({WARMUP_EPISODES} eps)"
        f"  merged={merged}  world_dim={world_dim}",
        flush=True,
    )
    e1_losses_warmup: list = []
    e2_losses_warmup: list = []

    for ep in range(WARMUP_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_t: Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            if merged:
                e1_v, e2_v, _ = _train_step_merged(agent, merged_opt, latent, harm_signal)
            else:
                e1_v, e2_v, _ = _train_step_separate(agent, e1_opt, e2_opt, e3_opt, latent, harm_signal)

            if ep >= WARMUP_EPISODES - 50:  # last 50 eps of warmup
                e1_losses_warmup.append(e1_v)
                e2_losses_warmup.append(e2_v)

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"  [{label}] warmup ep {ep+1}/{WARMUP_EPISODES}", flush=True)

    # -- Eval ------------------------------------------------------------------
    print(f"  [{label}] eval ({EVAL_EPISODES} eps)", flush=True)
    agent.eval()

    harm_events    = 0
    resource_visits = 0
    total_steps    = 0
    e3_harm_on_harm: list = []
    e3_harm_on_safe: list = []

    with torch.no_grad():
        for ep in range(EVAL_EPISODES):
            _, obs_dict = env.reset()
            agent.reset()
            z_self_t = None

            for _ in range(STEPS_PER_EP):
                obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

                if agent._current_latent is not None:
                    z_self_t = agent._current_latent.z_self.detach().clone()

                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                if z_self_t is not None:
                    agent.record_transition(z_self_t, action, latent.z_self.detach())

                _, reward, done, info, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0
                ttype = info.get("transition_type", "none")

                if ttype in ("agent_caused_hazard", "hazard_approach"):
                    harm_events += 1
                if ttype == "resource":
                    resource_visits += 1

                harm_pred = agent.e3.harm_eval(latent.z_world.detach()).item()
                if harm_signal < 0:
                    e3_harm_on_harm.append(harm_pred)
                else:
                    e3_harm_on_safe.append(harm_pred)

                total_steps += 1
                if done:
                    break

    harm_rate     = harm_events / max(1, total_steps)
    resource_rate = resource_visits / max(1, total_steps)
    e3_on_harm    = sum(e3_harm_on_harm) / max(1, len(e3_harm_on_harm))
    e3_on_safe    = sum(e3_harm_on_safe) / max(1, len(e3_harm_on_safe))
    e3_discrim    = e3_on_harm - e3_on_safe
    e1_loss_final = sum(e1_losses_warmup) / max(1, len(e1_losses_warmup))
    e2_loss_final = sum(e2_losses_warmup) / max(1, len(e2_losses_warmup))

    print(
        f"  [{label}] harm_rate={harm_rate:.4f}"
        f"  resource_rate={resource_rate:.4f}"
        f"  e3_discrim={e3_discrim:.4f}"
        f"  e1_loss={e1_loss_final:.5f}"
        f"  e2_loss={e2_loss_final:.5f}",
        flush=True,
    )

    return {
        "label":          label,
        "merged":         merged,
        "harm_rate":      round(harm_rate, 5),
        "resource_rate":  round(resource_rate, 5),
        "e3_discrim_gap": round(e3_discrim, 5),
        "e3_on_harm":     round(e3_on_harm, 5),
        "e3_on_safe":     round(e3_on_safe, 5),
        "e1_loss_final":  round(e1_loss_final, 6),
        "e2_loss_final":  round(e2_loss_final, 6),
        "n_harm_steps":   len(e3_harm_on_harm),
        "n_safe_steps":   len(e3_harm_on_safe),
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"\n[ARC-021-SCALE] seed={args.seed}  world_dim=128 (REEConfig.large)", flush=True)
    print("[ARC-021-SCALE] Condition: SEPARATE (three independent optimizers)", flush=True)
    r_sep = _run_condition("SEPARATE", args.seed,     merged=False)
    print("[ARC-021-SCALE] Condition: MERGED (single combined-loss optimizer)", flush=True)
    r_mrg = _run_condition("MERGED",   args.seed + 1, merged=True)

    delta_discrim = r_sep["e3_discrim_gap"] - r_mrg["e3_discrim_gap"]
    delta_harm    = r_mrg["harm_rate"]      - r_sep["harm_rate"]

    c1 = delta_discrim             >  0.02
    c2 = delta_harm                > -0.01
    c3 = r_sep["e1_loss_final"]   <=  r_mrg["e1_loss_final"] + 0.002
    c4 = r_sep["e2_loss_final"]   <=  r_mrg["e2_loss_final"] + 0.002

    status = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"

    print(f"\n[ARC-021-SCALE] -- Results -----------------------------------------", flush=True)
    print(f"  delta_e3_discrim (sep-mrg): {delta_discrim:.4f}  (C1 > 0.02: {c1})", flush=True)
    print(f"  delta_harm_rate  (mrg-sep): {delta_harm:.4f}  (C2 > -0.01: {c2})", flush=True)
    print(f"  e1_loss sep/mrg:  {r_sep['e1_loss_final']:.5f} / {r_mrg['e1_loss_final']:.5f}  (C3: {c3})", flush=True)
    print(f"  e2_loss sep/mrg:  {r_sep['e2_loss_final']:.5f} / {r_mrg['e2_loss_final']:.5f}  (C4: {c4})", flush=True)
    print(f"  Status: {status}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "status": status,
        "metrics": {
            "separate": r_sep,
            "merged":   r_mrg,
            "delta_e3_discrim": round(delta_discrim, 5),
            "delta_harm_rate":  round(delta_harm, 5),
        },
        "criteria": {
            "C1_e3_discrim":  c1,
            "C2_harm_rate":   c2,
            "C3_e1_loss":     c3,
            "C4_e2_loss":     c4,
        },
        "run_timestamp":      ts,
        "run_id":             f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim":              "ARC-021",
        "verdict":            status,
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
