"""
V3-ONBOARD-smoke-Daniel-PC — Contributor Onboarding Smoke Test

Purpose: verify the full experiment pipeline round-trip on a newly onboarded
machine. This is NOT a scientific experiment. It tests that:
  1. ree_core imports correctly
  2. CausalGridWorldV2 + REEAgent instantiate
  3. The full sense→clock→e1→trajectories→step loop runs
  4. CUDA / GPU is detected and usable (if present)
  5. Result JSON is written to REE_assembly and pushed via auto-sync

Pass criteria (deliberately lenient — just prove the plumbing works):
  C1: training loop completed all episodes without exception
  C2: total_steps > 0
  C3: steps_per_second > 0
  C4 (advisory only): cuda_available — logged but does not gate PASS/FAIL

Records: gpu_name, cuda_available, steps_per_second, torch_version, platform
so the contributor registry can be updated with real measured throughput.

Machine affinity: Daniel-PC (set in experiment_queue.json).
Estimated runtime: ~3 min on RTX 2060 Super, ~8 min on CPU.
"""

import sys
import time
import random
import platform
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_onboard_smoke_Daniel_PC"
CLAIM_IDS: list = []   # not tied to a scientific claim

WARMUP_EPISODES  = 20
EVAL_EPISODES    = 5
STEPS_PER_EP     = 100


def run(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    )
    torch_version = torch.__version__
    platform_str  = platform.platform()
    cuda_available = torch.cuda.is_available()

    print(
        f"[ONBOARD] Device: {device_str}  GPU: {gpu_name}\n"
        f"[ONBOARD] torch={torch_version}  platform={platform_str}",
        flush=True,
    )

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)
    agent.to(device)

    # Minimal optimizers — exercise the full parameter graph
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    total_steps = 0
    episodes_completed = 0
    t0 = time.monotonic()

    agent.train()
    print(
        f"[ONBOARD] Training {WARMUP_EPISODES} warmup + {EVAL_EPISODES} eval episodes "
        f"× {STEPS_PER_EP} steps …",
        flush=True,
    )

    for ep in range(WARMUP_EPISODES + EVAL_EPISODES):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for step in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, 32, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action_idx, action_vec = agent.select_action(candidates, latent)

            flat_obs, obs_dict, reward, done, info = env.step(action_idx)

            z_self_prev = latent.z_self.detach()
            action_prev = action_vec.detach()

            total_steps += 1
            if done:
                break

        episodes_completed += 1
        if (ep + 1) % 5 == 0:
            elapsed = time.monotonic() - t0
            sps = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"[ONBOARD] ep {ep+1}/{WARMUP_EPISODES + EVAL_EPISODES} "
                f"steps={total_steps}  {sps:.0f} steps/s",
                flush=True,
            )

    elapsed_total = time.monotonic() - t0
    steps_per_second = total_steps / elapsed_total if elapsed_total > 0 else 0.0

    passed = total_steps > 0 and steps_per_second > 0
    status = "PASS" if passed else "FAIL"

    print(
        f"\n[ONBOARD] Done — {total_steps} steps in {elapsed_total:.1f}s "
        f"({steps_per_second:.0f} steps/s)  status={status}",
        flush=True,
    )
    if not cuda_available:
        print("[ONBOARD] Advisory: CUDA not detected — experiments will run on CPU (slower)", flush=True)

    return {
        "status": status,
        "metrics": {
            "total_steps":      total_steps,
            "episodes_completed": episodes_completed,
            "elapsed_seconds":  round(elapsed_total, 2),
            "steps_per_second": round(steps_per_second, 1),
            "cuda_available":   cuda_available,
            "gpu_name":         gpu_name,
            "torch_version":    torch_version,
            "platform":         platform_str,
            "device":           device_str,
        },
        "criteria": {
            "C1_episodes_completed": episodes_completed == WARMUP_EPISODES + EVAL_EPISODES,
            "C2_total_steps_gt0":   total_steps > 0,
            "C3_sps_gt0":           steps_per_second > 0,
            "C4_cuda_advisory":     cuda_available,
        },
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    result = run(seed=42)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim"]              = "onboarding"
    result["verdict"]            = result["status"]

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
