"""
V3-ONBOARD-smoke-EWINPC -- Contributor Onboarding Smoke Test

Purpose: verify the full experiment pipeline round-trip on a newly onboarded
machine AND benchmark CPU vs GPU compute separately so experiment routing
(smart_assign) can match experiment type to the best available machine.

Three benchmark blocks:
  1. ENV-ONLY  -- pure environment stepping, no neural net. Isolates Python/CPU.
  2. CPU       -- full training loop forced to CPU. Baseline for env-heavy experiments.
  3. GPU       -- full training loop on CUDA (skipped if unavailable). For net-heavy experiments.

Metrics written to result JSON and used by smart_assign() for routing:
  env_steps_per_second    -- pure env throughput (CPU-bound ceiling)
  steps_per_second_cpu    -- integrated training throughput on CPU
  steps_per_second_gpu    -- integrated training throughput on GPU (0.0 if no CUDA)
  gpu_name                -- GPU model string
  cuda_available          -- bool

Pass criteria (pipeline verification -- not scientific):
  C1: all three benchmark blocks completed without exception
  C2: env_steps_per_second > 0
  C3: steps_per_second_cpu > 0
  C4 (advisory): cuda_available -- logged but does not gate PASS/FAIL

Machine affinity: EWINPC (set in experiment_queue.json).
Estimated runtime: ~10-20 min depending on hardware.
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


EXPERIMENT_TYPE = "v3_onboard_smoke_EWINPC"
CLAIM_IDS: list = []  # not tied to a scientific claim

# Episode counts per block -- kept short, this is a benchmark not a training run
ENV_ONLY_EPISODES = 50    # pure env stepping, no net
CPU_EPISODES      = 15    # full loop forced to CPU
GPU_EPISODES      = 15    # full loop on CUDA (skipped if unavailable)
STEPS_PER_EP      = 100


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_agent(env: CausalGridWorldV2, device: torch.device) -> REEAgent:
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
    agent.device = device  # config.device defaults to "cpu"; must update after .to()
    return agent


def _run_training_block(
    env: CausalGridWorldV2,
    agent: REEAgent,
    n_episodes: int,
    label: str,
    world_dim: int = 32,
) -> float:
    """Run n_episodes of the full training loop. Returns steps/second."""
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    total_steps = 0
    agent.train()

    t0 = time.monotonic()
    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"].to(agent.device) if isinstance(obs_dict["body_state"], torch.Tensor) else obs_dict["body_state"]
            obs_world = obs_dict["world_state"].to(agent.device) if isinstance(obs_dict["world_state"], torch.Tensor) else obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action_vec = agent.select_action(candidates, ticks)
            action_idx = action_vec.argmax().item() if action_vec.dim() > 0 else action_vec.item()
            _, _, done, _, obs_dict = env.step(action_idx)

            z_self_prev = latent.z_self.detach()
            action_prev = action_vec.detach()
            total_steps += 1
            if done:
                break

        if (ep + 1) % 5 == 0:
            elapsed = time.monotonic() - t0
            sps = total_steps / elapsed if elapsed > 0 else 0
            print(f"[ONBOARD/{label}] ep {ep+1}/{n_episodes}  {sps:.0f} steps/s", flush=True)

    elapsed = time.monotonic() - t0
    sps = total_steps / elapsed if elapsed > 0 else 0.0
    print(f"[ONBOARD/{label}] done -- {total_steps} steps in {elapsed:.1f}s -> {sps:.0f} steps/s", flush=True)
    return sps


def run(seed: int = 42) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cuda_available = torch.cuda.is_available()
    gpu_name       = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    torch_version  = torch.__version__
    platform_str   = platform.platform()

    print(
        f"[ONBOARD] GPU: {gpu_name}  CUDA: {cuda_available}\n"
        f"[ONBOARD] torch={torch_version}  platform={platform_str}",
        flush=True,
    )

    # -- Block 1: ENV-ONLY (pure Python, no net) ------------------------------
    print(f"\n[ONBOARD] Block 1/3 -- ENV-ONLY ({ENV_ONLY_EPISODES} eps x {STEPS_PER_EP} steps)", flush=True)
    env = _make_env(seed)
    action_dim = env.action_dim
    total_env_steps = 0
    t0 = time.monotonic()
    for ep in range(ENV_ONLY_EPISODES):
        _, obs_dict = env.reset()
        for _ in range(STEPS_PER_EP):
            action_idx = random.randint(0, action_dim - 1)
            _, _, done, _, obs_dict = env.step(action_idx)
            total_env_steps += 1
            if done:
                break
    env_elapsed = time.monotonic() - t0
    env_sps = total_env_steps / env_elapsed if env_elapsed > 0 else 0.0
    print(f"[ONBOARD/ENV] {total_env_steps} steps in {env_elapsed:.1f}s -> {env_sps:.0f} steps/s", flush=True)

    # -- Block 2: CPU training loop --------------------------------------------
    print(f"\n[ONBOARD] Block 2/3 -- CPU training ({CPU_EPISODES} eps x {STEPS_PER_EP} steps)", flush=True)
    cpu_device = torch.device("cpu")
    env_cpu    = _make_env(seed + 1)
    agent_cpu  = _make_agent(env_cpu, cpu_device)
    cpu_sps    = _run_training_block(env_cpu, agent_cpu, CPU_EPISODES, "CPU")

    # -- Block 3: GPU training loop (skipped if no CUDA) ----------------------
    gpu_sps = 0.0
    if cuda_available:
        print(f"\n[ONBOARD] Block 3/3 -- GPU training ({GPU_EPISODES} eps x {STEPS_PER_EP} steps)", flush=True)
        gpu_device = torch.device("cuda")
        env_gpu    = _make_env(seed + 2)
        agent_gpu  = _make_agent(env_gpu, gpu_device)
        gpu_sps    = _run_training_block(env_gpu, agent_gpu, GPU_EPISODES, "GPU")
    else:
        print("\n[ONBOARD] Block 3/3 -- GPU training SKIPPED (no CUDA)", flush=True)
        print("[ONBOARD] Advisory: experiments will run on CPU (slower for net-heavy workloads)", flush=True)

    # -- Summary ----------------------------------------------------------------
    gpu_speedup = (gpu_sps / cpu_sps) if cpu_sps > 0 and gpu_sps > 0 else None
    passed = env_sps > 0 and cpu_sps > 0
    status = "PASS" if passed else "FAIL"

    print(f"\n[ONBOARD] -- Results ------------------------------------------", flush=True)
    print(f"[ONBOARD]   env_steps/s (CPU-only):  {env_sps:.0f}", flush=True)
    print(f"[ONBOARD]   training steps/s (CPU):  {cpu_sps:.0f}", flush=True)
    print(f"[ONBOARD]   training steps/s (GPU):  {gpu_sps:.0f}", flush=True)
    if gpu_speedup:
        print(f"[ONBOARD]   GPU speedup vs CPU:      {gpu_speedup:.1f}x", flush=True)
    print(f"[ONBOARD]   status: {status}", flush=True)

    return {
        "status": status,
        "metrics": {
            "env_steps_per_second":   round(env_sps, 1),
            "steps_per_second_cpu":   round(cpu_sps, 1),
            "steps_per_second_gpu":   round(gpu_sps, 1),
            "gpu_speedup":            round(gpu_speedup, 2) if gpu_speedup else None,
            "cuda_available":         cuda_available,
            "gpu_name":               gpu_name,
            "torch_version":          torch_version,
            "platform":               platform_str,
            "env_only_episodes":      ENV_ONLY_EPISODES,
            "cpu_episodes":           CPU_EPISODES,
            "gpu_episodes":           GPU_EPISODES if cuda_available else 0,
            "steps_per_episode":      STEPS_PER_EP,
        },
        "criteria": {
            "C1_all_blocks_completed": True,   # reaching here means no exception
            "C2_env_sps_gt0":         env_sps > 0,
            "C3_cpu_sps_gt0":         cpu_sps > 0,
            "C4_cuda_advisory":       cuda_available,
        },
        "routing_hint": {
            "note": "env_steps_per_second governs env-heavy experiment routing; "
                    "steps_per_second_gpu governs net-heavy routing",
            "prefer_env_heavy":  env_sps > 500,
            "prefer_net_heavy":  gpu_sps > cpu_sps * 1.5 if cuda_available else False,
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
    print("\nBenchmark summary:", flush=True)
    m = result["metrics"]
    print(f"  env steps/s:      {m['env_steps_per_second']}", flush=True)
    print(f"  training (CPU):   {m['steps_per_second_cpu']} steps/s", flush=True)
    print(f"  training (GPU):   {m['steps_per_second_gpu']} steps/s", flush=True)
    if m.get("gpu_speedup"):
        print(f"  GPU speedup:      {m['gpu_speedup']}x", flush=True)
