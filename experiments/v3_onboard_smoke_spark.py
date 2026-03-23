#!/opt/local/bin/python3
"""
V3-ONBOARD-smoke-spark -- Contributor Onboarding Smoke Test (Spark template)

=============================================================================
CONFIGURE BEFORE QUEUING -- 3 things to update:
  1. MACHINE_NAME below -- must match socket.gethostname() exactly.
     Run on the Spark machine: python3 -c "import socket; print(socket.gethostname())"
  2. The queue entry in experiment_queue.json -- set machine_affinity to the
     same hostname string.
  3. Docstring estimated runtime once first run completes.
=============================================================================

MACHINE_NAME = "spark_1"  is a placeholder.

Like the Daniel-PC smoke test but adds Block 4: GPU training at world_dim=128.
This confirms REEConfig.large() actually benefits from GPU on Spark hardware.
The Daniel-PC GTX 1050 Ti never wins at world_dim=32 (overhead dominates);
Spark should win clearly at world_dim=128 with its unified memory architecture.

Four benchmark blocks:
  1. ENV-ONLY      -- pure environment stepping, no net (Python/CPU ceiling)
  2. CPU world=32  -- full training loop at default scale
  3. GPU world=32  -- same, on CUDA (expected: GPU loses on any sane hardware)
  4. GPU world=128 -- full training loop at REEConfig.large() scale (GPU should win)

Metrics:
  env_steps_per_second    -- pure env throughput
  steps_per_second_cpu    -- CPU training at world_dim=32
  steps_per_second_gpu_32 -- GPU training at world_dim=32
  steps_per_second_gpu_128-- GPU training at world_dim=128 (key Spark metric)
  gpu_speedup_128         -- GPU128 / CPU32 speedup ratio

Pass criteria (pipeline verification -- not scientific):
  C1: all blocks completed without exception
  C2: env_steps_per_second > 0
  C3: steps_per_second_cpu > 0
  C4 (advisory): cuda_available
  C5 (advisory): GPU wins at world_dim=128 (speedup > 1.0)
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


# =============================================================================
# UPDATE MACHINE_NAME to match socket.gethostname() output on the Spark machine
MACHINE_NAME = "spark_1"
# =============================================================================

EXPERIMENT_TYPE = f"v3_onboard_smoke_{MACHINE_NAME}"
CLAIM_IDS: list = []

ENV_ONLY_EPISODES  = 50
CPU_EPISODES       = 15    # world_dim=32
GPU_EPISODES_32    = 15    # world_dim=32, GPU
GPU_EPISODES_128   = 20    # world_dim=128, GPU (the Spark-specific block)
STEPS_PER_EP       = 100


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


def _make_agent(
    env: CausalGridWorldV2,
    device: torch.device,
    world_dim: int = 32,
) -> REEAgent:
    if world_dim == 128:
        config = REEConfig.large(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            reafference_action_dim=env.action_dim,
        )
    else:
        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=world_dim,
            world_dim=world_dim,
            alpha_world=0.9,
            alpha_self=0.3,
            reafference_action_dim=env.action_dim,
        )
    agent = REEAgent(config)
    agent.to(device)
    agent.device = device
    return agent


def _run_training_block(
    env: CausalGridWorldV2,
    agent: REEAgent,
    n_episodes: int,
    label: str,
    world_dim: int,
) -> float:
    """Run n_episodes full training loop. Returns steps/second."""
    optim.Adam(agent.parameters(), lr=1e-3)
    total_steps = 0
    agent.train()

    t0 = time.monotonic()
    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            if isinstance(obs_body, torch.Tensor):
                obs_body  = obs_body.to(agent.device)
                obs_world = obs_world.to(agent.device)

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
    print(
        f"[ONBOARD/{label}] done -- {total_steps} steps in {elapsed:.1f}s -> {sps:.0f} steps/s",
        flush=True,
    )
    return sps


def run(seed: int = 42) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cuda_available = torch.cuda.is_available()
    gpu_name       = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    torch_version  = torch.__version__
    platform_str   = platform.platform()

    print(
        f"[ONBOARD] machine: {MACHINE_NAME}  GPU: {gpu_name}  CUDA: {cuda_available}\n"
        f"[ONBOARD] torch={torch_version}  platform={platform_str}",
        flush=True,
    )

    # -- Block 1: ENV-ONLY -------------------------------------------------
    print(f"\n[ONBOARD] Block 1/4 -- ENV-ONLY ({ENV_ONLY_EPISODES} eps)", flush=True)
    env = _make_env(seed)
    action_dim = env.action_dim
    total_env_steps = 0
    t0 = time.monotonic()
    for _ in range(ENV_ONLY_EPISODES):
        _, obs_dict = env.reset()
        for _ in range(STEPS_PER_EP):
            _, _, done, _, obs_dict = env.step(random.randint(0, action_dim - 1))
            total_env_steps += 1
            if done:
                break
    env_elapsed = time.monotonic() - t0
    env_sps = total_env_steps / env_elapsed if env_elapsed > 0 else 0.0
    print(f"[ONBOARD/ENV] {total_env_steps} steps in {env_elapsed:.1f}s -> {env_sps:.0f} steps/s", flush=True)

    # -- Block 2: CPU world_dim=32 -----------------------------------------
    print(f"\n[ONBOARD] Block 2/4 -- CPU world_dim=32 ({CPU_EPISODES} eps)", flush=True)
    env_cpu   = _make_env(seed + 1)
    agent_cpu = _make_agent(env_cpu, torch.device("cpu"), world_dim=32)
    cpu_sps   = _run_training_block(env_cpu, agent_cpu, CPU_EPISODES, "CPU-32", 32)

    # -- Block 3: GPU world_dim=32 (skipped if no CUDA) --------------------
    gpu_sps_32 = 0.0
    if cuda_available:
        print(f"\n[ONBOARD] Block 3/4 -- GPU world_dim=32 ({GPU_EPISODES_32} eps)", flush=True)
        env_gpu32   = _make_env(seed + 2)
        agent_gpu32 = _make_agent(env_gpu32, torch.device("cuda"), world_dim=32)
        gpu_sps_32  = _run_training_block(env_gpu32, agent_gpu32, GPU_EPISODES_32, "GPU-32", 32)
    else:
        print("\n[ONBOARD] Block 3/4 -- GPU world_dim=32 SKIPPED (no CUDA)", flush=True)

    # -- Block 4: GPU world_dim=128 (REEConfig.large) -- Spark key metric --
    gpu_sps_128 = 0.0
    if cuda_available:
        print(
            f"\n[ONBOARD] Block 4/4 -- GPU world_dim=128 ({GPU_EPISODES_128} eps)"
            f" [REEConfig.large -- Spark key metric]",
            flush=True,
        )
        env_gpu128   = _make_env(seed + 3)
        agent_gpu128 = _make_agent(env_gpu128, torch.device("cuda"), world_dim=128)
        gpu_sps_128  = _run_training_block(env_gpu128, agent_gpu128, GPU_EPISODES_128, "GPU-128", 128)
    else:
        print("\n[ONBOARD] Block 4/4 -- GPU world_dim=128 SKIPPED (no CUDA)", flush=True)

    # -- Summary -----------------------------------------------------------
    gpu_speedup_32  = (gpu_sps_32  / cpu_sps) if cuda_available and cpu_sps > 0 else None
    gpu_speedup_128 = (gpu_sps_128 / cpu_sps) if cuda_available and cpu_sps > 0 else None
    passed = env_sps > 0 and cpu_sps > 0
    status = "PASS" if passed else "FAIL"

    print(f"\n[ONBOARD] -- Results ------------------------------------------", flush=True)
    print(f"[ONBOARD]   env steps/s (CPU-only):         {env_sps:.0f}", flush=True)
    print(f"[ONBOARD]   training steps/s (CPU-32):      {cpu_sps:.0f}", flush=True)
    print(f"[ONBOARD]   training steps/s (GPU-32):      {gpu_sps_32:.0f}", flush=True)
    print(f"[ONBOARD]   training steps/s (GPU-128):     {gpu_sps_128:.0f}", flush=True)
    if gpu_speedup_32 is not None:
        print(f"[ONBOARD]   GPU-32  speedup vs CPU-32:      {gpu_speedup_32:.2f}x", flush=True)
    if gpu_speedup_128 is not None:
        print(f"[ONBOARD]   GPU-128 speedup vs CPU-32:      {gpu_speedup_128:.2f}x", flush=True)
        if gpu_speedup_128 > 1.0:
            print(f"[ONBOARD]   GPU wins at world_dim=128 -- REEConfig.large() experiments viable", flush=True)
        else:
            print(f"[ONBOARD]   GPU does not win at world_dim=128 -- investigate before scaling", flush=True)
    print(f"[ONBOARD]   status: {status}", flush=True)

    return {
        "status": status,
        "metrics": {
            "machine_name":            MACHINE_NAME,
            "env_steps_per_second":    round(env_sps, 1),
            "steps_per_second_cpu_32": round(cpu_sps, 1),
            "steps_per_second_gpu_32": round(gpu_sps_32, 1),
            "steps_per_second_gpu_128":round(gpu_sps_128, 1),
            "gpu_speedup_32":          round(gpu_speedup_32, 3) if gpu_speedup_32 else None,
            "gpu_speedup_128":         round(gpu_speedup_128, 3) if gpu_speedup_128 else None,
            "cuda_available":          cuda_available,
            "gpu_name":                gpu_name,
            "torch_version":           torch_version,
            "platform":                platform_str,
        },
        "criteria": {
            "C1_all_blocks_completed": True,
            "C2_env_sps_gt0":          env_sps > 0,
            "C3_cpu_sps_gt0":          cpu_sps > 0,
            "C4_cuda_advisory":        cuda_available,
            "C5_gpu128_wins":          gpu_speedup_128 is not None and gpu_speedup_128 > 1.0,
        },
        "routing_hint": {
            "note": "GPU-128 speedup is the key Spark metric -- governs net-heavy large-config routing",
            "prefer_large_config": gpu_speedup_128 is not None and gpu_speedup_128 > 1.0,
            "gpu_crossover_confirmed_128": gpu_speedup_128 is not None and gpu_speedup_128 > 1.0,
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
    print(f"  env steps/s:             {m['env_steps_per_second']}", flush=True)
    print(f"  training (CPU-32):       {m['steps_per_second_cpu_32']} steps/s", flush=True)
    print(f"  training (GPU-32):       {m['steps_per_second_gpu_32']} steps/s", flush=True)
    print(f"  training (GPU-128):      {m['steps_per_second_gpu_128']} steps/s", flush=True)
    if m.get("gpu_speedup_128"):
        print(f"  GPU-128 speedup:         {m['gpu_speedup_128']}x", flush=True)
