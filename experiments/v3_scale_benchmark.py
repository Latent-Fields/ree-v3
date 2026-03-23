#!/opt/local/bin/python3
"""
V3 GPU/CPU Scale Crossover Benchmark

Operational benchmark (not a scientific experiment). Sweeps world_dim across
[32, 64, 128, 256] and measures CPU vs GPU throughput at each scale. Identifies
where GPU becomes faster than CPU on the host machine.

Purpose:
  - Establish baseline data on current hardware (Mac, Daniel-PC)
  - Compare against NVIDIA Spark when available
  - Confirm world_dim >= 128 threshold documented in CLAUDE.md

Result JSON written to evidence/experiments/v3_scale_benchmark/ for archiving.
Not queued with scientific EXQ number -- queue_id: V3-BENCH-scale-001.

Estimated runtime:
  Daniel-PC (i5-8600K + GTX 1050 Ti): ~40 min
  Mac (M2 Air, CPU only):             ~25 min
  NVIDIA Spark (expected):             ~5 min with GPU wins at world_dim >= 128
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


EXPERIMENT_TYPE = "v3_scale_benchmark"
CLAIM_IDS: list = []

WORLD_DIMS = [32, 64, 128, 256]
EPISODES_PER_BLOCK = 5   # small -- timing not training
STEPS_PER_EP = 100
SEED = 42


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


def _make_agent(env: CausalGridWorldV2, world_dim: int, device: torch.device) -> REEAgent:
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
    # Scale hidden dims at larger world_dim to match REEConfig.large/xlarge presets
    if world_dim >= 128:
        config.e1.hidden_dim = world_dim * 2
        config.e2.hidden_dim = world_dim * 2
        config.e3.hidden_dim = world_dim
        config.hippocampal.hidden_dim = world_dim * 2
        config.residue.hidden_dim = world_dim
        config.residue.num_basis_functions = world_dim // 2
    elif world_dim >= 64:
        config.e1.hidden_dim = 128
        config.e2.hidden_dim = 128
        config.e3.hidden_dim = 64
    agent = REEAgent(config)
    agent.to(device)
    agent.device = device
    return agent


def _run_block(
    env: CausalGridWorldV2,
    agent: REEAgent,
    world_dim: int,
    label: str,
) -> float:
    """Run full training loop for EPISODES_PER_BLOCK episodes. Returns steps/sec."""
    optim.Adam(agent.parameters(), lr=1e-3)
    total_steps = 0
    agent.train()

    t0 = time.monotonic()
    for ep in range(EPISODES_PER_BLOCK):
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

    elapsed = time.monotonic() - t0
    sps = total_steps / elapsed if elapsed > 0 else 0.0
    print(
        f"  [{label:>3}] world_dim={world_dim:>3}"
        f"  {total_steps} steps in {elapsed:.1f}s -> {sps:.0f} steps/s",
        flush=True,
    )
    return sps


def run(seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    torch_version = torch.__version__
    platform_str = platform.platform()

    print(f"\n[BENCH] GPU: {gpu_name}  CUDA: {cuda_available}", flush=True)
    print(f"[BENCH] torch={torch_version}", flush=True)
    print(f"[BENCH] platform: {platform_str}", flush=True)
    print(f"[BENCH] Sweeping world_dim: {WORLD_DIMS}", flush=True)
    print(f"[BENCH] {EPISODES_PER_BLOCK} eps x {STEPS_PER_EP} steps per block", flush=True)

    results_by_dim: dict = {}

    for world_dim in WORLD_DIMS:
        print(
            f"\n[BENCH] world_dim={world_dim}"
            f" -------------------------------------------",
            flush=True,
        )

        # CPU block
        env_cpu   = _make_env(seed)
        agent_cpu = _make_agent(env_cpu, world_dim, torch.device("cpu"))
        cpu_sps   = _run_block(env_cpu, agent_cpu, world_dim, "CPU")

        # GPU block
        gpu_sps = 0.0
        if cuda_available:
            env_gpu   = _make_env(seed + 1)
            agent_gpu = _make_agent(env_gpu, world_dim, torch.device("cuda"))
            gpu_sps   = _run_block(env_gpu, agent_gpu, world_dim, "GPU")

        gpu_speedup = (gpu_sps / cpu_sps) if cuda_available and cpu_sps > 0 else None
        gpu_wins    = gpu_speedup is not None and gpu_speedup > 1.0

        if gpu_speedup is not None:
            print(
                f"  GPU speedup at world_dim={world_dim}: {gpu_speedup:.2f}x"
                f"  {'(GPU WINS)' if gpu_wins else '(CPU wins)'}",
                flush=True,
            )

        results_by_dim[str(world_dim)] = {
            "cpu_sps":    round(cpu_sps, 1),
            "gpu_sps":    round(gpu_sps, 1),
            "gpu_speedup": round(gpu_speedup, 3) if gpu_speedup is not None else None,
            "gpu_wins":   gpu_wins,
        }

    # Find crossover
    crossover_dim = None
    for wd in WORLD_DIMS:
        if results_by_dim[str(wd)]["gpu_wins"]:
            crossover_dim = wd
            break

    # Summary table
    print(f"\n[BENCH] -- Summary ------------------------------------------", flush=True)
    print(f"  {'world_dim':>9}  {'CPU sps':>9}  {'GPU sps':>9}  {'speedup':>10}", flush=True)
    for wd in WORLD_DIMS:
        r  = results_by_dim[str(wd)]
        sp = f"{r['gpu_speedup']:.2f}x" if r["gpu_speedup"] is not None else "N/A"
        win_marker = " <--" if r["gpu_wins"] else ""
        print(
            f"  {wd:>9}  {r['cpu_sps']:>9.0f}  {r['gpu_sps']:>9.0f}  {sp:>10}{win_marker}",
            flush=True,
        )

    if crossover_dim is not None:
        print(f"\n[BENCH] GPU crossover: world_dim >= {crossover_dim}", flush=True)
    elif cuda_available:
        print(f"\n[BENCH] GPU did not win at any tested world_dim -- CPU always faster", flush=True)
    else:
        print(f"\n[BENCH] No CUDA -- CPU-only baseline recorded", flush=True)

    return {
        "status": "PASS",
        "metrics": {
            "cuda_available":      cuda_available,
            "gpu_name":            gpu_name,
            "torch_version":       torch_version,
            "platform":            platform_str,
            "world_dims_tested":   WORLD_DIMS,
            "episodes_per_block":  EPISODES_PER_BLOCK,
            "steps_per_episode":   STEPS_PER_EP,
            "results_by_dim":      results_by_dim,
            "crossover_dim":       crossover_dim,
        },
        "criteria": {
            "C1_all_dims_completed": True,   # reaching here means no exception
            "C2_crossover_found":    crossover_dim is not None,
        },
        "routing_hint": {
            "note": "GPU wins above crossover_dim -- use REEConfig.large() for Spark experiments",
            "gpu_beneficial_from": crossover_dim,
        },
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    result = run()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim"]              = "operational_benchmark"
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
    if result["metrics"]["crossover_dim"] is not None:
        print(
            f"GPU crossover confirmed at world_dim >= {result['metrics']['crossover_dim']}",
            flush=True,
        )
