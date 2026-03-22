"""
V3-EXQ-070 -- Daniel-PC Batch GPU Speedup Calibration

Claims: None (infrastructure diagnostic)

Purpose: Determine the batch size at which Daniel-PC's RTX 2060 Super beats its
CPU for the world_forward (E2 transition model) kernel. The onboarding smoke test
only measured batch=1 training (GPU 3x slower than CPU at that scale). This experiment
sweeps batch sizes [1, 4, 8, 16, 32, 64, 128, 256, 512] and measures gradient-update
throughput (samples/sec) on both CPU and GPU for each.

The world_forward kernel is the bottleneck in SD-003 counterfactual attribution:
  causal_sig = E3(E2(z_world, a_actual)) - E3(E2(z_world, a_cf))
It is also the training target for the E2 world_transition head. Knowing the
crossover batch size guides routing of future experiments that use experience
replay or batched attribution (SD-010, EXQ-071+).

Protocol:
  Phase 1 -- Collect 5000 world transitions using the agent + env (CPU, no GPU needed)
  Phase 2 -- For each batch_size in BATCH_SIZES:
      a. Benchmark CPU: run N_ITERS gradient update steps (world_forward + MSE + backward)
         Record: samples_per_second = batch_size * iters / elapsed
      b. Benchmark GPU (skip if no CUDA): same kernel on cuda device
  Phase 3 -- Find crossover batch size (smallest where GPU > CPU throughput)
  Phase 4 -- Write preferred_batch_mode annotation to Daniel-PC.json

Pass criteria (diagnostic -- not scientific):
  C1: All batch sizes complete without exception on CPU
  C2: env collection completed (n_transitions >= MIN_TRANSITIONS)
  C3: At least one batch size measured on both CPU and GPU (advisory if no CUDA)

Machine affinity: Daniel-PC (machine-specific benchmark)
Estimated runtime: ~35 min on RTX 2060 Super + i5-8600K
"""

import sys
import time
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_070_batch_gpu_calibration"
CLAIM_IDS: list = []

BATCH_SIZES     = [1, 4, 8, 16, 32, 64, 128, 256, 512]
N_ITERS         = 300      # gradient update iterations per (batch_size, device) cell
MIN_TRANSITIONS = 4000     # collect at least this many before benchmarking
COLLECT_EPISODES = 60      # env collection episodes
STEPS_PER_EP    = 200


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


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )


def _collect_transitions(seed: int, n_episodes: int, steps_per_ep: int):
    """Collect world transitions for replay buffer. Returns list of (z_world, action, z_world_next) CPU tensors."""
    env    = _make_env(seed)
    config = _make_config(env)
    agent  = REEAgent(config)
    agent.eval()

    transitions = []
    num_actions = env.action_dim

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev       = None

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            z_world_curr = latent.z_world.detach().cpu()

            action_idx = random.randint(0, num_actions - 1)
            a_onehot   = torch.zeros(1, num_actions)
            a_onehot[0, action_idx] = 1.0

            _, _, done, _, obs_dict = env.step(action_idx)

            if z_world_prev is not None and a_prev is not None:
                transitions.append((z_world_prev, a_prev, z_world_curr))

            z_world_prev = z_world_curr
            a_prev       = a_onehot

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"[EXQ-070] Collect ep {ep+1}/{n_episodes}  transitions={len(transitions)}",
                flush=True,
            )

    print(f"[EXQ-070] Collected {len(transitions)} transitions", flush=True)
    return transitions, config, env.action_dim


def _benchmark_kernel(world_dim: int, num_actions: int, transitions: list,
                      batch_size: int, device: torch.device, n_iters: int) -> float:
    """
    Benchmark E2 world_forward + backprop at a given batch_size on given device.
    Returns samples_per_second (= batch_size * iters / elapsed).
    """
    # Build a fresh minimal world_forward module (Linear chain matching E2 world_transition)
    # We use a fresh model so benchmarks are not confounded by prior optimiser state
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    # Minimal config just for the E2 head dimensions
    class _FakeEnv:
        body_obs_dim = 12
        world_obs_dim = 200   # typical CausalGridWorldV2
        action_dim   = num_actions

    fenv   = _FakeEnv()
    config = REEConfig.from_dims(
        body_obs_dim=fenv.body_obs_dim,
        world_obs_dim=fenv.world_obs_dim,
        action_dim=fenv.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=fenv.action_dim,
    )
    agent = REEAgent(config).to(device)
    wt_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    opt = optim.Adam(wt_params, lr=1e-3)

    n   = len(transitions)
    t0  = time.monotonic()

    for _ in range(n_iters):
        idxs  = [random.randint(0, n - 1) for _ in range(batch_size)]
        zw_b  = torch.cat([transitions[i][0] for i in idxs], dim=0).to(device)
        a_b   = torch.cat([transitions[i][1] for i in idxs], dim=0).to(device)
        zw1_b = torch.cat([transitions[i][2] for i in idxs], dim=0).to(device)

        pred = agent.e2.world_forward(zw_b, a_b)
        loss = F.mse_loss(pred, zw1_b)
        opt.zero_grad()
        loss.backward()
        opt.step()

    elapsed = time.monotonic() - t0
    sps = (batch_size * n_iters) / elapsed if elapsed > 0 else 0.0
    return sps


def run(seed: int = 42) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cuda_available = torch.cuda.is_available()
    gpu_name       = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    print(
        f"[EXQ-070] Batch GPU Calibration  CUDA={cuda_available}  GPU={gpu_name}",
        flush=True,
    )
    print(
        f"[EXQ-070] Collecting {COLLECT_EPISODES} eps of world transitions...",
        flush=True,
    )

    transitions, config, num_actions = _collect_transitions(seed, COLLECT_EPISODES, STEPS_PER_EP)

    c2_pass = len(transitions) >= MIN_TRANSITIONS
    if not c2_pass:
        print(
            f"[EXQ-070] WARNING: only {len(transitions)} transitions (need {MIN_TRANSITIONS})",
            flush=True,
        )

    # Truncate to a round number to avoid repeated random-without-replacement issues
    transitions = transitions[:min(len(transitions), 6000)]

    world_dim   = 32  # matches config
    cpu_device  = torch.device("cpu")
    gpu_device  = torch.device("cuda") if cuda_available else None

    cpu_sps: dict[int, float] = {}
    gpu_sps: dict[int, float] = {}

    for bs in BATCH_SIZES:
        # Skip batch sizes larger than buffer
        if bs > len(transitions):
            print(f"[EXQ-070] batch={bs}: skipped (buffer too small)", flush=True)
            cpu_sps[bs] = 0.0
            gpu_sps[bs] = 0.0
            continue

        print(f"[EXQ-070] Benchmarking batch={bs}...", flush=True)

        c = _benchmark_kernel(world_dim, num_actions, transitions, bs, cpu_device, N_ITERS)
        cpu_sps[bs] = round(c, 1)
        print(f"  CPU: {c:.0f} samples/s", flush=True)

        if gpu_device is not None:
            g = _benchmark_kernel(world_dim, num_actions, transitions, bs, gpu_device, N_ITERS)
            gpu_sps[bs] = round(g, 1)
            print(f"  GPU: {g:.0f} samples/s  speedup={g/c:.2f}x" if c > 0 else f"  GPU: {g:.0f} samples/s", flush=True)
        else:
            gpu_sps[bs] = 0.0

    # Find crossover
    crossover_batch = None
    if cuda_available:
        for bs in BATCH_SIZES:
            c = cpu_sps.get(bs, 0.0)
            g = gpu_sps.get(bs, 0.0)
            if c > 0 and g > c:
                crossover_batch = bs
                break

    # Routing recommendation
    if crossover_batch is not None:
        preferred_batch_mode = f"gpu_at_batch_{crossover_batch}_or_larger"
        routing_note = (
            f"GPU faster than CPU at batch>={crossover_batch}. "
            f"Route replay-heavy experiments with batch>={crossover_batch} to Daniel-PC GPU."
        )
    elif cuda_available:
        preferred_batch_mode = "cpu_preferred"
        routing_note = (
            "GPU never exceeded CPU throughput at tested batch sizes (max={} samples/s GPU vs {} samples/s CPU). "
            "Larger networks (world_dim>=128) or more complex models may shift crossover."
        ).format(
            max(gpu_sps.values()) if gpu_sps else 0,
            max(cpu_sps.values()) if cpu_sps else 0,
        )
    else:
        preferred_batch_mode = "cpu_only"
        routing_note = "No CUDA available. All experiments run on CPU."

    print(f"\n[EXQ-070] -- Results ----------------------------------------", flush=True)
    print(f"[EXQ-070]   crossover_batch: {crossover_batch}", flush=True)
    print(f"[EXQ-070]   preferred_batch_mode: {preferred_batch_mode}", flush=True)
    for bs in BATCH_SIZES:
        c = cpu_sps.get(bs, 0.0)
        g = gpu_sps.get(bs, 0.0)
        speedup = f"{g/c:.2f}x" if c > 0 and g > 0 else "N/A"
        print(f"[EXQ-070]   batch={bs:4d}  CPU={c:8.0f} smp/s  GPU={g:8.0f} smp/s  speedup={speedup}", flush=True)
    print(f"[EXQ-070]   routing note: {routing_note}", flush=True)

    c1_pass = True  # reaching here means no exception
    c3_pass = cuda_available  # advisory

    all_pass = c1_pass and c2_pass
    status   = "PASS" if all_pass else "FAIL"

    metrics: dict = {
        "crossover_batch":       crossover_batch,
        "preferred_batch_mode":  preferred_batch_mode,
        "cuda_available":        cuda_available,
        "gpu_name":              gpu_name,
        "n_transitions":         len(transitions),
        "n_iters_per_cell":      N_ITERS,
        "batch_sizes_tested":    BATCH_SIZES,
        "crit1_pass":            1.0 if c1_pass else 0.0,
        "crit2_transitions_ok":  1.0 if c2_pass else 0.0,
        "crit3_cuda_advisory":   1.0 if c3_pass else 0.0,
    }
    for bs in BATCH_SIZES:
        metrics[f"cpu_sps_batch{bs}"] = cpu_sps.get(bs, 0.0)
        metrics[f"gpu_sps_batch{bs}"] = gpu_sps.get(bs, 0.0)

    summary_markdown = f"""# V3-EXQ-070 -- Batch GPU Speedup Calibration (Daniel-PC)

**Status:** {status}
**Machine:** Daniel-PC (RTX 2060 Super 8GB, i5-8600K)
**Kernel:** E2 world_forward + MSE + backward (world_dim=32, num_actions={num_actions})
**Transitions collected:** {len(transitions)}
**Iters per cell:** {N_ITERS}

## Throughput by Batch Size (samples/sec)

| batch | CPU smp/s | GPU smp/s | speedup |
|---|---|---|---|
{"".join(f"| {bs} | {cpu_sps.get(bs,0):.0f} | {gpu_sps.get(bs,0):.0f} | {f'{gpu_sps.get(bs,0)/cpu_sps.get(bs,0):.2f}x' if cpu_sps.get(bs,0) > 0 else 'N/A'} |\n" for bs in BATCH_SIZES)}

## Routing Recommendation

- **Crossover batch:** {crossover_batch if crossover_batch else 'none found'}
- **preferred_batch_mode:** `{preferred_batch_mode}`
- {routing_note}

## Pass Criteria

| Criterion | Result |
|---|---|
| C1: no exception in any benchmark cell | PASS |
| C2: n_transitions >= {MIN_TRANSITIONS} | {"PASS" if c2_pass else "FAIL"} ({len(transitions)}) |
| C3: CUDA available (advisory) | {"PASS" if c3_pass else "N/A (no CUDA)"} |

**Overall:** {status}
"""

    return {
        "status":           status,
        "metrics":          metrics,
        "summary_markdown": summary_markdown,
        "claim_ids":        CLAIM_IDS,
        "evidence_direction": "supports",
        "experiment_type":  EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "routing_recommendation": {
            "crossover_batch":      crossover_batch,
            "preferred_batch_mode": preferred_batch_mode,
            "routing_note":         routing_note,
        },
    }


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(seed=args.seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim"]              = "infrastructure"
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

    # Update Daniel-PC.json with calibration results
    machine_json_path = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "contributors" / "machines" / "Daniel-PC.json"
    )
    if machine_json_path.exists():
        try:
            machine_data = json.loads(machine_json_path.read_text(encoding="utf-8"))
            machine_data["batch_calibration"] = {
                "run_id":               result["run_id"],
                "crossover_batch":      result["routing_recommendation"]["crossover_batch"],
                "preferred_batch_mode": result["routing_recommendation"]["preferred_batch_mode"],
                "routing_note":         result["routing_recommendation"]["routing_note"],
                "calibrated_at":        ts,
            }
            machine_json_path.write_text(
                json.dumps(machine_data, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Daniel-PC.json updated with batch calibration results.", flush=True)
        except Exception as e:
            print(f"WARNING: could not update Daniel-PC.json: {e}", flush=True)
    else:
        print(f"WARNING: Daniel-PC.json not found at {machine_json_path}", flush=True)
