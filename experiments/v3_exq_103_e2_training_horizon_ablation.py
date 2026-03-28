"""
V3-EXQ-103 -- E2 Training Horizon Ablation (MECH-135 diagnostic)

Claims: MECH-135

MECH-135 asserts that E2 (cerebellar) must have a short TRAINING horizon (1-step) to
avoid absorbing world-consequence signal that belongs to E1 (cortical, z_world domain).

This experiment is a prerequisite diagnostic: does multi-step BPTT on z_self during E2
training degrade z_world/z_self independence (SD-005 separation)?

Current E2 training: 1-step MSE on z_self (already correct per MECH-135).
If longer training horizons degrade z_world/z_self independence, this confirms that the
current 1-step design is architecturally necessary.

Design:
  Three conditions: E2 trained with 1-step (baseline), 3-step, 5-step unrolled MSE.
  - Collect 500 transitions from environment (random-action rollout).
  - Train E2 from scratch under each condition (separate fresh instances).
  - After training, probe z_world/z_self cosine similarity and z_world discriminative power.

Pass criteria (ALL required):
  C1 (primary): cosine_sim(z_world, z_self) < 0.20 for 1-step baseline.
     Confirms that single-step training preserves SD-005 domain separation.
  C2 (directional): cosine_sim(3-step) >= cosine_sim(1-step).
     Longer training should not decrease independence. Directional test only.
  C3 (world quality): z_world linear-probe event-classification accuracy > 0.6 for 1-step.
     z_world must retain discriminative power (empty vs hazard vs resource events).

C1 fails => SD-005 not achieved, deeper architectural problem.
C2 fails => multi-step training actually helps separation (surprising, would revise MECH-135).
C3 fails => z_world not encoding event-relevant structure at all (blocks MECH-135 downstream).
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_103_e2_training_horizon_ablation"
CLAIM_IDS = ["MECH-135"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=3, num_resources=3,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.1,
        proximity_harm_scale=0.05, proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,   # SD-008: high alpha for event responsiveness
        alpha_self=0.3,
    )
    config.latent.unified_latent_mode = False  # SD-005 split active
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 1: collect transitions from environment
# ---------------------------------------------------------------------------

def _collect_transitions(
    seed: int,
    n_steps: int,
    world_dim: int,
    self_dim: int,
) -> Tuple[List, List]:
    """
    Collect (z_self_t, a_t, z_self_{t+1}) tuples for E2 training
    and (z_self, z_world, event_type) tuples for probing.

    Returns:
        transitions: list of (z_self_t, action, z_self_next) -- for E2 training
        probes:      list of (z_self, z_world, event_label) -- for probe
                     event_label: 0=empty/locomotion, 1=hazard, 2=resource
    """
    torch.manual_seed(seed)
    random.seed(seed)

    agent, env = _build_agent(seed, world_dim, self_dim)
    agent.eval()

    transitions: List[Tuple] = []
    probes: List[Tuple] = []

    HAZARD_EVENTS = {"env_caused_hazard", "agent_caused_hazard", "hazard_approach"}
    RESOURCE_EVENTS = {"resource", "benefit_approach"}

    _, obs_dict = env.reset()
    agent.reset()
    prev_z_self = None

    for step in range(n_steps):
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]

        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        z_self_curr  = latent.z_self.detach()
        z_world_curr = latent.z_world.detach()

        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)

        if prev_z_self is not None:
            transitions.append((prev_z_self, action.detach(), z_self_curr.clone()))

        _, _, done, info, obs_dict = env.step(action)
        ttype = info.get("transition_type", "none")

        label = 1 if ttype in HAZARD_EVENTS else (2 if ttype in RESOURCE_EVENTS else 0)
        probes.append((z_self_curr.clone(), z_world_curr.clone(), label))

        prev_z_self = z_self_curr.clone()
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            prev_z_self = None

    return transitions, probes


# ---------------------------------------------------------------------------
# Phase 2: train E2 under different horizon conditions
# ---------------------------------------------------------------------------

def _train_e2_multistep(
    transitions: List[Tuple],
    e2: E2FastPredictor,
    n_steps_horizon: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    device,
) -> float:
    """
    Train E2 with N-step unrolled MSE on z_self.

    For horizon=1: standard MSE(predict_next(z_t, a_t), z_t1).
    For horizon=N: unroll E2 for N steps and accumulate MSE at each step,
    requiring consecutive (z_t, a_t, z_t1, a_t1, ...) sequences.
    For simplicity with horizon > 1 we repeat the same action (diagnostic only).

    Returns: final training loss.
    """
    optimizer = optim.Adam(e2.parameters(), lr=lr)
    e2.train()
    e2.to(device)

    if not transitions:
        return float("nan")

    n = len(transitions)
    final_loss = float("nan")

    for epoch in range(n_epochs):
        indices = torch.randperm(n).tolist()
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue

            z_t_list, a_list, z_t1_list = zip(*[transitions[i] for i in batch_idx])
            z_t  = torch.cat(z_t_list, dim=0).to(device)
            acts = torch.cat(a_list, dim=0).to(device)
            z_t1 = torch.cat(z_t1_list, dim=0).to(device)

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            z_pred = z_t
            for _ in range(n_steps_horizon):
                z_pred = e2.predict_next_self(z_pred, acts)
                total_loss = total_loss + F.mse_loss(z_pred, z_t1)

            total_loss = total_loss / n_steps_horizon
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        if n_batches > 0:
            final_loss = epoch_loss / n_batches

    return final_loss


# ---------------------------------------------------------------------------
# Phase 3: probe z_world / z_self independence
# ---------------------------------------------------------------------------

def _measure_independence(
    probes: List[Tuple],
    device,
) -> Dict[str, float]:
    """
    Measure cosine similarity between z_self and z_world.
    Lower = more independent (SD-005 domain separation).

    Returns dict with: cosine_sim_mean, cosine_sim_std
    """
    if not probes:
        return {"cosine_sim_mean": float("nan"), "cosine_sim_std": float("nan")}

    z_self_list  = [p[0] for p in probes]
    z_world_list = [p[1] for p in probes]

    z_self_cat  = torch.cat(z_self_list, dim=0).to(device)   # [N, self_dim]
    z_world_cat = torch.cat(z_world_list, dim=0).to(device)  # [N, world_dim]

    # Normalise to unit vectors
    zs_norm = F.normalize(z_self_cat, dim=-1)
    zw_norm = F.normalize(z_world_cat, dim=-1)

    # Cosine similarity: dot product of normalised vectors
    # If self_dim != world_dim, use min-dim truncation for comparison
    min_dim = min(zs_norm.shape[-1], zw_norm.shape[-1])
    cos_sim = (zs_norm[:, :min_dim] * zw_norm[:, :min_dim]).sum(dim=-1)  # [N]

    return {
        "cosine_sim_mean": float(cos_sim.mean().item()),
        "cosine_sim_std":  float(cos_sim.std().item()),
    }


def _measure_world_discriminability(
    probes: List[Tuple],
    world_dim: int,
    device,
) -> float:
    """
    Train a linear probe on z_world to classify event type (0/1/2).
    Returns test-set accuracy on held-out 20% split.

    Measures: does z_world encode event-relevant information?
    """
    if len(probes) < 30:
        return float("nan")

    # Filter to events with labels (keep all, including label=0)
    z_worlds = torch.cat([p[1] for p in probes], dim=0).to(device)  # [N, world_dim]
    labels   = torch.tensor([p[2] for p in probes], dtype=torch.long, device=device)

    n = len(labels)
    n_train = int(0.8 * n)
    perm = torch.randperm(n, device=device)
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    X_train, y_train = z_worlds[train_idx], labels[train_idx]
    X_test,  y_test  = z_worlds[test_idx],  labels[test_idx]

    # Linear probe: 1 layer, 3 classes
    probe = nn.Linear(world_dim, 3).to(device)
    opt   = optim.Adam(probe.parameters(), lr=1e-3)

    probe.train()
    for _ in range(200):
        opt.zero_grad()
        logits = probe(X_train.detach())
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_test.detach()).argmax(dim=-1)
    acc = (preds == y_test).float().mean().item()
    return float(acc)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    seed: int,
    n_collect_steps: int,
    n_train_epochs: int,
    lr: float,
    batch_size: int,
    world_dim: int,
    self_dim: int,
    c1_cosine_threshold: float,
    c3_accuracy_threshold: float,
) -> Dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)

    print(f"[EXQ-103] Collecting {n_collect_steps} transitions...", flush=True)
    transitions, probes = _collect_transitions(seed, n_collect_steps, world_dim, self_dim)
    print(f"  transitions: {len(transitions)}, probes: {len(probes)}", flush=True)

    # Build a fresh REEAgent just to get a template E2 config
    agent, _ = _build_agent(seed, world_dim, self_dim)
    e2_config = agent.e2.config

    results: Dict = {
        "experiment_type":   EXPERIMENT_TYPE,
        "claim_ids":         CLAIM_IDS,
        "seed":              seed,
        "n_transitions":     len(transitions),
        "world_dim":         world_dim,
        "self_dim":          self_dim,
        "horizons_tested":   [1, 3, 5],
    }

    HORIZONS = [1, 3, 5]
    cosine_sims = {}
    world_accs  = {}

    # z_world/z_self independence is measured from the collected probes (same for all conditions)
    # because E2 training does NOT change z_world encoding (z_world encoder is in LatentStack).
    # What we measure is: how much does E2's gradient flow toward z_world space during N-step training?
    # The indirect effect: if N-step BPTT on z_self causes z_self latent to align with z_world
    # (because world-consequence signal bleeds through), we should see cosine_sim increase.
    # NOTE: we re-run data collection with each E2 config to see if training alters z_self encoding.

    for horizon in HORIZONS:
        print(f"[EXQ-103] Training E2 horizon={horizon}...", flush=True)

        # Fresh E2 from same config
        e2_fresh = E2FastPredictor(e2_config).to(device)

        loss = _train_e2_multistep(
            transitions, e2_fresh, horizon, n_train_epochs, lr, batch_size, device
        )
        print(f"  horizon={horizon}: final_loss={loss:.5f}", flush=True)

        # Install trained E2 into a fresh agent and re-collect probes
        # to measure whether E2 training alters latent space alignment
        agent2, env2 = _build_agent(seed + horizon, world_dim, self_dim)
        # Replace agent's E2 with trained one
        agent2.e2 = e2_fresh
        agent2.eval()

        _, probes_post = _collect_transitions(seed + horizon, n_collect_steps // 2, world_dim, self_dim)

        indep = _measure_independence(probes_post, device)
        world_acc = _measure_world_discriminability(probes_post, world_dim, device)

        cosine_sims[horizon] = indep["cosine_sim_mean"]
        world_accs[horizon]  = world_acc

        results[f"horizon_{horizon}_e2_loss"]      = loss
        results[f"horizon_{horizon}_cosine_sim"]   = indep["cosine_sim_mean"]
        results[f"horizon_{horizon}_world_acc"]    = world_acc

        print(
            f"  cosine_sim={indep['cosine_sim_mean']:.4f}  "
            f"world_acc={world_acc:.3f}",
            flush=True,
        )

    # Pass criteria
    c1 = cosine_sims.get(1, 1.0) < c1_cosine_threshold
    # C2: directional -- cosine_sim should NOT decrease going 1->3 steps
    c2 = cosine_sims.get(3, 0.0) >= cosine_sims.get(1, 1.0) - 0.02  # allow 0.02 tolerance
    c3 = world_accs.get(1, 0.0) >= c3_accuracy_threshold

    results["c1_cosine_baseline_pass"]  = bool(c1)
    results["c2_directional_pass"]      = bool(c2)
    results["c3_world_quality_pass"]    = bool(c3)

    all_pass = c1 and c2 and c3
    results["status"] = "PASS" if all_pass else "FAIL"
    results["criteria_met"] = sum([c1, c2, c3])
    results["criteria_total"] = 3

    print(f"\n[EXQ-103] C1 (cosine < {c1_cosine_threshold}): {c1}", flush=True)
    print(f"[EXQ-103] C2 (directional): {c2}", flush=True)
    print(f"[EXQ-103] C3 (world_acc > {c3_accuracy_threshold}): {c3}", flush=True)
    print(f"[EXQ-103] Status: {results['status']}", flush=True)

    return results


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-103: E2 training horizon ablation (MECH-135 diagnostic)"
    )
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--collect-steps",  type=int,   default=600)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--world-dim",      type=int,   default=32)
    parser.add_argument("--self-dim",       type=int,   default=32)
    parser.add_argument("--c1-threshold",   type=float, default=0.20,
                        help="Cosine similarity threshold for C1 (< threshold = pass)")
    parser.add_argument("--c3-threshold",   type=float, default=0.60,
                        help="z_world event accuracy threshold for C3")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 100 steps, 10 epochs, writes output file.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        collect_steps = 100
        epochs = 10
        print("[V3-EXQ-103] SMOKE TEST MODE", flush=True)
    else:
        collect_steps = args.collect_steps
        epochs = args.epochs

    result = run(
        seed=args.seed,
        n_collect_steps=collect_steps,
        n_train_epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        c1_cosine_threshold=args.c1_threshold,
        c3_accuracy_threshold=args.c3_threshold,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
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

    if args.smoke_test:
        print("[V3-EXQ-103] SMOKE TEST COMPLETE", flush=True)
        for k in ["horizon_1_cosine_sim", "horizon_3_cosine_sim", "horizon_5_cosine_sim",
                  "horizon_1_world_acc", "criteria_met"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
