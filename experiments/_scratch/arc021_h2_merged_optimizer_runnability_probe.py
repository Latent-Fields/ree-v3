#!/opt/local/bin/python3
"""Reproducer: the ARC-021 H2 leg's MERGED arm does not run on current ree_core.

Context. The confirmed autopsy failure_autopsy_V3-EXQ-993a_2026-09-05 routes its
H2 (representation-axis) leg through the EXISTING driver
`experiments/v3_spark_arc021_three_loop_scale.py`, and records that the driver
"has NEVER produced a manifest", that "RUNNABILITY IS UNVERIFIED", and that "a
smoke run against current agent.py is the FIRST step of this leg, not an
afterthought". This file is that smoke run, and its answer is NO.

WHAT IT SHOWS (measured 2026-09-08 on DLAPTOP, torch 2.10.0, ree-v3 8a80ee33b5):

  SEPARATE arm (three independent optimizers)  -> runs; 4/4 steps clean.
  MERGED   arm (one optimizer over agent.parameters(), combined loss E1+E2+E3)
                                               -> CRASHES on step 1 (the SECOND
                                                  step), every time.

    RuntimeError: one of the variables needed for gradient computation has been
    modified by an inplace operation: [torch.FloatTensor [16, 256]], which is
    output 0 of AsStridedBackward0, is at version 1; expected version 0 instead.

MECHANISM (pinned with torch.autograd.set_detect_anomaly). The [16, 256] tensor
is the learnable parameter `e1.context_memory.memory`
(`ree_core/predictors/e1_deep.py:127`, `nn.Parameter(torch.randn(num_slots,
memory_dim) * 0.01)`). `ContextMemory.write` mutates it IN PLACE through `.data`
(`e1_deep.py:272-273`):

    self.memory.data[min_idx] = 0.9 * self.memory.data[min_idx] + 0.1 * write_signal.mean(0)

which bumps the parameter's autograd version counter. The anomaly traceback names
the invalidated forward as the PREVIOUS step's `_e1_tick` ->
`E1Deep.forward` -> `generate_prior` -> `ContextMemory.read` -> `value_proj(memory)`
(`agent.py:5878`, `e1_deep.py:1741`, `:1002`, `:226`). So the merged arm's combined
loss retains a graph that reaches back across an environment step into a read of
`self.memory`, and the in-place write invalidates it before backward.

WHY ONLY THE MERGED ARM. The SEPARATE arm backwards each of its three losses
immediately and independently, so no graph survives long enough to be
invalidated. The merged arm is defined by the opposite property -- ONE backward
over a combined objective spanning all three modules -- so the entanglement is
intrinsic to the manipulation under test, not to the driver's coding.

CONSEQUENCE. This is not a driver bug that a queue-experiment session can repair.
`self.memory` is a learnable parameter inside `ree_core`; excluding it from the
merged optimizer, or wrapping the write in `torch.no_grad()`, would change what
the MERGED arm IS -- and the merged arm is the ablation. The fix belongs in
`ree_core/predictors/e1_deep.py` (i.e. `/implement-substrate`), not here.

Step 2.5c independently reaches the same stop: `ContextMemory.write` is the
`substrate_paths` entry of the OPEN, severity `corrupting` substrate-queue item
`contextmemory-write-path-addressing-degeneracy` (status
`implemented_pending_validation`, which the gate counts as open). Note the two are
DIFFERENT defects in the same function -- that entry is about hard-argmin
addressing degeneracy; this is an autograd-version violation -- so this
reproducer is a NEW finding, not a restatement.

Writes nothing. Run: /opt/local/bin/python3 experiments/_scratch/arc021_h2_merged_optimizer_runnability_probe.py
"""

from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

N_STEPS = 4


def _make_env(seed: int):
    """Byte-identical kwargs to v3_spark_arc021_three_loop_scale._make_env."""
    return CausalGridWorldV2(
        seed=seed, size=10, num_hazards=3, num_resources=5, hazard_harm=0.02,
        env_drift_interval=10, env_drift_prob=0.05, proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03, proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def probe(merged: bool, anomaly: bool = False) -> bool:
    """Run N_STEPS of the spark driver's own train loop. True = ran clean."""
    torch.autograd.set_detect_anomaly(anomaly, check_nan=False)
    torch.manual_seed(0)
    random.seed(0)

    env = _make_env(0)
    cfg = REEConfig.large(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, reafference_action_dim=env.action_dim,
    )
    world_dim = cfg.latent.world_dim
    agent = REEAgent(cfg)

    if merged:
        merged_opt = optim.Adam(agent.parameters(), lr=1e-3)
    else:
        e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-4)
        e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-4)
        e3_opt = optim.Adam(
            list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
            lr=1e-3,
        )
    agent.train()

    _, obs_dict = env.reset()
    agent.reset()
    z_self_t = None

    for i in range(N_STEPS):
        obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
        obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
        if agent._current_latent is not None:
            z_self_t = agent._current_latent.z_self.detach().clone()

        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
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

        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        harm_target = torch.tensor([[1.0 if harm_signal < 0 else 0.0]])
        try:
            if merged:
                # UNDETACHED z_world -- the spark driver's own comment calls this
                # "allow contamination gradient". It is the manipulation.
                harm_loss = F.mse_loss(agent.e3.harm_eval(latent.z_world), harm_target)
                combined = (
                    (e1_loss if e1_loss.requires_grad else torch.tensor(0.0))
                    + (e2_loss if e2_loss.requires_grad else torch.tensor(0.0))
                    + harm_loss
                )
                if combined.requires_grad:
                    merged_opt.zero_grad()
                    combined.backward()
                    merged_opt.step()
            else:
                if e1_loss.requires_grad:
                    e1_opt.zero_grad(); e1_loss.backward(); e1_opt.step()
                if e2_loss.requires_grad:
                    e2_opt.zero_grad(); e2_loss.backward(); e2_opt.step()
                harm_loss = F.mse_loss(
                    agent.e3.harm_eval(latent.z_world.detach()), harm_target
                )
                e3_opt.zero_grad(); harm_loss.backward(); e3_opt.step()
        except RuntimeError:
            arm = "MERGED" if merged else "SEPARATE"
            print(f"  {arm}: CRASHED on step {i} (0-indexed)")
            for line in traceback.format_exc().split("\n"):
                s = line.strip()
                if s.startswith("RuntimeError") or "ree_core" in s:
                    print("    " + s)
            return False

        agent.update_residue(harm_signal)
        if done:
            break

    arm = "MERGED" if merged else "SEPARATE"
    print(f"  {arm}: ran {N_STEPS} step(s) clean (world_dim={world_dim})")
    return True


if __name__ == "__main__":
    print("[arc021-h2-runnability] spark driver, current ree_core")
    sep_ok = probe(merged=False)
    mrg_ok = probe(merged=True)
    print()
    print(f"  SEPARATE runnable: {sep_ok}")
    print(f"  MERGED   runnable: {mrg_ok}")
    if not mrg_ok:
        print("  -> H2 leg NOT buildable on this substrate; see module docstring.")
