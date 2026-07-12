#!/opt/local/bin/python3
"""
V3-EXQ-680 -- EXP-0380 MECH-423 cross-model SUPER-ADDITIVITY 3-arm ablation.
SLEEP DRIVER: N/A (R3 readiness calls CrossModuleConsolidator.consolidate()
              directly; no SleepLoopManager / sleep cycle is driven).

EXPERIMENT_PURPOSE = "evidence". claim_ids = ["MECH-423"].

HYPOTHESIS (MECH-423)
---------------------
Cross-model integration is SUPER-ADDITIVE: a representation co-trained over a
SHARED latent by two task objectives exceeds the SUM of what each objective
achieves in isolation (Caruana 1997 shared-rep MTL; "cross-pollination
turbo-charges development"), made falsifiable on the V3 substrate. The
integration locus is the shared encoder (LatentStack): when both task heads'
gradients flow into the shared latent the encoder develops features shaped by
both tasks; with a gradient stop between the streams it does not.

THREE ARMS (param- and compute-matched; the ONLY structural difference is the
gradient-stop / proxy-head, so a delta cannot be a capacity artefact)
----------------------------------------------------------------------------
  ARM_ISOLATED          -- two task heads train on a DETACHED shared latent:
                           NO gradient flows from either head into the shared
                           encoder ("gradient stop between streams -- no shared
                           features"). Cross-module consolidation runs on the
                           BLOCKED schedule (the McClelland 1995 catastrophic-
                           interference control). This is the ADDITIVE baseline
                           = sum of the two isolated marginal gains.
  ARM_INTEGRATED_PAIR   -- both task heads train on the LIVE shared latent
                           (cross-module gradient flow into the shared encoder);
                           iterative-inference settling ON (R2); cross-module
                           consolidation on the INTERLEAVED schedule (R3). This
                           is the proposal's registered E1<->E2 overlap instance
                           and is the LOAD-BEARING arm for the MECH-423 verdict.
  ARM_INTEGRATED_TRIPLE -- PAIR plus a third PROXY object-binding head that also
                           co-shapes the shared latent. The proxy is a stand-in
                           ATTACHMENT SURFACE for the ARC-080 object spine, which
                           is candidate / v3_pending / implementation_phase v4
                           and NOT yet built in V3 (the only V3 object module,
                           entities/object_file_buffer.py, is explicitly non-
                           trainable -- no nn.Module, no gradient). The TRIPLE
                           delta is therefore EXPLORATORY ONLY and is NOT tagged
                           to ARC-080: this run writes evidence to MECH-423 only.
                           Whether adding the object stream FURTHER super-adds is
                           a genuinely V4 question (it concerns ARC-080's own
                           contribution), so MECH-423's V3 verdict rests on the
                           ISOLATED-vs-PAIR contrast and does not depend on it.

OPERATIONALISATION (the two task objectives)
--------------------------------------------
The "world-model" and "affordance" objectives are the two streams predicting
each other through the shared latent (a cross-stream alignment task that is
non-trivial and genuinely benefits from a co-shaped shared representation):
  * world_head : z_self -> z_world   (world-model: recover world structure from
                 body state through the shared latent)
  * self_head  : z_world -> z_self   (affordance: recover the body/affordance
                 state from world context through the shared latent)
Score per arm = world_R2 + affordance_R2 (held-out R2 of each head). The
super-additivity test is on the DELTA of this combined score, integrated minus
isolated -- the random/chance baselines cancel in the difference, so this equals
"integrated gain minus the sum of isolated marginal gains".

PHASED TRAINING
---------------
P0 integration phase: rollout; per step the two (PAIR: three) heads co-train
   over the shared latent -- LIVE for the integrated arms (gradient into the
   encoder = the integration under test), DETACHED for the isolated arm
   (gradient stop). Buffers are also populated for the R3 readiness readout.
P1 frozen-head finetune: the encoder (LatentStack) is FROZEN; the heads are
   finetuned on DETACHED latents IDENTICALLY across arms, so the eval readout is
   calibrated to each arm's final frozen encoder (no moving-target; the arm
   difference is the P0 encoder co-shaping, not the readout).
P2 eval: everything frozen; a fresh held-out rollout; world_R2 + affordance_R2.

PRE-REGISTERED READINESS GATE (else substrate_not_ready_requeue, NOT a FAIL)
---------------------------------------------------------------------------
Measured on the LOAD-BEARING ARM_INTEGRATED_PAIR (worst case across seeds). The
integrated arm is only interpretable as evidence about MECH-423 once the
integration machinery is demonstrably doing cross-module work (lit-pull
targeted_review_mech_423_integration_prerequisites 2026-06-12):
  R1 COUPLING   -- shared_latent_gradient_probe: min per-module grad norm on the
                   shared latent > MIN_GRAD_FLOOR (else uncoupled => integrated
                   == isolated by construction) AND mean pairwise gradient cosine
                   >= 0 (a NET-NEGATIVE cosine is the negative-transfer regime
                   where sub-additivity is the EXPECTED consequence of gradient
                   conflict, NOT a refutation -> route substrate_not_ready).
  R2 INFERENCE  -- agent.last_inference_convergence: loop actually iterated
                   (n_iters >= 2) AND settled (final_rel_delta < REL_TOL).
  R3 INTERLEAVE -- CrossModuleConsolidator: interleaved cross_module_replay_share
                   > 0 AND blocked share == 0 AND interleaved touched E2
                   (updates_e2 >= 1; a starved E2 buffer would collapse the share
                   to 0 and masquerade as the blocked control).
Any readiness precondition unmet -> p0_readiness_gate raises -> the run writes a
substrate_not_ready_requeue manifest and does NOT emit a super-additivity verdict
(the V3-EXQ-642/643 lesson: a starved criterion must not masquerade as a
falsification). Each readiness precondition asserts the SAME statistic its
load-bearing criterion routes on, measured on the trained substrate.

PRE-REGISTERED SUPER-ADDITIVITY MARGIN (statistical, not absolute)
-----------------------------------------------------------------
With R1-R3 satisfied:
  additive_baseline[seed] = integration_score(ARM_ISOLATED, seed)
  delta_pair[seed]        = integration_score(ARM_INTEGRATED_PAIR, seed)
                            - additive_baseline[seed]
  margin                  = SUPERADD_SD_MULT * across-seed SD of the baseline
  PASS  if delta_pair[seed] > margin on >= MIN_SEEDS_PASS of the seeds
          (super-additive: integration beats the sum of isolated gains).
  FAIL/weakens if mean delta_pair <= 0 with R1-R3 satisfied and cosine >= 0
          (integration is merely additive or sub-additive).
  inconclusive (positive but below margin) -> outcome FAIL, evidence_direction
          "inconclusive" (additive-only, NOT weakens).
delta_triple is reported alongside as exploratory (proxy object spine), never as
the verdict driver.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_680_mech423_superadditivity_ablation.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_680_mech423_superadditivity_ablation.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.utils.shared_latent_probe import shared_latent_gradient_probe
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_680_mech423_superadditivity_ablation"
QUEUE_ID = "V3-EXQ-680"
CLAIM_IDS: List[str] = ["MECH-423"]
EXPERIMENT_PURPOSE = "evidence"

ARMS = ("ARM_ISOLATED", "ARM_INTEGRATED_PAIR", "ARM_INTEGRATED_TRIPLE")

# ---- design parameters (PRE-REGISTERED) ----
SEEDS = (42, 123, 456)            # >= 2/3 seeds margin in the acceptance check
TRAIN_EPISODES = 40               # P0 integration-phase episodes (the [train] denominator)
STEPS_PER_EPISODE = 80
P1_STEPS = 300                    # P1 frozen-encoder head finetune steps
EVAL_STEPS = 300                  # P2 held-out eval rollout
HEAD_LR = 1e-3
GRID_SIZE = 6
N_HAZARDS = 4
N_RESOURCES = 3

# R2 iterative-inference settling (integrated arms)
SETTLE_ITERS = 10
REL_TOL = 0.05

# R3 cross-module consolidation readiness
CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

# R1 readiness floors
MIN_GRAD_FLOOR = 1e-6
MIN_COSINE = 0.0                  # net-negative cosine = negative-transfer regime

# super-additivity margin (pre-registered, statistical)
SUPERADD_SD_MULT = 2.0
MIN_SEEDS_PASS = 2               # of len(SEEDS)

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Proxy object-binding stream -- a stand-in ATTACHMENT SURFACE for the ARC-080
# object spine (V4, not in V3). NOT ARC-080; the TRIPLE delta is exploratory.
# --------------------------------------------------------------------------- #
class ProxyObjectStream(nn.Module):
    """A trainable third stream: a bottlenecked autoencoder over the shared
    latent. When fed the LIVE shared latent its gradient co-shapes the encoder
    (a third cross-pollinating stream). A placeholder for the ARC-080 object
    spine -- it does NOT implement type/token/anchor binding and tags no claim."""

    def __init__(self, combined_dim: int):
        super().__init__()
        bottleneck = max(4, combined_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(combined_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, combined_dim),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        return self.net(combined)


def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, integrated: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # R2 iterative-inference settling: ON only for the integrated arms.
        use_iterative_inference=integrated,
        inference_settle_iters=SETTLE_ITERS if integrated else 1,
        inference_convergence_rel_tol=REL_TOL,
        # R3 consolidator built on every arm (standalone readiness readout);
        # the SCHEDULE differs by arm at consolidate() call time.
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved" if integrated else "blocked",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
    )
    return REEAgent(cfg)


def _set_requires_grad(modules, flag: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Variance-explained R2 of pred vs target over a [N, D] batch."""
    sse = float(((pred - target) ** 2).sum().item())
    var = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum().item())
    if var <= EPS:
        return 0.0
    return 1.0 - sse / var


def _frozen_rollout(agent, env, device, rng, n_steps: int):
    """Roll the env for n_steps with the encoder frozen; return detached
    (z_self [N, self_dim], z_world [N, world_dim]). Used for the P1 head
    finetune set and the P2 held-out eval set, both on the SAME final frozen
    encoder so the readout is calibrated to what eval measures."""
    self_list: List[torch.Tensor] = []
    world_list: List[torch.Tensor] = []
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    with torch.no_grad():
        for _step in range(n_steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            self_list.append(latent.z_self.detach().clone())
            world_list.append(latent.z_world.detach().clone())
            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            _, harm_signal, done, info, obs_dict = env.step(action)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()
    return torch.cat(self_list, dim=0), torch.cat(world_list, dim=0)


def run_cell(arm: str, seed: int, train_eps: int, steps: int,
             p1_steps: int, eval_steps: int) -> Dict:
    """Train + eval one (arm, seed) cell; return its row (incl. readiness on
    the integrated arms)."""
    print(f"Seed {seed} Condition {arm}", flush=True)
    integrated = arm in ("ARM_INTEGRATED_PAIR", "ARM_INTEGRATED_TRIPLE")
    triple = arm == "ARM_INTEGRATED_TRIPLE"

    env = _make_env(seed)
    agent = _make_agent(env, integrated)
    device = agent.device

    self_dim = agent.config.latent.self_dim
    world_dim = agent.config.latent.world_dim
    combined_dim = self_dim + world_dim

    # Two task heads (+ optional proxy object stream).
    world_head = nn.Linear(self_dim, world_dim).to(device)   # z_self -> z_world
    self_head = nn.Linear(world_dim, self_dim).to(device)    # z_world -> z_self
    proxy_obj = ProxyObjectStream(combined_dim).to(device) if triple else None

    head_params = list(world_head.parameters()) + list(self_head.parameters())
    if proxy_obj is not None:
        head_params += list(proxy_obj.parameters())
    opt = torch.optim.Adam(head_params, lr=HEAD_LR)

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    rng = torch.Generator(device="cpu").manual_seed(seed)

    last_combined_live: Optional[torch.Tensor] = None

    # ---------------- P0: integration phase ----------------
    for ep in range(train_eps):
        if ep % 8 == 0 or ep == train_eps - 1:
            print(f"  [train] {arm} seed={seed} ep {ep+1}/{train_eps}", flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        for _step in range(steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)

            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            if prev_z_self is not None:
                agent.record_transition(prev_z_self, action, latent.z_self.detach())

            # Shared latent. LIVE for integrated (grad flows into the encoder =
            # the integration under test); DETACHED for isolated (gradient stop).
            combined_full = torch.cat([latent.z_self, latent.z_world], dim=-1)
            last_combined_live = combined_full

            z_self_in = latent.z_self if integrated else latent.z_self.detach()
            z_world_in = latent.z_world if integrated else latent.z_world.detach()
            comb_in = combined_full if integrated else combined_full.detach()

            # Cross-stream task losses (targets always detached).
            loss = (
                F.mse_loss(world_head(z_self_in), latent.z_world.detach())
                + F.mse_loss(self_head(z_world_in), latent.z_self.detach())
            )
            if proxy_obj is not None:
                loss = loss + F.mse_loss(proxy_obj(comb_in), comb_in.detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            _, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()

    n_world_buffer = len(agent._world_experience_buffer)
    n_e2_buffer = len(agent._e2_transition_buffer)

    # ---------------- P1: frozen-encoder head finetune (identical all arms) ----
    # Encoder frozen; the heads finetune on a frozen-encoder rollout so the
    # readout is calibrated to the SAME encoder distribution P2 evaluates on.
    _set_requires_grad([agent.e1, agent.e2, agent.latent_stack], False)
    ft_self, ft_world = _frozen_rollout(agent, env, device, rng, eval_steps)
    n_ft = ft_self.shape[0]
    if n_ft >= CMC_BATCH:
        for _ps in range(p1_steps):
            idx = torch.randint(0, n_ft, (CMC_BATCH,), generator=rng)
            b_self = ft_self[idx]
            b_world = ft_world[idx]
            loss = (
                F.mse_loss(world_head(b_self), b_world)
                + F.mse_loss(self_head(b_world), b_self)
            )
            if proxy_obj is not None:
                loss = loss + F.mse_loss(
                    proxy_obj(torch.cat([b_self, b_world], dim=-1)),
                    torch.cat([b_self, b_world], dim=-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---------------- P2: held-out eval (everything frozen) ----------------
    world_head.eval()
    self_head.eval()
    ez_self, ez_world = _frozen_rollout(agent, env, device, rng, eval_steps)
    with torch.no_grad():
        world_r2 = _r2(world_head(ez_self), ez_world)
        affordance_r2 = _r2(self_head(ez_world), ez_self)
    integration_score = world_r2 + affordance_r2

    # ---------------- readiness readouts (integrated arms only) ----------------
    r1 = r2 = r3 = None
    if integrated:
        world_head.train()
        self_head.train()
        # R1: shared-latent gradient coupling on the two task heads.
        if last_combined_live is not None:
            z_shared = last_combined_live.detach().clone().requires_grad_(True)
            probe = shared_latent_gradient_probe(
                z_shared,
                {
                    "world": lambda z: F.mse_loss(
                        world_head(z[:, :self_dim]), z[:, self_dim:].detach()),
                    "self": lambda z: F.mse_loss(
                        self_head(z[:, self_dim:]), z[:, :self_dim].detach()),
                },
            )
            r1 = {
                "coupled": bool(probe["coupled"]),
                "min_grad_norm": float(probe["min_grad_norm"]),
                "mean_pairwise_cosine": float(probe["mean_pairwise_cosine"]),
                "n_modules": int(probe["n_modules"]),
            }
        else:
            r1 = {"coupled": False, "min_grad_norm": 0.0,
                  "mean_pairwise_cosine": -1.0, "n_modules": 0}

        # R2: iterative-inference convergence from one sense() on the substrate.
        _, r2_obs = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        obs_body = _to_batched(r2_obs["body_state"], device)
        obs_world = _to_batched(r2_obs["world_state"], device)
        obs_harm = r2_obs.get("harm_obs", None)
        if obs_harm is not None:
            obs_harm = _to_batched(obs_harm, device)
        _ = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        ic = agent.last_inference_convergence
        if ic is None:
            r2 = {"present": False, "converged": False,
                  "final_rel_delta": float("nan"), "n_iters": 0}
        else:
            r2 = {
                "present": True,
                "converged": bool(ic.get("converged", False)),
                "final_rel_delta": float(ic.get("final_rel_delta", float("nan"))),
                "n_iters": int(ic.get("n_iters", 0)),
            }

        # R3: interleaved vs blocked cross-module consolidation share.
        _set_requires_grad([agent.e1, agent.e2], True)
        cmc = agent.cross_module_consolidator
        module_losses = {
            "e1": lambda: agent.compute_prediction_loss(),
            "e2": lambda: agent.compute_e2_loss(batch_size=CMC_BATCH),
        }
        module_params = {
            "e1": list(agent.e1.parameters()),
            "e2": list(agent.e2.parameters()),
        }
        inter = cmc.consolidate(
            module_losses=module_losses, module_params=module_params,
            n_steps=CMC_STEPS, schedule="interleaved", lr=CMC_LR, simulation_mode=False)
        blocked = cmc.consolidate(
            module_losses=module_losses, module_params=module_params,
            n_steps=CMC_STEPS, schedule="blocked", lr=CMC_LR, simulation_mode=False)
        r3 = {
            "interleaved_share": float(inter["cross_module_replay_share"]),
            "blocked_share": float(blocked["cross_module_replay_share"]),
            "interleaved_n_updates": float(inter["n_updates"]),
            "updates_e2_interleaved": float(inter.get("updates_e2", 0.0)),
            "interleaved_flag": float(inter["interleaved"]),
        }

    print(
        f"  {arm} seed={seed} world_R2={world_r2:.4f} aff_R2={affordance_r2:.4f} "
        f"score={integration_score:.4f}"
        + (f" | R1 cos={r1['mean_pairwise_cosine']:.3f} min_grad={r1['min_grad_norm']:.3g}"
           f" | R2 conv={r2['converged']} n_iters={r2['n_iters']}"
           f" | R3 inter={r3['interleaved_share']:.2f} blk={r3['blocked_share']:.2f}"
           if integrated else ""),
        flush=True,
    )
    # Per-cell progress verdict (cell completed with a finite score).
    cell_ok = bool(torch.isfinite(torch.tensor(integration_score)).item())
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "world_r2": world_r2,
        "affordance_r2": affordance_r2,
        "integration_score": integration_score,
        "n_world_buffer": n_world_buffer,
        "n_e2_buffer": n_e2_buffer,
        "R1": r1,
        "R2": r2,
        "R3": r3,
    }


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    train_eps = 3 if dry_run else TRAIN_EPISODES
    steps = 15 if dry_run else STEPS_PER_EPISODE
    p1_steps = 20 if dry_run else P1_STEPS
    eval_steps = 20 if dry_run else EVAL_STEPS
    seeds = (SEEDS[0],) if dry_run else SEEDS

    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) "
          f"arms={ARMS} seeds={seeds} train_eps={train_eps}", flush=True)
    t0 = time.time()

    # cells keyed by (arm, seed)
    rows: Dict[tuple, Dict] = {}
    arm_results: List[Dict] = []
    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "train_episodes": train_eps,
                "steps_per_episode": steps,
                "p1_steps": p1_steps,
                "eval_steps": eval_steps,
                "settle_iters": SETTLE_ITERS,
                "rel_tol": REL_TOL,
                "cmc_steps": CMC_STEPS,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=["fresh_evidence_run_no_reuse"]) as cell:
                row = run_cell(arm, seed, train_eps, steps, p1_steps, eval_steps)
                cell.stamp(row)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    # ---------------- readiness gate (load-bearing ARM_INTEGRATED_PAIR) -------
    pair_rows = [rows[("ARM_INTEGRATED_PAIR", s)] for s in seeds]
    min_min_grad = min(r["R1"]["min_grad_norm"] for r in pair_rows)
    min_cos = min(r["R1"]["mean_pairwise_cosine"] for r in pair_rows)
    min_n_iters = min(r["R2"]["n_iters"] for r in pair_rows)
    min_updates_e2 = min(r["R3"]["updates_e2_interleaved"] for r in pair_rows)
    inter_share_ok = all(r["R3"]["interleaved_share"] > 0.0 for r in pair_rows)
    blocked_share_zero = all(r["R3"]["blocked_share"] == 0.0 for r in pair_rows)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    readiness_checks = [
        {"name": "r1_shared_latent_grad_coupled", "measured": float(min_min_grad),
         "threshold": float(MIN_GRAD_FLOOR), "direction": "lower"},
        {"name": "r1_grad_cosine_not_net_negative", "measured": float(min_cos),
         "threshold": float(MIN_COSINE), "direction": "lower"},
        {"name": "r2_iterative_loop_iterated", "measured": float(min_n_iters),
         "threshold": 2.0, "direction": "lower"},
        {"name": "r3_e2_touchable_under_interleaved", "measured": float(min_updates_e2),
         "threshold": 1.0, "direction": "lower"},
        {"name": "r3_interleaved_share_positive", "measured": float(1.0 if inter_share_ok else 0.0),
         "threshold": 1.0, "direction": "lower"},
        {"name": "r3_blocked_share_zero", "measured": float(1.0 if blocked_share_zero else 0.0),
         "threshold": 1.0, "direction": "lower"},
    ]

    def _write_manifest(manifest: Dict) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        # Audit link to the MANUAL proposal EXP-0380 (manual proposals carry no
        # EVB backlog_id; the evidence link is claim_ids). NOTE: EVB-0380 is a
        # DIFFERENT proposal (SD-027) -- do not conflate.
        "proposal_id": "EXP-0380",
        "sleep_driver_pattern": "N/A (CrossModuleConsolidator.consolidate() called directly; no sleep cycle)",
        "thresholds": {
            "min_grad_floor": MIN_GRAD_FLOOR,
            "min_cosine": MIN_COSINE,
            "rel_tol": REL_TOL,
            "settle_iters": SETTLE_ITERS,
            "superadd_sd_mult": SUPERADD_SD_MULT,
            "min_seeds_pass": MIN_SEEDS_PASS,
        },
        "config": {
            "arms": list(ARMS),
            "seeds": list(seeds),
            "train_episodes": train_eps,
            "steps_per_episode": steps,
            "p1_steps": p1_steps,
            "eval_steps": eval_steps,
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
        },
        "arm_results": arm_results,
        "proxy_object_spine_note": (
            "ARM_INTEGRATED_TRIPLE uses a PROXY object-binding head (ProxyObjectStream), "
            "NOT the ARC-080 object spine (candidate/v3_pending/v4; the only V3 object "
            "module entities/object_file_buffer.py is non-trainable). The TRIPLE delta is "
            "EXPLORATORY only and is NOT evidence about ARC-080; this run tags MECH-423 only."),
    }

    # P0 readiness abort gate.
    try:
        preconditions = p0_readiness_gate(readiness_checks)
    except P0NotReady as e:
        print(f"\n[{EXPERIMENT_TYPE}] readiness UNMET -> substrate_not_ready_requeue: "
              f"{e.reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Readiness precondition(s) unmet on the trained ARM_INTEGRATED_PAIR "
                "substrate -- the integration machinery is not demonstrably doing "
                "cross-module work, so the integrated arm equals the isolated arm by "
                "construction and a super-additivity verdict would be vacuous. Re-queue "
                "at a longer training budget / fix wiring. " + e.reason),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + e.reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write_manifest(manifest)

    # ---------------- super-additivity verdict ----------------
    iso_scores = [rows[("ARM_ISOLATED", s)]["integration_score"] for s in seeds]
    pair_scores = [rows[("ARM_INTEGRATED_PAIR", s)]["integration_score"] for s in seeds]
    triple_scores = [rows[("ARM_INTEGRATED_TRIPLE", s)]["integration_score"] for s in seeds]

    delta_pair = [p - i for p, i in zip(pair_scores, iso_scores)]
    delta_triple = [t - i for t, i in zip(triple_scores, iso_scores)]
    baseline_sd = float(statistics.pstdev(iso_scores)) if len(iso_scores) > 1 else 0.0
    margin = SUPERADD_SD_MULT * baseline_sd
    n_seeds_pass = sum(1 for d in delta_pair if d > margin)
    mean_delta_pair = float(sum(delta_pair) / len(delta_pair))

    super_additive = n_seeds_pass >= MIN_SEEDS_PASS

    # Non-degeneracy net: the integration effect is the WITHIN-SEED isolated-vs-
    # pair difference; the run is degenerate only if EVERY seed has iso == pair
    # (the integration had no measurable effect anywhere = vacuous criterion).
    # NOT keyed on across-seed delta spread: a consistent delta across seeds
    # (low spread) is a STRONG super-additive signal, not degeneracy.
    degen = check_degeneracy({
        "integration_score_iso_vs_pair": {
            "groups": [[iso_scores[k], pair_scores[k]] for k in range(len(seeds))]},
    })

    if super_additive:
        outcome = "PASS"
        evidence_direction = "supports"
        label = "super_additive"
        note = (f"Integration is super-additive: delta_pair > margin ({margin:.4g}) on "
                f"{n_seeds_pass}/{len(seeds)} seeds (mean delta {mean_delta_pair:.4g}).")
    elif mean_delta_pair <= 0.0:
        outcome = "FAIL"
        evidence_direction = "weakens"
        label = "sub_or_merely_additive"
        note = (f"Integrated arm at or below the additive baseline (mean delta "
                f"{mean_delta_pair:.4g} <= 0) with readiness satisfied and gradient "
                f"cosine >= 0: integration is merely additive / sub-additive.")
    else:
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        label = "additive_below_margin"
        note = (f"Positive but below the pre-registered margin: mean delta "
                f"{mean_delta_pair:.4g} > 0 but only {n_seeds_pass}/{len(seeds)} seeds "
                f"clear margin {margin:.4g}. Additive-only (NOT weakens).")

    print(f"\n[{EXPERIMENT_TYPE}] aggregate:")
    print(f"  iso_scores={[round(x,4) for x in iso_scores]} sd={baseline_sd:.4g}")
    print(f"  pair_scores={[round(x,4) for x in pair_scores]} delta={[round(x,4) for x in delta_pair]}")
    print(f"  triple_scores={[round(x,4) for x in triple_scores]} delta={[round(x,4) for x in delta_triple]} (exploratory)")
    print(f"  margin={margin:.4g} n_seeds_pass={n_seeds_pass}/{len(seeds)} -> {label} ({outcome})")
    print(f"  non_degenerate={degen['non_degenerate']} elapsed={elapsed:.1f}s")

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": note,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": [
                {"name": "superadditivity_margin_pair", "load_bearing": True,
                 "passed": bool(super_additive)},
            ],
        },
        "superadditivity": {
            "iso_scores": iso_scores,
            "pair_scores": pair_scores,
            "triple_scores_exploratory": triple_scores,
            "delta_pair": delta_pair,
            "delta_triple_exploratory": delta_triple,
            "baseline_across_seed_sd": baseline_sd,
            "margin": margin,
            "n_seeds_pass": n_seeds_pass,
            "mean_delta_pair": mean_delta_pair,
            "super_additive": super_additive,
        },
        "elapsed_seconds": elapsed,
    })
    manifest.update(degen)
    return outcome, _write_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _outcome_clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_outcome_clean, manifest_path=_manifest_path)
    sys.exit(0)
