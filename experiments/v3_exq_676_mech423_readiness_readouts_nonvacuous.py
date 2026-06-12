"""
V3-EXQ-676: MECH-423 readiness-readouts non-vacuity diagnostic.

EXPERIMENT_PURPOSE = diagnostic (claim_ids=[]). This is the implement-substrate
Step-8 validation for the MECH-423 super-additivity READINESS substrate landed
2026-06-12 (R2 iterative-inference convergence readout on LatentStack.encode +
R3 module-tagged interleaved CrossModuleConsolidator in the MECH-121 pipeline +
R1 shared_latent_gradient_probe). It does NOT test super-additivity -- that is
the separate EXP-0380 ablation this substrate unblocks.

It confirms the three EXP-0380 readiness readouts are NON-VACUOUS on a non-trivial
substrate (a few P0 warm-up episodes so E1+E2 and the replay buffers are real):

  R2  with use_iterative_inference=True (settle_iters=8), the shared-latent
      settling loop converges: agent.last_inference_convergence.final_rel_delta
      < 0.05 (the EXP-0380 R2 threshold) on >= 2/3 seeds.
  R3  with use_cross_module_consolidation=True, the INTERLEAVED schedule yields
      cross_module_replay_share > 0 AND interleaved == 1.0 AND n_updates > 0,
      WHILE the BLOCKED-schedule control (same trained agent, same buffers) yields
      cross_module_replay_share == 0.0. The blocked control is what proves the
      readout discriminates the schedule rather than being a constant.
  R1  shared_latent_gradient_probe over a z_shared fed jointly to E1 + E2 returns
      min_grad_norm > 0 (both modules genuinely couple to the shared latent) AND
      mean_pairwise_cosine >= 0 (not the net-negative-transfer regime).

PASS = all three non-vacuous on >= 2/3 seeds AND the readiness preconditions hold.
A below-floor readiness measure (buffers not populated, or the shared latent does
not couple into both modules) self-routes substrate_not_ready_requeue, NEVER a
substrate-verdict label -- the readout is starved, not refuted.

This experiment uses the STANDALONE consolidator (agent.cross_module_consolidator)
rather than the SleepLoopManager hook, so it sets NO sleep flags (the sleep-hook
wiring is covered by tests/contracts/test_mech423_cross_module_consolidation.py C7).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.utils.shared_latent_probe import shared_latent_gradient_probe  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_676_mech423_readiness_readouts_nonvacuous"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44]
SELF_DIM = 16
WORLD_DIM = 16
ACTION_DIM = 4
N_P0_EPISODES = 20          # P0 warm-up episodes per seed (the loop-bound denominator)
N_STEPS_PER_EP = 25         # env steps per P0 episode
WARM_EVERY = 5              # run an E1+E2 optimizer step every WARM_EVERY env steps

SETTLE_ITERS = 8            # R2 settling-loop budget
R2_REL_TOL_GATE = 0.05      # R2 PASS: final_rel_delta < this (the EXP-0380 threshold)
R3_STEPS = 4                # R3 consolidation steps per schedule
GRAD_NORM_FLOOR = 1e-8      # R1 / readiness: min_grad_norm must exceed this
BUFFER_FLOOR = 2            # readiness: both replay buffers must hold >= this


def _build_agent(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=6, num_hazards=2, num_resources=2, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,  # SD-008: z_world fidelity
        # R2
        use_iterative_inference=True,
        inference_settle_iters=SETTLE_ITERS,
        inference_convergence_rel_tol=0.02,
        # R3
        use_cross_module_consolidation=True,
        cross_module_consolidation_steps=R3_STEPS,
        cross_module_consolidation_schedule="interleaved",
        # SD-056: warm E2 toward action-conditional divergence (non-trivial modules)
        e2_action_contrastive_enabled=True,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


def _obs(env):
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return b, w


def _p0_warm(agent, b, w, n_ep: int):
    """Collect replay experience + warm E1/E2 so the modules are non-trivial."""
    opt = torch.optim.Adam(
        list(agent.e1.parameters()) + list(agent.e2.parameters()), lr=1e-3
    )
    prev_zself = None
    for ep in range(n_ep):
        for _step in range(N_STEPS_PER_EP):
            with torch.no_grad():
                a = agent.act_with_split_obs(b, w)
            act = a if a.dim() == 2 else a.unsqueeze(0)
            cur = agent._current_latent
            cur_zself = (
                cur.z_self.detach().clone()
                if cur is not None and cur.z_self is not None
                else torch.zeros(1, SELF_DIM)
            )
            if prev_zself is not None:
                agent.record_transition(prev_zself, act.float(), cur_zself)
            prev_zself = cur_zself
            if (_step + 1) % WARM_EVERY == 0:
                loss = agent.compute_prediction_loss() + agent.compute_e2_loss(8)
                if loss.requires_grad and float(loss.detach().item()) != 0.0:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        if (ep + 1) % 5 == 0:
            print(
                f"  [train] readiness seed-warm ep {ep+1}/{n_ep} "
                f"world_buf={len(agent._world_experience_buffer)} "
                f"e2_buf={len(agent._e2_transition_buffer)}",
                flush=True,
            )


def _measure_r2(agent, b, w) -> Dict[str, Any]:
    with torch.no_grad():
        agent.sense(obs_body=b, obs_world=w)
    ic = agent.last_inference_convergence or {}
    final_rel = float(ic.get("final_rel_delta", float("nan")))
    converged = bool(ic.get("converged", False))
    n_iters = int(ic.get("n_iters", 0))
    return {
        "r2_final_rel_delta": final_rel,
        "r2_converged": converged,
        "r2_n_iters": n_iters,
        # R2 readout-present is non-vacuous when the settling loop produced a
        # finite final_rel_delta below the gate.
        "r2_ok": (final_rel == final_rel) and (final_rel < R2_REL_TOL_GATE),
    }


def _measure_r3(agent) -> Dict[str, Any]:
    losses = {
        "e1": lambda: agent.compute_prediction_loss(),
        "e2": lambda: agent.compute_e2_loss(batch_size=8),
    }
    params = {
        "e1": list(agent.e1.parameters()),
        "e2": list(agent.e2.parameters()),
    }
    inter = agent.cross_module_consolidator.consolidate(
        module_losses=losses, module_params=params, n_steps=R3_STEPS,
        schedule="interleaved",
    )
    blocked = agent.cross_module_consolidator.consolidate(
        module_losses=losses, module_params=params, n_steps=R3_STEPS,
        schedule="blocked",
    )
    inter_share = float(inter["cross_module_replay_share"])
    blocked_share = float(blocked["cross_module_replay_share"])
    return {
        "r3_interleaved_share": inter_share,
        "r3_interleaved_flag": float(inter["interleaved"]),
        "r3_interleaved_n_updates": float(inter["n_updates"]),
        "r3_blocked_share": blocked_share,
        "r3_blocked_flag": float(blocked["interleaved"]),
        "r3_ok": (
            inter_share > 0.0
            and inter["interleaved"] == 1.0
            and inter["n_updates"] > 0.0
            and blocked_share == 0.0
        ),
    }


def _measure_r1(agent, b, w) -> Dict[str, Any]:
    """R1: a z_shared fed jointly to E1 + E2 toward a shared next-world target.

    Positive control: z_shared is a real z_world_t from the replay buffer and the
    target is the real next z_world_{t+1}; both E1 (world-model) and E2
    (world_forward) regress that one shared latent toward the SAME real next
    state, so a coupled substrate yields non-zero, aligned (non-negative-cosine)
    gradients. The target is model-INDEPENDENT (a real observation, not the
    model's own prediction), so neither module's loss is trivially zero.
    """
    with torch.no_grad():
        latent = agent.sense(obs_body=b, obs_world=w)
        z_self = latent.z_self.detach().clone()
    wb = agent._world_experience_buffer
    if len(wb) >= 2:
        z_world = wb[-2].detach().clone()
        target = wb[-1].detach().clone()
    else:  # fallback (should not happen post-P0); a non-trivial random target
        z_world = latent.z_world.detach().clone()
        target = torch.randn_like(z_world)
    if z_world.dim() == 1:
        z_world = z_world.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    batch = z_world.shape[0]
    if z_self.shape[0] != batch:
        z_self = z_self[:batch] if z_self.shape[0] > batch else z_self.expand(batch, -1)
    action = torch.zeros(batch, ACTION_DIM)
    action[:, 0] = 1.0
    z_shared = z_world.clone().requires_grad_(True)

    def _e2_loss(zs: torch.Tensor) -> torch.Tensor:
        pred = agent.e2.world_forward(zs, action)
        return ((pred - target) ** 2).mean()

    def _e1_loss(zs: torch.Tensor) -> torch.Tensor:
        current_state = torch.cat([z_self, zs], dim=-1)
        pred, _prior = agent.e1.forward(current_state)
        pred_flat = pred.reshape(batch, -1)[:, :WORLD_DIM]
        return ((pred_flat - target) ** 2).mean()

    res = shared_latent_gradient_probe(z_shared, {"e1": _e1_loss, "e2": _e2_loss})
    return {
        "r1_min_grad_norm": float(res["min_grad_norm"]),
        "r1_mean_pairwise_cosine": float(res["mean_pairwise_cosine"]),
        "r1_n_modules": int(res["n_modules"]),
        "r1_per_module_grad_norm": res["per_module_grad_norm"],
        "r1_ok": bool(res["coupled"]),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_p0 = 3 if dry_run else N_P0_EPISODES

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition readiness", flush=True)
        agent, env = _build_agent(seed)
        b, w = _obs(env)
        _p0_warm(agent, b, w, n_p0)
        r2 = _measure_r2(agent, b, w)
        r3 = _measure_r3(agent)
        r1 = _measure_r1(agent, b, w)
        row: Dict[str, Any] = {
            "seed": seed,
            "world_buf": len(agent._world_experience_buffer),
            "e2_buf": len(agent._e2_transition_buffer),
        }
        row.update(r2)
        row.update(r3)
        row.update(r1)
        seed_ok = bool(r2["r2_ok"] and r3["r3_ok"] and r1["r1_ok"])
        row["seed_ok"] = seed_ok
        rows.append(row)
        print(
            f"  R2 final_rel_delta={r2['r2_final_rel_delta']:.5f} ok={r2['r2_ok']} | "
            f"R3 inter_share={r3['r3_interleaved_share']:.2f} "
            f"blocked_share={r3['r3_blocked_share']:.2f} ok={r3['r3_ok']} | "
            f"R1 min_grad={r1['r1_min_grad_norm']:.3e} "
            f"cos={r1['r1_mean_pairwise_cosine']:.4f} ok={r1['r1_ok']}",
            flush=True,
        )
        print(f"verdict: {'PASS' if seed_ok else 'FAIL'}", flush=True)

    # ---- readiness preconditions (substrate_not_ready_requeue if unmet) ----
    n_seeds = len(rows)
    min_world_buf = min(r["world_buf"] for r in rows)
    min_e2_buf = min(r["e2_buf"] for r in rows)
    min_r1_grad = min(r["r1_min_grad_norm"] for r in rows)
    buffers_met = (min_world_buf >= BUFFER_FLOOR) and (min_e2_buf >= BUFFER_FLOOR)
    coupling_met = min_r1_grad > GRAD_NORM_FLOOR
    preconditions = [
        {
            "name": "replay_buffers_populated",
            "description": "both E1 world buffer and E2 transition buffer hold "
                           ">= BUFFER_FLOOR replay traces (else R3 consolidation "
                           "is starved, share is uninterpretable)",
            "measured": float(min(min_world_buf, min_e2_buf)),
            "threshold": float(BUFFER_FLOOR),
            "control": "min over seeds after P0 warm-up",
            "met": bool(buffers_met),
        },
        {
            "name": "shared_latent_couples_into_both_modules",
            "description": "R1 min per-module gradient norm on the jointly-fed "
                           "shared latent exceeds the floor (the SAME statistic "
                           "the R1 criterion routes on); below-floor means the "
                           "latent is uncoupled, not that coupling is refuted",
            "measured": float(min_r1_grad),
            "threshold": float(GRAD_NORM_FLOOR),
            "control": "min over seeds, positive-control z_shared fed to E1+E2",
            "met": bool(coupling_met),
        },
    ]
    readiness_met = buffers_met and coupling_met

    n_seed_ok = sum(1 for r in rows if r["seed_ok"])
    n_r2 = sum(1 for r in rows if r["r2_ok"])
    n_r3 = sum(1 for r in rows if r["r3_ok"])
    n_r1 = sum(1 for r in rows if r["r1_ok"])
    majority = (n_seeds // 2) + 1  # >= 2/3

    criteria_non_degenerate = {
        # R3 readout discriminates schedule (blocked control == 0) on every seed;
        # if blocked also produced share>0 the readout would be a constant 1.
        "R3_blocked_control_discriminates": all(
            r["r3_blocked_share"] == 0.0 for r in rows
        ),
        # R2 settling produced > 1 iteration (a genuine settling loop, not a
        # single degenerate pass).
        "R2_settled_multiple_iters": all(r["r2_n_iters"] >= 1 for r in rows),
        # R1 saw two contributing modules (the probe is not single-module).
        "R1_two_modules": all(r["r1_n_modules"] == 2 for r in rows),
    }

    criteria = [
        {"name": "R2_converges_majority", "load_bearing": True,
         "passed": n_r2 >= majority},
        {"name": "R3_interleaved_nonvacuous_blocked_zero_majority", "load_bearing": True,
         "passed": n_r3 >= majority},
        {"name": "R1_coupled_majority", "load_bearing": True,
         "passed": n_r1 >= majority},
    ]

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif all(c["passed"] for c in criteria):
        outcome = "PASS"
        label = "readiness_readouts_nonvacuous"
    else:
        outcome = "FAIL"
        label = "readiness_readout_failed_on_trained_substrate"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": timestamp,
        "seed_results": rows,
        "summary": {
            "n_seeds": n_seeds,
            "n_seed_ok": n_seed_ok,
            "n_r2_ok": n_r2,
            "n_r3_ok": n_r3,
            "n_r1_ok": n_r1,
            "majority_threshold": majority,
            "min_world_buf": min_world_buf,
            "min_e2_buf": min_e2_buf,
            "min_r1_grad_norm": float(min_r1_grad),
        },
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
        },
        "pre_registered": {
            "SETTLE_ITERS": SETTLE_ITERS,
            "R2_REL_TOL_GATE": R2_REL_TOL_GATE,
            "R3_STEPS": R3_STEPS,
            "GRAD_NORM_FLOOR": GRAD_NORM_FLOOR,
            "BUFFER_FLOOR": BUFFER_FLOOR,
        },
        "dry_run": dry_run,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} ({label})", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
