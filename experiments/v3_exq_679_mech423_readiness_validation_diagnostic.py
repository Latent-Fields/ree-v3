#!/opt/local/bin/python3
"""
V3-EXQ-679 -- MECH-423 super-additivity readiness substrate VALIDATION diagnostic.
SLEEP DRIVER: N/A (R3 calls CrossModuleConsolidator.consolidate() directly; no
              SleepLoopManager / sleep cycle is driven).

PURPOSE
-------
EXPERIMENT_PURPOSE = "diagnostic". claim_ids = [] -- this validates the MECH-423
readiness INSTRUMENTATION (R1/R2/R3), it does NOT test the MECH-423 claim.

The MECH-423 super-additivity readiness substrate (R1 shared-latent gradient
probe, R2 iterative-inference convergence, R3 module-tagged interleaved
cross-module consolidation) landed 2026-06-12, no-op-default + contract-tested.
The EXP-0380 3-arm super-additivity ablation reads all three readouts and
self-routes substrate_not_ready_requeue whenever any of R1/R2/R3 is vacuous on a
trained substrate. Before that bespoke ablation is worth building, this
diagnostic confirms the three readouts are NON-VACUOUS on a briefly-but-genuinely
trained substrate -- i.e. that EXP-0380 would actually get a meaningful reading
rather than self-routing substrate_not_ready every run.

WHAT EACH READOUT NEEDS TO BE NON-VACUOUS (and how this script supplies it)
--------------------------------------------------------------------------
R1 (shared-latent gradient coupling; ree_core/utils/shared_latent_probe.py):
    A shared latent fed to >= 2 modules must carry a non-zero gradient to EACH
    (min_grad_norm > 0) and the per-module gradients must not be net-conflicting
    (mean_pairwise_cosine >= 0). Supplied here by taking the trained agent's
    combined [z_self, z_world] latent as the shared latent and feeding it to two
    trained reconstruction heads (one -> z_world, one -> z_self). Related tasks
    over a shared substrate-derived latent => coupled verdict expected; a
    degenerate (~0) shared-latent gradient self-routes substrate_not_ready.
    NOTE: the full E1<->E2 integrated-arm construction of z_shared is EXP-0380's
    job; R1 here validates the PROBE instrument on a substrate-derived latent.

R2 (iterative-inference convergence; ree_core/latent/stack.py encode loop):
    With use_iterative_inference=True the encode() top-down round becomes a
    predictive-coding settling loop that exposes per-round ||delta z_shared||.
    Non-vacuous => the loop actually iterates (n_iters >= 2) AND settles
    (final_rel_delta < rel_tol). Read off agent.last_inference_convergence after
    a sense() on the trained substrate.

R3 (module-tagged interleaved cross-module consolidation;
    ree_core/sleep/cross_module_consolidation.py):
    The interleaved schedule must produce cross-module replay traces
    (cross_module_replay_share > 0) while the blocked control produces none
    (share == 0). This requires BOTH the E1 world buffer AND the E2 transition
    buffer to be populated so both module loss closures return non-zero loss
    (an exactly-zero loss == "no replay content" => module not touched). This
    script populates E1's _world_experience_buffer (via _e1_tick) and E2's
    _e2_transition_buffer (via record_transition) every warmup step, and
    re-enables requires_grad on e1/e2 before the consolidate pass so the locally
    constructed per-module optimisers find parameters.

PRE-REGISTERED READINESS CRITERIA (per seed; PASS = all seeds pass all three)
----------------------------------------------------------------------------
R1: coupled == True (min_grad_norm > MIN_GRAD_FLOOR AND mean_pairwise_cosine >= 0)
    AND n_modules == 2.                                          [load-bearing]
R2: inference_convergence present AND converged == True AND
    final_rel_delta < REL_TOL AND n_iters >= 2.                  [load-bearing]
R3: interleaved cross_module_replay_share > 0 AND blocked share == 0 AND
    interleaved n_updates > 0 AND both updates_e1/updates_e2 > 0. [load-bearing]

ADJUDICATION (diagnostic self-route is a hypothesis, not a verdict)
-------------------------------------------------------------------
Readiness-kind preconditions assert the SAME statistic each load-bearing
criterion routes on, measured on a known-positive control (the trained
substrate), so a STARVED criterion self-routes substrate_not_ready_requeue
instead of masquerading as a falsification (the V3-EXQ-643 lesson):
  * R1 precondition  -> min_grad_norm (the stat R1 routes on) > floor
  * R2 precondition  -> n_iters (loop actually iterated) >= 2
  * R3 precondition  -> updates_e2 under interleaved (the module the share needs
                        touched) >= 1
If any precondition is unmet  -> label substrate_not_ready_requeue (warmup too
short / wiring broken), NOT a readout-vacuous verdict.
If preconditions met but a load-bearing criterion fails (e.g. R2 fired but did
not settle below tol) -> label readiness_not_confirmed, a genuine FAIL: the
readout is non-vacuous but does not meet the EXP-0380 gate threshold on this
substrate; revisit warmup length / settle_iters / rel_tol before building
EXP-0380.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_679_mech423_readiness_validation_diagnostic.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_679_mech423_readiness_validation_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.utils.shared_latent_probe import shared_latent_gradient_probe
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_679_mech423_readiness_validation_diagnostic"
QUEUE_ID = "V3-EXQ-679"
CLAIM_IDS: List[str] = []  # validates the R1/R2/R3 instrumentation, not a claim
EXPERIMENT_PURPOSE = "diagnostic"

# ---- design parameters (pre-registered) ----
SEEDS = (42, 123, 456)
WARMUP_EPISODES = 24
P0_END = 16                 # P0 0..P0_END: encoder warmup; P1 P0_END..end: R1 heads
STEPS_PER_EPISODE = 100
GRID_SIZE = 6
N_HAZARDS = 4
N_RESOURCES = 3

# R2 iterative-inference settling
SETTLE_ITERS = 10
REL_TOL = 0.05

# R3 cross-module consolidation
CMC_STEPS = 8               # interleaved traces / blocked steps-per-module
CMC_LR = 1e-3
CMC_BATCH = 16

# R1 readiness floor (the smallest acceptable shared-latent gradient norm)
MIN_GRAD_FLOOR = 1e-6


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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # MECH-423 R2: iterative-inference settling loop ON (the substrate under test).
        use_iterative_inference=True,
        inference_settle_iters=SETTLE_ITERS,
        inference_convergence_rel_tol=REL_TOL,
        # MECH-423 R3: build agent.cross_module_consolidator (driven directly here).
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
    )
    return REEAgent(cfg)


def _set_requires_grad(modules, flag: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = flag


def run_seed(seed: int, warmup_eps: int, steps: int, p0_end: int) -> Dict:
    """Warm up a substrate, then measure R1/R2/R3 readouts on it."""
    print(f"Seed {seed} Condition readiness_validation", flush=True)
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device

    self_dim = agent.config.latent.self_dim
    world_dim = agent.config.latent.world_dim
    total_dim = self_dim + world_dim

    # R1 reconstruction heads (consume the shared [z_self, z_world] latent).
    e1_head = torch.nn.Linear(total_dim, world_dim).to(device)
    e2_head = torch.nn.Linear(total_dim, self_dim).to(device)

    # P0 optimiser: encoder warmup (E1 + E2 + LatentStack).
    p0_opt = torch.optim.Adam(
        list(agent.e1.parameters())
        + list(agent.e2.parameters())
        + list(agent.latent_stack.parameters()),
        lr=1e-4,
    )
    # P1 optimiser: frozen encoder, train ONLY the R1 heads on detached latents.
    p1_opt = torch.optim.Adam(
        list(e1_head.parameters()) + list(e2_head.parameters()), lr=1e-3
    )

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    rng = torch.Generator(device="cpu").manual_seed(seed)

    last_combined = None  # combined [z_self, z_world] latent for R1 measurement

    for ep in range(warmup_eps):
        if ep % 8 == 0 or ep == warmup_eps - 1:
            print(f"  [train] readiness seed={seed} ep {ep+1}/{warmup_eps}", flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        phase1 = ep >= p0_end
        if phase1:
            # Freeze encoder for the R1-head training phase (phased training:
            # heads train on detached latents, encoder is fixed).
            _set_requires_grad([agent.e1, agent.e2, agent.latent_stack], False)

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
            # Populate E1 world buffer (append happens inside _e1_tick).
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)

            # Random action (a readiness probe does not need behavioural policy).
            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0

            # Populate E2 transition buffer.
            if prev_z_self is not None:
                agent.record_transition(prev_z_self, action, latent.z_self.detach())

            combined = torch.cat([latent.z_self, latent.z_world], dim=-1)
            last_combined = combined.detach().clone()

            if not phase1:
                # P0: encoder warmup on the live replay losses.
                pred_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss(batch_size=CMC_BATCH)
                loss = pred_loss + e2_loss
                if loss.requires_grad:
                    p0_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e1.parameters())
                        + list(agent.e2.parameters())
                        + list(agent.latent_stack.parameters()),
                        1.0,
                    )
                    p0_opt.step()
            else:
                # P1: train R1 heads to reconstruct z_self / z_world from the
                # DETACHED shared latent (frozen encoder -> no moving target).
                comb_det = combined.detach()
                head_loss = F.mse_loss(
                    e1_head(comb_det), latent.z_world.detach()
                ) + F.mse_loss(e2_head(comb_det), latent.z_self.detach())
                p1_opt.zero_grad()
                head_loss.backward()
                p1_opt.step()

            _, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset()
                agent.reset()
                agent.e1.reset_hidden_state()

    n_world_buffer = len(agent._world_experience_buffer)
    n_e2_buffer = len(agent._e2_transition_buffer)

    # ---------- R2: iterative-inference convergence ----------
    # One more sense() on the trained substrate; read the cached readout.
    obs_body = _to_batched(obs_dict["body_state"], device)
    obs_world = _to_batched(obs_dict["world_state"], device)
    obs_harm = obs_dict.get("harm_obs", None)
    if obs_harm is not None:
        obs_harm = _to_batched(obs_harm, device)
    _ = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
    ic = agent.last_inference_convergence  # plain-float dict or None
    if ic is None:
        r2 = {
            "present": False, "converged": False, "final_rel_delta": float("nan"),
            "n_iters": 0, "per_step_rel_delta": [],
        }
    else:
        r2 = {
            "present": True,
            "converged": bool(ic.get("converged", False)),
            "final_rel_delta": float(ic.get("final_rel_delta", float("nan"))),
            "n_iters": int(ic.get("n_iters", 0)),
            "per_step_rel_delta": [float(x) for x in ic.get("per_step_rel_delta", [])],
        }

    # ---------- R3: interleaved vs blocked cross-module consolidation ----------
    # Re-enable grads so the consolidator's local per-module optimisers find params.
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
        n_steps=CMC_STEPS, schedule="interleaved", lr=CMC_LR, simulation_mode=False,
    )
    blocked = cmc.consolidate(
        module_losses=module_losses, module_params=module_params,
        n_steps=CMC_STEPS, schedule="blocked", lr=CMC_LR, simulation_mode=False,
    )
    r3 = {
        "interleaved_share": float(inter["cross_module_replay_share"]),
        "blocked_share": float(blocked["cross_module_replay_share"]),
        "interleaved_n_updates": float(inter["n_updates"]),
        "interleaved_n_cross_module_traces": float(inter["n_cross_module_traces"]),
        "updates_e1_interleaved": float(inter.get("updates_e1", 0.0)),
        "updates_e2_interleaved": float(inter.get("updates_e2", 0.0)),
        "interleaved_flag": float(inter["interleaved"]),
    }

    # ---------- R1: shared-latent gradient coupling ----------
    if last_combined is None:
        r1 = {"coupled": False, "min_grad_norm": 0.0, "mean_pairwise_cosine": 1.0,
              "n_modules": 0, "per_module_grad_norm": {}}
    else:
        z_shared = last_combined.clone().requires_grad_(True)
        z_self_target = z_shared.detach()[:, :self_dim]
        z_world_target = z_shared.detach()[:, self_dim:]
        probe = shared_latent_gradient_probe(
            z_shared,
            {
                "e1_world": lambda z: F.mse_loss(e1_head(z), z_world_target),
                "e2_self": lambda z: F.mse_loss(e2_head(z), z_self_target),
            },
        )
        r1 = {
            "coupled": bool(probe["coupled"]),
            "min_grad_norm": float(probe["min_grad_norm"]),
            "mean_pairwise_cosine": float(probe["mean_pairwise_cosine"]),
            "n_modules": int(probe["n_modules"]),
            "per_module_grad_norm": probe["per_module_grad_norm"],
        }

    # ---------- per-seed criteria ----------
    r1_pass = bool(r1["coupled"] and r1["n_modules"] == 2 and r1["min_grad_norm"] > MIN_GRAD_FLOOR)
    r2_pass = bool(r2["present"] and r2["converged"] and r2["n_iters"] >= 2
                   and r2["final_rel_delta"] < REL_TOL)
    r3_pass = bool(r3["interleaved_share"] > 0.0 and r3["blocked_share"] == 0.0
                   and r3["interleaved_n_updates"] > 0.0
                   and r3["updates_e1_interleaved"] > 0.0
                   and r3["updates_e2_interleaved"] > 0.0)
    seed_pass = r1_pass and r2_pass and r3_pass

    print(
        f"  seed={seed} R1 coupled={r1['coupled']} min_grad={r1['min_grad_norm']:.4g} "
        f"cos={r1['mean_pairwise_cosine']:.3f} | R2 conv={r2['converged']} "
        f"final={r2['final_rel_delta']:.4g} n_iters={r2['n_iters']} | "
        f"R3 inter={r3['interleaved_share']:.2f} blocked={r3['blocked_share']:.2f} "
        f"upd_e2={r3['updates_e2_interleaved']:.0f}",
        flush=True,
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "n_world_buffer": n_world_buffer,
        "n_e2_buffer": n_e2_buffer,
        "R1": r1, "R2": r2, "R3": r3,
        "r1_pass": r1_pass, "r2_pass": r2_pass, "r3_pass": r3_pass,
        "seed_pass": seed_pass,
    }


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    warmup = 3 if dry_run else WARMUP_EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE
    seeds = (SEEDS[0],) if dry_run else SEEDS
    p0_end = 2 if dry_run else P0_END

    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) "
          f"warmup={warmup} steps={steps} seeds={seeds}", flush=True)
    t0 = time.time()
    per_seed: List[Dict] = []
    for seed in seeds:
        per_seed.append(run_seed(seed, warmup, steps, p0_end))
    elapsed = time.time() - t0

    # ---- aggregate (worst-case across seeds for the readiness preconditions) ----
    min_min_grad = min(s["R1"]["min_grad_norm"] for s in per_seed)
    min_n_iters = min(s["R2"]["n_iters"] for s in per_seed)
    min_updates_e2 = min(s["R3"]["updates_e2_interleaved"] for s in per_seed)
    min_world_buf = min(s["n_world_buffer"] for s in per_seed)
    min_e2_buf = min(s["n_e2_buffer"] for s in per_seed)

    all_r1 = all(s["r1_pass"] for s in per_seed)
    all_r2 = all(s["r2_pass"] for s in per_seed)
    all_r3 = all(s["r3_pass"] for s in per_seed)
    all_seeds_pass = all(s["seed_pass"] for s in per_seed)

    # ---- readiness-kind preconditions (same statistic each criterion routes on) ----
    preconditions = [
        {
            "name": "r1_shared_latent_grad_coupled",
            "description": "trained shared latent must carry a non-zero gradient to each module (the stat R1 routes on)",
            "kind": "readiness",
            "measured": float(min_min_grad),
            "threshold": float(MIN_GRAD_FLOOR),
            "control": "combined [z_self,z_world] from trained encoder fed to two trained reconstruction heads",
            "met": bool(min_min_grad > MIN_GRAD_FLOOR),
        },
        {
            "name": "r2_iterative_loop_iterated",
            "description": "iterative-inference settling loop must actually iterate (>=2 rounds) for the convergence readout to be non-vacuous",
            "kind": "readiness",
            "measured": float(min_n_iters),
            "threshold": 2.0,
            "control": "use_iterative_inference=True, inference_settle_iters=%d on trained substrate" % SETTLE_ITERS,
            "met": bool(min_n_iters >= 2),
        },
        {
            "name": "r3_e2_touchable_under_interleaved",
            "description": "interleaved schedule must touch E2 (>=1 update) -- the module the cross-module share depends on; a starved E2 buffer would collapse the share to 0 and masquerade as the blocked control",
            "kind": "readiness",
            "measured": float(min_updates_e2),
            "threshold": 1.0,
            "control": "E2 _e2_transition_buffer populated via record_transition during warmup",
            "met": bool(min_updates_e2 >= 1.0),
        },
    ]
    all_preconditions_met = all(p["met"] for p in preconditions)

    criteria_non_degenerate = {
        "R1": bool(all(s["R1"]["n_modules"] == 2 for s in per_seed)),
        "R2": bool(min_n_iters >= 2),
        "R3": bool(all(s["R3"]["interleaved_share"] != s["R3"]["blocked_share"] for s in per_seed)),
    }

    criteria = [
        {"name": "R1_coupled", "load_bearing": True, "passed": bool(all_r1)},
        {"name": "R2_converged_below_tol", "load_bearing": True, "passed": bool(all_r2)},
        {"name": "R3_interleaved_share_positive_blocked_zero", "load_bearing": True, "passed": bool(all_r3)},
    ]

    # ---- self-route ----
    if not all_preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        note = ("A readiness precondition is unmet -- the substrate the warmup "
                "produced cannot exercise one of R1/R2/R3, so the criteria are "
                "starved, not falsified. Re-queue at a longer warmup / fix wiring.")
    elif all_seeds_pass:
        label = "readiness_validated"
        outcome = "PASS"
        evidence_direction = "supports"
        note = ("R1/R2/R3 readouts are non-vacuous on a trained substrate. EXP-0380 "
                "would receive a meaningful reading; the readiness substrate is "
                "validated.")
    else:
        label = "readiness_not_confirmed"
        outcome = "FAIL"
        evidence_direction = "weakens"
        note = ("Preconditions met (readouts non-vacuous) but a load-bearing "
                "criterion did not meet its EXP-0380 gate threshold on this "
                "substrate (e.g. R2 fired but did not settle below tol). Revisit "
                "warmup length / settle_iters / rel_tol before building EXP-0380.")

    print(f"\n[{EXPERIMENT_TYPE}] aggregate:")
    print(f"  preconditions_met={all_preconditions_met} "
          f"(min_grad={min_min_grad:.4g} min_n_iters={min_n_iters} min_upd_e2={min_updates_e2:.0f})")
    print(f"  world_buf>={min_world_buf} e2_buf>={min_e2_buf}")
    print(f"  R1_all={all_r1} R2_all={all_r2} R3_all={all_r3}")
    print(f"  label={label} outcome={outcome} elapsed={elapsed:.1f}s")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return outcome, None

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": note,
        "sleep_driver_pattern": "N/A (CrossModuleConsolidator.consolidate() called directly; no sleep cycle)",
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "note": note,
        },
        "aggregate": {
            "all_preconditions_met": all_preconditions_met,
            "all_seeds_pass": all_seeds_pass,
            "R1_all": all_r1, "R2_all": all_r2, "R3_all": all_r3,
            "min_min_grad_norm": min_min_grad,
            "min_n_iters": min_n_iters,
            "min_updates_e2_interleaved": min_updates_e2,
            "min_world_buffer": min_world_buf,
            "min_e2_buffer": min_e2_buf,
        },
        "thresholds": {
            "min_grad_floor": MIN_GRAD_FLOOR,
            "rel_tol": REL_TOL,
            "settle_iters": SETTLE_ITERS,
            "cmc_steps": CMC_STEPS,
        },
        "config": {
            "seeds": list(seeds),
            "warmup_episodes": warmup,
            "p0_end": p0_end,
            "steps_per_episode": steps,
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
        },
        "per_seed": per_seed,
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}")
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _outcome_clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_outcome_clean, manifest_path=_manifest_path)
    sys.exit(0)
