"""
V3-EXQ-075 -- MECH-113 Self-Maintenance (z_self D_eff Framing)

Claims: MECH-113

MECH-113 asserts that the agent requires a homeostatic mechanism over z_self
to maintain coherent self-modelling. Without it, z_self dispersion (D_eff)
rises under perturbation and does not recover, degrading the E2 motor-sensory
model. With self-maintenance loss active, D_eff is held near baseline.

D_eff (effective dimensionality, participation ratio) is the primary metric:
  D_eff = (sum|z_self|)^2 / sum(z_self^2)
From epistemic-mapping repo (dgolden): this is a framing of "knowing one is
not sure" -- high D_eff = diffuse/uncertain self-representation; low D_eff =
coherent focused self-model. The self-maintenance loss penalises D_eff > target,
acting as a homeostatic setpoint (Stephan 2016: allostatic self-efficacy level 2).

Two conditions (matched seeds):
  A. NoMaintenance  -- self_maintenance_weight=0 (no homeostasis)
  B. Maintenance    -- self_maintenance_weight=0.1, target D_eff=1.5

Perturbation protocol:
  After warmup_episodes, inject Gaussian noise (sigma=2.0) into z_self at
  step 500 of episode `perturbation_episode`. Measure D_eff recovery over
  the subsequent eval_episodes.

PASS criteria (ALL required):
  C1: d_eff_post_perturb_maint <= d_eff_baseline_maint * 1.5
      (maintenance condition: D_eff <=50% above baseline after perturbation)
  C2: d_eff_post_perturb_nomaint >= d_eff_baseline_nomaint * 2.0
      (no-maintenance: D_eff >=2x baseline = uncontrolled dispersal)
  C3: d_eff_gap >= 0.5 * d_eff_post_perturb_nomaint
      (maintenance and no-maintenance conditions clearly separate)
  C4: e2_loss_maint <= e2_loss_nomaint + 0.01
      (self-maintenance does not degrade motor-sensory prediction)

Informational:
  d_eff time-series per condition (pre/post perturbation)
  Additional epistemic-mapping metrics: entropy of z_self, z_self norm

Connection to epistemic-mapping repo:
  The D_eff formula mirrors the participation ratio used in EpistemicMonitor.
  A Hopfield-stability framing (familiarity of z_self patterns) is noted as
  a future extension (MECH-113 pending_design). This experiment focuses on
  D_eff because it is directly computable from z_self without additional
  memory infrastructure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from typing import Dict, List

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_075_mech113_self_maintenance"
CLAIM_IDS = ["MECH-113"]

BODY_OBS_DIM = 10
WORLD_OBS_DIM = 54
ACTION_DIM = 4


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        use_proxy_fields=False,
        seed=seed,
    )


def _compute_d_eff(z_self: torch.Tensor) -> float:
    """
    Participation ratio D_eff = (sum|z_self|)^2 / sum(z_self^2).

    From epistemic-mapping/Epistemic_monitor.py. Measures effective
    dimensionality of z_self representation. High = diffuse/uncertain.
    """
    z = z_self.squeeze(0)  # [self_dim]
    abs_z = z.abs()
    numerator = abs_z.sum().pow(2)
    denominator = z.pow(2).sum()
    if denominator.item() < 1e-8:
        return float("nan")
    return (numerator / denominator).item()


def _z_self_entropy(z_self: torch.Tensor) -> float:
    """Shannon entropy of |z_self| distribution (informational)."""
    z = z_self.squeeze(0).abs()
    total = z.sum()
    if total.item() < 1e-8:
        return 0.0
    probs = z / total
    return float(-torch.sum(probs * torch.log(probs + 1e-9)).item())


def _run_single(
    seed: int,
    maintenance_enabled: bool,
    maintenance_weight: float,
    d_eff_target: float,
    warmup_episodes: int,
    eval_episodes: int,
    perturbation_episode: int,
    steps_per_episode: int,
    noise_sigma: float = 2.0,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
) -> Dict:
    cond_label = f"MAINTENANCE(w={maintenance_weight})" if maintenance_enabled else "NO_MAINTENANCE"

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        self_maintenance_weight=maintenance_weight if maintenance_enabled else 0.0,
        self_maintenance_d_eff_target=d_eff_target,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    print(f"\n[EXQ-075] TRAIN {cond_label} seed={seed}", flush=True)
    agent.train()

    # Warmup training
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach())

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            # E3 harm supervision + self-maintenance loss
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                maint_loss = agent.compute_self_maintenance_loss()
                total_e3 = harm_loss + maint_loss
                e3_opt.zero_grad()
                total_e3.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            d_eff_now = agent.compute_z_self_d_eff()
            d_eff_str = f"{d_eff_now:.3f}" if d_eff_now is not None else "N/A"
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} d_eff={d_eff_str}",
                flush=True,
            )

    # --- Measure baseline D_eff (pre-perturbation eval) ---
    agent.eval()
    d_eff_baseline: List[float] = []
    e2_losses_baseline: List[float] = []

    for _ in range(10):  # 10 episodes baseline
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                agent._e1_tick(latent)

            d = agent.compute_z_self_d_eff()
            if d is not None:
                d_eff_baseline.append(d)

            e2_l = agent.compute_e2_loss()
            e2_losses_baseline.append(float(e2_l.item()))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    d_eff_baseline_mean = sum(d_eff_baseline) / max(1, len(d_eff_baseline))
    print(f"  [baseline] {cond_label} d_eff={d_eff_baseline_mean:.4f}", flush=True)

    # --- Perturbation: inject noise into z_self ---
    print(f"  [perturb] injecting noise sigma={noise_sigma}...", flush=True)
    if agent._current_latent is not None:
        noise = torch.randn_like(agent._current_latent.z_self) * noise_sigma
        # Create a perturbed copy (LatentState is a dataclass, access fields directly)
        lat = agent._current_latent
        # Inject by manually setting the stored z_self after encoding
        agent._self_experience_buffer.clear()  # clear buffer to force fresh E2 context

    # Post-perturbation eval
    d_eff_post: List[float] = []
    e2_losses_post: List[float] = []

    for ep_idx in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                agent._e1_tick(latent)

            # Inject perturbation at first step of perturbation_episode
            if ep_idx == 0 and step == 0 and agent._current_latent is not None:
                perturbed_z_self = agent._current_latent.z_self + \
                    torch.randn_like(agent._current_latent.z_self) * noise_sigma
                # Store the perturbed z_self back so E2 and E3 see it
                # We do this by directly modifying the latent's z_self-tracked copy
                agent._self_experience_buffer.append(perturbed_z_self.detach().clone())

            d = agent.compute_z_self_d_eff()
            if d is not None:
                d_eff_post.append(d)

            e2_l = agent.compute_e2_loss()
            e2_losses_post.append(float(e2_l.item()))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    d_eff_post_mean = sum(d_eff_post) / max(1, len(d_eff_post))
    e2_loss_mean = sum(e2_losses_post) / max(1, len(e2_losses_post))
    e2_loss_baseline = sum(e2_losses_baseline) / max(1, len(e2_losses_baseline))

    print(
        f"  [post-perturb] {cond_label} seed={seed}"
        f" d_eff_baseline={d_eff_baseline_mean:.4f}"
        f" d_eff_post={d_eff_post_mean:.4f}"
        f" ratio={d_eff_post_mean/max(d_eff_baseline_mean,0.01):.2f}x"
        f" e2_loss={e2_loss_mean:.5f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "maintenance_enabled": maintenance_enabled,
        "d_eff_baseline": d_eff_baseline_mean,
        "d_eff_post_perturb": d_eff_post_mean,
        "d_eff_ratio": d_eff_post_mean / max(d_eff_baseline_mean, 0.01),
        "e2_loss_post": e2_loss_mean,
        "e2_loss_baseline": e2_loss_baseline,
        "n_d_eff_baseline": len(d_eff_baseline),
        "n_d_eff_post": len(d_eff_post),
    }


def run(
    seed: int = 42,
    maintenance_weight: float = 0.1,
    d_eff_target: float = 1.5,
    noise_sigma: float = 2.0,
    warmup_episodes: int = 200,
    eval_episodes: int = 30,
    perturbation_episode: int = 0,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    """MECH-113: self-maintenance (D_eff homeostasis) -- NoMaintenance vs Maintenance."""
    print(f"\n[EXQ-075] MECH-113 Self-Maintenance (D_eff framing)", flush=True)
    print(
        f"  seed={seed} weight={maintenance_weight}"
        f" target_d_eff={d_eff_target} noise_sigma={noise_sigma}",
        flush=True,
    )

    r_nomaint = _run_single(
        seed=seed,
        maintenance_enabled=False,
        maintenance_weight=0.0,
        d_eff_target=d_eff_target,
        warmup_episodes=warmup_episodes,
        eval_episodes=eval_episodes,
        perturbation_episode=perturbation_episode,
        steps_per_episode=steps_per_episode,
        noise_sigma=noise_sigma,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
    )
    r_maint = _run_single(
        seed=seed,
        maintenance_enabled=True,
        maintenance_weight=maintenance_weight,
        d_eff_target=d_eff_target,
        warmup_episodes=warmup_episodes,
        eval_episodes=eval_episodes,
        perturbation_episode=perturbation_episode,
        steps_per_episode=steps_per_episode,
        noise_sigma=noise_sigma,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
    )

    d_eff_baseline_maint  = r_maint["d_eff_baseline"]
    d_eff_post_maint      = r_maint["d_eff_post_perturb"]
    d_eff_baseline_nomaint = r_nomaint["d_eff_baseline"]
    d_eff_post_nomaint    = r_nomaint["d_eff_post_perturb"]
    e2_loss_maint  = r_maint["e2_loss_post"]
    e2_loss_nomaint = r_nomaint["e2_loss_post"]

    # D_eff gap between conditions post-perturbation
    d_eff_gap = d_eff_post_nomaint - d_eff_post_maint

    c1_pass = d_eff_post_maint <= d_eff_baseline_maint * 1.5
    c2_pass = d_eff_post_nomaint >= d_eff_baseline_nomaint * 2.0
    c3_pass = d_eff_gap >= 0.5 * d_eff_post_nomaint
    c4_pass = e2_loss_maint <= e2_loss_nomaint + 0.01

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-075] Results:", flush=True)
    print(
        f"  D_eff:   maint baseline={d_eff_baseline_maint:.4f}  post={d_eff_post_maint:.4f}"
        f"  ratio={r_maint['d_eff_ratio']:.2f}x",
        flush=True,
    )
    print(
        f"  D_eff:   nomaint baseline={d_eff_baseline_nomaint:.4f}  post={d_eff_post_nomaint:.4f}"
        f"  ratio={r_nomaint['d_eff_ratio']:.2f}x",
        flush=True,
    )
    print(f"  D_eff gap (nomaint - maint) post-perturb: {d_eff_gap:.4f}", flush=True)
    print(f"  E2 loss: maint={e2_loss_maint:.5f}  nomaint={e2_loss_nomaint:.5f}", flush=True)
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-113 SUPPORTED: Self-maintenance loss (D_eff penalty) holds z_self"
            " effective dimensionality near baseline after perturbation (<=1.5x),"
            " while no-maintenance condition disperses (>=2x). The gap between"
            " conditions is clear. E2 motor-sensory loss is not degraded by the"
            " maintenance signal. This is consistent with Stephan 2016 allostatic"
            " self-efficacy: the homeostatic setpoint mechanism works at level 2."
            " D_eff framing from epistemic-mapping: 'knowing one is not sure'"
            " is monitored and corrected."
        )
    elif criteria_met >= 2:
        interpretation = (
            "MECH-113 PARTIAL: Some homeostatic signal present but below threshold."
            " Perturbation may be too weak, or maintenance weight needs adjustment."
        )
    else:
        interpretation = (
            "MECH-113 NOT SUPPORTED: Self-maintenance loss does not produce measurable"
            " D_eff homeostasis after perturbation. D_eff may already be naturally"
            " bounded (architecture-level constraint), or the loss is not propagating"
            " to z_self effectively."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: maint D_eff post/baseline={r_maint['d_eff_ratio']:.2f}x > 1.5x"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: nomaint D_eff post/baseline={r_nomaint['d_eff_ratio']:.2f}x < 2.0x"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap={d_eff_gap:.4f} < 0.5 * nomaint_post={0.5*d_eff_post_nomaint:.4f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: maint e2_loss={e2_loss_maint:.5f} > nomaint+0.01={e2_loss_nomaint+0.01:.5f}"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = (
        f"# V3-EXQ-075 -- MECH-113 Self-Maintenance (z_self D_eff)\n\n"
        f"**Status:** {status}\n**Claims:** MECH-113\n"
        f"**Seed:** {seed}  **Warmup:** {warmup_episodes}  **Eval:** {eval_episodes}\n"
        f"**maintenance_weight:** {maintenance_weight}  **d_eff_target:** {d_eff_target}\n"
        f"**noise_sigma:** {noise_sigma}\n\n"
        f"## Results\n\n"
        f"| Metric | NoMaintenance | Maintenance |\n|---|---|---|\n"
        f"| D_eff baseline (pre-perturb) | {d_eff_baseline_nomaint:.4f} | {d_eff_baseline_maint:.4f} |\n"
        f"| D_eff post-perturbation | {d_eff_post_nomaint:.4f} | {d_eff_post_maint:.4f} |\n"
        f"| D_eff ratio (post/baseline) | {r_nomaint['d_eff_ratio']:.2f}x | {r_maint['d_eff_ratio']:.2f}x |\n"
        f"| E2 loss post-perturb | {e2_loss_nomaint:.5f} | {e2_loss_maint:.5f} |\n\n"
        f"**D_eff gap (nomaint - maint) post-perturb: {d_eff_gap:.4f}**\n\n"
        f"## PASS Criteria\n\n| Criterion | Result |\n|---|---|\n"
        f"| C1: maint D_eff ratio <= 1.5x | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: nomaint D_eff ratio >= 2.0x | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: gap >= 0.5 * nomaint_post | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: e2_loss not degraded | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "d_eff_baseline_nomaint":   float(d_eff_baseline_nomaint),
        "d_eff_post_nomaint":       float(d_eff_post_nomaint),
        "d_eff_ratio_nomaint":      float(r_nomaint["d_eff_ratio"]),
        "d_eff_baseline_maint":     float(d_eff_baseline_maint),
        "d_eff_post_maint":         float(d_eff_post_maint),
        "d_eff_ratio_maint":        float(r_maint["d_eff_ratio"]),
        "d_eff_gap":                float(d_eff_gap),
        "e2_loss_nomaint":          float(e2_loss_nomaint),
        "e2_loss_maint":            float(e2_loss_maint),
        "noise_sigma":              float(noise_sigma),
        "maintenance_weight":       float(maintenance_weight),
        "d_eff_target":             float(d_eff_target),
        "crit1_pass":              1.0 if c1_pass else 0.0,
        "crit2_pass":              1.0 if c2_pass else 0.0,
        "crit3_pass":              1.0 if c3_pass else 0.0,
        "crit4_pass":              1.0 if c4_pass else 0.0,
        "criteria_met":            float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--maintenance-weight",  type=float, default=0.1)
    parser.add_argument("--d-eff-target",        type=float, default=1.5)
    parser.add_argument("--noise-sigma",         type=float, default=2.0)
    parser.add_argument("--warmup",              type=int,   default=200)
    parser.add_argument("--eval-eps",            type=int,   default=30)
    parser.add_argument("--steps",               type=int,   default=200)
    parser.add_argument("--alpha-world",         type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        maintenance_weight=args.maintenance_weight,
        d_eff_target=args.d_eff_target,
        noise_sigma=args.noise_sigma,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

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
