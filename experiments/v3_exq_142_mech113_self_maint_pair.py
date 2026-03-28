#!/opt/local/bin/python3
"""
V3-EXQ-142 -- MECH-113 Self-Maintenance (z_self D_eff) Discriminative Pair

Claims: MECH-113
Proposal: EXP-0074 / EVB-0062

MECH-113 asserts:
  "The agent requires a self-maintenance error signal monitoring internal latent
  coherence, independent of external harm signals."

Specifically, the D_eff (participation ratio) of z_self must be monitored and
kept near a homeostatic setpoint. Without this signal, z_self disperses after
perturbation and does not recover -- the agent silently loses self-model
coherence even when no external harm is occurring.

This experiment is a proper matched-seed discriminative pair extending EXQ-075d
(single seed, single condition comparison) to the canonical 2-seed design.

If MECH-113 is correct:
  SELF_MAINT_ON: homeostatic loss keeps D_eff near baseline after perturbation
  (D_eff ratio post/baseline <= 1.5x).
  SELF_MAINT_ABLATED: no homeostatic signal -- D_eff disperses after perturbation
  (D_eff ratio post/baseline >= 2.0x).

Conditions
----------
SELF_MAINT_ON:
  - self_maintenance_weight=0.1, d_eff_target=1.5
  - The homeostatic loss penalises D_eff > target, acting as a reactive setpoint
    (Stephan 2016: allostatic self-efficacy Level 1).
  - E2 motor-sensory loss should not be degraded (self-maintenance is orthogonal
    to motor prediction quality).

SELF_MAINT_ABLATED:
  - self_maintenance_weight=0.0 (no homeostasis; L-space learns from harm and E2
    motor-sensory error only).
  - Expected: z_self D_eff rises significantly after perturbation and does not
    recover, because no gradient pushes it back toward a coherent setpoint.

Design rationale
----------------
MECH-113 claims homeostatic self-coherence is an ACTIVE signal, not just a
consequence of other losses. The ablation isolates exactly this: all training
is identical except the self_maintenance_weight. If the maintenance loss is the
cause of D_eff recovery, removing it should produce measurable dispersal.

Perturbation protocol:
  - After warmup, run 10 baseline eval episodes to measure D_eff at rest.
  - Inject Gaussian noise (sigma=2.0) into z_self at the start of the first
    eval episode (stress test).
  - Run 30 post-perturbation eval episodes. Measure D_eff recovery vs baseline.

The D_eff metric (participation ratio):
  D_eff = (sum|z_self|)^2 / sum(z_self^2)
High D_eff = diffuse/uncertain self-representation.
Low D_eff = coherent, focused self-model.
From epistemic-mapping repo (dgolden): "knowing one is not sure."

Pre-registered thresholds
--------------------------
C1: d_eff_ratio_ON (post/baseline) <= THRESH_D_EFF_RATIO_ON both seeds
    (ON condition: D_eff held near baseline after perturbation)
C2: d_eff_ratio_ABLATED (post/baseline) >= THRESH_D_EFF_RATIO_ABLATED both seeds
    (ABLATED condition: D_eff disperses significantly after perturbation)
C3: d_eff_gap (ABLATED_post - ON_post) >= THRESH_D_EFF_GAP_FRAC * ABLATED_post
    both seeds (the two conditions clearly separate)
C4: e2_loss_ON <= e2_loss_ABLATED + THRESH_E2_LOSS_MARGIN both seeds
    (homeostasis does not degrade motor-sensory prediction)
C5: n_d_eff_samples >= THRESH_N_D_EFF per condition per seed (data quality)

Interpretation:
  C1+C2+C3+C4+C5 ALL PASS: MECH-113 Level 1 SUPPORTED. Homeostatic D_eff
    signal is causally responsible for z_self coherence recovery. Removing it
    produces uncontrolled dispersal. E2 motor prediction is not degraded.
    This is consistent with Stephan 2016 Level 1 allostatic self-efficacy.
  C1 FAIL only: ON condition does not recover -- maintenance weight may be
    too low, or D_eff is architecture-bounded and never disperses.
  C2 FAIL only: ABLATED condition does not disperse -- D_eff may be naturally
    bounded by the architecture regardless of maintenance signal.
  C4 FAIL: self-maintenance loss interferes with E2 motor model -- weight
    may need reduction or training schedule needs separation.
  C5 FAIL: data quality; increase eval episodes.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorld size=10, 3 hazards, 5 resources, nav_bias=0.35
Warmup: 200 episodes x 200 steps
Eval:  10 baseline + 30 post-perturbation episodes x 200 steps
Estimated runtime: ~90 min any machine
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_142_mech113_self_maint_pair"
CLAIM_IDS = ["MECH-113"]

# Pre-registered thresholds
# C1: ON condition D_eff ratio (post/baseline) must be <= this (both seeds)
THRESH_D_EFF_RATIO_ON = 1.5
# C2: ABLATED condition D_eff ratio (post/baseline) must be >= this (both seeds)
THRESH_D_EFF_RATIO_ABLATED = 2.0
# C3: gap must be >= this fraction * ablated_post (both seeds)
THRESH_D_EFF_GAP_FRAC = 0.5
# C4: E2 loss delta (ON - ABLATED) must be <= this (both seeds)
THRESH_E2_LOSS_MARGIN = 0.01
# C5: minimum D_eff samples per condition per seed (data quality)
THRESH_N_D_EFF = 50

# Env / training configuration
BODY_OBS_DIM = 10
WORLD_OBS_DIM = 200  # CausalGridWorld size=10, use_proxy_fields=False
ACTION_DIM = 4

# Homeostatic signal parameters
MAINT_WEIGHT = 0.1       # self_maintenance_weight for ON condition
D_EFF_TARGET  = 1.5      # setpoint for ON condition
NOISE_SIGMA   = 2.0      # perturbation sigma


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
    Participation ratio: D_eff = (sum|z|)^2 / sum(z^2).
    Measures effective dimensionality of z_self.
    High = diffuse; Low = coherent.
    """
    z = z_self.detach().squeeze(0)
    abs_sum = z.abs().sum()
    sq_sum  = z.pow(2).sum()
    if sq_sum.item() < 1e-9:
        return float("nan")
    return float((abs_sum.pow(2) / sq_sum).item())


def _run_single(
    seed: int,
    maint_enabled: bool,
    maint_weight: float,
    d_eff_target: float,
    warmup_episodes: int,
    baseline_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    noise_sigma: float,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell.

    Returns per-seed D_eff and E2 metrics for the paired comparison.
    SELF_MAINT_ON: homeostatic loss active; expected D_eff recovery after noise.
    SELF_MAINT_ABLATED: no homeostatic signal; expected D_eff dispersal.
    """
    cond_label = (
        f"SELF_MAINT_ON(w={maint_weight},target={d_eff_target})"
        if maint_enabled
        else "SELF_MAINT_ABLATED"
    )

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
        alpha_self=alpha_self,
        self_maintenance_weight=maint_weight if maint_enabled else 0.0,
        self_maintenance_d_eff_target=d_eff_target,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    if dry_run:
        warmup_episodes = 3
        baseline_episodes = 2
        eval_episodes = 2

    print(
        f"\n[V3-EXQ-142] TRAIN {cond_label} seed={seed}"
        f" warmup={warmup_episodes} baseline={baseline_episodes}"
        f" eval={eval_episodes} nav_bias={nav_bias}",
        flush=True,
    )
    agent.train()

    # --- Warmup training ---
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
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            # Nav bias: pushes agent toward hazards to ensure harm supervision data
            if random.random() < nav_bias:
                action = torch.zeros(1, ACTION_DIM)
                action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E1 + E2 combined (avoids inplace conflict)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_e1_e2 = e1_loss + e2_loss
            if total_e1_e2.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total_e1_e2.backward()
                e1_opt.step()
                e2_opt.step()

            # E3 harm supervision + self-maintenance loss
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss  = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
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
            d_eff_str = f"{d_eff_now:.4f}" if d_eff_now is not None else "N/A"
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} d_eff={d_eff_str}",
                flush=True,
            )

    # --- Baseline eval (pre-perturbation) ---
    agent.eval()
    d_eff_baseline: List[float] = []
    e2_loss_baseline: List[float] = []

    for _ in range(baseline_episodes):
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
            if d is not None and not (d != d):  # skip nan
                d_eff_baseline.append(d)

            e2_l = agent.compute_e2_loss()
            e2_loss_baseline.append(float(e2_l.item()))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    d_eff_baseline_mean = (
        sum(d_eff_baseline) / len(d_eff_baseline) if d_eff_baseline else float("nan")
    )
    print(
        f"  [baseline] {cond_label} seed={seed} d_eff={d_eff_baseline_mean:.4f}"
        f" n_samples={len(d_eff_baseline)}",
        flush=True,
    )

    # --- Post-perturbation eval ---
    # Inject noise at the first step of the first eval episode
    d_eff_post: List[float] = []
    e2_loss_post: List[float] = []

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

            # Inject perturbation at first step of first eval episode
            if ep_idx == 0 and step == 0 and agent._current_latent is not None:
                perturbed = (
                    agent._current_latent.z_self
                    + torch.randn_like(agent._current_latent.z_self) * noise_sigma
                )
                agent._self_experience_buffer.append(perturbed.detach().clone())
                if not dry_run:
                    print(
                        f"  [perturb] {cond_label} seed={seed}"
                        f" noise sigma={noise_sigma} injected",
                        flush=True,
                    )

            d = agent.compute_z_self_d_eff()
            if d is not None and not (d != d):
                d_eff_post.append(d)

            e2_l = agent.compute_e2_loss()
            e2_loss_post.append(float(e2_l.item()))

            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    d_eff_post_mean = (
        sum(d_eff_post) / len(d_eff_post) if d_eff_post else float("nan")
    )
    e2_loss_post_mean   = sum(e2_loss_post)   / max(1, len(e2_loss_post))
    e2_loss_base_mean   = sum(e2_loss_baseline) / max(1, len(e2_loss_baseline))

    d_eff_ratio = (
        d_eff_post_mean / d_eff_baseline_mean
        if d_eff_baseline_mean > 0 and not (d_eff_baseline_mean != d_eff_baseline_mean)
        else float("nan")
    )

    print(
        f"  [post-perturb] {cond_label} seed={seed}"
        f" d_eff_baseline={d_eff_baseline_mean:.4f}"
        f" d_eff_post={d_eff_post_mean:.4f}"
        f" ratio={d_eff_ratio:.3f}x"
        f" e2_loss_post={e2_loss_post_mean:.5f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "maint_enabled": maint_enabled,
        "d_eff_baseline": d_eff_baseline_mean,
        "d_eff_post": d_eff_post_mean,
        "d_eff_ratio": d_eff_ratio,
        "e2_loss_baseline": e2_loss_base_mean,
        "e2_loss_post": e2_loss_post_mean,
        "n_d_eff_baseline": len(d_eff_baseline),
        "n_d_eff_post": len(d_eff_post),
    }


def run(
    seeds: Tuple[int, ...] = (42, 123),
    maint_weight: float = MAINT_WEIGHT,
    d_eff_target: float = D_EFF_TARGET,
    noise_sigma: float = NOISE_SIGMA,
    warmup_episodes: int = 200,
    baseline_episodes: int = 10,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.35,
    dry_run: bool = False,
) -> dict:
    """MECH-113: SELF_MAINT_ON vs SELF_MAINT_ABLATED discriminative pair.

    Matched-seed design: same env per seed across conditions.
    Tests whether homeostatic D_eff signal is causally responsible for z_self
    coherence recovery after perturbation.
    """
    print(
        f"\n[V3-EXQ-142] MECH-113 Self-Maintenance Discriminative Pair"
        f" seeds={list(seeds)}"
        f" maint_weight={maint_weight}"
        f" d_eff_target={d_eff_target}"
        f" noise_sigma={noise_sigma}",
        flush=True,
    )

    results_on:  List[Dict] = []
    results_abl: List[Dict] = []

    for seed in seeds:
        for maint_on in [True, False]:
            r = _run_single(
                seed=seed,
                maint_enabled=maint_on,
                maint_weight=maint_weight,
                d_eff_target=d_eff_target,
                warmup_episodes=warmup_episodes,
                baseline_episodes=baseline_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                noise_sigma=noise_sigma,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if maint_on:
                results_on.append(r)
            else:
                results_abl.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results if r[key] == r[key]]  # skip nan
        return float(sum(vals) / max(1, len(vals)))

    # --- Pre-registered criteria (per-seed paired comparisons) ---

    # C1: ON ratio <= THRESH both seeds
    c1_per_seed = [
        r["d_eff_ratio"] <= THRESH_D_EFF_RATIO_ON
        for r in results_on
        if r["d_eff_ratio"] == r["d_eff_ratio"]  # skip nan
    ]
    c1_pass = len(c1_per_seed) >= len(seeds) and all(c1_per_seed)

    # C2: ABLATED ratio >= THRESH both seeds
    c2_per_seed = [
        r["d_eff_ratio"] >= THRESH_D_EFF_RATIO_ABLATED
        for r in results_abl
        if r["d_eff_ratio"] == r["d_eff_ratio"]
    ]
    c2_pass = len(c2_per_seed) >= len(seeds) and all(c2_per_seed)

    # C3: gap >= THRESH_GAP_FRAC * ablated_post both seeds
    c3_per_seed = []
    per_seed_gap: List[float] = []
    for r_on in results_on:
        matching = [r for r in results_abl if r["seed"] == r_on["seed"]]
        if matching:
            r_abl = matching[0]
            gap = r_abl["d_eff_post"] - r_on["d_eff_post"]
            per_seed_gap.append(gap)
            c3_per_seed.append(
                gap >= THRESH_D_EFF_GAP_FRAC * r_abl["d_eff_post"]
                if r_abl["d_eff_post"] > 0 else False
            )
    c3_pass = len(c3_per_seed) >= len(seeds) and all(c3_per_seed)

    # C4: E2 loss ON <= E2 loss ABLATED + margin both seeds
    c4_per_seed = []
    for r_on in results_on:
        matching = [r for r in results_abl if r["seed"] == r_on["seed"]]
        if matching:
            r_abl = matching[0]
            c4_per_seed.append(
                r_on["e2_loss_post"] <= r_abl["e2_loss_post"] + THRESH_E2_LOSS_MARGIN
            )
    c4_pass = len(c4_per_seed) >= len(seeds) and all(c4_per_seed)

    # C5: data quality -- sufficient D_eff samples both conditions both seeds
    n_d_eff_min = min(
        r["n_d_eff_post"] for r in results_on + results_abl
    )
    c5_pass = n_d_eff_min >= THRESH_N_D_EFF

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass or c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    # Summary statistics
    mean_d_eff_baseline_on  = _avg(results_on,  "d_eff_baseline")
    mean_d_eff_post_on      = _avg(results_on,  "d_eff_post")
    mean_ratio_on           = _avg(results_on,  "d_eff_ratio")
    mean_d_eff_baseline_abl = _avg(results_abl, "d_eff_baseline")
    mean_d_eff_post_abl     = _avg(results_abl, "d_eff_post")
    mean_ratio_abl          = _avg(results_abl, "d_eff_ratio")
    mean_e2_on              = _avg(results_on,  "e2_loss_post")
    mean_e2_abl             = _avg(results_abl, "e2_loss_post")

    print(
        f"\n[V3-EXQ-142] Results:"
        f" ON d_eff_base={mean_d_eff_baseline_on:.4f} post={mean_d_eff_post_on:.4f}"
        f" ratio={mean_ratio_on:.3f}x",
        flush=True,
    )
    print(
        f"  ABL d_eff_base={mean_d_eff_baseline_abl:.4f} post={mean_d_eff_post_abl:.4f}"
        f" ratio={mean_ratio_abl:.3f}x",
        flush=True,
    )
    print(
        f"  per_seed_gap={[round(g, 4) for g in per_seed_gap]}"
        f" e2_on={mean_e2_on:.5f} e2_abl={mean_e2_abl:.5f}"
        f" n_d_eff_min={n_d_eff_min}",
        flush=True,
    )
    print(
        f"  C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass} C5={c5_pass}"
        f" decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    # Failure notes
    failure_notes: List[str] = []
    if not c1_pass:
        on_ratios = [round(r["d_eff_ratio"], 4) for r in results_on]
        failure_notes.append(
            f"C1 FAIL: ON d_eff_ratio {on_ratios} > {THRESH_D_EFF_RATIO_ON}"
            " -- homeostatic loss does not hold D_eff near baseline;"
            " maintenance weight may need increase or architecture bounds D_eff"
        )
    if not c2_pass:
        abl_ratios = [round(r["d_eff_ratio"], 4) for r in results_abl]
        failure_notes.append(
            f"C2 FAIL: ABLATED d_eff_ratio {abl_ratios} < {THRESH_D_EFF_RATIO_ABLATED}"
            " -- ablated condition does not disperse; D_eff may be architecture-bounded"
            " independent of maintenance signal"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per-seed gap {[round(g, 4) for g in per_seed_gap]}"
            f" < {THRESH_D_EFF_GAP_FRAC} * ablated_post in some seeds;"
            " ON and ABLATED conditions do not clearly separate"
        )
    if not c4_pass:
        c4_vals = [
            (r["seed"], round(r["e2_loss_post"], 6)) for r in results_on
        ]
        c4_abl_vals = [
            (r["seed"], round(r["e2_loss_post"], 6)) for r in results_abl
        ]
        failure_notes.append(
            f"C4 FAIL: e2_loss_ON {c4_vals} >"
            f" e2_loss_ABL {c4_abl_vals} + {THRESH_E2_LOSS_MARGIN}"
            " -- self-maintenance loss interferes with E2 motor prediction;"
            " reduce maintenance_weight or separate training phases"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_d_eff_min={n_d_eff_min} < {THRESH_N_D_EFF}"
            " -- insufficient D_eff samples; increase eval_episodes or steps"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Interpretation text
    if all_pass:
        interpretation = (
            f"MECH-113 Level 1 SUPPORTED: Homeostatic self-maintenance loss"
            f" (D_eff penalty) causally holds z_self coherence near baseline"
            f" after Gaussian perturbation (sigma={noise_sigma}). ON ratio"
            f" {mean_ratio_on:.3f}x <= {THRESH_D_EFF_RATIO_ON}x (recovery);"
            f" ABLATED ratio {mean_ratio_abl:.3f}x >= {THRESH_D_EFF_RATIO_ABLATED}x"
            f" (dispersal). Per-seed gap {[round(g, 4) for g in per_seed_gap]}."
            f" E2 motor-sensory prediction unaffected (C4). This is consistent with"
            f" Stephan 2016 Level 1 allostatic self-efficacy: the reactive homeostatic"
            f" setpoint mechanism works. D_eff framing (epistemic-mapping) confirmed:"
            f" 'knowing one is not sure' is monitored and corrected."
        )
    elif c1_pass or c2_pass:
        interpretation = (
            f"Partial support: directional D_eff effect observed but not all criteria"
            f" met. C1={c1_pass} (ON recovery) C2={c2_pass} (ABL dispersal)"
            f" C3={c3_pass} (separation) C4={c4_pass} (E2 unaffected)"
            f" C5={c5_pass} (data quality)."
            f" Consider adjusting maint_weight or noise_sigma."
        )
    else:
        interpretation = (
            f"MECH-113 Level 1 NOT SUPPORTED: Self-maintenance loss does not produce"
            f" measurable D_eff homeostasis after perturbation. ON ratio"
            f" {mean_ratio_on:.3f}x; ABLATED ratio {mean_ratio_abl:.3f}x."
            f" Possible causes: D_eff is architecture-bounded (does not vary"
            f" regardless of maintenance signal); maintenance loss is not propagating"
            f" through z_self gradient path; or perturbation sigma={noise_sigma}"
            f" is too small to exceed the architecture's implicit regularisation."
        )

    # Per-seed detail rows
    per_on_rows = "\n".join(
        f"  seed={r['seed']}: d_eff_baseline={r['d_eff_baseline']:.4f}"
        f" d_eff_post={r['d_eff_post']:.4f}"
        f" ratio={r['d_eff_ratio']:.3f}x"
        f" e2_loss={r['e2_loss_post']:.5f}"
        f" n_samples={r['n_d_eff_post']}"
        for r in results_on
    )
    per_abl_rows = "\n".join(
        f"  seed={r['seed']}: d_eff_baseline={r['d_eff_baseline']:.4f}"
        f" d_eff_post={r['d_eff_post']:.4f}"
        f" ratio={r['d_eff_ratio']:.3f}x"
        f" e2_loss={r['e2_loss_post']:.5f}"
        f" n_samples={r['n_d_eff_post']}"
        for r in results_abl
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-142 -- MECH-113 Self-Maintenance Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-113\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** SELF_MAINT_ON vs SELF_MAINT_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Baseline eval:** {baseline_episodes} eps"
        f"  **Post-perturb eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorld size=10, 3 hazards, 5 resources"
        f" nav_bias={nav_bias}\n"
        f"**Perturbation:** Gaussian noise sigma={noise_sigma} injected into z_self"
        f" at first step of eval\n"
        f"**maint_weight:** {maint_weight}  **d_eff_target:** {d_eff_target}\n\n"
        f"## Design\n\n"
        f"MECH-113 asserts a homeostatic error signal monitors internal latent"
        f" coherence (z_self D_eff) independently of external harm signals."
        f" SELF_MAINT_ON activates the D_eff homeostatic loss;"
        f" SELF_MAINT_ABLATED removes it. A Gaussian noise perturbation is injected"
        f" into z_self at eval start to stress-test recovery. If MECH-113 Level 1"
        f" is correct, ON holds D_eff near baseline; ABLATED disperses.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: ON d_eff_ratio (post/baseline) <= {THRESH_D_EFF_RATIO_ON} both seeds"
        f" (homeostatic recovery)\n"
        f"C2: ABLATED d_eff_ratio (post/baseline) >= {THRESH_D_EFF_RATIO_ABLATED} both seeds"
        f" (uncontrolled dispersal without homeostasis)\n"
        f"C3: per-seed gap (ABLATED_post - ON_post) >= {THRESH_D_EFF_GAP_FRAC}x ABLATED_post"
        f" both seeds (clear separation)\n"
        f"C4: e2_loss_ON <= e2_loss_ABLATED + {THRESH_E2_LOSS_MARGIN} both seeds"
        f" (homeostasis does not degrade E2 motor model)\n"
        f"C5: n_d_eff_samples >= {THRESH_N_D_EFF} per condition per seed (data quality)\n\n"
        f"## Results\n\n"
        f"| Condition | D_eff baseline | D_eff post | ratio | E2 loss |\n"
        f"|-----------|----------------|------------|-------|----------|\n"
        f"| SELF_MAINT_ON      | {mean_d_eff_baseline_on:.4f}"
        f" | {mean_d_eff_post_on:.4f}"
        f" | {mean_ratio_on:.3f}x"
        f" | {mean_e2_on:.5f} |\n"
        f"| SELF_MAINT_ABLATED | {mean_d_eff_baseline_abl:.4f}"
        f" | {mean_d_eff_post_abl:.4f}"
        f" | {mean_ratio_abl:.3f}x"
        f" | {mean_e2_abl:.5f} |\n\n"
        f"**per-seed gap (ABLATED_post - ON_post):"
        f" {[round(g, 4) for g in per_seed_gap]}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: ON ratio <= {THRESH_D_EFF_RATIO_ON} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {[round(r['d_eff_ratio'], 4) for r in results_on]} |\n"
        f"| C2: ABLATED ratio >= {THRESH_D_EFF_RATIO_ABLATED} (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(r['d_eff_ratio'], 4) for r in results_abl]} |\n"
        f"| C3: gap >= {THRESH_D_EFF_GAP_FRAC}x ABLATED_post (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {[round(g, 4) for g in per_seed_gap]} |\n"
        f"| C4: e2_ON <= e2_ABL + {THRESH_E2_LOSS_MARGIN} (both seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | ON={mean_e2_on:.5f} ABL={mean_e2_abl:.5f} |\n"
        f"| C5: n_d_eff_min >= {THRESH_N_D_EFF}"
        f" | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_d_eff_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed Detail\n\n"
        f"SELF_MAINT_ON:\n{per_on_rows}\n\n"
        f"SELF_MAINT_ABLATED:\n{per_abl_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "mean_d_eff_baseline_on":     float(mean_d_eff_baseline_on),
        "mean_d_eff_post_on":         float(mean_d_eff_post_on),
        "mean_d_eff_ratio_on":        float(mean_ratio_on),
        "mean_d_eff_baseline_ablated": float(mean_d_eff_baseline_abl),
        "mean_d_eff_post_ablated":    float(mean_d_eff_post_abl),
        "mean_d_eff_ratio_ablated":   float(mean_ratio_abl),
        "mean_e2_loss_on":            float(mean_e2_on),
        "mean_e2_loss_ablated":       float(mean_e2_abl),
        "per_seed_gap":               [round(g, 6) for g in per_seed_gap],
        "per_seed_ratio_on":          [round(r["d_eff_ratio"], 6) for r in results_on],
        "per_seed_ratio_ablated":     [round(r["d_eff_ratio"], 6) for r in results_abl],
        "n_d_eff_min":                float(n_d_eff_min),
        "maint_weight":               float(maint_weight),
        "d_eff_target":               float(d_eff_target),
        "noise_sigma":                float(noise_sigma),
        "crit1_pass":                 1.0 if c1_pass else 0.0,
        "crit2_pass":                 1.0 if c2_pass else 0.0,
        "crit3_pass":                 1.0 if c3_pass else 0.0,
        "crit4_pass":                 1.0 if c4_pass else 0.0,
        "crit5_pass":                 1.0 if c5_pass else 0.0,
        "criteria_met":               float(criteria_met),
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
        "decision": decision,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--maint-weight",    type=float, default=MAINT_WEIGHT)
    parser.add_argument("--d-eff-target",    type=float, default=D_EFF_TARGET)
    parser.add_argument("--noise-sigma",     type=float, default=NOISE_SIGMA)
    parser.add_argument("--warmup",          type=int,   default=200)
    parser.add_argument("--baseline-eps",    type=int,   default=10)
    parser.add_argument("--eval-eps",        type=int,   default=30)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        maint_weight=args.maint_weight,
        d_eff_target=args.d_eff_target,
        noise_sigma=args.noise_sigma,
        warmup_episodes=args.warmup,
        baseline_episodes=args.baseline_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        dry_run=args.dry_run,
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
