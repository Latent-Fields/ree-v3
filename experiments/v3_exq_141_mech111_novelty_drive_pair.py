#!/opt/local/bin/python3
"""
V3-EXQ-141 -- MECH-111 Novelty/Curiosity Drive Discriminative Pair

Claims: MECH-111
Proposal: EXP-0072 / EVB-0061

MECH-111 asserts:
  "E1 prediction error surprise at moderate magnitudes generates intrinsic
  positive valence (curiosity/novelty drive); information-seeking is
  architecturally grounded."

Key architectural requirement: the agent must distinguish alarm-surprise
(high magnitude, harmful context) from curiosity-surprise (moderate magnitude,
safe novel context). Without a novelty drive, the gradient minimum under
harm-avoidance-only training is quiescence -- the agent that does nothing
accrues no harm signal and appears to perform optimally. The novelty drive
provides the Go-channel signal (MECH-112 / ARC-030 context) that prevents
behavioral flatness.

This experiment extends EXQ-073b (single seed) to a proper matched-seed
discriminative pair, with 5 pre-registered criteria.

Conditions
----------
NOVELTY_DRIVE_ON:
  - E1 prediction error EMA feeds back as an exploration bonus to E3
    trajectory scoring. Trajectories leading to novel (high-E1-error) states
    are preferred.
  - novelty_bonus_weight = 0.10 (moderate drive strength).
  - The agent should show higher policy entropy and greater cell coverage
    without substantially increasing harm contact.

NOVELTY_DRIVE_ABLATED:
  - No novelty bonus (novelty_bonus_weight = 0.0).
  - Pure harm-avoidance training. Exploration is incidental, not driven.
  - Expected: lower policy entropy, fewer novel cells visited, possibly
    converging toward quiescent near-center policy.

Design rationale
----------------
If MECH-111 is correct, the novelty drive should produce measurably more
exploratory behavior (higher entropy, wider cell coverage) while keeping harm
exposure bounded. The harm-safety constraint (C3) confirms the drive is
curiosity-surprise (moderate novelty, safe) not alarm-surprise (high magnitude,
harmful): the novelty EMA should not push the agent into dangerous zones.

Manipulation check (C4/C5):
  - C4: novelty_ema at eval must be non-zero in NOVELTY_DRIVE_ON (signal
    actually operates).
  - C5: sufficient harm contacts in both conditions (agent is not degenerate;
    harm-avoidance context exists).

Pre-registered thresholds
--------------------------
C1: entropy_gap = entropy_ON - entropy_ABLATED >= THRESH_ENT_GAP (both seeds)
    (novelty drive increases action diversity)
C2: cell_gap = novel_cells_ON - novel_cells_ABLATED >= THRESH_CELL_GAP (both seeds)
    (novelty drive increases cell coverage)
C3: harm_delta = harm_rate_ON - harm_rate_ABLATED <= THRESH_HARM_DELTA (both seeds)
    (novelty drive does not substantially increase harm exposure)
C4: novelty_ema_on > THRESH_NOVELTY_EMA (both seeds; drive signal is non-zero)
C5: n_harm_min >= THRESH_N_HARM both conditions both seeds (data quality)

Interpretation:
  C1+C2+C3+C4+C5 PASS: MECH-111 SUPPORTED. E1 novelty EMA drives approach
    toward unexplored states; curiosity-surprise (moderate, safe) is distinct
    from alarm-surprise (harm). Information-seeking is architecturally grounded.
  C1 or C2 fail, C3 pass: exploration signal present but below threshold;
    increase novelty_bonus_weight or warmup.
  C3 fail: novelty drive causes harm increase; drive is not curiosity-safe;
    may conflate alarm-surprise and curiosity-surprise.
  C4 fail: novelty EMA absent; check E1 prediction loss flow.
  C5 fail: data quality; increase nav_bias or eval episodes.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorld size=10, 5 hazards, 5 resources, nav_bias=0.30
       (lower bias than harm-only experiments -- allows exploration)
Warmup: 200 episodes x 200 steps
Eval:   50 episodes x 200 steps
Estimated runtime: ~90 min any machine
"""

import sys
import random
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_141_mech111_novelty_drive_pair"
CLAIM_IDS = ["MECH-111"]

# Pre-registered thresholds
# C1: entropy gap (novelty ON - OFF) must be >= this both seeds
THRESH_ENT_GAP = 0.10
# C2: novel cell coverage gap (novelty ON - OFF) must be >= this both seeds
THRESH_CELL_GAP = 3
# C3: harm rate delta (ON - OFF) must be <= this both seeds (drive is safe)
THRESH_HARM_DELTA = 0.02
# C4: novelty EMA at eval must be above this in NOVELTY_DRIVE_ON both seeds
THRESH_NOVELTY_EMA = 1e-6
# C5: minimum harm contacts both conditions both seeds (data quality)
THRESH_N_HARM = 10

# Env / training configuration
BODY_OBS_DIM = 10    # no proxy fields needed
WORLD_OBS_DIM = 200  # CausalGridWorld size=10, use_proxy_fields=False
ACTION_DIM = 4
NOVELTY_BONUS_WEIGHT = 0.10  # strength of novelty bonus in NOVELTY_DRIVE_ON


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=5,
        use_proxy_fields=False,
        seed=seed,
    )


def _action_entropy(action_counts: List[int]) -> float:
    """Compute Shannon entropy of action distribution."""
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _run_single(
    seed: int,
    novelty_enabled: bool,
    novelty_bonus_weight: float,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Run one (seed, condition) cell.

    Returns per-seed metrics for the paired comparison.
    NOVELTY_DRIVE_ON: novelty EMA bonus active; expected higher entropy/coverage.
    NOVELTY_DRIVE_ABLATED: no bonus; pure harm-avoidance baseline.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = f"NOVELTY_ON(w={novelty_bonus_weight})" if novelty_enabled else "NOVELTY_ABLATED"

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        novelty_bonus_weight=novelty_bonus_weight if novelty_enabled else 0.0,
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
        eval_episodes = 2

    print(
        f"\n[V3-EXQ-141] TRAIN {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}"
        f" nav_bias={nav_bias}",
        flush=True,
    )
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )

            # Action selection before record_transition so action tensor is available
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            if z_self_t is not None:
                agent.record_transition(z_self_t, action, latent.z_self.detach().clone())

            # E1 prediction loss; also drives novelty EMA update
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
                if novelty_enabled:
                    agent.e3.update_novelty_ema(float(e1_loss.item()))

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            # Nav bias: with probability nav_bias, override action toward hazard
            # (ensures enough harm contacts for harm_eval supervision)
            if random.random() < nav_bias:
                action = torch.randint(0, ACTION_DIM, (1, ACTION_DIM), dtype=torch.float32)

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} novelty_ema={agent.e3._novelty_ema:.5f}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    action_counts = [0] * ACTION_DIM
    harm_events = 0
    total_steps = 0
    visited_cells: Set[tuple] = set()
    e1_errors: List[float] = []
    novelty_ema_at_eval = float(agent.e3._novelty_ema)

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                    1, world_dim, device=agent.device
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)

                # Capture E1 error for informational purposes
                e1_loss_val = agent.compute_prediction_loss()
                if float(e1_loss_val) > 0.0:
                    e1_errors.append(float(e1_loss_val.item()))

            action_idx = int(action.squeeze().argmax().item())
            action_counts[action_idx] += 1

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            # Track visited grid cells from body_state position estimate
            pos_x = int(obs_dict["body_state"][0] * 10)
            pos_y = int(obs_dict["body_state"][1] * 10)
            visited_cells.add((pos_x, pos_y))
            total_steps += 1

            if done:
                break

    policy_entropy = _action_entropy(action_counts)
    harm_rate = harm_events / max(1, total_steps)
    novel_cell_visits = len(visited_cells)
    mean_e1_error = sum(e1_errors) / max(1, len(e1_errors))

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" entropy={policy_entropy:.4f}"
        f" cells={novel_cell_visits}"
        f" harm_rate={harm_rate:.4f}"
        f" harm_events={harm_events}"
        f" novelty_ema={novelty_ema_at_eval:.5f}"
        f" mean_e1_err={mean_e1_error:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "novelty_enabled": novelty_enabled,
        "policy_entropy": policy_entropy,
        "novel_cell_visits": novel_cell_visits,
        "harm_rate": harm_rate,
        "harm_events": harm_events,
        "novelty_ema_at_eval": novelty_ema_at_eval,
        "mean_e1_prediction_error": mean_e1_error,
        "total_steps": total_steps,
    }


def run(
    seeds: Tuple[int, ...] = (42, 123),
    novelty_bonus_weight: float = NOVELTY_BONUS_WEIGHT,
    warmup_episodes: int = 200,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    nav_bias: float = 0.30,
    dry_run: bool = False,
) -> dict:
    """MECH-111 novelty/curiosity drive: NOVELTY_DRIVE_ON vs NOVELTY_DRIVE_ABLATED.

    Paired discriminative design: each seed runs both conditions (same env, same
    init). NOVELTY_DRIVE_ON uses E1 error EMA as exploration bonus; ABLATED uses
    no bonus. If MECH-111 is correct, ON condition shows higher entropy and cell
    coverage without substantially worse harm avoidance.
    """
    print(
        f"\n[V3-EXQ-141] MECH-111 Novelty Drive Discriminative Pair"
        f" seeds={list(seeds)} novelty_bonus_weight={novelty_bonus_weight}",
        flush=True,
    )

    results_on: List[Dict] = []
    results_abl: List[Dict] = []

    for seed in seeds:
        for novelty_on in [True, False]:
            r = _run_single(
                seed=seed,
                novelty_enabled=novelty_on,
                novelty_bonus_weight=novelty_bonus_weight,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                nav_bias=nav_bias,
                dry_run=dry_run,
            )
            if novelty_on:
                results_on.append(r)
            else:
                results_abl.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    # Per-seed gaps (paired comparison)
    per_seed_ent_gap:  List[float] = []
    per_seed_cell_gap: List[int]   = []
    per_seed_harm_delta: List[float] = []

    for r_on in results_on:
        matching = [r for r in results_abl if r["seed"] == r_on["seed"]]
        if matching:
            r_abl = matching[0]
            per_seed_ent_gap.append(r_on["policy_entropy"] - r_abl["policy_entropy"])
            per_seed_cell_gap.append(r_on["novel_cell_visits"] - r_abl["novel_cell_visits"])
            per_seed_harm_delta.append(r_on["harm_rate"] - r_abl["harm_rate"])

    # Aggregate means
    mean_entropy_on  = _avg(results_on,  "policy_entropy")
    mean_entropy_abl = _avg(results_abl, "policy_entropy")
    mean_cells_on    = _avg(results_on,  "novel_cell_visits")
    mean_cells_abl   = _avg(results_abl, "novel_cell_visits")
    mean_harm_on     = _avg(results_on,  "harm_rate")
    mean_harm_abl    = _avg(results_abl, "harm_rate")

    n_harm_min = min(r["harm_events"] for r in results_on + results_abl)

    # Pre-registered PASS criteria
    # C1: entropy gap >= THRESH_ENT_GAP both seeds
    c1_pass = (
        len(per_seed_ent_gap) > 0
        and all(g >= THRESH_ENT_GAP for g in per_seed_ent_gap)
    )
    # C2: cell coverage gap >= THRESH_CELL_GAP both seeds
    c2_pass = (
        len(per_seed_cell_gap) > 0
        and all(g >= THRESH_CELL_GAP for g in per_seed_cell_gap)
    )
    # C3: harm delta <= THRESH_HARM_DELTA both seeds (drive is harm-safe)
    c3_pass = (
        len(per_seed_harm_delta) > 0
        and all(d <= THRESH_HARM_DELTA for d in per_seed_harm_delta)
    )
    # C4: novelty EMA non-zero in NOVELTY_DRIVE_ON both seeds
    c4_pass = all(r["novelty_ema_at_eval"] > THRESH_NOVELTY_EMA for r in results_on)
    # C5: data quality -- sufficient harm events both conditions both seeds
    c5_pass = n_harm_min >= THRESH_N_HARM

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif (c1_pass or c2_pass) and c3_pass:
        decision = "hybridize"
    elif not c3_pass:
        decision = "retire_ree_claim"
    else:
        decision = "hybridize"

    print(
        f"\n[V3-EXQ-141] Results:"
        f" entropy ON={mean_entropy_on:.4f} ABL={mean_entropy_abl:.4f}"
        f" cells ON={mean_cells_on:.1f} ABL={mean_cells_abl:.1f}"
        f" harm ON={mean_harm_on:.4f} ABL={mean_harm_abl:.4f}",
        flush=True,
    )
    print(
        f"  per_seed_ent_gap={[round(g, 4) for g in per_seed_ent_gap]}"
        f" per_seed_cell_gap={per_seed_cell_gap}"
        f" per_seed_harm_delta={[round(d, 4) for d in per_seed_harm_delta]}"
        f" n_harm_min={n_harm_min}"
        f" decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing = [r["seed"] for r in results_on
                   if (results_abl[results_on.index(r)]["policy_entropy"]
                       if results_on.index(r) < len(results_abl) else 0)
                   > r["policy_entropy"] - THRESH_ENT_GAP]
        failure_notes.append(
            f"C1 FAIL: per-seed entropy_gap {[round(g, 4) for g in per_seed_ent_gap]}"
            f" < {THRESH_ENT_GAP}"
            " -- novelty bonus does not produce measurable entropy increase"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed cell_gap {per_seed_cell_gap}"
            f" < {THRESH_CELL_GAP}"
            " -- novelty drive does not increase novel cell coverage"
        )
    if not c3_pass:
        failing_seeds = [
            results_on[i]["seed"] for i, d in enumerate(per_seed_harm_delta)
            if d > THRESH_HARM_DELTA
        ]
        failure_notes.append(
            f"C3 FAIL: per-seed harm_delta {[round(d, 4) for d in per_seed_harm_delta]}"
            f" > {THRESH_HARM_DELTA} in seeds {failing_seeds}"
            " -- novelty drive causes harm increase; curiosity-surprise not isolated"
        )
    if not c4_pass:
        failing_seeds = [r["seed"] for r in results_on
                         if r["novelty_ema_at_eval"] <= THRESH_NOVELTY_EMA]
        failure_notes.append(
            f"C4 FAIL: novelty_ema <= {THRESH_NOVELTY_EMA} in seeds {failing_seeds}"
            " -- E1 novelty EMA is zero; check E1 prediction loss flow"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_harm_min={n_harm_min} < {THRESH_N_HARM}"
            " -- insufficient harm contacts; increase nav_bias or eval episodes"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            f"MECH-111 SUPPORTED: E1 prediction-error EMA bonus increases exploration"
            f" (entropy and cell coverage up, harm not substantially increased)."
            f" NOVELTY_DRIVE_ON: entropy={mean_entropy_on:.4f}"
            f" cells={mean_cells_on:.1f} harm={mean_harm_on:.4f}."
            f" NOVELTY_DRIVE_ABLATED: entropy={mean_entropy_abl:.4f}"
            f" cells={mean_cells_abl:.1f} harm={mean_harm_abl:.4f}."
            f" per-seed entropy_gap={[round(g, 4) for g in per_seed_ent_gap]}"
            f" cell_gap={per_seed_cell_gap}"
            f" harm_delta={[round(d, 4) for d in per_seed_harm_delta]}."
            f" Supports MECH-111: novelty drive is architecturally grounded;"
            f" information-seeking approach is distinct from alarm-surprise"
            f" (harm not elevated with drive active)."
        )
    elif (c1_pass or c2_pass) and c3_pass:
        interpretation = (
            f"Partial support: directional exploration increase observed (entropy or"
            f" cell coverage) without harm elevation, but below pre-registered"
            f" threshold on one criterion. Consider increasing novelty_bonus_weight"
            f" or warmup episodes. C1={c1_pass} C2={c2_pass} C3={c3_pass}."
        )
    elif not c3_pass:
        interpretation = (
            f"MECH-111 NOT SUPPORTED: novelty drive increases harm exposure."
            f" per-seed harm_delta={[round(d, 4) for d in per_seed_harm_delta]}."
            f" The E1 error EMA does not selectively reward safe-novel states;"
            f" curiosity-surprise and alarm-surprise are not architecturally"
            f" distinguished at this scale. Drive blends with harm-approach signal."
        )
    else:
        interpretation = (
            f"MECH-111 NOT SUPPORTED: novelty bonus does not produce measurable"
            f" exploration increase. E1 error may be too uniform across states,"
            f" or the EMA is not feeding back into E3 selection effectively."
            f" Criteria: C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}"
            f" C5={c5_pass}."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: entropy={r['policy_entropy']:.4f}"
        f" cells={r['novel_cell_visits']} harm_rate={r['harm_rate']:.4f}"
        f" novelty_ema={r['novelty_ema_at_eval']:.5f}"
        for r in results_on
    )
    per_abl_rows = "\n".join(
        f"  seed={r['seed']}: entropy={r['policy_entropy']:.4f}"
        f" cells={r['novel_cell_visits']} harm_rate={r['harm_rate']:.4f}"
        for r in results_abl
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-141 -- MECH-111 Novelty/Curiosity Drive Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-111\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** NOVELTY_DRIVE_ON vs NOVELTY_DRIVE_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps x {steps_per_episode} steps"
        f"  **Eval:** {eval_episodes} eps x {steps_per_episode} steps\n"
        f"**Env:** CausalGridWorld size=10, 5 hazards, 5 resources"
        f" nav_bias={nav_bias}\n"
        f"**novelty_bonus_weight:** {novelty_bonus_weight}\n\n"
        f"## Design\n\n"
        f"MECH-111 asserts E1 prediction-error surprise at moderate magnitudes"
        f" generates intrinsic positive valence (curiosity/novelty drive)."
        f" The experiment compares NOVELTY_DRIVE_ON (E1 EMA bonus active) against"
        f" NOVELTY_DRIVE_ABLATED (no bonus) across {len(seeds)} matched seeds."
        f" Key test: does the drive increase exploration (entropy, cell coverage)"
        f" without increasing harm (curiosity-surprise is safe, not alarm-surprise)?\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: per-seed entropy_gap (ON-ABLATED) >= {THRESH_ENT_GAP} (both seeds)\n"
        f"C2: per-seed cell_gap (ON-ABLATED) >= {THRESH_CELL_GAP} (both seeds)\n"
        f"C3: per-seed harm_delta (ON-ABLATED) <= {THRESH_HARM_DELTA} (both seeds)\n"
        f"C4: novelty_ema_ON > {THRESH_NOVELTY_EMA} both seeds (signal non-zero)\n"
        f"C5: n_harm_min >= {THRESH_N_HARM} both conditions (data quality)\n\n"
        f"## Results\n\n"
        f"| Condition | entropy | novel_cells | harm_rate |\n"
        f"|-----------|---------|-------------|----------|\n"
        f"| NOVELTY_DRIVE_ON      | {mean_entropy_on:.4f}"
        f" | {mean_cells_on:.1f} | {mean_harm_on:.4f} |\n"
        f"| NOVELTY_DRIVE_ABLATED | {mean_entropy_abl:.4f}"
        f" | {mean_cells_abl:.1f} | {mean_harm_abl:.4f} |\n\n"
        f"**per-seed entropy_gap: {[round(g, 4) for g in per_seed_ent_gap]}**\n"
        f"**per-seed cell_gap: {per_seed_cell_gap}**\n"
        f"**per-seed harm_delta: {[round(d, 4) for d in per_seed_harm_delta]}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: entropy_gap >= {THRESH_ENT_GAP} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {[round(g, 4) for g in per_seed_ent_gap]} |\n"
        f"| C2: cell_gap >= {THRESH_CELL_GAP} (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {per_seed_cell_gap} |\n"
        f"| C3: harm_delta <= {THRESH_HARM_DELTA} (both seeds)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {[round(d, 4) for d in per_seed_harm_delta]} |\n"
        f"| C4: novelty_ema_ON > {THRESH_NOVELTY_EMA} (both seeds)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {[round(r['novelty_ema_at_eval'], 7) for r in results_on]} |\n"
        f"| C5: n_harm_min >= {THRESH_N_HARM}"
        f" | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_harm_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed Detail\n\n"
        f"NOVELTY_DRIVE_ON:\n{per_on_rows}\n\n"
        f"NOVELTY_DRIVE_ABLATED:\n{per_abl_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "mean_entropy_on":           float(mean_entropy_on),
        "mean_entropy_ablated":      float(mean_entropy_abl),
        "mean_entropy_gap":          float(mean_entropy_on - mean_entropy_abl),
        "mean_cells_on":             float(mean_cells_on),
        "mean_cells_ablated":        float(mean_cells_abl),
        "mean_cell_gap":             float(mean_cells_on - mean_cells_abl),
        "mean_harm_rate_on":         float(mean_harm_on),
        "mean_harm_rate_ablated":    float(mean_harm_abl),
        "mean_harm_delta":           float(mean_harm_on - mean_harm_abl),
        "per_seed_entropy_gap_min":  float(min(per_seed_ent_gap)) if per_seed_ent_gap else 0.0,
        "per_seed_entropy_gap_max":  float(max(per_seed_ent_gap)) if per_seed_ent_gap else 0.0,
        "per_seed_cell_gap_min":     float(min(per_seed_cell_gap)) if per_seed_cell_gap else 0.0,
        "per_seed_harm_delta_max":   float(max(per_seed_harm_delta)) if per_seed_harm_delta else 0.0,
        "novelty_ema_min":           float(min(r["novelty_ema_at_eval"] for r in results_on)),
        "n_harm_min":                float(n_harm_min),
        "n_seeds":                   float(len(seeds)),
        "novelty_bonus_weight":      float(novelty_bonus_weight),
        "nav_bias":                  float(nav_bias),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--novelty-weight",  type=float, default=NOVELTY_BONUS_WEIGHT)
    parser.add_argument("--warmup",          type=int,   default=200)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--nav-bias",        type=float, default=0.30)
    parser.add_argument("--dry-run",         action="store_true",
                        help="3 warmup + 2 eval eps per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        novelty_bonus_weight=args.novelty_weight,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_per_seed_entropy_gap":    THRESH_ENT_GAP,
        "C2_per_seed_cell_gap":       THRESH_CELL_GAP,
        "C3_per_seed_harm_delta":     THRESH_HARM_DELTA,
        "C4_novelty_ema_min":         THRESH_NOVELTY_EMA,
        "C5_n_harm_min":              THRESH_N_HARM,
    }
    result["seeds"] = list(args.seeds)
    result["conditions"] = ["NOVELTY_DRIVE_ON", "NOVELTY_DRIVE_ABLATED"]
    result["dispatch_mode"] = "discriminative_pair"
    result["backlog_id"] = "EVB-0061"
    result["evidence_class"] = "discriminative_pair"
    result["claim_ids_tested"] = CLAIM_IDS

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

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
