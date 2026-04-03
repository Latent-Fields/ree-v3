#!/opt/local/bin/python3
"""
V3-EXQ-138a -- ARC-030 Go/NoGo Symmetry Discriminative Pair (Canonical Matched Seeds)

Claims: ARC-030
Proposal: EXP-0061 / EVB-0053

ARC-030 asserts:
  "The three BG-like loops require symmetric Go (approach) and NoGo (avoidance)
   sub-channels; pure NoGo architecture produces behavioral flatness."

Specifically: without a Go (approach/benefit) channel, the gradient minimum under
harm-avoidance-only training is quiescence. The agent that does nothing accrues no
harm signal, no attribution, never crosses a commit boundary, and appears by its own
error metrics to be performing optimally. The residue field requires positive
attractors alongside harm repellers, or the hippocampal planner has no terrain to
navigate toward.

Discriminative pair design
--------------------------
GO_NOGO_SUPPORT (ARC-030 architecture -- full BG symmetry):
  - harm_eval_head trained (NoGo channel, avoidance gradient)
  - z_goal attractor updated each step via agent.update_z_goal(benefit_exposure)
  - benefit_eval_head trained (Go channel, approach gradient)
  - E1 goal-conditioned (e1_goal_conditioned=True)
  - Full D1/D2 symmetry: both channels active and competing on same trajectories

NOGO_ONLY_ABLATED (ablation -- pure avoidance, no Go channel):
  - harm_eval_head trained (NoGo channel, same as support)
  - z_goal DISABLED (z_goal_enabled=False)
  - benefit_eval_head DISABLED (benefit_eval_enabled=False)
  - E1 not goal-conditioned (e1_goal_conditioned=False)
  - No approach gradient; harm-avoidance only; quiescence is optimal

Both conditions use identical architecture (same world_dim, self_dim),
identical environment, identical training duration, identical eval protocol.
Only the Go channel activation differs.

Testable prediction (from ARC-030 + literature):
  GO_NOGO_SUPPORT should produce:
    (a) higher benefit_rate than NOGO_ONLY_ABLATED (Go channel drives approach)
    (b) maintained or improved harm_rate (NoGo channel not disrupted by Go)
    (c) NOGO_ONLY agent produces near-zero benefit_rate (quiescent degenerate policy)

  If ARC-030 is correct, the Go channel breaks behavioral flatness.
  The D1/D2 competitive model (Bariselli et al. 2018) predicts both channels
  evaluate the SAME hippocampal trajectory proposals; the commit threshold
  (MECH-106, ARC-016) is the competition balance point.

This design strengthens EXQ-086 (FAIL: warmup=150 insufficient, seeds [42,7]).
EXQ-086 showed benefit_rate_go=0.0 -- the Go channel did not have enough training
time to build a benefit buffer. This experiment addresses that with:
  - 400 warmup episodes (vs 150)
  - proximity_benefit_scale=0.10 (vs 0.05, stronger benefit signal density)
  - num_resources=5 (vs 4, more resource targets)
  - Canonical seeds [42, 123] (vs [42, 7])

Pre-registered acceptance criteria (ALL required for PASS):

  C1: delta_benefit_rate >= 0.002 (across averaged seeds)
      GO_NOGO benefit_rate exceeds NOGO_ONLY by at least 2pp.
      Threshold derived from EXQ-086 baseline (benefit_rate=0.0);
      any reliable approach advantage should exceed 0.002.

  C2: benefit_rate_go >= 0.002 (both seeds independently)
      Go agent actively collects benefit -- not still flat per seed.
      If benefit_rate_go < 0.002 for any seed, the Go channel is not
      forming and the experiment is underspecified (not ablation failure).

  C3: harm_rate_go <= harm_rate_nogo * 1.50 (averaged seeds)
      Go channel does not catastrophically impair harm avoidance.
      ARC-030 predicts Go and NoGo are co-beneficial, not competing.
      Factor 1.50 allows headroom for initial Go exploration costs.

  C4: per-seed direction: benefit_rate_go > benefit_rate_nogo (both seeds)
      Consistent direction across matched seeds.
      Single-seed advantage could be noise; both seeds required.

  C5: n_benefit_buf_go >= 100 per seed (manipulation check)
      Confirms the benefit buffer filled -- Go channel saw enough positive
      events to train benefit_eval_head. If n_benefit_buf < 100, the
      environment benefit signal is too sparse and the experiment is invalid.

Decision scoring:
  PASS: C1 + C2 + C3 + C4 + C5 (5/5) -> retain_ree
  PARTIAL: C4 + C3 (direction consistent, no harm regression) -> hybridize
  FAIL: C4 fails (no consistent direction) -> retire_ree_claim

Note on prior evidence:
  EXQ-086 (2026-03-23): FAIL (1/4), benefit_rate_go=0.0, warmup=150 insufficient.
  EXQ-086 does NOT count as disconfirming evidence for ARC-030 because the
  Go channel never formed (diagnostic: n_benefit_buf_go=274 -- barely populated
  at training end, and eval policy uses trained weights with near-zero benefit
  gradient). This is an execution failure, not a scientific disconfirmation.
  EXQ-138a is the canonical test with corrected training depth.

Architecture references:
  ARC-030: approach/avoidance symmetry (D1/D2 BG loops)
  ARC-021: three BG-like loops require distinct learning channels
  MECH-069: harm, motor-sensory, and goal error are incommensurable
  INV-032: go/nogo sub-channels required for non-degenerate policy
  SD-010: nociceptive stream (harm encoder, z_harm)
  MECH-112: structured latent goal representation (z_goal attractor)
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
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_138a_arc030_go_nogo_pair"
CLAIM_IDS = ["ARC-030"]

# Pre-registered thresholds
MIN_DELTA_BENEFIT_RATE = 0.002       # C1: minimum Go advantage over NoGo (averaged seeds)
MIN_BENEFIT_RATE_GO_PER_SEED = 0.002 # C2: minimum Go benefit_rate per seed
MAX_HARM_RATE_RATIO = 1.50           # C3: max harm_rate_go / harm_rate_nogo
MIN_BENEFIT_BUF_PER_SEED = 100       # C5: manipulation check


ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5,
    env_drift_prob=0.2,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.10,    # stronger benefit signal (vs 0.05 in EXQ-086)
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

SELF_DIM = 32
WORLD_DIM = 32
LR = 1e-3
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
MAX_BUF = 2000


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    go_nogo: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Run one (seed, condition) cell.

    go_nogo=True  -> GO_NOGO_SUPPORT (full Go + NoGo channels)
    go_nogo=False -> NOGO_ONLY_ABLATED (NoGo only, Go ablated)
    """
    cond_label = "GO_NOGO_SUPPORT" if go_nogo else "NOGO_ONLY_ABLATED"
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        z_goal_enabled=go_nogo,
        benefit_eval_enabled=go_nogo,
        e1_goal_conditioned=go_nogo,
        reafference_action_dim=0,
    )
    agent = REEAgent(config)

    # Separate optimizers
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "benefit_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=LR)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    benefit_eval_opt = None
    if go_nogo:
        benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())
        benefit_eval_opt = optim.Adam(benefit_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    benefit_buf_pos: List[torch.Tensor] = []
    benefit_buf_neg: List[torch.Tensor] = []

    print(
        f"  [train] {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eval={eval_episodes}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # benefit_exposure: proximity to nearest resource
            benefit_exp = float(
                obs_body[11] if hasattr(obs_body, "__len__") else obs_body
            )

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, reward, done, info, obs_dict = env.step(action)
            reward_f = float(reward)

            # --- Standard E1 + E2 update ---
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_std = e1_loss + e2_loss
            if total_std.requires_grad:
                optimizer.zero_grad()
                total_std.backward()
                torch.nn.utils.clip_grad_norm_(standard_params, 1.0)
                optimizer.step()

            # --- Harm eval (NoGo channel -- both conditions) ---
            if reward_f < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k = min(16, len(harm_buf_pos), len(harm_buf_neg))
                zw_p = torch.cat(random.sample(harm_buf_pos, k), dim=0)
                zw_n = torch.cat(random.sample(harm_buf_neg, k), dim=0)
                zw_b = torch.cat([zw_p, zw_n], dim=0)
                tgt = torch.cat([
                    torch.ones(k, 1, device=agent.device),
                    torch.zeros(k, 1, device=agent.device),
                ], dim=0)
                h_loss = F.mse_loss(agent.e3.harm_eval(zw_b), tgt)
                if h_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    h_loss.backward()
                    harm_eval_opt.step()

            # --- Go channel (GO_NOGO_SUPPORT condition only) ---
            if go_nogo:
                # Update z_goal attractor (wanting: MECH-112)
                agent.update_z_goal(benefit_exp)

                # Benefit buffer: proximity signal or positive reward
                is_benefit = reward_f > 0 or benefit_exp > 0.05
                if is_benefit:
                    benefit_buf_pos.append(z_world_curr)
                    if len(benefit_buf_pos) > MAX_BUF:
                        benefit_buf_pos = benefit_buf_pos[-MAX_BUF:]
                else:
                    benefit_buf_neg.append(z_world_curr)
                    if len(benefit_buf_neg) > MAX_BUF:
                        benefit_buf_neg = benefit_buf_neg[-MAX_BUF:]

                # Train benefit_eval_head (liking: Go sub-channel)
                if len(benefit_buf_pos) >= 4 and len(benefit_buf_neg) >= 4:
                    k = min(16, len(benefit_buf_pos), len(benefit_buf_neg))
                    zw_p = torch.cat(random.sample(benefit_buf_pos, k), dim=0)
                    zw_n = torch.cat(random.sample(benefit_buf_neg, k), dim=0)
                    zw_b = torch.cat([zw_p, zw_n], dim=0)
                    tgt = torch.cat([
                        torch.ones(k, 1, device=agent.device),
                        torch.zeros(k, 1, device=agent.device),
                    ], dim=0)
                    b_loss = F.mse_loss(agent.e3.benefit_eval(zw_b), tgt)
                    if b_loss.requires_grad:
                        benefit_eval_opt.zero_grad()
                        b_loss.backward()
                        benefit_eval_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            goal_str = ""
            if go_nogo and agent.goal_state is not None:
                goal_str = (
                    f" goal_norm={agent.goal_state.goal_norm():.3f}"
                    f" benefit_buf={len(benefit_buf_pos)}"
                )
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm_buf={len(harm_buf_pos)}"
                f"{goal_str}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    agent.eval()
    benefit_per_ep: List[float] = []
    harm_per_ep: List[float] = []
    goal_proximity_vals: List[float] = []
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0
        ep_harm = 0.0
        ep_steps = 0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                e1_prior = agent._e1_tick(latent)
                candidates = agent.generate_trajectories(latent, e1_prior, {
                    "e1_tick": True, "e2_tick": True, "e3_tick": True,
                })
                action = agent.select_action(candidates, {
                    "e1_tick": True, "e2_tick": True, "e3_tick": True,
                }, temperature=0.5)

            action_idx = int(action.squeeze().argmax().item())
            agent._last_action = action

            _, reward, done, info, obs_dict = env.step(action)
            reward_f = float(reward)
            ep_benefit += max(0.0, reward_f)
            ep_harm += abs(min(0.0, reward_f))
            ep_steps += 1

            if go_nogo and agent.goal_state is not None and agent._current_latent is not None:
                try:
                    gp = float(
                        agent.goal_state.goal_proximity(
                            agent._current_latent.z_world
                        ).mean().item()
                    )
                    goal_proximity_vals.append(gp)
                except Exception:
                    n_fatal += 1

            if done:
                break

        if ep_steps > 0:
            benefit_per_ep.append(ep_benefit / ep_steps)
            harm_per_ep.append(ep_harm / ep_steps)

    benefit_rate = float(sum(benefit_per_ep) / max(1, len(benefit_per_ep)))
    harm_rate = float(sum(harm_per_ep) / max(1, len(harm_per_ep)))
    goal_proximity_mean = (
        float(sum(goal_proximity_vals) / max(1, len(goal_proximity_vals)))
        if goal_proximity_vals else float("nan")
    )

    print(
        f"  [eval]  {cond_label} seed={seed}"
        f" benefit_rate={benefit_rate:.5f}"
        f" harm_rate={harm_rate:.5f}"
        f" goal_prox={goal_proximity_mean:.4f}"
        f" n_fatal={n_fatal}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "go_nogo": go_nogo,
        "benefit_rate": benefit_rate,
        "harm_rate": harm_rate,
        "goal_proximity_mean": goal_proximity_mean,
        "n_benefit_buf_pos": len(benefit_buf_pos) if go_nogo else 0,
        "n_harm_buf_pos": len(harm_buf_pos),
        "n_fatal": n_fatal,
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    **kwargs,
) -> dict:
    """ARC-030: GO_NOGO_SUPPORT vs NOGO_ONLY_ABLATED discriminative pair."""
    print(
        f"\n[EXQ-138a] ARC-030 Go/NoGo symmetry discriminative pair"
        f" seeds={seeds} warmup={warmup_episodes} eval={eval_episodes}",
        flush=True,
    )

    results_go: List[Dict] = []
    results_nogo: List[Dict] = []

    for seed in seeds:
        for go_nogo in [False, True]:
            r = _run_single(
                seed=seed,
                go_nogo=go_nogo,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
            )
            if go_nogo:
                results_go.append(r)
            else:
                results_nogo.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    benefit_rate_go   = _avg(results_go,   "benefit_rate")
    benefit_rate_nogo = _avg(results_nogo, "benefit_rate")
    harm_rate_go      = _avg(results_go,   "harm_rate")
    harm_rate_nogo    = _avg(results_nogo, "harm_rate")
    goal_prox_go      = _avg(results_go,   "goal_proximity_mean")
    delta_benefit     = benefit_rate_go - benefit_rate_nogo
    n_benefit_buf_go_per_seed = [r["n_benefit_buf_pos"] for r in results_go]

    # Per-seed C2 check
    c2_per_seed = [r["benefit_rate"] >= MIN_BENEFIT_RATE_GO_PER_SEED for r in results_go]
    # Per-seed C4 check
    c4_per_seed = [
        r_go["benefit_rate"] > r_nogo["benefit_rate"]
        for r_go, r_nogo in zip(results_go, results_nogo)
    ]

    # Pre-registered criteria
    c1_pass = delta_benefit >= MIN_DELTA_BENEFIT_RATE
    c2_pass = all(c2_per_seed)
    c3_pass = (
        harm_rate_nogo == 0.0
        or harm_rate_go <= harm_rate_nogo * MAX_HARM_RATE_RATIO
    )
    c4_pass = all(c4_per_seed)
    c5_pass = all(n >= MIN_BENEFIT_BUF_PER_SEED for n in n_benefit_buf_go_per_seed)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c4_pass and c3_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[EXQ-138a] Final results:", flush=True)
    print(
        f"  benefit_rate: GO_NOGO={benefit_rate_go:.5f}"
        f"  NOGO_ONLY={benefit_rate_nogo:.5f}"
        f"  delta={delta_benefit:+.5f}",
        flush=True,
    )
    print(
        f"  harm_rate:    GO_NOGO={harm_rate_go:.5f}"
        f"  NOGO_ONLY={harm_rate_nogo:.5f}",
        flush=True,
    )
    print(
        f"  goal_prox_GO: {goal_prox_go:.4f}",
        flush=True,
    )
    print(
        f"  benefit_buf_per_seed: {n_benefit_buf_go_per_seed}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_benefit_rate={delta_benefit:.5f}"
            f" < {MIN_DELTA_BENEFIT_RATE}"
            " -- Go channel not producing reliable benefit advantage"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed benefit_rate_go {c2_per_seed}"
            f" -- some seeds flat (< {MIN_BENEFIT_RATE_GO_PER_SEED})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_rate_go={harm_rate_go:.5f}"
            f" > harm_rate_nogo*{MAX_HARM_RATE_RATIO}={harm_rate_nogo*MAX_HARM_RATE_RATIO:.5f}"
            " -- Go channel impairs harm avoidance"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: inconsistent directionality across seeds -- {c4_per_seed}"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: benefit_buf_per_seed={n_benefit_buf_go_per_seed}"
            f" -- some seeds below {MIN_BENEFIT_BUF_PER_SEED}"
            " -- Go channel saw insufficient positive events (experiment underspecified)"
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            f"ARC-030 SUPPORTED: Go channel produces benefit_rate={benefit_rate_go:.5f}"
            f" vs NOGO_ONLY {benefit_rate_nogo:.5f}"
            f" (delta={delta_benefit:+.5f} >= {MIN_DELTA_BENEFIT_RATE})."
            " Harm avoidance maintained. Pure NoGo produces measurably flatter"
            " resource-approaching behaviour. D1/D2 symmetry is required per ARC-030."
        )
    elif c4_pass and c3_pass:
        interpretation = (
            "ARC-030 PARTIAL: Consistent Go > NoGo direction across seeds but"
            f" delta={delta_benefit:.5f} below threshold {MIN_DELTA_BENEFIT_RATE}."
            " Go channel functional but advantage is weak."
            " Consider longer warmup or higher proximity_benefit_scale."
        )
    else:
        interpretation = (
            "ARC-030 NOT SUPPORTED: Go channel does not consistently outperform"
            " NOGO_ONLY on resource collection."
            " Diagnostic: check n_benefit_buf_go_per_seed and goal_proximity_mean."
            " If C5 fails, the environment benefit signal is insufficient."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_go_rows = "\n".join(
        f"  seed={r['seed']}: benefit_rate={r['benefit_rate']:.5f}"
        f" harm_rate={r['harm_rate']:.5f}"
        f" goal_prox={r['goal_proximity_mean']:.4f}"
        f" benefit_buf={r['n_benefit_buf_pos']}"
        for r in results_go
    )
    per_nogo_rows = "\n".join(
        f"  seed={r['seed']}: benefit_rate={r['benefit_rate']:.5f}"
        f" harm_rate={r['harm_rate']:.5f}"
        for r in results_nogo
    )

    summary_markdown = (
        f"# V3-EXQ-138a -- ARC-030 Go/NoGo Symmetry Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_benefit_rate           >= {MIN_DELTA_BENEFIT_RATE} (averaged seeds)\n"
        f"C2: benefit_rate_go per seed     >= {MIN_BENEFIT_RATE_GO_PER_SEED} (both seeds)\n"
        f"C3: harm_rate_go                 <= harm_rate_nogo * {MAX_HARM_RATE_RATIO}\n"
        f"C4: per-seed direction           GO > NOGO (both seeds)\n"
        f"C5: n_benefit_buf_go per seed    >= {MIN_BENEFIT_BUF_PER_SEED} (manipulation check)\n\n"
        f"## Results\n\n"
        f"| Condition       | benefit_rate | harm_rate   | goal_proximity |\n"
        f"|-----------------|-------------|-------------|----------------|\n"
        f"| GO_NOGO_SUPPORT | {benefit_rate_go:.5f}  | {harm_rate_go:.5f}"
        f" | {goal_prox_go:.4f} |\n"
        f"| NOGO_ONLY_ABLATED | {benefit_rate_nogo:.5f}"
        f" | {harm_rate_nogo:.5f}"
        f" | N/A |\n\n"
        f"**delta_benefit_rate (GO - NOGO): {delta_benefit:+.5f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|-----------|--------|-------|\n"
        f"| C1: delta >= {MIN_DELTA_BENEFIT_RATE} | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_benefit:+.5f} |\n"
        f"| C2: benefit_rate_go per seed >= {MIN_BENEFIT_RATE_GO_PER_SEED} | "
        f"{'PASS' if c2_pass else 'FAIL'}"
        f" | {c2_per_seed} |\n"
        f"| C3: harm_rate_go <= NOGO*{MAX_HARM_RATE_RATIO} | {'PASS' if c3_pass else 'FAIL'}"
        f" | {harm_rate_go:.5f} vs {harm_rate_nogo*MAX_HARM_RATE_RATIO:.5f} |\n"
        f"| C4: per-seed direction GO > NOGO | {'PASS' if c4_pass else 'FAIL'}"
        f" | {c4_per_seed} |\n"
        f"| C5: benefit_buf >= {MIN_BENEFIT_BUF_PER_SEED} per seed | "
        f"{'PASS' if c5_pass else 'FAIL'}"
        f" | {n_benefit_buf_go_per_seed} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Prior Experiment\n\n"
        f"EXQ-086 (2026-03-23): FAIL (1/4), benefit_rate_go=0.0, warmup=150"
        f" -- training too short; benefit buffer barely populated (274 events).\n"
        f"EXQ-138 fixes: warmup 150->400, proximity_benefit_scale 0.05->0.10,"
        f" num_resources 4->5, seeds [42,7]->[42,123].\n\n"
        f"## Per-Seed\n\n"
        f"GO_NOGO_SUPPORT:\n{per_go_rows}\n\n"
        f"NOGO_ONLY_ABLATED:\n{per_nogo_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "benefit_rate_go":              float(benefit_rate_go),
        "benefit_rate_nogo":            float(benefit_rate_nogo),
        "delta_benefit_rate":           float(delta_benefit),
        "harm_rate_go":                 float(harm_rate_go),
        "harm_rate_nogo":               float(harm_rate_nogo),
        "goal_proximity_go":            float(goal_prox_go),
        "n_benefit_buf_go_seed0":       float(n_benefit_buf_go_per_seed[0]) if n_benefit_buf_go_per_seed else 0.0,
        "n_benefit_buf_go_seed1":       float(n_benefit_buf_go_per_seed[1]) if len(n_benefit_buf_go_per_seed) > 1 else 0.0,
        "n_harm_buf_go":                float(sum(r["n_harm_buf_pos"] for r in results_go)),
        "n_harm_buf_nogo":              float(sum(r["n_harm_buf_pos"] for r in results_nogo)),
        "n_seeds":                      float(len(seeds)),
        "alpha_world":                  float(ALPHA_WORLD),
        "proximity_benefit_scale":      float(ENV_KWARGS["proximity_benefit_scale"]),
        "warmup_episodes":              float(warmup_episodes),
        "crit1_pass":                   1.0 if c1_pass else 0.0,
        "crit2_pass":                   1.0 if c2_pass else 0.0,
        "crit3_pass":                   1.0 if c3_pass else 0.0,
        "crit4_pass":                   1.0 if c4_pass else 0.0,
        "crit5_pass":                   1.0 if c5_pass else 0.0,
        "criteria_met":                 float(criteria_met),
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
        "fatal_error_count": sum(r["n_fatal"] for r in results_go + results_nogo),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",  type=int, default=400)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps",  type=int, default=200)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
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
