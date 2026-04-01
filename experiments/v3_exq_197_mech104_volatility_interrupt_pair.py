#!/opt/local/bin/python3
"""
V3-EXQ-197 -- MECH-104: Volatility Interrupt Discriminative Pair (Matched-Seed)

Claims: MECH-104
Proposal: EXP-0041 (backlog EVB-0041), also EXP-0086 Route-2 validation

MECH-104 asserts that unexpected harm events spike commitment uncertainty
(LC-NE volatility interrupt via beta_variance / running_variance), enabling
de-commitment. This is the norepinephrine-driven surprise interrupt: NE release
is triggered by unexpected outcomes, not routine harm or baseline jitter.

Context:
  EXQ-049e PASS: Route-1 validated -- jitter noise floor prevents permanent
  variance collapse and enables de-commitment.
  EXQ-061  PASS: Route-1 -- any harm contact triggers variance spike (coarse).
  EXQ-064  PASS: Route-2 -- surprise gate raises variance selectively on
  unexpected harm; 5/5 criteria met. Single seed only.
  EXQ-126  PASS: Matched-seed pair (2 seeds), 6/6 criteria. Confirmed
  discriminative delta between SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED.

  Problem: EXQ-049e/061/062b/064 were not matched-seed discriminative pairs.
  EXQ-126 used only 2 seeds. This experiment provides the definitive matched-seed
  pair with 3 seeds [42, 7, 13] and pre-registered thresholds for final MECH-104
  evidence (Route-2 harm-surprise triggering volatility spike).

Design -- SURPRISE_GATE_ACTIVE vs SURPRISE_GATE_ABLATED x 3 matched seeds:

  For each seed, train one agent (400 warmup episodes to committed state).
  Then evaluate 2 conditions on the SAME trained agent:

  Condition A -- SURPRISE_GATE_ACTIVE (harm-surprise gate drives beta_variance spike)
    When committed AND actual_harm < HARM_CONTACT_THRESHOLD:
      surprise = |actual_harm - predicted_harm|
      if surprise > SURPRISE_THRESHOLD:
        running_variance += SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
    Harm events classified as "unexpected" (spike fired) or "expected" (no spike).

  Condition B -- SURPRISE_GATE_ABLATED (gate disabled, only baseline jitter noise floor)
    Same trained agent weights, same eval env seed sequence. Surprise is computed
    and classified identically, but no variance impulse is applied. Isolates the
    gate's causal contribution.

Pre-registered PASS criteria (ALL must hold across ALL 3 seeds):

  C1: ACTIVE delta_var_unexpected >= 0.005 (each seed)
      Surprise gate raises variance on unexpected harm. Replicates EXQ-064 C1.

  C2: ACTIVE delta_var_expected < 0.002 (each seed)
      Gate is selective -- does NOT spike on expected harm.

  C3: ABLATED delta_var_unexpected < 0.002 (each seed)
      Ablated baseline stays flat on unexpected harm -- confirms variance rise
      in ACTIVE is gate-specific, not an artifact of harm contact.

  C4: (ACTIVE delta_var_unexpected) - (ABLATED delta_var_unexpected) >= 0.004 (each seed)
      Discriminative criterion: gate produces meaningfully more variance on
      unexpected harm than ablated condition.

  C5: n_unexpected_harm_ACTIVE >= 10 (each seed)
      Sufficient unexpected harm events for reliable measurement.

  C6: No fatal errors across all conditions and seeds.

Decision scoring:
  6/6 all seeds -> PASS, supports, retain_ree
  5/6           -> FAIL, mixed, inconclusive
  4/6           -> FAIL, mixed, inconclusive
  <=3/6         -> FAIL, weakens, retire_ree_claim
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_197_mech104_volatility_interrupt_pair"
CLAIM_IDS = ["MECH-104"]

# Pre-registered thresholds
THRESH_C1 = 0.005   # ACTIVE delta_var_unexpected >= 0.005 (each seed)
THRESH_C2 = 0.002   # ACTIVE delta_var_expected < 0.002 (each seed)
THRESH_C3 = 0.002   # ABLATED delta_var_unexpected < 0.002 (each seed)
THRESH_C4 = 0.004   # (ACTIVE - ABLATED) delta_var_unexpected >= 0.004 (each seed)
THRESH_C5 = 10      # n_unexpected_harm_ACTIVE >= 10 (each seed)

HARM_CONTACT_THRESHOLD = -0.01  # harm_signal below this = harm contact event
SPIKE_MAGNITUDE = 0.05
SURPRISE_THRESHOLD = 0.02

MATCHED_SEEDS = [42, 7, 13]

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    env: CausalGridWorldV2,
) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(config)


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    seed: int,
) -> Dict:
    """Train agent until running_variance collapses (committed state)."""
    agent.train()
    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000:
                harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000:
                harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train seed={seed}] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    return {"final_running_variance": float(agent.e3._running_variance)}


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    gate_active: bool,
    label: str,
    initial_variance: float,
) -> Dict:
    """Eval SURPRISE_GATE_ACTIVE or SURPRISE_GATE_ABLATED on same trained agent.

    gate_active=True:  surprise gate fires on unexpected committed harm steps.
    gate_active=False: ablation -- gate disabled, variance unchanged on harm steps.

    Both conditions record variance before/after harm events, classified
    unexpected vs expected, with identical surprise computation.
    """
    agent.eval()

    var_before_unexpected: List[float] = []
    var_after_unexpected: List[float] = []
    var_before_expected: List[float] = []
    var_after_expected: List[float] = []

    n_surprise_spikes = 0
    n_committed_harm = 0
    n_uncommitted_harm = 0
    fatal = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = initial_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Query predicted harm BEFORE env step (needed for surprise)
            with torch.no_grad():
                theta_z = agent.theta_buffer.summary()
                predicted_harm = float(agent.e3.harm_eval(theta_z).item())

            # Commitment state and variance BEFORE step
            is_committed_pre = agent.e3._running_variance < agent.e3.commit_threshold
            variance_pre = float(agent.e3._running_variance)

            try:
                with torch.no_grad():
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                flat_obs, actual_harm, done, info, obs_dict = env.step(action)

                # Process harm contact events
                if actual_harm < HARM_CONTACT_THRESHOLD:
                    surprise = abs(actual_harm - predicted_harm)
                    is_unexpected = surprise > SURPRISE_THRESHOLD

                    if is_committed_pre:
                        n_committed_harm += 1
                        # Gate: apply variance impulse only in ACTIVE condition
                        if gate_active and is_unexpected:
                            impulse = SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
                            if impulse > 0:
                                agent.e3._running_variance += impulse
                                n_surprise_spikes += 1
                    else:
                        n_uncommitted_harm += 1

                    variance_post = float(agent.e3._running_variance)

                    if is_unexpected:
                        var_before_unexpected.append(variance_pre)
                        var_after_unexpected.append(variance_post)
                    else:
                        var_before_expected.append(variance_pre)
                        var_after_expected.append(variance_post)

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_var_before_unexpected = _mean_safe(var_before_unexpected)
    mean_var_after_unexpected = _mean_safe(var_after_unexpected)
    mean_var_before_expected = _mean_safe(var_before_expected)
    mean_var_after_expected = _mean_safe(var_after_expected)

    delta_unexpected = mean_var_after_unexpected - mean_var_before_unexpected
    delta_expected = mean_var_after_expected - mean_var_before_expected

    print(
        f"\n  [{label}] gate_active={gate_active}"
        f"\n    n_unexpected_harm={len(var_before_unexpected)}"
        f"  n_expected_harm={len(var_before_expected)}"
        f"  n_surprise_spikes={n_surprise_spikes}"
        f"\n    delta_var_unexpected={delta_unexpected:.6f}"
        f"  delta_var_expected={delta_expected:.6f}"
        f"\n    var_before_unexpected={mean_var_before_unexpected:.6f}"
        f"  var_after_unexpected={mean_var_after_unexpected:.6f}"
        f"\n    committed_harm_steps={n_committed_harm}"
        f"  uncommitted_harm_steps={n_uncommitted_harm}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "n_unexpected_harm": len(var_before_unexpected),
        "n_expected_harm": len(var_before_expected),
        "n_surprise_spikes": n_surprise_spikes,
        "mean_var_before_unexpected": mean_var_before_unexpected,
        "mean_var_after_unexpected": mean_var_after_unexpected,
        "delta_var_unexpected": delta_unexpected,
        "mean_var_before_expected": mean_var_before_expected,
        "mean_var_after_expected": mean_var_after_expected,
        "delta_var_expected": delta_expected,
        "n_committed_harm": n_committed_harm,
        "n_uncommitted_harm": n_uncommitted_harm,
        "fatal_errors": fatal,
    }


def _run_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
) -> Dict:
    """Run one seed: train, then eval ACTIVE and ABLATED conditions."""
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-197] Seed {seed}", flush=True)
    print('=' * 60, flush=True)

    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = _make_agent(seed, self_dim, world_dim, alpha_world, env)

    train_out = _train_agent(
        agent, env, warmup_episodes, steps_per_episode, world_dim, seed
    )
    initial_variance = train_out["final_running_variance"]

    print(
        f"\n  Post-train running_variance={initial_variance:.6f}"
        f"  commit_threshold={agent.e3.commit_threshold:.4f}"
        f"  committed={initial_variance < agent.e3.commit_threshold}",
        flush=True,
    )

    if initial_variance >= agent.e3.commit_threshold:
        print(
            "  WARNING: agent not in committed state after training. "
            "Surprise gate will not fire (only fires on committed steps).",
            flush=True,
        )

    # Snapshot env RNG state after training so both conditions see identical layouts
    import copy
    rng_state_post_train = copy.deepcopy(env._rng)

    # Condition A: SURPRISE_GATE_ACTIVE
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-197] Seed {seed} -- SURPRISE_GATE_ACTIVE", flush=True)
    print('=' * 60, flush=True)
    result_active = _eval_condition(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        gate_active=True,
        label=f"ACTIVE seed={seed}",
        initial_variance=initial_variance,
    )

    # Restore env RNG so ABLATED sees the same episode sequence
    env._rng = rng_state_post_train

    # Condition B: SURPRISE_GATE_ABLATED
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-197] Seed {seed} -- SURPRISE_GATE_ABLATED", flush=True)
    print('=' * 60, flush=True)
    result_ablated = _eval_condition(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        gate_active=False,
        label=f"ABLATED seed={seed}",
        initial_variance=initial_variance,
    )

    return {
        "seed": seed,
        "initial_variance": initial_variance,
        "active": result_active,
        "ablated": result_ablated,
    }


def run(
    seeds: List[int] = None,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = list(MATCHED_SEEDS)

    if dry_run:
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 10

    print(
        f"[V3-EXQ-197] MECH-104: Volatility Interrupt Discriminative Pair\n"
        f"  Conditions: SURPRISE_GATE_ACTIVE vs SURPRISE_GATE_ABLATED\n"
        f"  Seeds: {seeds} ({len(seeds)} seeds x 2 conditions = {len(seeds)*2} cells)\n"
        f"  SPIKE_MAGNITUDE={SPIKE_MAGNITUDE}  SURPRISE_THRESHOLD={SURPRISE_THRESHOLD}\n"
        f"  HARM_CONTACT_THRESHOLD={HARM_CONTACT_THRESHOLD}\n"
        f"  Pre-registered: C1>={THRESH_C1}, C2<{THRESH_C2}, C3<{THRESH_C3},"
        f" C4>={THRESH_C4}, C5>={THRESH_C5}\n"
        f"  Warmup={warmup_episodes} eps  Eval={eval_episodes} eps"
        f"  Steps={steps_per_episode}  alpha_world={alpha_world}",
        flush=True,
    )

    results_by_seed: List[Dict] = []
    for seed in seeds:
        seed_result = _run_seed(
            seed=seed,
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            alpha_world=alpha_world,
        )
        results_by_seed.append(seed_result)

    # -------------------------------------------------------------------------
    # Per-seed criterion evaluation
    # -------------------------------------------------------------------------
    c1_per_seed = []
    c2_per_seed = []
    c3_per_seed = []
    c4_per_seed = []
    c5_per_seed = []

    for sr in results_by_seed:
        act = sr["active"]
        abl = sr["ablated"]
        delta_act = act["delta_var_unexpected"]
        delta_abl = abl["delta_var_unexpected"]
        disc_delta = delta_act - delta_abl

        c1_per_seed.append(delta_act >= THRESH_C1)
        c2_per_seed.append(act["delta_var_expected"] < THRESH_C2)
        c3_per_seed.append(delta_abl < THRESH_C3)
        c4_per_seed.append(disc_delta >= THRESH_C4)
        c5_per_seed.append(act["n_unexpected_harm"] >= THRESH_C5)

    # All criteria must hold across ALL seeds
    c1_pass = all(c1_per_seed)
    c2_pass = all(c2_per_seed)
    c3_pass = all(c3_per_seed)
    c4_pass = all(c4_per_seed)
    c5_pass = all(c5_per_seed)
    c6_pass = all(
        sr["active"]["fatal_errors"] + sr["ablated"]["fatal_errors"] == 0
        for sr in results_by_seed
    )

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
        evidence_direction = "supports"
    elif criteria_met >= 4:
        decision = "inconclusive"
        evidence_direction = "mixed"
    else:
        decision = "retire_ree_claim"
        evidence_direction = "weakens"

    failure_notes = []
    for i, sr in enumerate(results_by_seed):
        seed = sr["seed"]
        act = sr["active"]
        abl = sr["ablated"]
        delta_act = act["delta_var_unexpected"]
        delta_abl = abl["delta_var_unexpected"]
        disc = delta_act - delta_abl
        if not c1_per_seed[i]:
            failure_notes.append(
                f"C1 FAIL seed={seed}: ACTIVE delta_var_unexpected="
                f"{delta_act:.6f} < {THRESH_C1}"
            )
        if not c2_per_seed[i]:
            failure_notes.append(
                f"C2 FAIL seed={seed}: ACTIVE delta_var_expected="
                f"{act['delta_var_expected']:.6f} >= {THRESH_C2}"
            )
        if not c3_per_seed[i]:
            failure_notes.append(
                f"C3 FAIL seed={seed}: ABLATED delta_var_unexpected="
                f"{delta_abl:.6f} >= {THRESH_C3}"
            )
        if not c4_per_seed[i]:
            failure_notes.append(
                f"C4 FAIL seed={seed}: discriminative_delta="
                f"{disc:.6f} < {THRESH_C4}"
            )
        if not c5_per_seed[i]:
            failure_notes.append(
                f"C5 FAIL seed={seed}: n_unexpected_harm_ACTIVE="
                f"{act['n_unexpected_harm']} < {THRESH_C5}"
            )

    if not c6_pass:
        total_fatal = sum(
            sr["active"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
            for sr in results_by_seed
        )
        failure_notes.append(f"C6 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-197 verdict: {status}  ({criteria_met}/6)", flush=True)
    print(f"  decision: {decision}", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # -------------------------------------------------------------------------
    # Diagnostics (not PASS/FAIL)
    # -------------------------------------------------------------------------
    for sr in results_by_seed:
        act = sr["active"]
        abl = sr["ablated"]
        delta_act = act["delta_var_unexpected"]
        delta_abl = abl["delta_var_unexpected"]
        ratio = (delta_act / delta_abl) if abs(delta_abl) > 1e-9 else float("inf")
        print(
            f"  [D] seed={sr['seed']}"
            f"  spikes/unexpected={act['n_surprise_spikes']}/{act['n_unexpected_harm']}"
            f"  effect_ratio={ratio:.2f}"
            f"  init_var={sr['initial_variance']:.6f}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Build flat metrics dict (per-seed prefixed)
    # -------------------------------------------------------------------------
    metrics: Dict[str, float] = {}
    for sr in results_by_seed:
        seed = sr["seed"]
        act = sr["active"]
        abl = sr["ablated"]
        pfx = f"s{seed}"
        metrics[f"{pfx}_active_n_unexpected_harm"] = float(act["n_unexpected_harm"])
        metrics[f"{pfx}_active_n_expected_harm"] = float(act["n_expected_harm"])
        metrics[f"{pfx}_active_n_surprise_spikes"] = float(act["n_surprise_spikes"])
        metrics[f"{pfx}_active_delta_var_unexpected"] = float(act["delta_var_unexpected"])
        metrics[f"{pfx}_active_delta_var_expected"] = float(act["delta_var_expected"])
        metrics[f"{pfx}_active_var_before_unexpected"] = float(act["mean_var_before_unexpected"])
        metrics[f"{pfx}_active_var_after_unexpected"] = float(act["mean_var_after_unexpected"])
        metrics[f"{pfx}_active_n_committed_harm"] = float(act["n_committed_harm"])
        metrics[f"{pfx}_ablated_n_unexpected_harm"] = float(abl["n_unexpected_harm"])
        metrics[f"{pfx}_ablated_delta_var_unexpected"] = float(abl["delta_var_unexpected"])
        metrics[f"{pfx}_ablated_delta_var_expected"] = float(abl["delta_var_expected"])
        disc_delta = act["delta_var_unexpected"] - abl["delta_var_unexpected"]
        metrics[f"{pfx}_discriminative_delta"] = float(disc_delta)
        metrics[f"{pfx}_initial_variance"] = float(sr["initial_variance"])
        metrics[f"{pfx}_fatal_errors"] = float(
            act["fatal_errors"] + abl["fatal_errors"]
        )

    metrics["crit1_pass"] = 1.0 if c1_pass else 0.0
    metrics["crit2_pass"] = 1.0 if c2_pass else 0.0
    metrics["crit3_pass"] = 1.0 if c3_pass else 0.0
    metrics["crit4_pass"] = 1.0 if c4_pass else 0.0
    metrics["crit5_pass"] = 1.0 if c5_pass else 0.0
    metrics["crit6_pass"] = 1.0 if c6_pass else 0.0
    metrics["criteria_met"] = float(criteria_met)
    metrics["fatal_error_count"] = float(sum(
        sr["active"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
        for sr in results_by_seed
    ))

    # -------------------------------------------------------------------------
    # Summary markdown
    # -------------------------------------------------------------------------
    seed_rows_active = "\n".join(
        f"| {sr['seed']} | {sr['active']['n_unexpected_harm']} "
        f"| {sr['active']['n_surprise_spikes']} "
        f"| {sr['active']['delta_var_unexpected']:.6f} "
        f"| {sr['active']['delta_var_expected']:.6f} |"
        for sr in results_by_seed
    )
    seed_rows_ablated = "\n".join(
        f"| {sr['seed']} | {sr['ablated']['n_unexpected_harm']} "
        f"| {sr['ablated']['delta_var_unexpected']:.6f} "
        f"| {sr['ablated']['delta_var_expected']:.6f} |"
        for sr in results_by_seed
    )
    seed_rows_disc = "\n".join(
        f"| {sr['seed']} | "
        f"{sr['active']['delta_var_unexpected'] - sr['ablated']['delta_var_unexpected']:.6f} |"
        for sr in results_by_seed
    )

    failure_section = ""
    if failure_notes:
        failure_section = (
            "\n## Failure Notes\n\n"
            + "\n".join(f"- {n}" for n in failure_notes)
        )

    summary_markdown = f"""# V3-EXQ-197 -- MECH-104: Volatility Interrupt Discriminative Pair

**Status:** {status}
**Claims:** MECH-104
**Decision:** {decision}
**Design:** SURPRISE_GATE_ACTIVE vs SURPRISE_GATE_ABLATED x seeds {[sr['seed'] for sr in results_by_seed]}
**SPIKE_MAGNITUDE:** {SPIKE_MAGNITUDE}  |  **SURPRISE_THRESHOLD:** {SURPRISE_THRESHOLD}
**Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Steps:** {steps_per_episode}

## Design Rationale

MECH-104 (LC-NE volatility interrupt) predicts that unexpected harm events spike
commitment uncertainty (running_variance) via a surprise gate. This is the Route-2
validation: norepinephrine release is driven by unexpected outcomes, not routine harm
or background jitter.

Prior evidence: EXQ-049e (Route-1 jitter), EXQ-061 (any-harm spike), EXQ-064 (Route-2
single seed 5/5), EXQ-126 (matched-pair 2 seeds 6/6). All PASS but lacked the
definitive 3-seed matched-seed discriminative pair with pre-registered thresholds.

This experiment provides the final matched-seed pair:
(1) Three seeds [42, 7, 13] for cross-seed replication.
(2) SURPRISE_GATE_ACTIVE vs SURPRISE_GATE_ABLATED on the SAME trained agent per seed.
(3) Pre-registered numeric thresholds for all 6 criteria.

## SURPRISE_GATE_ACTIVE Results

| seed | n_unexpected | n_spikes | delta_var_unexpected | delta_var_expected |
|------|-------------|---------|---------------------|--------------------|
{seed_rows_active}

## SURPRISE_GATE_ABLATED Results

| seed | n_unexpected | delta_var_unexpected | delta_var_expected |
|------|-------------|---------------------|-------------------|
{seed_rows_ablated}

## Discriminative Delta (ACTIVE - ABLATED, delta_var_unexpected)

| seed | discriminative_delta |
|------|---------------------|
{seed_rows_disc}

## PASS Criteria (pre-registered, all must hold across all 3 seeds)

| Criterion | Result | Values |
|-----------|--------|--------|
| C1: ACTIVE delta_var_unexpected >= {THRESH_C1} (each seed) | {"PASS" if c1_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['active']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C2: ACTIVE delta_var_expected < {THRESH_C2} (each seed) | {"PASS" if c2_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['active']['delta_var_expected']:.6f}" for sr in results_by_seed)} |
| C3: ABLATED delta_var_unexpected < {THRESH_C3} (each seed) | {"PASS" if c3_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['ablated']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C4: (ACTIVE-ABLATED) delta >= {THRESH_C4} (each seed) | {"PASS" if c4_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['active']['delta_var_unexpected']-sr['ablated']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C5: n_unexpected_harm_ACTIVE >= {THRESH_C5} (each seed) | {"PASS" if c5_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['active']['n_unexpected_harm']}" for sr in results_by_seed)} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {int(metrics['fatal_error_count'])} |

Criteria met: {criteria_met}/6 -> **{status}**

## Decision Scoring

| Criteria met | Decision |
|--------------|----------|
| 6/6 | retain_ree (supports MECH-104) |
| 4-5/6 | inconclusive (mixed evidence) |
| <=3/6 | retire_ree_claim (weakens MECH-104) |

**This run:** {criteria_met}/6 -> **{decision}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "decision": decision,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(metrics["fatal_error_count"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(MATCHED_SEEDS))
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
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
    print(f"Decision: {result['decision']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}", flush=True)
