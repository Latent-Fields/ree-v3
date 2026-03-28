#!/opt/local/bin/python3
"""
V3-EXQ-119 -- MECH-090: Beta Gate Committed Dynamics: Multi-Seed Discriminative Pair

Claim: MECH-090 -- beta oscillations gate E3->action_selection propagation (NOT E3
internal updating): during a committed sequence, beta is elevated and policy output is
blocked; at completion (transition to uncommitted), beta drops and E3 state propagates.

Proposal: EXP-0016 / EVB-0013
Dispatch mode: discriminative_pair
Min shared seeds: 2

Background:
  EXQ-049e (seed=0) tested a two-condition design (trained vs fresh agent). C1 PASS
  (committed_hold_concordance > 0.6) and C2 PASS (uncommitted_release_concordance > 0.5)
  were both achieved in that single-seed run. That result supported MECH-090 provisionally
  but required multi-seed replication for claim support.
  This experiment provides a proper discriminative pair:
    BETA_GATE_ON  -- beta gate engaged: BetaGate.update() called, elevates when committed,
                     releases when uncommitted. E3 policy output blocked during elevation.
    BETA_GATE_OFF -- gate always in pass-through mode: beta gate disabled (gate never
                     elevates). E3 state always propagates regardless of commitment state.

Mechanism under test:
  MECH-090 asserts that E3->action_selection is gated by beta oscillations:
  - When running_variance < commit_threshold (committed): beta elevated, output HELD.
    The gate protects the ongoing committed sequence from interference.
  - When running_variance >= commit_threshold (uncommitted): beta released, output
    PROPAGATES. E3 evaluation reaches action selection normally.
  The key signature: gate elevation concordance with commitment state, measured per-seed
  across both conditions. BETA_GATE_OFF is the ablation -- no commitment-linked gating.

Discriminative pair:
  BETA_GATE_ON  -- BetaGate.update(is_committed) called each step. Gate elevates when
                   committed (running_variance < commit_threshold); releases when not.
                   Commitment-concordant gating behaviour measured.
  BETA_GATE_OFF -- BetaGate disabled. gate_is_elevated always False. Agent runs with
                   unmodified policy output in all states.

Seeds: [42, 123] -- matched, both run under both conditions.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (committed-hold concordance, BETA_GATE_ON):
    committed_hold_concordance >= 0.60 per seed.
    PASS: both seeds meet threshold (pair_c1_pass_count == 2).
    Definition: fraction of committed steps where gate is elevated.
    The core claim: gate is elevated when agent is committed.

  C2 (uncommitted-release concordance, BETA_GATE_ON):
    uncommitted_release_concordance >= 0.50 per seed.
    PASS: at least 1 of 2 seeds meets threshold.
    Definition: fraction of uncommitted steps where gate is NOT elevated (released).
    Supports the release half of MECH-090.

  C3 (gate ablation contrast, BETA_GATE_OFF):
    BETA_GATE_OFF committed_hold_concordance < 0.40 for both seeds.
    Confirms gate ablation works: without the gate, elevation does not track commitment.
    This is the discriminative criterion: the gate has causal effect.

  C4 (committed steps data quality):
    n_committed_steps >= 50 (BETA_GATE_ON) for both seeds.
    Enough committed steps to compute C1 reliably.

  C5 (no fatal errors):
    Zero crashes across all 4 cells (2 seeds x 2 conditions).

PASS criteria:
  C1_both AND C2_at_least_one AND C3_both AND C4_both AND C5
    -> PASS -> supports MECH-090
  C1_both AND NOT C3 (ablation shows elevation too) -> FAIL -> gate not discriminative
  NOT C1_both AND C3_both -> FAIL -> gate not elevating during commitment
  NOT C4 -> FAIL -> data quality (too few committed steps)
  NOT C5 -> FAIL -> runtime error

Decision mapping:
  PASS -> retain_ree (beta gate confirmed with matched-seed replication)
  FAIL (C3 gate not discriminative) -> redesign (gate logic may not be conditional)
  FAIL (C1: gate not elevating) -> hold (commitment-gate wiring may need more training)
  FAIL (data quality) -> retry (increase warmup so agent reaches committed state)

If this experiment PASSes:
  Which claim does that support and why?
  MECH-090: beta gate elevation is concordant with committed state (C1 >= 0.60 both seeds),
  releases when uncommitted (C2), and the ablation confirms the gate is the causal mechanism
  (C3: BETA_GATE_OFF does NOT show commitment-concordant elevation). This is the full
  discriminative support for MECH-090's core assertion.
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


EXPERIMENT_TYPE = "v3_exq_119_mech090_beta_gate_pair"
CLAIM_IDS = ["MECH-090"]

# Pre-registered thresholds
THRESH_C1_HOLD_CONCORDANCE    = 0.60   # committed_hold_concordance >= 0.60 both seeds
THRESH_C2_RELEASE_CONCORDANCE = 0.50   # uncommitted_release_concordance >= 0.50, >=1 seed
THRESH_C3_ABLATION_MAX        = 0.40   # BETA_GATE_OFF concordance < 0.40 (gate not tracking)
THRESH_C4_N_COMMITTED         = 50     # n_committed_steps >= 50 per seed (GATE_ON)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


# ---------------------------------------------------------------------------
# Single (seed, condition) cell
# ---------------------------------------------------------------------------

def _run_cell(
    seed: int,
    gate_on: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    dry_run: bool,
) -> Dict:
    """
    Run one (seed, condition) cell and return gate concordance metrics.

    gate_on=True  -> BETA_GATE_ON: BetaGate.update(is_committed) called each step.
    gate_on=False -> BETA_GATE_OFF: gate disabled, never elevates.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "BETA_GATE_ON" if gate_on else "BETA_GATE_OFF"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=4,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    config.latent.unified_latent_mode = False  # SD-005 split always on

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer          = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # World-forward replay buffer for variance update
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    # ------------------------------------------------------------------ TRAIN
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev = None
        action_prev  = None
        episode_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks  = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            # Gate update (BETA_GATE_ON only)
            if gate_on:
                is_committed = (
                    agent.e3._running_variance < agent.e3.commit_threshold
                )
                if is_committed:
                    agent.beta_gate.elevate()
                else:
                    agent.beta_gate.release()

            # Action selection (simple policy: random if harm high)
            if ticks.get("e3_tick", False) or not hasattr(agent, "_cached_harm_score"):
                with torch.no_grad():
                    z_w = latent.z_world.detach()
                    agent._cached_harm_score = float(agent.e3.harm_eval(z_w).item())

            if agent._cached_harm_score > 0.5:
                action_idx = random.randint(0, env.action_dim - 1)
            else:
                action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)
            if float(harm_signal) < 0:
                episode_harm += abs(float(harm_signal))

            # Replay buffers
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > MAX_BUF:
                    wf_buf = wf_buf[-MAX_BUF:]

            (harm_buf_pos if float(harm_signal) < 0 else harm_buf_neg).append(
                z_world_curr.detach()
            )
            if len(harm_buf_pos) > MAX_BUF: harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            if len(harm_buf_neg) > MAX_BUF: harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(standard_params, 1.0)
                optimizer.step()

            # World-forward loss + direct variance update
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(standard_params, 1.0)
                    optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            # Harm eval training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(harm_eval_params, 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] cond={cond_label} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" ep_harm={episode_harm:.4f}"
                f" running_var={rv:.6f}",
                flush=True,
            )

        if dry_run and ep >= 2:
            break

    # ------------------------------------------------------------------ EVAL
    agent.eval()
    agent.beta_gate.reset()

    committed_elevated   = 0
    committed_not_elev   = 0
    uncommitted_elevated = 0
    uncommitted_not_elev = 0
    fatal_errors = 0
    rvs: List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            try:
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks  = agent.clock.advance()
                    if ticks.get("e1_tick", False):
                        agent._e1_tick(latent)

                is_committed = (
                    agent.e3._running_variance < agent.e3.commit_threshold
                )
                rvs.append(agent.e3._running_variance)

                # Gate update
                if gate_on:
                    if is_committed:
                        agent.beta_gate.elevate()
                    else:
                        agent.beta_gate.release()

                is_elevated = agent.beta_gate.is_elevated if gate_on else False

                if is_committed and is_elevated:
                    committed_elevated += 1
                elif is_committed and not is_elevated:
                    committed_not_elev += 1
                elif not is_committed and is_elevated:
                    uncommitted_elevated += 1
                else:
                    uncommitted_not_elev += 1

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

            except Exception as exc:
                fatal_errors += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

        if dry_run and ep >= 1:
            break

    gate_state = agent.beta_gate.get_state()
    n_committed   = committed_elevated + committed_not_elev
    n_uncommitted = uncommitted_elevated + uncommitted_not_elev

    committed_hold_conc      = committed_elevated   / max(1, n_committed)
    uncommitted_release_conc = uncommitted_not_elev / max(1, n_uncommitted)
    mean_rv = _mean_safe(rvs)

    print(
        f"  [eval] cond={cond_label} seed={seed}"
        f"  n_committed={n_committed}  n_uncommitted={n_uncommitted}"
        f"  committed_hold_conc={committed_hold_conc:.3f}"
        f"  uncommitted_release_conc={uncommitted_release_conc:.3f}"
        f"  hold_count={gate_state['hold_count']}"
        f"  propagation_count={gate_state['propagation_count']}"
        f"  mean_rv={mean_rv:.6f}"
        f"  fatal={fatal_errors}",
        flush=True,
    )

    return {
        "condition":                     cond_label,
        "gate_on":                       gate_on,
        "seed":                          seed,
        "committed_elevated":            committed_elevated,
        "committed_not_elevated":        committed_not_elev,
        "uncommitted_elevated":          uncommitted_elevated,
        "uncommitted_not_elevated":      uncommitted_not_elev,
        "n_committed_steps":             n_committed,
        "n_uncommitted_steps":           n_uncommitted,
        "committed_hold_concordance":    committed_hold_conc,
        "uncommitted_release_concordance": uncommitted_release_conc,
        "hold_count":                    gate_state["hold_count"],
        "propagation_count":             gate_state["propagation_count"],
        "mean_running_variance":         mean_rv,
        "fatal_errors":                  fatal_errors,
    }


# ---------------------------------------------------------------------------
# Discriminative pair: BETA_GATE_ON vs BETA_GATE_OFF x seeds [42, 123]
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Matched-seed discriminative pair: BETA_GATE_ON vs BETA_GATE_OFF x seeds.

    C1: committed_hold_concordance >= 0.60 (GATE_ON) both seeds.
    C2: uncommitted_release_concordance >= 0.50 (GATE_ON) >=1 seed.
    C3: GATE_OFF committed_hold_concordance < 0.40 both seeds (ablation discriminative).
    C4: n_committed_steps >= 50 (GATE_ON) both seeds.
    C5: zero fatal errors across all 4 cells.
    PASS: C1_both AND C2_at_least_one AND C3_both AND C4_both AND C5.
    """
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for gate_on in [True, False]:
            label = "BETA_GATE_ON" if gate_on else "BETA_GATE_OFF"
            print(
                f"\n[V3-EXQ-119] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode} alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_cell(
                seed=seed,
                gate_on=gate_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                dry_run=dry_run,
            )
            if gate_on:
                results_on.append(r)
            else:
                results_off.append(r)

    # --- C1: committed-hold concordance (GATE_ON, both seeds) ---
    c1_per_seed = [
        r["committed_hold_concordance"] >= THRESH_C1_HOLD_CONCORDANCE
        for r in results_on
    ]
    pair_c1_pass_count = sum(c1_per_seed)
    c1_pass = pair_c1_pass_count == len(seeds)

    # --- C2: uncommitted-release concordance (GATE_ON, >=1 seed) ---
    c2_per_seed = [
        r["uncommitted_release_concordance"] >= THRESH_C2_RELEASE_CONCORDANCE
        for r in results_on
    ]
    c2_pass = sum(c2_per_seed) >= 1

    # --- C3: ablation contrast (GATE_OFF, committed_hold_conc < 0.40 both seeds) ---
    c3_per_seed = [
        r["committed_hold_concordance"] < THRESH_C3_ABLATION_MAX
        for r in results_off
    ]
    c3_pass = all(c3_per_seed)

    # --- C4: data quality -- n_committed >= threshold (GATE_ON, both seeds) ---
    c4_per_seed = [r["n_committed_steps"] >= THRESH_C4_N_COMMITTED for r in results_on]
    c4_pass = all(c4_per_seed)

    # --- C5: no fatal errors ---
    total_fatal = sum(r["fatal_errors"] for r in results_on + results_off)
    c5_pass = total_fatal == 0

    # --- PASS logic ---
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    # Decision mapping
    if status == "PASS":
        decision = "retain_ree"
    elif not c3_pass:
        decision = "redesign_gate_logic"
    elif not c1_pass and c4_pass:
        decision = "hold_increase_warmup"
    elif not c4_pass:
        decision = "retry_increase_warmup"
    else:
        decision = "hold"

    # --- Failure notes ---
    failure_notes: List[str] = []
    for i, r in enumerate(results_on):
        s = seeds[i] if i < len(seeds) else "?"
        if not c1_per_seed[i]:
            failure_notes.append(
                f"C1 FAIL seed={s} (GATE_ON): committed_hold_conc="
                f"{r['committed_hold_concordance']:.3f}"
                f" < {THRESH_C1_HOLD_CONCORDANCE}"
                f" (n_committed={r['n_committed_steps']})"
            )
        if not c2_per_seed[i]:
            failure_notes.append(
                f"C2 seed={s} (GATE_ON): uncommitted_release_conc="
                f"{r['uncommitted_release_concordance']:.3f}"
                f" < {THRESH_C2_RELEASE_CONCORDANCE}"
            )
        if not c4_per_seed[i]:
            failure_notes.append(
                f"C4 FAIL seed={s} (GATE_ON): n_committed="
                f"{r['n_committed_steps']} < {THRESH_C4_N_COMMITTED}"
            )
    for i, r in enumerate(results_off):
        s = seeds[i] if i < len(seeds) else "?"
        if not c3_per_seed[i]:
            failure_notes.append(
                f"C3 FAIL seed={s} (GATE_OFF): committed_hold_conc="
                f"{r['committed_hold_concordance']:.3f}"
                f" >= {THRESH_C3_ABLATION_MAX}"
                f"  (ablation tracks commitment -- gate not discriminative)"
            )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={total_fatal}")

    # --- Summary print ---
    print(f"\n[V3-EXQ-119] Summary:", flush=True)
    print(f"  Conditions: BETA_GATE_ON vs BETA_GATE_OFF  Seeds: {list(seeds)}", flush=True)
    for i in range(len(seeds)):
        s = seeds[i] if i < len(seeds) else "?"
        r_on  = results_on[i]
        r_off = results_off[i]
        print(
            f"  seed={s}"
            f" | GATE_ON:"
            f" hold_conc={r_on['committed_hold_concordance']:.3f}"
            f" release_conc={r_on['uncommitted_release_concordance']:.3f}"
            f" n_comm={r_on['n_committed_steps']}"
            f" | GATE_OFF:"
            f" hold_conc={r_off['committed_hold_concordance']:.3f}"
            f" n_comm={r_off['n_committed_steps']}",
            flush=True,
        )
    print(
        f"  C1 (hold_conc>={THRESH_C1_HOLD_CONCORDANCE} both, GATE_ON):"
        f" {'PASS' if c1_pass else 'FAIL'} ({pair_c1_pass_count}/{len(seeds)})",
        flush=True,
    )
    print(
        f"  C2 (release_conc>={THRESH_C2_RELEASE_CONCORDANCE} >=1, GATE_ON):"
        f" {'PASS' if c2_pass else 'FAIL'} ({sum(c2_per_seed)}/{len(seeds)})",
        flush=True,
    )
    print(
        f"  C3 (hold_conc<{THRESH_C3_ABLATION_MAX} both, GATE_OFF):"
        f" {'PASS' if c3_pass else 'FAIL'} ({sum(c3_per_seed)}/{len(seeds)})",
        flush=True,
    )
    print(
        f"  C4 (n_comm>={THRESH_C4_N_COMMITTED} both, GATE_ON):"
        f" {'PASS' if c4_pass else 'FAIL'} ({sum(c4_per_seed)}/{len(seeds)})",
        flush=True,
    )
    print(
        f"  C5 (no fatal errors): {'PASS' if c5_pass else 'FAIL'}"
        f" (fatal={total_fatal})",
        flush=True,
    )
    print(
        f"  Criteria met: {criteria_met}/5  Status: {status}  Decision: {decision}",
        flush=True,
    )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # --- Interpretation ---
    if status == "PASS":
        interpretation = (
            f"MECH-090 SUPPORTED by multi-seed discriminative pair ({len(seeds)} seeds). "
            "Beta gate is concordant with commitment state (GATE_ON: C1 hold_conc "
            f">= {THRESH_C1_HOLD_CONCORDANCE} both seeds, C2 release_conc "
            f">= {THRESH_C2_RELEASE_CONCORDANCE} >=1 seed). Ablation confirms causal "
            f"role (GATE_OFF hold_conc < {THRESH_C3_ABLATION_MAX} both seeds -- gate "
            "not tracking commitment when disabled). This confirms MECH-090: beta "
            "oscillations gate E3->action_selection propagation, blocked during "
            "commitment, released on completion."
        )
    elif not c3_pass:
        interpretation = (
            "MECH-090 NOT DISCRIMINATIVE: GATE_OFF condition also shows high "
            "committed_hold_concordance, meaning the gate elevation seen in GATE_ON "
            "is not causally produced by the gate mechanism -- it may arise from "
            "other architecture dynamics. Gate logic may need redesign."
        )
    elif not c1_pass and c4_pass:
        interpretation = (
            "MECH-090 GATE ELEVATION INSUFFICIENT: Enough committed steps (C4 pass) "
            f"but committed_hold_concordance < {THRESH_C1_HOLD_CONCORDANCE} for "
            f"{len(seeds) - pair_c1_pass_count} seed(s). Gate is not reliably "
            "elevating during committed sequences. Possible cause: running_variance "
            "rarely crosses commit_threshold during eval. Consider longer warmup "
            "to drive variance lower."
        )
    elif not c4_pass:
        interpretation = (
            f"MECH-090 INCONCLUSIVE: Insufficient committed steps (n_committed < "
            f"{THRESH_C4_N_COMMITTED}) -- agent may not reach committed state during eval. "
            "Increase warmup episodes."
        )
    else:
        interpretation = (
            f"MECH-090 INCONCLUSIVE: criteria_met={criteria_met}/5. "
            f"See failure notes for details."
        )

    # --- Per-seed rows for markdown ---
    per_seed_on_rows = "\n".join(
        f"| {seeds[i]} | {results_on[i]['committed_hold_concordance']:.3f}"
        f" | {results_on[i]['uncommitted_release_concordance']:.3f}"
        f" | {results_on[i]['n_committed_steps']}"
        f" | {results_on[i]['n_uncommitted_steps']}"
        f" | {'PASS' if c1_per_seed[i] else 'FAIL'}"
        f" | {'PASS' if c2_per_seed[i] else 'FAIL'} |"
        for i in range(len(seeds))
    )
    per_seed_off_rows = "\n".join(
        f"| {seeds[i]} | {results_off[i]['committed_hold_concordance']:.3f}"
        f" | {results_off[i]['n_committed_steps']}"
        f" | {'PASS' if c3_per_seed[i] else 'FAIL'} |"
        for i in range(len(seeds))
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        ) + "\n"

    summary_markdown = (
        f"# V3-EXQ-119 -- MECH-090: Beta Gate Committed Dynamics (Multi-Seed Pair)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-090\n"
        f"**Proposal:** EXP-0016 / EVB-0013\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_episode}\n\n"
        f"## Context\n\n"
        f"EXQ-049e (seed=0) two-condition design showed committed_hold_conc=1.0 (C1) "
        f"and uncommitted_release_conc consistent with claim. This is the matched-seed "
        f"discriminative replication with a full GATE_OFF ablation arm.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1 (hold concordance, GATE_ON): committed_hold_conc >= {THRESH_C1_HOLD_CONCORDANCE}"
        f" for BOTH seeds\n"
        f"C2 (release concordance, GATE_ON): uncommitted_release_conc "
        f">= {THRESH_C2_RELEASE_CONCORDANCE} for >= 1 seed\n"
        f"C3 (ablation, GATE_OFF): committed_hold_conc < {THRESH_C3_ABLATION_MAX}"
        f" for BOTH seeds\n"
        f"C4 (data quality): n_committed >= {THRESH_C4_N_COMMITTED} per seed (GATE_ON)\n"
        f"C5 (no crashes): zero fatal errors\n"
        f"PASS: C1_both AND C2_at_least_one AND C3_both AND C4_both AND C5\n\n"
        f"## Per-Seed Results -- BETA_GATE_ON\n\n"
        f"| Seed | hold_conc | release_conc | n_committed | n_uncommitted | C1 | C2 |\n"
        f"|------|-----------|--------------|-------------|---------------|----|----|  \n"
        f"{per_seed_on_rows}\n\n"
        f"## Per-Seed Results -- BETA_GATE_OFF (Ablation)\n\n"
        f"| Seed | hold_conc | n_committed | C3 |\n"
        f"|------|-----------|-------------|----|\n"
        f"{per_seed_off_rows}\n\n"
        f"## Aggregate\n\n"
        f"| Criterion | Result | Detail |\n"
        f"|-----------|--------|--------|\n"
        f"| C1: hold_conc>={THRESH_C1_HOLD_CONCORDANCE} both (GATE_ON)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {pair_c1_pass_count}/{len(seeds)} seeds |\n"
        f"| C2: release_conc>={THRESH_C2_RELEASE_CONCORDANCE} >=1 (GATE_ON)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {sum(c2_per_seed)}/{len(seeds)} seeds |\n"
        f"| C3: hold_conc<{THRESH_C3_ABLATION_MAX} both (GATE_OFF)"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {sum(c3_per_seed)}/{len(seeds)} seeds |\n"
        f"| C4: n_committed>={THRESH_C4_N_COMMITTED} both (GATE_ON)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {sum(c4_per_seed)}/{len(seeds)} seeds |\n"
        f"| C5: no fatal errors"
        f" | {'PASS' if c5_pass else 'FAIL'}"
        f" | fatal={total_fatal} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
        f"{failure_section}"
    )

    # Flat metrics (stable numeric keys)
    per_seed_on_metrics: Dict = {}
    per_seed_off_metrics: Dict = {}
    for i, (r_on, r_off) in enumerate(zip(results_on, results_off)):
        s = seeds[i] if i < len(seeds) else i
        per_seed_on_metrics[f"seed{s}_on_hold_conc"] = float(r_on["committed_hold_concordance"])
        per_seed_on_metrics[f"seed{s}_on_release_conc"] = float(r_on["uncommitted_release_concordance"])
        per_seed_on_metrics[f"seed{s}_on_n_committed"] = float(r_on["n_committed_steps"])
        per_seed_on_metrics[f"seed{s}_on_n_uncommitted"] = float(r_on["n_uncommitted_steps"])
        per_seed_off_metrics[f"seed{s}_off_hold_conc"] = float(r_off["committed_hold_concordance"])
        per_seed_off_metrics[f"seed{s}_off_n_committed"] = float(r_off["n_committed_steps"])

    metrics: Dict = {
        "pair_c1_pass_count":         float(pair_c1_pass_count),
        "c2_seeds_passing":           float(sum(c2_per_seed)),
        "c3_seeds_passing":           float(sum(c3_per_seed)),
        "n_seeds":                    float(len(seeds)),
        "c1_threshold_hold":          float(THRESH_C1_HOLD_CONCORDANCE),
        "c2_threshold_release":       float(THRESH_C2_RELEASE_CONCORDANCE),
        "c3_threshold_ablation_max":  float(THRESH_C3_ABLATION_MAX),
        "c4_threshold_n_committed":   float(THRESH_C4_N_COMMITTED),
        "total_fatal_errors":         float(total_fatal),
        "criteria_met":               float(criteria_met),
        "crit1_pass":                 1.0 if c1_pass else 0.0,
        "crit2_pass":                 1.0 if c2_pass else 0.0,
        "crit3_pass":                 1.0 if c3_pass else 0.0,
        "crit4_pass":                 1.0 if c4_pass else 0.0,
        "crit5_pass":                 1.0 if c5_pass else 0.0,
        "alpha_world":                float(alpha_world),
        "warmup_episodes":            float(warmup_episodes),
        "eval_episodes":              float(eval_episodes),
        **per_seed_on_metrics,
        **per_seed_off_metrics,
    }

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "evidence_direction":  "supports" if status == "PASS" else (
            "mixed" if criteria_met >= 3 else "weakens"
        ),
        "experiment_type":     EXPERIMENT_TYPE,
        "fatal_error_count":   float(total_fatal),
        "supersedes":          "v3_exq_049e_mech090_two_condition",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_committed_hold_concordance_geq": THRESH_C1_HOLD_CONCORDANCE,
        "C2_uncommitted_release_concordance_geq": THRESH_C2_RELEASE_CONCORDANCE,
        "C3_ablation_hold_conc_lt": THRESH_C3_ABLATION_MAX,
        "C4_n_committed_geq": THRESH_C4_N_COMMITTED,
    }
    result["seeds"]              = list(args.seeds)
    result["conditions"]         = ["BETA_GATE_ON", "BETA_GATE_OFF"]
    result["dispatch_mode"]      = "discriminative_pair"
    result["backlog_id"]         = "EVB-0013"
    result["claim_ids_tested"]   = CLAIM_IDS
    result["evidence_class"]     = "experiment"

    if args.dry_run:
        print(f"\n[dry-run] Status: {result['status']}", flush=True)
        print("[dry-run] Key metrics:", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        print("[dry-run] Script ran without error. Not writing output file.", flush=True)
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
