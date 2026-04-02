"""
V3-EXQ-204 -- MECH-104: Volatility Interrupt Discriminative Pair

Claims: MECH-104

MECH-104 hypothesis: Unexpected harm events spike commitment uncertainty
(LC-NE volatility interrupt), enabling de-commitment. The mechanism is:
harm causes large world-forward prediction error -> large EMA update to
_running_variance -> variance may exceed commit_threshold -> de-commitment.

Design: Two matched conditions per seed
  VOLATILITY_ON  (Control):  normal _ema_alpha (0.05), variance responds to errors
  VOLATILITY_OFF (Ablation): _ema_alpha = 0.0, variance frozen (cannot spike)

Per-step variance tracking:
  After each env.step, compute wf_pred = E2.world_forward(z_world_prev, action_prev)
  Record rv_before, call update_running_variance, record rv_after.
  Harm label: harm_signal < 0 at the step that produced the current z_world_curr.

Pre-registered thresholds (discriminative_pair):
  C1: spike_contrast (harm_spike - nonharm_spike) > 0 in >= 2/3 seeds (VOLATILITY_ON)
  C2: all harm-step spikes < 1e-10 in ALL seeds (VOLATILITY_OFF: _ema_alpha=0.0 exact)
  C3: pairwise_delta (ON harm spike - OFF harm spike) > 0 in >= 2/3 seeds
  C4: n_harm_steps > 0 in ALL seeds (sanity: harm actually occurred during eval)
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


EXPERIMENT_TYPE = "v3_exq_204_mech104_surprise_gate_pair"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-104"]

# Pre-registered thresholds
THRESH_C1_CONTRAST   = 0.0    # harm_spike > nonharm_spike (ON condition)
THRESH_C2_FROZEN     = 1e-10  # OFF condition harm spike must be < this
THRESH_C3_PAIRWISE   = 0.0    # ON harm spike > OFF harm spike
THRESH_C4_HARM_STEPS = 1      # at least 1 harm step per seed

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=3,
    hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent_and_env(
    seed: int, self_dim: int, world_dim: int
) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)
    return agent, env


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    lr: float = 1e-3,
) -> Dict:
    """
    Train agent's E2 world-forward model so prediction errors are meaningful.
    Variance updates via update_running_variance during training (standard path).
    """
    agent.train()

    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=lr,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    variance_log: List[float] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates   = agent.generate_trajectories(latent, e1_prior, ticks)
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            variance_log.append(agent.e3._running_variance)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            if len(wf_buf) >= 16:
                k     = min(32, len(wf_buf))
                idxs  = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
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

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] ep {ep+1}/{num_episodes}  running_var={rv:.7f}",
                flush=True,
            )

    return {
        "final_running_variance":      agent.e3._running_variance,
        "mean_variance_final_100eps":  _mean_safe(variance_log[-2000:]),
    }


def _eval_variance_spikes(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    condition_label: str,
    ema_alpha_override: Optional[float] = None,
    rv_init_override: Optional[float] = None,
) -> Dict:
    """
    Eval: per-step variance deltas on harm vs non-harm transitions.

    Pairing: z_world_prev (t-1), action_prev (t-1), z_world_curr (t),
             harm_prev = harm_signal from env.step(action_{t-1}).
    This correctly attributes prediction error to the transition that caused it.

    ema_alpha_override: if set, overrides agent.e3._ema_alpha for this eval.
    rv_init_override:   if set, resets agent.e3._running_variance to this value.
    """
    agent.eval()

    orig_alpha = agent.e3._ema_alpha
    if ema_alpha_override is not None:
        agent.e3._ema_alpha = ema_alpha_override
    if rv_init_override is not None:
        agent.e3._running_variance = rv_init_override

    harm_spike_deltas:    List[float] = []
    nonharm_spike_deltas: List[float] = []
    decommit_events: int = 0
    n_harm_steps:    int = 0
    n_nonharm_steps: int = 0
    rv_trajectory:   List[float] = []

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # reset() may overwrite _running_variance; re-apply override if needed
        if rv_init_override is not None:
            agent.e3._running_variance = rv_init_override
        if ema_alpha_override is not None:
            agent.e3._ema_alpha = ema_alpha_override

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        harm_prev:    Optional[float]        = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            z_world_curr = latent.z_world.detach()
            rv_trajectory.append(agent.e3._running_variance)

            # Per-step variance update using the PREVIOUS transition
            # harm_prev = harm from action_prev (which produced z_world_curr)
            if z_world_prev is not None and action_prev is not None and harm_prev is not None:
                rv_before     = agent.e3._running_variance
                was_committed = rv_before < agent.e3.commit_threshold
                with torch.no_grad():
                    wf_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(
                        (wf_pred - z_world_curr).detach()
                    )
                rv_after      = agent.e3._running_variance
                now_committed = rv_after < agent.e3.commit_threshold
                spike         = rv_after - rv_before

                if was_committed and not now_committed:
                    decommit_events += 1

                if harm_prev < 0:
                    harm_spike_deltas.append(spike)
                    n_harm_steps += 1
                else:
                    nonharm_spike_deltas.append(spike)
                    n_nonharm_steps += 1

            with torch.no_grad():
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            with torch.no_grad():
                action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            z_world_prev = z_world_curr
            action_prev  = action.detach()
            harm_prev    = float(harm_signal)

            if done:
                break

    # Restore original alpha
    agent.e3._ema_alpha = orig_alpha

    mean_harm    = _mean_safe(harm_spike_deltas)
    mean_nonharm = _mean_safe(nonharm_spike_deltas)
    contrast     = mean_harm - mean_nonharm

    print(
        f"\n  [{condition_label}] Variance spike eval:"
        f"\n    harm steps: {n_harm_steps}  nonharm steps: {n_nonharm_steps}"
        f"\n    mean harm spike:    {mean_harm:.9f}"
        f"\n    mean nonharm spike: {mean_nonharm:.9f}"
        f"\n    spike_contrast:     {contrast:.9f}"
        f"\n    decommit_events:    {decommit_events}"
        f"\n    final running_var:  {agent.e3._running_variance:.9f}",
        flush=True,
    )

    return {
        "mean_harm_spike":    mean_harm,
        "mean_nonharm_spike": mean_nonharm,
        "spike_contrast":     contrast,
        "decommit_events":    decommit_events,
        "n_harm_steps":       n_harm_steps,
        "n_nonharm_steps":    n_nonharm_steps,
        "final_rv":           agent.e3._running_variance,
        "mean_rv":            _mean_safe(rv_trajectory),
    }


def run_seed(
    seed:              int,
    warmup_episodes:   int,
    eval_episodes:     int,
    steps_per_episode: int,
    self_dim:  int = 32,
    world_dim: int = 32,
) -> Dict:
    """
    One seed: train a single agent, then eval under VOLATILITY_ON and VOLATILITY_OFF.
    VOLATILITY_OFF resets _running_variance to post-train value, sets _ema_alpha=0.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\n{'='*60}", flush=True)
    print(f"[EXQ-204] Seed {seed} -- Training ({warmup_episodes} eps)", flush=True)
    print('='*60, flush=True)

    agent, env = _make_agent_and_env(seed, self_dim, world_dim)
    print(
        f"  _ema_alpha={agent.e3._ema_alpha:.4f}"
        f"  commit_threshold={agent.e3.commit_threshold:.4f}"
        f"  precision_init={agent.e3._running_variance:.4f}",
        flush=True,
    )

    train_out = _train_agent(
        agent, env, warmup_episodes, steps_per_episode, world_dim
    )
    post_train_rv = train_out["final_running_variance"]
    print(
        f"\n  Post-train running_var={post_train_rv:.7f}",
        flush=True,
    )

    # ── VOLATILITY_ON eval ───────────────────────────────────────────────
    print(f"\n[EXQ-204] Seed {seed} -- Eval VOLATILITY_ON ({eval_episodes} eps)", flush=True)
    result_on = _eval_variance_spikes(
        agent, env,
        eval_episodes, steps_per_episode, world_dim,
        condition_label="VOLATILITY_ON",
        ema_alpha_override=None,    # keep default
        rv_init_override=None,
    )

    # ── VOLATILITY_OFF eval (ablation) ───────────────────────────────────
    # Restore post-train rv, freeze alpha=0.0
    print(f"\n[EXQ-204] Seed {seed} -- Eval VOLATILITY_OFF ({eval_episodes} eps)", flush=True)
    result_off = _eval_variance_spikes(
        agent, env,
        eval_episodes, steps_per_episode, world_dim,
        condition_label="VOLATILITY_OFF",
        ema_alpha_override=0.0,
        rv_init_override=post_train_rv,
    )

    pairwise_delta = result_on["mean_harm_spike"] - result_off["mean_harm_spike"]

    return {
        "seed":               seed,
        "post_train_rv":      post_train_rv,
        # VOLATILITY_ON
        "on_harm_spike":      result_on["mean_harm_spike"],
        "on_nonharm_spike":   result_on["mean_nonharm_spike"],
        "on_spike_contrast":  result_on["spike_contrast"],
        "on_decommit_events": result_on["decommit_events"],
        "on_n_harm_steps":    result_on["n_harm_steps"],
        "on_n_nonharm_steps": result_on["n_nonharm_steps"],
        # VOLATILITY_OFF
        "off_harm_spike":     result_off["mean_harm_spike"],
        "off_spike_contrast": result_off["spike_contrast"],
        "off_decommit_events":result_off["decommit_events"],
        "off_n_harm_steps":   result_off["n_harm_steps"],
        # Pairwise
        "pairwise_delta":     pairwise_delta,
    }


def run(
    seeds:             Optional[List[int]] = None,
    warmup_episodes:   int   = 200,
    eval_episodes:     int   = 50,
    steps_per_episode: int   = 200,
    self_dim:  int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    print(
        f"[V3-EXQ-204] MECH-104: Volatility Interrupt Discriminative Pair\n"
        f"  Seeds: {seeds}  Warmup: {warmup_episodes} eps  Eval: {eval_episodes} eps\n"
        f"  VOLATILITY_ON:  _ema_alpha=default (normal)\n"
        f"  VOLATILITY_OFF: _ema_alpha=0.0 (frozen -- ablation)\n"
        f"  Pre-registered thresholds:\n"
        f"    C1: spike_contrast > {THRESH_C1_CONTRAST} (harm > nonharm, ON, >=2/3 seeds)\n"
        f"    C2: harm_spike_off < {THRESH_C2_FROZEN} (OFF frozen, ALL seeds)\n"
        f"    C3: pairwise_delta > {THRESH_C3_PAIRWISE} (ON > OFF on harm, >=2/3 seeds)\n"
        f"    C4: n_harm_steps >= {THRESH_C4_HARM_STEPS} (harm occurred, ALL seeds)",
        flush=True,
    )

    seed_results = []
    for seed in seeds:
        sr = run_seed(seed, warmup_episodes, eval_episodes, steps_per_episode, self_dim, world_dim)
        seed_results.append(sr)

    n_seeds  = len(seeds)
    majority = max(2, (2 * n_seeds + 2) // 3)   # ceil(2/3 * n_seeds), minimum 2

    c1_seeds = sum(1 for r in seed_results if r["on_spike_contrast"] > THRESH_C1_CONTRAST)
    c2_seeds = sum(1 for r in seed_results if abs(r["off_harm_spike"]) < THRESH_C2_FROZEN)
    c3_seeds = sum(1 for r in seed_results if r["pairwise_delta"] > THRESH_C3_PAIRWISE)
    c4_seeds = sum(1 for r in seed_results if r["on_n_harm_steps"] >= THRESH_C4_HARM_STEPS)

    c1_pass = c1_seeds >= majority
    c2_pass = c2_seeds == n_seeds   # ALL seeds must show frozen (alpha=0 is exact)
    c3_pass = c3_seeds >= majority
    c4_pass = c4_seeds == n_seeds   # sanity: must have harm in ALL seeds

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status       = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not c1_pass:
        contrasts = [r["on_spike_contrast"] for r in seed_results]
        failure_notes.append(
            f"C1 FAIL: spike_contrast > {THRESH_C1_CONTRAST} in {c1_seeds}/{n_seeds} seeds"
            f" (need {majority}); values={[f'{v:.7f}' for v in contrasts]}"
        )
    if not c2_pass:
        off_vals = [r["off_harm_spike"] for r in seed_results]
        failure_notes.append(
            f"C2 FAIL: OFF harm spike not frozen in {c2_seeds}/{n_seeds} seeds"
            f"; values={[f'{v:.2e}' for v in off_vals]}"
        )
    if not c3_pass:
        pairs = [r["pairwise_delta"] for r in seed_results]
        failure_notes.append(
            f"C3 FAIL: pairwise_delta > {THRESH_C3_PAIRWISE} in {c3_seeds}/{n_seeds} seeds"
            f" (need {majority}); values={[f'{v:.7f}' for v in pairs]}"
        )
    if not c4_pass:
        nharm = [r["on_n_harm_steps"] for r in seed_results]
        failure_notes.append(
            f"C4 FAIL: n_harm_steps < {THRESH_C4_HARM_STEPS} in some seeds"
            f"; values={nharm}"
        )

    print(f"\nV3-EXQ-204 verdict: {status}  ({criteria_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # Flat metrics
    metrics: Dict = {
        "c1_seeds_pass": float(c1_seeds),
        "c2_seeds_pass": float(c2_seeds),
        "c3_seeds_pass": float(c3_seeds),
        "c4_seeds_pass": float(c4_seeds),
        "criteria_met":  float(criteria_met),
        "n_seeds":       float(n_seeds),
        "majority_threshold": float(majority),
    }
    for r in seed_results:
        s = r["seed"]
        for key in (
            "post_train_rv", "on_harm_spike", "on_nonharm_spike", "on_spike_contrast",
            "on_decommit_events", "on_n_harm_steps",
            "off_harm_spike", "off_spike_contrast", "off_decommit_events", "off_n_harm_steps",
            "pairwise_delta",
        ):
            metrics[f"seed{s}_{key}"] = float(r[key])

    # Summary markdown table
    rows_seed = ""
    for r in seed_results:
        rows_seed += (
            f"| {r['seed']} | {r['post_train_rv']:.7f}"
            f" | {r['on_harm_spike']:.8f} | {r['on_nonharm_spike']:.8f}"
            f" | {r['on_spike_contrast']:.8f}"
            f" | {r['off_harm_spike']:.2e}"
            f" | {r['pairwise_delta']:.8f}"
            f" | {r['on_decommit_events']} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-204 -- MECH-104: Volatility Interrupt Discriminative Pair

**Status:** {status}
**Claim:** MECH-104 -- unexpected harm events spike commitment uncertainty (LC-NE analog)
**Design:** Matched discriminative pair per seed (VOLATILITY_ON vs VOLATILITY_OFF ablation)
**Warmup:** {warmup_episodes} eps | **Eval:** {eval_episodes} eps | **Seeds:** {seeds}

## Mechanism Under Test

Harm events cause large world-forward prediction error -> _running_variance spikes
via EMA update (alpha * error_var). Ablation: set _ema_alpha=0.0 freezes variance.
Paired comparison per step: harm transitions vs non-harm transitions.
Transition labeling: harm_prev (from action that produced z_world_curr).

## Pre-registered Thresholds

| Criterion | Threshold | Seed requirement |
|-----------|-----------|-----------------|
| C1: spike_contrast (harm-nonharm) | > {THRESH_C1_CONTRAST} | >= {majority}/{n_seeds} seeds |
| C2: OFF harm spike frozen | < {THRESH_C2_FROZEN} | ALL {n_seeds} seeds |
| C3: pairwise_delta (ON-OFF) | > {THRESH_C3_PAIRWISE} | >= {majority}/{n_seeds} seeds |
| C4: n_harm_steps | >= {THRESH_C4_HARM_STEPS} | ALL {n_seeds} seeds |

## Results by Seed

| Seed | post-train rv | ON harm spike | ON nonharm spike | ON contrast | OFF harm spike | Pairwise delta | Decommits (ON) |
|------|-------------|-------------|----------------|-------------|--------------|---------------|---------------|
{rows_seed}

## PASS Criteria

| Criterion | Seeds passing | Required | Result |
|-----------|------------|---------|--------|
| C1 spike_contrast > {THRESH_C1_CONTRAST} | {c1_seeds}/{n_seeds} | >={majority} | {"PASS" if c1_pass else "FAIL"} |
| C2 OFF harm spike frozen | {c2_seeds}/{n_seeds} | =={n_seeds} | {"PASS" if c2_pass else "FAIL"} |
| C3 pairwise_delta > {THRESH_C3_PAIRWISE} | {c3_seeds}/{n_seeds} | >={majority} | {"PASS" if c3_pass else "FAIL"} |
| C4 n_harm_steps >= {THRESH_C4_HARM_STEPS} | {c4_seeds}/{n_seeds} | =={n_seeds} | {"PASS" if c4_pass else "FAIL"} |

Criteria met: {criteria_met}/4 -> **{status}**
{failure_section}
"""

    evidence_direction = (
        "supports" if all_pass
        else ("mixed" if criteria_met >= 2 else "weakens")
    )

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type":    EXPERIMENT_TYPE,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  type=int,   nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup", type=int,   default=200)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps",  type=int,   default=200)
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]     = ts
    result["run_id"]            = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"]= "ree_hybrid_guardrails_v1"
    result["experiment_purpose"]= EXPERIMENT_PURPOSE
    result["claim_ids"]         = CLAIM_IDS

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
