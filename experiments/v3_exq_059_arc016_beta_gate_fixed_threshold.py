"""
V3-EXQ-059 — ARC-016 / MECH-057b / MECH-090: Beta Gate with Fixed Commit Threshold

Claims: ARC-016, MECH-057b, MECH-090

Context (2026-03-20):
  EXQ-048 and EXQ-049 both FAILED on C1 (committed_step_count == 0).
  Root cause: commit_threshold was 0.003, but actual running_variance converges to ~0.33
  in trained environments (EXQ-038). The threshold was 25,000x below actual values, so
  `running_variance < commit_threshold` was NEVER true → result.committed never fired →
  beta_gate.elevate() was never called → hold_rate_during_committed was undefined (0/0).

  Fix (2026-03-20): E3Config.commitment_threshold recalibrated to 0.40.
  - Trained environments: running_variance ~0.33 < 0.40 → commits
  - Untrained environments: running_variance ~0.50 ≥ 0.40 → stays uncommitted
  - This allows ARC-016 (dynamic precision gating) and MECH-090 (beta gate) to exercise.

  ARC-016 semantics: precision gate fires when agent has "settled" into a stable
  world model (variance reduced by training). Commitment represents the transition
  from exploratory to reliable forward-model use.

  MECH-090: BetaGate elevates during committed trajectory execution, blocking
  policy output propagation. Releases at E3 tick boundary (trajectory completion).

  MECH-057b: The gate opens specifically at completion, not initiation.

Bug fixes applied (2026-03-20, second run):
  Bug 1: agent.e3.get_diagnostics() does not exist → replaced with
         agent.e3.get_commitment_state() which returns running_variance, precision,
         commit_threshold, committed_now.  This caused 84 fatal errors in the first
         run, aborting the entire try block (including all accounting) on every e3_tick.

  Bug 2: _running_variance never updated during training (chicken-and-egg).
         _running_variance starts at precision_init=0.5 > commit_threshold=0.40, so
         result.committed is always False → _committed_trajectory always None →
         post_action_update() never updates _running_variance.  Fix: explicitly call
         agent.e3.update_running_variance(prediction_error) in the training loop
         after computing the world_forward loss.

  Bug 3: beta_gate.release() never called in eval.  MECH-057b says the gate releases
         at trajectory completion (E3 tick boundary).  Fix: call release() at the
         start of each e3_tick, before the new commitment decision.

  Bug 4: is_committed used agent._committed_candidates is not None, which is True
         after the very first generate_trajectories() call and never False again.
         Fix: is_committed = agent.e3._running_variance < agent.e3.commit_threshold.

Design:
  Training: 400 episodes, full pipeline (same as EXQ-048)
  Eval: 50 episodes x 200 steps
  Same env: CausalGridWorldV2, 12x12, 4 hazards

PASS criteria:
  C1: committed_step_count >= 10        (ARC-016: commitment gate fires with corrected threshold)
  C2: committed_and_elevated > 0        (MECH-090: beta_gate.elevate() called when committed)
  C3: hold_rate_during_committed > 0.3  (MECH-057b: gate holds during committed steps)
  C4: gate_release_events > 0           (MECH-057b: gate releases at trajectory completion)
  C5: calibration_gap_approach > 0.0   (E3 still functional)
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

EXPERIMENT_TYPE = "v3_exq_060_arc016_beta_gate_fixed_threshold"
CLAIM_IDS = ["ARC-016", "MECH-057b", "MECH-090"]

WARMUP_EPISODES = 400
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200
SEED            = 0
ALPHA_WORLD     = 0.9
LR              = 1e-3
WORLD_DIM       = 32


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent_and_env(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=2,
        proximity_harm_scale=0.05, use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    # commitment_threshold is already 0.40 in E3Config default (fixed 2026-03-20)
    agent = REEAgent(cfg)
    print(
        f"[EXQ-057] commit_threshold={cfg.e3.commitment_threshold}  "
        f"precision_init={cfg.e3.precision_init}",
        flush=True,
    )
    return agent, env, cfg


def _train(agent: REEAgent, env: CausalGridWorldV2, n_episodes: int):
    opt_e1   = optim.Adam(list(agent.e1.parameters()), lr=LR)
    opt_wf   = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    opt_harm = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple] = []

    for ep in range(n_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
                # MECH-090: elevate beta gate if committed
                if result.committed:
                    agent.beta_gate.elevate()
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # E2 world_forward buffer
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            # Harm eval buffers
            if harm_signal < 0:
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                opt_e1.zero_grad(); e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                opt_e1.step()

            # E2 world_forward loss
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zt_b  = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zt_b)
                if wf_loss.requires_grad:
                    opt_wf.zero_grad(); wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    opt_wf.step()
                # Bug 2 fix: update E3 running_variance from world_forward error.
                # _running_variance starts at precision_init=0.5 and is only updated
                # via post_action_update() → but that requires _committed_trajectory,
                # which requires result.committed=True, which requires
                # running_variance < commit_threshold — a chicken-and-egg deadlock.
                # Directly updating from the wf error breaks the cycle.
                with torch.no_grad():
                    wf_err = (wf_pred.detach() - zt_b).detach()
                    agent.e3.update_running_variance(wf_err)

            # E3 harm_eval loss (balanced batch)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                tgt = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                harm_loss = F.mse_loss(agent.e3.harm_eval(zw_b), tgt)
                if harm_loss.requires_grad:
                    opt_harm.zero_grad(); harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    opt_harm.step()

            z_world_prev = z_world_curr
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == n_episodes - 1:
            print(f"  [train] ep {ep+1}/{n_episodes}", flush=True)


def _eval(agent: REEAgent, env: CausalGridWorldV2) -> Dict:
    """Track beta gate state and ARC-016 commitment step-by-step."""
    agent.eval()
    agent.beta_gate.reset()

    committed_steps       = 0
    uncommitted_steps     = 0
    committed_and_elevated = 0
    gate_release_events   = 0
    running_var_vals: List[float] = []
    approach_scores: List[float] = []
    none_scores:     List[float] = []
    fatal = 0
    prev_elevated = False

    for _ in range(EVAL_EPISODES):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        prev_elevated = False

        for _ in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent   = agent.sense(obs_body, obs_world)
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, WORLD_DIM, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                theta_z = agent.theta_buffer.summary()

            try:
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        # Bug 3 fix: release gate at E3 tick boundary before new
                        # commitment decision (MECH-057b: gate releases at completion).
                        agent.beta_gate.release()

                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action
                        if result.committed:
                            agent.beta_gate.elevate()
                        # Bug 1 fix: get_diagnostics() does not exist; use
                        # get_commitment_state() which has the running_variance key.
                        diag = agent.e3.get_commitment_state()
                        running_var_vals.append(float(diag["running_variance"]))
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )
                        agent._last_action = action

                # Bug 4 fix: _committed_candidates is non-None after the very first
                # generate_trajectories() call and never cleared — useless as a proxy.
                # Use E3's own commitment criterion directly.
                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                is_elevated  = agent.beta_gate.is_elevated

                if is_committed:
                    committed_steps += 1
                    if is_elevated:
                        committed_and_elevated += 1
                else:
                    uncommitted_steps += 1

                if prev_elevated and not is_elevated:
                    gate_release_events += 1
                prev_elevated = is_elevated

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(theta_z).item())
                if ttype == "hazard_approach":
                    approach_scores.append(score)
                elif ttype == "none":
                    none_scores.append(score)
            except Exception:
                pass

            if done:
                break

    beta_state = agent.beta_gate.get_state()
    hold_rate = committed_and_elevated / max(1, committed_steps)
    cal_gap   = _mean(approach_scores) - _mean(none_scores)
    mean_rv   = _mean(running_var_vals)

    print(
        f"  committed_steps={committed_steps}  uncommitted={uncommitted_steps}\n"
        f"  committed_and_elevated={committed_and_elevated}  "
        f"hold_rate={hold_rate:.3f}\n"
        f"  gate_releases={gate_release_events}  "
        f"hold_count={beta_state['hold_count']}  "
        f"prop_count={beta_state['propagation_count']}\n"
        f"  mean_running_variance={mean_rv:.4f}  cal_gap={cal_gap:.4f}",
        flush=True,
    )

    return {
        "committed_step_count":        committed_steps,
        "uncommitted_step_count":      uncommitted_steps,
        "committed_and_elevated":      committed_and_elevated,
        "hold_rate_during_committed":  hold_rate,
        "gate_release_events":         gate_release_events,
        "hold_count_total":            beta_state["hold_count"],
        "propagation_count_total":     beta_state["propagation_count"],
        "mean_running_variance":       mean_rv,
        "calibration_gap_approach":    cal_gap,
        "n_approach":                  len(approach_scores),
        "n_none":                      len(none_scores),
        "fatal_errors":                fatal,
    }


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone

    agent, env, cfg = _make_agent_and_env(SEED)
    print(f"[EXQ-057] Training {WARMUP_EPISODES} eps...")
    _train(agent, env, WARMUP_EPISODES)

    print("\n[EXQ-057] Evaluating beta gate / ARC-016 commitment...")
    m = _eval(agent, env)

    c1 = m["committed_step_count"] >= 10
    c2 = m["committed_and_elevated"] > 0
    c3 = m["hold_rate_during_committed"] > 0.3
    c4 = m["gate_release_events"] > 0
    c5 = m["calibration_gap_approach"] > 0.0
    n_pass = sum([c1, c2, c3, c4, c5])
    status = "PASS" if n_pass >= 4 else "FAIL"

    print(f"\n[EXQ-057] {status} ({n_pass}/5 criteria)")
    print(f"  committed_step_count:    {m['committed_step_count']}")
    print(f"  committed_and_elevated:  {m['committed_and_elevated']}")
    print(f"  hold_rate:               {m['hold_rate_during_committed']:.3f}")
    print(f"  gate_release_events:     {m['gate_release_events']}")
    print(f"  mean_running_variance:   {m['mean_running_variance']:.4f}")
    print(f"  cal_gap_approach:        {m['calibration_gap_approach']:.4f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "run_id": f"{ts}_{EXPERIMENT_TYPE}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "metrics": m,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        "hyperparams": {
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "alpha_world": ALPHA_WORLD,
            "commit_threshold": cfg.e3.commitment_threshold,
            "precision_init": cfg.e3.precision_init,
            "seed": SEED,
        },
    }
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[EXQ-057] Result written to {out_path.name}")
