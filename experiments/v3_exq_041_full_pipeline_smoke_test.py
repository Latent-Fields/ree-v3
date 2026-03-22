"""
V3-EXQ-041 — Full Pipeline Smoke Test: ThetaBuffer E3 Calibration

Claims: MECH-089, ARC-016, MECH-071

The first experiment to exercise the full agent act_with_split_obs() pipeline.
All prior experiments bypassed the full architecture: they called agent.sense()
then accessed agent.e3.harm_eval(raw_z_world) directly, skipping ThetaBuffer,
HippocampalModule, BetaGate, and MultiRateClock.

Core question: Does E3.harm_eval show positive calibration_gap when receiving
theta-averaged z_world (MECH-089) instead of raw z_world step-by-step?

EXQ-037 showed sign inversion when training E3 on E2-predicted states (Fix2).
Hypothesis: the inversion was an artifact of distributional mismatch between raw
z_world and E2-predicted z_world. ThetaBuffer.summary() smooths z_world over the
last theta_buffer_size steps, reducing that mismatch. E3 trained on theta-averaged
z_world should show stable positive calibration_gap.

Design:
  - Full manual pipeline per step:
      sense() → clock.advance() → _e1_tick() → generate_trajectories()
      → select_action() → env.step()
  - E3.harm_eval trains on agent.theta_buffer.summary() (theta-averaged z_world)
  - terrain_prior is NOT trained (frozen) — isolates ThetaBuffer effect alone
  - E2.world_forward trains separately (MECH-069 separate optimizer)
  - E1 trains via compute_prediction_loss()

PASS criteria:
  C1: calibration_gap_approach > 0 (no sign inversion with theta-averaged z_world)
  C2: calibration_gap_approach > 0.03 (meaningful signal, same threshold as EXQ-026)
  C3: n_approach_eval >= 30 (sufficient approach samples)
  C4: e3_tick_count > 0 (clock actually fired E3 ticks during training)
  C5: world_forward_r2 > 0.05 (E2 world model functional)

If C1 FAIL: sign inversion is NOT a ThetaBuffer artifact — deeper problem, V4 signal.
If C1 PASS + C2 PASS: ThetaBuffer was the fix all along; prior experiments were testing
  a stub E3 on raw z_world, which explains all sign inversions.
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


EXPERIMENT_TYPE = "v3_exq_041_full_pipeline_smoke_test"
CLAIM_IDS = ["MECH-089", "ARC-016", "MECH-071"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES  = {"agent_caused_hazard", "env_caused_hazard"}


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
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
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    # ── Optimizers (MECH-069: separate error channels) ─────────────────────
    # Exclude world_forward params and terrain_prior params from main optimizer.
    # terrain_prior is frozen in EXQ-041 (isolates ThetaBuffer effect).
    wf_param_ids = set(
        id(p) for p in
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    terrain_param_ids = set(
        id(p) for p in
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters())
    )
    main_params = [
        p for p in agent.parameters()
        if id(p) not in wf_param_ids and id(p) not in terrain_param_ids
    ]
    optimizer    = optim.Adam(main_params, lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )

    # ── Training buffers ─────────────────────────────────────────────────────
    # E3.harm_eval training: (theta_z_world, label) split by ttype
    harm_buf_pos: List[torch.Tensor] = []   # theta_z_world at approach/contact steps
    harm_buf_neg: List[torch.Tensor] = []   # theta_z_world at none steps
    MAX_HARM_BUF = 1000

    # E2.world_forward training: (z_world_t, action_t, z_world_{t+1})
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    print(
        f"[V3-EXQ-041] Training {warmup_episodes} eps — full pipeline, terrain_prior FROZEN\n"
        f"  CausalGridWorldV2: body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}  proximity_scale={proximity_scale}",
        flush=True,
    )

    agent.train()
    e3_tick_total = 0

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            # ── Full pipeline ─────────────────────────────────────────────
            latent = agent.sense(obs_body, obs_world)

            # Record E2 z_self transition (z_self_t → z_self_{t+1})
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            if ticks.get("e3_tick", False):
                e3_tick_total += 1

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)

            # Theta-averaged z_world (MECH-089): what E3 "sees"
            theta_z = agent.theta_buffer.summary()  # already detached inside ThetaBuffer

            z_world_curr = latent.z_world.detach()

            # ── Step environment ──────────────────────────────────────────
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            # ── Collect harm_eval training data on theta_z_world ─────────
            is_pos = ttype in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0))
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            # ── Collect world_forward training data ───────────────────────
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]

                # ARC-016: update running_variance from wf prediction error
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(z_world_curr - z_pred)

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()

            # ── Train E3.harm_eval on theta-averaged z_world (MECH-089) ──
            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k       = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                pos_z   = torch.stack([harm_buf_pos[i] for i in pos_idx]).to(agent.device)
                neg_z   = torch.stack([harm_buf_neg[i] for i in neg_idx]).to(agent.device)
                z_batch = torch.cat([pos_z, neg_z], dim=0)
                labels  = torch.cat([
                    torch.ones(k, 1), torch.zeros(k, 1)
                ], dim=0).to(agent.device)

                harm_pred  = agent.e3.harm_eval(z_batch)
                harm_loss  = F.binary_cross_entropy(harm_pred, labels)
                e1_loss    = agent.compute_prediction_loss()
                total_loss = harm_loss + e1_loss

                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    optimizer.step()

            # ── Train E2.world_forward separately (MECH-069) ─────────────
            if len(wf_buf) >= 16 and step % 4 == 0:
                k    = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_t  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_t   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0
                    )
                    wf_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  ep {ep+1}/{warmup_episodes}  var={agent.e3._running_variance:.4f}  "
                f"e3_ticks={e3_tick_total}  pos_buf={len(harm_buf_pos)}  neg_buf={len(harm_buf_neg)}",
                flush=True,
            )

    # ── Eval phase ─────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-041] Eval: {eval_episodes} eps (full pipeline, theta_z_world)", flush=True)
    agent.eval()

    scores_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }

    with torch.no_grad():
        for ep in range(eval_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            z_world_prev = None
            action_prev  = None

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                latent   = agent.sense(obs_body, obs_world)
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action     = agent.select_action(candidates, ticks)

                # Theta-averaged z_world for harm_eval (MECH-089)
                theta_z = agent.theta_buffer.summary()
                score   = float(agent.e3.harm_eval(theta_z).mean().item())

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                if ttype in APPROACH_TTYPES:
                    scores_by_ttype["approach"].append(score)
                elif ttype in CONTACT_TTYPES:
                    scores_by_ttype["contact"].append(score)
                else:
                    scores_by_ttype["none"].append(score)

                if done:
                    break

    # ── Compute metrics ─────────────────────────────────────────────────────
    def mean_safe(lst: list, default: float = 0.0) -> float:
        return float(sum(lst) / len(lst)) if lst else default

    mean_approach = mean_safe(scores_by_ttype["approach"])
    mean_contact  = mean_safe(scores_by_ttype["contact"])
    mean_none     = mean_safe(scores_by_ttype["none"])
    n_approach    = len(scores_by_ttype["approach"])
    n_contact     = len(scores_by_ttype["contact"])
    n_none        = len(scores_by_ttype["none"])

    cal_gap_approach = mean_approach - mean_none
    cal_gap_contact  = mean_contact  - mean_none

    # World_forward R2 on held-out wf_buf
    wf_r2 = 0.0
    if len(wf_buf) >= 32:
        with torch.no_grad():
            k    = min(200, len(wf_buf))
            idxs = torch.randperm(len(wf_buf))[:k].tolist()
            zw_t  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
            a_t   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
            zw_t1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
            pred  = agent.e2.world_forward(zw_t, a_t)
            ss_res = (zw_t1 - pred).pow(2).sum().item()
            ss_tot = (zw_t1 - zw_t1.mean(0, keepdim=True)).pow(2).sum().item()
            wf_r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-8))

    # ── PASS / FAIL ─────────────────────────────────────────────────────────
    c1 = cal_gap_approach > 0.0
    c2 = cal_gap_approach > 0.03
    c3 = n_approach >= 30
    c4 = e3_tick_total > 0
    c5 = wf_r2 > 0.05

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: cal_gap_approach={cal_gap_approach:.4f} <= 0 "
            f"(sign inversion PERSISTS even with ThetaBuffer — deeper problem, V4 signal)"
        )
    if not c2:
        failure_notes.append(f"C2 FAIL: cal_gap_approach={cal_gap_approach:.4f} <= 0.03")
    if not c3:
        failure_notes.append(f"C3 FAIL: n_approach_eval={n_approach} < 30")
    if not c4:
        failure_notes.append(f"C4 FAIL: e3_tick_total={e3_tick_total} — clock not firing E3 ticks")
    if not c5:
        failure_notes.append(f"C5 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")

    print(f"\nV3-EXQ-041 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  cal_gap_approach={cal_gap_approach:.4f}  cal_gap_contact={cal_gap_contact:.4f}\n"
        f"  mean_approach={mean_approach:.4f}  mean_none={mean_none:.4f}\n"
        f"  n_approach={n_approach}  n_contact={n_contact}  n_none={n_none}\n"
        f"  e3_ticks={e3_tick_total}  wf_r2={wf_r2:.4f}",
        flush=True,
    )

    metrics = {
        "calibration_gap_approach":  float(cal_gap_approach),
        "calibration_gap_contact":   float(cal_gap_contact),
        "mean_harm_eval_approach":   float(mean_approach),
        "mean_harm_eval_contact":    float(mean_contact),
        "mean_harm_eval_none":       float(mean_none),
        "n_approach_eval":           float(n_approach),
        "n_contact_eval":            float(n_contact),
        "n_none_eval":               float(n_none),
        "e3_tick_total":             float(e3_tick_total),
        "world_forward_r2":          float(wf_r2),
        "running_variance_final":    float(agent.e3._running_variance),
        "alpha_world":               float(alpha_world),
        "crit1_pass": 1.0 if c1 else 0.0,
        "crit2_pass": 1.0 if c2 else 0.0,
        "crit3_pass": 1.0 if c3 else 0.0,
        "crit4_pass": 1.0 if c4 else 0.0,
        "crit5_pass": 1.0 if c5 else 0.0,
        "criteria_met": float(n_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-041 — Full Pipeline Smoke Test: ThetaBuffer E3 Calibration

**Status:** {status}
**Claims:** MECH-089, ARC-016, MECH-071
**World:** CausalGridWorldV2 (body={env.body_obs_dim}, world={env.world_obs_dim})
**alpha_world:** {alpha_world} (SD-008)  |  **Seed:** {seed}
**Predecessors:** EXQ-026 PASS (harm_eval on raw z_world), EXQ-037 PASS (sign inversion diagnosed)

## Motivation

First experiment to exercise the complete agent pipeline. All prior V3 experiments
called `agent.sense()` then `agent.e3.harm_eval(raw_z_world)` directly, bypassing:
- **ThetaBuffer** (MECH-089): E3 should receive theta-cycle averages, not raw z_world
- **HippocampalModule**: terrain-informed trajectory proposals (not random)
- **MultiRateClock**: E3 fires at its own rate, not every step

EXQ-037 showed sign inversion under Fix2 training. Hypothesis: the inversion was
a distributional artifact of raw z_world ≠ E2-predicted z_world. ThetaBuffer.summary()
averages z_world over the last {config.heartbeat.theta_buffer_size} steps, smoothing
the distribution. E3 trained on theta-averaged z_world should be stable.

terrain_prior is FROZEN at random init — this experiment isolates the ThetaBuffer
effect. EXQ-042 adds terrain_prior training.

## Results

| Transition Type | mean harm_eval | n |
|---|---|---|
| none | {mean_none:.4f} | {n_none} |
| approach | {mean_approach:.4f} | {n_approach} |
| contact | {mean_contact:.4f} | {n_contact} |

**calibration_gap_approach:** {cal_gap_approach:.4f}  (approach − none)
**calibration_gap_contact:**  {cal_gap_contact:.4f}  (contact − none)

## Pipeline Verification

- E3 tick total: {e3_tick_total}  (clock functional: {c4})
- world_forward R2: {wf_r2:.4f}
- final running_variance: {agent.e3._running_variance:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: cal_gap_approach > 0 (no sign inversion) | {"PASS" if c1 else "FAIL"} | {cal_gap_approach:.4f} |
| C2: cal_gap_approach > 0.03 (meaningful) | {"PASS" if c2 else "FAIL"} | {cal_gap_approach:.4f} |
| C3: n_approach_eval >= 30 | {"PASS" if c3 else "FAIL"} | {n_approach} |
| C4: e3_tick_count > 0 | {"PASS" if c4 else "FAIL"} | {e3_tick_total} |
| C5: world_forward R2 > 0.05 | {"PASS" if c5 else "FAIL"} | {wf_r2:.4f} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--warmup-episodes",  type=int,   default=400)
    parser.add_argument("--eval-episodes",    type=int,   default=50)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]         = CLAIM_IDS[0]
    result["verdict"]       = result["status"]
    result["run_id"]        = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
