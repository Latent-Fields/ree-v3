"""
V3-EXQ-043 — SD-003 via Trajectory Attribution (SD-003, MECH-102)

Claims: SD-003, MECH-102

Prerequisites: EXQ-041 PASS (ThetaBuffer stable), EXQ-042 PASS (terrain-informed proposals).

Prior SD-003 attempts (EXQ-030b, 036) failed because:
  1. 1-step E2 counterfactuals produce indistinguishable next-states
  2. Training E3 on E2-predicted states (Fix2) causes sign inversion
  3. Multi-step rollout (k=5) still insufficient divergence

This experiment implements SD-003 as TRAJECTORY ATTRIBUTION via the
hippocampal proposal distribution:

  At each eval step, generate N hippocampal candidate trajectories.
  E3.score_trajectory() evaluates all N candidates via J(ζ) = F + λM + ρΦ_R.
  The agent's selection (lowest-score trajectory) is the "actual path."
  All other candidates are "counterfactual paths" — terrain alternatives
  that could have been taken from the same starting z_world_theta.

  causal_sig_t = mean(cf_scores) - selected_score

  Positive causal_sig = agent's choice mattered (best path meaningfully
  better than alternatives). Near hazards, trajectory scores diverge more
  (some paths lead through residue, others avoid it), so causal_sig should
  be higher at approach and contact states than at safe states.

This is architecturally correct SD-003:
  - Counterfactuals are TERRAIN PATHS, not raw single-step actions
  - E3 evaluates all paths on the SAME z_world_theta (theta-averaged)
  - The "causal" question is: did the agent's terrain navigation matter?

MECH-102 (energy escalation ladder) is tested simultaneously:
  causal_sig_contact > causal_sig_approach > causal_sig_none
  = trajectory decision consequence escalates near hazards

PASS criteria:
  C1: causal_sig_approach > causal_sig_none (approach states have higher decision consequence)
  C2: causal_sig_contact > causal_sig_approach (graduated escalation — MECH-102)
  C3: causal_sig_approach > 0.001 (above noise floor)
  C4: calibration_gap_approach > 0.03 (E3 still calibrated; needed for scores to be meaningful)
  C5: n_approach_eval >= 30

Training: identical to EXQ-042 (full pipeline + terrain_prior training).
Evaluation: same training, then compute causal_sig by transition_type.
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


EXPERIMENT_TYPE = "v3_exq_043_sd003_trajectory_attribution"
CLAIM_IDS = ["SD-003", "MECH-102"]

APPROACH_TTYPES = {"hazard_approach"}
CONTACT_TTYPES  = {"agent_caused_hazard", "env_caused_hazard"}
N_CANDIDATES    = 16   # hippocampal proposals for attribution measurement
CANDIDATE_HORIZON = 5


def run(
    seed: int = 0,
    warmup_episodes: int = 600,
    eval_episodes: int = 100,
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

    # ── Optimizers ──────────────────────────────────────────────────────────
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
    optimizer = optim.Adam(main_params, lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    terrain_optimizer = optim.Adam(
        list(agent.hippocampal.terrain_prior.parameters()) +
        list(agent.hippocampal.action_object_decoder.parameters()),
        lr=5e-4,
    )

    # ── Buffers ─────────────────────────────────────────────────────────────
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_HARM_BUF = 1000
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF_BUF = 2000

    print(
        f"[V3-EXQ-043] Training {warmup_episodes} eps — full pipeline + terrain_prior\n"
        f"  SD-003 trajectory attribution: causal_sig by transition_type\n"
        f"  CausalGridWorldV2: body={env.body_obs_dim}  world={env.world_obs_dim}",
        flush=True,
    )

    # ── Training (identical to EXQ-042) ─────────────────────────────────────
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

            latent = agent.sense(obs_body, obs_world)
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z    = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action

                # Terrain_prior behavioral cloning
                selected_ao = result.selected_trajectory.get_action_object_sequence()
                if selected_ao is not None:
                    ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                        theta_z, e1_prior=e1_prior.detach()
                    )
                    terrain_loss = F.mse_loss(ao_mean_pred, selected_ao.detach())
                    terrain_optimizer.zero_grad()
                    terrain_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.hippocampal.terrain_prior.parameters()) +
                        list(agent.hippocampal.action_object_decoder.parameters()), 1.0
                    )
                    terrain_optimizer.step()
            else:
                action = agent._last_action
                if action is None:
                    action_idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            is_pos = ttype in APPROACH_TTYPES | CONTACT_TTYPES
            if is_pos:
                harm_buf_pos.append(theta_z.squeeze(0))
                if len(harm_buf_pos) > MAX_HARM_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
            else:
                harm_buf_neg.append(theta_z.squeeze(0))
                if len(harm_buf_neg) > MAX_HARM_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > MAX_WF_BUF:
                    wf_buf = wf_buf[-MAX_WF_BUF:]
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, action_prev)
                    agent.e3.update_running_variance(z_world_curr - z_pred)

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()

            if len(harm_buf_pos) >= 8 and len(harm_buf_neg) >= 8 and step % 8 == 0:
                k       = min(16, len(harm_buf_pos), len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k].tolist()
                z_batch = torch.cat([
                    torch.stack([harm_buf_pos[i] for i in pos_idx]),
                    torch.stack([harm_buf_neg[i] for i in neg_idx]),
                ], dim=0).to(agent.device)
                labels  = torch.cat([torch.ones(k, 1), torch.zeros(k, 1)], dim=0).to(agent.device)
                harm_loss  = F.binary_cross_entropy(agent.e3.harm_eval(z_batch), labels)
                e1_loss    = agent.compute_prediction_loss()
                total_loss = harm_loss + e1_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                    optimizer.step()

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
                f"  ep {ep+1}/{warmup_episodes}  e3_ticks={e3_tick_total}  "
                f"pos_buf={len(harm_buf_pos)}  var={agent.e3._running_variance:.4f}",
                flush=True,
            )

    # ── Eval phase: SD-003 trajectory attribution ───────────────────────────
    print(
        f"\n[V3-EXQ-043] Eval: {eval_episodes} eps — SD-003 trajectory attribution",
        flush=True,
    )
    agent.eval()

    causal_sig_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }
    harm_eval_by_ttype: Dict[str, List[float]] = {
        "approach": [], "contact": [], "none": [],
    }

    with torch.no_grad():
        for ep in range(eval_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                latent   = agent.sense(obs_body, obs_world)
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                theta_z  = agent.theta_buffer.summary()
                z_self   = latent.z_self.detach()

                # ── SD-003 trajectory attribution ─────────────────────────
                # Generate N hippocampal candidates from theta-averaged z_world.
                # E3 scores all N via J(ζ). Selected = lowest J(ζ).
                # causal_sig = mean(other_scores) - selected_score
                # Positive = agent's terrain choice was meaningfully better.
                hippo_trajs = agent.hippocampal.propose_trajectories(
                    z_world=theta_z, z_self=z_self,
                    num_candidates=N_CANDIDATES, e1_prior=e1_prior.detach(),
                )

                if hippo_trajs:
                    scores = torch.stack([
                        agent.e3.score_trajectory(t).mean() for t in hippo_trajs
                    ])
                    selected_idx  = int(scores.argmin().item())
                    selected_score = float(scores[selected_idx].item())
                    cf_scores = [
                        float(scores[i].item())
                        for i in range(len(scores))
                        if i != selected_idx
                    ]
                    mean_cf_score = sum(cf_scores) / len(cf_scores) if cf_scores else selected_score
                    causal_sig    = mean_cf_score - selected_score  # positive = selection mattered
                    action        = hippo_trajs[selected_idx].actions[:, 0, :].detach()
                else:
                    causal_sig = 0.0
                    action_idx = random.randint(0, env.action_dim - 1)
                    action     = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, action_idx] = 1.0

                # Harm_eval calibration (still on theta_z)
                harm_score = float(agent.e3.harm_eval(theta_z).mean().item())

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)
                ttype = info.get("transition_type", "none")

                key = (
                    "approach" if ttype in APPROACH_TTYPES else
                    "contact"  if ttype in CONTACT_TTYPES  else
                    "none"
                )
                causal_sig_by_ttype[key].append(causal_sig)
                harm_eval_by_ttype[key].append(harm_score)

                if done:
                    break

    # ── Compute metrics ─────────────────────────────────────────────────────
    def mean_safe(lst: list, default: float = 0.0) -> float:
        return float(sum(lst) / len(lst)) if lst else default

    causal_sig_approach = mean_safe(causal_sig_by_ttype["approach"])
    causal_sig_contact  = mean_safe(causal_sig_by_ttype["contact"])
    causal_sig_none     = mean_safe(causal_sig_by_ttype["none"])

    mean_harm_approach  = mean_safe(harm_eval_by_ttype["approach"])
    mean_harm_none      = mean_safe(harm_eval_by_ttype["none"])
    cal_gap_approach    = mean_harm_approach - mean_harm_none

    n_approach = len(causal_sig_by_ttype["approach"])
    n_contact  = len(causal_sig_by_ttype["contact"])
    n_none     = len(causal_sig_by_ttype["none"])

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
    c1 = causal_sig_approach > causal_sig_none
    c2 = causal_sig_contact  > causal_sig_approach
    c3 = causal_sig_approach > 0.001
    c4 = cal_gap_approach    > 0.03
    c5 = n_approach          >= 30

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: causal_sig_approach={causal_sig_approach:.4f} <= "
            f"causal_sig_none={causal_sig_none:.4f}. "
            f"Approach states don't show higher decision consequence than safe states."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: causal_sig_contact={causal_sig_contact:.4f} <= "
            f"causal_sig_approach={causal_sig_approach:.4f}. "
            f"MECH-102 escalation ladder not present."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: causal_sig_approach={causal_sig_approach:.4f} <= 0.001 "
            f"(at noise floor — trajectory scores don't diverge near hazards)."
        )
    if not c4:
        failure_notes.append(
            f"C4 FAIL: cal_gap_approach={cal_gap_approach:.4f} <= 0.03 "
            f"(E3 not calibrated — trajectory scores not meaningful)."
        )
    if not c5:
        failure_notes.append(f"C5 FAIL: n_approach_eval={n_approach} < 30")

    print(f"\nV3-EXQ-043 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)
    print(
        f"  causal_sig: none={causal_sig_none:.4f}  "
        f"approach={causal_sig_approach:.4f}  contact={causal_sig_contact:.4f}\n"
        f"  cal_gap_approach={cal_gap_approach:.4f}  "
        f"n_approach={n_approach}  n_contact={n_contact}  n_none={n_none}\n"
        f"  e3_ticks={e3_tick_total}  wf_r2={wf_r2:.4f}",
        flush=True,
    )

    metrics = {
        "causal_sig_none":           float(causal_sig_none),
        "causal_sig_approach":       float(causal_sig_approach),
        "causal_sig_contact":        float(causal_sig_contact),
        "causal_sig_gap_approach":   float(causal_sig_approach - causal_sig_none),
        "causal_sig_gap_contact":    float(causal_sig_contact  - causal_sig_approach),
        "calibration_gap_approach":  float(cal_gap_approach),
        "mean_harm_eval_approach":   float(mean_harm_approach),
        "mean_harm_eval_none":       float(mean_harm_none),
        "n_approach_eval":           float(n_approach),
        "n_contact_eval":            float(n_contact),
        "n_none_eval":               float(n_none),
        "world_forward_r2":          float(wf_r2),
        "e3_tick_total":             float(e3_tick_total),
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

    summary_markdown = f"""# V3-EXQ-043 — SD-003 Trajectory Attribution

**Status:** {status}
**Claims:** SD-003, MECH-102
**World:** CausalGridWorldV2 (body={env.body_obs_dim}, world={env.world_obs_dim})
**alpha_world:** {alpha_world} (SD-008)  |  **Seed:** {seed}
**Prerequisites:** EXQ-041 PASS, EXQ-042 PASS (full pipeline + terrain_prior trained)

## SD-003 Redesign: Trajectory Attribution

Prior SD-003 tests (EXQ-030b, 036) computed causal signatures as:
  `E3(E2(z_world, a_actual)) − E3(E2(z_world, a_cf))`
This produced weak signals (causal_sig ≈ 0.003) because:
  1. Single-step E2 predictions are nearly identical for all actions
  2. Training E3 on E2-predicted states (Fix2) causes sign inversion

This experiment implements SD-003 as terrain attribution:
  `causal_sig = mean(E3_score(cf_trajectories)) − E3_score(selected_trajectory)`

- N={N_CANDIDATES} hippocampal candidates generated from theta-averaged z_world
- E3 scores all N via J(ζ) = F(ζ) + λM(ζ) + ρΦ_R(ζ)
- Selected trajectory = lowest J(ζ) (agent's actual terrain path)
- Counterfactuals = all other terrain paths
- causal_sig > 0 means the agent's terrain navigation was meaningfully better

Near hazards, terrain paths diverge more (some paths enter residue-heavy regions,
others avoid them), so causal_sig should be elevated at approach and contact.

## Attribution Results by Transition Type

| Transition Type | causal_sig | n |
|---|---|---|
| none | {causal_sig_none:.4f} | {n_none} |
| approach | {causal_sig_approach:.4f} | {n_approach} |
| contact | {causal_sig_contact:.4f} | {n_contact} |

approach − none gap: {causal_sig_approach - causal_sig_none:.4f}
contact − approach gap: {causal_sig_contact - causal_sig_approach:.4f}

## E3 Calibration (required for meaningful scores)

| Type | mean harm_eval |
|---|---|
| none | {mean_harm_none:.4f} |
| approach | {mean_harm_approach:.4f} |
calibration_gap_approach: {cal_gap_approach:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: causal_sig_approach > causal_sig_none | {"PASS" if c1 else "FAIL"} | {causal_sig_approach:.4f} vs {causal_sig_none:.4f} |
| C2: causal_sig_contact > causal_sig_approach (MECH-102) | {"PASS" if c2 else "FAIL"} | {causal_sig_contact:.4f} vs {causal_sig_approach:.4f} |
| C3: causal_sig_approach > 0.001 | {"PASS" if c3 else "FAIL"} | {causal_sig_approach:.4f} |
| C4: cal_gap_approach > 0.03 (E3 calibrated) | {"PASS" if c4 else "FAIL"} | {cal_gap_approach:.4f} |
| C5: n_approach_eval >= 30 | {"PASS" if c5 else "FAIL"} | {n_approach} |

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
    parser.add_argument("--warmup-episodes",  type=int,   default=600)
    parser.add_argument("--eval-episodes",    type=int,   default=100)
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
