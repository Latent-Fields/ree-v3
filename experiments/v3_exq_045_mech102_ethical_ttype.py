"""
V3-EXQ-045 — MECH-102: Energy Escalation Ladder (Ethical Policy + ttype Split)

Claims: MECH-102, ARC-024

EXQ-032b PASS confirmed the MECH-102 escalation ladder under random policy:
  none=-0.032  →  approach=+0.006  →  contact=+0.017
But random policy provides a one-sided test: the agent sometimes blunders into
hazards, producing positive causal_sig at contact (harm_actual > mean_cf).

EXQ-045 complements EXQ-032b with the ETHICAL policy variant. The agent always
picks the action that minimises predicted harm: argmin E3(E2(z_world, a)).

With an ethical policy the sign of causal_sig inverts:
  harm_actual  = E3(E2(z, a_best))   ← minimum possible harm
  mean_cf_harm = mean(E3(E2(z, a_cf))) ← average over all other actions
  causal_sig   = harm_actual - mean_cf   ← NEGATIVE (agent avoided harm)

The interesting metric becomes the ADVANTAGE of the ethical choice:
  advantage_sig = mean_cf_harm - harm_actual
  = how much harm did the ethical agent spare vs the average alternative?

MECH-102 with ethical policy predicts:
  advantage_sig_contact > advantage_sig_approach > advantage_sig_none

Interpretation:
  - At "none" (safe locomotion): all actions lead to similar low-harm outcomes.
    The ethical choice saves little because alternatives are also safe.
    advantage_sig ≈ 0.
  - At "approach" (near hazard): some actions go deeper into the hazard gradient,
    others retreat. The ethical agent actively avoids harm; advantage rises.
  - At "contact" (hazard overlap): action space has maximum gradient — some paths
    continue through residue, some escape. Ethical advantage is maximised.

Relationship to EXQ-032b:
  EXQ-032b measures consequentiality of harm INCURRED (random blunder).
  EXQ-045 measures consequentiality of harm AVOIDED (ethical choice).
  If both PASS, MECH-102 is robust across policy types:
  "actions are more consequential near hazards" regardless of whether you
  measure that by how much harm you cause or how much harm you prevent.

Training:
  Identical to EXQ-032b (random policy warmup, 500 eps). E3 trained on
  E2-predicted states (Fix2 pattern from EXQ-032b) so it's calibrated
  on the counterfactual input distribution used at eval.

Eval:
  Ethical policy: a* = argmin_{a} E3(E2(z_world, a)) over all discrete actions.
  Same transition_type split as EXQ-032b.

PASS criteria (ALL must hold):
  C1: advantage_sig_contact > advantage_sig_none   (core MECH-102)
  C2: advantage_sig_approach > advantage_sig_none  (gradient before contact)
  C3: advantage_sig_contact > 0.001                (non-trivial advantage at contact)
  C4: world_forward_r2 > 0.05
  C5: n_contact >= 50
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_045_mech102_ethical_ttype"
CLAIM_IDS = ["MECH-102", "ARC-024"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 100,
    steps_per_episode: int = 300,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=6, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.2,
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

    print(
        f"[V3-EXQ-045] MECH-102 Ethical Policy + ttype Split\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  num_hazards=6  Training: RANDOM  Eval: ETHICAL (argmin E3(E2(z,a)))\n"
        f"  Metric: advantage_sig = mean_cf_harm - harm_actual",
        flush=True,
    )

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer               = optim.Adam(standard_params,      lr=lr)
    world_forward_optimizer = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer     = optim.Adam(harm_eval_params,      lr=1e-4)

    # ── Training (identical to EXQ-032b: random policy, Fix2 E3 training) ───
    print(f"\n[V3-EXQ-045] Training ({warmup_episodes} eps, random policy)...", flush=True)
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    train_counts: Dict[str, int] = {}
    num_actions = env.action_dim

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            train_counts[ttype] = train_counts.get(ttype, 0) + 1

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_optimizer.step()

            # E3.harm_eval: Fix2 — train on both observed and E2-predicted states
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
                    zw_pos_pred = agent.e2.world_forward(zw_pos_obs.to(agent.device), a_rand_pos)
                    zw_neg_pred = agent.e2.world_forward(zw_neg_obs.to(agent.device), a_rand_neg)

                zw_b = torch.cat([
                    zw_pos_obs.to(agent.device), zw_neg_obs.to(agent.device),
                    zw_pos_pred, zw_neg_pred,
                ], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            approach = train_counts.get("hazard_approach", 0)
            contact  = (train_counts.get("env_caused_hazard", 0) +
                        train_counts.get("agent_caused_hazard", 0))
            print(
                f"  [train] ep {ep+1}/{warmup_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    # ── world_forward R² ────────────────────────────────────────────────────
    wf_r2 = 0.0
    if len(wf_data) >= 20:
        n = len(wf_data)
        n_train = int(n * 0.8)
        with torch.no_grad():
            zw_all  = torch.cat([d[0] for d in wf_data], dim=0).to(agent.device)
            a_all   = torch.cat([d[1] for d in wf_data], dim=0).to(agent.device)
            zw1_all = torch.cat([d[2] for d in wf_data], dim=0).to(agent.device)
            pred_all  = agent.e2.world_forward(zw_all, a_all)
            pred_test = pred_all[n_train:]
            tgt_test  = zw1_all[n_train:]
            if pred_test.shape[0] > 0:
                ss_res = ((tgt_test - pred_test) ** 2).sum()
                ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
                wf_r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R² (test): {wf_r2:.4f}", flush=True)

    # ── Eval: ethical policy + advantage_sig by ttype ───────────────────────
    print(
        f"\n[V3-EXQ-045] Eval ({eval_episodes} eps, ethical policy)...",
        flush=True,
    )
    agent.eval()

    advantage_sigs_by_ttype: Dict[str, List[float]] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

                # Compute E3(E2(z, a)) for all discrete actions
                harm_per_action: List[float] = []
                for a_idx in range(num_actions):
                    a_oh = _action_to_onehot(a_idx, num_actions, agent.device)
                    z_next = agent.e2.world_forward(z_world, a_oh)
                    harm_per_action.append(float(agent.e3.harm_eval(z_next).item()))

                # Ethical policy: pick action with minimum predicted harm
                best_idx   = int(np.argmin(harm_per_action))
                harm_actual = harm_per_action[best_idx]
                cf_harms    = [h for i, h in enumerate(harm_per_action) if i != best_idx]
                mean_cf     = float(np.mean(cf_harms)) if cf_harms else harm_actual

                # advantage_sig = how much harm the ethical choice spared
                advantage_sig = mean_cf - harm_actual

            action = _action_to_onehot(best_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype not in advantage_sigs_by_ttype:
                advantage_sigs_by_ttype[ttype] = []
            advantage_sigs_by_ttype[ttype].append(advantage_sig)

            if done:
                break

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    contact_sigs = (
        advantage_sigs_by_ttype.get("agent_caused_hazard", []) +
        advantage_sigs_by_ttype.get("env_caused_hazard", [])
    )

    mean_none     = _mean(advantage_sigs_by_ttype.get("none", []))
    mean_approach = _mean(advantage_sigs_by_ttype.get("hazard_approach", []))
    mean_contact  = _mean(contact_sigs)
    n_none        = len(advantage_sigs_by_ttype.get("none", []))
    n_approach    = len(advantage_sigs_by_ttype.get("hazard_approach", []))
    n_contact     = len(contact_sigs)

    print(f"\n  --- MECH-102 Ethical Advantage Ladder (EXQ-045) ---", flush=True)
    print(f"  none (baseline):     advantage_sig={mean_none:.6f}  n={n_none}", flush=True)
    print(f"  hazard_approach:     advantage_sig={mean_approach:.6f}  n={n_approach}", flush=True)
    print(f"  contact (combined):  advantage_sig={mean_contact:.6f}  n={n_contact}", flush=True)
    print(f"  Ladder: none={mean_none:.4f} → approach={mean_approach:.4f} → contact={mean_contact:.4f}", flush=True)
    print(f"\n  By individual ttype:", flush=True)
    for tt, sigs in sorted(advantage_sigs_by_ttype.items()):
        print(f"    {tt:28s}: advantage_sig={_mean(sigs):.6f}  n={len(sigs)}", flush=True)

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1 = mean_contact  > mean_none
    c2 = mean_approach > mean_none
    c3 = mean_contact  > 0.001
    c4 = wf_r2         > 0.05
    c5 = n_contact     >= 50

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: advantage_sig_contact={mean_contact:.6f} <= "
            f"advantage_sig_none={mean_none:.6f}. "
            f"Ethical choice doesn't save more harm at contact than at baseline."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: advantage_sig_approach={mean_approach:.6f} <= "
            f"advantage_sig_none={mean_none:.6f}. "
            f"No gradient before contact."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: advantage_sig_contact={mean_contact:.6f} <= 0.001 "
            f"(all actions equally harmful at contact — no room for ethical choice)"
        )
    if not c4:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5:
        failure_notes.append(f"C5 FAIL: n_contact={n_contact} < 50")

    print(f"\nV3-EXQ-045 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":                     float(alpha_world),
        "proximity_scale":                 float(proximity_scale),
        "world_forward_r2":                float(wf_r2),
        "advantage_sig_none":              float(mean_none),
        "advantage_sig_approach":          float(mean_approach),
        "advantage_sig_contact":           float(mean_contact),
        "n_none":                          float(n_none),
        "n_approach":                      float(n_approach),
        "n_contact":                       float(n_contact),
        "train_contact_events": float(
            train_counts.get("env_caused_hazard", 0) +
            train_counts.get("agent_caused_hazard", 0)
        ),
        "train_approach_events": float(train_counts.get("hazard_approach", 0)),
        "crit1_pass":    1.0 if c1 else 0.0,
        "crit2_pass":    1.0 if c2 else 0.0,
        "crit3_pass":    1.0 if c3 else 0.0,
        "crit4_pass":    1.0 if c4 else 0.0,
        "crit5_pass":    1.0 if c5 else 0.0,
        "criteria_met":  float(n_met),
        "fatal_error_count": 0.0,
    }
    for tt, sigs in advantage_sigs_by_ttype.items():
        metrics[f"advantage_sig_ttype_{tt.replace(' ', '_')}"] = float(_mean(sigs))

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-045 — MECH-102: Energy Escalation Ladder (Ethical Policy + ttype Split)

**Status:** {status}
**Claims:** MECH-102, ARC-024
**World:** CausalGridWorldV2 (6 hazards, 3 resources)
**Training policy:** RANDOM  |  **Eval policy:** ETHICAL (argmin E3(E2(z, a)))
**Metric:** advantage_sig = mean_cf_harm − harm_actual
**alpha_world:** {alpha_world}  (SD-008)  |  **Seed:** {seed}
**Complement:** EXQ-032b PASS (same claim, random policy)

## Design Rationale

EXQ-032b used random eval policy: causal_sig = harm_actual − mean_cf (positive at
contact because agent stepped into hazard). EXQ-045 tests the complementary case.

With ethical policy, harm_actual is the *minimum* over all actions.
advantage_sig = mean_cf − harm_actual = how much harm the ethical choice spared.

MECH-102 prediction: advantage_sig escalates with transition energy:
- none (safe): all actions harmless → advantage ≈ 0
- approach: action space starts to diverge (toward vs away from hazard) → advantage rises
- contact: maximum action-space gradient → advantage maximised

If PASS: ethical agency matters MOST when stakes are highest.
Both EXQ-032b (harm incurred) and EXQ-045 (harm avoided) confirm MECH-102.

## Results — Ethical Advantage Ladder

| State Energy Level | advantage_sig | n steps |
|---|---|---|
| none (safe locomotion)    | {mean_none:.6f} | {n_none} |
| hazard_approach (medium)  | {mean_approach:.6f} | {n_approach} |
| contact (high — combined) | {mean_contact:.6f} | {n_contact} |

- **world_forward R²**: {wf_r2:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: advantage_sig_contact > advantage_sig_none | {"PASS" if c1 else "FAIL"} | {mean_contact:.6f} vs {mean_none:.6f} |
| C2: advantage_sig_approach > advantage_sig_none (gradient precedes contact) | {"PASS" if c2 else "FAIL"} | {mean_approach:.6f} vs {mean_none:.6f} |
| C3: advantage_sig_contact > 0.001 (non-trivial advantage at contact) | {"PASS" if c3 else "FAIL"} | {mean_contact:.6f} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4 else "FAIL"} | {wf_r2:.4f} |
| C5: n_contact >= 50 | {"PASS" if c5 else "FAIL"} | {n_contact} |

Criteria met: {n_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=500)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=300)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
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
