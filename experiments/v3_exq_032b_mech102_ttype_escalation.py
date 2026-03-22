"""
V3-EXQ-032b -- MECH-102: Energy Escalation Ladder via Transition-Type Split

Claims: MECH-102, ARC-024, SD-003

EXQ-032 FAIL post-mortem:
    E3-guided harm-minimizing policy on 6-hazard, 3-resource world never entered
    high harm_exposure states (mean=0.0018, n_high=0). The agent was too effective
    at avoidance -- viability was never genuinely threatened.

EXQ-032b redesign: avoid the harm_exposure EMA entirely.
    Instead, split steps directly by transition_type:
        none             -> baseline: no hazard proximity
        hazard_approach  -> medium-energy: near hazard, gradient increasing
        contact          -> high-energy: actual hazard contact (agent + env combined)

    Use RANDOM policy during eval. This:
      (a) Guarantees the agent enters all ttype categories naturally
      (b) Removes confound between policy ethics and measurement
      (c) Provides a clean test of the MECH-102 escalation ladder:
          "actions are more consequential when the agent is in higher-energy states"

    With random policy, causal_sig = E3(E2(z, a_rand)) - mean_cf(E3(E2(z, a_cf))).
    At contact steps (agent in hazard): the actual action (stepped into hazard)
    produces a high-harm predicted next state; counterfactuals (step away) produce
    lower-harm predicted states -> positive causal_sig.
    At none steps (safe zone): E3 varies little across actions -> small causal_sig.

MECH-102 prediction (reformulated for ttype split):
    causal_sig_contact > causal_sig_approach > causal_sig_none
    This is the "energy escalation ladder": the agent's action is increasingly
    consequential as it moves into higher-energy interaction states with the world.

PASS criteria (ALL must hold):
    C1: causal_sig_contact > causal_sig_none
        Core MECH-102: contact-state actions more consequential than baseline
    C2: causal_sig_approach > causal_sig_none
        Approach already elevates attribution (gradient precedes contact)
    C3: causal_sig_contact > 0.001  (positive signal, not noise)
    C4: world_forward_r2 > 0.05
    C5: n_contact >= 50  (enough contact events for reliable measurement)

Architecture basis:
    MECH-102 (violence as terminal error correction)
    ARC-024 (gradient world -- approach events are meaningful intermediate states)
    SD-003 (counterfactual causal_sig via E2.world_forward + E3.harm_eval)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_032b_mech102_ttype_escalation"
CLAIM_IDS = ["MECH-102", "ARC-024", "SD-003"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all  = agent.e2.world_forward(zw_all, a_all)
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R2 (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Random-policy warmup with E3 Fix 2 (observed + E2-predicted states)."""
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

    for ep in range(num_episodes):
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
            counts[ttype] = counts.get(ttype, 0) + 1

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
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_optimizer.step()

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
                    zw_pos_obs.to(agent.device),
                    zw_neg_obs.to(agent.device),
                    zw_pos_pred,
                    zw_neg_pred,
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

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _eval_ttype_ladder(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """
    Random-policy eval. Measure causal_sig by transition_type.
    Group "env_caused_hazard" + "agent_caused_hazard" -> "contact" for MECH-102.
    """
    agent.eval()
    num_actions = env.action_dim

    causal_sigs_by_ttype: Dict[str, List[float]] = {}

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                z_actual   = agent.e2.world_forward(z_world, action)
                harm_actual = float(agent.e3.harm_eval(z_actual).item())

                cf_harms = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf = agent.e2.world_forward(z_world, a_cf)
                    cf_harms.append(float(agent.e3.harm_eval(z_cf).item()))

                mean_cf = float(np.mean(cf_harms)) if cf_harms else harm_actual
                causal_sig = harm_actual - mean_cf

            if ttype not in causal_sigs_by_ttype:
                causal_sigs_by_ttype[ttype] = []
            causal_sigs_by_ttype[ttype].append(causal_sig)

            if done:
                break

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    # Group contact types
    contact_sigs = (
        causal_sigs_by_ttype.get("agent_caused_hazard", []) +
        causal_sigs_by_ttype.get("env_caused_hazard", [])
    )

    mean_none     = _mean(causal_sigs_by_ttype.get("none", []))
    mean_approach = _mean(causal_sigs_by_ttype.get("hazard_approach", []))
    mean_contact  = _mean(contact_sigs)

    n_none     = len(causal_sigs_by_ttype.get("none", []))
    n_approach = len(causal_sigs_by_ttype.get("hazard_approach", []))
    n_contact  = len(contact_sigs)

    print(f"\n  --- MECH-102 Energy Escalation Ladder (EXQ-032b) ---", flush=True)
    print(f"  none (baseline):   causal_sig={mean_none:.6f}  n={n_none}", flush=True)
    print(f"  hazard_approach:   causal_sig={mean_approach:.6f}  n={n_approach}", flush=True)
    print(f"  contact (combined): causal_sig={mean_contact:.6f}  n={n_contact}", flush=True)
    print(f"  Ladder: none={mean_none:.4f} -> approach={mean_approach:.4f} -> contact={mean_contact:.4f}", flush=True)
    print(f"\n  By individual ttype:", flush=True)
    for tt, sigs in sorted(causal_sigs_by_ttype.items()):
        print(f"    {tt:28s}: causal_sig={_mean(sigs):.6f}  n={len(sigs)}", flush=True)

    return {
        "mean_causal_sig_none":     mean_none,
        "mean_causal_sig_approach": mean_approach,
        "mean_causal_sig_contact":  mean_contact,
        "n_none":                   n_none,
        "n_approach":               n_approach,
        "n_contact":                n_contact,
        "by_ttype":                 {t: _mean(s) for t, s in causal_sigs_by_ttype.items()},
    }


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
        f"[V3-EXQ-032b] MECH-102 Energy Escalation Ladder (ttype split)\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  num_hazards=6  Policy: RANDOM (avoids EMA-avoidance problem)\n"
        f"  Split: none / hazard_approach / contact(agent+env combined)",
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
    harm_eval_optimizer     = optim.Adam(harm_eval_params,     lr=1e-4)

    print(f"\n[V3-EXQ-032b] Training ({warmup_episodes} eps)...", flush=True)
    train_out = _train(
        agent, env, optimizer, harm_eval_optimizer, world_forward_optimizer,
        warmup_episodes, steps_per_episode,
    )

    wf_r2 = _compute_world_forward_r2(agent, train_out["wf_data"])

    print(f"\n[V3-EXQ-032b] Eval ({eval_episodes} eps, random policy)...", flush=True)
    eval_out = _eval_ttype_ladder(agent, env, eval_episodes, steps_per_episode)

    c1_pass = eval_out["mean_causal_sig_contact"]  > eval_out["mean_causal_sig_none"]
    c2_pass = eval_out["mean_causal_sig_approach"] > eval_out["mean_causal_sig_none"]
    c3_pass = eval_out["mean_causal_sig_contact"]  > 0.001
    c4_pass = wf_r2 > 0.05
    c5_pass = eval_out["n_contact"] >= 50

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: causal_sig_contact={eval_out['mean_causal_sig_contact']:.6f} <= causal_sig_none={eval_out['mean_causal_sig_none']:.6f}"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: causal_sig_approach={eval_out['mean_causal_sig_approach']:.6f} <= causal_sig_none={eval_out['mean_causal_sig_none']:.6f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: causal_sig_contact={eval_out['mean_causal_sig_contact']:.6f} <= 0.001"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: world_forward_r2={wf_r2:.4f} <= 0.05")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: n_contact={eval_out['n_contact']} < 50")

    print(f"\nV3-EXQ-032b verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    tc = train_out["counts"]
    metrics = {
        "alpha_world":                    float(alpha_world),
        "proximity_scale":                float(proximity_scale),
        "world_forward_r2":               float(wf_r2),
        "mean_causal_sig_none":           float(eval_out["mean_causal_sig_none"]),
        "mean_causal_sig_approach":       float(eval_out["mean_causal_sig_approach"]),
        "mean_causal_sig_contact":        float(eval_out["mean_causal_sig_contact"]),
        "n_none":                         float(eval_out["n_none"]),
        "n_approach":                     float(eval_out["n_approach"]),
        "n_contact":                      float(eval_out["n_contact"]),
        "train_contact_events":           float(tc.get("env_caused_hazard", 0) + tc.get("agent_caused_hazard", 0)),
        "train_approach_events":          float(tc.get("hazard_approach", 0)),
        "crit1_pass":    1.0 if c1_pass else 0.0,
        "crit2_pass":    1.0 if c2_pass else 0.0,
        "crit3_pass":    1.0 if c3_pass else 0.0,
        "crit4_pass":    1.0 if c4_pass else 0.0,
        "crit5_pass":    1.0 if c5_pass else 0.0,
        "criteria_met":  float(n_met),
        "fatal_error_count": 0.0,
    }
    for tt, sig in eval_out["by_ttype"].items():
        metrics[f"causal_sig_ttype_{tt.replace(' ','_')}"] = float(sig)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-032b -- MECH-102: Energy Escalation Ladder (ttype split, random policy)

**Status:** {status}
**Claims:** MECH-102, ARC-024, SD-003
**World:** CausalGridWorldV2 (6 hazards, 3 resources)
**Policy:** RANDOM (avoids EMA-avoidance failure mode of EXQ-032)
**Split:** transition_type -> none / hazard_approach / contact
**alpha_world:** {alpha_world}  (SD-008)
**Seed:** {seed}

## Design Rationale

EXQ-032 used E3-guided (harm-minimizing) policy and split by harm_exposure EMA.
FAIL: the ethical policy was so effective that harm_exposure never exceeded 0.20
(n_high=0). No viability threat was measurable.

EXQ-032b replaces:
1. **Policy**: random -> agent naturally enters all ttype states
2. **Split**: harm_exposure EMA -> transition_type (directly reflects state-space energy)

The random policy is not less ethical -- MECH-102 is about the *environment constraining
the option space*, not about unethical choices. With random policy, when the agent
happens to step into a hazard (agent_caused_hazard), that action had higher causal_sig
than alternatives (step away). The escalation ladder tests whether state-level energy
(ttype) predicts action-level consequentiality (causal_sig).

## Results -- Energy Escalation Ladder

| State Energy Level | causal_sig | n steps |
|---|---|---|
| none (safe locomotion) | {eval_out['mean_causal_sig_none']:.6f} | {eval_out['n_none']} |
| hazard_approach (medium) | {eval_out['mean_causal_sig_approach']:.6f} | {eval_out['n_approach']} |
| contact (high -- agent+env) | {eval_out['mean_causal_sig_contact']:.6f} | {eval_out['n_contact']} |

- **world_forward R2**: {wf_r2:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: causal_sig_contact > causal_sig_none (escalation from safe to contact) | {"PASS" if c1_pass else "FAIL"} | {eval_out['mean_causal_sig_contact']:.6f} vs {eval_out['mean_causal_sig_none']:.6f} |
| C2: causal_sig_approach > causal_sig_none (gradient precedes contact) | {"PASS" if c2_pass else "FAIL"} | {eval_out['mean_causal_sig_approach']:.6f} vs {eval_out['mean_causal_sig_none']:.6f} |
| C3: causal_sig_contact > 0.001 (positive signal at contact) | {"PASS" if c3_pass else "FAIL"} | {eval_out['mean_causal_sig_contact']:.6f} |
| C4: world_forward_r2 > 0.05 | {"PASS" if c4_pass else "FAIL"} | {wf_r2:.4f} |
| C5: n_contact >= 50 | {"PASS" if c5_pass else "FAIL"} | {eval_out['n_contact']} |

Criteria met: {n_met}/5 -> **{status}**
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
    result["claim"]  = CLAIM_IDS[0]
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
