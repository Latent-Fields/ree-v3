#!/opt/local/bin/python3
"""
V3-EXQ-449 -- SD-016 cue_action_proj wiring diagnostic probe

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  Why does EXQ-418a / 418b produce action_bias_divergence=0.0 with sd016_enabled=True
  and terrain_loss active? Hypothesis: cue_action_proj receives NO training signal.

ROOT CAUSE (to confirm):
  agent.py line 694 detaches action_bias before CEM. Even without detach, CEM selection
  is a non-differentiable argmax, so trajectory-selection gradient cannot reach
  cue_action_proj. ree-v3/CLAUDE.md:735 asserts "implicit via E3 trajectory selection
  gradient (no new loss)" -- this probe tests that claim directly.

DESIGN:
  Two arms, 2 seeds each. Training loop mirrors EXQ-418a (WITHOUT_SLEEP condition,
  no sleep cycles, terrain_loss active, shy_enabled=False).

  ARM_BASELINE:   EXACTLY EXQ-418a's training loop. No auxiliary cue_action_loss.
  ARM_SUPERVISED: Same + auxiliary loss:
                    L_cue_action = MSE(cue_action_proj(cue_context_live),
                                       E2.action_object(z_world, a_executed).detach())
                  where cue_context_live is re-extracted WITHOUT detach so grad flows
                  back through cue_action_proj -> output_proj -> world_query_proj.

INSTRUMENTATION:
  - Backward hook on e1.cue_action_proj.weight records grad.norm() per step.
  - Weight delta ||w_final - w_init||_F per arm per seed.
  - action_bias_divergence across SAFE vs DANGEROUS contexts (as EXQ-418a).

ACCEPTANCE CRITERIA:
  C1 BASELINE zero-signal confirmed:
       mean_grad_norm(BASELINE) < 1e-7  AND  action_bias_div(BASELINE) < 0.01
       in both seeds.
  C2 SUPERVISED path works:
       mean_grad_norm(SUPERVISED) > 1e-5  AND  action_bias_div(SUPERVISED) > 0.05
       in both seeds.
  C3 Weight drift ratio:
       ||w_delta(SUPERVISED)|| / ||w_delta(BASELINE)|| > 10  in both seeds.

PASS: C1 AND C2. C3 is secondary.

A PASS confirms the diagnosis (BASELINE wiring gap is real) AND that supervising
cue_action_proj is a viable fix. If C1 fails (BASELINE shows non-zero grad), the
diagnosis is wrong -- some other gradient path exists and we must re-investigate.
If C2 fails (supervised arm also zero), something deeper is broken in the forward
pass and the fix strategy must change.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE    = "v3_exq_449_sd016_cue_action_proj_wiring_probe"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES          = 25
P1_EPISODES          = 50
EVAL_EPISODES        = 12
STEPS_PER_EPISODE    = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN       = 0.1
LAMBDA_CUE_ACTION    = 0.5    # SUPERVISED arm only

LR    = 1e-4
SEEDS = [42, 49]


def get_hazard_max(obs_dict, world_obs):
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, "shape") and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, "shape"):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def compute_terrain_loss(agent, z_world, hazard_max):
    _, terrain_weight = agent.e1.extract_cue_context(z_world)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    target = torch.tensor([[w_harm_target, w_goal_target]],
                          dtype=terrain_weight.dtype,
                          device=terrain_weight.device)
    return F.mse_loss(terrain_weight, target)


def compute_cue_action_loss(agent, z_world, action):
    """Auxiliary supervision: cue_action_proj output should reconstruct the
    action_object that E2 actually produces for the executed action, given
    the current z_world.

    action_object(z_world, action) is a low-level feature of the realised
    transition. Training cue_action_proj(cue_context) to predict it forces
    the projection to encode context-conditioned affordance information.
    """
    action_bias, _ = agent.e1.extract_cue_context(z_world)
    with torch.no_grad():
        ao_target = agent.e2.action_object(z_world.detach(), action.detach())
    return F.mse_loss(action_bias, ao_target.detach())


def _make_env_safe(seed):
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=1, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed):
    return CausalGridWorldV2(
        seed=seed + 1000, size=8, num_hazards=5, num_resources=3,
        hazard_harm=0.04, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sd016_enabled=True,
        sws_enabled=False,
        rem_enabled=False,
        shy_enabled=False,
    )
    return REEAgent(cfg)


def _onehot(idx, n, device):
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _extract_action_bias_eval(agent, z_world):
    if not hasattr(agent.e1, "world_query_proj"):
        return None
    with torch.no_grad():
        z_w = z_world.detach()
        if z_w.dim() == 1:
            z_w = z_w.unsqueeze(0)
        action_bias, _ = agent.e1.extract_cue_context(z_w)
        return action_bias.squeeze(0)


def _action_bias_divergence(safe_biases, dang_biases):
    if len(safe_biases) < 10 or len(dang_biases) < 10:
        return 0.0
    with torch.no_grad():
        smat = torch.stack(safe_biases[:50])
        dmat = torch.stack(dang_biases[:50])
        sn = F.normalize(smat, dim=-1)
        dn = F.normalize(dmat, dim=-1)
        sim = torch.mm(sn, dn.t())
        return max(0.0, 1.0 - float(sim.mean().item()))


def _run_training_episode(agent, env, optimizer, arm, grad_norms):
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_steps = 0

    for _step in range(STEPS_PER_EPISODE):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)

        latent = agent.sense(ob, ow)
        agent.clock.advance()

        hazard_max = get_hazard_max(obs_dict, ow)
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, _info, obs_dict = env.step(action)
        ep_steps += 1

        pred_loss = agent.compute_prediction_loss()
        t_loss    = compute_terrain_loss(agent, latent.z_world, hazard_max)
        total_loss = pred_loss + LAMBDA_TERRAIN * t_loss

        if arm == "SUPERVISED":
            ca_loss = compute_cue_action_loss(agent, latent.z_world, action)
            total_loss = total_loss + LAMBDA_CUE_ACTION * ca_loss

        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            # Capture cue_action_proj grad BEFORE clip for fidelity.
            w = agent.e1.cue_action_proj.weight
            if w.grad is not None:
                grad_norms.append(float(w.grad.norm().item()))
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    return ep_steps


def _run_eval_episode(agent, env, label, safe_biases, dang_biases):
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(STEPS_PER_EPISODE):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)

        latent = agent.sense(ob, ow)
        agent.clock.advance()

        if latent.z_world is not None:
            bias = _extract_action_bias_eval(agent, latent.z_world)
            if bias is not None:
                if label == "SAFE":
                    safe_biases.append(bias.cpu())
                else:
                    dang_biases.append(bias.cpu())

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            break


def run_arm(arm, seed, dry_run=False):
    torch.manual_seed(seed)
    random.seed(seed)

    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    ev = EVAL_EPISODES if not dry_run else 4

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    w_init = agent.e1.cue_action_proj.weight.detach().clone()
    grad_norms: List[float] = []

    print(f"  [arm] arm={arm} seed={seed} p0={p0} p1={p1} eval={ev}", flush=True)

    for ep in range(p0):
        use_dang = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, arm, grad_norms)

    for ep in range(p1):
        abs_ep = p0 + ep
        use_dang = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, arm, grad_norms)

    w_final = agent.e1.cue_action_proj.weight.detach().clone()
    weight_delta_norm = float((w_final - w_init).norm().item())

    safe_biases: List[torch.Tensor] = []
    dang_biases: List[torch.Tensor] = []
    for i in range(ev):
        if i % 2 == 0:
            _run_eval_episode(agent, env_safe, "SAFE", safe_biases, dang_biases)
        else:
            _run_eval_episode(agent, env_dang, "DANGEROUS", safe_biases, dang_biases)

    action_bias_div = _action_bias_divergence(safe_biases, dang_biases)
    mean_grad_norm = float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0
    max_grad_norm  = float(max(grad_norms)) if grad_norms else 0.0
    nonzero_frac   = (sum(1 for g in grad_norms if g > 1e-10) / len(grad_norms)) if grad_norms else 0.0

    print(
        f"  verdict: arm={arm} seed={seed}"
        f" mean_grad_norm={mean_grad_norm:.3e}"
        f" max_grad_norm={max_grad_norm:.3e}"
        f" nonzero_frac={nonzero_frac:.3f}"
        f" weight_delta_norm={weight_delta_norm:.4e}"
        f" action_bias_div={action_bias_div:.4f}"
        f" n_safe={len(safe_biases)} n_dang={len(dang_biases)}",
        flush=True,
    )

    return {
        "arm": arm,
        "seed": seed,
        "mean_grad_norm": mean_grad_norm,
        "max_grad_norm": max_grad_norm,
        "nonzero_grad_fraction": nonzero_frac,
        "weight_delta_norm": weight_delta_norm,
        "action_bias_divergence": action_bias_div,
        "n_safe_bias_samples": len(safe_biases),
        "n_dang_bias_samples": len(dang_biases),
        "n_grad_samples": len(grad_norms),
    }


def main(dry_run=False):
    arms = ["BASELINE", "SUPERVISED"]
    all_results: Dict[str, List[Dict]] = {a: [] for a in arms}

    for arm in arms:
        for seed in SEEDS:
            print(f"[arm] arm={arm} seed={seed}", flush=True)
            res = run_arm(arm, seed, dry_run=dry_run)
            all_results[arm].append(res)

    BASELINE_GRAD_THRESH = 1e-7
    BASELINE_BIAS_THRESH = 0.01
    SUPERVISED_GRAD_THRESH = 1e-5
    SUPERVISED_BIAS_THRESH = 0.05
    DRIFT_RATIO_THRESH = 10.0

    c1_pass_seeds = 0
    c2_pass_seeds = 0
    c3_pass_seeds = 0
    per_seed_comparisons = []

    for base_r, sup_r in zip(all_results["BASELINE"], all_results["SUPERVISED"]):
        assert base_r["seed"] == sup_r["seed"]
        s = base_r["seed"]

        c1 = (base_r["mean_grad_norm"] < BASELINE_GRAD_THRESH
              and base_r["action_bias_divergence"] < BASELINE_BIAS_THRESH)
        c2 = (sup_r["mean_grad_norm"] > SUPERVISED_GRAD_THRESH
              and sup_r["action_bias_divergence"] > SUPERVISED_BIAS_THRESH)
        base_wd = base_r["weight_delta_norm"]
        c3 = (base_wd > 1e-10
              and (sup_r["weight_delta_norm"] / base_wd) > DRIFT_RATIO_THRESH)

        c1_pass_seeds += int(c1)
        c2_pass_seeds += int(c2)
        c3_pass_seeds += int(c3)

        per_seed_comparisons.append({
            "seed": s,
            "baseline_mean_grad_norm": base_r["mean_grad_norm"],
            "supervised_mean_grad_norm": sup_r["mean_grad_norm"],
            "baseline_action_bias_div": base_r["action_bias_divergence"],
            "supervised_action_bias_div": sup_r["action_bias_divergence"],
            "baseline_weight_delta_norm": base_r["weight_delta_norm"],
            "supervised_weight_delta_norm": sup_r["weight_delta_norm"],
            "drift_ratio": (sup_r["weight_delta_norm"] / base_wd) if base_wd > 1e-12 else float("inf"),
            "c1_baseline_zero_signal": c1,
            "c2_supervised_wiring_works": c2,
            "c3_drift_ratio_gt_10": c3,
        })

    n_seeds = len(SEEDS)
    c1_pass = c1_pass_seeds == n_seeds
    c2_pass = c2_pass_seeds == n_seeds
    c3_pass = c3_pass_seeds == n_seeds
    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    print(
        f"verdict: C1_baseline_zero_signal={c1_pass} ({c1_pass_seeds}/{n_seeds}),"
        f" C2_supervised_wiring_works={c2_pass} ({c2_pass_seeds}/{n_seeds}),"
        f" C3_drift_ratio={c3_pass} ({c3_pass_seeds}/{n_seeds})"
        f" => {outcome}",
        flush=True,
    )

    # Interpret the result for downstream governance.
    if c1_pass and c2_pass:
        diagnosis = (
            "confirmed: cue_action_proj receives no gradient under baseline wiring "
            "(detached action_bias + non-differentiable CEM argmax). "
            "Auxiliary supervision restores training signal and produces non-zero "
            "action_bias divergence. Fix path: add a dedicated cue_action_loss "
            "analogous to terrain_loss, or replace CEM with a differentiable "
            "approximation (Gumbel-softmax) plus non-detached action_bias."
        )
    elif not c1_pass:
        diagnosis = (
            "diagnosis rejected: baseline arm shows non-zero gradient on "
            "cue_action_proj.weight. A gradient path exists that was not "
            "identified. Re-investigate computation graph before any fix."
        )
    else:
        diagnosis = (
            "supervised arm failed to train cue_action_proj. Something in the "
            "forward pass or loss construction is broken; the fix strategy must "
            "change. Inspect gradient hooks, loss magnitudes, and extract_cue_context."
        )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "diagnostic_probe",
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "Diagnostic probe of cue_action_proj wiring under sd016_enabled. "
            "Outcome should NOT be used to update SD-016 claim confidence. Result "
            "informs whether the existing wiring has any training signal path and "
            "whether auxiliary supervision is a viable fix. " + diagnosis
        ),
        "acceptance_checks": {
            "C1_baseline_zero_signal": c1_pass,
            "C1_pass_seeds": c1_pass_seeds,
            "C2_supervised_wiring_works": c2_pass,
            "C2_pass_seeds": c2_pass_seeds,
            "C3_drift_ratio_gt_10": c3_pass,
            "C3_pass_seeds": c3_pass_seeds,
            "primary_pass": c1_pass and c2_pass,
            "secondary_pass": c3_pass,
            "baseline_grad_threshold": BASELINE_GRAD_THRESH,
            "baseline_bias_threshold": BASELINE_BIAS_THRESH,
            "supervised_grad_threshold": SUPERVISED_GRAD_THRESH,
            "supervised_bias_threshold": SUPERVISED_BIAS_THRESH,
            "drift_ratio_threshold": DRIFT_RATIO_THRESH,
        },
        "diagnosis_summary": diagnosis,
        "per_seed_comparisons": per_seed_comparisons,
        "all_results": all_results,
        "params": {
            "p0_episodes": P0_EPISODES if not dry_run else 2,
            "p1_episodes": P1_EPISODES if not dry_run else 3,
            "eval_episodes": EVAL_EPISODES if not dry_run else 4,
            "steps_per_episode": STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain": LAMBDA_TERRAIN,
            "lambda_cue_action": LAMBDA_CUE_ACTION,
            "seeds": SEEDS,
            "sd016_enabled": True,
            "sws_enabled": False,
            "rem_enabled": False,
            "shy_enabled": False,
            "dry_run": dry_run,
        },
    }

    if not dry_run:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "REE_assembly", "evidence", "experiments",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_id}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {out_path}", flush=True)
    else:
        out_path = None
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)

    if not dry_run:
        from experiment_protocol import emit_outcome
        emit_outcome(outcome=outcome, manifest_path=str(out_path))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal episodes to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
