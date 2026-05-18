#!/opt/local/bin/python3
"""
V3-EXQ-418h -- SD-016 env-entropy precondition probe (SD-023 landmarks ON vs OFF)

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  V3-EXQ-418f + V3-EXQ-418g (2026-04-28) together established that
  SD-016's mechanism (cue-indexed retrieval over a stored set of
  associations) does not produce a behavioural delta in the current
  CausalGridWorldV2 configs because z_world produces near-identical
  vectors across observations. EXQ-418f probe: pairwise cos(z_world)
  ~ 0.998 across 64 random-policy samples; slot_argmax_histogram =
  [0,0,0,0,64,0,...] (slot 4 wins ALL samples). EXQ-418g 4-arm:
  attention can be made delta-peaky (B1/B3 attn_entropy = 0.000)
  AND slots can be perfectly orthogonal (B2/B3 slot_diversity =
  1.000) yet action_class_entropy stays at 1.105e-10 IDENTICALLY
  across all four arms. The substrate works as designed; the
  bottleneck is upstream -- the env does not generate enough
  cross-context z_world variation for any retrieval substrate to
  exercise.

  The user's diagnosis (2026-04-28): "the world is too simple. not
  enough entropy in every single configuration to support as much
  information extraction". SD-016 is now parked pending an
  env-entropy precondition.

  This probe asks: do the existing-but-unused SD-023 landmarks
  (n_landmarks_a, n_landmarks_b in CausalGridWorldV2) supply enough
  additional cross-context z_world variation to satisfy the
  precondition? The 418e/418f/418g configs all set
  n_landmarks_a=0, n_landmarks_b=0 (defaults); SD-023 was
  implemented 2026-04-09 but has never been enabled in an SD-016
  context.

DESIGN:
  Two arms, two seeds [42, 43]. Each arm trains an agent for the
  same shortened schedule used in EXQ-418f (P0=10 P1=20, ~1 min on
  Mac), then constructs TWO eval batches per arm:
    safe batch       (16 samples) -- random-policy rollout in safe env
    dangerous batch  (16 samples) -- random-policy rollout in dangerous env

  Reports per arm:
    cos_within_safe       mean pairwise cos(z_world_safe_i, z_world_safe_j)
    cos_within_danger     mean pairwise cos(z_world_dang_i, z_world_dang_j)
    cos_cross             mean pairwise cos(z_world_safe_i, z_world_dang_j)
    separation_gap        mean(within_safe, within_danger) - cos_cross
    z_world_norm_stats    mean / std / min / max across the combined 32

  Arms:
    H0_landmarks_off    n_landmarks_a=0   n_landmarks_b=0  (current default,
                          replicates EXQ-418f sampling regime)
    H1_landmarks_on     n_landmarks_a=3   n_landmarks_b=3  (SD-023 design)

  Both arms hold sd016_enabled=True for parity; world_query_proj is
  not exercised (this is a z_world probe, not a substrate retest).

ACCEPTANCE CRITERIA:
  E1  cos_cross < 0.95 in H1_landmarks_on, both seeds. Substrate-
      readiness gate: at least the env-on config must produce
      cross-context z_world separation that the cosine-attention
      substrate could plausibly retrieve over.
  E2  separation_gap(H1) > separation_gap(H0) + 0.02 across the
      seed mean. Confirms the landmark addition contributes
      cross-context variation rather than noise.
  E3  H0 reproduces the EXQ-418f / 418g failure mode: cos_cross(H0)
      >= 0.95 (sanity check that the eval batch construction is
      sensitive enough to detect the regime that failed before).

  PASS = E1 AND E2 AND E3.
  PASS unblocks SD-016 retest under H1 env config (queue an
  EXQ-418i 4-arm reusing 418g's substrate matrix on the
  SD-023-enabled env).
  FAIL routes to broader env enrichment scoping (substrate_queue
  SD-016 stays parked; open a thought-doc on multi-context env
  design).

  PASS DOES NOT update SD-016 claim confidence directly --
  experiment_purpose=diagnostic, evidence_class=diagnostic_probe.
  It updates the env-readiness gate state, which then licenses
  (or refuses to license) downstream substrate retests.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418g (in spirit -- this is the precondition the
  4-arm cannot satisfy in the current env)
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE    = "v3_exq_418h_sd016_env_entropy_precondition"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES        = 10
P1_EPISODES        = 20
STEPS_PER_EPISODE  = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN     = 0.1
LAMBDA_CUE_ACTION  = 0.5
EVAL_BATCH_PER_CTX = 16
LR                 = 1e-4
SEEDS              = [42, 43]

# Arm spec: (label, n_landmarks_a, n_landmarks_b).
ARMS: List[Tuple[str, int, int]] = [
    ("H0_landmarks_off", 0, 0),
    ("H1_landmarks_on",  3, 3),
]

# Acceptance thresholds.
E1_CROSS_COSINE_CEILING = 0.95
E2_SEPARATION_GAP_DELTA = 0.02
E3_H0_CROSS_COSINE_FLOOR = 0.95


def _make_env_safe(seed: int, n_a: int, n_b: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=1, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
        n_landmarks_a=n_a, n_landmarks_b=n_b,
    )


def _make_env_dangerous(seed: int, n_a: int, n_b: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=8, num_hazards=5, num_resources=3,
        hazard_harm=0.04, use_proxy_fields=True,
        resource_respawn_on_consume=True,
        n_landmarks_a=n_a, n_landmarks_b=n_b,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9, alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="off",
        sws_enabled=False, rem_enabled=False, shy_enabled=False,
    )
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


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
    action_bias, _ = agent.e1.extract_cue_context(z_world)
    with torch.no_grad():
        ao_target = agent.e2.action_object(z_world.detach(), action.detach())
    return F.mse_loss(action_bias, ao_target.detach())


def _run_training_episode(agent, env, optimizer, phase: str) -> int:
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
        if ob.dim() == 1: ob = ob.unsqueeze(0)
        if ow.dim() == 1: ow = ow.unsqueeze(0)

        latent = agent.sense(ob, ow)
        agent._self_experience_buffer.append(latent.z_self.detach().clone())
        agent._world_experience_buffer.append(latent.z_world.detach().clone())
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
        if phase == "P1":
            ca_loss = compute_cue_action_loss(agent, latent.z_world, action)
            total_loss = total_loss + LAMBDA_CUE_ACTION * ca_loss

        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    return ep_steps


def _collect_z_world_batch(agent, env, n_samples: int) -> torch.Tensor:
    device = agent.device
    out: List[torch.Tensor] = []
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    for _ in range(STEPS_PER_EPISODE * 4):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1: ob = ob.unsqueeze(0)
        if ow.dim() == 1: ow = ow.unsqueeze(0)
        latent = agent.sense(ob, ow)
        out.append(latent.z_world.detach().clone().squeeze(0))
        if len(out) >= n_samples:
            break
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
    return torch.stack(out[:n_samples], dim=0)


def _mean_pairwise_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean pairwise cosine across batches a and b. If a is b (same object),
    compute over off-diagonal. Otherwise compute the full cross-product mean.
    """
    with torch.no_grad():
        an = F.normalize(a, dim=-1)
        bn = F.normalize(b, dim=-1)
        sim = an @ bn.t()
        if a.data_ptr() == b.data_ptr():
            n = sim.shape[0]
            mask = 1 - torch.eye(n, device=sim.device)
            return float((sim * mask).sum().item() / max(1, n * (n - 1)))
        return float(sim.mean().item())


def _norm_stats(t: torch.Tensor) -> Dict[str, float]:
    n = t.norm(dim=-1)
    return {
        "mean": float(n.mean().item()),
        "std":  float(n.std().item()),
        "min":  float(n.min().item()),
        "max":  float(n.max().item()),
    }


def _run_one_arm_seed(arm_label: str, n_a: int, n_b: int, seed: int,
                     p0: int, p1: int, eval_n: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed, n_a, n_b)
    env_dang = _make_env_dangerous(seed, n_a, n_b)

    agent = _make_agent(env_safe)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    print(
        f"  [arm {arm_label} seed {seed}] n_lm_a={n_a} n_lm_b={n_b} "
        f"world_obs_dim={env_safe.world_obs_dim} p0={p0} p1={p1}",
        flush=True,
    )

    for ep in range(p0):
        env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P0")
    for ep in range(p1):
        abs_ep = p0 + ep
        env = env_dang if (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P1")

    safe_batch = _collect_z_world_batch(agent, env_safe, eval_n)
    dang_batch = _collect_z_world_batch(agent, env_dang, eval_n)

    cos_within_safe   = _mean_pairwise_cos(safe_batch, safe_batch)
    cos_within_danger = _mean_pairwise_cos(dang_batch, dang_batch)
    cos_cross         = _mean_pairwise_cos(safe_batch, dang_batch)
    within_avg        = 0.5 * (cos_within_safe + cos_within_danger)
    separation_gap    = within_avg - cos_cross

    combined = torch.cat([safe_batch, dang_batch], dim=0)
    norm_stats = _norm_stats(combined)

    print(
        f"  [arm {arm_label} seed {seed}] "
        f"cos_within_safe={cos_within_safe:.4f} "
        f"cos_within_danger={cos_within_danger:.4f} "
        f"cos_cross={cos_cross:.4f} "
        f"sep_gap={separation_gap:.4f} "
        f"z_norm_mean={norm_stats['mean']:.3f}",
        flush=True,
    )

    return {
        "arm":               arm_label,
        "n_landmarks_a":     n_a,
        "n_landmarks_b":     n_b,
        "world_obs_dim":     env_safe.world_obs_dim,
        "seed":              seed,
        "cos_within_safe":   cos_within_safe,
        "cos_within_danger": cos_within_danger,
        "cos_cross":         cos_cross,
        "separation_gap":    separation_gap,
        "z_world_norm":      norm_stats,
    }


def _per_arm_summary(rs: List[Dict]) -> Dict:
    n = len(rs)
    return {
        "n_seeds":                  n,
        "cos_within_safe_mean":     sum(r["cos_within_safe"] for r in rs) / n,
        "cos_within_danger_mean":   sum(r["cos_within_danger"] for r in rs) / n,
        "cos_cross_mean":           sum(r["cos_cross"] for r in rs) / n,
        "separation_gap_mean":      sum(r["separation_gap"] for r in rs) / n,
        "cos_cross_max":            max(r["cos_cross"] for r in rs),
        "cos_cross_min":            min(r["cos_cross"] for r in rs),
        "z_world_norm_mean_mean":   sum(r["z_world_norm"]["mean"] for r in rs) / n,
    }


def _evaluate_acceptance(per_arm: Dict[str, Dict], per_seed: Dict[str, List[Dict]]) -> Dict:
    h0 = per_arm.get("H0_landmarks_off", {})
    h1 = per_arm.get("H1_landmarks_on", {})

    # E1: cos_cross < 0.95 in H1, ALL seeds.
    h1_seeds = per_seed.get("H1_landmarks_on", [])
    e1_per_seed = [r["cos_cross"] < E1_CROSS_COSINE_CEILING for r in h1_seeds]
    e1_pass = bool(h1_seeds) and all(e1_per_seed)

    # E2: separation_gap delta H1 - H0 > 0.02 (mean across seeds).
    e2_delta = h1.get("separation_gap_mean", 0.0) - h0.get("separation_gap_mean", 0.0)
    e2_pass = e2_delta > E2_SEPARATION_GAP_DELTA

    # E3: H0 reproduces failure (cos_cross >= 0.95 in mean).
    e3_pass = h0.get("cos_cross_mean", 0.0) >= E3_H0_CROSS_COSINE_FLOOR

    overall_pass = e1_pass and e2_pass and e3_pass

    return {
        "E1_h1_cross_cosine_below_ceiling_all_seeds": {
            "pass": e1_pass,
            "per_seed": e1_per_seed,
            "ceiling": E1_CROSS_COSINE_CEILING,
            "h1_cos_cross_per_seed": [r["cos_cross"] for r in h1_seeds],
        },
        "E2_landmarks_increase_separation_gap": {
            "pass": e2_pass,
            "delta_h1_minus_h0": e2_delta,
            "threshold": E2_SEPARATION_GAP_DELTA,
            "h0_separation_gap_mean": h0.get("separation_gap_mean", 0.0),
            "h1_separation_gap_mean": h1.get("separation_gap_mean", 0.0),
        },
        "E3_h0_reproduces_failure_regime": {
            "pass": e3_pass,
            "floor": E3_H0_CROSS_COSINE_FLOOR,
            "h0_cos_cross_mean": h0.get("cos_cross_mean", 0.0),
        },
        "overall_pass": overall_pass,
    }


def main(dry_run: bool = False):
    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    eval_n = EVAL_BATCH_PER_CTX if not dry_run else 4

    print(
        f"V3-EXQ-418h SD-016 env-entropy precondition probe "
        f"(seeds={SEEDS} dry_run={dry_run})",
        flush=True,
    )

    all_results: List[Dict] = []
    per_arm: Dict[str, List[Dict]] = {arm: [] for arm, _, _ in ARMS}

    for arm_label, n_a, n_b in ARMS:
        for seed in SEEDS:
            r = _run_one_arm_seed(arm_label, n_a, n_b, seed, p0, p1, eval_n)
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries = {arm: _per_arm_summary(rs) for arm, rs in per_arm.items()}
    acceptance = _evaluate_acceptance(summaries, per_arm)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(
        f"  [summary] "
        f"E1={acceptance['E1_h1_cross_cosine_below_ceiling_all_seeds']['pass']} "
        f"E2={acceptance['E2_landmarks_increase_separation_gap']['pass']} "
        f"E3={acceptance['E3_h0_reproduces_failure_regime']['pass']} "
        f"-> outcome={outcome}",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    output = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "claim_ids_tested":   CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class":     "diagnostic_probe",
        "outcome":            outcome,
        "timestamp_utc":      ts,
        "supersedes":         "V3-EXQ-418g",
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "SD-016 env-entropy precondition probe. PURPOSE: test whether "
            "SD-023 landmarks (n_landmarks_a, n_landmarks_b in CausalGridWorldV2) "
            "supply enough cross-context z_world variation to satisfy the "
            "substrate-readiness gate that EXQ-418f/g established was missing. "
            "PASS = SD-023 landmarks ON produce cross-context cos_cross < 0.95 "
            "with separation_gap > H0+0.02 AND H0 (landmarks OFF) reproduces "
            "the EXQ-418f/g failure regime (cos_cross >= 0.95). PASS unblocks "
            "SD-016 retest in the H1 env config (queue EXQ-418i 4-arm reusing "
            "418g substrate matrix). FAIL routes to broader env enrichment "
            "scoping. MUST NOT update SD-016 claim confidence directly -- "
            "this is an env-readiness gate probe, not a substrate retest."
        ),
        "acceptance_checks":  acceptance,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "e1_cross_cosine_ceiling":   E1_CROSS_COSINE_CEILING,
            "e2_separation_gap_delta":   E2_SEPARATION_GAP_DELTA,
            "e3_h0_cross_cosine_floor":  E3_H0_CROSS_COSINE_FLOOR,
        },
        "params": {
            "seeds":                SEEDS,
            "p0_episodes":          p0,
            "p1_episodes":          p1,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":       LAMBDA_TERRAIN,
            "lambda_cue_action":    LAMBDA_CUE_ACTION,
            "eval_batch_per_ctx":   eval_n,
            "lr":                   LR,
            "arms": [
                {"label": label, "n_landmarks_a": n_a, "n_landmarks_b": n_b}
                for label, n_a, n_b in ARMS
            ],
            "sd016_enabled":        True,
            "sd016_writepath_mode": "off",
            "dry_run":              dry_run,
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
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"verdict: {outcome}", flush=True)
    print(f"Outcome: {outcome}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal episodes to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
