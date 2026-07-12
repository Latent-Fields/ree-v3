#!/opt/local/bin/python3
"""
V3-EXQ-418i -- SD-016 Path 1 div_weight sweep (1.0 / 2.0 / 5.0)

EXPERIMENT_PURPOSE: evidence

QUESTION:
  EXQ-418e (div_weight=0.5) established that the auxiliary diversification
  loss orthogonalizes slot content (slot_diversity_mean = 0.9999 in the
  A2_div_only arm) but CANNOT move attention away from the uniform rail
  (attn_entropy_mean pinned at ln(16) = 2.7726). EXQ-418g confirmed the
  root cause: the bottleneck is query selectivity, not slot content. The
  world_query_proj produces nearly identical q vectors across observations
  because the CausalGridWorldV2 environment does not supply enough
  cross-context z_world variation for retrieval to exercise (EXQ-418h:
  pairwise cos(z_world) ~ 0.996 with or without SD-023 landmarks).

  However, the diversification loss also applies a gradient through
  key_proj (slots are differentiated via their key projections when
  key_proj.bias=False is not set). It is therefore possible -- though
  not guaranteed -- that a STRONGER diversification gradient (at weights
  1.0, 2.0, 5.0) reshapes key_proj enough to create non-trivial
  inner-product variation with world_query_proj outputs, thereby moving
  attention even in the low-entropy environment.

  This sweep tests whether there is a div_weight threshold above 0.5 at
  which Path 1 alone begins to break the uniform attention rail.

DESIGN:
  Four arms, three seeds [42, 43, 44].
  All arms: writepath_mode="off", sd016_enabled=True, sd016_temperature_learnable=False.
  "Off" isolates the diversification gradient from any write-side confound
  (matching EXQ-418e A2_div_only, which is the cleanest ablation).

    A0_baseline   div_weight=0.0   -- replicates EXQ-418e A2_div_only baseline
    A1_div_1p0    div_weight=1.0
    A2_div_2p0    div_weight=2.0
    A3_div_5p0    div_weight=5.0

  Training schedule matches EXQ-418e (P0=20 P1=40 STEPS=150
  LAMBDA_TERRAIN=0.1 LAMBDA_CUE_ACTION=0.5).
  key_proj.bias=False not explicitly set here (default True) to allow
  key_proj to adapt its bias -- the sweep is testing whether any
  div_weight reshapes the joint {memory, key_proj} representation.

PER-ARM METRICS:
  attn_entropy_mean       attention entropy over eval batch;
                          uniform reference ln(16) ~= 2.7726.
  slot_diversity          1 - mean off-diagonal pairwise cosine similarity
                          of cm.memory rows post-train.
  action_class_entropy    entropy of cue_action_proj argmax over eval batch.

ACCEPTANCE CRITERIA:
  C1  attn_entropy_mean < 2.65 in ALL seeds for AT LEAST ONE of
      A1_div_1p0, A2_div_2p0, A3_div_5p0. (~5% reduction from uniform.)
      PASS = Path 1 CAN break the uniform rail at some weight > 0.5.
  C2  slot_diversity > 0.10 in ALL seeds for ALL non-baseline arms.
      Sanity check that stronger div_weight does not collapse slots.
  C3  |action_class_entropy[best_arm] - action_class_entropy[A0]| >= 0.20
      nats, where best_arm = arm with lowest attn_entropy_mean_mean.
      Confirms attention delta translates to behavioural change.
  C4  A0_baseline replicates EXQ-418e A2_div_only failure:
      attn_entropy_mean_mean > 2.65 (near-uniform), slot_diversity_mean > 0.90.
      Sanity check that the substrate is consistent.

  PASS: C1 AND C2 AND C3 AND C4.
  C1 FAIL with C2 PASS = Path 1 is categorically insufficient; attention
    bottleneck is entirely in query selectivity regardless of slot pressure.
  C3 FAIL with C1+C2 PASS = attention breaks the uniform rail but the
    behavioural output (cue_action_proj) remains uninformative.
  C4 FAIL = substrate inconsistency; investigate before interpreting other
    criteria.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418e (weight-sweep follow-up to the div_weight=0.5 FAIL)
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_418i_sd016_divweight_sweep"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES         = 20
P1_EPISODES         = 40
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
EVAL_BATCH_SIZE     = 32
LR                  = 1e-4
SEEDS               = [42, 43, 44]

# Arms: (label, div_weight)
ARMS: List[Tuple[str, float]] = [
    ("A0_baseline", 0.0),
    ("A1_div_1p0",  1.0),
    ("A2_div_2p0",  2.0),
    ("A3_div_5p0",  5.0),
]

ATTN_ENTROPY_C1_THRESHOLD   = 2.65
SLOT_DIVERSITY_C2_THRESHOLD = 0.10
ACTION_ENTROPY_C3_DELTA     = 0.20
C4_ATTN_ENTROPY_FLOOR       = 2.65
C4_SLOT_DIV_FLOOR           = 0.90


# ---------------------------------------------------------------------------
# Env + agent helpers (mirrors EXQ-418e).
# ---------------------------------------------------------------------------

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=1, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=8, num_hazards=5, num_resources=3,
        hazard_harm=0.04, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, div_weight: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="off",
        sws_enabled=False,
        rem_enabled=False,
        shy_enabled=False,
    )
    cfg.sd016_diversification_weight = div_weight
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


# ---------------------------------------------------------------------------
# Slot probes.
# ---------------------------------------------------------------------------

def _slot_diversity(cm) -> float:
    with torch.no_grad():
        mn = F.normalize(cm.memory, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        mean_pair = float(off.sum().item() / max(1, n * (n - 1)))
        return max(0.0, 1.0 - mean_pair)


def _attn_entropy_mean(cm, world_query_proj, z_world_batch) -> float:
    with torch.no_grad():
        batch_size = z_world_batch.shape[0]
        memory_dim = cm.memory_dim
        q = world_query_proj(z_world_batch).unsqueeze(1)
        k = cm.key_proj(cm.memory).unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1).squeeze(1)
        probs = weights.clamp(min=1e-12)
        return float(-(probs * probs.log()).sum(dim=-1).mean().item())


def _action_class_entropy(agent, z_world_batch) -> float:
    with torch.no_grad():
        action_bias, _ = agent.e1.extract_cue_context(z_world_batch)
        action_dim = agent.e2.config.action_dim
        cls = action_bias[..., :action_dim].argmax(dim=-1)
        hist = torch.bincount(cls, minlength=action_dim).float()
        if hist.sum() <= 0:
            return 0.0
        probs = hist / hist.sum()
        probs = probs.clamp(min=1e-12)
        return float(-(probs * probs.log()).sum().item())


# ---------------------------------------------------------------------------
# Eval batch collector.
# ---------------------------------------------------------------------------

def _collect_eval_batch(agent, env, n_samples: int) -> torch.Tensor:
    device = agent.device
    z_world_list: List[torch.Tensor] = []
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    for _ in range(STEPS_PER_EPISODE * 4):
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)
        latent = agent.sense(ob, ow)
        z_world_list.append(latent.z_world.detach().clone().squeeze(0))
        if len(z_world_list) >= n_samples:
            break
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
    return torch.stack(z_world_list[:n_samples], dim=0)


# ---------------------------------------------------------------------------
# Training episode.
# ---------------------------------------------------------------------------

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
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)

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

        pred_loss  = agent.compute_prediction_loss()
        t_loss     = compute_terrain_loss(agent, latent.z_world, hazard_max)
        total_loss = pred_loss + LAMBDA_TERRAIN * t_loss

        if phase == "P1":
            ca_loss    = compute_cue_action_loss(agent, latent.z_world, action)
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


# ---------------------------------------------------------------------------
# Per-arm per-seed run.
# ---------------------------------------------------------------------------

def _run_one(arm_label: str, div_weight: float, seed: int,
             p0: int, p1: int, eval_n: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent    = _make_agent(env_safe, div_weight)
    cm       = agent.e1.context_memory
    wqp      = agent.e1.world_query_proj

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    print(f"  [arm {arm_label} seed {seed}] div_weight={div_weight} "
          f"p0={p0} p1={p1}", flush=True)

    for ep in range(p0):
        env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P0")

    for ep in range(p1):
        abs_ep = p0 + ep
        env = env_dang if (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P1")

    z_batch   = _collect_eval_batch(agent, env_safe, eval_n)
    slot_div  = _slot_diversity(cm)
    attn_ent  = _attn_entropy_mean(cm, wqp, z_batch)
    act_ent   = _action_class_entropy(agent, z_batch)

    print(f"  [arm {arm_label} seed {seed}] "
          f"slot_div={slot_div:.3f} attn_ent={attn_ent:.4f} act_ent={act_ent:.4f}",
          flush=True)

    return {
        "arm":                   arm_label,
        "diversification_weight": div_weight,
        "seed":                  seed,
        "slot_diversity":        slot_div,
        "attn_entropy_mean":     attn_ent,
        "action_class_entropy":  act_ent,
    }


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------

def _summarise(arm_results: List[Dict]) -> Dict:
    n = len(arm_results)
    return {
        "n_seeds":                   n,
        "slot_diversity_mean":       sum(r["slot_diversity"] for r in arm_results) / n,
        "slot_diversity_min":        min(r["slot_diversity"] for r in arm_results),
        "slot_diversity_seeds_pass": sum(1 for r in arm_results
                                         if r["slot_diversity"] > SLOT_DIVERSITY_C2_THRESHOLD),
        "attn_entropy_mean_mean":    sum(r["attn_entropy_mean"] for r in arm_results) / n,
        "attn_entropy_mean_min":     min(r["attn_entropy_mean"] for r in arm_results),
        "attn_entropy_seeds_pass":   sum(1 for r in arm_results
                                         if r["attn_entropy_mean"] < ATTN_ENTROPY_C1_THRESHOLD),
        "action_class_entropy_mean": sum(r["action_class_entropy"] for r in arm_results) / n,
    }


def _evaluate(per_arm: Dict[str, Dict]) -> Dict:
    n_seeds    = len(SEEDS)
    non_base   = [lab for lab, _ in ARMS if lab != "A0_baseline"]

    # C1: at least one non-baseline arm breaks attn uniform rail in ALL seeds.
    c1_per_arm = {arm: per_arm[arm]["attn_entropy_seeds_pass"] == n_seeds
                  for arm in non_base if arm in per_arm}
    c1_pass    = any(c1_per_arm.values()) if c1_per_arm else False

    # C2: slot_diversity > threshold in ALL seeds for ALL non-baseline arms.
    c2_per_arm = {arm: per_arm[arm]["slot_diversity_seeds_pass"] == n_seeds
                  for arm in non_base if arm in per_arm}
    c2_pass    = all(c2_per_arm.values()) if c2_per_arm else False

    # C3: behavioural delta between best non-baseline arm and A0.
    a0_act_ent = per_arm.get("A0_baseline", {}).get("action_class_entropy_mean", 0.0)
    best_arm   = min(
        non_base,
        key=lambda arm: per_arm.get(arm, {}).get("attn_entropy_mean_mean", 999.0),
    ) if non_base else None
    best_act_ent = per_arm.get(best_arm, {}).get("action_class_entropy_mean", 0.0) \
                   if best_arm else 0.0
    c3_delta   = abs(best_act_ent - a0_act_ent)
    c3_pass    = c3_delta >= ACTION_ENTROPY_C3_DELTA

    # C4: baseline replicates prior failure regime.
    a0 = per_arm.get("A0_baseline", {})
    c4_attn    = a0.get("attn_entropy_mean_mean", 0.0) > C4_ATTN_ENTROPY_FLOOR
    c4_slot    = a0.get("slot_diversity_mean", 0.0) > C4_SLOT_DIV_FLOOR
    c4_pass    = c4_attn and c4_slot

    return {
        "C1_attn_breaks_uniform_any_arm": {
            "pass":       c1_pass,
            "per_arm":    c1_per_arm,
            "threshold":  ATTN_ENTROPY_C1_THRESHOLD,
        },
        "C2_slot_diversity_survives_all_arms": {
            "pass":       c2_pass,
            "per_arm":    c2_per_arm,
            "threshold":  SLOT_DIVERSITY_C2_THRESHOLD,
        },
        "C3_behavioural_delta_above_0_20": {
            "pass":       c3_pass,
            "best_arm":   best_arm,
            "delta":      c3_delta,
            "threshold":  ACTION_ENTROPY_C3_DELTA,
            "a0_act_ent": a0_act_ent,
            "best_act_ent": best_act_ent,
        },
        "C4_baseline_replicates_prior_fail": {
            "pass":               c4_pass,
            "attn_entropy_mean":  a0.get("attn_entropy_mean_mean", 0.0),
            "attn_floor":         C4_ATTN_ENTROPY_FLOOR,
            "slot_diversity_mean": a0.get("slot_diversity_mean", 0.0),
            "slot_div_floor":     C4_SLOT_DIV_FLOOR,
        },
        "overall_pass": c1_pass and c2_pass and c3_pass and c4_pass,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False):
    p0     = P0_EPISODES     if not dry_run else 2
    p1     = P1_EPISODES     if not dry_run else 3
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    print(f"V3-EXQ-418i SD-016 div_weight sweep "
          f"(seeds={SEEDS} dry_run={dry_run})", flush=True)

    all_results: List[Dict]      = []
    per_arm: Dict[str, List]     = {lab: [] for lab, _ in ARMS}

    for arm_label, div_w in ARMS:
        for seed in SEEDS:
            r = _run_one(arm_label, div_w, seed, p0, p1, eval_n)
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries   = {arm: _summarise(rs) for arm, rs in per_arm.items()}
    acceptance  = _evaluate(summaries)
    outcome     = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(f"  [summary] C1={acceptance['C1_attn_breaks_uniform_any_arm']['pass']} "
          f"C2={acceptance['C2_slot_diversity_survives_all_arms']['pass']} "
          f"C3={acceptance['C3_behavioural_delta_above_0_20']['pass']} "
          f"C4={acceptance['C4_baseline_replicates_prior_fail']['pass']} "
          f"-> outcome={outcome}", flush=True)

    ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    output = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "claim_ids_tested":   CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "timestamp_utc":      ts,
        "supersedes":         "V3-EXQ-418e",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_note": (
            "SD-016 Path 1 div_weight sweep at 1.0/2.0/5.0 (all arms: "
            "writepath_mode=off, temperature_learnable=False). PASS = at "
            "least one weight above 0.5 breaks the uniform attention rail "
            "in all seeds AND produces a behavioural delta, confirming that "
            "Path 1 alone can establish cue-indexed retrieval given strong "
            "enough gradient pressure. FAIL on C1 with C2 PASS = the "
            "attention bottleneck is categorically in query selectivity, "
            "not slot orthogonality; Path 1 alone is insufficient regardless "
            "of weight. FAIL on C1 with C4 FAIL = substrate inconsistency. "
            "FAIL on C3 with C1+C2 PASS = slots differentiated and attention "
            "selective but cue_action_proj output still uninformative."
        ),
        "acceptance_checks":  acceptance,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "attn_entropy_c1_threshold":    ATTN_ENTROPY_C1_THRESHOLD,
            "slot_diversity_c2_threshold":  SLOT_DIVERSITY_C2_THRESHOLD,
            "action_entropy_c3_delta":      ACTION_ENTROPY_C3_DELTA,
            "c4_attn_entropy_floor":        C4_ATTN_ENTROPY_FLOOR,
            "c4_slot_div_floor":            C4_SLOT_DIV_FLOOR,
        },
        "params": {
            "seeds":                SEEDS,
            "p0_episodes":          p0,
            "p1_episodes":          p1,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":       LAMBDA_TERRAIN,
            "lambda_cue_action":    LAMBDA_CUE_ACTION,
            "eval_batch_size":      eval_n,
            "lr":                   LR,
            "arms": [{"label": lab, "div_weight": dw} for lab, dw in ARMS],
            "sd016_enabled":           True,
            "sd016_writepath_mode":    "off",
            "sd016_temperature_learnable": False,
            "sws_enabled":             False,
            "rem_enabled":             False,
            "shy_enabled":             False,
            "dry_run":                 dry_run,
        },
    }

    if not dry_run:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "REE_assembly", "evidence", "experiments",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = write_flat_manifest(
            output,
            out_dir,
            dry_run=False,
            config=output.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"verdict: {outcome}", flush=True)
    print(f"Outcome: {outcome}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
