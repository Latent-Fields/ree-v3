#!/opt/local/bin/python3
"""
V3-EXQ-418k -- SD-016 env-entropy reef fix: 2x2 factorial {SD-016, reef}

EXPERIMENT_PURPOSE: evidence

QUESTION:
  V3-EXQ-418i (2026-05-04) FAILed all four arms with
  action_class_entropy = 1.1052408294132121e-10, confirming the env-entropy
  precondition is not met in the default CausalGridWorldV2: pairwise
  cos(z_world) ~ 0.998 regardless of div_weight (1.0 / 2.0 / 5.0). SD-016's
  ContextMemory query mechanism cannot produce context-dependent behavior when
  z_world is near-constant across all env states.

  V3-EXQ-522 (2026-05-05, PASS) confirmed that reef enrichment (SD-050) reliably
  breaks monostrategy by adding reef_field_view (25 dims) to world_state
  (world_obs_dim: 250->275), creating two behavioral attractors (flee to reef /
  forage) with zone_transitions_per_ep ~49, reef_visit_fraction ~0.5.

  Hypothesis: reef's reef_field_view makes z_world context-dependent (reef zone
  vs food zone), providing the env-entropy precondition required for SD-016's
  ContextMemory queries to produce informative, context-specific outputs and
  behavioral divergence.

DESIGN:
  2x2 factorial: {SD-016: off/on} x {reef: off/on}. Three seeds [42, 43, 44].

    A0_off_no_reef      SD-016 OFF, reef OFF  -- absolute baseline
    A1_sd016_no_reef    SD-016 ON (Path4), reef OFF  -- replicates EXQ-418i failure
    A2_reef_no_sd016    SD-016 OFF, reef ON  -- reef provides behavioral diversity only
    A3_reef_sd016       SD-016 ON (Path4), reef ON  -- PRIMARY TEST

  Path 4 config (same as EXQ-418g B3_sel_plus_div):
    sd016_temperature_learnable=True, attn_entropy_weight=0.05,
    sd016_diversification_weight=0.5

ENV:
  reef OFF arms (A0, A1): size=8, num_hazards=2, hazard_harm=0.02, num_resources=3
  reef ON arms (A2, A3):  same + reef_enabled=True, reef_patch_radius=1,
                          hazard_food_attraction=0.7
  world_obs_dim: 250 (reef OFF) / 275 (reef ON)

PER-ARM METRICS:
  cos_cross           mean pairwise cos(z_world_reef_zone, z_world_food_zone)
  cos_within          mean pairwise cosine within each zone, averaged
  separation_gap      cos_within - cos_cross (higher = more zone separation)
  attn_entropy_mean   mean attention entropy over eval batch; uniform ref ln(16)=2.7726
  slot_diversity      1 - mean off-diagonal pairwise cosine of cm.memory rows
  action_class_entropy entropy of cue_action_proj argmax over eval batch

ACCEPTANCE CRITERIA:
  C0  A1_sd016_no_reef replicates EXQ-418i failure:
      attn_entropy_mean > 2.65 in all 3 seeds (env-entropy bottleneck confirmed).
  C1  A2_reef_no_sd016 AND A3_reef_sd016 show reef zone separation:
      cos_cross < 0.90 in >=2/3 seeds for both arms.
  C2  A3_reef_sd016 breaks uniform attention rail:
      attn_entropy_mean < 2.65 in >=2/3 seeds.
  C3  A3_reef_sd016 produces behavioral delta:
      action_class_entropy > 0.30 nats in >=2/3 seeds.

  PASS = C0 AND C1 AND C2 AND C3.
  C1 PASS + C2 FAIL = reef provides env variation but SD-016 substrate cannot
    leverage it; deeper diagnosis of gradient path needed.
  C1 PASS + C2 PASS + C3 FAIL = attention breaks uniform rail but cue_action_proj
    inert; EXP-0155 gradient path still blocked; supervised loss needed.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418j
bugfix: torch.cat([reef_z, food_z], dim=0) crashed when reef_z=torch.empty(0,1)
  (shape [0,1]) and food_z has shape [n,32]; filter out zero-row tensors before cat.
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


EXPERIMENT_TYPE    = "v3_exq_418k_sd016_context_memory_reef"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES        = 20
P1_EPISODES        = 40
STEPS_PER_EPISODE  = 150
LAMBDA_TERRAIN     = 0.1
LAMBDA_CUE_ACTION  = 0.5
LAMBDA_DIVERSIFY   = 0.5
LAMBDA_ATTN_ENT    = 0.05
EVAL_N_PER_ZONE    = 24  # per zone; collect 2x this, then median-split

LR    = 1e-4
SEEDS = [42, 43, 44]

# Arm spec: (label, sd016_on, reef_on)
ARMS: List[Tuple[str, bool, bool]] = [
    ("A0_off_no_reef",    False, False),
    ("A1_sd016_no_reef",  True,  False),
    ("A2_reef_no_sd016",  False, True),
    ("A3_reef_sd016",     True,  True),
]

# Acceptance thresholds
ATTN_ENT_C0_FLOOR        = 2.65   # A1 must stay ABOVE (env-entropy bottleneck)
COS_CROSS_C1_CEIL        = 0.90   # reef arms must go BELOW (zone separation)
ATTN_ENT_C2_THRESH       = 2.65   # A3 must go BELOW (breaks uniform rail)
ACT_ENT_C3_THRESH        = 0.30   # A3 must exceed (behavioral delta)


def _make_env(seed: int, reef: bool) -> CausalGridWorldV2:
    if reef:
        return CausalGridWorldV2(
            seed=seed, size=8, num_hazards=2, num_resources=3,
            hazard_harm=0.02, use_proxy_fields=True,
            resource_respawn_on_consume=True,
            reef_enabled=True, reef_patch_radius=1, hazard_food_attraction=0.7,
        )
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=2, num_resources=3,
        hazard_harm=0.02, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, sd016_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9, alpha_self=0.3,
        sd016_enabled=sd016_on,
        sd016_writepath_mode="off",
        sd016_temperature_learnable=sd016_on,
        sws_enabled=False, rem_enabled=False, shy_enabled=False,
    )
    if sd016_on:
        cfg.sd016_diversification_weight = LAMBDA_DIVERSIFY
    return REEAgent(cfg)


def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def get_hazard_max(obs_dict, world_obs):
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, "shape"):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def _reef_zone_score(world_obs: torch.Tensor, world_obs_dim: int) -> float:
    """Return mean of reef_field_view if reef dims present, else 0."""
    if world_obs_dim <= 250:
        return 0.0
    reef_fv = world_obs[..., 250:275]
    return float(reef_fv.mean().item())


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


def _slot_diversity(cm) -> float:
    with torch.no_grad():
        mn = F.normalize(cm.memory, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        return max(0.0, 1.0 - float(off.sum().item() / max(1, n * (n - 1))))


def _attn_entropy_mean(agent, z_world_batch) -> float:
    with torch.no_grad():
        cm = agent.e1.context_memory
        wq = agent.e1.world_query_proj
        memory_dim = cm.memory_dim
        batch_size = z_world_batch.shape[0]
        q = wq(z_world_batch).unsqueeze(1)
        k = cm.key_proj(cm.memory).unsqueeze(0).expand(batch_size, -1, -1)
        scale = float(agent.e1._sd016_attention_scale(memory_dim))
        scores = torch.bmm(q, k.transpose(1, 2)) / scale
        weights = F.softmax(scores, dim=-1).squeeze(1)
        probs = weights.clamp(min=1e-12)
        return float(-(probs * probs.log()).sum(dim=-1).mean().item())


def _action_class_entropy_under_cue_bias(agent, z_world_batch) -> float:
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


def _pairwise_cos_mean(z_batch: torch.Tensor) -> float:
    """Mean off-diagonal pairwise cosine similarity."""
    if z_batch.shape[0] < 2:
        return 1.0
    with torch.no_grad():
        norm = F.normalize(z_batch, dim=-1)
        sim = torch.mm(norm, norm.t())
        n = sim.shape[0]
        off_sum = sim.sum().item() - n  # subtract diagonal (=1.0 each)
        return float(off_sum / max(1, n * (n - 1)))


def _cross_cos_mean(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Mean pairwise cross-group cosine similarity."""
    if z_a.shape[0] < 1 or z_b.shape[0] < 1:
        return 1.0
    with torch.no_grad():
        na = F.normalize(z_a, dim=-1)
        nb = F.normalize(z_b, dim=-1)
        sim = torch.mm(na, nb.t())
        return float(sim.mean().item())


def _collect_eval_batches(
    agent, env, n_per_zone: int, reef_on: bool
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Collect z_world and split by reef proximity via median-split.
    Collects 2*n_per_zone total samples, then assigns top-half (high reef
    proximity) as reef zone and bottom-half (low reef proximity) as food zone.
    This is robust: no absolute threshold needed; always produces balanced groups.
    For reef-OFF arms: all samples go to food_z (reef_z is empty).
    Returns (reef_z_batch, food_z_batch, reef_fraction).
    """
    device = agent.device
    n_total = n_per_zone * 2
    world_obs_dim = env.world_obs_dim
    samples: List[Tuple[float, torch.Tensor]] = []  # (reef_score, z_world)

    _, obs_dict = env.reset()
    agent.reset()
    if hasattr(agent, "e1") and hasattr(agent.e1, "reset_hidden_state"):
        agent.e1.reset_hidden_state()

    for _ in range(STEPS_PER_EPISODE * 8):
        if len(samples) >= n_total:
            break
        ob = obs_dict["body_state"]
        ow = obs_dict["world_state"]
        ob = ob.to(device) if torch.is_tensor(ob) else torch.tensor(ob, dtype=torch.float32, device=device)
        ow = ow.to(device) if torch.is_tensor(ow) else torch.tensor(ow, dtype=torch.float32, device=device)
        if ob.dim() == 1:
            ob = ob.unsqueeze(0)
        if ow.dim() == 1:
            ow = ow.unsqueeze(0)
        latent = agent.sense(ob, ow)
        z = latent.z_world.detach().squeeze(0)
        score = _reef_zone_score(ow, world_obs_dim)
        samples.append((score, z))
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            if hasattr(agent, "e1") and hasattr(agent.e1, "reset_hidden_state"):
                agent.e1.reset_hidden_state()

    if not samples:
        return torch.empty(0, 1), torch.empty(0, 1), 0.0

    if not reef_on:
        # No reef dims; all samples classified as food zone
        food_t = torch.stack([s[1] for s in samples], dim=0)
        return torch.empty(0, 1), food_t, 0.0

    # Median-split by reef proximity score
    samples_sorted = sorted(samples, key=lambda x: x[0], reverse=True)
    mid = len(samples_sorted) // 2
    reef_z = [s[1] for s in samples_sorted[:mid]]   # high reef proximity
    food_z = [s[1] for s in samples_sorted[mid:]]   # low reef proximity (food zone)
    reef_t = torch.stack(reef_z, dim=0) if reef_z else torch.empty(0, 1)
    food_t = torch.stack(food_z, dim=0) if food_z else torch.empty(0, 1)
    reef_frac = len(reef_z) / max(1, len(samples))
    return reef_t, food_t, reef_frac


def _run_training_episode(agent, env, optimizer, phase: str) -> int:
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    if hasattr(agent, "e1") and hasattr(agent.e1, "reset_hidden_state"):
        agent.e1.reset_hidden_state()
    sd016 = agent.config.e1.sd016_enabled if hasattr(agent.config, "e1") else False

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
        if hasattr(agent, "_self_experience_buffer"):
            agent._self_experience_buffer.append(latent.z_self.detach().clone())
        if hasattr(agent, "_world_experience_buffer"):
            agent._world_experience_buffer.append(latent.z_world.detach().clone())
        agent.clock.advance()

        hazard_max = get_hazard_max(obs_dict, ow)
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, _info, obs_dict = env.step(action)

        pred_loss = agent.compute_prediction_loss()
        total_loss = pred_loss

        if sd016:
            t_loss = compute_terrain_loss(agent, latent.z_world, hazard_max)
            ent_loss = agent.e1.compute_attention_entropy_loss(latent.z_world)
            total_loss = total_loss + LAMBDA_TERRAIN * t_loss + LAMBDA_ATTN_ENT * ent_loss
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

    return STEPS_PER_EPISODE


def _run_one_arm_seed(
    arm_label: str, sd016_on: bool, reef_on: bool, seed: int
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed, reef_on)
    agent = _make_agent(env, sd016_on)
    device = agent.device

    param_groups = list(agent.e1.parameters()) + list(agent.latent_stack.parameters())
    optimizer = optim.Adam(param_groups, lr=LR)

    print(
        f"  [arm {arm_label} seed {seed}] sd016={sd016_on} reef={reef_on} "
        f"world_obs_dim={env.world_obs_dim} p0={P0_EPISODES} p1={P1_EPISODES} "
        f"steps/ep={STEPS_PER_EPISODE}",
        flush=True,
    )

    for ep in range(P0_EPISODES):
        _run_training_episode(agent, env, optimizer, "P0")
    for ep in range(P1_EPISODES):
        _run_training_episode(agent, env, optimizer, "P1")

    reef_z, food_z, reef_frac = _collect_eval_batches(
        agent, env, EVAL_N_PER_ZONE, reef_on
    )

    # z_world variation metrics
    if reef_on and reef_z.shape[0] >= 2 and food_z.shape[0] >= 2:
        cos_cross   = _cross_cos_mean(reef_z.to(device), food_z.to(device))
        cos_w_reef  = _pairwise_cos_mean(reef_z.to(device))
        cos_w_food  = _pairwise_cos_mean(food_z.to(device))
        cos_within  = (cos_w_reef + cos_w_food) / 2.0
        sep_gap     = cos_within - cos_cross
    else:
        # reef OFF or insufficient zone samples: use food_z as the full batch
        combined = food_z.to(device) if food_z.shape[0] > 0 else torch.zeros(2, 32)
        cos_cross  = _pairwise_cos_mean(combined)
        cos_within = cos_cross
        sep_gap    = 0.0
        reef_frac  = 0.0

    # SD-016 attention / behavioral metrics
    # Filter out empty placeholder tensors (shape [0,1]) before cat to avoid
    # dim-1 mismatch when reef_on=False returns torch.empty(0,1) for reef_z.
    _parts = [t for t in (reef_z, food_z) if t.shape[0] > 0]
    combined_z = (torch.cat(_parts, dim=0) if _parts else torch.zeros(4, 32)).to(device)

    if sd016_on and hasattr(agent.e1, "context_memory") and agent.e1.context_memory is not None:
        attn_ent = _attn_entropy_mean(agent, combined_z)
        act_ent  = _action_class_entropy_under_cue_bias(agent, combined_z)
        slot_div = _slot_diversity(agent.e1.context_memory)
        log_tau  = float(agent.e1.sd016_log_temperature.item()) if agent.e1.sd016_log_temperature is not None else None
    else:
        attn_ent = float("nan")
        act_ent  = float("nan")
        slot_div = float("nan")
        log_tau  = None

    print(
        f"  [arm {arm_label} seed {seed}] reef_frac={reef_frac:.3f} "
        f"cos_cross={cos_cross:.4f} sep_gap={sep_gap:.4f} "
        f"attn_ent={attn_ent:.4f} act_ent={act_ent:.4f} slot_div={slot_div:.3f}",
        flush=True,
    )

    return {
        "arm":               arm_label,
        "sd016_on":          sd016_on,
        "reef_on":           reef_on,
        "seed":              seed,
        "world_obs_dim":     env.world_obs_dim,
        "cos_cross":         cos_cross,
        "cos_within":        cos_within,
        "separation_gap":    sep_gap,
        "reef_zone_fraction": reef_frac,
        "attn_entropy_mean": attn_ent,
        "action_class_entropy": act_ent,
        "slot_diversity":    slot_div,
        "log_tau_final":     log_tau,
        "n_reef_samples":    reef_z.shape[0] if reef_z.shape[0] > 0 else 0,
        "n_food_samples":    food_z.shape[0] if food_z.shape[0] > 0 else 0,
    }


def _aggregate_arm(results: List[Dict]) -> Dict:
    def mean(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float)) and r.get(key) == r.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")
    return {
        "arm":                  results[0]["arm"],
        "sd016_on":             results[0]["sd016_on"],
        "reef_on":              results[0]["reef_on"],
        "world_obs_dim":        results[0]["world_obs_dim"],
        "cos_cross_mean":       mean("cos_cross"),
        "cos_within_mean":      mean("cos_within"),
        "separation_gap_mean":  mean("separation_gap"),
        "reef_zone_fraction_mean": mean("reef_zone_fraction"),
        "attn_entropy_mean_mean": mean("attn_entropy_mean"),
        "action_class_entropy_mean": mean("action_class_entropy"),
        "slot_diversity_mean":  mean("slot_diversity"),
        "per_seed":             results,
    }


def _evaluate_criteria(aggregates: Dict[str, Dict]) -> Dict:
    a0 = aggregates.get("A0_off_no_reef", {})
    a1 = aggregates.get("A1_sd016_no_reef", {})
    a2 = aggregates.get("A2_reef_no_sd016", {})
    a3 = aggregates.get("A3_reef_sd016", {})

    def seeds_passing(arm_results, key, threshold, op="lt") -> int:
        count = 0
        for r in arm_results.get("per_seed", []):
            v = r.get(key)
            if v is None or v != v:
                continue
            if op == "lt" and v < threshold:
                count += 1
            elif op == "gt" and v > threshold:
                count += 1
        return count

    # C0: A1 attn_entropy > 2.65 in ALL 3 seeds (env-entropy bottleneck)
    c0_pass = seeds_passing(a1, "attn_entropy_mean", ATTN_ENT_C0_FLOOR, "gt") == 3

    # C1: cos_cross < 0.90 in >=2/3 seeds for BOTH A2 and A3
    c1_a2 = seeds_passing(a2, "cos_cross", COS_CROSS_C1_CEIL, "lt") >= 2
    c1_a3 = seeds_passing(a3, "cos_cross", COS_CROSS_C1_CEIL, "lt") >= 2
    c1_pass = c1_a2 and c1_a3

    # C2: A3 attn_entropy < 2.65 in >=2/3 seeds
    c2_pass = seeds_passing(a3, "attn_entropy_mean", ATTN_ENT_C2_THRESH, "lt") >= 2

    # C3: A3 action_class_entropy > 0.30 nats in >=2/3 seeds
    c3_pass = seeds_passing(a3, "action_class_entropy", ACT_ENT_C3_THRESH, "gt") >= 2

    overall = c0_pass and c1_pass and c2_pass and c3_pass

    return {
        "C0_env_entropy_bottleneck": c0_pass,
        "C1_reef_zone_separation": c1_pass,
        "C1a_A2_separation": c1_a2,
        "C1b_A3_separation": c1_a3,
        "C2_uniform_rail_broken": c2_pass,
        "C3_behavioral_delta": c3_pass,
        "overall_pass": overall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run: smoke test one arm one seed.", flush=True)
        env = _make_env(42, reef=True)
        agent = _make_agent(env, sd016_on=True)
        optimizer = optim.Adam(
            list(agent.e1.parameters()) + list(agent.latent_stack.parameters()), lr=LR
        )
        for ep in range(2):
            _run_training_episode(agent, env, optimizer, "P0")
        reef_z, food_z, reef_frac = _collect_eval_batches(agent, env, 4, reef_on=True)
        print(f"  reef_samples={reef_z.shape[0]} food_samples={food_z.shape[0]} reef_frac={reef_frac:.3f}", flush=True)
        print("Dry-run PASS.", flush=True)
        return

    print(f"=== {EXPERIMENT_TYPE} ===", flush=True)
    all_results: List[Dict] = []
    for arm_label, sd016_on, reef_on in ARMS:
        print(f"\n--- Arm {arm_label} ---", flush=True)
        arm_seed_results = []
        for seed in SEEDS:
            r = _run_one_arm_seed(arm_label, sd016_on, reef_on, seed)
            arm_seed_results.append(r)
            all_results.append(r)
        agg = _aggregate_arm(arm_seed_results)
        print(
            f"  [arm {arm_label} AGG] cos_cross={agg['cos_cross_mean']:.4f} "
            f"sep_gap={agg['separation_gap_mean']:.4f} "
            f"attn_ent={agg['attn_entropy_mean_mean']:.4f} "
            f"act_ent={agg['action_class_entropy_mean']:.4f}",
            flush=True,
        )

    aggregates = {}
    for arm_label, _, _ in ARMS:
        seed_results = [r for r in all_results if r["arm"] == arm_label]
        aggregates[arm_label] = _aggregate_arm(seed_results)

    criteria = _evaluate_criteria(aggregates)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    print(f"\n=== OUTCOME: {outcome} ===", flush=True)
    for k, v in criteria.items():
        print(f"  {k}: {v}", flush=True)

    # Write result manifest
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_v3"
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{EXPERIMENT_TYPE}_{run_id}.json"
    manifest = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             run_id,
        "claim_ids":          CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "supersedes":         ["V3-EXQ-418j"],
        "criteria":           criteria,
        "arm_aggregates":     aggregates,
        "all_seed_results":   all_results,
        "hyperparams": {
            "P0_episodes":     P0_EPISODES,
            "P1_episodes":     P1_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "lambda_terrain":  LAMBDA_TERRAIN,
            "lambda_cue_action": LAMBDA_CUE_ACTION,
            "lambda_diversify": LAMBDA_DIVERSIFY,
            "lambda_attn_ent": LAMBDA_ATTN_ENT,
            "eval_n_per_zone": EVAL_N_PER_ZONE,
            "zone_split": "median (top-half high-proximity = reef zone)",
            "seeds":           SEEDS,
            "lr":              LR,
        },
    }
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nResult written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
