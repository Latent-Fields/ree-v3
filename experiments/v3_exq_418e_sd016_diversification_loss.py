#!/opt/local/bin/python3
"""
V3-EXQ-418e -- SD-016 ContextMemory Path 1 auxiliary diversification loss

EXPERIMENT_PURPOSE: evidence

QUESTION:
  EXQ-418d 4-arm SD-016 write-path comparison FAILed across all four arms
  (A0_off, A1_train_only, A2_sense_only, A3_both): attn_entropy_mean
  collapsed to ln(16) = 2.7726 (uniform reference) on every arm under
  matched seeds [42, 43, 44]; slot_diversity was bimodal across seeds
  (seed 42 escaped to ~0.46, seeds 43/44 collapsed to numerically zero).
  The v2 substrate (read-side gradient + EMA argmin write rule) cannot
  break slot symmetry: read-side cannot differentiate slots that all
  contribute equally under uniform attention; the write rule is
  luck-dependent on init.

  Path 1 hypothesis: an explicit auxiliary diversification loss term
  (mean squared off-diagonal cosine similarity of normalised slot
  vectors) provides direct gradient pressure on ContextMemory.memory
  to push slot vectors toward mutual orthogonality, breaking symmetry
  independently of the write rule.

DESIGN:
  Four arms, three seeds each. Part A (key_proj.bias=False) applied
  uniformly. Part B routing held at "off" or "sense_only".
    A0_off              writepath_mode="off"        div_weight=0.0
                          baseline; replicates EXQ-418d A0_off
    A1_writes_only      writepath_mode="sense_only" div_weight=0.0
                          replicates EXQ-418d A2_sense_only
                          (sanity check the v2 path still FAILs)
    A2_div_only         writepath_mode="off"        div_weight=0.5
                          tests if diversification ALONE breaks symmetry
                          (no write contribution; pure gradient pressure)
    A3_writes_plus_div  writepath_mode="sense_only" div_weight=0.5
                          full hypothesis bootstrap

  All arms use sd016_enabled=True so cue_terrain_proj / cue_action_proj
  exist; SHY / sleep disabled to isolate the substrate effect.

  Training schedule matches EXQ-418d (P0=20 P1=40 STEPS=150
  LAMBDA_TERRAIN=0.1 LAMBDA_CUE_ACTION=0.5).

PER-ARM METRICS:
  attn_entropy_mean       attention entropy over the eval batch;
                          uniform reference for num_slots=16 is
                          ln(16) ~= 2.7726.
  slot_diversity          1 - mean off-diagonal pairwise cosine
                          similarity of cm.memory rows post-train.
  action_class_entropy    entropy of action class argmax over
                          cue_action_proj output across the eval batch.
  n_writes                total ContextMemory.write() calls
                          (instrumentation; expected 0 for div_only).

ACCEPTANCE CRITERIA (script-level PASS/FAIL):
  C1  attn_entropy_mean < 2.65 in ALL three seeds for A2_div_only
      AND A3_writes_plus_div. (uniform = 2.7726; cutoff 2.65 means
      ~5% reduction in attention entropy from uniform.)
  C2  slot_diversity > 0.10 in ALL three seeds for A2_div_only AND
      A3_writes_plus_div. (Hard symmetry-breaking criterion: every
      seed must escape collapse, not just the lucky one. EXQ-418d
      had seeds 43/44 collapse to ~0 on every arm.)
  C3  |action_class_entropy[A3] - action_class_entropy[A0]| >= 0.20
      nats. Confirms diversified slots produce behaviourally
      distinguishable cue_action_proj outputs (production target).
  C4  A1_writes_only replicates EXQ-418d FAIL pattern:
      slot_diversity_seeds_pass < 3 (i.e. NOT all seeds escape
      collapse) AND attn_entropy_mean_mean > 2.65. Sanity check
      that the new substrate did not accidentally fix the v2 path.

  PASS: C1 AND C2 AND C3 AND C4.
  C4 FAIL with C1+C2+C3 PASS would invalidate the ablation logic.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418d
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


EXPERIMENT_TYPE    = "v3_exq_418e_sd016_diversification_loss"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "evidence"

# Training schedule (matched to EXQ-418d).
P0_EPISODES         = 20
P1_EPISODES         = 40
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
LAMBDA_DIVERSIFY    = 0.5     # SD-016 Path 1 auxiliary diversification weight
EVAL_BATCH_SIZE     = 32

LR    = 1e-4
SEEDS = [42, 43, 44]

# Arm spec: (label, sd016_writepath_mode, sd016_diversification_weight)
ARMS: List[Tuple[str, str, float]] = [
    ("A0_off",             "off",        0.0),
    ("A1_writes_only",     "sense_only", 0.0),
    ("A2_div_only",        "off",        LAMBDA_DIVERSIFY),
    ("A3_writes_plus_div", "sense_only", LAMBDA_DIVERSIFY),
]

# Acceptance thresholds.
ATTN_ENTROPY_C1_THRESHOLD     = 2.65   # ~5% reduction from uniform ln(16)=2.7726
SLOT_DIVERSITY_C2_THRESHOLD   = 0.10   # symmetry-breaking floor
ACTION_ENTROPY_C3_DELTA       = 0.20   # nats; behavioural delta vs A0
A1_C4_DIVERSITY_PASS_CEILING  = 2      # must NOT have all 3 seeds pass C2
A1_C4_ATTN_ENTROPY_FLOOR      = 2.65   # mean must remain above this


# ---------------------------------------------------------------------------
# Env + agent plumbing (mirrors EXQ-418d).
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


def _make_agent(env: CausalGridWorldV2, writepath_mode: str, div_weight: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode=writepath_mode,
        sws_enabled=False,
        rem_enabled=False,
        shy_enabled=False,
    )
    # SD-016 Path 1: top-level REEConfig field; from_dims does not pipe it
    # through (no kwarg added on purpose -- experiments opt in by direct set).
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
# Write-call instrumentation.
# ---------------------------------------------------------------------------

def _install_write_counter(cm) -> List[int]:
    import types
    counter: List[int] = []
    orig_write = cm.write

    def _counted_write(self, state: torch.Tensor) -> None:
        counter.append(0)
        return orig_write(state)

    cm.write = types.MethodType(_counted_write, cm)
    return counter


# ---------------------------------------------------------------------------
# Slot-store + attention probes.
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


# ---------------------------------------------------------------------------
# Eval batch + behavioural action class probe.
# ---------------------------------------------------------------------------

def _collect_eval_batch(agent, env, n_samples: int) -> Dict[str, torch.Tensor]:
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

    return {
        "z_world_batch": torch.stack(z_world_list[:n_samples], dim=0),
    }


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


# ---------------------------------------------------------------------------
# Training episode (mirrors EXQ-418d).
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

        # compute_prediction_loss adds the SD-016 Path 1 diversification term
        # automatically when sd016_diversification_weight > 0 and sd016_enabled.
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


# ---------------------------------------------------------------------------
# Per-arm per-seed run.
# ---------------------------------------------------------------------------

def _run_one_arm_seed(
    arm_label: str,
    writepath_mode: str,
    div_weight: float,
    seed: int,
    p0: int,
    p1: int,
    eval_n: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe, writepath_mode, div_weight)
    cm = agent.e1.context_memory
    world_query_proj = agent.e1.world_query_proj

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    write_counter = _install_write_counter(cm)

    print(
        f"  [arm {arm_label} seed {seed}] mode={writepath_mode} "
        f"div_w={div_weight} p0={p0} p1={p1} steps/ep={STEPS_PER_EPISODE}",
        flush=True,
    )

    for ep in range(p0):
        use_dang = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "P0")

    for ep in range(p1):
        abs_ep = p0 + ep
        use_dang = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "P1")

    eval_batch = _collect_eval_batch(agent, env_safe, eval_n)
    z_world_batch = eval_batch["z_world_batch"]

    n_writes = len(write_counter)
    slot_div = _slot_diversity(cm)
    attn_ent = _attn_entropy_mean(cm, world_query_proj, z_world_batch)
    act_ent  = _action_class_entropy_under_cue_bias(agent, z_world_batch)

    print(
        f"  [arm {arm_label} seed {seed}] writes={n_writes} "
        f"slot_div={slot_div:.3f} attn_ent={attn_ent:.3f} act_ent={act_ent:.3f}",
        flush=True,
    )

    return {
        "arm":                  arm_label,
        "writepath_mode":       writepath_mode,
        "diversification_weight": div_weight,
        "seed":                 seed,
        "n_writes":             n_writes,
        "slot_diversity":       slot_div,
        "attn_entropy_mean":    attn_ent,
        "action_class_entropy": act_ent,
    }


# ---------------------------------------------------------------------------
# Aggregation + acceptance.
# ---------------------------------------------------------------------------

def _per_arm_summary(arm_results: List[Dict]) -> Dict:
    n_seeds = len(arm_results)
    return {
        "n_seeds":                  n_seeds,
        "n_writes_min":             min(r["n_writes"] for r in arm_results),
        "n_writes_max":             max(r["n_writes"] for r in arm_results),
        "slot_diversity_mean":      sum(r["slot_diversity"] for r in arm_results) / n_seeds,
        "slot_diversity_min":       min(r["slot_diversity"] for r in arm_results),
        "slot_diversity_seeds_pass": sum(1 for r in arm_results
                                         if r["slot_diversity"] > SLOT_DIVERSITY_C2_THRESHOLD),
        "attn_entropy_mean_mean":   sum(r["attn_entropy_mean"] for r in arm_results) / n_seeds,
        "attn_entropy_mean_max":    max(r["attn_entropy_mean"] for r in arm_results),
        "attn_entropy_seeds_pass":  sum(1 for r in arm_results
                                        if r["attn_entropy_mean"] < ATTN_ENTROPY_C1_THRESHOLD),
        "action_class_entropy_mean": sum(r["action_class_entropy"] for r in arm_results) / n_seeds,
    }


def _evaluate_acceptance(per_arm: Dict[str, Dict]) -> Dict:
    n_seeds = len(SEEDS)

    # C1: attn_entropy_mean < 2.65 in ALL three seeds for A2 and A3.
    c1_per_arm = {
        arm: per_arm[arm]["attn_entropy_seeds_pass"] == n_seeds
        for arm in ("A2_div_only", "A3_writes_plus_div")
        if arm in per_arm
    }
    c1_pass = all(c1_per_arm.values()) if c1_per_arm else False

    # C2: slot_diversity > 0.10 in ALL three seeds for A2 and A3.
    c2_per_arm = {
        arm: per_arm[arm]["slot_diversity_seeds_pass"] == n_seeds
        for arm in ("A2_div_only", "A3_writes_plus_div")
        if arm in per_arm
    }
    c2_pass = all(c2_per_arm.values()) if c2_per_arm else False

    # C3: |action_class_entropy[A3] - action_class_entropy[A0]| >= 0.20.
    a0_act = per_arm.get("A0_off", {}).get("action_class_entropy_mean", 0.0)
    a3_act = per_arm.get("A3_writes_plus_div", {}).get("action_class_entropy_mean", 0.0)
    c3_delta = abs(a3_act - a0_act)
    c3_pass = c3_delta >= ACTION_ENTROPY_C3_DELTA

    # C4: A1_writes_only must replicate EXQ-418d FAIL pattern (sanity check).
    a1 = per_arm.get("A1_writes_only", {})
    a1_diversity_pass_count = a1.get("slot_diversity_seeds_pass", 0)
    a1_attn_mean = a1.get("attn_entropy_mean_mean", 0.0)
    c4_diversity_replicates = a1_diversity_pass_count <= A1_C4_DIVERSITY_PASS_CEILING
    c4_attn_replicates = a1_attn_mean > A1_C4_ATTN_ENTROPY_FLOOR
    c4_pass = c4_diversity_replicates and c4_attn_replicates

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "C1_attn_entropy_below_2_65_all_seeds": {
            "pass": c1_pass,
            "per_arm": c1_per_arm,
            "threshold": ATTN_ENTROPY_C1_THRESHOLD,
            "applies_to": ["A2_div_only", "A3_writes_plus_div"],
        },
        "C2_slot_diversity_above_0_10_all_seeds": {
            "pass": c2_pass,
            "per_arm": c2_per_arm,
            "threshold": SLOT_DIVERSITY_C2_THRESHOLD,
            "applies_to": ["A2_div_only", "A3_writes_plus_div"],
        },
        "C3_behavioural_delta_above_0_20": {
            "pass": c3_pass,
            "delta_a3_minus_a0": c3_delta,
            "threshold": ACTION_ENTROPY_C3_DELTA,
            "a0_action_class_entropy_mean": a0_act,
            "a3_action_class_entropy_mean": a3_act,
        },
        "C4_a1_writes_only_replicates_418d_fail": {
            "pass": c4_pass,
            "diversity_pass_count": a1_diversity_pass_count,
            "diversity_pass_ceiling": A1_C4_DIVERSITY_PASS_CEILING,
            "attn_entropy_mean": a1_attn_mean,
            "attn_entropy_floor": A1_C4_ATTN_ENTROPY_FLOOR,
            "rule": "A1 must NOT have all seeds escape collapse, AND attn entropy must remain near uniform",
        },
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# main() driver.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False):
    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    print(
        f"V3-EXQ-418e SD-016 Path 1 diversification 4-arm "
        f"(seeds={SEEDS} dry_run={dry_run})",
        flush=True,
    )

    all_results: List[Dict] = []
    per_arm: Dict[str, List[Dict]] = {arm: [] for arm, _, _ in ARMS}

    for arm_label, mode, div_w in ARMS:
        for seed in SEEDS:
            r = _run_one_arm_seed(arm_label, mode, div_w, seed, p0, p1, eval_n)
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries = {arm: _per_arm_summary(rs) for arm, rs in per_arm.items()}
    acceptance = _evaluate_acceptance(summaries)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(
        f"  [summary] C1={acceptance['C1_attn_entropy_below_2_65_all_seeds']['pass']} "
        f"C2={acceptance['C2_slot_diversity_above_0_10_all_seeds']['pass']} "
        f"C3={acceptance['C3_behavioural_delta_above_0_20']['pass']} "
        f"C4={acceptance['C4_a1_writes_only_replicates_418d_fail']['pass']} "
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
        "outcome":            outcome,
        "timestamp_utc":      ts,
        "supersedes":         "V3-EXQ-418d",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_note": (
            "SD-016 ContextMemory Path 1 auxiliary diversification loss "
            "4-arm ablation. PASS = the diversification loss term breaks "
            "slot symmetry where the v2 read+write substrate cannot "
            "(EXQ-418d FAIL across all 4 arms with bimodal seed pattern). "
            "FAIL on C1 or C2 at div_weight=0.5 invalidates Path 1 at this "
            "weight; follow-up is either a weight sweep (EXQ-418f at 1.0/2.0/5.0) "
            "or Path 2/3 from the original 3-path proposal. FAIL on C3 with "
            "C1+C2 PASS would mean diversification works at the slot level "
            "but does not propagate to behaviour (cue_action_proj path "
            "uninformative even with non-uniform attention). FAIL on C4 "
            "with C1+C2+C3 PASS would invalidate the ablation logic by "
            "showing the new substrate accidentally fixed the v2 path."
        ),
        "acceptance_checks":  acceptance,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "attn_entropy_c1_threshold":    ATTN_ENTROPY_C1_THRESHOLD,
            "slot_diversity_c2_threshold":  SLOT_DIVERSITY_C2_THRESHOLD,
            "action_entropy_c3_delta":      ACTION_ENTROPY_C3_DELTA,
            "a1_c4_diversity_pass_ceiling": A1_C4_DIVERSITY_PASS_CEILING,
            "a1_c4_attn_entropy_floor":     A1_C4_ATTN_ENTROPY_FLOOR,
        },
        "params": {
            "seeds":                SEEDS,
            "p0_episodes":          p0,
            "p1_episodes":          p1,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":       LAMBDA_TERRAIN,
            "lambda_cue_action":    LAMBDA_CUE_ACTION,
            "lambda_diversify":     LAMBDA_DIVERSIFY,
            "eval_batch_size":      eval_n,
            "lr":                   LR,
            "arms": [
                {
                    "label": label,
                    "writepath_mode": mode,
                    "diversification_weight": div_w,
                }
                for label, mode, div_w in ARMS
            ],
            "sd016_enabled":        True,
            "sws_enabled":          False,
            "rem_enabled":          False,
            "shy_enabled":          False,
            "dry_run":              dry_run,
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal episodes to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
