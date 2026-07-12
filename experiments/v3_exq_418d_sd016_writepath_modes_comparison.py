#!/opt/local/bin/python3
"""
V3-EXQ-418d -- SD-016 ContextMemory write-path modes 4-arm comparison

EXPERIMENT_PURPOSE: evidence

QUESTION:
  EXQ-477 / EXP-0155 diagnosed the SD-016 ContextMemory uniform-attention
  collapse as bias-dominance: with memory init scale 0.01 and a default-
  init Linear bias on key_proj, every key collapsed to ~b_K and softmax
  was uniform regardless of slot content. Part A of the fix
  (key_proj=Linear(memory_dim, memory_dim, bias=False)) is now structural
  in e1_deep.py and lands in every agent.

  Part B is the observation-conditioned write path. Without writes, slots
  remain at randn*0.01 init and Part A produces non-uniform attention only
  over noise. Two routing options were proposed in the design doc v2:
    B1  REEAgent.compute_prediction_loss   -- training-time write hook
    B2  REEAgent.sense()                   -- per-tick write hook
  This experiment compares all four combinations under matched training
  to identify the production default.

DESIGN:
  Four arms, three seeds each. Part A applied uniformly (structural).
    A0  sd016_writepath_mode = "off"          -- Part A only; bias-fix
                                                 baseline; no writes.
    A1  sd016_writepath_mode = "train_only"   -- B1 hook active.
    A2  sd016_writepath_mode = "sense_only"   -- B2 hook active.
    A3  sd016_writepath_mode = "both"         -- B1 + B2 active.
  All arms use sd016_enabled=True so cue_terrain_proj / cue_action_proj
  exist; SHY is disabled (shy_enabled=False) so Part B writes are not
  immediately decayed away. Sleep is disabled (sws_enabled=False,
  rem_enabled=False) -- this experiment isolates the write-path mode,
  not sleep consolidation.

  Training:
    P0 (P0_EPISODES warmup):   E1 prediction loss + cue_terrain_loss
                               (LAMBDA_TERRAIN=0.1, supervises
                               cue_terrain_proj per the EXQ-182 pattern).
    P1 (P1_EPISODES training): same loss + cue_action_loss
                               (LAMBDA_CUE_ACTION=0.5, supervises
                               cue_action_proj toward E2.action_object).
    Alternating SAFE / DANGEROUS contexts every CONTEXT_SWITCH_EVERY=5 eps
    so cue_terrain has signal to track.

  Evaluation: held-out EVAL_BATCH_SIZE z_world snapshots run through the
  ContextMemory attention path and write_log instrumentation tallied.

PER-ARM METRICS:
  attn_entropy_mean          attention entropy over the eval batch;
                             uniform reference for num_slots=16 is
                             ln(16) ~= 2.7726.
  n_writes                   total ContextMemory.write() calls during
                             training (instrumentation hook).
  slot_diversity             1 - mean off-diagonal pairwise cosine
                             similarity of cm.memory rows post-train
                             (higher = more differentiated).
  action_class_entropy       entropy of the agent's action class
                             distribution over the eval batch (a
                             behavioural ablation signature for
                             cue_action_proj usefulness).

ACCEPTANCE CRITERIA (script-level PASS/FAIL):
  C1  attn_entropy_mean < 2.5 in >= 2/3 seeds for arms A1, A2, A3.
      (A0 is the bias-fix baseline; without writes, entropy may stay
      near uniform over noise; we do NOT require A0 to satisfy C1.)
  C2  n_writes > 0 in ALL 3 seeds for A1, A2, A3; n_writes == 0 in
      ALL 3 seeds for A0.
  C3  slot_diversity > 0.5 in >= 2/3 seeds for A1, A2, A3 post-train.
  C4  Production default selected by combined rule:
        smallest slot_diversity collapse (1 - mean_diversity is
        smallest) + largest action_class_entropy ablation delta vs
        A0 (action_class_entropy[arm] - action_class_entropy[A0] is
        largest). Ties go to A1 (train_only) per design doc v2.
      C4 reports the winner string in production_default; PASS does
      not gate on which arm wins.

  PASS: C1 AND C2 AND C3.

claim_ids: ["SD-016", "SD-017"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418c
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_418d_sd016_writepath_modes_comparison"
CLAIM_IDS          = ["SD-016", "SD-017"]
EXPERIMENT_PURPOSE = "evidence"

# Training schedule (calibrated against EXQ-477 + EXQ-418a; ~5 min/seed/arm
# on DLAPTOP-4.local at 150 steps/ep -- 4 arms x 3 seeds = 12 runs ~= 60 min).
P0_EPISODES         = 20
P1_EPISODES         = 40
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
EVAL_BATCH_SIZE     = 32

LR    = 1e-4
SEEDS = [42, 43, 44]

ARMS: List[Tuple[str, str]] = [
    ("A0_off",         "off"),
    ("A1_train_only",  "train_only"),
    ("A2_sense_only",  "sense_only"),
    ("A3_both",        "both"),
]

# Acceptance thresholds.
ATTN_ENTROPY_C1_THRESHOLD   = 2.5     # below this => meaningfully non-uniform
N_WRITES_NONZERO_THRESHOLD  = 1       # at least one write counts as "writes happen"
SLOT_DIVERSITY_C3_THRESHOLD = 0.5     # post-eval slot-store differentiation


# ---------------------------------------------------------------------------
# Env + agent plumbing (mirrors EXQ-477 / EXQ-418a contract).
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


def _make_agent(env: CausalGridWorldV2, writepath_mode: str) -> REEAgent:
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
# Write-call instrumentation: tally context_memory.write() invocations.
# ---------------------------------------------------------------------------

def _install_write_counter(cm) -> List[int]:
    """Returns a list whose length is incremented on every write() call.

    Cheaper than the EXQ-477 full trace; we only need a per-arm count.
    """
    import types
    counter: List[int] = []
    orig_write = cm.write

    def _counted_write(self, state: torch.Tensor) -> None:
        counter.append(0)
        return orig_write(state)

    cm.write = types.MethodType(_counted_write, cm)
    return counter


# ---------------------------------------------------------------------------
# Slot-store + attention probes (lifted from EXQ-477).
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
    obs_world_list: List[torch.Tensor] = []

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
        obs_world_list.append(ow.detach().clone().squeeze(0))

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
        "obs_world_batch": torch.stack(obs_world_list[:n_samples], dim=0),
    }


def _action_class_entropy_under_cue_bias(agent, z_world_batch) -> float:
    """Entropy over action-class argmax of cue_action_proj output.

    With sd016_enabled=True, extract_cue_context returns a per-state
    (action_bias, terrain_weight) pair. We inspect the action_bias
    distribution across the eval batch by argmaxing into action_dim
    classes (action_object is action_dim-shaped) and computing the
    entropy of the resulting class distribution. Higher entropy =
    cue_action_proj is content-conditioned across states; near-zero
    entropy = collapsed to a single class regardless of input.
    """
    with torch.no_grad():
        action_bias, _ = agent.e1.extract_cue_context(z_world_batch)
        # action_bias is action_object_dim -- project to action_dim via
        # E2's action_object decoder if available; otherwise argmax over
        # the bias vector directly. We argmax over the leading
        # action_dim entries which correspond to the action embedding
        # in E2.action_object's contract.
        action_dim = agent.e2.config.action_dim
        cls = action_bias[..., :action_dim].argmax(dim=-1)
        # Histogram across the batch.
        hist = torch.bincount(cls, minlength=action_dim).float()
        if hist.sum() <= 0:
            return 0.0
        probs = hist / hist.sum()
        probs = probs.clamp(min=1e-12)
        return float(-(probs * probs.log()).sum().item())


# ---------------------------------------------------------------------------
# Training episode (mirrors EXQ-449b WARMUP / SUPERVISED arm).
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
        # B1 hook in compute_prediction_loss requires _world_experience_buffer
        # populated; sense() does not push (that is done by _e1_tick from
        # select_action). Push manually so train-time prediction loss is real.
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


# ---------------------------------------------------------------------------
# Per-arm per-seed run.
# ---------------------------------------------------------------------------

def _run_one_arm_seed(
    arm_label: str,
    writepath_mode: str,
    seed: int,
    p0: int,
    p1: int,
    eval_n: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe, writepath_mode)
    cm = agent.e1.context_memory
    world_query_proj = agent.e1.world_query_proj

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    write_counter = _install_write_counter(cm)

    print(
        f"  [arm {arm_label} seed {seed}] mode={writepath_mode} p0={p0} p1={p1} "
        f"steps/ep={STEPS_PER_EPISODE}",
        flush=True,
    )

    # P0 warmup
    for ep in range(p0):
        use_dang = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "P0")

    # P1 supervised
    for ep in range(p1):
        abs_ep = p0 + ep
        use_dang = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "P1")

    # Evaluation
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
    """Per-arm summary across seeds."""
    n_seeds = len(arm_results)
    return {
        "n_seeds":                  n_seeds,
        "n_writes_min":             min(r["n_writes"] for r in arm_results),
        "n_writes_max":             max(r["n_writes"] for r in arm_results),
        "n_writes_seeds_nonzero":   sum(1 for r in arm_results
                                        if r["n_writes"] >= N_WRITES_NONZERO_THRESHOLD),
        "slot_diversity_mean":      sum(r["slot_diversity"] for r in arm_results) / n_seeds,
        "slot_diversity_seeds_pass": sum(1 for r in arm_results
                                         if r["slot_diversity"] > SLOT_DIVERSITY_C3_THRESHOLD),
        "attn_entropy_mean_mean":   sum(r["attn_entropy_mean"] for r in arm_results) / n_seeds,
        "attn_entropy_seeds_pass":  sum(1 for r in arm_results
                                        if r["attn_entropy_mean"] < ATTN_ENTROPY_C1_THRESHOLD),
        "action_class_entropy_mean": sum(r["action_class_entropy"] for r in arm_results) / n_seeds,
    }


def _evaluate_acceptance(per_arm: Dict[str, Dict]) -> Dict:
    a0 = per_arm.get("A0_off", {})
    n_seeds = a0.get("n_seeds", 0)

    # C1: attn_entropy < 2.5 in 2/3 seeds for arms A1, A2, A3.
    c1_pass_per_arm = {
        arm: per_arm[arm]["attn_entropy_seeds_pass"] >= 2
        for arm in ("A1_train_only", "A2_sense_only", "A3_both")
        if arm in per_arm
    }
    c1_pass = all(c1_pass_per_arm.values()) if c1_pass_per_arm else False

    # C2: n_writes>0 in ALL seeds for A1/A2/A3; ==0 for A0.
    c2_a0_pass = (a0.get("n_writes_max", 0) == 0) and (a0.get("n_writes_min", 0) == 0)
    c2_active_pass = all(
        per_arm[arm]["n_writes_seeds_nonzero"] == n_seeds
        for arm in ("A1_train_only", "A2_sense_only", "A3_both")
        if arm in per_arm
    )
    c2_pass = c2_a0_pass and c2_active_pass

    # C3: slot_diversity > 0.5 in 2/3 seeds for A1/A2/A3.
    c3_pass_per_arm = {
        arm: per_arm[arm]["slot_diversity_seeds_pass"] >= 2
        for arm in ("A1_train_only", "A2_sense_only", "A3_both")
        if arm in per_arm
    }
    c3_pass = all(c3_pass_per_arm.values()) if c3_pass_per_arm else False

    # C4: production default selector. Among A1/A2/A3, pick the one with
    # smallest slot_diversity collapse (== largest mean diversity) plus
    # largest action_class_entropy ablation delta vs A0. Combine via sum
    # of per-metric ranks (1=best). Ties go to A1_train_only.
    candidate_arms = [a for a in ("A1_train_only", "A2_sense_only", "A3_both")
                      if a in per_arm]
    # Largest slot_diversity_mean is best (rank 1).
    sd_rank = sorted(
        candidate_arms,
        key=lambda a: -per_arm[a]["slot_diversity_mean"],
    )
    a0_act = a0.get("action_class_entropy_mean", 0.0)
    # Largest delta vs A0 is best (rank 1).
    act_rank = sorted(
        candidate_arms,
        key=lambda a: -(per_arm[a]["action_class_entropy_mean"] - a0_act),
    )
    rank_score = {a: sd_rank.index(a) + act_rank.index(a) for a in candidate_arms}
    min_score = min(rank_score.values()) if rank_score else 0
    tied = [a for a in candidate_arms if rank_score[a] == min_score]
    if "A1_train_only" in tied:
        production_default = "A1_train_only"
    else:
        production_default = tied[0] if tied else "A1_train_only"

    overall_pass = c1_pass and c2_pass and c3_pass

    return {
        "C1_attn_entropy_below_2_5": {
            "pass": c1_pass,
            "per_arm": c1_pass_per_arm,
            "threshold": ATTN_ENTROPY_C1_THRESHOLD,
        },
        "C2_writes_routing_correct": {
            "pass": c2_pass,
            "A0_zero_writes": c2_a0_pass,
            "active_arms_all_seeds_nonzero": c2_active_pass,
        },
        "C3_slot_diversity_above_0_5": {
            "pass": c3_pass,
            "per_arm": c3_pass_per_arm,
            "threshold": SLOT_DIVERSITY_C3_THRESHOLD,
        },
        "C4_production_default": {
            "winner": production_default,
            "rank_score_lower_is_better": rank_score,
            "tie_break_rule": "ties_go_to_A1_train_only",
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
        f"V3-EXQ-418d 4-arm SD-016 write-path comparison "
        f"(seeds={SEEDS} dry_run={dry_run})",
        flush=True,
    )

    all_results: List[Dict] = []
    per_arm: Dict[str, List[Dict]] = {arm: [] for arm, _ in ARMS}

    for arm_label, mode in ARMS:
        for seed in SEEDS:
            r = _run_one_arm_seed(arm_label, mode, seed, p0, p1, eval_n)
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries = {arm: _per_arm_summary(rs) for arm, rs in per_arm.items()}
    acceptance = _evaluate_acceptance(summaries)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(
        f"  [summary] C1={acceptance['C1_attn_entropy_below_2_5']['pass']} "
        f"C2={acceptance['C2_writes_routing_correct']['pass']} "
        f"C3={acceptance['C3_slot_diversity_above_0_5']['pass']} "
        f"-> outcome={outcome} (production_default="
        f"{acceptance['C4_production_default']['winner']})",
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
        "supersedes":         "V3-EXQ-418c",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_note": (
            "SD-016 ContextMemory write-path 4-arm comparison (Part B routing "
            "study). Part A (key_proj.bias=False) is structural and applied "
            "uniformly across arms. Outcome PASS = the chosen production "
            "default (sd016_writepath_mode in {train_only, sense_only, both}) "
            "is empirically grounded; outcome FAIL = none of the active write "
            "modes restore content-conditioned attention + slot differentiation "
            "under matched training, indicating the design doc v2 substrate is "
            "insufficient and a deeper rework is required."
        ),
        "acceptance_checks":  acceptance,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "attn_entropy_c1_threshold":   ATTN_ENTROPY_C1_THRESHOLD,
            "n_writes_nonzero_threshold":  N_WRITES_NONZERO_THRESHOLD,
            "slot_diversity_c3_threshold": SLOT_DIVERSITY_C3_THRESHOLD,
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
            "arms":                 [{"label": a, "writepath_mode": m} for a, m in ARMS],
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
