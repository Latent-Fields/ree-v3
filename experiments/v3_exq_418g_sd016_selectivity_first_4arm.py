#!/opt/local/bin/python3
"""
V3-EXQ-418g -- SD-016 Path 4: query-selectivity-first 4-arm

EXPERIMENT_PURPOSE: evidence

QUESTION:
  EXQ-418d FAILed all writepath arms; EXQ-418e Path 1 FAILed at
  div_weight=0.5 with the cleanest decomposition we have:
    - A2_div_only:    slot_diversity_mean = 0.9999 (slots fully orthogonal)
                      attn_entropy_mean    = 2.7726 (uniform rail)
                      action_class_entropy = 1.1e-10 (cue_context inert)
    - A3_writes+div:  slot_diversity_mean = 0.6107
                      attn_entropy_mean    = 2.7524 (still ~uniform)
                      action_class_entropy = 0.4556 (= A1, all from writes)
  Slot diversification works at the slot level but cannot move attention.
  Behavioural delta in C3 came entirely from writes (A1 = A3 = 0.46);
  diversification contributed zero behavioural change. Cue_context is
  functionally inert because (a) world_query_proj produces queries that
  cannot pick a slot under sqrt(memory_dim) softmax, and (b) the
  EXQ-449a band-aid lets z_world bypass cue_context in cue_action_proj.

  Path 4 hypothesis: query selectivity, not slot content, is the
  bottleneck. Adding (i) a learnable attention temperature
  (sd016_temperature_learnable=True; exposes
  E1.sd016_log_temperature, replacing the fixed sqrt(memory_dim)
  divisor) and (ii) an attention-entropy minimisation loss
  (lambda_attn_ent * E1.compute_attention_entropy_loss(z_world))
  applies direct gradient pressure on world_query_proj + key_proj +
  log_temperature toward queries that select specific slots.
  Pair with diversification (Path 1) so slots are simultaneously
  differentiated and selectively retrievable.

DESIGN:
  Four arms, three seeds [42, 43, 44]. Part A (key_proj.bias=False)
  applied uniformly; writepath_mode held at "off" across all four
  arms to isolate the read path from the write rule (writes were
  shown to give bimodal slot collapse and only contribute via
  pre-existing seed luck; we do not want them confounding the
  selectivity signal here).

    B0_off              temp=False   ent_w=0.0   div_w=0.0
                          baseline; replicates EXQ-418e A0_off
                          (uniform-rail attention, inert cue_context)
    B1_sel_only         temp=True    ent_w=0.05  div_w=0.0
                          selectivity in isolation: does query-side
                          gradient pressure ALONE break the uniform
                          rail and produce behavioural delta?
    B2_div_only         temp=False   ent_w=0.0   div_w=0.5
                          replicates EXQ-418e A2_div_only as the
                          control / null arm: slot diversification
                          alone does not move attention or behaviour.
    B3_sel_plus_div     temp=True    ent_w=0.05  div_w=0.5
                          full Path-4 hypothesis: selectivity AND
                          diversification together produce sharp
                          attention against differentiated slots,
                          cue_context becomes informative end-to-end.

  All arms share sd016_enabled=True. SHY / sleep / serotonin disabled
  so the substrate effect is isolated. Training schedule matches
  EXQ-418e (P0=20 P1=40 STEPS=150 LAMBDA_TERRAIN=0.1 LAMBDA_CUE_ACTION=0.5).

PER-ARM METRICS:
  attn_entropy_mean       attention entropy under post-training queries.
                          uniform reference for num_slots=16: ln(16)=2.7726.
  slot_diversity          1 - mean off-diagonal pairwise cosine of
                          context_memory.memory rows post-train.
  action_class_entropy    entropy of action class argmax over
                          cue_action_proj outputs across the eval batch.
  log_tau_final           final value of E1.sd016_log_temperature
                          (None for arms B0_off, B2_div_only).
  n_writes                total ContextMemory.write() calls (expected
                          0 for all arms since writepath_mode='off').

ACCEPTANCE CRITERIA (per acceptance reframe in the brainstorm plan,
section F):
  C1  attn_entropy_mean < 2.65 in ALL three seeds for B1_sel_only AND
      B3_sel_plus_div. Selectivity arms must clear the uniform rail.
      (cutoff 2.65 = ~5% reduction from ln(16) = 2.7726.)
  C2  attn_entropy_mean > 2.65 in B0_off AND B2_div_only mean across
      seeds. No-selectivity arms must NOT move attention. Sanity check
      that diversification alone cannot move queries (replicates
      EXQ-418e A2_div_only verdict).
  C3  slot_diversity > 0.10 in ALL three seeds for B2_div_only AND
      B3_sel_plus_div. Diversification arms achieve symmetry break.
  C4  action_class_entropy(B1_sel_only) - action_class_entropy(B0_off)
      >= 0.10 nats. Selectivity ALONE produces behavioural delta --
      cue_context becomes informative without any write or
      diversification contribution. Resolves the open EXQ-418e
      question "does cue_context need to be informative for SD-016
      to do real work?".
  C5  action_class_entropy(B3_sel_plus_div) - action_class_entropy(B2_div_only)
      >= 0.10 nats. Adding selectivity ON TOP of diversification
      produces additional behavioural delta. Distinguishes the
      Path-4 hypothesis from a "writes alone explain everything"
      null.

  PASS = C1 AND C2 AND C3 AND C4 AND C5.
  C4 FAIL with C1 PASS would mean attention sharpened but cue_context
  still does not propagate behaviourally -- the bottleneck is then
  downstream of ContextMemory and Path 4 is not sufficient.
  C5 FAIL with C4 PASS would mean diversification adds nothing on top
  of selectivity (acceptable -- Path 4 alone suffices) but the
  experiment would still PASS scientifically (the hypothesis was
  selectivity is necessary, not strictly that they combine).
  We require C4 AND C5 here as a strong test; if only C4 PASSes the
  follow-up is a 2-arm sweep on selectivity weight.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418e
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


EXPERIMENT_TYPE    = "v3_exq_418g_sd016_selectivity_first_4arm"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES         = 20
P1_EPISODES         = 40
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
LAMBDA_DIVERSIFY    = 0.5
LAMBDA_ATTN_ENTROPY = 0.05
EVAL_BATCH_SIZE     = 32

LR    = 1e-4
SEEDS = [42, 43, 44]

# Arm spec: (label, temp_learnable, attn_entropy_weight, diversification_weight).
ARMS: List[Tuple[str, bool, float, float]] = [
    ("B0_off",            False, 0.0,                   0.0),
    ("B1_sel_only",       True,  LAMBDA_ATTN_ENTROPY,   0.0),
    ("B2_div_only",       False, 0.0,                   LAMBDA_DIVERSIFY),
    ("B3_sel_plus_div",   True,  LAMBDA_ATTN_ENTROPY,   LAMBDA_DIVERSIFY),
]

# Acceptance thresholds.
ATTN_ENTROPY_C1_THRESHOLD     = 2.65
ATTN_ENTROPY_C2_FLOOR         = 2.65   # no-sel arms must stay above this
SLOT_DIVERSITY_C3_THRESHOLD   = 0.10
ACTION_ENTROPY_C4_DELTA       = 0.10
ACTION_ENTROPY_C5_DELTA       = 0.10


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


def _make_agent(env: CausalGridWorldV2,
                temp_learnable: bool,
                div_weight: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9, alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="off",
        sd016_temperature_learnable=temp_learnable,
        sws_enabled=False, rem_enabled=False, shy_enabled=False,
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


def _install_write_counter(cm) -> List[int]:
    import types
    counter: List[int] = []
    orig_write = cm.write

    def _counted_write(self, state: torch.Tensor) -> None:
        counter.append(0)
        return orig_write(state)

    cm.write = types.MethodType(_counted_write, cm)
    return counter


def _slot_diversity(cm) -> float:
    with torch.no_grad():
        mn = F.normalize(cm.memory, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        mean_pair = float(off.sum().item() / max(1, n * (n - 1)))
        return max(0.0, 1.0 - mean_pair)


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


def _run_training_episode(agent, env, optimizer,
                          phase: str, attn_entropy_weight: float) -> int:
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

        # compute_prediction_loss adds the diversification term automatically
        # when sd016_diversification_weight > 0 and sd016_enabled.
        pred_loss = agent.compute_prediction_loss()
        t_loss    = compute_terrain_loss(agent, latent.z_world, hazard_max)
        total_loss = pred_loss + LAMBDA_TERRAIN * t_loss

        # Path 4 attention-entropy minimisation. The experiment script owns
        # the loss composition (parallel to terrain_loss / cue_action_loss);
        # the substrate exposes the loss method, the script weights it.
        if attn_entropy_weight > 0.0:
            ent_loss = agent.e1.compute_attention_entropy_loss(latent.z_world)
            total_loss = total_loss + attn_entropy_weight * ent_loss

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


def _run_one_arm_seed(
    arm_label: str,
    temp_learnable: bool,
    attn_entropy_weight: float,
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

    agent = _make_agent(env_safe, temp_learnable, div_weight)
    cm = agent.e1.context_memory

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    write_counter = _install_write_counter(cm)

    log_tau_init: Optional[float] = None
    if agent.e1.sd016_log_temperature is not None:
        log_tau_init = float(agent.e1.sd016_log_temperature.item())

    print(
        f"  [arm {arm_label} seed {seed}] temp={temp_learnable} "
        f"ent_w={attn_entropy_weight} div_w={div_weight} "
        f"log_tau_init={log_tau_init} p0={p0} p1={p1} steps/ep={STEPS_PER_EPISODE}",
        flush=True,
    )

    for ep in range(p0):
        env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P0", attn_entropy_weight)

    for ep in range(p1):
        abs_ep = p0 + ep
        env = env_dang if (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P1", attn_entropy_weight)

    z_world_batch = _collect_eval_batch(agent, env_safe, eval_n)

    n_writes = len(write_counter)
    slot_div = _slot_diversity(cm)
    attn_ent = _attn_entropy_mean(agent, z_world_batch)
    act_ent  = _action_class_entropy_under_cue_bias(agent, z_world_batch)

    log_tau_final: Optional[float] = None
    if agent.e1.sd016_log_temperature is not None:
        log_tau_final = float(agent.e1.sd016_log_temperature.item())

    print(
        f"  [arm {arm_label} seed {seed}] writes={n_writes} "
        f"slot_div={slot_div:.3f} attn_ent={attn_ent:.3f} act_ent={act_ent:.3f} "
        f"log_tau_final={log_tau_final}",
        flush=True,
    )

    return {
        "arm":                  arm_label,
        "temperature_learnable": temp_learnable,
        "attn_entropy_weight":  attn_entropy_weight,
        "diversification_weight": div_weight,
        "seed":                 seed,
        "n_writes":             n_writes,
        "slot_diversity":       slot_div,
        "attn_entropy_mean":    attn_ent,
        "action_class_entropy": act_ent,
        "log_tau_init":         log_tau_init,
        "log_tau_final":        log_tau_final,
    }


def _per_arm_summary(arm_results: List[Dict]) -> Dict:
    n_seeds = len(arm_results)
    return {
        "n_seeds":                  n_seeds,
        "n_writes_min":             min(r["n_writes"] for r in arm_results),
        "n_writes_max":             max(r["n_writes"] for r in arm_results),
        "slot_diversity_mean":      sum(r["slot_diversity"] for r in arm_results) / n_seeds,
        "slot_diversity_min":       min(r["slot_diversity"] for r in arm_results),
        "slot_diversity_seeds_pass": sum(1 for r in arm_results
                                         if r["slot_diversity"] > SLOT_DIVERSITY_C3_THRESHOLD),
        "attn_entropy_mean_mean":   sum(r["attn_entropy_mean"] for r in arm_results) / n_seeds,
        "attn_entropy_mean_max":    max(r["attn_entropy_mean"] for r in arm_results),
        "attn_entropy_seeds_below_c1": sum(1 for r in arm_results
                                            if r["attn_entropy_mean"] < ATTN_ENTROPY_C1_THRESHOLD),
        "action_class_entropy_mean": sum(r["action_class_entropy"] for r in arm_results) / n_seeds,
    }


def _evaluate_acceptance(per_arm: Dict[str, Dict]) -> Dict:
    n_seeds = len(SEEDS)

    # C1: attn_entropy_mean < 2.65 in ALL three seeds for B1 and B3.
    c1_per_arm = {
        arm: per_arm[arm]["attn_entropy_seeds_below_c1"] == n_seeds
        for arm in ("B1_sel_only", "B3_sel_plus_div")
        if arm in per_arm
    }
    c1_pass = all(c1_per_arm.values()) if c1_per_arm else False

    # C2: attn_entropy_mean_mean > 2.65 for B0_off AND B2_div_only.
    c2_per_arm = {
        arm: per_arm[arm]["attn_entropy_mean_mean"] > ATTN_ENTROPY_C2_FLOOR
        for arm in ("B0_off", "B2_div_only")
        if arm in per_arm
    }
    c2_pass = all(c2_per_arm.values()) if c2_per_arm else False

    # C3: slot_diversity > 0.10 in ALL three seeds for B2 and B3.
    c3_per_arm = {
        arm: per_arm[arm]["slot_diversity_seeds_pass"] == n_seeds
        for arm in ("B2_div_only", "B3_sel_plus_div")
        if arm in per_arm
    }
    c3_pass = all(c3_per_arm.values()) if c3_per_arm else False

    # C4: action_class_entropy(B1) - action_class_entropy(B0) >= 0.10
    b0_act = per_arm.get("B0_off", {}).get("action_class_entropy_mean", 0.0)
    b1_act = per_arm.get("B1_sel_only", {}).get("action_class_entropy_mean", 0.0)
    c4_delta = b1_act - b0_act
    c4_pass = c4_delta >= ACTION_ENTROPY_C4_DELTA

    # C5: action_class_entropy(B3) - action_class_entropy(B2) >= 0.10
    b2_act = per_arm.get("B2_div_only", {}).get("action_class_entropy_mean", 0.0)
    b3_act = per_arm.get("B3_sel_plus_div", {}).get("action_class_entropy_mean", 0.0)
    c5_delta = b3_act - b2_act
    c5_pass = c5_delta >= ACTION_ENTROPY_C5_DELTA

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass

    return {
        "C1_selectivity_arms_clear_uniform_rail": {
            "pass": c1_pass,
            "per_arm": c1_per_arm,
            "threshold": ATTN_ENTROPY_C1_THRESHOLD,
            "applies_to": ["B1_sel_only", "B3_sel_plus_div"],
        },
        "C2_no_selectivity_arms_stay_at_uniform_rail": {
            "pass": c2_pass,
            "per_arm": c2_per_arm,
            "floor": ATTN_ENTROPY_C2_FLOOR,
            "applies_to": ["B0_off", "B2_div_only"],
        },
        "C3_diversification_arms_break_symmetry": {
            "pass": c3_pass,
            "per_arm": c3_per_arm,
            "threshold": SLOT_DIVERSITY_C3_THRESHOLD,
            "applies_to": ["B2_div_only", "B3_sel_plus_div"],
        },
        "C4_selectivity_alone_produces_behavioural_delta": {
            "pass": c4_pass,
            "delta_b1_minus_b0": c4_delta,
            "threshold": ACTION_ENTROPY_C4_DELTA,
            "b0_action_class_entropy_mean": b0_act,
            "b1_action_class_entropy_mean": b1_act,
        },
        "C5_selectivity_adds_to_diversification": {
            "pass": c5_pass,
            "delta_b3_minus_b2": c5_delta,
            "threshold": ACTION_ENTROPY_C5_DELTA,
            "b2_action_class_entropy_mean": b2_act,
            "b3_action_class_entropy_mean": b3_act,
        },
        "overall_pass": overall_pass,
    }


def main(dry_run: bool = False):
    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    print(
        f"V3-EXQ-418g SD-016 Path 4 selectivity-first 4-arm "
        f"(seeds={SEEDS} dry_run={dry_run})",
        flush=True,
    )

    all_results: List[Dict] = []
    per_arm: Dict[str, List[Dict]] = {arm: [] for arm, _, _, _ in ARMS}

    for arm_label, temp_learnable, ent_w, div_w in ARMS:
        for seed in SEEDS:
            r = _run_one_arm_seed(
                arm_label, temp_learnable, ent_w, div_w,
                seed, p0, p1, eval_n,
            )
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries = {arm: _per_arm_summary(rs) for arm, rs in per_arm.items()}
    acceptance = _evaluate_acceptance(summaries)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(
        f"  [summary] "
        f"C1={acceptance['C1_selectivity_arms_clear_uniform_rail']['pass']} "
        f"C2={acceptance['C2_no_selectivity_arms_stay_at_uniform_rail']['pass']} "
        f"C3={acceptance['C3_diversification_arms_break_symmetry']['pass']} "
        f"C4={acceptance['C4_selectivity_alone_produces_behavioural_delta']['pass']} "
        f"C5={acceptance['C5_selectivity_adds_to_diversification']['pass']} "
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
        "supersedes":         "V3-EXQ-418e",
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_note": (
            "SD-016 Path 4 query-selectivity-first 4-arm. PASS = learnable "
            "attention temperature + entropy-minimisation loss (B1, B3) "
            "clears the uniform attention rail AND produces behavioural "
            "delta vs no-selectivity controls (B0, B2). The strong test: "
            "selectivity alone produces a behavioural delta (C4) AND adds "
            "on top of diversification (C5). FAIL on C1 invalidates "
            "Path 4 as wired (raise lambda_attn_ent or rescale init "
            "log_tau before re-running). FAIL on C2 invalidates the "
            "ablation -- something other than selectivity moved attention. "
            "FAIL on C4 with C1 PASS would mean attention sharpened but "
            "cue_context still does not propagate behaviourally; bottleneck "
            "is downstream of ContextMemory and Path 4 alone is "
            "insufficient. FAIL on C5 with C4 PASS = selectivity alone "
            "suffices, diversification adds nothing -- still informative "
            "but the experiment-level PASS bar is C1+C2+C3+C4+C5."
        ),
        "acceptance_checks":  acceptance,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "attn_entropy_c1_threshold":  ATTN_ENTROPY_C1_THRESHOLD,
            "attn_entropy_c2_floor":      ATTN_ENTROPY_C2_FLOOR,
            "slot_diversity_c3_threshold": SLOT_DIVERSITY_C3_THRESHOLD,
            "action_entropy_c4_delta":    ACTION_ENTROPY_C4_DELTA,
            "action_entropy_c5_delta":    ACTION_ENTROPY_C5_DELTA,
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
            "lambda_attn_entropy":  LAMBDA_ATTN_ENTROPY,
            "eval_batch_size":      eval_n,
            "lr":                   LR,
            "arms": [
                {
                    "label": label,
                    "temperature_learnable": temp_learnable,
                    "attn_entropy_weight":   ent_w,
                    "diversification_weight": div_w,
                }
                for label, temp_learnable, ent_w, div_w in ARMS
            ],
            "sd016_enabled":        True,
            "sd016_writepath_mode": "off",
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
