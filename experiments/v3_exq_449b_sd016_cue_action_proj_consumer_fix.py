#!/opt/local/bin/python3
"""
V3-EXQ-449b -- SD-016 cue_action_proj consumer-fix verification (supersedes EXQ-449a)

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  EXQ-449a (PASS) localised the SD-016 cue_action_proj forward-path collapse to a
  uniform-attention bottleneck inside extract_cue_context: ContextMemory slots
  initialise at randn*0.01 so key_proj(memory) is dominated by its bias term, all
  keys look approximately identical, softmax produces exactly uniform weights
  (entropy = ln(num_slots) = 2.7726), and bmm(uniform, v) = mean(v) is constant
  across batch. cue_context = output_proj(constant) is constant; cue_action_proj(
  constant) is constant -- per-channel std collapsed to 2.7e-8 in g2.

  Fix (this experiment verifies): cue_action_proj now consumes [cue_context, z_world]
  instead of cue_context alone. This guarantees a non-collapsed input path even when
  the attention degenerates. cue_terrain_proj is left unchanged (terrain_loss already
  trains it usefully and is not the bottleneck under investigation here).

DESIGN:
  Same three-regime protocol as EXQ-449a (g1 supervised-active eval, g2 frozen post-
  training, g3 detach-bypassed-during-training frozen). Single seed, 25 + 50 episodes
  warmup + supervised, 12 eval episodes per regime. The instrumentation,
  thresholds, and reporting structure are reused so the result is directly
  comparable to v3_exq_449a_*.json.

  Acceptance pivots from "find the offender" (449a) to "verify the offender is gone":

  PRIMARY (the criterion the task brief specifies):
    P1 g2 action_bias_per_channel_std > 1e-3 (was 2.7e-8 under EXQ-449a)

  SECONDARY (the original 449a sanity gates, retained for context):
    S1 g2 action_bias_norm_mean > 1e-3      (path still produces non-zero output)
    S2 g2 ao_target_coverage > 80%          (supervised target sanity, was C4)
    S3 g2 action_bias_div > 0.05            (safe vs dangerous contexts diverge)

  PASS = P1 AND S1 AND S2.  S3 is a stronger behavioural signature; reported but
  not required for the script-level smoke pass (a successful S3 confirms the fix
  also unblocks EXQ-418b downstream).

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-449a
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_449b_sd016_cue_action_proj_consumer_fix"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES         = "V3-EXQ-449a"

P0_EPISODES         = 25
P1_EPISODES         = 50
EVAL_EPISODES       = 12
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5

LR   = 1e-4
SEED = 42

# Acceptance thresholds for the consumer-fix verification.
PRIMARY_PER_CHANNEL_STD_THRESH = 1e-3   # g2 must clear this (was 2.7e-8 in EXQ-449a)
NORM_THRESH                    = 1e-3
COVERAGE_THRESH                = 0.80
DIVERGENCE_THRESH              = 0.05


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
    return F.mse_loss(action_bias, ao_target.detach()), ao_target.detach()


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


def _monkeypatch_remove_detach(agent):
    """Replace agent._e1_tick with a copy that does NOT detach action_bias before
    caching to self._cue_action_bias. Used for the W_no_detach training run."""
    import types

    def _e1_tick_nodetach(self, latent_state):
        total_state = torch.cat([latent_state.z_self, latent_state.z_world], dim=-1)
        self._self_experience_buffer.append(latent_state.z_self.detach().clone())
        self._world_experience_buffer.append(latent_state.z_world.detach().clone())
        for buf in [self._self_experience_buffer, self._world_experience_buffer]:
            if len(buf) > 1000:
                del buf[:-1000]

        _z_goal_input = None
        _goal_cfg = getattr(self.config, "goal", None)
        if (self.goal_state is not None
                and _goal_cfg is not None
                and _goal_cfg.e1_goal_conditioned
                and self.goal_state.is_active()):
            _z_goal_input = self.goal_state.z_goal
        _, e1_prior = self.e1(total_state, z_goal=_z_goal_input)

        self.theta_buffer.update(latent_state.z_world, latent_state.z_self)
        self.clock.update_e3_rate_from_beta(latent_state.z_beta)

        schema_sal = self.e1.get_schema_salience()
        self._schema_salience = schema_sal.detach() if schema_sal is not None else None

        if hasattr(self.e1, 'world_query_proj'):
            action_bias, terrain_weight = self.e1.extract_cue_context(
                latent_state.z_world
            )
            self._cue_action_bias    = action_bias
            self._cue_terrain_weight = terrain_weight
        else:
            self._cue_action_bias    = None
            self._cue_terrain_weight = None

        return e1_prior

    agent._e1_tick = types.MethodType(_e1_tick_nodetach, agent)


def _context_memory_stats(agent):
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        per_slot_norms = mem.norm(dim=-1)
        mn = F.normalize(mem, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        mean_pair_sim = float(off.sum().item() / max(1, n * (n - 1)))
        slot_diversity = max(0.0, 1.0 - mean_pair_sim)
        return {
            "slot_norm_mean": float(per_slot_norms.mean().item()),
            "slot_norm_min":  float(per_slot_norms.min().item()),
            "slot_norm_max":  float(per_slot_norms.max().item()),
            "slot_diversity": slot_diversity,
            "num_slots":      int(n),
        }


def _attention_stats(agent, z_world):
    with torch.no_grad():
        if z_world.dim() == 1:
            z_world = z_world.unsqueeze(0)
        batch_size = z_world.shape[0]
        memory_dim = agent.e1.context_memory.memory_dim
        q = agent.e1.world_query_proj(z_world).unsqueeze(1)
        memory = agent.e1.context_memory.memory
        k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1).squeeze(1)
        probs = weights.clamp(min=1e-12)
        entropy = float(-(probs * probs.log()).sum(dim=-1).mean().item())
        max_weight = float(weights.max(dim=-1)[0].mean().item())
        return entropy, max_weight


def _extract_eval_record(agent, z_world, action_idx_onehot):
    with torch.no_grad():
        zw = z_world.detach()
        if zw.dim() == 1:
            zw = zw.unsqueeze(0)
        action_bias, terrain_weight = agent.e1.extract_cue_context(zw)
        ao_target = agent.e2.action_object(zw, action_idx_onehot)
        entropy, max_w = _attention_stats(agent, zw)
        ab = action_bias.squeeze(0)
        tw = terrain_weight.squeeze(0)
        ao = ao_target.squeeze(0)
        return {
            "action_bias":              ab.cpu(),
            "terrain_weight":           tw.cpu(),
            "ao_target":                ao.cpu(),
            "action_bias_norm":         float(ab.norm().item()),
            "terrain_weight_norm":      float(tw.norm().item()),
            "ao_target_norm":           float(ao.norm().item()),
            "attn_entropy":             entropy,
            "attn_max_weight":          max_w,
        }


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


def _per_channel_std(biases: List[torch.Tensor]) -> float:
    if len(biases) < 2:
        return 0.0
    with torch.no_grad():
        mat = torch.stack(biases)
        return float(mat.std(dim=0).mean().item())


def _run_training_episode(agent, env, optimizer, arm, cue_action_coverage):
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
            ca_loss, ao_target = compute_cue_action_loss(agent, latent.z_world, action)
            total_loss = total_loss + LAMBDA_CUE_ACTION * ca_loss
            cue_action_coverage.append(float(ao_target.norm().item()))

        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    return ep_steps


def _run_eval_episode(agent, env, label, safe_records, dang_records, regime, optimizer=None):
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

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        rec = _extract_eval_record(agent, latent.z_world, action)
        if label == "SAFE":
            safe_records.append(rec)
        else:
            dang_records.append(rec)

        if regime == "g1" and optimizer is not None:
            hazard_max = get_hazard_max(obs_dict, ow)
            pred_loss = agent.compute_prediction_loss()
            t_loss    = compute_terrain_loss(agent, latent.z_world, hazard_max)
            ca_loss, _ = compute_cue_action_loss(agent, latent.z_world, action)
            total_loss = pred_loss + LAMBDA_TERRAIN * t_loss + LAMBDA_CUE_ACTION * ca_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            break


def _train_variant(label, no_detach, dry_run, total_training_eps):
    torch.manual_seed(SEED)
    random.seed(SEED)

    env_safe = _make_env_safe(SEED)
    env_dang = _make_env_dangerous(SEED)

    agent = _make_agent(env_safe)
    if no_detach:
        _monkeypatch_remove_detach(agent)

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3

    cue_action_coverage: List[float] = []
    w_init = agent.e1.cue_action_proj.weight.detach().clone()

    print(f"Seed {SEED} Condition {label}", flush=True)
    print(f"  [variant] label={label} no_detach={no_detach} p0={p0} p1={p1}", flush=True)

    for ep in range(p0):
        if (ep + 1) % 10 == 0 or (ep + 1) == p0:
            print(f"  [train] label={label} ep {ep+1}/{total_training_eps} phase=P0", flush=True)
        use_dang = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "WARMUP", cue_action_coverage)

    for ep in range(p1):
        abs_ep = p0 + ep
        if (abs_ep + 1) % 10 == 0 or (ep + 1) == p1:
            print(f"  [train] label={label} ep {abs_ep+1}/{total_training_eps} phase=P1", flush=True)
        use_dang = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "SUPERVISED", cue_action_coverage)

    w_final = agent.e1.cue_action_proj.weight.detach().clone()
    weight_delta_norm = float((w_final - w_init).norm().item())
    weight_norm       = float(w_final.norm().item())
    weight_max_abs    = float(w_final.abs().max().item())
    weight_nonzero_frac = float((w_final.abs() > 1e-8).float().mean().item())

    target_positive = [v for v in cue_action_coverage if v > NORM_THRESH]
    target_coverage = (len(target_positive) / len(cue_action_coverage)) if cue_action_coverage else 0.0
    target_norm_mean = (sum(cue_action_coverage) / len(cue_action_coverage)) if cue_action_coverage else 0.0

    diag = {
        "label":                 label,
        "no_detach":             no_detach,
        "weight_delta_norm":     weight_delta_norm,
        "weight_norm_final":     weight_norm,
        "weight_max_abs_final":  weight_max_abs,
        "weight_nonzero_frac":   weight_nonzero_frac,
        "target_norm_mean":      target_norm_mean,
        "target_coverage_gt_1e-3": target_coverage,
        "n_target_samples":      len(cue_action_coverage),
    }
    print(f"verdict: variant={label} weight_delta_norm={weight_delta_norm:.4e} target_cov={target_coverage:.3f}", flush=True)
    return agent, optimizer, diag, env_safe, env_dang


def _run_eval_regime(agent, env_safe, env_dang, regime, optimizer, dry_run):
    ev = EVAL_EPISODES if not dry_run else 4
    safe_records: List[Dict] = []
    dang_records: List[Dict] = []

    for i in range(ev):
        if i % 2 == 0:
            _run_eval_episode(
                agent, env_safe, "SAFE", safe_records, dang_records,
                regime=regime, optimizer=optimizer,
            )
        else:
            _run_eval_episode(
                agent, env_dang, "DANGEROUS", safe_records, dang_records,
                regime=regime, optimizer=optimizer,
            )

    all_records = safe_records + dang_records
    if not all_records:
        return {
            "regime": regime,
            "n_safe": 0,
            "n_dang": 0,
            "error": "no eval records",
        }

    safe_biases = [r["action_bias"] for r in safe_records]
    dang_biases = [r["action_bias"] for r in dang_records]
    action_bias_div = _action_bias_divergence(safe_biases, dang_biases)

    ab_norms = [r["action_bias_norm"] for r in all_records]
    tw_norms = [r["terrain_weight_norm"] for r in all_records]
    ao_norms = [r["ao_target_norm"] for r in all_records]
    attn_ent = [r["attn_entropy"] for r in all_records]
    attn_max = [r["attn_max_weight"] for r in all_records]

    ab_coverage = sum(1 for v in ab_norms if v > NORM_THRESH) / max(1, len(ab_norms))
    ao_coverage = sum(1 for v in ao_norms if v > NORM_THRESH) / max(1, len(ao_norms))

    per_channel_std_all = _per_channel_std([r["action_bias"] for r in all_records])
    cm_stats = _context_memory_stats(agent)

    sampled_biases = []
    if len(all_records) >= 10:
        stride = max(1, len(all_records) // 10)
        for i in range(0, len(all_records), stride):
            if len(sampled_biases) >= 10:
                break
            sampled_biases.append(all_records[i]["action_bias"].tolist())

    result = {
        "regime":                  regime,
        "n_safe":                  len(safe_records),
        "n_dang":                  len(dang_records),
        "action_bias_div":         action_bias_div,
        "action_bias_norm_mean":   float(sum(ab_norms) / len(ab_norms)),
        "action_bias_norm_min":    float(min(ab_norms)),
        "action_bias_norm_max":    float(max(ab_norms)),
        "action_bias_per_channel_std": per_channel_std_all,
        "action_bias_coverage_gt_1e-3": ab_coverage,
        "terrain_weight_norm_mean": float(sum(tw_norms) / len(tw_norms)),
        "ao_target_norm_mean":     float(sum(ao_norms) / len(ao_norms)),
        "ao_target_coverage_gt_1e-3": ao_coverage,
        "attn_entropy_mean":       float(sum(attn_ent) / len(attn_ent)),
        "attn_max_weight_mean":    float(sum(attn_max) / len(attn_max)),
        "context_memory":          cm_stats,
        "sampled_action_biases":   sampled_biases,
    }

    print(
        f"  verdict: regime={regime}"
        f" action_bias_div={action_bias_div:.4f}"
        f" ab_norm_mean={result['action_bias_norm_mean']:.3e}"
        f" ab_channel_std={per_channel_std_all:.3e}"
        f" ao_coverage={ao_coverage:.3f}"
        f" attn_entropy={result['attn_entropy_mean']:.3f}"
        f" slot_div={cm_stats['slot_diversity']:.3f}",
        flush=True,
    )
    return result


def main(dry_run=False):
    total_training_eps = (P0_EPISODES + P1_EPISODES) if not dry_run else (2 + 3)

    agent_std, opt_std, diag_std, env_safe_std, env_dang_std = _train_variant(
        label="W_std", no_detach=False, dry_run=dry_run,
        total_training_eps=total_training_eps,
    )

    agent_nd, opt_nd, diag_nd, env_safe_nd, env_dang_nd = _train_variant(
        label="W_no_detach", no_detach=True, dry_run=dry_run,
        total_training_eps=total_training_eps,
    )

    print(f"  [eval] regime=g1 W_std supervised_active", flush=True)
    res_g1 = _run_eval_regime(agent_std, env_safe_std, env_dang_std,
                              regime="g1", optimizer=opt_std, dry_run=dry_run)
    print(f"  [eval] regime=g2 W_std frozen", flush=True)
    res_g2 = _run_eval_regime(agent_std, env_safe_std, env_dang_std,
                              regime="g2", optimizer=None, dry_run=dry_run)
    print(f"  [eval] regime=g3 W_no_detach frozen", flush=True)
    res_g3 = _run_eval_regime(agent_nd, env_safe_nd, env_dang_nd,
                              regime="g3", optimizer=None, dry_run=dry_run)

    # ------- Acceptance evaluation (consumer-fix verification) -------
    g2_per_channel_std = res_g2.get("action_bias_per_channel_std", 0.0)
    g2_norm_mean       = res_g2.get("action_bias_norm_mean", 0.0)
    g2_div             = res_g2.get("action_bias_div", 0.0)

    p1_pass = g2_per_channel_std > PRIMARY_PER_CHANNEL_STD_THRESH
    s1_pass = g2_norm_mean > NORM_THRESH
    s2_pass = diag_std.get("target_coverage_gt_1e-3", 0.0) >= COVERAGE_THRESH
    s3_pass = g2_div > DIVERGENCE_THRESH

    primary_pass = p1_pass and s1_pass and s2_pass
    outcome = "PASS" if primary_pass else "FAIL"

    print(
        f"  [summary] P1_per_channel_std={p1_pass} (g2={g2_per_channel_std:.3e} > {PRIMARY_PER_CHANNEL_STD_THRESH:.0e}),"
        f" S1_norm_mean={s1_pass} (g2={g2_norm_mean:.3e}),"
        f" S2_target_sanity={s2_pass} (P1_cov={diag_std.get('target_coverage_gt_1e-3', 0.0):.3f}),"
        f" S3_action_bias_div={s3_pass} (g2={g2_div:.4f})"
        f" => primary_pass={primary_pass} -- {outcome}",
        flush=True,
    )

    if primary_pass and s3_pass:
        diagnosis = (
            "Consumer fix verified. action_bias per-channel std exceeds 1e-3 in g2 "
            "(was 2.7e-8 in EXQ-449a) and safe vs dangerous contexts diverge "
            "above the 0.05 threshold. Cleared to queue EXQ-418b integration test."
        )
    elif primary_pass and not s3_pass:
        diagnosis = (
            "Consumer fix verified at the projection level (per-channel std clears "
            "1e-3) but safe vs dangerous divergence is below 0.05. The fix removes "
            "the constant-output failure mode but contextual selectivity is weak "
            "under random-action eval; integration test EXQ-418b can still proceed "
            "(the original failure was action_bias_divergence=0.0; non-zero is the "
            "necessary precondition the integration test was blocked on)."
        )
    elif not p1_pass:
        diagnosis = (
            "Consumer fix did NOT clear the per-channel std threshold. Output is "
            "still constant across contexts. Re-examine cue_action_proj input "
            "wiring; the [cue_context, z_world] concat may not have propagated."
        )
    elif not s2_pass:
        diagnosis = (
            "Supervised target sanity check failed (P1 coverage below 80 percent). "
            "E2.action_object head is producing degenerate output during training; "
            "address before re-evaluating the consumer fix."
        )
    else:
        diagnosis = (
            "Consumer fix produces non-zero output but g2 norm is below 1e-3. "
            "Inspect cue_action_proj initialisation and gradient flow."
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
        "supersedes":         SUPERSEDES,
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "Diagnostic verification of the EXQ-449a consumer fix for SD-016 "
            "cue_action_proj. Outcome MUST NOT be used to update SD-016 claim "
            "confidence. Verifies that cue_action_proj output has per-channel "
            "variation across contexts after the [cue_context, z_world] concat "
            "fix in extract_cue_context. " + diagnosis
        ),
        "acceptance_checks": {
            "P1_per_channel_std":      p1_pass,
            "S1_norm_mean":            s1_pass,
            "S2_target_sanity":        s2_pass,
            "S3_action_bias_div":      s3_pass,
            "primary_pass":            primary_pass,
            "primary_per_channel_std_thresh": PRIMARY_PER_CHANNEL_STD_THRESH,
            "norm_thresh":             NORM_THRESH,
            "coverage_thresh":         COVERAGE_THRESH,
            "divergence_thresh":       DIVERGENCE_THRESH,
        },
        "diagnosis_summary": diagnosis,
        "training_variants": {
            "W_std":       diag_std,
            "W_no_detach": diag_nd,
        },
        "eval_regimes": {
            "g1_supervised_active_during_eval": res_g1,
            "g2_frozen_weights_post_training":  res_g2,
            "g3_detach_bypassed_during_training": res_g3,
        },
        "params": {
            "seed":                SEED,
            "p0_episodes":         P0_EPISODES if not dry_run else 2,
            "p1_episodes":         P1_EPISODES if not dry_run else 3,
            "eval_episodes":       EVAL_EPISODES if not dry_run else 4,
            "steps_per_episode":   STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":      LAMBDA_TERRAIN,
            "lambda_cue_action":   LAMBDA_CUE_ACTION,
            "sd016_enabled":       True,
            "sws_enabled":         False,
            "rem_enabled":         False,
            "shy_enabled":         False,
            "dry_run":             dry_run,
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
            seeds=None,
            script_path=Path(__file__),
        )
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal episodes to verify wiring")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
