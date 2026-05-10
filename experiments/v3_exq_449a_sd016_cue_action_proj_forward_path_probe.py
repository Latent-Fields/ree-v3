#!/opt/local/bin/python3
"""
V3-EXQ-449a -- SD-016 cue_action_proj forward-path instrumentation probe (EXP-0155)

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  V3-EXQ-449 confirmed (C1 PASS) that the baseline wiring gives cue_action_proj zero
  gradient (~0.0 mean grad norm, ~0.0 weight delta, 2 seeds, ~1.7k steps). The
  SUPERVISED arm DID move the weights (grad ~0.013, weight_delta ~0.21) but
  action_bias_divergence stayed at exactly 0.0 in both seeds. Weights train; agent
  action distribution does not change. Something downstream of cue_action_proj
  zeroes the signal before it reaches the selection path.

  Identify the specific downstream blocker before any EXQ-418b successor is queued.

DESIGN:
  Single-seed (42) short-protocol instrumentation run. P0=25 eps warmup, P1=50 eps
  supervised training (reuses EXQ-449 SUPERVISED arm unchanged), 12 eps eval x three
  regimes. Single configuration (sd016_enabled=True, sws_enabled=False,
  rem_enabled=False, shy_enabled=False, SD-017 orthogonal). No ree_core edits --
  script-level hook logging only.

  Training variants (to make regime g3 meaningful):
    W_std        : one training run with standard agent code (detach present at
                   agent.py line 723 inside _e1_tick).
    W_no_detach  : one training run with the line-723 detach monkey-patched out
                   so that _cue_action_bias is NOT detached before being cached.

  Eval regimes:
    g1 (supervised_loss_on_during_eval)   : use W_std, supervised loss ACTIVE during
                                            eval (weights continue to move).
    g2 (frozen_weights_post_training)     : use W_std, supervised loss OFF, frozen.
    g3 (detach_bypassed_during_training)  : use W_no_detach, supervised loss OFF,
                                            frozen.

INSTRUMENTATION (per-eval-step, logged to arrays):
  (a) extract_cue_context return -- action_bias and terrain_weight norms, per-channel std,
      nonzero fraction. Sampled full action_bias vector at 10 regular intervals per regime.
  (b) cue_action_proj(cue_context) output -- norm, per-channel value spread (std across
      the 16-dim action_object_dim), correlation with the supervised target for this step.
  (c) cue_terrain_proj(cue_context) output -- norm (control: should be non-zero since
      terrain_loss trained it).
  (d) ContextMemory.memory slot statistics -- per-slot norm, slot diversity (mean
      pairwise cosine distance across memory_dim), attention weight entropy under
      world_query_proj queries at this z_world.
  (e) cue_action_proj.weight summary -- Frobenius norm, max abs, nonzero fraction.
  (f) supervised target E2.action_object(z_world, a_executed).detach() norm.

  NOTE: The full chain from cue_action_proj output into E3.select() score
  composition runs entirely through _e3_tick -> hippocampal.propose_trajectories ->
  e2.rollout_with_world -> _score_trajectory. The eval here extracts action_bias
  directly via e1.extract_cue_context(z_world) (bypassing the cached
  self._cue_action_bias). The action_bias_divergence metric therefore isolates the
  PROJECTION output (cue_action_proj) from the selection path. If divergence is
  high at the projection output but behavioural divergence is zero, the blocker is
  between the projection and E3.select (inspected by reading the recorded action
  series and per-step first-action distributions).

ACCEPTANCE CRITERIA:
  C1 diagnosis-localised:
      At least one of the following forward-path tensors has mean norm below 1e-4
      across the g2 eval window, or is constant across contexts (per-channel std
      below 1e-4 on the full action_bias eval tensor):
        extract_cue_context.action_bias
        cue_action_proj output post-training
        ContextMemory attention weights diversity
      Offending tensor name recorded in the result pack.

  C2 detach-isolated (bonus):
      Regime g3 (detach bypassed during training, frozen eval) produces
      action_bias_divergence > 0.05. This would confirm the line-723 detach as a
      primary blocker and the fix becomes a single-line change.

  C3 context-memory-pre-zero check:
      extract_cue_context output norm > 1e-3 in at least 80 percent of eval steps
      under regime g2 (after P0 warmup). Rules out ContextMemory cold-zero at init.

  C4 supervised-target sanity:
      E2.action_object(z_world, a_executed).detach() norm > 1e-3 in at least 80
      percent of steps across P1 supervision. Rules out 'cue_action_proj trained to
      predict zero'.

  primary_pass = C1 AND C3 AND C4. (C2 is bonus confirmation.)

DECISION OUTCOMES (for downstream EXQ-418b successor):
  - C1+C3+C4 PASS and C2 PASS: queue EXQ-418b successor with a one-line
    agent.py fix (remove the line-723 detach) plus the EXQ-449 supervised loss.
  - C1+C3+C4 PASS, C2 FAIL: detach is not the sole blocker. Forward-path
    redesign needed (learnable scale, different injection point, or fold
    cue_action into E2.action_object() directly).
  - C3 FAIL: ContextMemory query path is the primary blocker. world_query_proj
    needs its own training signal, or ContextMemory writes must be activated.
  - C4 FAIL: supervised target is degenerate. E2.action_object head needs its
    own training first, or a different target must be chosen.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
"""

import os
import sys
import json
import math
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


EXPERIMENT_TYPE    = "v3_exq_449a_sd016_cue_action_proj_forward_path_probe"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES         = 25
P1_EPISODES         = 50
EVAL_EPISODES       = 12
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5

LR   = 1e-4
SEED = 42

BASELINE_NORM_THRESH      = 1e-4
CONTEXT_MEM_NORM_THRESH   = 1e-3
TARGET_NORM_THRESH        = 1e-3
COVERAGE_THRESH           = 0.80
DETACH_BYPASS_DIV_THRESH  = 0.05


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
    """Replace agent._e1_tick with a copy that does NOT detach action_bias
    before caching to self._cue_action_bias (agent.py line 723). Used for the
    W_no_detach training run. Preserves all other behaviour."""
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
                latent_state.z_world  # NOTE: no .detach() here (was line 721)
            )
            # NOTE: removed .detach() on the cached fields (was line 723-724)
            self._cue_action_bias    = action_bias
            self._cue_terrain_weight = terrain_weight
        else:
            self._cue_action_bias    = None
            self._cue_terrain_weight = None

        return e1_prior

    agent._e1_tick = types.MethodType(_e1_tick_nodetach, agent)


def _context_memory_stats(agent):
    """Per-slot norm and pairwise diversity of ContextMemory slots."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory  # [num_slots, memory_dim]
        per_slot_norms = mem.norm(dim=-1)
        mean_slot_norm = float(per_slot_norms.mean().item())
        min_slot_norm  = float(per_slot_norms.min().item())
        max_slot_norm  = float(per_slot_norms.max().item())

        mn = F.normalize(mem, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        mean_pair_sim = float(off.sum().item() / max(1, n * (n - 1)))
        slot_diversity = max(0.0, 1.0 - mean_pair_sim)

        return {
            "slot_norm_mean": mean_slot_norm,
            "slot_norm_min":  min_slot_norm,
            "slot_norm_max":  max_slot_norm,
            "slot_diversity": slot_diversity,
            "num_slots":      int(n),
        }


def _attention_stats(agent, z_world):
    """Attention weight entropy for the extract_cue_context query at this z_world."""
    with torch.no_grad():
        if z_world.dim() == 1:
            z_world = z_world.unsqueeze(0)
        batch_size = z_world.shape[0]
        memory_dim = agent.e1.context_memory.memory_dim
        q = agent.e1.world_query_proj(z_world).unsqueeze(1)
        memory = agent.e1.context_memory.memory
        k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1).squeeze(1)  # [batch, num_slots]
        probs = weights.clamp(min=1e-12)
        entropy = float(-(probs * probs.log()).sum(dim=-1).mean().item())
        max_weight = float(weights.max(dim=-1)[0].mean().item())
        return entropy, max_weight


def _extract_eval_record(agent, z_world, action_idx_onehot):
    """Extract one-step forward-path record for eval logging (no backprop)."""
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


def _run_eval_episode(
    agent,
    env,
    label,
    safe_records,
    dang_records,
    regime,
    optimizer=None,
):
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

        # Regime g1: supervised loss ACTIVE during eval -- weights keep moving.
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
    """Run one training variant (W_std or W_no_detach). Returns (agent,
    optimizer, training_diagnostics)."""
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

    target_positive = [v for v in cue_action_coverage if v > TARGET_NORM_THRESH]
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

    ab_coverage = sum(1 for v in ab_norms if v > CONTEXT_MEM_NORM_THRESH) / max(1, len(ab_norms))
    ao_coverage = sum(1 for v in ao_norms if v > TARGET_NORM_THRESH) / max(1, len(ao_norms))

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

    # --- Training variant A: W_std (standard agent code) ---
    agent_std, opt_std, diag_std, env_safe_std, env_dang_std = _train_variant(
        label="W_std", no_detach=False, dry_run=dry_run,
        total_training_eps=total_training_eps,
    )

    # --- Training variant B: W_no_detach (detach monkey-patched out) ---
    agent_nd, opt_nd, diag_nd, env_safe_nd, env_dang_nd = _train_variant(
        label="W_no_detach", no_detach=True, dry_run=dry_run,
        total_training_eps=total_training_eps,
    )

    # Regime g1: supervised loss ACTIVE during eval, W_std.
    print(f"  [eval] regime=g1 W_std supervised_active", flush=True)
    res_g1 = _run_eval_regime(agent_std, env_safe_std, env_dang_std,
                              regime="g1", optimizer=opt_std, dry_run=dry_run)
    # Regime g2: frozen eval, W_std.
    print(f"  [eval] regime=g2 W_std frozen", flush=True)
    res_g2 = _run_eval_regime(agent_std, env_safe_std, env_dang_std,
                              regime="g2", optimizer=None, dry_run=dry_run)
    # Regime g3: frozen eval, W_no_detach (trained with detach bypassed).
    print(f"  [eval] regime=g3 W_no_detach frozen", flush=True)
    res_g3 = _run_eval_regime(agent_nd, env_safe_nd, env_dang_nd,
                              regime="g3", optimizer=None, dry_run=dry_run)

    # ------- Acceptance evaluation -------
    # C1: some forward-path tensor is below threshold or is constant across contexts.
    ab_norm_mean_g2 = res_g2.get("action_bias_norm_mean", 0.0)
    ab_channel_std_g2 = res_g2.get("action_bias_per_channel_std", 0.0)
    cm_slot_div_g2    = res_g2.get("context_memory", {}).get("slot_diversity", 0.0)
    attn_entropy_g2   = res_g2.get("attn_entropy_mean", 0.0)

    c1_offenders = []
    if ab_norm_mean_g2 < BASELINE_NORM_THRESH:
        c1_offenders.append("extract_cue_context.action_bias norm < 1e-4 (g2)")
    if ab_channel_std_g2 < BASELINE_NORM_THRESH:
        c1_offenders.append("action_bias per-channel std < 1e-4 (constant output across contexts, g2)")
    if cm_slot_div_g2 < 0.01:
        c1_offenders.append("ContextMemory slot_diversity < 0.01 (slots collapsed, g2)")
    if attn_entropy_g2 < 0.05:
        c1_offenders.append("attention entropy < 0.05 (query picks a single slot deterministically, g2)")
    c1_pass = len(c1_offenders) > 0

    # C2: detach bypassed during training -> divergence > 0.05.
    c2_pass = res_g3.get("action_bias_div", 0.0) > DETACH_BYPASS_DIV_THRESH

    # C3: extract_cue_context output norm > 1e-3 in >=80 percent of g2 steps.
    c3_pass = res_g2.get("action_bias_coverage_gt_1e-3", 0.0) >= COVERAGE_THRESH

    # C4: supervised target norm > 1e-3 in >=80 percent of steps across P1 training.
    c4_pass = diag_std.get("target_coverage_gt_1e-3", 0.0) >= COVERAGE_THRESH

    primary_pass = c1_pass and c3_pass and c4_pass
    outcome = "PASS" if primary_pass else "FAIL"

    print(
        f"  [summary] C1_diagnosis_localised={c1_pass} (offenders={len(c1_offenders)}),"
        f" C2_detach_isolated={c2_pass} (g3_div={res_g3.get('action_bias_div', 0.0):.4f}),"
        f" C3_context_memory_pre_zero={c3_pass} (g2_coverage={res_g2.get('action_bias_coverage_gt_1e-3', 0.0):.3f}),"
        f" C4_supervised_target_sanity={c4_pass} (P1_coverage={diag_std.get('target_coverage_gt_1e-3', 0.0):.3f})"
        f" => primary_pass={primary_pass} -- {outcome}",
        flush=True,
    )

    # Diagnosis string for downstream governance.
    if primary_pass and c2_pass:
        diagnosis = (
            "line-723 detach in agent._e1_tick is the primary blocker. "
            "Removing it during training enables action_bias to diverge across "
            "contexts. Queue EXQ-418b successor with a one-line fix (drop the "
            "detach) plus the EXQ-449 supervised loss."
        )
    elif primary_pass and not c2_pass:
        diagnosis = (
            "detach is not the sole blocker. Forward-path tensor(s) below threshold: "
            + "; ".join(c1_offenders) + ". Requires redesigning how cue_action_proj "
            "output is consumed (learnable scale, different injection point, or "
            "folding cue_action into E2.action_object() directly)."
        )
    elif not c3_pass:
        diagnosis = (
            "ContextMemory query path is the primary blocker. extract_cue_context "
            "output norm falls below 1e-3 in more than 20 percent of eval steps. "
            "world_query_proj needs its own training signal or ContextMemory "
            "writes must be activated earlier."
        )
    elif not c4_pass:
        diagnosis = (
            "Supervised target is degenerate. E2.action_object(z_world, a).detach() "
            "norm falls below 1e-3 in more than 20 percent of training steps. The "
            "E2.action_object head needs its own training first, or a different "
            "supervised target must be chosen."
        )
    else:
        diagnosis = (
            "Diagnosis inconclusive under the current acceptance thresholds. "
            "Review per-regime metrics and raise a follow-up probe."
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
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "Diagnostic probe of SD-016 cue_action_proj forward path (EXP-0155). "
            "Outcome MUST NOT be used to update SD-016 claim confidence. Result "
            "identifies the specific blocker in the forward path and determines "
            "the fix plan for EXQ-418b re-run. " + diagnosis
        ),
        "acceptance_checks": {
            "C1_diagnosis_localised":      c1_pass,
            "C1_offenders":                c1_offenders,
            "C2_detach_isolated_bonus":    c2_pass,
            "C3_context_memory_pre_zero":  c3_pass,
            "C4_supervised_target_sanity": c4_pass,
            "primary_pass":                primary_pass,
            "baseline_norm_thresh":        BASELINE_NORM_THRESH,
            "context_memory_norm_thresh":  CONTEXT_MEM_NORM_THRESH,
            "target_norm_thresh":          TARGET_NORM_THRESH,
            "coverage_thresh":             COVERAGE_THRESH,
            "detach_bypass_div_thresh":    DETACH_BYPASS_DIV_THRESH,
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
