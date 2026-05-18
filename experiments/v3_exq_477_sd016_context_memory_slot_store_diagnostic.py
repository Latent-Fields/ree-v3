#!/opt/local/bin/python3
"""
V3-EXQ-477 -- SD-016 ContextMemory slot-store / attention-uniformity diagnostic (EXP-0155)

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  EXQ-449b (PASS) verified a consumer-side fix for cue_action_proj (concat z_world
  bypasses the uniform-attention bottleneck). The narrower upstream bug -- the
  ContextMemory slot-store producing uniform softmax -- is still present. Before
  any EXQ-418b successor can meaningfully exercise the SD-016 path, we need to
  localise WHICH component of the ContextMemory forward path keeps attention
  uniform. The 2026-04-24T07:36 retag note flagged this as a blocker on
  EXQ-436a, and the ree-v3 CLAUDE.md SD-016 section documents the symptom:
  entropy = ln(num_slots) = 2.7726 and cue_context constant across batch.

  Candidate root causes (non-exhaustive):

    (a) slot content stays near randn*0.01 because ContextMemory.write() is
        rarely called or always picks the same slot.
    (b) key_proj.bias norm dominates key_proj(memory_i) for every slot i, so
        all keys look approximately identical regardless of slot content.
    (c) write_gate sigmoid saturates to ~0.5 (scale ~50x slot scale), so
        writes overwrite slot content with a signal that is itself constant
        across batch, keeping keys undifferentiated even if write() fires.
    (d) query_proj output has a large bias relative to its z_world-dependent
        component, so q is batch-constant, collapsing scores to a single
        shared row regardless of key differentiation.

  This diagnostic instruments each of (a)-(d) independently so the result
  pinpoints the responsible path rather than reporting downstream symptom.

DESIGN:
  Single-seed, single-arm training trace over P0 warmup + P1 training under
  the same env schedule as EXQ-449b (safe / dangerous env context switches
  every 5 episodes, sd016_enabled=True). At three checkpoints -- pre-training,
  post-P0, post-P1 -- record the full diagnostic tensor bank plus per-channel
  std of cue_context across a held-out eval batch. Also log per-step write
  activity during training (write called? which slot? write_signal magnitude?).

  NO acceptance PASS/FAIL mapping to SD-016 confidence (diagnostic_probe,
  evidence_direction=diagnostic). The PASS/FAIL at script level indicates
  whether the instrumentation ran to completion and produced interpretable
  telemetry; the diagnosis_summary string carries the substantive result.

ACCEPTANCE (script-level, NOT claim evidence):
  C1 all three checkpoints produced (pre / post_p0 / post_p1).
  C2 at least 50 write() calls instrumented across training.
  C3 held-out eval batch produced non-empty cue_context samples.
  C4 no NaN/Inf in any instrumented tensor.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
proposal_id: EXP-0155 (the ContextMemory diagnostic referred to in CLAUDE.md
             SD-016 section and the 2026-04-24T07:36 TASK_CLAIMS retag note;
             distinct from the EXP-0155 proposal_id for Q-038 in proposals.v1.json,
             which has been marked blocked_substrate).
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE    = "v3_exq_477_sd016_context_memory_slot_store_diagnostic"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES         = 25
P1_EPISODES         = 50
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
EVAL_BATCH_SIZE     = 32

LR   = 1e-4
SEED = 42

# Thresholds used for diagnostic flag interpretation (NOT governance PASS/FAIL).
ATTENTION_UNIFORM_THRESH   = 2.67       # ln(16) - 0.1; below => meaningfully non-uniform
KEY_BIAS_DOMINANCE_WARN    = 1.0        # ||bias|| > mean(||W @ slot||) flags (b)
WRITE_GATE_SATURATION_WARN = 0.9        # sigmoid output > 0.9 flags write_gate saturation
CUE_CONTEXT_STD_MIN        = 1e-3       # per-channel std below this flags constant-output
SLOT_CONCENTRATION_WARN    = 0.5        # one slot chosen > 50% of writes => argmin degeneracy


# ---------------------------------------------------------------------------
# Env + agent plumbing (reuses EXQ-449b contract so results are comparable).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Instrumentation -- the substantive diagnostic probes.
# ---------------------------------------------------------------------------

def _slot_stats(cm) -> Dict:
    """Slot-store statistics: norms + pairwise cosine diversity."""
    with torch.no_grad():
        mem = cm.memory
        per_slot_norms = mem.norm(dim=-1)
        mn = F.normalize(mem, dim=-1)
        sim = torch.mm(mn, mn.t())
        n = sim.shape[0]
        off = sim - torch.eye(n, device=sim.device)
        mean_pair_sim = float(off.sum().item() / max(1, n * (n - 1)))
        slot_diversity = max(0.0, 1.0 - mean_pair_sim)
        return {
            "num_slots":       int(n),
            "slot_norm_mean":  float(per_slot_norms.mean().item()),
            "slot_norm_min":   float(per_slot_norms.min().item()),
            "slot_norm_max":   float(per_slot_norms.max().item()),
            "slot_norm_std":   float(per_slot_norms.std().item()),
            "slot_diversity":  slot_diversity,
            "mean_pair_cosine_sim": mean_pair_sim,
        }


def _key_bias_dominance(cm) -> Dict:
    """Candidate (b): key_proj.bias dominates key_proj(memory_i)."""
    with torch.no_grad():
        mem = cm.memory  # [num_slots, memory_dim]
        W = cm.key_proj.weight  # [memory_dim, memory_dim]
        b = cm.key_proj.bias    # [memory_dim]
        # linear output per slot: W @ slot + b
        out = mem @ W.t() + b           # [num_slots, memory_dim]
        out_minus_bias = mem @ W.t()    # content-only part
        bias_norm = float(b.norm().item())
        out_norm_mean = float(out.norm(dim=-1).mean().item())
        content_norm_mean = float(out_minus_bias.norm(dim=-1).mean().item())
        # Dominance ratio: bias norm vs mean content norm
        dominance_ratio = bias_norm / max(1e-12, content_norm_mean)
        # Pairwise cosine similarity BEFORE vs AFTER bias addition:
        on = F.normalize(out, dim=-1)
        cn = F.normalize(out_minus_bias, dim=-1)
        def _mean_offdiag(m):
            n = m.shape[0]
            off = m - torch.eye(n, device=m.device)
            return float(off.sum().item() / max(1, n * (n - 1)))
        keys_pair_sim_with_bias = _mean_offdiag(torch.mm(on, on.t()))
        keys_pair_sim_content_only = _mean_offdiag(torch.mm(cn, cn.t()))
        return {
            "bias_norm":                  bias_norm,
            "content_norm_mean":          content_norm_mean,
            "out_norm_mean":              out_norm_mean,
            "bias_over_content_ratio":    dominance_ratio,
            "keys_pair_sim_with_bias":    keys_pair_sim_with_bias,
            "keys_pair_sim_content_only": keys_pair_sim_content_only,
        }


def _query_stats(cm, z_world_batch) -> Dict:
    """Candidate (d): world_query_proj output constancy across batch."""
    # world_query_proj lives on E1 (module scope), not on cm; accept via hook:
    # We expect z_world_batch shape [B, world_dim].
    with torch.no_grad():
        # Callers pass world_query_proj separately via _forward_trace.
        return {}


def _attention_trace(cm, world_query_proj, z_world_batch) -> Dict:
    """Candidate (a)/(b)/(d) composite: full attention forward over a batch."""
    with torch.no_grad():
        batch_size = z_world_batch.shape[0]
        memory_dim = cm.memory_dim
        q = world_query_proj(z_world_batch).unsqueeze(1)  # [B, 1, D]
        mem = cm.memory
        k = cm.key_proj(mem).unsqueeze(0).expand(batch_size, -1, -1)
        v = cm.value_proj(mem).unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)  # [B,1,N]
        weights = F.softmax(scores, dim=-1).squeeze(1)                   # [B, N]
        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)          # [B, D]
        cue_context = cm.output_proj(context)                            # [B, latent]

        probs = weights.clamp(min=1e-12)
        entropy = float(-(probs * probs.log()).sum(dim=-1).mean().item())
        max_weight_mean = float(weights.max(dim=-1)[0].mean().item())

        # q stats across batch:
        q_flat = q.squeeze(1)  # [B, D]
        q_per_channel_std = float(q_flat.std(dim=0).mean().item())
        q_norm_mean = float(q_flat.norm(dim=-1).mean().item())
        q_pair_cos = torch.mm(
            F.normalize(q_flat, dim=-1), F.normalize(q_flat, dim=-1).t()
        )
        n = q_pair_cos.shape[0]
        q_pair_off = (q_pair_cos - torch.eye(n, device=q_pair_cos.device)).sum().item() / max(1, n * (n - 1))

        # cue_context stats across batch:
        cc_per_channel_std = float(cue_context.std(dim=0).mean().item())
        cc_norm_mean = float(cue_context.norm(dim=-1).mean().item())

        # scores stats: if all scores look similar across the slot dim, softmax is near uniform.
        scores_flat = scores.squeeze(1)  # [B, N]
        scores_range_mean = float((scores_flat.max(dim=-1)[0] - scores_flat.min(dim=-1)[0]).mean().item())

        return {
            "attn_entropy_mean":             entropy,
            "attn_entropy_uniform_reference": float(torch.log(torch.tensor(float(cm.num_slots))).item()),
            "attn_max_weight_mean":          max_weight_mean,
            "scores_range_mean":             scores_range_mean,
            "q_per_channel_std":             q_per_channel_std,
            "q_norm_mean":                   q_norm_mean,
            "q_pair_cosine_sim":             q_pair_off,
            "cue_context_per_channel_std":   cc_per_channel_std,
            "cue_context_norm_mean":         cc_norm_mean,
        }


def _write_gate_probe(cm, state_batch) -> Dict:
    """Candidate (c): write_gate sigmoid saturation and write_signal magnitude."""
    with torch.no_grad():
        write_signal = cm.write_gate(state_batch)    # [B, memory_dim] in (0, 1)
        mean_activation = float(write_signal.mean().item())
        saturated_frac  = float((write_signal > WRITE_GATE_SATURATION_WARN).float().mean().item())
        signal_norm_mean = float(write_signal.norm(dim=-1).mean().item())
        # Compared to slot scale (mem norm mean):
        mem_norm_mean = float(cm.memory.norm(dim=-1).mean().item())
        scale_ratio = signal_norm_mean / max(1e-12, mem_norm_mean)
        return {
            "write_gate_mean_activation": mean_activation,
            "write_gate_saturated_frac":  saturated_frac,
            "write_signal_norm_mean":     signal_norm_mean,
            "mem_norm_mean_at_probe":     mem_norm_mean,
            "write_signal_vs_slot_scale": scale_ratio,
        }


def _checkpoint(cm, world_query_proj, z_world_batch, state_batch, label) -> Dict:
    snap = {
        "checkpoint_label": label,
        "slot_stats":       _slot_stats(cm),
        "key_bias":         _key_bias_dominance(cm),
        "attention":        _attention_trace(cm, world_query_proj, z_world_batch),
        "write_gate":       _write_gate_probe(cm, state_batch),
    }
    # Sanity -- detect NaN / Inf anywhere in the checkpoint:
    def _scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{path}.{k}" if path else k)
        else:
            if isinstance(obj, float):
                if obj != obj or obj in (float("inf"), float("-inf")):
                    raise ValueError(f"NaN/Inf in checkpoint at {path}")
    _scan(snap)
    return snap


# ---------------------------------------------------------------------------
# Write-call instrumentation via monkeypatch: records every ContextMemory.write
# invocation during training. Records the argmin slot idx, write_signal norm,
# and query_norm to disambiguate (a) vs (c).
# ---------------------------------------------------------------------------

def _install_write_trace(cm, write_log: List[Dict]):
    import types

    orig_write = cm.write

    def _traced_write(self, state: torch.Tensor) -> None:
        with torch.no_grad():
            write_signal = self.write_gate(state)
            q = self.query_proj(state)
            scores = torch.mm(q, self.memory.t())
            min_idx = int(scores.mean(0).argmin().item())
            write_log.append({
                "step":                 len(write_log),
                "selected_slot":        min_idx,
                "write_signal_norm":    float(write_signal.norm(dim=-1).mean().item()),
                "write_gate_mean":      float(write_signal.mean().item()),
                "query_norm":           float(q.norm(dim=-1).mean().item()),
                "scores_range":         float((scores.max(dim=-1)[0]
                                               - scores.min(dim=-1)[0]).mean().item()),
            })
        return orig_write(state)

    cm.write = types.MethodType(_traced_write, cm)


def _summarise_write_log(write_log: List[Dict], num_slots: int) -> Dict:
    if not write_log:
        return {
            "n_writes": 0,
            "slot_histogram": [0] * num_slots,
            "slot_concentration_max": 0.0,
            "unique_slots_chosen": 0,
            "write_signal_norm_mean": 0.0,
            "write_gate_mean_mean":   0.0,
            "query_norm_mean":        0.0,
            "scores_range_mean":      0.0,
        }
    hist = [0] * num_slots
    for w in write_log:
        if 0 <= w["selected_slot"] < num_slots:
            hist[w["selected_slot"]] += 1
    total = sum(hist)
    concentration = max(hist) / total if total > 0 else 0.0
    unique = sum(1 for c in hist if c > 0)
    return {
        "n_writes": len(write_log),
        "slot_histogram": hist,
        "slot_concentration_max": concentration,
        "unique_slots_chosen": unique,
        "write_signal_norm_mean": sum(w["write_signal_norm"] for w in write_log) / len(write_log),
        "write_gate_mean_mean":   sum(w["write_gate_mean"]   for w in write_log) / len(write_log),
        "query_norm_mean":        sum(w["query_norm"]        for w in write_log) / len(write_log),
        "scores_range_mean":      sum(w["scores_range"]      for w in write_log) / len(write_log),
    }


# ---------------------------------------------------------------------------
# Held-out eval batch: pulls EVAL_BATCH_SIZE distinct z_world snapshots so the
# attention trace measures cross-input variation rather than within-episode
# autocorrelation.
# ---------------------------------------------------------------------------

def _collect_eval_batch(agent, env, n_samples) -> Dict[str, torch.Tensor]:
    device = agent.device
    z_world_list = []
    state_list = []

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(STEPS_PER_EPISODE * 4):
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
        state_list.append(
            torch.cat([latent.z_self, latent.z_world], dim=-1)
            .detach().clone().squeeze(0)
        )

        if len(z_world_list) >= n_samples:
            break

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)
        _, _h, done, _i, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()

    z_world_batch = torch.stack(z_world_list[:n_samples], dim=0)   # [n, world_dim]
    state_batch   = torch.stack(state_list[:n_samples],   dim=0)   # [n, total_dim]
    return {"z_world_batch": z_world_batch, "state_batch": state_batch}


# ---------------------------------------------------------------------------
# Training episode (mirrors EXQ-449b WARMUP/SUPERVISED arm).
# ---------------------------------------------------------------------------

def _run_training_episode(agent, env, optimizer, arm):
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
            ca_loss, _ = compute_cue_action_loss(agent, latent.z_world, action)
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
# Diagnosis synthesis: reads checkpoint deltas + write log and identifies
# which of the four candidates best explains the observed collapse.
# ---------------------------------------------------------------------------

def _synthesise_diagnosis(ckpts: Dict[str, Dict], write_summary: Dict) -> str:
    post = ckpts.get("post_p1", ckpts.get("post_p0"))
    pre  = ckpts.get("pre_training")
    if post is None or pre is None:
        return "Insufficient checkpoints for diagnosis."

    att = post["attention"]
    kb  = post["key_bias"]
    ss  = post["slot_stats"]
    wg  = post["write_gate"]

    flags = []
    if att["attn_entropy_mean"] > ATTENTION_UNIFORM_THRESH:
        flags.append(
            f"attention is uniform (entropy={att['attn_entropy_mean']:.3f} > "
            f"{ATTENTION_UNIFORM_THRESH}; reference={att['attn_entropy_uniform_reference']:.3f})"
        )
    if att["cue_context_per_channel_std"] < CUE_CONTEXT_STD_MIN:
        flags.append(
            f"cue_context is batch-constant (per-channel std={att['cue_context_per_channel_std']:.3e} "
            f"< {CUE_CONTEXT_STD_MIN:.0e})"
        )
    if kb["bias_over_content_ratio"] > KEY_BIAS_DOMINANCE_WARN:
        flags.append(
            f"key_proj.bias dominates (bias_over_content_ratio={kb['bias_over_content_ratio']:.2f} > "
            f"{KEY_BIAS_DOMINANCE_WARN}); keys_pair_sim_with_bias={kb['keys_pair_sim_with_bias']:.3f} vs "
            f"keys_pair_sim_content_only={kb['keys_pair_sim_content_only']:.3f}"
        )
    if wg["write_gate_saturated_frac"] > 0.5:
        flags.append(
            f"write_gate saturates (frac>0.9 = {wg['write_gate_saturated_frac']:.2f})"
        )
    if wg["write_signal_vs_slot_scale"] > 50.0:
        flags.append(
            f"write_signal >> slot scale ({wg['write_signal_vs_slot_scale']:.1f}x)"
        )
    if ss["slot_diversity"] < 0.1:
        flags.append(
            f"slot diversity collapsed (slot_diversity={ss['slot_diversity']:.3f})"
        )
    if write_summary["slot_concentration_max"] > SLOT_CONCENTRATION_WARN:
        flags.append(
            f"write() argmin degenerate (one slot chosen "
            f"{100.0 * write_summary['slot_concentration_max']:.0f}% of writes, "
            f"{write_summary['unique_slots_chosen']}/{ss['num_slots']} slots ever chosen)"
        )
    if att["q_per_channel_std"] < CUE_CONTEXT_STD_MIN:
        flags.append(
            f"world_query_proj output is batch-constant (per-channel std={att['q_per_channel_std']:.3e})"
        )

    # Candidate ranking
    primary = None
    if kb["bias_over_content_ratio"] > KEY_BIAS_DOMINANCE_WARN and kb["keys_pair_sim_content_only"] < kb["keys_pair_sim_with_bias"] - 0.1:
        primary = "(b) key_proj.bias dominates key content -- keys look identical across slots"
    elif att["q_per_channel_std"] < CUE_CONTEXT_STD_MIN:
        primary = "(d) world_query_proj output is batch-constant -- scores collapse regardless of key differentiation"
    elif ss["slot_diversity"] < 0.1 and write_summary["n_writes"] > 0:
        primary = "(a)/(c) slot content collapsed -- either writes rarely fire or write_signal is constant and overwrites slots with identical content"
    elif write_summary["n_writes"] == 0:
        primary = "(a) ContextMemory.write() never fired during training -- slots remain at randn*0.01 init"
    elif att["attn_entropy_mean"] <= ATTENTION_UNIFORM_THRESH:
        primary = "no candidate flagged -- attention is meaningfully non-uniform; investigate downstream output_proj or cue_action_proj"
    else:
        primary = "ambiguous -- multiple flags set but no single candidate dominates; inspect the per-checkpoint deltas"

    if not flags:
        flags_text = "no diagnostic flags triggered"
    else:
        flags_text = "; ".join(flags)

    return f"PRIMARY: {primary}. FLAGS: {flags_text}."


# ---------------------------------------------------------------------------
# main() driver.
# ---------------------------------------------------------------------------

def main(dry_run=False):
    torch.manual_seed(SEED)
    random.seed(SEED)

    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    total_training_eps = p0 + p1
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    env_safe = _make_env_safe(SEED)
    env_dang = _make_env_dangerous(SEED)

    agent = _make_agent(env_safe)
    cm = agent.e1.context_memory
    world_query_proj = agent.e1.world_query_proj

    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    write_log: List[Dict] = []
    _install_write_trace(cm, write_log)

    print(f"Seed {SEED} Condition DIAGNOSTIC", flush=True)
    print(
        f"  [setup] p0={p0} p1={p1} steps/ep={STEPS_PER_EPISODE} "
        f"eval_batch={eval_n} sd016_enabled=True",
        flush=True,
    )

    # --- Pre-training checkpoint ---
    eval_batch = _collect_eval_batch(agent, env_safe, eval_n)
    ckpt_pre = _checkpoint(
        cm, world_query_proj,
        eval_batch["z_world_batch"], eval_batch["state_batch"],
        "pre_training",
    )
    print(
        f"  [ckpt pre] slot_div={ckpt_pre['slot_stats']['slot_diversity']:.3f}"
        f" slot_norm_mean={ckpt_pre['slot_stats']['slot_norm_mean']:.3e}"
        f" bias_ratio={ckpt_pre['key_bias']['bias_over_content_ratio']:.2f}"
        f" attn_entropy={ckpt_pre['attention']['attn_entropy_mean']:.3f}"
        f" cc_std={ckpt_pre['attention']['cue_context_per_channel_std']:.3e}",
        flush=True,
    )

    # --- P0 warmup ---
    for ep in range(p0):
        if (ep + 1) % 10 == 0 or (ep + 1) == p0:
            print(
                f"  [train] label=DIAGNOSTIC ep {ep+1}/{total_training_eps} phase=P0",
                flush=True,
            )
        use_dang = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "WARMUP")

    writes_after_p0 = len(write_log)
    eval_batch = _collect_eval_batch(agent, env_safe, eval_n)
    ckpt_p0 = _checkpoint(
        cm, world_query_proj,
        eval_batch["z_world_batch"], eval_batch["state_batch"],
        "post_p0",
    )
    print(
        f"  [ckpt p0] writes={writes_after_p0}"
        f" slot_div={ckpt_p0['slot_stats']['slot_diversity']:.3f}"
        f" slot_norm_mean={ckpt_p0['slot_stats']['slot_norm_mean']:.3e}"
        f" bias_ratio={ckpt_p0['key_bias']['bias_over_content_ratio']:.2f}"
        f" attn_entropy={ckpt_p0['attention']['attn_entropy_mean']:.3f}"
        f" cc_std={ckpt_p0['attention']['cue_context_per_channel_std']:.3e}",
        flush=True,
    )

    # --- P1 supervised ---
    for ep in range(p1):
        abs_ep = p0 + ep
        if (abs_ep + 1) % 10 == 0 or (ep + 1) == p1:
            print(
                f"  [train] label=DIAGNOSTIC ep {abs_ep+1}/{total_training_eps} phase=P1",
                flush=True,
            )
        use_dang = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dang else env_safe
        _run_training_episode(agent, env, optimizer, "SUPERVISED")

    writes_after_p1 = len(write_log)
    eval_batch = _collect_eval_batch(agent, env_safe, eval_n)
    ckpt_p1 = _checkpoint(
        cm, world_query_proj,
        eval_batch["z_world_batch"], eval_batch["state_batch"],
        "post_p1",
    )
    print(
        f"  [ckpt p1] writes={writes_after_p1}"
        f" slot_div={ckpt_p1['slot_stats']['slot_diversity']:.3f}"
        f" slot_norm_mean={ckpt_p1['slot_stats']['slot_norm_mean']:.3e}"
        f" bias_ratio={ckpt_p1['key_bias']['bias_over_content_ratio']:.2f}"
        f" attn_entropy={ckpt_p1['attention']['attn_entropy_mean']:.3f}"
        f" cc_std={ckpt_p1['attention']['cue_context_per_channel_std']:.3e}",
        flush=True,
    )

    # --- Diagnosis ---
    write_summary = _summarise_write_log(write_log, num_slots=cm.num_slots)
    ckpts = {
        "pre_training": ckpt_pre,
        "post_p0":      ckpt_p0,
        "post_p1":      ckpt_p1,
    }
    diagnosis = _synthesise_diagnosis(ckpts, write_summary)

    # Script-level acceptance checks (NOT claim evidence).
    c1_pass = all(k in ckpts for k in ("pre_training", "post_p0", "post_p1"))
    c2_pass = write_summary["n_writes"] >= (50 if not dry_run else 5)
    c3_pass = ckpt_p1["attention"]["cue_context_norm_mean"] > 0.0
    # NaN scan already done inside _checkpoint; if no exception was raised, C4 PASS.
    c4_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    outcome = "PASS" if all_pass else "FAIL"

    print(
        f"  [summary] C1_ckpts={c1_pass} C2_writes={c2_pass}({write_summary['n_writes']})"
        f" C3_eval={c3_pass} C4_no_nan={c4_pass} => script_outcome={outcome}",
        flush=True,
    )
    print(f"  [diagnosis] {diagnosis}", flush=True)

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
            "Diagnostic probe of the SD-016 ContextMemory slot-store / attention-"
            "uniformity bug referenced in CLAUDE.md (SD-016 section) and in the "
            "2026-04-24T07:36 TASK_CLAIMS retag note. Outcome MUST NOT be used to "
            "update SD-016 claim confidence. Result provides targeted localisation "
            "of which ContextMemory forward-path component produces uniform "
            "softmax. " + diagnosis
        ),
        "acceptance_checks": {
            "C1_all_checkpoints_present": c1_pass,
            "C2_min_write_calls":         c2_pass,
            "C3_eval_batch_nonempty":     c3_pass,
            "C4_no_nan_inf":              c4_pass,
            "script_outcome":             outcome,
        },
        "diagnosis_summary": diagnosis,
        "checkpoints": {
            "pre_training": ckpt_pre,
            "post_p0":      ckpt_p0,
            "post_p1":      ckpt_p1,
        },
        "write_log_summary": write_summary,
        "thresholds": {
            "attention_uniform_thresh":    ATTENTION_UNIFORM_THRESH,
            "key_bias_dominance_warn":     KEY_BIAS_DOMINANCE_WARN,
            "write_gate_saturation_warn":  WRITE_GATE_SATURATION_WARN,
            "cue_context_std_min":         CUE_CONTEXT_STD_MIN,
            "slot_concentration_warn":     SLOT_CONCENTRATION_WARN,
        },
        "params": {
            "seed":                SEED,
            "p0_episodes":         p0,
            "p1_episodes":         p1,
            "steps_per_episode":   STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":      LAMBDA_TERRAIN,
            "lambda_cue_action":   LAMBDA_CUE_ACTION,
            "eval_batch_size":     eval_n,
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

    print(f"verdict: {outcome}", flush=True)
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
