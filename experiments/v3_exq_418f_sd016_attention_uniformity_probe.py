#!/opt/local/bin/python3
"""
V3-EXQ-418f -- SD-016 attention uniformity diagnostic probe

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  EXQ-418d FAILed across all 4 SD-016 writepath arms; EXQ-418e FAILed
  Path 1 (slot diversification) at div_weight=0.5. Re-reading EXQ-418e
  decomposes the failure cleanly: A2_div_only achieved near-perfect slot
  orthogonality (slot_diversity_mean = 0.9999) yet attn_entropy_mean
  stayed pinned at the uniform rail (2.7726 = ln(16)) AND action_class
  entropy was identical between A0_off and A2_div_only (~1e-10 in both).
  Diversification fixes slot content but cannot move attention. The
  bottleneck is therefore QUERY SELECTIVITY -- world_query_proj produces
  q vectors whose dot products with the slot keys are too flat for
  softmax to pick anyone.

  Before committing to a fix (learned temperature, attention-entropy
  minimisation, sparse attention, codebook substrate), instrument the
  attention path to localise the cause: is the query magnitude too
  small? Are the keys too small? Is the sqrt(memory_dim) scale too
  aggressive? Is the dot-product distribution flat in absolute terms,
  or only after the divisor?

DESIGN:
  Single seed (42), single config matching EXQ-418e A1_writes_only
  (sd016_enabled=True, sd016_writepath_mode='sense_only',
  alpha_world=0.9, div_weight=0.0). Reduced training schedule (P0=10,
  P1=20) -- the failure mode reproduces from the very first step;
  longer training is unnecessary for diagnosis. Match seeds to EXQ-418e
  for reproducibility against the prior FAIL.

  After training, collect EVAL_BATCH_SIZE z_world samples from a
  random-policy rollout in the safe env (mirroring EXQ-418e), then
  recompute the SD-016 attention path manually and dump per-batch
  statistics on:

    q = world_query_proj(z_world)       [batch, memory_dim]
    k = key_proj(context_memory.memory) [num_slots, memory_dim]
    raw_scores = q @ k.T / scale        [batch, num_slots]
    weights    = softmax(raw_scores)    [batch, num_slots]

  Specifically:
    q_norm_mean / std / min / max
    k_norm_mean / std / min / max
    raw_dotprod_mean / std / min / max  (BEFORE the sqrt(memory_dim) divisor)
    scaled_dotprod_mean / std / min / max
    attn_entropy_mean / min / max
    attn_max_weight_mean / min / max
    target_scale_for_entropy_1p5  (what divisor would produce mean entropy 1.5?)
    slot_argmax_histogram         (which slot wins per sample, count of K samples)

ACCEPTANCE:
  D1  n_samples >= 16. Data was actually collected.
  D2  0 <= attn_entropy_mean <= ln(num_slots) + 1e-3. Entropy is in
      bounds (sanity check on the manual recomputation).
  D3  raw_dotprod_std > 1e-6. Queries actually vary across the batch
      (rules out the trivial "world_query_proj is identically zero"
      explanation).
  D4  q_norm_mean > 1e-4 AND k_norm_mean > 1e-4. Both sides of the
      dot-product carry signal.

  PASS = all four. This is diagnostic-only -- "PASS" means the probe
  ran successfully and produced interpretable measurements; FAIL means
  the probe itself broke. PASS DOES NOT CHANGE SD-016 CLAIM CONFIDENCE.

claim_ids: ["SD-016"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418e (in spirit -- this localises the failure mode)
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


EXPERIMENT_TYPE    = "v3_exq_418f_sd016_attention_uniformity_probe"
CLAIM_IDS          = ["SD-016"]
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES        = 10
P1_EPISODES        = 20
STEPS_PER_EPISODE  = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN     = 0.1
LAMBDA_CUE_ACTION  = 0.5
EVAL_BATCH_SIZE    = 64
LR                 = 1e-4
SEED               = 42

# Acceptance thresholds.
D1_MIN_SAMPLES         = 16
D3_DOTPROD_STD_FLOOR   = 1e-6
D4_NORM_FLOOR          = 1e-4


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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9, alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="sense_only",
        sd016_diversification_weight=0.0,
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


def _stat_block(t: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(t.mean().item()),
        "std":  float(t.std().item()),
        "min":  float(t.min().item()),
        "max":  float(t.max().item()),
    }


def _probe_attention(agent, z_world_batch: torch.Tensor) -> Dict:
    """Recompute the SD-016 attention path manually and return diagnostics."""
    cm = agent.e1.context_memory
    world_query_proj = agent.e1.world_query_proj
    memory_dim = cm.memory_dim
    num_slots  = cm.num_slots
    batch_size = z_world_batch.shape[0]

    with torch.no_grad():
        q = world_query_proj(z_world_batch)                          # [batch, memory_dim]
        k = cm.key_proj(cm.memory)                                   # [num_slots, memory_dim]
        # Raw dot products before scaling.
        raw = q @ k.t()                                              # [batch, num_slots]
        scale_used = float(agent.e1._sd016_attention_scale(memory_dim))
        scaled = raw / scale_used
        weights = F.softmax(scaled, dim=-1)                          # [batch, num_slots]
        probs = weights.clamp(min=1e-12)
        per_sample_entropy = -(probs * probs.log()).sum(dim=-1)      # [batch]
        per_sample_max     = weights.max(dim=-1).values              # [batch]
        slot_argmax        = weights.argmax(dim=-1)                  # [batch]
        slot_hist          = torch.bincount(slot_argmax, minlength=num_slots).cpu().tolist()

        q_norms = q.norm(dim=-1)                                     # [batch]
        k_norms = k.norm(dim=-1)                                     # [num_slots]

        # What divisor would produce mean entropy ~ 1.5 nats? Rough estimate
        # via a binary search over scale_factor in [0.05x, 50x]. Crude but
        # informative; only needed for diagnostic reporting.
        target_entropy = 1.5
        lo, hi = scale_used * 0.05, scale_used * 50.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            scaled_mid = raw / max(mid, 1e-6)
            w = F.softmax(scaled_mid, dim=-1).clamp(min=1e-12)
            ent = -(w * w.log()).sum(dim=-1).mean().item()
            # Smaller scale -> sharper attention -> lower entropy.
            if ent > target_entropy:
                hi = mid     # need sharper, decrease scale
            else:
                lo = mid     # need flatter, increase scale
        target_scale_for_entropy_1p5 = float((lo + hi) / 2.0)

    return {
        "n_samples":              int(batch_size),
        "num_slots":              int(num_slots),
        "memory_dim":             int(memory_dim),
        "scale_used":             scale_used,
        "uniform_entropy":        float(torch.log(torch.tensor(float(num_slots))).item()),
        "q_norm":                 _stat_block(q_norms),
        "k_norm":                 _stat_block(k_norms),
        "raw_dotprod":            _stat_block(raw),
        "scaled_dotprod":         _stat_block(scaled),
        "attn_entropy_mean":      float(per_sample_entropy.mean().item()),
        "attn_entropy_min":       float(per_sample_entropy.min().item()),
        "attn_entropy_max":       float(per_sample_entropy.max().item()),
        "attn_max_weight_mean":   float(per_sample_max.mean().item()),
        "attn_max_weight_min":    float(per_sample_max.min().item()),
        "attn_max_weight_max":    float(per_sample_max.max().item()),
        "slot_argmax_histogram":  slot_hist,
        "target_scale_for_entropy_1p5": target_scale_for_entropy_1p5,
    }


def _evaluate_acceptance(probe: Dict) -> Dict:
    n_samples = probe["n_samples"]
    uniform   = probe["uniform_entropy"]

    d1_pass = n_samples >= D1_MIN_SAMPLES
    d2_pass = (
        probe["attn_entropy_mean"] >= 0.0
        and probe["attn_entropy_mean"] <= uniform + 1e-3
    )
    d3_pass = probe["raw_dotprod"]["std"] > D3_DOTPROD_STD_FLOOR
    d4_pass = (
        probe["q_norm"]["mean"] > D4_NORM_FLOOR
        and probe["k_norm"]["mean"] > D4_NORM_FLOOR
    )
    overall = d1_pass and d2_pass and d3_pass and d4_pass

    return {
        "D1_data_collected": {
            "pass": d1_pass,
            "n_samples": n_samples,
            "threshold": D1_MIN_SAMPLES,
        },
        "D2_entropy_in_bounds": {
            "pass": d2_pass,
            "attn_entropy_mean": probe["attn_entropy_mean"],
            "uniform_reference": uniform,
        },
        "D3_queries_vary": {
            "pass": d3_pass,
            "raw_dotprod_std": probe["raw_dotprod"]["std"],
            "threshold": D3_DOTPROD_STD_FLOOR,
        },
        "D4_norms_nonzero": {
            "pass": d4_pass,
            "q_norm_mean": probe["q_norm"]["mean"],
            "k_norm_mean": probe["k_norm"]["mean"],
            "threshold": D4_NORM_FLOOR,
        },
        "overall_pass": overall,
    }


def main(dry_run: bool = False):
    p0 = P0_EPISODES if not dry_run else 2
    p1 = P1_EPISODES if not dry_run else 3
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    print(
        f"V3-EXQ-418f SD-016 attention-uniformity probe seed={SEED} dry_run={dry_run}",
        flush=True,
    )

    torch.manual_seed(SEED)
    random.seed(SEED)

    env_safe = _make_env_safe(SEED)
    env_dang = _make_env_dangerous(SEED)
    agent = _make_agent(env_safe)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    for ep in range(p0):
        env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P0")
    for ep in range(p1):
        abs_ep = p0 + ep
        env = env_dang if (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P1")

    z_world_batch = _collect_z_world_batch(agent, env_safe, eval_n)
    probe = _probe_attention(agent, z_world_batch)
    acceptance = _evaluate_acceptance(probe)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(
        f"  [probe] q_norm_mean={probe['q_norm']['mean']:.4f} "
        f"k_norm_mean={probe['k_norm']['mean']:.4f} "
        f"raw_dot_std={probe['raw_dotprod']['std']:.4e} "
        f"scale_used={probe['scale_used']:.3f} "
        f"attn_ent={probe['attn_entropy_mean']:.4f} "
        f"target_scale_ent1p5={probe['target_scale_for_entropy_1p5']:.3f}",
        flush=True,
    )
    print(f"  [acceptance] D1={acceptance['D1_data_collected']['pass']} "
          f"D2={acceptance['D2_entropy_in_bounds']['pass']} "
          f"D3={acceptance['D3_queries_vary']['pass']} "
          f"D4={acceptance['D4_norms_nonzero']['pass']} -> outcome={outcome}",
          flush=True)

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
        "supersedes":         "V3-EXQ-418e",
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "SD-016 attention-uniformity diagnostic probe. PURPOSE: localise "
            "the cause of the EXQ-418d/418e attn_entropy_mean=ln(16) "
            "uniform-rail failure (slot-side fixes alone cannot move "
            "attention; queries must be the bottleneck). Reports query/key "
            "norms, raw and scaled dot-product distributions, slot argmax "
            "histogram, attention entropy stats, and the divisor that would "
            "produce target mean entropy 1.5 nats. PASS = the probe ran "
            "successfully (diagnostic-only outcome). MUST NOT update SD-016 "
            "claim confidence -- this is instrumentation, not evidence."
        ),
        "acceptance_checks":  acceptance,
        "probe":              probe,
        "thresholds": {
            "d1_min_samples":          D1_MIN_SAMPLES,
            "d3_dotprod_std_floor":    D3_DOTPROD_STD_FLOOR,
            "d4_norm_floor":           D4_NORM_FLOOR,
        },
        "params": {
            "seed":                 SEED,
            "p0_episodes":          p0,
            "p1_episodes":          p1,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain":       LAMBDA_TERRAIN,
            "lambda_cue_action":    LAMBDA_CUE_ACTION,
            "eval_batch_size":      eval_n,
            "lr":                   LR,
            "sd016_enabled":        True,
            "sd016_writepath_mode": "sense_only",
            "sd016_diversification_weight": 0.0,
            "sd016_temperature_learnable":  False,
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
            seeds=None,
            script_path=Path(__file__),
        )
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
