#!/opt/local/bin/python3
"""
V3-EXQ-418m -- SD-016 Path 3 feedforward cue->slot tagger (substrate readiness)

EXPERIMENT_PURPOSE: diagnostic

QUESTION:
  V3-EXQ-418i (the recommended div_weight sweep at 1.0/2.0/5.0) FAILed C1 at
  every weight and concluded Path 1 (auxiliary slot diversification loss) is
  "insufficient regardless of weight; the attention bottleneck is categorically
  in query selectivity, not slot orthogonality." The z_world-only q.k attention
  inside extract_cue_context is pinned at the uniform ln(num_slots)=2.7726
  saddle: with memory init 0.01 the slot keys are near-identical, the softmax
  over them is near-uniform, and the softmax Jacobian at uniform is flat, so the
  cue_terrain_proj terrain_loss gradient cannot push attention off the rail.

  SD-016 Path 3 (implemented 2026-06-05) replaces ONLY the slot-SELECTION
  scoring inside extract_cue_context with a fresh feedforward MLP cue_slot_tagger
  (z_world -> slot logits). A random MLP produces non-uniform logits from step 0,
  so it sits OFF the saddle and the existing terrain_loss gradient flows into it
  and shapes contextual selectivity. The slot-CONTENT path (value_proj ->
  output_proj -> cue_context) and both downstream projections (cue_action_proj
  retaining the EXQ-449a z_world concat; cue_terrain_proj) are untouched.

  Does enabling the tagger break the uniform selection saddle that Path 1 could
  not? This is a substrate-readiness MEASUREMENT, not governance evidence.

DESIGN:
  Two arms, three seeds [42, 43, 44]. Both arms sd016_enabled=True,
  writepath_mode="off", temperature_learnable=False. Training schedule + losses
  match V3-EXQ-418i (P0=20 P1=40 STEPS=150 LAMBDA_TERRAIN=0.1
  LAMBDA_CUE_ACTION=0.5) so the only inter-arm difference is the tagger flag.

    A0_OFF   sd016_cue_slot_tagger=False  (legacy q.k attention -- the 418i saddle)
    A1_ON    sd016_cue_slot_tagger=True   (Path 3 feedforward tagger)

  Selection entropy is measured uniformly across both arms by calling
  extract_cue_context over the eval batch and reading the cached
  E1DeepPredictor._last_cue_slot_weights -- the OFF arm caches the q.k softmax
  weights, the ON arm caches the tagger softmax weights, so the metric compares
  the active selection mechanism in each arm on equal footing.

PER-ARM METRICS:
  sel_entropy_mean          mean slot-selection entropy over eval batch;
                            uniform reference ln(16) ~= 2.7726.
  action_bias_per_channel_std  std of cue_action_proj output across the eval
                            batch, mean over channels (the EXQ-449 collapsed-
                            output metric: 2.7e-8 collapsed; 449b cleared 1e-3).
  action_bias_div           safe-vs-dangerous mean action_bias divergence (L2),
                            the EXQ-449b S3 metric (threshold 0.05). REPORTED,
                            NOT GATED: full propagation depends on cue_action_proj
                            whose gradient path is the separate SD-055 concern.

ACCEPTANCE CRITERIA:
  C1 (PRIMARY)  A1_ON sel_entropy_mean < 2.5 on a MAJORITY of seeds (>=2/3).
                PASS = the tagger breaks the uniform selection saddle that
                Path 1 (418i) could not, regardless of slot pressure.
  C2 (CONTROL)  A0_OFF sel_entropy_mean > 2.65 on a majority of seeds
                (reproduces the ~ln(16) uniform saddle). Confirms the ablation
                isolates the tagger and the substrate is consistent with 418i.

  PASS: C1 AND C2.
  C1 FAIL with C2 PASS = the tagger does not establish selectivity under the
    terrain_loss gradient at this schedule -> route to /failure-autopsy (tagger
    capacity / training-signal, NOT the saddle).
  C2 FAIL = substrate inconsistency (OFF arm not on the saddle); investigate
    before interpreting C1.

  SECONDARY (reported, not gated): action_bias_per_channel_std and
    action_bias_div per arm. A lift in the ON arm is consistent with retrieval
    selectivity propagating toward action bias, but full action_bias_div>=0.05
    is gated on cue_action_proj / SD-055 differentiable CEM, so it is recorded
    as context, not a pass gate.

claim_ids: []   (diagnostic -- substrate readiness, weights no claim)
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
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
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_418m_sd016_cue_slot_tagger"
CLAIM_IDS: List[str] = []          # diagnostic: substrate readiness, weights no claim
EXPERIMENT_PURPOSE = "diagnostic"

P0_EPISODES         = 20
P1_EPISODES         = 40
STEPS_PER_EPISODE   = 150
CONTEXT_SWITCH_EVERY = 5
LAMBDA_TERRAIN      = 0.1
LAMBDA_CUE_ACTION   = 0.5
EVAL_BATCH_SIZE     = 32
LR                  = 1e-4
SEEDS               = [42, 43, 44]

# Arms: (label, cue_slot_tagger flag)
ARMS: List[Tuple[str, bool]] = [
    ("A0_OFF", False),
    ("A1_ON",  True),
]

SEL_ENTROPY_C1_THRESHOLD = 2.5    # ON arm must drop below this (off the saddle)
SEL_CONTEXT_DIV_THRESHOLD = 0.1   # ON selection must differ this much safe-vs-dangerous (L1)
SEL_ENTROPY_C2_FLOOR     = 2.65   # OFF arm must stay above this (on the saddle)
UNIFORM_REFERENCE        = None   # filled at runtime = ln(num_slots)


# ---------------------------------------------------------------------------
# Env + agent helpers (mirror V3-EXQ-418i).
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


def _make_agent(env: CausalGridWorldV2, cue_slot_tagger: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        sd016_enabled=True,
        sd016_writepath_mode="off",
        sd016_cue_slot_tagger=cue_slot_tagger,
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
# Selection / output probes.
# ---------------------------------------------------------------------------

def _selection_entropy_mean(agent, z_world_batch) -> float:
    """Mean entropy of the ACTIVE slot-selection mechanism over the batch.

    Calls extract_cue_context (which caches _last_cue_slot_weights for BOTH the
    legacy q.k branch and the Path-3 tagger branch) and reads the cached
    distribution -- so OFF and ON are measured on equal footing.
    """
    with torch.no_grad():
        agent.e1.extract_cue_context(z_world_batch)
        w = agent.e1._last_cue_slot_weights.clamp(min=1e-12)   # [batch, num_slots]
        return float(-(w * w.log()).sum(dim=-1).mean().item())


def _action_bias_per_channel_std(agent, z_world_batch) -> float:
    """Std of action_bias across the batch, mean over channels (EXQ-449 metric)."""
    with torch.no_grad():
        action_bias, _ = agent.e1.extract_cue_context(z_world_batch)
        return float(action_bias.std(dim=0).mean().item())


def _action_bias_divergence(agent, z_safe, z_dang) -> float:
    """Safe-vs-dangerous mean action_bias L2 divergence (EXQ-449b S3 metric)."""
    with torch.no_grad():
        ab_safe, _ = agent.e1.extract_cue_context(z_safe)
        ab_dang, _ = agent.e1.extract_cue_context(z_dang)
        return float((ab_safe.mean(dim=0) - ab_dang.mean(dim=0)).norm().item())


def _selection_context_divergence(agent, z_safe, z_dang) -> float:
    """L1 distance between the mean slot-selection distributions for safe vs
    dangerous contexts. Anti-degeneracy guard: a tagger that collapsed to one
    slot for ALL contexts would pass the per-sample entropy gate but score ~0
    here (selection must actually DIFFER across contexts to be useful, and
    terrain_loss can only be minimised if it does)."""
    with torch.no_grad():
        agent.e1.extract_cue_context(z_safe)
        w_safe = agent.e1._last_cue_slot_weights.mean(dim=0)   # [num_slots]
        agent.e1.extract_cue_context(z_dang)
        w_dang = agent.e1._last_cue_slot_weights.mean(dim=0)
        return float((w_safe - w_dang).abs().sum().item())


# ---------------------------------------------------------------------------
# Eval batch collector (mirrors V3-EXQ-418i).
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
# Training episode (mirrors V3-EXQ-418i).
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

def _run_one(arm_label: str, tagger: bool, seed: int,
             p0: int, p1: int, eval_n: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent    = _make_agent(env_safe, tagger)

    total_eps = p0 + p1
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    print(f"Seed {seed} Condition {arm_label}", flush=True)
    print(f"  [arm {arm_label} seed {seed}] cue_slot_tagger={tagger} "
          f"p0={p0} p1={p1}", flush=True)

    for ep in range(p0):
        env = env_dang if (ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P0")
        if (ep + 1) % 50 == 0 or (ep + 1) == p0:
            print(f"  [train] {arm_label} seed={seed} ep {ep+1}/{total_eps}", flush=True)

    for ep in range(p1):
        abs_ep = p0 + ep
        env = env_dang if (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1 else env_safe
        _run_training_episode(agent, env, optimizer, "P1")
        if (abs_ep + 1) % 50 == 0 or (abs_ep + 1) == total_eps:
            print(f"  [train] {arm_label} seed={seed} ep {abs_ep+1}/{total_eps}", flush=True)

    z_safe = _collect_eval_batch(agent, env_safe, eval_n)
    z_dang = _collect_eval_batch(agent, env_dang, eval_n)

    sel_ent = _selection_entropy_mean(agent, z_safe)
    sel_ctx = _selection_context_divergence(agent, z_safe, z_dang)
    ab_std  = _action_bias_per_channel_std(agent, z_safe)
    ab_div  = _action_bias_divergence(agent, z_safe, z_dang)

    print(f"  [arm {arm_label} seed {seed}] "
          f"sel_entropy={sel_ent:.4f} sel_ctx_div={sel_ctx:.4f} "
          f"action_bias_std={ab_std:.6f} action_bias_div={ab_div:.4f}", flush=True)
    print(f"verdict: PASS", flush=True)   # per-run completion (gating is aggregate)

    return {
        "arm":                         arm_label,
        "cue_slot_tagger":             tagger,
        "seed":                        seed,
        "sel_entropy_mean":            sel_ent,
        "sel_context_divergence":      sel_ctx,
        "action_bias_per_channel_std": ab_std,
        "action_bias_div":             ab_div,
    }


# ---------------------------------------------------------------------------
# Aggregation + acceptance.
# ---------------------------------------------------------------------------

def _summarise(arm_results: List[Dict]) -> Dict:
    n = len(arm_results)
    return {
        "n_seeds":                          n,
        "sel_entropy_mean_mean":            sum(r["sel_entropy_mean"] for r in arm_results) / n,
        "sel_entropy_mean_min":             min(r["sel_entropy_mean"] for r in arm_results),
        "sel_entropy_mean_max":             max(r["sel_entropy_mean"] for r in arm_results),
        "sel_context_divergence_mean":      sum(r["sel_context_divergence"] for r in arm_results) / n,
        "sel_context_divergence_min":       min(r["sel_context_divergence"] for r in arm_results),
        "action_bias_per_channel_std_mean": sum(r["action_bias_per_channel_std"] for r in arm_results) / n,
        "action_bias_div_mean":             sum(r["action_bias_div"] for r in arm_results) / n,
    }


def _evaluate(per_arm_results: Dict[str, List[Dict]]) -> Dict:
    n_seeds  = len(SEEDS)
    majority = (n_seeds // 2) + 1

    on   = per_arm_results.get("A1_ON", [])
    off  = per_arm_results.get("A0_OFF", [])

    # C1 PRIMARY: ON arm drops below the saddle threshold on a majority of seeds.
    c1_seeds_pass = sum(1 for r in on if r["sel_entropy_mean"] < SEL_ENTROPY_C1_THRESHOLD)
    c1_pass       = c1_seeds_pass >= majority

    # C1b ANTI-DEGENERACY: ON selection actually DIFFERS across safe vs dangerous
    # contexts (not a degenerate always-one-slot collapse that would also pass C1).
    c1b_seeds_pass = sum(1 for r in on if r["sel_context_divergence"] > SEL_CONTEXT_DIV_THRESHOLD)
    c1b_pass       = c1b_seeds_pass >= majority

    # C2 CONTROL: OFF arm stays on the uniform saddle on a majority of seeds.
    c2_seeds_pass = sum(1 for r in off if r["sel_entropy_mean"] > SEL_ENTROPY_C2_FLOOR)
    c2_pass       = c2_seeds_pass >= majority

    return {
        "C1_tagger_breaks_saddle": {
            "pass":        c1_pass,
            "seeds_pass":  c1_seeds_pass,
            "majority":    majority,
            "threshold":   SEL_ENTROPY_C1_THRESHOLD,
        },
        "C1b_selection_context_dependent": {
            "pass":        c1b_pass,
            "seeds_pass":  c1b_seeds_pass,
            "majority":    majority,
            "threshold":   SEL_CONTEXT_DIV_THRESHOLD,
        },
        "C2_off_arm_on_saddle": {
            "pass":        c2_pass,
            "seeds_pass":  c2_seeds_pass,
            "majority":    majority,
            "floor":       SEL_ENTROPY_C2_FLOOR,
        },
        "overall_pass": c1_pass and c1b_pass and c2_pass,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    import math
    global UNIFORM_REFERENCE
    UNIFORM_REFERENCE = math.log(16)   # ContextMemory default num_slots

    p0     = P0_EPISODES     if not dry_run else 2
    p1     = P1_EPISODES     if not dry_run else 3
    eval_n = EVAL_BATCH_SIZE if not dry_run else 4

    print(f"V3-EXQ-418m SD-016 Path 3 cue-slot tagger "
          f"(seeds={SEEDS} dry_run={dry_run} uniform_ref={UNIFORM_REFERENCE:.4f})",
          flush=True)

    all_results: List[Dict]            = []
    per_arm: Dict[str, List[Dict]]     = {lab: [] for lab, _ in ARMS}

    for arm_label, tagger in ARMS:
        for seed in SEEDS:
            r = _run_one(arm_label, tagger, seed, p0, p1, eval_n)
            all_results.append(r)
            per_arm[arm_label].append(r)

    summaries  = {arm: _summarise(rs) for arm, rs in per_arm.items()}
    acceptance = _evaluate(per_arm)
    outcome    = "PASS" if acceptance["overall_pass"] else "FAIL"

    print(f"  [summary] C1={acceptance['C1_tagger_breaks_saddle']['pass']} "
          f"C1b={acceptance['C1b_selection_context_dependent']['pass']} "
          f"C2={acceptance['C2_off_arm_on_saddle']['pass']} -> outcome={outcome}",
          flush=True)
    print(f"  [summary] OFF sel_entropy={summaries['A0_OFF']['sel_entropy_mean_mean']:.4f} "
          f"ON sel_entropy={summaries['A1_ON']['sel_entropy_mean_mean']:.4f} "
          f"ON sel_ctx_div={summaries['A1_ON']['sel_context_divergence_mean']:.4f} "
          f"(uniform={UNIFORM_REFERENCE:.4f})", flush=True)

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
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "SD-016 Path 3 (feedforward cue->slot tagger) substrate-readiness "
            "diagnostic; claim_ids=[] (weights no claim). 2-arm ablation OFF "
            "(legacy q.k attention -- the V3-EXQ-418i saddle) vs ON (Path 3 "
            "tagger), matched terrain_loss training. C1 PRIMARY: ON selection "
            "entropy < 2.5 on a majority of seeds = tagger breaks the uniform "
            "ln(16)=2.773 saddle Path 1 could not. C2 CONTROL: OFF stays > 2.65 "
            "(on the saddle), isolating the tagger. SECONDARY (reported, not "
            "gated): action_bias_per_channel_std + safe-vs-dangerous "
            "action_bias_div -- full action_bias_div>=0.05 propagation is gated "
            "on cue_action_proj / SD-055 differentiable CEM, so it is context "
            "not a pass gate. C1 FAIL + C2 PASS -> /failure-autopsy (tagger "
            "training-signal, not the saddle). C2 FAIL -> substrate inconsistency."
        ),
        "acceptance_checks":  acceptance,
        "uniform_reference":  UNIFORM_REFERENCE,
        "per_arm_summaries":  summaries,
        "per_seed_results":   all_results,
        "thresholds": {
            "sel_entropy_c1_threshold":  SEL_ENTROPY_C1_THRESHOLD,
            "sel_context_div_threshold": SEL_CONTEXT_DIV_THRESHOLD,
            "sel_entropy_c2_floor":      SEL_ENTROPY_C2_FLOOR,
            "uniform_reference":         UNIFORM_REFERENCE,
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
            "arms": [{"label": lab, "cue_slot_tagger": tg} for lab, tg in ARMS],
            "sd016_enabled":               True,
            "sd016_writepath_mode":        "off",
            "sd016_temperature_learnable": False,
            "sws_enabled":                 False,
            "rem_enabled":                 False,
            "shy_enabled":                 False,
            "dry_run":                     dry_run,
        },
    }

    out_path = None
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

    print(f"Outcome: {outcome}", flush=True)
    output["_manifest_path"] = out_path
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)

    _manifest_path = result.get("_manifest_path")
    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
    )
