#!/opt/local/bin/python3
"""V3-EXQ-504 -- MECH-153 supervised context labeling, 3-arm extension.

Claim: MECH-153 (e1.context_memory_supervised_training_requirement)
Status: candidate (exp_conf=0.43, 1 PASS / 3 FAIL across 11 runs)

Why this experiment exists
--------------------------
EXQ-181b PASS-on-substrate-side but FAIL on metric (cosine_sim=0.9999) made
the supervised-objective requirement clear, then EXQ-187a was inconclusive
(lambda=0.1, 150 warmup eps too weak), then EXQ-239 ran the 2-arm version
(supervised CE vs ablated, lambda=0.5, 500 warmup eps).

This experiment adds the missing third arm: a SOFT contrastive loss on
hazard-proximity rather than discrete CE on a tertiled label. The intuition
(claim text): the right inductive signal is "context vectors near a hazard
should be more similar than vectors far from it." Hard-CE labels (3-class
hazard tertile) impose discrete bins on a continuous quantity and may be
losing useful gradient on the hazard transitions where MECH-150 retrieval
is most useful. If soft contrastive does at least as well as hard CE AND
beats the ablated arm, that supports the claim's strong reading.

PASS strongly favors MECH-153. FAIL with the no-loss arm collapsing
cosine_sim ~1.0 while either supervised arm works points to the
supervised-objective requirement; FAIL with both supervised arms also
collapsing weakens MECH-153 (hazard signal alone is not enough; the
upstream world latent may be too coarse to differentiate by any read-side
loss).

Three arms (5 seeds each)
-------------------------
ARM_A (NO_LOSS):
  lambda_terrain=0.0. context_memory.write() called per step. No supervision.
  Predicts: cosine_sim near 1.0 (context vectors degenerate, EXQ-181b
  reproduction).

ARM_B (HARD_CE):
  lambda_terrain=0.5. CE loss on terrain class (OPEN/ROCKY/FOREST tertiles
  of hazard_max). Same training scaffolding as EXQ-239 SUPERVISED arm.
  Predicts: cosine_sim noticeably below ARM_A (supervised differentiation
  lands).

ARM_C (SOFT_CONTRASTIVE):
  lambda_terrain=0.5. Soft contrastive loss: pairs of cue_context vectors
  from the same tick batch are pushed together when their hazard_max
  distance is small, pushed apart when large. No discrete labels; gradient
  scales with hazard-similarity. The MECH-153 strong-reading test.
  Predicts: cosine_sim at least as low as ARM_B; harm_r2 (linear probe on
  cue_context predicting hazard_max) >= ARM_B.

PASS criteria (>= 3/5 seeds for each)
-------------------------------------
  C1: ARM_A cosine_sim >= 0.95          (no-loss collapses, baseline)
  C2: ARM_B cosine_sim < 0.80           (hard-CE differentiates)
  C3: ARM_C cosine_sim < 0.80           (soft-contrastive differentiates)
  C4: ARM_C harm_r2 >= 0.40             (cue_context encodes hazard
                                          continuously, not just bins)
  C5: ARM_C harm_r2 >= ARM_B harm_r2 - 0.05  (soft >= hard within tolerance)

PASS = C1 AND C2 AND C3 AND C4 AND C5  (all five required)
PASS supports MECH-153 strong reading.
FAIL with C1 only PASSing -> supervised objective is necessary AND
  insufficient; upstream latent capacity is the blocker.
FAIL with C1 + C2 PASSing but C3/C4 failing -> hard CE works but soft
  doesn't (claim narrows to discrete-label-required).
FAIL with C1 + C3 + C4 PASSing but C2 failing -> hard CE collapses while
  soft works (interesting; favors continuous-hazard reading over tertiles).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_504_mech153_supervised_context_labeling_3arm.py
  /opt/local/bin/python3 experiments/v3_exq_504_mech153_supervised_context_labeling_3arm.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_504_mech153_supervised_context_labeling_3arm"
CLAIM_IDS = ["MECH-153"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 7, 11, 99, 31)
WARMUP_EPISODES = 500
COLLECT_EPISODES = 100
STEPS_PER_EPISODE = 200
SELF_DIM = 32
WORLD_DIM = 32
LR = 1e-3
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
NUM_HAZARDS = 1
LAMBDA_TERRAIN = 0.5
CONTRASTIVE_TEMPERATURE = 0.5  # soft-contrastive scale on hazard distance
PROBE_TRAIN_STEPS = 50
N_TERRAIN_CLASSES = 3

# Pre-registered thresholds
C1_ARM_A_MIN_COSINE = 0.95
C2_ARM_B_MAX_COSINE = 0.80
C3_ARM_C_MAX_COSINE = 0.80
C4_ARM_C_MIN_HARM_R2 = 0.40
C5_HARM_R2_TOLERANCE = 0.05
PASS_FRACTION_REQUIRED = 3.0 / 5.0


# --- Helpers ------------------------------------------------------------
def _action_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _hazard_max(obs_dict: Dict, obs_world: Optional[torch.Tensor]) -> float:
    if "harm_obs" in obs_dict:
        h = obs_dict["harm_obs"]
        if hasattr(h, "shape") and h.shape[-1] >= 26:
            return float(h[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, "shape"):
            return float(hfv.max().item())
    if obs_world is not None and obs_world.shape[-1] >= 225:
        return float(obs_world[..., 200:225].max().item())
    return 0.0


def _hazard_to_class(h: float) -> int:
    if h > 0.70:
        return 2
    if h >= 0.33:
        return 1
    return 0


def _extract_cue_context(agent: REEAgent, z_world: torch.Tensor) -> torch.Tensor:
    """Pull cue_context vector via SD-016 world_query_proj path."""
    batch = z_world.shape[0]
    mem_dim = agent.e1.context_memory.memory_dim
    q = agent.e1.world_query_proj(z_world).unsqueeze(1)
    memory = agent.e1.context_memory.memory
    k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch, -1, -1)
    v = agent.e1.context_memory.value_proj(memory).unsqueeze(0).expand(batch, -1, -1)
    scores = torch.bmm(q, k.transpose(1, 2)) / (mem_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    ctx = torch.bmm(weights, v).squeeze(1)
    return agent.e1.context_memory.output_proj(ctx)


def _soft_contrastive_loss_against_bank(
    current_cue: torch.Tensor,        # [1, latent_dim] WITH grad
    current_haz: float,
    bank_cues: torch.Tensor,          # [N, latent_dim] DETACHED
    bank_hazs: torch.Tensor,          # [N]
    temperature: float,
) -> torch.Tensor:
    """Soft contrastive: current cue paired against a detached memory bank.

    Gradient flows only through current_cue. bank_* are stored across ticks
    as detached tensors so cross-tick in-place mutations of ContextMemory
    don't poison the autograd graph.
    target_sim = exp(- |haz_dist| / temperature) in [0, 1]
    Loss = MSE(normalised_cue_sim, target_sim) across the N pairs.
    """
    if bank_cues.shape[0] == 0:
        return torch.tensor(0.0, device=current_cue.device)
    cur_n = F.normalize(current_cue, dim=-1)         # [1, D]
    bank_n = F.normalize(bank_cues, dim=-1)          # [N, D]
    sim = (cur_n @ bank_n.t()).squeeze(0)            # [N], in [-1, 1]
    sim_norm = (sim + 1.0) / 2.0
    haz_dist = (bank_hazs - current_haz).abs()       # [N]
    target = torch.exp(-haz_dist / max(temperature, 1e-6))
    return F.mse_loss(sim_norm, target)


# --- Per-arm runner -----------------------------------------------------
def run_arm(seed: int, arm_label: str, loss_mode: str,
            warmup_episodes: int, collect_episodes: int) -> Dict:
    """One arm run.

    loss_mode in {"none", "hard_ce", "soft_contrastive"}.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=NUM_HAZARDS,
        num_resources=3,
        hazard_harm=0.5,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )
    action_dim = env.action_dim

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
        sd016_enabled=True,
    )
    agent = REEAgent(cfg)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    latent_dim = SELF_DIM + WORLD_DIM
    ce_head = nn.Linear(latent_dim, N_TERRAIN_CLASSES) if loss_mode == "hard_ce" else None
    if ce_head is not None:
        ce_opt = optim.Adam(ce_head.parameters(), lr=LR)

    # Detached memory bank for the soft-contrastive arm. The current tick's
    # cue carries gradient; bank entries are .detach()'d so cross-tick
    # in-place mutations of ContextMemory don't break autograd version
    # tracking. FIFO ring of latest BANK_CAPACITY observations.
    BANK_CAPACITY = 64
    bank_cue_list: List[torch.Tensor] = []
    bank_haz_list: List[float] = []

    # ---- Warmup training ----
    agent.train()
    if ce_head is not None:
        ce_head.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            obs_state = torch.cat(
                [latent.z_self.detach(), latent.z_world.detach()], dim=-1
            )
            agent.e1.context_memory.write(obs_state)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            aux_loss = torch.tensor(0.0)

            if loss_mode == "hard_ce":
                cue = _extract_cue_context(agent, latent.z_world)
                hmax = _hazard_max(obs_dict, obs_world)
                label = torch.tensor([_hazard_to_class(hmax)], dtype=torch.long)
                logits = ce_head(cue)
                aux_loss = F.cross_entropy(logits, label)
            elif loss_mode == "soft_contrastive":
                cue = _extract_cue_context(agent, latent.z_world)
                hmax = _hazard_max(obs_dict, obs_world)
                if bank_cue_list:
                    bank_cues = torch.stack(bank_cue_list, dim=0)
                    bank_hazs = torch.tensor(bank_haz_list, dtype=torch.float32)
                    aux_loss = _soft_contrastive_loss_against_bank(
                        cue, float(hmax), bank_cues, bank_hazs,
                        CONTRASTIVE_TEMPERATURE,
                    )
                # Update bank AFTER computing loss (so current cue isn't its own neighbour).
                bank_cue_list.append(cue.detach().squeeze(0).clone())
                bank_haz_list.append(float(hmax))
                if len(bank_cue_list) > BANK_CAPACITY:
                    bank_cue_list.pop(0)
                    bank_haz_list.pop(0)

            total = e1_loss + e2_loss + LAMBDA_TERRAIN * aux_loss
            if total.requires_grad:
                optimizer.zero_grad()
                if ce_head is not None:
                    ce_opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                if ce_head is not None:
                    torch.nn.utils.clip_grad_norm_(ce_head.parameters(), 1.0)
                optimizer.step()
                if ce_head is not None:
                    ce_opt.step()

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(f"  [{arm_label}] seed={seed} ep {ep+1}/{warmup_episodes}", flush=True)

    # ---- Collect ----
    agent.eval()
    if ce_head is not None:
        ce_head.eval()

    cue_proximate: List[torch.Tensor] = []  # FOREST
    cue_distal: List[torch.Tensor] = []     # OPEN
    cues_all: List[torch.Tensor] = []
    hazards_all: List[float] = []

    for ep in range(collect_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                cue = _extract_cue_context(agent, latent.z_world.detach())
            agent.clock.advance()

            hmax = _hazard_max(obs_dict, obs_world)
            cues_all.append(cue.squeeze(0).cpu())
            hazards_all.append(float(hmax))
            if hmax > 0.70:
                cue_proximate.append(cue.squeeze(0).cpu())
            elif hmax < 0.33:
                cue_distal.append(cue.squeeze(0).cpu())

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    n_prox = len(cue_proximate)
    n_dist = len(cue_distal)
    n_total = len(cues_all)

    cosine_sim = float("nan")
    if n_prox >= 10 and n_dist >= 10:
        a = torch.stack(cue_proximate).mean(dim=0)
        b = torch.stack(cue_distal).mean(dim=0)
        cosine_sim = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    # harm_r2: linear probe predicting continuous hazard_max from cue_context.
    harm_r2 = float("nan")
    if n_total >= 50:
        X = torch.stack(cues_all).detach().float()
        y = torch.tensor(hazards_all, dtype=torch.float32)
        n_test = max(10, n_total // 5)
        X_tr, y_tr = X[n_test:], y[n_test:]
        X_te, y_te = X[:n_test], y[:n_test]
        probe = nn.Linear(X.shape[1], 1)
        probe_opt = optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(PROBE_TRAIN_STEPS):
            probe_opt.zero_grad()
            pred = probe(X_tr).squeeze(-1)
            loss = F.mse_loss(pred, y_tr)
            loss.backward()
            probe_opt.step()
        with torch.no_grad():
            pred_te = probe(X_te).squeeze(-1)
            ss_res = float(((pred_te - y_te) ** 2).sum().item())
            ss_tot = float(((y_te - y_te.mean()) ** 2).sum().item())
            if ss_tot > 1e-8:
                harm_r2 = 1.0 - ss_res / ss_tot

    print(
        f"  [{arm_label}] seed={seed} cosine_sim={cosine_sim:.4f} "
        f"harm_r2={harm_r2:.4f}  n_prox={n_prox} n_dist={n_dist} n_total={n_total}",
        flush=True,
    )
    return {
        "seed": seed,
        "arm_label": arm_label,
        "loss_mode": loss_mode,
        "cosine_sim": None if cosine_sim != cosine_sim else cosine_sim,
        "harm_r2": None if harm_r2 != harm_r2 else harm_r2,
        "n_proximate": n_prox,
        "n_distal": n_dist,
        "n_total": n_total,
    }


# --- Aggregation --------------------------------------------------------
def _evaluate(arm_a: List[Dict], arm_b: List[Dict], arm_c: List[Dict]) -> Dict:
    n = len(arm_a)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)

    def _cs(rows): return [r["cosine_sim"] for r in rows if r["cosine_sim"] is not None]
    def _hr(rows): return [r["harm_r2"] for r in rows if r["harm_r2"] is not None]

    a_cs, b_cs, c_cs = _cs(arm_a), _cs(arm_b), _cs(arm_c)
    b_hr, c_hr = _hr(arm_b), _hr(arm_c)

    c1 = sum(1 for x in a_cs if x >= C1_ARM_A_MIN_COSINE)
    c2 = sum(1 for x in b_cs if x < C2_ARM_B_MAX_COSINE)
    c3 = sum(1 for x in c_cs if x < C3_ARM_C_MAX_COSINE)
    c4 = sum(1 for x in c_hr if x >= C4_ARM_C_MIN_HARM_R2)

    paired = list(zip(arm_b, arm_c))
    c5 = 0
    for rb, rc in paired:
        if rb["harm_r2"] is None or rc["harm_r2"] is None:
            continue
        if rc["harm_r2"] >= rb["harm_r2"] - C5_HARM_R2_TOLERANCE:
            c5 += 1

    return {
        "n_seeds": n,
        "min_seeds_required": required,
        "c1_arm_a_collapses_seeds_pass": c1,
        "c2_arm_b_differentiates_seeds_pass": c2,
        "c3_arm_c_differentiates_seeds_pass": c3,
        "c4_arm_c_harm_r2_seeds_pass": c4,
        "c5_arm_c_ge_arm_b_seeds_pass": c5,
        "c1_pass": c1 >= required,
        "c2_pass": c2 >= required,
        "c3_pass": c3 >= required,
        "c4_pass": c4 >= required,
        "c5_pass": c5 >= required,
        "overall_pass": (
            c1 >= required and c2 >= required and c3 >= required
            and c4 >= required and c5 >= required
        ),
        "mean_arm_a_cosine": float(sum(a_cs) / len(a_cs)) if a_cs else None,
        "mean_arm_b_cosine": float(sum(b_cs) / len(b_cs)) if b_cs else None,
        "mean_arm_c_cosine": float(sum(c_cs) / len(c_cs)) if c_cs else None,
        "mean_arm_b_harm_r2": float(sum(b_hr) / len(b_hr)) if b_hr else None,
        "mean_arm_c_harm_r2": float(sum(c_hr) / len(c_hr)) if c_hr else None,
    }


# --- Driver -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--warmup", type=int, default=WARMUP_EPISODES)
    parser.add_argument("--collect", type=int, default=COLLECT_EPISODES)
    args = parser.parse_args()

    if args.dry_run:
        seeds = (args.seeds[0],)
        warmup = 2
        collect = 2
        print("[DRY-RUN] 1 seed, 2 warmup eps, 2 collect eps -- smoke only.", flush=True)
    else:
        seeds = tuple(args.seeds)
        warmup = args.warmup
        collect = args.collect

    t0 = time.time()
    arm_a_results = [run_arm(s, "ARM_A_no_loss", "none", warmup, collect) for s in seeds]
    arm_b_results = [run_arm(s, "ARM_B_hard_ce", "hard_ce", warmup, collect) for s in seeds]
    arm_c_results = [run_arm(s, "ARM_C_soft_contrastive", "soft_contrastive", warmup, collect) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate(arm_a_results, arm_b_results, arm_c_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-504 (MECH-153) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    for name, val in (
        ("C1 ARM_A_no_loss collapses (cosine>=0.95)", criteria["c1_pass"]),
        ("C2 ARM_B_hard_ce differentiates (cosine<0.80)", criteria["c2_pass"]),
        ("C3 ARM_C_soft_contrastive differentiates (cosine<0.80)", criteria["c3_pass"]),
        ("C4 ARM_C harm_r2 >= 0.40", criteria["c4_pass"]),
        ("C5 ARM_C harm_r2 >= ARM_B harm_r2 - 0.05", criteria["c5_pass"]),
    ):
        print(f"  {name}: {'PASS' if val else 'FAIL'}", flush=True)

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-153": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_ARM_A_MIN_COSINE": C1_ARM_A_MIN_COSINE,
            "C2_ARM_B_MAX_COSINE": C2_ARM_B_MAX_COSINE,
            "C3_ARM_C_MAX_COSINE": C3_ARM_C_MAX_COSINE,
            "C4_ARM_C_MIN_HARM_R2": C4_ARM_C_MIN_HARM_R2,
            "C5_HARM_R2_TOLERANCE": C5_HARM_R2_TOLERANCE,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "config": {
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "warmup_episodes": warmup,
            "collect_episodes": collect,
            "steps_per_episode": STEPS_PER_EPISODE,
            "lambda_terrain": LAMBDA_TERRAIN,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
            "num_hazards": NUM_HAZARDS,
            "seeds": list(seeds),
        },
        "results_arm_a_no_loss": arm_a_results,
        "results_arm_b_hard_ce": arm_b_results,
        "results_arm_c_soft_contrastive": arm_c_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "Three-arm extension of EXQ-239 (which ran the no-loss vs hard-CE pair). "
            "Adds a soft contrastive arm whose pair similarity targets exp(-|haz_dist|/T), "
            "providing continuous-hazard supervision without discrete tertile binning. "
            "PASS supports MECH-153 strong reading. FAIL with C1 only PASSing -> upstream "
            "z_world capacity is the blocker. FAIL with C1+C2 PASSing but C3/C4 failing -> "
            "claim narrows to discrete-label-required."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
