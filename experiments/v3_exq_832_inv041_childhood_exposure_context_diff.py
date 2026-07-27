"""
V3-EXQ-832: INV-041 -- childhood exposure regime is necessary for E1
ContextMemory differentiation (commitment-free, exposure-only diagnostic).

Scientific question (INV-041): Is the childhood phase's committed + constrained
exposure regime -- specifically, FORCED engagement with hazard-proximate contexts
rather than free-affordance avoidance of them -- a necessary architectural
prerequisite for E1 ContextMemory to develop differentiated hazard-proximate vs
hazard-distal context representations? INV-041's distinctive assertion (beyond
MECH-153's "supervised labeling is required") is that AVOIDANCE ITSELF, by
under-exposing the agent to hazard-proximate contexts, prevents ContextMemory
from receiving the hazard-context signal -- so an avoidance-shaped training
distribution fails to differentiate EVEN WITH the supervised labeling objective on.

Why this is commitment-free / substrate-ready: the childhood regime's defining
constraint is IMPOSED here (the training exposure distribution is set directly at
the minibatch-sampling level), not elicited from an emergent multi-step commitment
layer. The F-dominant / conversion-ceiling avoidance the substrate already exhibits
is the MOTIVATING phenomenon, not a confound: it is exactly the avoidance-shaped
exposure that ARM_ADULT simulates. No basal-ganglia commitment substrate is needed,
so a FAIL here is a real verdict on INV-041, not a re-derivation of the F-dominance
ceiling.

Design (3 arms x 5 seeds). Each (arm, seed) cell is an independent, RNG-reset run
so its P0 encoder warmup + context-pool collection are bit-identical across arms at
a fixed seed; the arms diverge ONLY in the P1 ContextMemory-training exposure
distribution and objective:
  ARM_CHILDHOOD       -- balanced/forced exposure (A_FRAC=0.50), supervised
                         context-labeling ON. The childhood regime.
  ARM_ADULT           -- avoidance-shaped exposure (A_FRAC=0.05, hazard-proximate
                         contexts under-sampled as a free-affordance avoider would),
                         supervised context-labeling ON. The no-childhood regime.
  ARM_CHILDHOOD_UNSUP -- balanced exposure (A_FRAC=0.50) but UNSUPERVISED (next-
                         z_world prediction, no context labels). MECH-153 contrast:
                         experience without labels.

Phased training (mandatory): P0 = encoder warmup (E1 prediction + E2 world-forward);
P1 = FROZEN encoder, train context_memory + prior_generator (+ label head when
supervised) on the DETACHED pooled latents at the arm's exposure ratio; P2 = eval.

DV: cosine_sim(mean_prior_A, mean_prior_B) computed on a HELD-OUT BALANCED probe
pool (identical across arms). LOWER cosine = better differentiation. Same metric
EXQ-181b found inadequate in the untrained system and that INV-041's notes name as
the childhood-exit criterion.

Pre-registered criteria:
  C1 (INV-041, LOAD-BEARING): mean_cos(ADULT) - mean_cos(CHILDHOOD) >= 0.10 AND
     mean_cos(CHILDHOOD) < 0.85. Childhood differentiates; avoidance does not,
     despite identical labeling -> exposure regime is necessary.
  C2 (MECH-153, secondary): mean_cos(CHILDHOOD_UNSUP) - mean_cos(CHILDHOOD) >= 0.10.
     Labeling helps beyond balanced exposure alone.
Overall PASS iff C1 (INV-041 is the target claim).

evidence_direction_per_claim:
  INV-041   -- supports if C1 else weakens.
  MECH-153  -- supports if C2 else weakens.

Claim IDs: INV-041 (primary), MECH-153 (secondary).
"""

import sys
import copy
import argparse
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.manifest_core import stamp_recording_core

EXPERIMENT_TYPE = "v3_exq_832_inv041_childhood_exposure_context_diff"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["INV-041", "MECH-153"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---- Arms: (label, a_frac, supervised) ----
ARMS = [
    ("ARM_CHILDHOOD", 0.50, True),
    ("ARM_ADULT", 0.05, True),
    ("ARM_CHILDHOOD_UNSUP", 0.50, False),
]

SEEDS = [0, 1, 2, 3, 4]

# ---- Pre-registered thresholds (constants, not derived from run stats) ----
C1_DELTA_MIN = 0.10        # ADULT cosine must exceed CHILDHOOD cosine by >= this
CHILDHOOD_COS_MAX = 0.85   # CHILDHOOD must differentiate below this (EXQ-181b C1)
C2_DELTA_MIN = 0.10        # UNSUP cosine must exceed CHILDHOOD cosine by >= this

# ---- Context label thresholds (EXQ-181b revised, use_proxy_fields=True) ----
CTX_A_THRESH = 0.7         # hazard-proximate: hazard_field_view.max() > 0.7
CTX_B_THRESH = 0.33        # hazard-distal:    hazard_field_view.max() < 0.33
MIN_PROBE_PER_CLASS = 10   # eval SKIP-guard per class

# ---- Run sizes ----
WARMUP_EPISODES = 60
COLLECT_EPISODES = 40
PROBE_EPISODES = 20
STEPS_PER_EPISODE = 60
P1_STEPS = 300
BATCH = 64
LR = 1e-3
ALPHA_WORLD = 0.9          # z_world fidelity needed (SD-008); default 0.3 too low
ALPHA_SELF = 0.1
SELF_DIM = 32
WORLD_DIM = 64
NUM_HAZARDS = 1            # single hazard -> spatial gradient (EXQ-181b)


# ------------------------------------------------------------------ #
# Helpers (env/obs API identical to EXQ-181b)                          #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def get_hazard_max(obs_dict: Dict, world_obs: Optional[torch.Tensor]) -> float:
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


def get_harm_scalar(obs_dict: Dict) -> float:
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, "shape") and harm_obs.shape[-1] > 50:
            return float(harm_obs[..., 50].mean().item())
    return 0.0


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
    )
    return REEAgent(config)


# ------------------------------------------------------------------ #
# P0: encoder warmup                                                   #
# ------------------------------------------------------------------ #

def _warmup_encoder(agent: REEAgent, env: CausalGridWorldV2, seed: int,
                    warmup_episodes: int, steps_per_episode: int,
                    arm_label: str) -> Tuple[float, float]:
    """Train E1 prediction + E2 world-forward so z_world is meaningful. Random policy."""
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    action_dim = env.action_dim
    last_e1 = last_e2 = 0.0

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _harm, done, _info, obs_dict = env.step(action_oh)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            last_e1 = float(e1_loss.detach())
            last_e2 = float(e2_loss.detach())
            if done:
                break

        if (ep + 1) % 20 == 0 or ep == warmup_episodes - 1:
            print(f"  [train] {arm_label} seed={seed} ep {ep+1}/{warmup_episodes}",
                  flush=True)
    return last_e1, last_e2


# ------------------------------------------------------------------ #
# Collect context pool (total_state, context_label, harm, next_zworld) #
# ------------------------------------------------------------------ #

def _collect_pool(agent: REEAgent, env: CausalGridWorldV2, seed: int,
                  n_episodes: int, steps_per_episode: int
                  ) -> Tuple[List[Dict], List[Dict]]:
    """Random-policy rollout under the FROZEN encoder. Returns (pool_A, pool_B).

    Each entry: {"state": [total_dim] detached, "next_zworld": [world_dim] detached,
    "harm": float}. Pairs total_state_t with z_world_{t+1} for the unsupervised head.
    """
    agent.eval()
    action_dim = env.action_dim
    pool_A: List[Dict] = []
    pool_B: List[Dict] = []

    for _ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_state: Optional[torch.Tensor] = None
        prev_label: Optional[str] = None
        prev_harm: float = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            z_self = latent.z_self.detach()
            z_world = latent.z_world.detach()
            total_state = torch.cat([z_self, z_world], dim=-1).squeeze(0)  # [total_dim]

            hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
            harm_val = get_harm_scalar(obs_dict)
            label = "A" if hazard_max > CTX_A_THRESH else ("B" if hazard_max < CTX_B_THRESH else None)

            # Complete the previous transition with THIS step's z_world as next_zworld.
            if prev_state is not None and prev_label is not None:
                entry = {"state": prev_state, "next_zworld": z_world.squeeze(0), "harm": prev_harm}
                (pool_A if prev_label == "A" else pool_B).append(entry)

            prev_state = total_state
            prev_label = label
            prev_harm = harm_val

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _harm, done, _info, obs_dict = env.step(action_oh)
            if done:
                break

    return pool_A, pool_B


# ------------------------------------------------------------------ #
# P1: train ContextMemory under the arm's exposure distribution        #
# ------------------------------------------------------------------ #

def _train_context_memory(agent: REEAgent, pool_A: List[Dict], pool_B: List[Dict],
                          a_frac: float, supervised: bool, seed: int,
                          arm_label: str) -> Dict:
    """P1: frozen encoder; train context_memory + prior_generator (+ label head)."""
    device = agent.device
    label_head = nn.Linear(WORLD_DIM, 2).to(device)

    train_params = (list(agent.e1.context_memory.parameters())
                    + list(agent.e1.prior_generator.parameters()))
    if supervised:
        train_params += list(label_head.parameters())
    optimizer = optim.Adam(train_params, lr=LR)

    rng = random.Random(seed + 9973)
    n_a_per_batch = max(0, int(round(BATCH * a_frac)))
    n_b_per_batch = BATCH - n_a_per_batch
    a_exposure = 0
    b_exposure = 0

    if not pool_A or not pool_B:
        return {"trained": False, "a_exposure": 0, "b_exposure": 0,
                "final_loss": float("nan")}

    agent.train()
    last_loss = float("nan")
    for step in range(P1_STEPS):
        idx_a = [rng.randrange(len(pool_A)) for _ in range(n_a_per_batch)]
        idx_b = [rng.randrange(len(pool_B)) for _ in range(n_b_per_batch)]
        batch = [pool_A[i] for i in idx_a] + [pool_B[i] for i in idx_b]
        labels_list = [1] * n_a_per_batch + [0] * n_b_per_batch
        a_exposure += n_a_per_batch
        b_exposure += n_b_per_batch

        states = torch.stack([e["state"] for e in batch], dim=0).to(device)  # [B, total_dim]
        prior = agent.e1.generate_prior(states)  # [B, world_dim]

        if supervised:
            logits = label_head(prior)
            targets = torch.tensor(labels_list, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)
        else:
            next_zw = torch.stack([e["next_zworld"] for e in batch], dim=0).to(device)
            loss = F.mse_loss(prior, next_zw)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        last_loss = float(loss.detach())

        if (step + 1) % 100 == 0 or step == P1_STEPS - 1:
            print(f"  [p1] {arm_label} seed={seed} step {step+1}/{P1_STEPS} "
                  f"loss={last_loss:.4f}", flush=True)

    return {"trained": True, "a_exposure": a_exposure, "b_exposure": b_exposure,
            "final_loss": last_loss}


# ------------------------------------------------------------------ #
# P2: eval cosine_sim on a HELD-OUT BALANCED probe                     #
# ------------------------------------------------------------------ #

def _eval_cosine(agent: REEAgent, probe_A: List[Dict], probe_B: List[Dict]
                 ) -> Dict:
    agent.eval()
    device = agent.device
    n_A = len(probe_A)
    n_B = len(probe_B)
    if n_A < MIN_PROBE_PER_CLASS or n_B < MIN_PROBE_PER_CLASS:
        return {"cosine_sim": float("nan"), "n_probe_A": n_A, "n_probe_B": n_B,
                "harm_r2": 0.0}

    with torch.no_grad():
        sa = torch.stack([e["state"] for e in probe_A], dim=0).to(device)
        sb = torch.stack([e["state"] for e in probe_B], dim=0).to(device)
        prior_A = agent.e1.generate_prior(sa)  # [n_A, world_dim]
        prior_B = agent.e1.generate_prior(sb)  # [n_B, world_dim]
        mean_A = prior_A.mean(dim=0)
        mean_B = prior_B.mean(dim=0)
        cos = float(F.cosine_similarity(mean_A.unsqueeze(0), mean_B.unsqueeze(0)).item())

    return {"cosine_sim": cos, "n_probe_A": n_A, "n_probe_B": n_B}


# ------------------------------------------------------------------ #
# One (arm, seed) cell                                                 #
# ------------------------------------------------------------------ #

def _arm_config_slice(arm_label: str, a_frac: float, supervised: bool) -> Dict:
    return {
        "arm_label": arm_label,
        "a_frac": a_frac,
        "supervised": supervised,
        "warmup_episodes": WARMUP_EPISODES,
        "collect_episodes": COLLECT_EPISODES,
        "probe_episodes": PROBE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "p1_steps": P1_STEPS,
        "batch": BATCH,
        "lr": LR,
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "num_hazards": NUM_HAZARDS,
        "ctx_a_thresh": CTX_A_THRESH,
        "ctx_b_thresh": CTX_B_THRESH,
    }


def _run_cell(arm_label: str, a_frac: float, supervised: bool, seed: int,
              warmup_episodes: int, collect_episodes: int, probe_episodes: int,
              steps_per_episode: int) -> Dict:
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    slice_ = _arm_config_slice(arm_label, a_frac, supervised)

    with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env = _make_env(seed)
        agent = _make_agent(env)

        # Snapshot the UNTRAINED context path so P1 starts from an undifferentiated
        # ContextMemory. The childhood/adult training phase must be the SOLE driver
        # of differentiation, not incidental warmup exposure -- otherwise a warmup
        # that already differentiates would confound the exposure contrast.
        init_ctx = copy.deepcopy(agent.e1.context_memory.state_dict())
        init_prior = copy.deepcopy(agent.e1.prior_generator.state_dict())

        last_e1, last_e2 = _warmup_encoder(agent, env, seed, warmup_episodes,
                                           steps_per_episode, arm_label)

        # Frozen encoder from here: pooled latents are collected under no_grad and
        # stored detached, so P1 gradients touch only context_memory + prior_generator.
        # Pool collection uses the encoder only (agent.sense), never the context path,
        # so resetting that path afterward does not affect the pool.
        pool_A, pool_B = _collect_pool(agent, env, seed, collect_episodes,
                                       steps_per_episode)
        probe_A, probe_B = _collect_pool(agent, env, seed, probe_episodes,
                                         steps_per_episode)

        # Reset the context path to its untrained init (encoder stays warmed).
        agent.e1.context_memory.load_state_dict(init_ctx)
        agent.e1.prior_generator.load_state_dict(init_prior)

        train_stats = _train_context_memory(agent, pool_A, pool_B, a_frac,
                                             supervised, seed, arm_label)
        eval_stats = _eval_cosine(agent, probe_A, probe_B)

        row = {
            "arm_label": arm_label,
            "seed": seed,
            "a_frac": a_frac,
            "supervised": supervised,
            "cosine_sim": eval_stats["cosine_sim"],
            "n_pool_A": len(pool_A),
            "n_pool_B": len(pool_B),
            "n_probe_A": eval_stats["n_probe_A"],
            "n_probe_B": eval_stats["n_probe_B"],
            "a_exposure": train_stats["a_exposure"],
            "b_exposure": train_stats["b_exposure"],
            "p1_final_loss": train_stats["final_loss"],
            "warmup_e1_loss": last_e1,
            "warmup_e2_loss": last_e2,
        }
        cell.stamp(row)

    cos = row["cosine_sim"]
    cos_s = f"{cos:.4f}" if cos == cos else "nan"
    print(f"  [cell] {arm_label} seed={seed} cosine_sim={cos_s} "
          f"a_exposure={row['a_exposure']} b_exposure={row['b_exposure']} "
          f"n_pool_A={row['n_pool_A']} n_pool_B={row['n_pool_B']}", flush=True)
    print(f"verdict: {'PASS' if (cos == cos) else 'FAIL'}", flush=True)
    return row


# ------------------------------------------------------------------ #
# Aggregate + criteria                                                 #
# ------------------------------------------------------------------ #

def _mean_cos(rows: List[Dict], arm_label: str) -> float:
    vals = [r["cosine_sim"] for r in rows
            if r["arm_label"] == arm_label and r["cosine_sim"] == r["cosine_sim"]]
    return statistics.fmean(vals) if vals else float("nan")


def run_experiment(warmup_episodes: int = WARMUP_EPISODES,
                   collect_episodes: int = COLLECT_EPISODES,
                   probe_episodes: int = PROBE_EPISODES,
                   steps_per_episode: int = STEPS_PER_EPISODE,
                   seeds: Optional[List[int]] = None) -> Dict:
    seeds = seeds if seeds is not None else SEEDS
    t0 = time.perf_counter()

    arm_results: List[Dict] = []
    for arm_label, a_frac, supervised in ARMS:
        for seed in seeds:
            arm_results.append(
                _run_cell(arm_label, a_frac, supervised, seed,
                          warmup_episodes, collect_episodes, probe_episodes,
                          steps_per_episode)
            )

    mean_child = _mean_cos(arm_results, "ARM_CHILDHOOD")
    mean_adult = _mean_cos(arm_results, "ARM_ADULT")
    mean_unsup = _mean_cos(arm_results, "ARM_CHILDHOOD_UNSUP")

    c1_delta = (mean_adult - mean_child) if (mean_adult == mean_adult and mean_child == mean_child) else float("nan")
    c2_delta = (mean_unsup - mean_child) if (mean_unsup == mean_unsup and mean_child == mean_child) else float("nan")

    c1_pass = bool(c1_delta == c1_delta and c1_delta >= C1_DELTA_MIN
                   and mean_child == mean_child and mean_child < CHILDHOOD_COS_MAX)
    c2_pass = bool(c2_delta == c2_delta and c2_delta >= C2_DELTA_MIN)

    overall_pass = c1_pass  # C1 is the load-bearing INV-041 criterion

    # Non-degeneracy: cosine must be computable for the two INV-041 arms, and their
    # priors must not be bit-identical (which would pin the delta at 0 vacuously).
    have_core = (mean_child == mean_child) and (mean_adult == mean_adult)
    non_degenerate = bool(have_core)
    degeneracy_reason = ""
    if not have_core:
        degeneracy_reason = ("cosine_sim uncomputable for a load-bearing arm "
                             "(probe class coverage below MIN_PROBE_PER_CLASS)")

    inv041_dir = "supports" if c1_pass else "weakens"
    mech153_dir = "supports" if c2_pass else "weakens"

    manifest = {
        "run_id": None,  # filled in __main__ with timestamp
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": "supports" if c1_pass else "weakens",
        "evidence_direction_per_claim": {
            "INV-041": inv041_dir,
            "MECH-153": mech153_dir,
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria": [
            {"name": "C1_childhood_exposure_necessary", "load_bearing": True,
             "passed": c1_pass},
            {"name": "C2_labeling_beyond_exposure", "load_bearing": False,
             "passed": c2_pass},
        ],
        "metrics": {
            "mean_cos_childhood": mean_child,
            "mean_cos_adult": mean_adult,
            "mean_cos_childhood_unsup": mean_unsup,
            "c1_delta_adult_minus_childhood": c1_delta,
            "c2_delta_unsup_minus_childhood": c2_delta,
            "c1_pass": c1_pass,
            "c2_pass": c2_pass,
        },
        "thresholds": {
            "C1_DELTA_MIN": C1_DELTA_MIN,
            "CHILDHOOD_COS_MAX": CHILDHOOD_COS_MAX,
            "C2_DELTA_MIN": C2_DELTA_MIN,
        },
        "arm_results": arm_results,
        "per_seed_rows": arm_results,
        "arms": [{"label": a, "a_frac": f, "supervised": s} for a, f, s in ARMS],
        "interpretation": (
            "C1 supports INV-041: the childhood (balanced/forced) exposure regime "
            "differentiates ContextMemory while the avoidance-shaped regime does not, "
            "despite identical supervised labeling -- so committed exposure is a "
            "necessary prerequisite, not merely the labeling objective (MECH-153). "
            "C1 fail weakens INV-041: exposure regime does not gate differentiation "
            "on this substrate."
        ),
    }

    full_config = _arm_config_slice("(multi-arm)", -1.0, True)
    full_config["arms"] = [{"label": a, "a_frac": f, "supervised": s} for a, f, s in ARMS]
    return manifest, full_config, t0


# ------------------------------------------------------------------ #
# __main__                                                             #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from datetime import datetime
    from experiments.pack_writer import write_flat_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Fast smoke: tiny sizes, 1 seed.")
    args = parser.parse_args()

    seeds_used = [0] if args.dry_run else SEEDS
    if args.dry_run:
        # ~17% of steps are hazard-proximate (A); use enough episodes that both
        # classes clear MIN_PROBE_PER_CLASS so the smoke exercises the cosine path.
        manifest, full_config, started_at = run_experiment(
            warmup_episodes=5, collect_episodes=15, probe_episodes=15,
            steps_per_episode=40, seeds=seeds_used)
    else:
        manifest, full_config, started_at = run_experiment(seeds=seeds_used)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id
    manifest["timestamp_utc"] = ts

    # Single sanctioned writer: stamps the always-core (recording_schema /
    # substrate_hash hoisted from arm_results / machine / machine_class /
    # elapsed_seconds / config / seeds) and enforces the run_id/_v3 invariants.
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=started_at,
    )

    print(f"[manifest] wrote {out_path}", flush=True)
    print(f"[result] outcome={manifest['outcome']} "
          f"mean_cos_childhood={manifest['metrics']['mean_cos_childhood']} "
          f"mean_cos_adult={manifest['metrics']['mean_cos_adult']} "
          f"mean_cos_unsup={manifest['metrics']['mean_cos_childhood_unsup']} "
          f"c1_delta={manifest['metrics']['c1_delta_adult_minus_childhood']} "
          f"c2_delta={manifest['metrics']['c2_delta_unsup_minus_childhood']}",
          flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=run_id,
        dry_run=args.dry_run,
    )
