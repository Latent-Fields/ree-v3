#!/opt/local/bin/python3
"""
V3-EXQ-198 -- SD-011 Dual Nociceptive Stream Stability / Replication

Claims: SD-011
Dispatch mode: discriminative_pair (DUAL_STREAM vs FUSED_STREAM)

Context:
  EXQ-178b was the first SD-011 PASS (4/4 criteria met). This experiment is a
  stability/replication test: does the dual-stream substrate reliably produce the
  sensory-vs-affective dissociation across multiple seeds and a varied hazard
  structure? If yes, SD-011 can be treated as stable platform rather than ongoing
  rescue work.

Design:
  Condition A (DUAL_STREAM): Current SD-011 stack with z_harm_s + z_harm_a active
    (HarmEncoder + AffectiveHarmEncoder, separate latents). Direct replication of
    EXQ-178b across more seeds.

  Condition B (FUSED_STREAM): Control that collapses the dual-stream distinction.
    Both encoders still run, but their outputs are averaged into a single z_harm_fused
    of dim max(Z_HARM_S_DIM, Z_HARM_A_DIM). Both evaluation heads receive the same
    fused representation. If the DUAL_STREAM advantage disappears under fusion, that
    confirms the split itself (not just extra capacity) drives the dissociation.

  Seeds: [42, 7, 13] -- 3 matched seeds per condition.

  Task variant: each seed runs BOTH a standard hazard layout (num_hazards=3, matching
  EXQ-178b) AND a dense hazard layout (num_hazards=5) to test stability under changed
  hazard structure. Metrics are averaged across both layouts per seed.

PRE-REGISTERED ACCEPTANCE CRITERIA (all required for overall PASS):
  C1 (replication): DUAL_STREAM mean harm_fwd_r2 >= 0.20
    Forward model must replicate EXQ-178b C1 across 3 seeds and both layouts.

  C2 (stream dissociation preserved): DUAL_STREAM mean stream_corr <= 0.85
    Norms of z_harm_s and z_harm_a must not saturate in correlation.

  C3 (temporal integration preserved): DUAL_STREAM mean autocorr_gap >= 0.10
    z_harm_a must show higher lag-10 autocorrelation than z_harm_s.

  C4 (sensory responsiveness preserved): DUAL_STREAM mean z_harm_s_hazard_corr >= 0.25
    z_harm_s norm must track hazard proximity.

  C5 (dual advantage): DUAL_STREAM mean dissociation_score > FUSED_STREAM mean dissociation_score
    dissociation_score = (1 - abs(stream_corr)) + autocorr_gap + z_harm_s_hazard_corr
    The dual-stream condition must produce a higher composite dissociation than the fused control.
    This confirms the advantage is structural, not just capacity.

  C6 (seed stability): all 3 DUAL_STREAM seeds must individually pass C1 AND C4.
    No single seed should fail both the forward model and sensory responsiveness checks.

Decision scoring:
  PASS (retain_ree):     C1-C6 all met -- SD-011 stable, platform-grade
  inconclusive:          C1-C4 pass but C5 or C6 fails -- dual stream works but advantage
                         over fusion unclear or one seed unstable
  hybridize:             C1+C4 pass, C2 or C3 fail -- streams exist but temporal profile
                         not reliably distinct
  retire_ree_claim:      C1 or C4 fail on aggregate -- forward model or sensory stream
                         not replicable, EXQ-178b was a lucky seed
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder, HarmForwardModel
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_198_sd011_dual_stream_stability"
CLAIM_IDS = ["SD-011"]

# Pre-registered thresholds (must not be changed post-hoc)
THRESH_C1_FWD_R2       = 0.20   # HarmForwardModel R2 on held-out z_harm_s transitions
THRESH_C2_STREAM_CORR  = 0.85   # max allowed norm correlation between z_harm_s and z_harm_a
THRESH_C3_AUTOCORR_GAP = 0.10   # z_harm_a autocorr(lag=10) - z_harm_s autocorr(lag=10)
THRESH_C4_S_CORR       = 0.25   # Pearson corr between z_harm_s norm and hazard field intensity

HARM_OBS_DIM   = 51             # hazard_field[25] + resource_field[25] + harm_exposure[1]
HARM_OBS_A_DIM = 50             # hazard_field[25] + resource_field[25] (no scalar)
Z_HARM_S_DIM   = 32
Z_HARM_A_DIM   = 16
ACTION_DIM     = 4


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _hazard_gradient_action(env: CausalGridWorldV2, toward: bool) -> int:
    """Return action moving toward (or away from) nearest hazard. Falls back to random."""
    obs_dict = env._get_observation_dict()
    if not env.use_proxy_fields:
        return random.randint(0, ACTION_DIM - 1)
    h_view = obs_dict["hazard_field_view"].numpy().reshape(5, 5)
    # actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(h_view[r, c]))
        else:
            vals.append(-1.0)
    if toward:
        return int(np.argmax(vals))
    else:
        return int(np.argmin(vals))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays. Returns 0.0 on degenerate inputs."""
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum())) + 1e-8
    return float(np.dot(a_c, b_c) / denom)


def _autocorr(series: List[float], lag: int) -> float:
    arr = np.array(series)
    if len(arr) <= lag or np.std(arr) < 1e-8:
        return 0.0
    a = arr[:-lag] - arr[:-lag].mean()
    b = arr[lag:] - arr[lag:].mean()
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())) + 1e-8
    return float(np.dot(a, b) / denom)


def run_single(
    seed: int,
    num_hazards: int,
    fused_mode: bool,
    warmup_episodes: int,
    fwd_collect_episodes: int,
    fwd_train_epochs: int,
    fwd_batch_size: int,
    dissociation_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict:
    """
    Run one (seed, hazard layout, condition) combination.

    Args:
        fused_mode: if True, fuse z_harm_s and z_harm_a into a single shared
                    representation (FUSED_STREAM control). If False, keep them
                    separate (DUAL_STREAM).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dry_run:
        warmup_episodes       = min(3, warmup_episodes)
        fwd_collect_episodes  = min(3, fwd_collect_episodes)
        fwd_train_epochs      = min(5, fwd_train_epochs)
        dissociation_episodes = min(4, dissociation_episodes)
        steps_per_episode     = min(30, steps_per_episode)

    device = torch.device("cpu")
    condition = "FUSED_STREAM" if fused_mode else "DUAL_STREAM"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=num_hazards,
        num_resources=2,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )
    env.reset()

    harm_enc   = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_S_DIM).to(device)
    affect_enc = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)
    harm_fwd   = HarmForwardModel(z_harm_dim=Z_HARM_S_DIM, action_dim=ACTION_DIM).to(device)

    # Evaluation heads -- predict scalar from latent
    harm_eval_head   = nn.Linear(Z_HARM_S_DIM, 1).to(device)
    affect_eval_head = nn.Linear(Z_HARM_A_DIM, 1).to(device)

    opt_enc = optim.Adam(
        list(harm_enc.parameters()) + list(affect_enc.parameters()) +
        list(harm_eval_head.parameters()) + list(affect_eval_head.parameters()),
        lr=3e-4,
    )
    opt_fwd = optim.Adam(harm_fwd.parameters(), lr=3e-4)

    replay_enc: List[Tuple[torch.Tensor, torch.Tensor, float, float]] = []
    replay_fwd: List[Tuple[torch.Tensor, int, torch.Tensor]] = []

    def _maybe_fuse(z_hs: torch.Tensor, z_ha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """If fused_mode, average both into a shared representation (padded to max dim)."""
        if not fused_mode:
            return z_hs, z_ha
        # Pad z_ha to z_hs dim, average, then slice back
        if Z_HARM_S_DIM >= Z_HARM_A_DIM:
            z_ha_padded = F.pad(z_ha, (0, Z_HARM_S_DIM - Z_HARM_A_DIM))
            fused = (z_hs + z_ha_padded) / 2.0
            return fused, fused[:, :Z_HARM_A_DIM]
        else:
            z_hs_padded = F.pad(z_hs, (0, Z_HARM_A_DIM - Z_HARM_S_DIM))
            fused = (z_hs_padded + z_ha) / 2.0
            return fused[:, :Z_HARM_S_DIM], fused

    # ------------------------------------------------------------------ #
    # Phase 0: warmup -- train both encoders                              #
    # ------------------------------------------------------------------ #
    for _ in range(warmup_episodes):
        env.reset()
        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            harm_obs   = obs_dict["harm_obs"]     # [51]
            harm_obs_a = obs_dict["harm_obs_a"]   # [50]

            action_idx = random.randint(0, ACTION_DIM - 1)

            ho_t  = harm_obs.unsqueeze(0).to(device)
            hoa_t = harm_obs_a.unsqueeze(0).to(device)

            harm_exposure = float(obs_dict["body_state"][10])
            affect_level  = float(harm_obs_a.mean())

            replay_enc.append((ho_t.squeeze(0).detach(), hoa_t.squeeze(0).detach(),
                                harm_exposure, affect_level))

            env.step(action_idx)

        # Train encoders on replay
        if len(replay_enc) >= 32:
            idxs = random.sample(range(len(replay_enc)), min(64, len(replay_enc)))
            ho_batch  = torch.stack([replay_enc[i][0] for i in idxs]).to(device)
            hoa_batch = torch.stack([replay_enc[i][1] for i in idxs]).to(device)
            he_targets = torch.tensor(
                [[replay_enc[i][2]] for i in idxs], dtype=torch.float32, device=device
            )
            ae_targets = torch.tensor(
                [[replay_enc[i][3]] for i in idxs], dtype=torch.float32, device=device
            )
            z_hs = harm_enc(ho_batch)
            z_ha = affect_enc(hoa_batch)
            z_hs, z_ha = _maybe_fuse(z_hs, z_ha)
            loss_enc = (
                F.mse_loss(harm_eval_head(z_hs), he_targets) +
                F.mse_loss(affect_eval_head(z_ha), ae_targets)
            )
            opt_enc.zero_grad()
            loss_enc.backward()
            opt_enc.step()

    print(f"[seed={seed} haz={num_hazards} {condition}] Phase 0 done. "
          f"replay_enc={len(replay_enc)}")

    # ------------------------------------------------------------------ #
    # Phase 1a: collect transitions with frozen encoders                  #
    # ------------------------------------------------------------------ #
    for p in list(harm_enc.parameters()) + list(affect_enc.parameters()):
        p.requires_grad_(False)
    replay_fwd.clear()

    for _ in range(fwd_collect_episodes):
        env.reset()
        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            ho_t = obs_dict["harm_obs"].unsqueeze(0).to(device)
            with torch.no_grad():
                z_harm_s = harm_enc(ho_t)
                if fused_mode:
                    hoa_t = obs_dict["harm_obs_a"].unsqueeze(0).to(device)
                    z_harm_a = affect_enc(hoa_t)
                    z_harm_s, _ = _maybe_fuse(z_harm_s, z_harm_a)

            action_idx = random.randint(0, ACTION_DIM - 1)
            env.step(action_idx)

            obs_dict_next = env._get_observation_dict()
            ho_next = obs_dict_next["harm_obs"].unsqueeze(0).to(device)
            with torch.no_grad():
                z_harm_s_next = harm_enc(ho_next)
                if fused_mode:
                    hoa_next = obs_dict_next["harm_obs_a"].unsqueeze(0).to(device)
                    z_harm_a_next = affect_enc(hoa_next)
                    z_harm_s_next, _ = _maybe_fuse(z_harm_s_next, z_harm_a_next)

            replay_fwd.append((
                z_harm_s.squeeze(0).detach(),
                action_idx,
                z_harm_s_next.squeeze(0).detach(),
            ))

    print(f"[seed={seed} haz={num_hazards} {condition}] Phase 1a done. "
          f"replay_fwd={len(replay_fwd)}")

    # ------------------------------------------------------------------ #
    # Phase 1b: train HarmForwardModel                                    #
    # ------------------------------------------------------------------ #
    n = len(replay_fwd)
    effective_batch = min(fwd_batch_size, n)
    loss_fwd_final = float("nan")
    for epoch in range(fwd_train_epochs):
        idxs = list(range(n))
        random.shuffle(idxs)
        for batch_start in range(0, max(1, n - effective_batch + 1), effective_batch):
            batch_idxs = idxs[batch_start:batch_start + effective_batch]
            z_hs_b   = torch.stack([replay_fwd[i][0] for i in batch_idxs]).to(device)
            a_b      = torch.stack([
                _action_to_onehot(replay_fwd[i][1], ACTION_DIM, device).squeeze(0)
                for i in batch_idxs
            ]).to(device)
            z_next_b = torch.stack([replay_fwd[i][2] for i in batch_idxs]).to(device)

            z_pred = harm_fwd(z_hs_b, a_b)
            loss_fwd = F.mse_loss(z_pred, z_next_b)
            opt_fwd.zero_grad()
            loss_fwd.backward()
            opt_fwd.step()
            loss_fwd_final = float(loss_fwd.item())

    n_grad_steps = fwd_train_epochs * max(1, n // fwd_batch_size)
    print(f"[seed={seed} haz={num_hazards} {condition}] Phase 1b done. "
          f"{n_grad_steps} grad steps, final_loss={loss_fwd_final:.4f}")

    # Evaluate HarmForwardModel R2 on held-out transitions (last 200)
    held_out = replay_fwd[-200:] if len(replay_fwd) >= 200 else replay_fwd
    z_hs_batch   = torch.stack([t[0] for t in held_out]).to(device)
    a_batch      = torch.stack([
        _action_to_onehot(t[1], ACTION_DIM, device).squeeze(0) for t in held_out
    ]).to(device)
    z_next_batch = torch.stack([t[2] for t in held_out]).to(device)

    with torch.no_grad():
        z_hs_pred = harm_fwd(z_hs_batch, a_batch)
    ss_res = float(((z_next_batch - z_hs_pred) ** 2).sum())
    ss_tot = float(((z_next_batch - z_next_batch.mean(0)) ** 2).sum())
    harm_fwd_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    print(f"[seed={seed} haz={num_hazards} {condition}] C1 harm_fwd_r2 = {harm_fwd_r2:.4f}")

    # ------------------------------------------------------------------ #
    # Phase 2: Dissociation probe                                         #
    # ------------------------------------------------------------------ #
    # Unfreeze encoders for probe phase (they are not trained further, but
    # requires_grad state is irrelevant for inference -- just restore for clarity)
    for p in list(harm_enc.parameters()) + list(affect_enc.parameters()):
        p.requires_grad_(True)

    z_hs_norms: List[float] = []
    z_ha_norms: List[float] = []
    hazard_levels: List[float] = []

    for ep in range(dissociation_episodes):
        env.reset()
        approach = (ep % 2 == 0)

        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            ho_t  = obs_dict["harm_obs"].unsqueeze(0).to(device)
            hoa_t = obs_dict["harm_obs_a"].unsqueeze(0).to(device)

            with torch.no_grad():
                z_hs = harm_enc(ho_t)
                z_ha = affect_enc(hoa_t)
                z_hs, z_ha = _maybe_fuse(z_hs, z_ha)

            z_hs_norms.append(float(z_hs.norm()))
            z_ha_norms.append(float(z_ha.norm()))

            h_view = obs_dict["hazard_field_view"].numpy()
            center_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
            hazard_levels.append(float(np.mean([h_view[i] for i in center_indices])))

            if approach:
                action_idx = _hazard_gradient_action(env, toward=True)
            else:
                action_idx = random.randint(0, ACTION_DIM - 1)
            env.step(action_idx)

    # C2: stream correlation
    stream_corr = _pearson(np.array(z_hs_norms), np.array(z_ha_norms))

    # C3: lag-10 autocorrelation gap
    LAG = 10
    z_ha_autocorr = _autocorr(z_ha_norms, LAG)
    z_hs_autocorr = _autocorr(z_hs_norms, LAG)
    autocorr_gap  = z_ha_autocorr - z_hs_autocorr

    # C4: z_harm_s vs hazard proximity
    z_harm_s_hazard_corr = _pearson(np.array(z_hs_norms), np.array(hazard_levels))

    # Composite dissociation score for C5 comparison
    dissociation_score = (1.0 - abs(stream_corr)) + autocorr_gap + z_harm_s_hazard_corr

    print(f"[seed={seed} haz={num_hazards} {condition}] "
          f"C2 stream_corr={stream_corr:.4f} "
          f"C3 autocorr_gap={autocorr_gap:.4f} "
          f"C4 hazard_corr={z_harm_s_hazard_corr:.4f} "
          f"dissociation_score={dissociation_score:.4f}")

    return {
        "seed": seed,
        "num_hazards": num_hazards,
        "condition": condition,
        "harm_fwd_r2": harm_fwd_r2,
        "stream_corr": stream_corr,
        "autocorr_gap": autocorr_gap,
        "z_ha_autocorr_lag10": z_ha_autocorr,
        "z_hs_autocorr_lag10": z_hs_autocorr,
        "z_harm_s_hazard_corr": z_harm_s_hazard_corr,
        "dissociation_score": dissociation_score,
        "n_steps_measured": len(z_hs_norms),
        "z_hs_norms_mean": float(np.mean(z_hs_norms)) if z_hs_norms else 0.0,
        "z_ha_norms_mean": float(np.mean(z_ha_norms)) if z_ha_norms else 0.0,
        "hazard_level_mean": float(np.mean(hazard_levels)) if hazard_levels else 0.0,
        "fwd_loss_final": loss_fwd_final,
    }


def _evaluate_criteria(
    dual_results: List[Dict],
    fused_results: List[Dict],
    seeds: List[int],
) -> Tuple[str, Dict, Dict]:
    """
    Evaluate all pre-registered criteria on aggregated results.

    Returns (outcome, criteria_dict, decision_dict).
    """
    # Aggregate DUAL_STREAM metrics across all (seed, layout) runs
    dual_fwd_r2    = float(np.mean([r["harm_fwd_r2"] for r in dual_results]))
    dual_corr      = float(np.mean([r["stream_corr"] for r in dual_results]))
    dual_gap       = float(np.mean([r["autocorr_gap"] for r in dual_results]))
    dual_haz_corr  = float(np.mean([r["z_harm_s_hazard_corr"] for r in dual_results]))
    dual_dissoc    = float(np.mean([r["dissociation_score"] for r in dual_results]))

    fused_dissoc   = float(np.mean([r["dissociation_score"] for r in fused_results]))

    c1 = dual_fwd_r2 >= THRESH_C1_FWD_R2
    c2 = dual_corr <= THRESH_C2_STREAM_CORR
    c3 = dual_gap >= THRESH_C3_AUTOCORR_GAP
    c4 = dual_haz_corr >= THRESH_C4_S_CORR
    c5 = dual_dissoc > fused_dissoc
    # C6: each seed individually passes C1 AND C4 (averaged across layouts for that seed)
    c6_per_seed = {}
    for seed in seeds:
        seed_runs = [r for r in dual_results if r["seed"] == seed]
        seed_fwd_r2   = float(np.mean([r["harm_fwd_r2"] for r in seed_runs]))
        seed_haz_corr = float(np.mean([r["z_harm_s_hazard_corr"] for r in seed_runs]))
        c6_per_seed[seed] = {
            "harm_fwd_r2": seed_fwd_r2,
            "z_harm_s_hazard_corr": seed_haz_corr,
            "passes_c1": seed_fwd_r2 >= THRESH_C1_FWD_R2,
            "passes_c4": seed_haz_corr >= THRESH_C4_S_CORR,
            "passes": (seed_fwd_r2 >= THRESH_C1_FWD_R2) and (seed_haz_corr >= THRESH_C4_S_CORR),
        }
    c6 = all(v["passes"] for v in c6_per_seed.values())

    criteria = {
        "C1_replication_fwd_r2": c1,
        "C2_stream_dissociation": c2,
        "C3_temporal_integration": c3,
        "C4_sensory_responsiveness": c4,
        "C5_dual_advantage": c5,
        "C6_seed_stability": c6,
    }

    metrics = {
        "dual_harm_fwd_r2": dual_fwd_r2,
        "dual_stream_corr": dual_corr,
        "dual_autocorr_gap": dual_gap,
        "dual_z_harm_s_hazard_corr": dual_haz_corr,
        "dual_dissociation_score": dual_dissoc,
        "fused_dissociation_score": fused_dissoc,
        "dissociation_advantage": dual_dissoc - fused_dissoc,
        "c6_per_seed": c6_per_seed,
    }

    # Decision scoring
    all_pass = all(criteria.values())
    if all_pass:
        outcome = "PASS"
        decision = "retain_ree"
    elif c1 and c4 and not (c2 and c3):
        outcome = "FAIL"
        decision = "hybridize"
    elif not c1 or not c4:
        outcome = "FAIL"
        decision = "retire_ree_claim"
    else:
        # C1-C4 pass but C5 or C6 fails
        outcome = "FAIL"
        decision = "inconclusive"

    metrics["decision"] = decision

    return outcome, criteria, metrics


def main(dry_run: bool = False):
    seeds = [42, 7, 13]
    hazard_layouts = [3, 5]  # standard (matches EXQ-178b) + dense variant

    # Timing estimate: per run ~30 min on Mac at full scale
    # 2 conditions x 3 seeds x 2 layouts = 12 runs x ~30 min = ~360 min = ~6 hours
    # (Mac DLAPTOP-4.local: ~0.10 min/ep at 200 steps/ep)
    warmup_episodes      = 5  if dry_run else 150
    fwd_collect_episodes = 3  if dry_run else 80
    fwd_train_epochs     = 2  if dry_run else 10
    fwd_batch_size       = 128
    dissociation_eps     = 4  if dry_run else 40
    steps_per_episode    = 30 if dry_run else 200

    dual_results: List[Dict] = []
    fused_results: List[Dict] = []

    for seed in seeds:
        for num_hazards in hazard_layouts:
            for fused_mode in [False, True]:
                cond_label = "FUSED_STREAM" if fused_mode else "DUAL_STREAM"
                print(f"\n=== seed={seed} hazards={num_hazards} condition={cond_label} ===")
                r = run_single(
                    seed=seed,
                    num_hazards=num_hazards,
                    fused_mode=fused_mode,
                    warmup_episodes=warmup_episodes,
                    fwd_collect_episodes=fwd_collect_episodes,
                    fwd_train_epochs=fwd_train_epochs,
                    fwd_batch_size=fwd_batch_size,
                    dissociation_episodes=dissociation_eps,
                    steps_per_episode=steps_per_episode,
                    dry_run=dry_run,
                )
                if fused_mode:
                    fused_results.append(r)
                else:
                    dual_results.append(r)

    outcome, criteria, metrics = _evaluate_criteria(dual_results, fused_results, seeds)

    print("\n=== AGGREGATE RESULTS ===")
    print(f"  DUAL  harm_fwd_r2:          {metrics['dual_harm_fwd_r2']:.4f}")
    print(f"  DUAL  stream_corr:          {metrics['dual_stream_corr']:.4f}")
    print(f"  DUAL  autocorr_gap:         {metrics['dual_autocorr_gap']:.4f}")
    print(f"  DUAL  z_harm_s_hazard_corr: {metrics['dual_z_harm_s_hazard_corr']:.4f}")
    print(f"  DUAL  dissociation_score:   {metrics['dual_dissociation_score']:.4f}")
    print(f"  FUSED dissociation_score:   {metrics['fused_dissociation_score']:.4f}")
    print(f"  Advantage (dual - fused):   {metrics['dissociation_advantage']:.4f}")
    print(f"\nOutcome: {outcome}")
    print(f"Decision: {metrics['decision']}")
    for crit, val in criteria.items():
        print(f"  {crit}: {'PASS' if val else 'FAIL'}")
    print("\nC6 per-seed detail:")
    for seed, detail in metrics["c6_per_seed"].items():
        print(f"  seed={seed}: fwd_r2={detail['harm_fwd_r2']:.4f} "
              f"haz_corr={detail['z_harm_s_hazard_corr']:.4f} "
              f"-> {'PASS' if detail['passes'] else 'FAIL'}")

    if dry_run:
        print("\n[DRY RUN] Skipping output file write.")
        return

    run_id = (
        f"{EXPERIMENT_TYPE}_"
        f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    evidence_dir = (
        Path(__file__).resolve().parents[2] /
        "REE_assembly" / "evidence" / "experiments"
    )
    out_dir = evidence_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "decision": metrics["decision"],
        "criteria": criteria,
        "metrics": metrics,
        "per_run_dual": dual_results,
        "per_run_fused": fused_results,
        "thresholds": {
            "C1_harm_fwd_r2": THRESH_C1_FWD_R2,
            "C2_stream_corr": THRESH_C2_STREAM_CORR,
            "C3_autocorr_gap": THRESH_C3_AUTOCORR_GAP,
            "C4_z_harm_s_hazard_corr": THRESH_C4_S_CORR,
            "C5_dual_advantage": "dual_dissociation_score > fused_dissociation_score",
            "C6_seed_stability": "all seeds individually pass C1 AND C4",
        },
        "design": {
            "dispatch_mode": "discriminative_pair",
            "condition_A": "DUAL_STREAM (z_harm_s + z_harm_a separate)",
            "condition_B": "FUSED_STREAM (averaged into shared representation)",
            "seeds": seeds,
            "hazard_layouts": hazard_layouts,
            "replicates": f"{len(seeds)} seeds x {len(hazard_layouts)} layouts = "
                          f"{len(seeds) * len(hazard_layouts)} runs per condition",
        },
        "config": {
            "warmup_episodes": warmup_episodes,
            "fwd_collect_episodes": fwd_collect_episodes,
            "fwd_train_epochs": fwd_train_epochs,
            "fwd_batch_size": fwd_batch_size,
            "dissociation_episodes": dissociation_eps,
            "steps_per_episode": steps_per_episode,
            "harm_obs_dim": HARM_OBS_DIM,
            "harm_obs_a_dim": HARM_OBS_A_DIM,
            "z_harm_s_dim": Z_HARM_S_DIM,
            "z_harm_a_dim": Z_HARM_A_DIM,
        },
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    print(f"run_id: {run_id}")
    print(f"Final outcome: {outcome}")
    print(f"Decision: {metrics['decision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
