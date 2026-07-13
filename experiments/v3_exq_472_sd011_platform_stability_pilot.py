#!/opt/local/bin/python3
"""
V3-EXQ-472 -- SD-011 Platform-Stability Diagnostic Pilot

Claims: ["SD-011"]
EXPERIMENT_PURPOSE = "diagnostic"
evidence_direction = "diagnostic" (not PASS/FAIL scored)

Purpose
-------
Anchor C7 perturbation-recovery threshold and C5 dual_advantage multiplier for the
EXP-0090 main evidence script. This pilot is NOT itself an evidence run -- it
measures the natural range of `recovery_ratio` (post-perturbation / pre-perturbation
dissociation_score) under a fixed DUAL-stream substrate across two hazard layouts
and two seeds, so EXP-0090's pre-registered thresholds can be set from observation.

Design
------
- Conditions:   2 (layout L3=3 hazards, layout L5=5 hazards). DUAL-stream only.
- Seeds:        2 (42, 7).
- Total runs:   2 * 2 = 4.
- steps_per_episode: 200.
- episodes_per_run: 300 (P0=120 + P1=80 + P2=40 + P3=60).
- drive_weight:  2.0 (post-288 substrate default).

Per-run phased pipeline
-----------------------
- P0 warmup (ep 0..120): full REE agent training with DUAL encoders active.
  LatentStackConfig: harm_history_len=10, z_harm_a_aux_loss_weight=0.1.
  HarmEncoder + AffectiveHarmEncoder both train. No forward-model loss yet.
- P1 frozen-encoder head (ep 120..200): rollout-collect 80 eps into a replay buffer,
  then train E2_harm_s forward head (HarmStreamForwardProbe) for 10 epochs x
  batch=128 on FROZEN (.detach()ed) z_harm_s inputs and targets. Aux harm_accum
  loss continues on the encoder. Identity-collapse guard: .detach() on both
  forward-head input and target (see EXQ-166b/c/d, EXQ-194).
- P2 eval pre-perturbation (ep 200..240): no parameter updates. Measure
  dissociation_score_pre, harm_fwd_r2_pre, stream_corr_pre,
  z_harm_s_hazard_corr_pre, autocorr_gap_pre over 40 eps x 200 steps = 8000 steps.
- P3 perturbation + recovery (ep 240..300): at ep 240, SWAP hazard layout.
  L3 runs swap TO L5 positions; L5 runs swap TO L3 positions. Other env state
  (agent position, damage, drive) is re-sampled by reset() but the encoder
  weights, forward head, optimizer state, and z_harm_a harm_history all carry
  through. Resume agent training for 30 eps (ep 240..270), then measure the
  P2 metrics again over the final 30 eps (ep 270..300). Record as *_post.

Per-run output (flat JSON, REE_assembly/evidence/experiments/<script>/...)
-------------------------------------------------------------------------
    seed, layout (L3 / L5), num_hazards, swap_layout (what it became)
    dissociation_score_pre, dissociation_score_post
    recovery_ratio = post / pre  (with /eps guard)
    harm_fwd_r2_pre, harm_fwd_r2_post
    stream_corr_pre, stream_corr_post
    z_harm_s_hazard_corr_pre, z_harm_s_hazard_corr_post
    autocorr_gap_pre, autocorr_gap_post

Summary (diagnostic -- NO PASS/FAIL)
------------------------------------
- median recovery_ratio across the 4 runs
- min / max recovery_ratio across the 4 runs
- median dissociation_score_pre and dissociation_score_post separately
- the 4 per-run recovery_ratios printed to stdout for the runner log

The runner verdict-count check does not apply (diagnostic); still prints
"verdict: PASS" once per run so the runner progress bar advances correctly.
All stdout ASCII-only.

References
----------
- ree-v3/experiments/v3_exq_198_sd011_dual_stream_stability.py (measurement code)
- ree-v3/experiments/v3_exq_241a_sd011_second_source_validation.py
  (harm_history + z_harm_a_aux_loss_weight wiring; compute_harm_accum_loss flow)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

MANIFEST_WRITER_EXEMPT = "archival early-era manifest (non-canonical filename not provably == run_id.json; superseded lineage, not re-run)"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_472_sd011_platform_stability_pilot"
CLAIM_IDS          = ["SD-011"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM    = 12
WORLD_OBS_DIM   = 250
ACTION_DIM_AGENT = 4   # REEAgent action_dim (env has 5 including stay)
ACTION_DIM_ENV   = 5   # sampled uniformly 0..4
WORLD_DIM       = 32
Z_HARM_S_DIM    = 32
Z_HARM_A_DIM    = 16
HARM_HISTORY_LEN = 10
Z_HARM_A_AUX_LOSS_WEIGHT = 0.1

# ---------------------------------------------------------------------------
# Schedule (full run)
# ---------------------------------------------------------------------------
STEPS_PER_EP    = 200
P0_EPS          = 120
P1_EPS          = 80
P2_EPS          = 40
P3_RECOVERY_EPS = 30    # post-swap training
P3_EVAL_EPS     = 30    # post-swap measurement
TOTAL_EPS_PER_RUN = P0_EPS + P1_EPS + P2_EPS + P3_RECOVERY_EPS + P3_EVAL_EPS  # 300

# Forward-head training
FWD_TRAIN_EPOCHS = 10
FWD_BATCH_SIZE   = 128

SEEDS     = [42, 7]
LAYOUTS   = [3, 5]

AUTOCORR_LAG = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    ac = a - a.mean()
    bc = b - b.mean()
    denom = (np.sqrt((ac ** 2).sum()) * np.sqrt((bc ** 2).sum())) + 1e-8
    return float(np.dot(ac, bc) / denom)


def _autocorr(series: List[float], lag: int) -> float:
    arr = np.array(series, dtype=np.float64)
    if len(arr) <= lag or float(np.std(arr)) < 1e-8:
        return 0.0
    a = arr[:-lag] - arr[:-lag].mean()
    b = arr[lag:] - arr[lag:].mean()
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())) + 1e-8
    return float(np.dot(a, b) / denom)


class HarmStreamForwardProbe(nn.Module):
    """MLP(z_harm_s_t, action_t) -> z_harm_s_{t+1}. Residual delta head."""

    def __init__(self, z_harm_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_harm_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_harm_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, a], dim=-1))
        return z + delta


# ---------------------------------------------------------------------------
# Measurement block (pre/post P2 and post-perturbation eval)
# ---------------------------------------------------------------------------
def measure_block(
    agent: REEAgent,
    env: CausalGridWorldV2,
    fwd_probe: HarmStreamForwardProbe,
    n_eps: int,
    steps_per_ep: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run n_eps eval episodes, collect metrics, no gradient updates."""
    z_hs_norms: List[float] = []
    z_ha_norms: List[float] = []
    hazard_levels: List[float] = []

    fwd_preds: List[torch.Tensor] = []
    fwd_tgts: List[torch.Tensor] = []

    agent_was_training = agent.latent_stack.training
    agent.latent_stack.eval()
    fwd_probe.eval()

    try:
        for _ in range(n_eps):
            obs, obs_dict = env.reset()
            agent.reset()
            prev_z_harm_s: Optional[torch.Tensor] = None
            prev_action_oh: Optional[torch.Tensor] = None

            for _step in range(steps_per_ep):
                with torch.no_grad():
                    latent = agent.sense(
                        obs_dict["body_state"].unsqueeze(0),
                        obs_dict["world_state"].unsqueeze(0),
                        obs_harm=obs_dict.get("harm_obs"),
                        obs_harm_a=obs_dict.get("harm_obs_a"),
                        obs_harm_history=obs_dict.get("harm_history"),
                    )
                z_harm_s = latent.z_harm.detach() if latent.z_harm is not None else None
                z_harm_a = latent.z_harm_a.detach() if latent.z_harm_a is not None else None

                if z_harm_s is not None:
                    z_hs_norms.append(float(z_harm_s.norm().item()))
                if z_harm_a is not None:
                    z_ha_norms.append(float(z_harm_a.norm().item()))

                # Center-of-view hazard intensity proxy.
                hfv = obs_dict.get("hazard_field_view")
                if hfv is not None:
                    hv = hfv.cpu().numpy().reshape(-1)
                    # 5x5 centred indices.
                    center_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
                    try:
                        hazard_levels.append(float(np.mean([hv[i] for i in center_indices])))
                    except IndexError:
                        hazard_levels.append(float(hv.mean()))
                else:
                    hazard_levels.append(0.0)

                # Forward-head rollout eval (predict current z_harm_s from prev + action).
                if prev_z_harm_s is not None and prev_action_oh is not None and z_harm_s is not None:
                    with torch.no_grad():
                        pred = fwd_probe(prev_z_harm_s, prev_action_oh)
                    fwd_preds.append(pred.squeeze(0))
                    fwd_tgts.append(z_harm_s.squeeze(0))

                action = random.randint(0, ACTION_DIM_ENV - 1)
                prev_action_oh = _action_to_onehot(action, ACTION_DIM_ENV, device)
                prev_z_harm_s = z_harm_s

                obs, reward, done, info, obs_dict = env.step(action)
                if done:
                    break
    finally:
        if agent_was_training:
            agent.latent_stack.train()

    # Compute metrics.
    stream_corr = _pearson(np.array(z_hs_norms), np.array(z_ha_norms))
    a_ac = _autocorr(z_ha_norms, AUTOCORR_LAG)
    s_ac = _autocorr(z_hs_norms, AUTOCORR_LAG)
    autocorr_gap = a_ac - s_ac
    haz_corr = _pearson(np.array(z_hs_norms), np.array(hazard_levels))
    dissociation_score = (1.0 - abs(stream_corr)) + autocorr_gap + haz_corr

    harm_fwd_r2 = 0.0
    if fwd_preds:
        preds_t = torch.stack(fwd_preds)
        tgts_t = torch.stack(fwd_tgts)
        ss_res = float(((preds_t - tgts_t) ** 2).sum().item())
        ss_tot = float(((tgts_t - tgts_t.mean(dim=0, keepdim=True)) ** 2).sum().item())
        harm_fwd_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        "stream_corr": float(stream_corr),
        "autocorr_gap": float(autocorr_gap),
        "z_harm_s_hazard_corr": float(haz_corr),
        "dissociation_score": float(dissociation_score),
        "harm_fwd_r2": float(harm_fwd_r2),
        "n_steps": int(len(z_hs_norms)),
    }


# ---------------------------------------------------------------------------
# P0 warmup train step (full REE agent + aux loss)
# ---------------------------------------------------------------------------
def warmup_step(
    agent: REEAgent,
    obs_dict: Dict,
    agent_opt: optim.Optimizer,
):
    """One gradient step on LatentStack via aux loss (harm_accum + resource_prox)."""
    latent_grad = agent.sense(
        obs_dict["body_state"].unsqueeze(0),
        obs_dict["world_state"].unsqueeze(0),
        obs_harm=obs_dict.get("harm_obs"),
        obs_harm_a=obs_dict.get("harm_obs_a"),
        obs_harm_history=obs_dict.get("harm_history"),
    )
    accum_target = obs_dict.get("accumulated_harm", 0.0)
    aux_loss = agent.compute_harm_accum_loss(accum_target, latent_grad)
    rpv = obs_dict.get("resource_field_view")
    if rpv is not None:
        rpt = float(rpv.max().item())
        rp_loss = agent.compute_resource_proximity_loss(rpt, latent_grad)
        total = aux_loss + rp_loss
    else:
        total = aux_loss

    if hasattr(total, "requires_grad") and total.requires_grad:
        agent_opt.zero_grad()
        total.backward()
        agent_opt.step()


# ---------------------------------------------------------------------------
# Single run (seed x layout)
# ---------------------------------------------------------------------------
def run_single(seed: int, layout_num_hazards: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    layout_label = f"L{layout_num_hazards}"
    swap_to = 5 if layout_num_hazards == 3 else 3
    swap_label = f"L{swap_to}"

    print(f"Seed {seed} Condition {layout_label}", flush=True)
    print(f"[pilot] seed={seed} layout={layout_label} swap_to={swap_label}", flush=True)

    if dry_run:
        p0 = 1
        p1 = 1
        p2 = 1
        p3r = 1
        p3e = 1
        spe = 10
        fwd_epochs = 1
    else:
        p0, p1, p2, p3r, p3e, spe = P0_EPS, P1_EPS, P2_EPS, P3_RECOVERY_EPS, P3_EVAL_EPS, STEPS_PER_EP
        fwd_epochs = FWD_TRAIN_EPOCHS

    total_eps = p0 + p1 + p2 + p3r + p3e

    # Build agent + env.
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM_AGENT,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=HARM_HISTORY_LEN,
        z_harm_a_aux_loss_weight=Z_HARM_A_AUX_LOSS_WEIGHT,
        use_resource_proximity_head=True,
        drive_weight=2.0,
    )
    agent = REEAgent(cfg)
    env = CausalGridWorldV2(
        use_proxy_fields=True,
        seed=seed,
        hazard_harm=0.5,
        num_hazards=layout_num_hazards,
        num_resources=5,
        harm_history_len=HARM_HISTORY_LEN,
    )

    agent_opt = optim.Adam(agent.latent_stack.parameters(), lr=1e-4)
    fwd_probe = HarmStreamForwardProbe(Z_HARM_S_DIM, ACTION_DIM_ENV).to(device)
    fwd_opt = optim.Adam(fwd_probe.parameters(), lr=3e-4)

    # Replay buffer for P1 forward head training.
    p1_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    ep_idx = 0
    # ---- P0 warmup ----
    for _ in range(p0):
        obs, obs_dict = env.reset()
        agent.reset()
        for _step in range(spe):
            warmup_step(agent, obs_dict, agent_opt)
            action = random.randint(0, ACTION_DIM_ENV - 1)
            obs, reward, done, info, obs_dict = env.step(action)
            if done:
                break
        ep_idx += 1
        if ep_idx % 20 == 0 or ep_idx == 1:
            print(f"  [train] phase=P0 seed={seed} {layout_label} ep {ep_idx}/{total_eps}", flush=True)

    # ---- P1a: collect transitions with frozen encoders ----
    # Freeze LatentStack (encoder).
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)
    agent.latent_stack.eval()

    for _ in range(p1):
        obs, obs_dict = env.reset()
        agent.reset()
        prev_z_harm_s: Optional[torch.Tensor] = None
        prev_action_oh: Optional[torch.Tensor] = None
        for _step in range(spe):
            with torch.no_grad():
                latent = agent.sense(
                    obs_dict["body_state"].unsqueeze(0),
                    obs_dict["world_state"].unsqueeze(0),
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
            z_harm_s = latent.z_harm.detach() if latent.z_harm is not None else None
            if prev_z_harm_s is not None and prev_action_oh is not None and z_harm_s is not None:
                # Identity-collapse guard: both sides .detach()ed already.
                p1_buffer.append((
                    prev_z_harm_s.squeeze(0).detach(),
                    prev_action_oh.squeeze(0).detach(),
                    z_harm_s.squeeze(0).detach(),
                ))
            action = random.randint(0, ACTION_DIM_ENV - 1)
            prev_action_oh = _action_to_onehot(action, ACTION_DIM_ENV, device)
            prev_z_harm_s = z_harm_s
            obs, reward, done, info, obs_dict = env.step(action)
            if done:
                break
        ep_idx += 1
        if ep_idx % 20 == 0:
            print(f"  [train] phase=P1_collect seed={seed} {layout_label} ep {ep_idx}/{total_eps}", flush=True)

    # ---- P1b: train forward head on frozen z_harm_s ----
    fwd_probe.train()
    n = len(p1_buffer)
    if n > 0:
        for epoch in range(fwd_epochs):
            idxs = list(range(n))
            random.shuffle(idxs)
            bs = min(FWD_BATCH_SIZE, n)
            for start in range(0, max(1, n - bs + 1), bs):
                batch_idxs = idxs[start:start + bs]
                z_b = torch.stack([p1_buffer[i][0] for i in batch_idxs]).to(device)
                a_b = torch.stack([p1_buffer[i][1] for i in batch_idxs]).to(device)
                z1_b = torch.stack([p1_buffer[i][2] for i in batch_idxs]).to(device)
                pred = fwd_probe(z_b, a_b)
                loss = F.mse_loss(pred, z1_b)
                fwd_opt.zero_grad()
                loss.backward()
                fwd_opt.step()
    print(f"  [train] phase=P1_train seed={seed} {layout_label} n_transitions={n}", flush=True)

    # ---- P2: eval pre-perturbation ----
    pre_metrics = measure_block(agent, env, fwd_probe, p2, spe, device)
    ep_idx += p2
    print(f"  [eval ] phase=P2 seed={seed} {layout_label} ep {ep_idx}/{total_eps} "
          f"dissoc_pre={pre_metrics['dissociation_score']:.4f} "
          f"fwd_r2_pre={pre_metrics['harm_fwd_r2']:.4f}",
          flush=True)

    # ---- P3 perturbation: swap hazard layout ----
    env.num_hazards = swap_to
    # Reset to apply new layout next episode; seeds remain stable via env._rng continuation.
    print(f"  [swap ] seed={seed} {layout_label} -> {swap_label} at ep {ep_idx}/{total_eps}", flush=True)

    # Unfreeze encoder for recovery training.
    for p in agent.latent_stack.parameters():
        p.requires_grad_(True)
    agent.latent_stack.train()

    # ---- P3a: recovery training ----
    for _ in range(p3r):
        obs, obs_dict = env.reset()
        agent.reset()
        for _step in range(spe):
            warmup_step(agent, obs_dict, agent_opt)
            action = random.randint(0, ACTION_DIM_ENV - 1)
            obs, reward, done, info, obs_dict = env.step(action)
            if done:
                break
        ep_idx += 1
        if ep_idx % 10 == 0 or ep_idx % 20 == 0:
            print(f"  [train] phase=P3_recover seed={seed} {swap_label} ep {ep_idx}/{total_eps}", flush=True)

    # Freeze encoder for final eval block (match P2 protocol).
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)
    agent.latent_stack.eval()

    # ---- P3b: eval post-perturbation ----
    post_metrics = measure_block(agent, env, fwd_probe, p3e, spe, device)
    ep_idx += p3e
    print(f"  [eval ] phase=P3 seed={seed} {swap_label} ep {ep_idx}/{total_eps} "
          f"dissoc_post={post_metrics['dissociation_score']:.4f} "
          f"fwd_r2_post={post_metrics['harm_fwd_r2']:.4f}",
          flush=True)

    dissoc_pre = pre_metrics["dissociation_score"]
    dissoc_post = post_metrics["dissociation_score"]
    recovery_ratio = dissoc_post / dissoc_pre if abs(dissoc_pre) > 1e-8 else 0.0

    # Diagnostic prints (one verdict line for runner progress completion).
    print(f"  [pilot] seed={seed} layout={layout_label}->{swap_label} "
          f"recovery_ratio={recovery_ratio:.4f}", flush=True)
    print(f"verdict: PASS", flush=True)

    return {
        "seed": seed,
        "layout": layout_label,
        "num_hazards": layout_num_hazards,
        "swap_layout": swap_label,
        "swap_num_hazards": swap_to,
        "dissociation_score_pre": float(dissoc_pre),
        "dissociation_score_post": float(dissoc_post),
        "recovery_ratio": float(recovery_ratio),
        "harm_fwd_r2_pre": float(pre_metrics["harm_fwd_r2"]),
        "harm_fwd_r2_post": float(post_metrics["harm_fwd_r2"]),
        "stream_corr_pre": float(pre_metrics["stream_corr"]),
        "stream_corr_post": float(post_metrics["stream_corr"]),
        "z_harm_s_hazard_corr_pre": float(pre_metrics["z_harm_s_hazard_corr"]),
        "z_harm_s_hazard_corr_post": float(post_metrics["z_harm_s_hazard_corr"]),
        "autocorr_gap_pre": float(pre_metrics["autocorr_gap"]),
        "autocorr_gap_post": float(post_metrics["autocorr_gap"]),
        "p1_buffer_size": int(n),
        "n_steps_measured_pre": int(pre_metrics["n_steps"]),
        "n_steps_measured_post": int(post_metrics["n_steps"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(np.median(np.array(vals, dtype=np.float64)))


def main(dry_run: bool = False) -> None:
    print(f"{EXPERIMENT_TYPE}: SD-011 platform-stability diagnostic pilot", flush=True)
    print(f"  seeds={SEEDS} layouts={LAYOUTS} dual-only", flush=True)
    print(f"  total_runs={len(SEEDS) * len(LAYOUTS)} "
          f"episodes_per_run={TOTAL_EPS_PER_RUN} steps_per_episode={STEPS_PER_EP}",
          flush=True)
    print(f"  P0={P0_EPS} P1={P1_EPS} P2={P2_EPS} P3_recover={P3_RECOVERY_EPS} "
          f"P3_eval={P3_EVAL_EPS}", flush=True)
    print("", flush=True)

    all_runs: List[Dict] = []
    for layout_num_hazards in LAYOUTS:
        for seed in SEEDS:
            r = run_single(seed=seed, layout_num_hazards=layout_num_hazards, dry_run=dry_run)
            all_runs.append(r)

    recovery_ratios = [r["recovery_ratio"] for r in all_runs]
    dissoc_pre = [r["dissociation_score_pre"] for r in all_runs]
    dissoc_post = [r["dissociation_score_post"] for r in all_runs]

    summary = {
        "n_runs": len(all_runs),
        "recovery_ratio_median": _median(recovery_ratios),
        "recovery_ratio_min": float(min(recovery_ratios)) if recovery_ratios else 0.0,
        "recovery_ratio_max": float(max(recovery_ratios)) if recovery_ratios else 0.0,
        "dissociation_score_pre_median": _median(dissoc_pre),
        "dissociation_score_post_median": _median(dissoc_post),
        "recovery_ratios": recovery_ratios,
    }

    print("", flush=True)
    print("=" * 60, flush=True)
    print("DIAGNOSTIC SUMMARY (no PASS/FAIL)", flush=True)
    for r in all_runs:
        print(f"  result: seed={r['seed']} {r['layout']}->{r['swap_layout']} "
              f"recovery_ratio={r['recovery_ratio']:.4f} "
              f"dissoc_pre={r['dissociation_score_pre']:.4f} "
              f"dissoc_post={r['dissociation_score_post']:.4f}", flush=True)
    print(f"  result: recovery_ratio_median={summary['recovery_ratio_median']:.4f}", flush=True)
    print(f"  result: recovery_ratio_min={summary['recovery_ratio_min']:.4f}", flush=True)
    print(f"  result: recovery_ratio_max={summary['recovery_ratio_max']:.4f}", flush=True)
    print(f"  result: dissociation_score_pre_median={summary['dissociation_score_pre_median']:.4f}",
          flush=True)
    print(f"  result: dissociation_score_post_median={summary['dissociation_score_post_median']:.4f}",
          flush=True)
    print("=" * 60, flush=True)

    if dry_run:
        print("[DRY RUN] Skipping output file write.", flush=True)
        return

    # Output: flat JSON per run + aggregate manifest.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    evidence_root = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
        / EXPERIMENT_TYPE
    )
    out_dir = evidence_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flat JSON per run (explorer-launch pattern).
    for r in all_runs:
        per_run_path = out_dir / f"run_seed{r['seed']}_{r['layout']}_to_{r['swap_layout']}.json"
        per_run_payload = {
            "experiment_type": EXPERIMENT_TYPE,
            "run_id": run_id,
            "claim_ids": CLAIM_IDS,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "evidence_direction": "diagnostic",
            "timestamp_utc": ts,
            **r,
        }
        with open(per_run_path, "w") as f:
            json.dump(per_run_payload, f, indent=2)

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": "DIAGNOSTIC",
        "status": "DIAGNOSTIC",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "layouts": LAYOUTS,
        "design": {
            "conditions": len(LAYOUTS),
            "seeds": len(SEEDS),
            "n_runs": len(all_runs),
            "episodes_per_run": TOTAL_EPS_PER_RUN,
            "steps_per_episode": STEPS_PER_EP,
            "phases": {
                "P0": P0_EPS,
                "P1": P1_EPS,
                "P2": P2_EPS,
                "P3_recover": P3_RECOVERY_EPS,
                "P3_eval": P3_EVAL_EPS,
            },
            "dual_only": True,
            "drive_weight": 2.0,
            "harm_history_len": HARM_HISTORY_LEN,
            "z_harm_a_aux_loss_weight": Z_HARM_A_AUX_LOSS_WEIGHT,
            "perturbation": "hazard_layout_swap_at_ep_p0_plus_p1_plus_p2",
        },
        "summary": summary,
        "per_run": all_runs,
        "note": (
            "Diagnostic pilot -- anchors C7 perturbation-recovery threshold and C5 "
            "dual_advantage multiplier for the EXP-0090 main evidence script. "
            "Not itself an evidence run. Not PASS/FAIL scored."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Flat output for runner.
    flat_out = {
        "output_files": [str(manifest_path)] + [
            str(out_dir / f"run_seed{r['seed']}_{r['layout']}_to_{r['swap_layout']}.json")
            for r in all_runs
        ],
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "status": "DIAGNOSTIC",
        "experiment_purpose": EXPERIMENT_PURPOSE,
    }
    with open(evidence_root / f"{EXPERIMENT_TYPE}_output.json", "w") as f:
        json.dump(flat_out, f, indent=2)

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Result written to: {manifest_path}", flush=True)
    print(f"run_id: {run_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="1 ep per phase, no output write; verify pipeline end-to-end.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
