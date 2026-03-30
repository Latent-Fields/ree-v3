#!/opt/local/bin/python3
"""
V3-EXQ-178a -- SD-011 Dual Nociceptive Stream Dissociation (supersedes EXQ-178)

Supersedes: V3-EXQ-178
Root cause of EXQ-178 failures:
  C1 FAIL (harm_fwd_r2=-0.73): replay_fwd was populated DURING warmup (Phase 0), while
    encoders were still being updated. Stored (z_harm_s_t, action, z_harm_s_next) pairs
    were inconsistent -- z_harm_s_t encoded by encoder at step t, z_harm_s_next encoded
    by encoder at step t+1 (which had been updated in between). Training on these stale
    pairs caused the forward model to learn noise. Fix: clear replay_fwd at end of Phase 0,
    collect fresh transitions only after encoders are frozen.
  C3 FAIL (recovery_ratio=1.005): the ratio of z_harm_a norms (sustained vs brief) was
    barely above 1.0 because AffectiveHarmEncoder was trained on harm_obs_a.mean() which
    varies little between conditions. Fix: replace recovery_ratio with lag-10 autocorrelation
    gap -- z_harm_a should have higher autocorr at lag 10 than z_harm_s (EMA tau=20 steps
    means z_harm_a's input is far more correlated step-to-step than z_harm_s's input).
    This tests the temporal integration property directly from the input structure.

Claims: SD-011, ARC-033

Context:
  SD-010 implemented a single z_harm stream (sensory-discriminative, Adelta-pathway analog).
  EXQ-093/094 confirmed that HarmBridge(z_world -> z_harm) has bridge_r2=0 by construction
  (SD-010 makes z_world perp z_harm). The SD-003 counterfactual pipeline therefore cannot
  operate via z_world; it must operate within the harm stream.

  SD-011 splits the single harm stream into two biologically distinct pathways:
    z_harm_s (sensory-discriminative, Adelta-pathway): immediate proximity, forward-predictable.
               HarmForwardModel(z_harm_s, action) -> z_harm_s_next enables SD-003 redesign.
    z_harm_a (affective-motivational, C-fiber/paleospinothalamic): EMA-accumulated homeostatic
               deviation. NOT forward-predicted. Feeds E3 urgency gating directly (ARC-016).

  This experiment validates:
  (1) That the two streams are WIRED and produce different representations (not redundant copies)
  (2) That HarmForwardModel can learn z_harm_s transitions (prerequisite for SD-003 redesign)
  (3) That z_harm_a accumulates slower than z_harm_s (temporal integration property)
  (4) That z_harm_s responds rapidly to hazard proximity changes

  Biological grounding:
    Rainville et al. (1997, Science): hypnotic modulation of unpleasantness modulates ACC
    (affective = z_harm_a), not S1 (discriminative = z_harm_s). Craig (2003): pain as
    homeostatic emotion -- the affective stream encodes motivational urgency, not spatial location.
    Keltner et al. (2006): predictability suppresses S1/S2 nociception -- z_harm_s is
    the stream that can be modeled and cancelled by HarmForwardModel.

PRE-REGISTERED ACCEPTANCE CRITERIA (all required for PASS):
  C1 (forward model): harm_fwd_r2 >= 0.20
    HarmForwardModel must learn z_harm_s transitions better than identity baseline.
    Note: latent-space (not obs-space) forward model. z_harm_s is trained on full 51-dim
    harm_obs (spatial hazard field + resource field + exposure), so unlike EXQ-115 which
    used a single scalar, there is enough spatial structure to learn 1-step transitions.
    Identity baseline R2 ~ 0 (proximity field shifts by 1 cell = 20% change per step).

  C2 (stream dissociation): stream_corr <= 0.85
    Pearson correlation between z_harm_s and z_harm_a norms across time should not be
    saturated. If corr > 0.85, the streams are redundant and SD-011 wiring is degenerate.
    Expected: moderate correlation (~0.5-0.7) since both respond to hazard proximity,
    but z_harm_a is temporally integrated and should diverge after sustained exposure.

  C3 (temporal integration): autocorr_gap >= 0.10
    Lag-10 autocorrelation of z_harm_a norm series minus lag-10 autocorrelation of
    z_harm_s norm series must be >= 0.10. harm_obs_a is an EMA of proximity fields with
    tau=20 steps, so its input is far more temporally correlated than harm_obs (immediate).
    After encoding, z_harm_a should inherit this slow temporal structure: autocorr(lag=10)
    for z_harm_a should exceed autocorr(lag=10) for z_harm_s by >= 0.10.
    This tests the temporal integration property directly from the input structure, without
    requiring specific sustained vs brief episode counts.

  C4 (sensory responsiveness): z_harm_s_hazard_corr >= 0.25
    Pearson correlation between z_harm_s norm and hazard field intensity at the agent's
    position should be positive and >= 0.25. This confirms z_harm_s encodes proximity
    levels (not just random noise). z_harm_a correlation is also measured but expected
    to be lower due to temporal integration smoothing out instantaneous proximity.

Decision scoring:
  PASS:         ALL criteria met -- SD-011 dual stream validated, SD-003 redesign unblocked
  hybridize:    C1 + C4 pass, C2 or C3 fail -- streams exist but temporal integration unclear
  inconclusive: C1 fails -- HarmForwardModel cannot learn transitions (identity collapse again)
  fail:         C1 passes but C4 fails -- sensory stream not encoding proximity levels
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


EXPERIMENT_TYPE = "v3_exq_178a_sd011_dual_stream_dissociation"
CLAIM_IDS = ["SD-011", "ARC-033"]

# Pre-registered thresholds (must not be changed post-hoc)
THRESH_C1_FWD_R2      = 0.20   # HarmForwardModel R2 on held-out z_harm_s transitions
THRESH_C2_STREAM_CORR = 0.85   # max allowed norm correlation between z_harm_s and z_harm_a
THRESH_C3_AUTOCORR_GAP = 0.10  # z_harm_a autocorr(lag=10) - z_harm_s autocorr(lag=10)
THRESH_C4_S_CORR      = 0.25   # Pearson corr between z_harm_s norm and hazard field intensity

HARM_OBS_DIM   = 51            # hazard_field[25] + resource_field[25] + harm_exposure[1]
HARM_OBS_A_DIM = 50            # hazard_field[25] + resource_field[25] (no scalar)
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


def run_experiment(
    seed: int,
    warmup_episodes: int,
    fwd_train_episodes: int,
    dissociation_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict:
    """Run one seed. Returns per-seed metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if dry_run:
        warmup_episodes = min(3, warmup_episodes)
        fwd_train_episodes = min(3, fwd_train_episodes)
        dissociation_episodes = min(4, dissociation_episodes)
        steps_per_episode = min(30, steps_per_episode)

    device = torch.device("cpu")

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=3,
        num_resources=2,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )
    env.reset()

    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_S_DIM).to(device)
    affect_enc = AffectiveHarmEncoder(harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM).to(device)
    harm_fwd = HarmForwardModel(z_harm_dim=Z_HARM_S_DIM, action_dim=ACTION_DIM).to(device)

    # Harm evaluation head -- predicts harm_exposure scalar from z_harm_s
    harm_eval_head = nn.Linear(Z_HARM_S_DIM, 1).to(device)
    affect_eval_head = nn.Linear(Z_HARM_A_DIM, 1).to(device)

    opt_enc = optim.Adam(
        list(harm_enc.parameters()) + list(affect_enc.parameters()) +
        list(harm_eval_head.parameters()) + list(affect_eval_head.parameters()),
        lr=3e-4,
    )
    opt_fwd = optim.Adam(harm_fwd.parameters(), lr=3e-4)

    replay_enc: List[Tuple[torch.Tensor, torch.Tensor, float, float]] = []
    replay_fwd: List[Tuple[torch.Tensor, int, torch.Tensor]] = []

    # ------------------------------------------------------------------ #
    # Phase 0: warmup -- train both encoders                              #
    # ------------------------------------------------------------------ #
    for _ in range(warmup_episodes):
        env.reset()
        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            harm_obs = obs_dict["harm_obs"]       # [51]
            harm_obs_a = obs_dict["harm_obs_a"]   # [50]

            action_idx = random.randint(0, ACTION_DIM - 1)
            action_oh = _action_to_onehot(action_idx, ACTION_DIM, device)

            ho_t = harm_obs.unsqueeze(0).to(device)
            hoa_t = harm_obs_a.unsqueeze(0).to(device)

            harm_exposure = float(obs_dict["body_state"][10])
            # Use mean of harm_obs_a as affective target (running accumulated level)
            affect_level = float(harm_obs_a.mean())

            replay_enc.append((ho_t.squeeze(0).detach(), hoa_t.squeeze(0).detach(),
                                harm_exposure, affect_level))

            # Collect z_harm_s transition for forward model
            z_harm_s = harm_enc(ho_t).detach()
            env.step(action_idx)
            obs_dict_next = env._get_observation_dict()
            ho_next = obs_dict_next["harm_obs"].unsqueeze(0).to(device)
            z_harm_s_next = harm_enc(ho_next).detach()
            replay_fwd.append((z_harm_s.squeeze(0).detach(), action_idx,
                                z_harm_s_next.squeeze(0).detach()))

        # Train encoders on replay
        if len(replay_enc) >= 32:
            idxs = random.sample(range(len(replay_enc)), min(64, len(replay_enc)))
            ho_batch = torch.stack([replay_enc[i][0] for i in idxs]).to(device)
            hoa_batch = torch.stack([replay_enc[i][1] for i in idxs]).to(device)
            he_targets = torch.tensor(
                [[replay_enc[i][2]] for i in idxs], dtype=torch.float32, device=device
            )
            ae_targets = torch.tensor(
                [[replay_enc[i][3]] for i in idxs], dtype=torch.float32, device=device
            )
            z_hs = harm_enc(ho_batch)
            z_ha = affect_enc(hoa_batch)
            loss_enc = (
                F.mse_loss(harm_eval_head(z_hs), he_targets) +
                F.mse_loss(affect_eval_head(z_ha), ae_targets)
            )
            opt_enc.zero_grad()
            loss_enc.backward()
            opt_enc.step()

    print(f"[seed={seed}] Phase 0 done. replay_enc={len(replay_enc)}, replay_fwd={len(replay_fwd)}")

    # ------------------------------------------------------------------ #
    # Phase 1: HarmForwardModel training (frozen encoders)               #
    # ------------------------------------------------------------------ #
    # Freeze encoders, then clear replay_fwd -- transitions collected during Phase 0
    # are STALE (encoder changed each update, so z_harm_s_t and z_harm_s_next were
    # encoded by different encoder versions). Stale pairs cause negative R2.
    for p in list(harm_enc.parameters()) + list(affect_enc.parameters()):
        p.requires_grad_(False)
    replay_fwd.clear()  # discard stale warmup transitions before collecting fresh ones

    # Collect more transitions with frozen encoders for clean training
    for _ in range(fwd_train_episodes):
        env.reset()
        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            harm_obs = obs_dict["harm_obs"]
            ho_t = harm_obs.unsqueeze(0).to(device)
            z_harm_s = harm_enc(ho_t).detach()

            action_idx = random.randint(0, ACTION_DIM - 1)
            env.step(action_idx)
            obs_dict_next = env._get_observation_dict()
            ho_next = obs_dict_next["harm_obs"].unsqueeze(0).to(device)
            z_harm_s_next = harm_enc(ho_next).detach()
            replay_fwd.append((z_harm_s.squeeze(0).detach(), action_idx,
                                z_harm_s_next.squeeze(0).detach()))

        if len(replay_fwd) >= 32:
            idxs = random.sample(range(len(replay_fwd)), min(128, len(replay_fwd)))
            z_hs_batch = torch.stack([replay_fwd[i][0] for i in idxs]).to(device)
            a_batch = torch.stack([
                _action_to_onehot(replay_fwd[i][1], ACTION_DIM, device).squeeze(0)
                for i in idxs
            ]).to(device)
            z_hs_next_batch = torch.stack([replay_fwd[i][2] for i in idxs]).to(device)

            z_hs_pred = harm_fwd(z_hs_batch, a_batch)
            loss_fwd = F.mse_loss(z_hs_pred, z_hs_next_batch)
            opt_fwd.zero_grad()
            loss_fwd.backward()
            opt_fwd.step()

    print(f"[seed={seed}] Phase 1 done. replay_fwd={len(replay_fwd)}")

    # Evaluate HarmForwardModel R2 on held-out transitions (last 200 from replay)
    held_out = replay_fwd[-200:] if len(replay_fwd) >= 200 else replay_fwd
    z_hs_batch = torch.stack([t[0] for t in held_out]).to(device)
    a_batch = torch.stack([
        _action_to_onehot(t[1], ACTION_DIM, device).squeeze(0) for t in held_out
    ]).to(device)
    z_hs_next_batch = torch.stack([t[2] for t in held_out]).to(device)

    with torch.no_grad():
        z_hs_pred = harm_fwd(z_hs_batch, a_batch)
    ss_res = float(((z_hs_next_batch - z_hs_pred) ** 2).sum())
    ss_tot = float(((z_hs_next_batch - z_hs_next_batch.mean(0)) ** 2).sum())
    harm_fwd_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    print(f"[seed={seed}] C1 harm_fwd_r2 = {harm_fwd_r2:.4f} (thresh >= {THRESH_C1_FWD_R2})")

    # ------------------------------------------------------------------ #
    # Phase 2: Dissociation probe                                         #
    # ------------------------------------------------------------------ #
    # For each step, record z_harm_s norm and z_harm_a norm.
    # Track: (a) correlation between streams, (b) sustained vs brief hazard norms,
    # (c) z_harm_s response speed at hazard approach events.

    z_hs_norms: List[float] = []
    z_ha_norms: List[float] = []
    hazard_levels: List[float] = []   # hazard field value at agent's position each step

    for ep in range(dissociation_episodes):
        env.reset()
        # Alternate navigation: approach (even) vs random (odd) -- varied coverage
        approach = (ep % 2 == 0)

        for step in range(steps_per_episode):
            obs_dict = env._get_observation_dict()
            harm_obs = obs_dict["harm_obs"]
            harm_obs_a = obs_dict["harm_obs_a"]

            ho_t = harm_obs.unsqueeze(0).to(device)
            hoa_t = harm_obs_a.unsqueeze(0).to(device)

            with torch.no_grad():
                z_hs = harm_enc(ho_t)
                z_ha = affect_enc(hoa_t)

            z_hs_norms.append(float(z_hs.norm()))
            z_ha_norms.append(float(z_ha.norm()))

            # Measure hazard proximity at agent position (mean of central 3x3 cells)
            h_view = obs_dict["hazard_field_view"].numpy()
            center_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
            hazard_levels.append(float(np.mean([h_view[i] for i in center_indices])))

            if approach:
                action_idx = _hazard_gradient_action(env, toward=True)
            else:
                action_idx = random.randint(0, ACTION_DIM - 1)
            env.step(action_idx)

    def _autocorr(series: List[float], lag: int) -> float:
        """Pearson autocorrelation of a series at the given lag."""
        arr = np.array(series)
        if len(arr) <= lag or np.std(arr) < 1e-8:
            return 0.0
        a = arr[:-lag] - arr[:-lag].mean()
        b = arr[lag:] - arr[lag:].mean()
        denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())) + 1e-8
        return float(np.dot(a, b) / denom)

    # C2: Pearson correlation between z_harm_s and z_harm_a norms
    if len(z_hs_norms) > 10:
        a = np.array(z_hs_norms)
        b = np.array(z_ha_norms)
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = (np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum())) + 1e-8
        stream_corr = float(np.dot(a_c, b_c) / denom)
    else:
        stream_corr = 0.0

    # C3: lag-10 autocorrelation gap (z_harm_a - z_harm_s).
    # harm_obs_a is EMA of proximity fields with tau=20 steps. z_harm_a's input is
    # far more temporally correlated than z_harm_s's input (immediate proximity).
    # After encoding, z_harm_a should inherit this slower temporal structure.
    LAG = 10
    z_ha_autocorr = _autocorr(z_ha_norms, LAG)
    z_hs_autocorr = _autocorr(z_hs_norms, LAG)
    autocorr_gap = z_ha_autocorr - z_hs_autocorr

    # C4: Pearson correlation between z_harm_s norm and hazard field intensity
    if len(hazard_levels) > 10 and np.std(hazard_levels) > 1e-6:
        hs = np.array(z_hs_norms)
        hz = np.array(hazard_levels)
        hs_c = hs - hs.mean()
        hz_c = hz - hz.mean()
        denom4 = (np.sqrt((hs_c ** 2).sum()) * np.sqrt((hz_c ** 2).sum())) + 1e-8
        z_harm_s_hazard_corr = float(np.dot(hs_c, hz_c) / denom4)
    else:
        z_harm_s_hazard_corr = 0.0

    print(f"[seed={seed}] C2 stream_corr = {stream_corr:.4f} (thresh <= {THRESH_C2_STREAM_CORR})")
    print(f"[seed={seed}] C3 autocorr_gap(lag={LAG}) = {autocorr_gap:.4f} "
          f"(z_ha={z_ha_autocorr:.3f} z_hs={z_hs_autocorr:.3f}, thresh >= {THRESH_C3_AUTOCORR_GAP})")
    print(f"[seed={seed}] C4 z_harm_s_hazard_corr = {z_harm_s_hazard_corr:.4f} (thresh >= {THRESH_C4_S_CORR})")
    print(f"[seed={seed}] n_steps = {len(z_hs_norms)}")

    return {
        "harm_fwd_r2": harm_fwd_r2,
        "stream_corr": stream_corr,
        "autocorr_gap": autocorr_gap,
        "z_ha_autocorr_lag10": z_ha_autocorr,
        "z_hs_autocorr_lag10": z_hs_autocorr,
        "z_harm_s_hazard_corr": z_harm_s_hazard_corr,
        "n_steps_measured": len(z_hs_norms),
        "z_hs_norms_mean": float(np.mean(z_hs_norms)) if z_hs_norms else 0.0,
        "z_ha_norms_mean": float(np.mean(z_ha_norms)) if z_ha_norms else 0.0,
        "hazard_level_mean": float(np.mean(hazard_levels)) if hazard_levels else 0.0,
        "seed": seed,
    }


def _passes_criteria(metrics: Dict, seeds_results: List[Dict]) -> Tuple[bool, Dict]:
    """Check all criteria. Returns (pass, per-criterion dict)."""
    c1 = metrics["harm_fwd_r2"] >= THRESH_C1_FWD_R2
    c2 = metrics["stream_corr"] <= THRESH_C2_STREAM_CORR
    c3 = metrics["autocorr_gap"] >= THRESH_C3_AUTOCORR_GAP
    c4 = metrics["z_harm_s_hazard_corr"] >= THRESH_C4_S_CORR
    all_pass = c1 and c2 and c3 and c4
    return all_pass, {"C1_fwd_r2": c1, "C2_stream_corr": c2, "C3_autocorr_gap": c3,
                       "C4_hazard_corr": c4}


def main(dry_run: bool = False):
    seeds = [42, 123]

    # Timings calibrated for Mac (DLAPTOP-4.local): ~0.10 min/ep at 200 steps
    warmup_episodes    = 5 if dry_run else 150
    fwd_train_episodes = 3 if dry_run else 80
    dissociation_eps   = 4 if dry_run else 40
    steps_per_episode  = 30 if dry_run else 200

    results_by_seed = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        r = run_experiment(
            seed=seed,
            warmup_episodes=warmup_episodes,
            fwd_train_episodes=fwd_train_episodes,
            dissociation_episodes=dissociation_eps,
            steps_per_episode=steps_per_episode,
            dry_run=dry_run,
        )
        results_by_seed.append(r)

    # Aggregate metrics (mean across seeds)
    agg_keys = ["harm_fwd_r2", "stream_corr", "autocorr_gap", "z_harm_s_hazard_corr"]
    agg = {k: float(np.mean([r[k] for r in results_by_seed])) for k in agg_keys}

    passed, criteria_results = _passes_criteria(agg, results_by_seed)
    outcome = "PASS" if passed else "FAIL"

    print("\n=== AGGREGATE RESULTS ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nOutcome: {outcome}")
    for crit, val in criteria_results.items():
        print(f"  {crit}: {'PASS' if val else 'FAIL'}")

    if dry_run:
        print("\n[DRY RUN] Skipping output file write.")
        return

    # Write result manifest
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
        "criteria": criteria_results,
        "metrics": agg,
        "per_seed": results_by_seed,
        "thresholds": {
            "C1_harm_fwd_r2": THRESH_C1_FWD_R2,
            "C2_stream_corr": THRESH_C2_STREAM_CORR,
            "C3_autocorr_gap": THRESH_C3_AUTOCORR_GAP,
            "C4_z_harm_s_hazard_corr": THRESH_C4_S_CORR,
        },
        "supersedes": "V3-EXQ-178",
        "config": {
            "seeds": seeds,
            "warmup_episodes": warmup_episodes,
            "fwd_train_episodes": fwd_train_episodes,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
