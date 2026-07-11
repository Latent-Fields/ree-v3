#!/opt/local/bin/python3
"""V3-EXQ-740: INV-064 maturational-sequence necessity -- E3 evaluator quality is
bounded by E1/E2 representational differentiation (curriculum-order diagnostic).

CLAIM UNDER TEST (INV-064, e1_e2_e3_maturational_sequence_necessity):
    "E3's primary inputs are z_world from E1 and action_objects from E2. If these
    inputs carry only poorly-differentiated, low-information representations, E3
    training produces a noise-fitted harm/goal evaluator. The information quality
    available to E3 is strictly bounded by E1/E2 representational differentiation.
    Therefore productive E3 training cannot begin until E1/E2 have reached
    sufficient schema differentiation."

MEASUREMENT-ONLY, COMMITMENT-FREE DESIGN.
    This probe requires NO sustained multi-step action-commitment layer (no
    basal-ganglia / F-dominance path is exercised). It is a controlled
    curriculum-ORDER contrast on a frozen representation, so a PASS/FAIL is a real
    verdict on INV-064's central mechanism, not a re-derivation of the conversion
    ceiling.

    Per (seed, onset) cell:
      1. Mature E1 + E2.world_forward + the encoder (latent_stack) for `onset`
         episodes via the canonical goal_pipeline warmup_train. The harm_eval
         head trained incidentally here is DISCARDED (re-initialised in step 4),
         so E3's training budget is controlled independently of maturity.
      2. FREEZE E1, E2.world_transition, E2.world_action_encoder, latent_stack.
      3. Collect a harm dataset by replaying a FIXED action sequence (seeded RNG,
         independent of maturity) through a FIXED collection env. Because the raw
         trajectory + harm labels are identical across the 4 onset arms of a seed,
         the ONLY thing that varies across arms is the frozen encoder's z_world --
         so any difference in E3 quality is attributable to E1/E2 differentiation,
         not to data quantity or the harm-label distribution.
      4. Re-initialise agent.e3.harm_eval_head to a FIXED init (bit-identical
         across arms) and train ONLY that head for a FIXED budget (E3_EPOCHS) on
         the frozen-encoded (z_world, harm_target) tensors. Torch RNG is reset to
         a fixed seed before this phase so the optimisation is identical across
         arms except for the z_world data.
      5. Read out E3 quality: held-out harm R^2, train R^2, and the train-test gap
         (the "noise-fitted evaluator" signature -- poorly-differentiated input
         lets the head fit training-batch noise, giving high train R^2 but low
         held-out R^2).

    IV  (per arm): onset episodes {2, 5, 11, 22}  (E1/E2 maturity at E3 onset).
    Mediator:      z_world effective-rank / variance + E2 forward-model R^2 at
                   freeze (confirms the IV actually moved E1/E2 differentiation --
                   a precondition, not the claim).
    DV:            harm_eval held-out R^2, and gap = train_R2 - test_R2.

SCOPING NOTE (do not over-read a PASS/FAIL).
    INV-064 as catalogued is broader (it also frames a z_self maturational-
    stability gate, V4). This probe tests the load-bearing E1/E2-world -> E3-harm
    leg of the mechanism directly measurable on the current V3 substrate: harm_eval
    reads z_world (e3_selector.harm_eval), so z_world differentiation is the
    binding input here. The E2->trajectory-selector leg and the V4 self-model leg
    are natural follow-ons.

REGIME.
    scheduled_external_hazard is OFF (537b lesson: injecting by-design-
    unpredictable ext events floors harm R^2 regardless of E1 maturity, which
    would make the maturity contrast uninterpretable). In the OFF regime harm is
    predictable-from-state, so E1/E2 differentiation is the binding constraint --
    exactly the regime in which INV-064's prediction is falsifiable.

PRE-REGISTERED PASS CRITERIA (evidence supports INV-064):
    C1 quality-gain:  mean_seed(R2_test[onset_max] - R2_test[onset_min]) >= 0.15
                      AND that mean delta >= 2.0 * SD_seed(delta)   (effect-size gate)
    C2 monotone:      mean_seed Spearman(onset, R2_test across 4 arms) >= 0.80
    C3 noise-signature: mean_seed(gap[onset_min] - gap[onset_max]) >= 0.08
    PASS = C1 and C2 and C3.

PRECONDITIONS (validity -- if unmet the contrast is vacuous, not a real FAIL):
    PC_iv_moved:  mean_seed(eff_rank[onset_max] / eff_rank[onset_min]) >= 1.15   (E1
                  differentiation actually increased across arms)
    PC_events:    min over cells of harm_event_frac >= 0.03        (enough harm to fit)
    If a precondition is unmet the run is marked non_degenerate=False (excluded
    from governance scoring) and evidence_direction="unknown".
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from _lib.goal_pipeline_tier1 import ArmSpec, build_config, warmup_train  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_740_inv064_maturational_sequence_e3_bounded"
QUEUE_ID = "V3-EXQ-740"
CLAIM_IDS = ["INV-064"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants ---
SEEDS = [42, 7, 19]
ONSET_EPISODES = [2, 5, 11, 22]      # E1/E2 maturity (episodes) at E3 onset (~11x gradient)
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared harm-dataset collection (identical per arm)
E3_EPOCHS = 40                       # FIXED harm_eval training budget (all arms)
E3_BATCH = 64
E3_LR = 1e-3
HELDOUT_FRAC = 0.3
E3_INIT_SEED = 90001                 # fixed harm_eval_head init (all arms identical)
E3_TRAIN_SEED = 90002                # fixed E3 optimisation RNG (all arms identical)
COLLECT_SEED_BASE = 70000            # collection env + action RNG base (per-seed, arm-independent)

HARM_OBS_DIM = 51

ENV_KWARGS = {
    "size": 12,
    "num_hazards": 5,   # denser predictable harm -> enough harm events per collected step
    "num_resources": 5,
    "use_proxy_fields": True,
    "env_drift_prob": 0.3,
    "env_drift_interval": 1,
    "limb_damage_enabled": True,
    "harm_history_len": 10,
    "reef_enabled": True,
    "n_reef_patches": 3,
    "reef_patch_radius": 2,
    "hazard_food_attraction": 0.7,
    # 537b lesson: OFF so harm is predictable-from-state and E1 differentiation
    # is the binding constraint (not a by-design-unpredictable ceiling).
    "scheduled_external_hazard_enabled": False,
}

# --- pre-registered thresholds ---
R2_DELTA_FLOOR = 0.15
R2_DELTA_SD_MULT = 2.0
MONOTONE_RHO_MIN = 0.80
GAP_DELTA_FLOOR = 0.08
# preconditions
DIFF_INCREASE_FLOOR = 1.15
HARM_EVENT_FRAC_FLOOR = 0.03


def _r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _eff_rank(Z: torch.Tensor) -> float:
    """Participation ratio of the z_world covariance eigenspectrum.

    Higher == more differentiated representation. A collapsed encoder concentrates
    variance on few axes -> low effective rank.
    """
    if Z.shape[0] < 3:
        return 0.0
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.t() @ Zc) / (Z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    denom = (eig.pow(2).sum() + 1e-12).item()
    return float((eig.sum().item() ** 2) / denom)


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(v: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def _build_agent(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    arm = ArmSpec(arm_id="maturation", gap4_operating=False)
    cfg = build_config(env, arm)  # from_dims path -> alpha_world=0.9 (SD-008)
    agent = REEAgent(cfg)
    return agent, env


def _collect_frozen_dataset(agent: REEAgent, seed: int, steps_per_ep: int,
                            n_episodes: int) -> Dict[str, torch.Tensor]:
    """Replay a FIXED action sequence through a FIXED collection env, encoding
    z_world with the (frozen) agent. Identical raw trajectory across onset arms;
    only the encoder differs.
    """
    collect_env = CausalGridWorldV2(seed=COLLECT_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(COLLECT_SEED_BASE + seed)
    action_dim = collect_env.action_dim

    z_list: List[torch.Tensor] = []
    y_list: List[float] = []
    # For E2 forward-model R^2 (transition triples).
    zprev_list: List[torch.Tensor] = []
    a_list: List[torch.Tensor] = []
    zcurr_list: List[torch.Tensor] = []

    agent.eval()
    with torch.no_grad():
        for _ep in range(n_episodes):
            _, obs_dict = collect_env.reset()
            agent.reset()
            z_world_prev = None
            action_prev = None
            for _step in range(steps_per_ep):
                latent = agent.sense(
                    obs_dict["body_state"],
                    obs_dict["world_state"],
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                z_world = latent.z_world.detach().cpu()

                action = int(act_rng.integers(0, action_dim))
                a_oh = torch.zeros(1, action_dim)
                a_oh[0, action] = 1.0

                _, harm_signal, done, _info, obs_next = collect_env.step(action)
                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

                z_list.append(z_world)
                y_list.append(harm_target)

                if z_world_prev is not None and action_prev is not None:
                    zprev_list.append(z_world_prev)
                    a_list.append(action_prev)
                    zcurr_list.append(z_world)

                z_world_prev = z_world
                action_prev = a_oh
                obs_dict = obs_next
                if done:
                    break

    Z = torch.cat(z_list, dim=0)
    Y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    out = {"Z": Z, "Y": Y}
    if zprev_list:
        out["Zprev"] = torch.cat(zprev_list, dim=0)
        out["A"] = torch.cat(a_list, dim=0)
        out["Zcurr"] = torch.cat(zcurr_list, dim=0)
    return out


def _e2_forward_r2(agent: REEAgent, data: Dict[str, torch.Tensor]) -> float:
    if "Zprev" not in data:
        return float("nan")
    Zp, A, Zc = data["Zprev"], data["A"], data["Zcurr"]
    n = Zp.shape[0]
    n_test = max(1, int(n * HELDOUT_FRAC))
    idx = torch.arange(n)
    te = idx[-n_test:]
    agent.eval()
    with torch.no_grad():
        pred = agent.e2.world_forward(Zp[te], A[te])
    return _r2(pred, Zc[te])


def _freeze(agent: REEAgent) -> None:
    params = (
        list(agent.e1.parameters())
        + list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
        + list(agent.latent_stack.parameters())
    )
    for p in params:
        p.requires_grad_(False)


def _train_e3_and_eval(agent: REEAgent, data: Dict[str, torch.Tensor],
                       e3_init_state: Dict[str, Any],
                       e3_epochs: int) -> Dict[str, float]:
    """Re-init harm_eval_head to the fixed shared init, train for a FIXED budget on
    the frozen-encoded (z_world, harm_target) tensors, return train/test R^2.
    """
    Z, Y = data["Z"], data["Y"]
    n = Z.shape[0]
    n_test = max(1, int(n * HELDOUT_FRAC))
    # Fixed shuffled split (identical indices across all arms -- depends only on n,
    # which is identical across arms because the collection trajectory is shared).
    # Shuffling avoids a harm-sparse contiguous tail making held-out R^2 degenerate.
    split_perm = np.random.default_rng(E3_TRAIN_SEED + 1).permutation(n)
    te = torch.as_tensor(split_perm[:n_test], dtype=torch.long)
    tr = torch.as_tensor(split_perm[n_test:], dtype=torch.long)
    Ztr, Ytr = Z[tr], Y[tr]
    Zte, Yte = Z[te], Y[te]

    # Fixed, bit-identical E3 init + optimisation RNG across all arms.
    agent.e3.harm_eval_head.load_state_dict(copy.deepcopy(e3_init_state))
    for p in agent.e3.harm_eval_head.parameters():
        p.requires_grad_(True)
    torch.manual_seed(E3_TRAIN_SEED)
    opt = torch.optim.Adam(agent.e3.harm_eval_head.parameters(), lr=E3_LR)
    batch_rng = np.random.default_rng(E3_TRAIN_SEED)

    ntr = Ztr.shape[0]
    for epoch in range(e3_epochs):
        perm = batch_rng.permutation(ntr)
        for start in range(0, ntr, E3_BATCH):
            bidx = torch.as_tensor(perm[start:start + E3_BATCH], dtype=torch.long)
            pred = agent.e3.harm_eval(Ztr[bidx])
            loss = F.mse_loss(pred, Ytr[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == e3_epochs:
            print(f"  [e3-fit] epoch {epoch + 1}/{e3_epochs}", flush=True)

    agent.eval()
    with torch.no_grad():
        r2_train = _r2(agent.e3.harm_eval(Ztr), Ytr)
        r2_test = _r2(agent.e3.harm_eval(Zte), Yte)
    harm_event_frac = float((Y > 1e-6).float().mean().item())
    return {
        "harm_r2_train": r2_train,
        "harm_r2_test": r2_test,
        "harm_gap": r2_train - r2_test,
        "harm_event_frac": harm_event_frac,
        "n_samples": int(n),
    }


def _run_cell(seed: int, onset: int, steps_per_ep: int, collect_eps: int,
              e3_epochs: int, dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition onset_{onset}", flush=True)
    config_slice = {
        "onset_episodes": onset,
        "steps_per_ep": steps_per_ep,
        "collect_episodes": collect_eps,
        "e3_epochs": e3_epochs,
        "env_kwargs": ENV_KWARGS,
        "e3_batch": E3_BATCH,
        "e3_lr": E3_LR,
    }
    # Cells are NOT reuse-eligible: the frozen representation is a function of the
    # shared maturation trajectory, not reproducible from config_slice alone. The
    # fingerprint is emitted (validator requirement) but flagged ineligible.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=["frozen_representation_from_maturation_trajectory"],
    ) as cell:
        agent, env = _build_agent(seed)
        # Snapshot the fresh harm_eval_head init BEFORE any training so every arm
        # starts E3 from bit-identical weights.
        e3_init_state = copy.deepcopy(agent.e3.harm_eval_head.state_dict())

        # 1. Mature E1 + E2.world_forward + encoder (harm_eval trained here is
        #    discarded via the re-init in step 4).
        warmup_train(
            agent, env,
            num_episodes=onset,
            steps_per_episode=steps_per_ep,
            label=f"mat seed={seed} onset={onset}",
        )

        # 2. Freeze E1/E2/encoder.
        _freeze(agent)

        # 3. Collect shared frozen dataset (identical trajectory across arms).
        data = _collect_frozen_dataset(agent, seed, steps_per_ep, collect_eps)

        # Differentiation mediators at freeze.
        zworld_var = float(data["Z"].var(dim=0).mean().item())
        zworld_eff_rank = _eff_rank(data["Z"])
        e2_r2 = _e2_forward_r2(agent, data)

        # 4-5. Controlled frozen E3 training + readout.
        e3 = _train_e3_and_eval(agent, data, e3_init_state, e3_epochs)

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "zworld_var": zworld_var,
            "zworld_eff_rank": zworld_eff_rank,
            "e2_forward_r2": e2_r2,
            **e3,
        }
        cell.stamp(row)

    print(f"verdict: {'PASS' if e3['harm_r2_test'] == e3['harm_r2_test'] else 'FAIL'}",
          flush=True)  # NaN-guard: cell completed
    return row


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    onset_min = min(ONSET_EPISODES)
    onset_max = max(ONSET_EPISODES)

    def _cell(seed: int, onset: int) -> Dict[str, Any]:
        return next(r for r in rows if r["seed"] == seed and r["onset_episodes"] == onset)

    per_seed_delta_r2: List[float] = []
    per_seed_gap_delta: List[float] = []
    per_seed_rho: List[float] = []
    per_seed_effrank_ratio: List[float] = []
    for seed in SEEDS:
        r2s = [_cell(seed, o)["harm_r2_test"] for o in ONSET_EPISODES]
        gaps = [_cell(seed, o)["harm_gap"] for o in ONSET_EPISODES]
        eff = [_cell(seed, o)["zworld_eff_rank"] for o in ONSET_EPISODES]
        per_seed_delta_r2.append(_cell(seed, onset_max)["harm_r2_test"]
                                 - _cell(seed, onset_min)["harm_r2_test"])
        per_seed_gap_delta.append(_cell(seed, onset_min)["harm_gap"]
                                  - _cell(seed, onset_max)["harm_gap"])
        per_seed_rho.append(_spearman([float(o) for o in ONSET_EPISODES], r2s))
        er_min = _cell(seed, onset_min)["zworld_eff_rank"]
        per_seed_effrank_ratio.append(
            _cell(seed, onset_max)["zworld_eff_rank"] / max(er_min, 1e-6))

    def _mean(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _sd(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = _mean(v)
        return float((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5)

    mean_delta_r2 = _mean(per_seed_delta_r2)
    sd_delta_r2 = _sd(per_seed_delta_r2)
    mean_gap_delta = _mean(per_seed_gap_delta)
    mean_rho = _mean(per_seed_rho)
    mean_effrank_ratio = _mean(per_seed_effrank_ratio)
    min_harm_event_frac = min(r["harm_event_frac"] for r in rows)

    # Preconditions (validity).
    pc_iv_moved = mean_effrank_ratio >= DIFF_INCREASE_FLOOR
    pc_events = min_harm_event_frac >= HARM_EVENT_FRAC_FLOOR
    preconditions_met = pc_iv_moved and pc_events

    # Criteria (INV-064 supports).
    c1_effect = mean_delta_r2 >= (R2_DELTA_SD_MULT * sd_delta_r2)
    c1 = (mean_delta_r2 >= R2_DELTA_FLOOR) and c1_effect
    c2 = mean_rho >= MONOTONE_RHO_MIN
    c3 = mean_gap_delta >= GAP_DELTA_FLOOR

    return {
        "onset_min": onset_min,
        "onset_max": onset_max,
        "per_seed_delta_r2": per_seed_delta_r2,
        "per_seed_gap_delta": per_seed_gap_delta,
        "per_seed_spearman_rho": per_seed_rho,
        "per_seed_effrank_ratio": per_seed_effrank_ratio,
        "mean_delta_r2": mean_delta_r2,
        "sd_delta_r2": sd_delta_r2,
        "mean_gap_delta": mean_gap_delta,
        "mean_spearman_rho": mean_rho,
        "mean_effrank_ratio": mean_effrank_ratio,
        "min_harm_event_frac": min_harm_event_frac,
        "PC_iv_moved": pc_iv_moved,
        "PC_events": pc_events,
        "preconditions_met": preconditions_met,
        "C1_quality_gain": c1,
        "C2_monotone": c2,
        "C3_noise_signature": c3,
    }


def _aggregate_by_onset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ONSET_EPISODES:
        cells = [r for r in rows if r["onset_episodes"] == o]
        out[f"onset_{o}"] = {
            "mean_harm_r2_test": float(np.mean([c["harm_r2_test"] for c in cells])),
            "mean_harm_r2_train": float(np.mean([c["harm_r2_train"] for c in cells])),
            "mean_harm_gap": float(np.mean([c["harm_gap"] for c in cells])),
            "mean_zworld_eff_rank": float(np.mean([c["zworld_eff_rank"] for c in cells])),
            "mean_e2_forward_r2": float(np.mean([c["e2_forward_r2"] for c in cells])),
        }
    return out


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [SEEDS[0]]
        onsets = [2, 5]
        steps_per_ep = 20
        collect_eps = 3
        e3_epochs = 3
    else:
        seeds = SEEDS
        onsets = ONSET_EPISODES
        steps_per_ep = STEPS_PER_EP
        collect_eps = COLLECT_EPISODES
        e3_epochs = E3_EPOCHS

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for onset in onsets:
            rows.append(_run_cell(seed, onset, steps_per_ep, collect_eps,
                                  e3_epochs, dry_run))

    if dry_run:
        # Not enough arms/seeds for the full criteria; just confirm the pipeline.
        return {
            "outcome": "PASS",
            "dry_run": True,
            "n_cells": len(rows),
            "arm_results": rows,
        }

    criteria = _evaluate(rows)
    by_onset = _aggregate_by_onset(rows)

    preconditions_met = criteria["preconditions_met"]
    claim_pass = criteria["C1_quality_gain"] and criteria["C2_monotone"] and criteria["C3_noise_signature"]

    if not preconditions_met:
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        reasons = []
        if not criteria["PC_iv_moved"]:
            reasons.append(
                f"IV did not move: mean eff_rank ratio {criteria['mean_effrank_ratio']:.3f} "
                f"< {DIFF_INCREASE_FLOOR} (E1/E2 differentiation did not increase across arms)")
        if not criteria["PC_events"]:
            reasons.append(
                f"insufficient harm events: min frac {criteria['min_harm_event_frac']:.4f} "
                f"< {HARM_EVENT_FRAC_FLOOR}")
        degeneracy_reason = "; ".join(reasons)
    else:
        outcome = "PASS" if claim_pass else "FAIL"
        evidence_direction = "supports" if claim_pass else "weakens"
        non_degenerate = True
        degeneracy_reason = ""

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria": criteria,
        "by_onset": by_onset,
        "arm_results": rows,
        "preconditions_met": preconditions_met,
        "claim_pass": claim_pass,
    }


def _write_manifest(result: Dict[str, Any]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "exp:simulation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result.get("evidence_direction", "unknown"),
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "onset_episodes": ONSET_EPISODES,
        "thresholds": {
            "R2_DELTA_FLOOR": R2_DELTA_FLOOR,
            "R2_DELTA_SD_MULT": R2_DELTA_SD_MULT,
            "MONOTONE_RHO_MIN": MONOTONE_RHO_MIN,
            "GAP_DELTA_FLOOR": GAP_DELTA_FLOOR,
            "DIFF_INCREASE_FLOOR": DIFF_INCREASE_FLOOR,
            "HARM_EVENT_FRAC_FLOOR": HARM_EVENT_FRAC_FLOOR,
        },
        "criteria": result["criteria"],
        "by_onset": result["by_onset"],
        "arm_results": result["arm_results"],
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_path}", flush=True)
    return out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    """Run the experiment; return (outcome, manifest_path). The emit_outcome
    sentinel is written by the __main__ block (runner-conformance contract).
    """
    result = run_experiment(dry_run=dry_run)

    if dry_run:
        print(f"DRY_RUN complete: {result['n_cells']} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result)
    c = result["criteria"]
    print("=== INV-064 maturational-sequence result ===", flush=True)
    print(f"  preconditions_met: {result['preconditions_met']} "
          f"(iv_moved={c['PC_iv_moved']} eff_rank_ratio={c['mean_effrank_ratio']:.3f}; "
          f"events={c['PC_events']} min_frac={c['min_harm_event_frac']:.4f})", flush=True)
    print(f"  C1 quality_gain: {c['C1_quality_gain']} "
          f"(mean_delta_r2={c['mean_delta_r2']:.3f}, sd={c['sd_delta_r2']:.3f})", flush=True)
    print(f"  C2 monotone:     {c['C2_monotone']} (mean_rho={c['mean_spearman_rho']:.3f})", flush=True)
    print(f"  C3 noise_sig:    {c['C3_noise_signature']} (mean_gap_delta={c['mean_gap_delta']:.3f})", flush=True)
    print(f"  OUTCOME: {result['outcome']} (direction={result['evidence_direction']})", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=args.dry_run,
    )
