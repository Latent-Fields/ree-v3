#!/opt/local/bin/python3
"""V3-EXQ-744a: INV-088 world/goal-evaluator DV-coupling -- higher-seed (3 -> 8)
RE-ESTIMATE of V3-EXQ-744, built on the shared maturation-curriculum module +
frozen-prefix tensor cache. EXTENDS (does not supersede) V3-EXQ-744 and V3-EXQ-740a.

WHY THIS RUN (the 744 result it responds to):
    V3-EXQ-744 landed FAIL / inconclusive (non_degenerate=True). The DV-coupling
    MAGNITUDE cleared its floor -- mean_seed(R2_test[onset_max] - R2_test[onset_min])
    = 0.232 >= 0.15 -- but with only 3 seeds the per-seed SD was 0.168, so C1's
    EFFECT-SIZE gate (mean_delta >= 2.0 * SD_seed) was 0.232 >= 0.336 = FALSE, and C2
    (monotone) also failed; C3 (noise-fit floor) passed. So the coupling looks real
    but the 3-seed variance was too high to clear the pre-registered effect-size gate.
    This run RE-ESTIMATES at 8 seeds with the IDENTICAL pre-registered criteria to
    resolve that inconclusive: a higher-n estimate tightens both mean and SD estimates.
    HONEST SCOPE: more seeds does not shrink the TRUE per-seed SD, so if the coupling's
    per-seed coefficient of variation is intrinsically ~0.7 the 2*SD gate may still not
    clear -- in which case 744a is the definitive higher-n read that the INV-088 DV
    coupling is real-but-variable rather than a clean threshold effect. Either way it
    is a decision-quality result, not a re-run hoping for a flip.

WHY THE SHARED MODULE (the reuse lever this run mints):
    740a (INV-064 IV leg), 744 (INV-088 DV coupling) and 743 (INV-089 z_harm leg) each
    independently ran an expensive maturation prefix -- warmup_train(onset) + a FIXED
    frozen-dataset collection -- then diverged only in a cheap tail. The prefix is a
    DETERMINISTIC pure function of (substrate, env_kwargs, recipe, seed, onset) within a
    machine_class (verified bit-identical 2026-07-12), so the earlier
    `frozen_representation_from_maturation_trajectory` reuse-ineligibility flag was
    EMPIRICALLY FALSE and is dropped here. This run builds its frozen z_world prefix via
    experiments/_lib/baselines/maturation_curriculum.mature_and_collect_world, which
    (a) makes the whole world leg share ONE recipe so sibling arm_fingerprints match by
    construction, and (b) MINTS a frozen-prefix TENSOR cache (frozen agent state_dict +
    dataset tensors) keyed on upstream-only params. The bit-identity test
    (test_maturation_bitidentity) proves the module reproduces 744/740a inline exactly.

    CACHE-MINT SCOPE NOTE: 744 ran INLINE and did NOT populate the tensor cache, so on
    744a's FIRST run its 3 original seeds are COLD (cache miss + store) just like the 5
    new seeds -- 744a MINTS all 40 (8 seed x 5 onset) prefix cells. The tensor-reuse
    saving is realized by a LATER world-leg sibling (or a 744a re-run) that shares a
    (seed, onset, env_kwargs, recipe) cell, per the mint-as-you-go default. The cells
    are ALSO emitted arm_fingerprint reuse-ELIGIBLE (scalar whole-cell reuse, a distinct
    mechanism) with include_driver_script_in_hash=False so a differently-driven sibling
    can cite them.

REPLICATION FIDELITY: because the module reproduces 744's prefix bit-identically and the
    tail (ridge probes + evaluator-head training) is byte-identical to 744, 744a's rows
    at seeds {42, 7, 19} reproduce 744's rows exactly; 744a merely ADDS 5 seeds.

CLAIM / DESIGN / CRITERIA are IDENTICAL to V3-EXQ-744 (INV-088,
world_goal_evaluator_bounded_by_z_world_differentiation, child (i) of INV-064). See the
V3-EXQ-744 docstring for the full claim text, the corrected-DV-target rationale (the
740a harm leg mis-streamed; the world/goal-feature target IS decodable from z_world),
the measurement-only / commitment-free design, and the regime (scheduled_external_hazard
OFF). Only the seed count and the module-based prefix differ.

    PRIMARY DV (load-bearing): agent.e3.harm_eval_head (SD-003), re-init to a fixed init,
        trained a FIXED budget on FROZEN z_world[t] -> next-step hazard-proximity world
        feature (mean hazard_field_view of harm_obs[t+1], normalised to [0,1]).
    SECONDARY DV (corroborating, non-gating): agent.e3.benefit_eval_head -> next-step
        resource-proximity world feature.
    IV OBSERVABLE (from 740a): world_feat_decode_r2 = held-out ridge R2 z_world[t] ->
        harm_obs[t+1] (predictive, JL-safe, rises with maturation).

PRE-REGISTERED PASS CRITERIA (identical to 744; on the PRIMARY harm_eval SD-003 DV):
    C1 quality-gain:  mean_seed(R2_test[onset_max] - R2_test[onset_min]) >= 0.15
                      AND that mean delta >= 2.0 * SD_seed(delta)   (effect-size gate)
    C2 monotone:      mean_seed Spearman(onset, R2_test across arms) >= 0.80
    C3 noise-fit floor: mean_seed(R2_test[onset_min]) <= 0.05
    PASS = C1 and C2 and C3.

PRECONDITIONS (validity -- identical to 744):
    PC_iv_moved:     mean_seed(world_feat_decode_r2[onset_max] - [onset_min]) >= 0.03 AND > 0
    PC_dv_decodable: mean_seed(hazard_feat_decode_r2[onset_max]) >= 0.05
    PC_target_var:   min over cells of the primary DV target std >= 0.02
    If a precondition is unmet the run is non_degenerate=False, evidence_direction="unknown".
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.baselines.maturation_curriculum import mature_and_collect_world  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_744a_inv088_world_goal_evaluator_dv_coupling"
QUEUE_ID = "V3-EXQ-744a"
SUPERSEDES = None  # EXTENDS 744 (higher-n re-estimate, same claim/criteria); does not supersede
CLAIM_IDS = ["INV-088"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants (mirror 744; seeds 3 -> 8) ---
SEEDS = [42, 7, 19, 3, 11, 23, 47, 101]     # first three reproduce 744 bit-identically
ONSET_EPISODES = [0, 1, 4, 12, 30]          # onset_0 = fresh encoder (immature anchor)
WARMUP_PROGRESS_DENOM = max(ONSET_EPISODES)  # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared frozen-dataset collection (identical per arm)
EVAL_EPOCHS = 40                     # FIXED evaluator-head training budget (all arms/heads)
EVAL_BATCH = 64
EVAL_LR = 1e-3
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear decode probes
EVAL_INIT_SEED = 90001              # (documentation) fixed head init is arm_cell-seeded
EVAL_TRAIN_SEED = 90002            # fixed evaluator optimisation RNG (all arms/heads identical)
DECODE_SPLIT_SEED = 90003          # fixed decode-probe train/test split (all arms identical)
COLLECT_SEED_BASE = 70000          # collection env + action RNG base (per-seed, arm-independent)

HARM_OBS_DIM = 51
# harm_obs [51] = hazard_field_view[0:25] + resource_field_view[25:50] + harm_exposure[50]
HAZARD_SLICE = (0, 25)             # harm-relevant proximity world features (SD-003 domain)
RESOURCE_SLICE = (25, 50)          # goal/benefit-relevant proximity world features

ENV_KWARGS = {
    "size": 12,
    "num_hazards": 5,
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
    # 537b lesson: OFF so world features are predictable-from-state and E1
    # differentiation is the binding constraint (not a by-design-unpredictable ceiling).
    "scheduled_external_hazard_enabled": False,
}

# --- pre-registered thresholds (identical to 744) ---
R2_DELTA_FLOOR = 0.15
R2_DELTA_SD_MULT = 2.0
MONOTONE_RHO_MIN = 0.80
NOISE_FIT_CEIL = 0.05          # immature-anchor evaluator held-out R2 must be <= this (noise-fitted)
# preconditions
IV_MOVE_FLOOR = 0.03            # min predictive-decodability increase onset_min->onset_max
DV_DECODABLE_FLOOR = 0.05      # mature-anchor primary DV-target decodability positive control
TARGET_STD_FLOOR = 0.02        # primary DV target must not be near-constant


def _r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _eff_rank(Z: torch.Tensor) -> float:
    """Participation ratio of the z_world covariance eigenspectrum. Diagnostic only."""
    if Z.shape[0] < 3:
        return 0.0
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.t() @ Zc) / (Z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    denom = (eig.pow(2).sum() + 1e-12).item()
    return float((eig.sum().item() ** 2) / denom)


def _ridge_heldout_r2(Z: torch.Tensor, T: torch.Tensor, split_seed: int,
                      lam: float = RIDGE_LAMBDA) -> float:
    """Held-out R^2 of a closed-form ridge linear probe Z -> T (fixed split; pooled over
    output dims). Pure read-out of the FROZEN representation (no gradients)."""
    n = Z.shape[0]
    if n < 4:
        return float("nan")
    if T.dim() == 1:
        T = T.unsqueeze(1)
    n_test = max(1, int(n * HELDOUT_FRAC))
    perm = np.random.default_rng(split_seed).permutation(n)
    te = torch.as_tensor(perm[:n_test], dtype=torch.long)
    tr = torch.as_tensor(perm[n_test:], dtype=torch.long)
    Ztr, Ttr = Z[tr], T[tr]
    Zte, Tte = Z[te], T[te]
    Xtr = torch.cat([Ztr, torch.ones(Ztr.shape[0], 1)], dim=1)
    Xte = torch.cat([Zte, torch.ones(Zte.shape[0], 1)], dim=1)
    d = Xtr.shape[1]
    reg = lam * torch.eye(d)
    reg[-1, -1] = 0.0  # do not regularise the bias term
    A = Xtr.t() @ Xtr + reg
    B = Xtr.t() @ Ttr
    W = torch.linalg.solve(A, B)
    pred = Xte @ W
    ss_res = ((pred - Tte) ** 2).sum().item()
    mu = Ttr.mean(dim=0, keepdim=True)
    ss_tot = ((Tte - mu) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


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


def _next_step_feature(Hnext: torch.Tensor, slc: Tuple[int, int]) -> torch.Tensor:
    """Mean proximity over a world-feature block of harm_obs[t+1] -> [n, 1] scalar target.
    NEXT-STEP (predictive) -> JL-safe. Dataset-normalised to [0,1] (min-max over the
    shared trajectory, identical across the onset arms of a seed)."""
    a, b = slc
    feat = Hnext[:, a:b].mean(dim=1, keepdim=True)
    fmin = feat.min()
    fmax = feat.max()
    denom = (fmax - fmin).clamp(min=1e-6)
    return ((feat - fmin) / denom).to(torch.float32)


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


def _split_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed shuffled train/test split -- identical indices across all arms/heads
    (depends only on n, which is identical across arms because the trajectory is shared)."""
    n_test = max(1, int(n * HELDOUT_FRAC))
    perm = np.random.default_rng(EVAL_TRAIN_SEED + 1).permutation(n)
    te = torch.as_tensor(perm[:n_test], dtype=torch.long)
    tr = torch.as_tensor(perm[n_test:], dtype=torch.long)
    return tr, te


def _train_eval_head(head: torch.nn.Module, init_state: Dict[str, Any],
                     Z: torch.Tensor, T: torch.Tensor,
                     eval_epochs: int, head_label: str) -> Dict[str, float]:
    """Re-init `head` to `init_state`, train ONLY that head for a FIXED budget on the
    frozen (z_world, next-step world-feature) tensors, return train/test R^2 + gap.
    Fixed, bit-identical init + optimisation RNG + split across all arms/heads."""
    n = Z.shape[0]
    tr, te = _split_indices(n)
    Ztr, Ttr = Z[tr], T[tr]
    Zte, Tte = Z[te], T[te]

    head.load_state_dict(copy.deepcopy(init_state))
    for p in head.parameters():
        p.requires_grad_(True)
    torch.manual_seed(EVAL_TRAIN_SEED)
    opt = torch.optim.Adam(head.parameters(), lr=EVAL_LR)
    batch_rng = np.random.default_rng(EVAL_TRAIN_SEED)

    ntr = Ztr.shape[0]
    for epoch in range(eval_epochs):
        perm = batch_rng.permutation(ntr)
        for start in range(0, ntr, EVAL_BATCH):
            bidx = torch.as_tensor(perm[start:start + EVAL_BATCH], dtype=torch.long)
            pred = head(Ztr[bidx])
            loss = F.mse_loss(pred, Ttr[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == eval_epochs:
            print(f"  [eval-fit:{head_label}] epoch {epoch + 1}/{eval_epochs}", flush=True)

    head.eval()
    with torch.no_grad():
        r2_train = _r2(head(Ztr), Ttr)
        r2_test = _r2(head(Zte), Tte)
    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "gap": r2_train - r2_test,
    }


def _run_cell(seed: int, onset: int, steps_per_ep: int, collect_eps: int,
              eval_epochs: int, dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition onset_{onset}", flush=True)
    config_slice = {
        "onset_episodes": onset,
        "steps_per_ep": steps_per_ep,
        "collect_episodes": collect_eps,
        "eval_epochs": eval_epochs,
        "env_kwargs": ENV_KWARGS,
        "eval_batch": EVAL_BATCH,
        "eval_lr": EVAL_LR,
        "ridge_lambda": RIDGE_LAMBDA,
    }
    # Cells ARE reuse-eligible: the frozen prefix is a DETERMINISTIC pure function of
    # (substrate, config_slice, seed) -- verified bit-identical (the earlier
    # frozen_representation_from_maturation_trajectory ineligible reason was FALSE).
    # include_driver_script_in_hash=False anchors the fingerprint on the shared
    # maturation_curriculum module (in the _lib substrate glob) so a differently-driven
    # sibling can cite these cells.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        # Frozen z_world prefix via the shared module (mint-as-you-go into the
        # frozen-prefix tensor cache). Called FIRST inside arm_cell (per the module's
        # bit-identity contract), so the fresh head inits + dataset match 744 exactly.
        agent, fresh_heads, data, prov = mature_and_collect_world(
            seed, onset,
            env_kwargs=ENV_KWARGS,
            steps_per_ep=steps_per_ep,
            collect_episodes=collect_eps,
            collect_seed_base=COLLECT_SEED_BASE,
            warmup_progress_denom=WARMUP_PROGRESS_DENOM,
        )
        Z = data["Z"]

        # DV targets: NEXT-STEP world/goal-feature scalars (JL-safe, decodable leg).
        hazard_next = _next_step_feature(data["Hnext"], HAZARD_SLICE)       # PRIMARY (SD-003)
        resource_next = _next_step_feature(data["Hnext"], RESOURCE_SLICE)   # SECONDARY (goal)

        # Differentiation mediators at freeze.
        zworld_var = float(Z.var(dim=0).mean().item())
        zworld_eff_rank = _eff_rank(Z)                 # diagnostic only
        e2_r2 = _e2_forward_r2(agent, data)
        # IV (reused from 740a): predictive full harm_obs decodability.
        world_feat_decode_r2 = _ridge_heldout_r2(Z, data["Hnext"], DECODE_SPLIT_SEED)
        # Positive controls: are the DV targets decodable-in-principle from z_world?
        hazard_feat_decode_r2 = _ridge_heldout_r2(Z, hazard_next, DECODE_SPLIT_SEED)   # PRIMARY PC
        resource_feat_decode_r2 = _ridge_heldout_r2(Z, resource_next, DECODE_SPLIT_SEED)  # advisory

        # Controlled frozen evaluator training + readout (both real substrate heads),
        # re-init to the FRESH pre-maturation inits the module captured.
        harm_eval = _train_eval_head(agent.e3.harm_eval_head, fresh_heads["harm_eval_head"],
                                     Z, hazard_next, eval_epochs, "harm_sd003")
        benefit_eval = _train_eval_head(agent.e3.benefit_eval_head, fresh_heads["benefit_eval_head"],
                                        Z, resource_next, eval_epochs, "benefit")

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "zworld_var": zworld_var,
            "zworld_eff_rank": zworld_eff_rank,
            "e2_forward_r2": e2_r2,
            "world_feat_decode_r2": world_feat_decode_r2,
            "hazard_feat_decode_r2": hazard_feat_decode_r2,
            "resource_feat_decode_r2": resource_feat_decode_r2,
            "hazard_target_std": float(hazard_next.std().item()),
            "resource_target_std": float(resource_next.std().item()),
            "n_samples": int(Z.shape[0]),
            "prefix_cache": prov["cache"],   # provenance: hit | miss | disabled
            # PRIMARY DV (harm_eval SD-003 on next-step hazard world feature)
            "harm_eval_r2_test": harm_eval["r2_test"],
            "harm_eval_r2_train": harm_eval["r2_train"],
            "harm_eval_gap": harm_eval["gap"],
            # SECONDARY DV (benefit_eval on next-step resource world feature)
            "benefit_eval_r2_test": benefit_eval["r2_test"],
            "benefit_eval_r2_train": benefit_eval["r2_train"],
            "benefit_eval_gap": benefit_eval["gap"],
        }
        cell.stamp(row)

    print(f"verdict: {'PASS' if row['harm_eval_r2_test'] == row['harm_eval_r2_test'] else 'FAIL'}",
          flush=True)  # NaN-guard: cell completed
    return row


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    onset_min = min(ONSET_EPISODES)
    onset_max = max(ONSET_EPISODES)

    def _cell(seed: int, onset: int) -> Dict[str, Any]:
        return next(r for r in rows if r["seed"] == seed and r["onset_episodes"] == onset)

    per_seed_delta_r2: List[float] = []
    per_seed_immature_r2: List[float] = []
    per_seed_gap_delta: List[float] = []   # diagnostic only (see C3 note)
    per_seed_rho: List[float] = []
    per_seed_iv_delta: List[float] = []
    per_seed_dv_decode_mature: List[float] = []
    per_seed_benefit_delta: List[float] = []
    per_seed_benefit_rho: List[float] = []
    for seed in SEEDS:
        r2s = [_cell(seed, o)["harm_eval_r2_test"] for o in ONSET_EPISODES]
        per_seed_delta_r2.append(_cell(seed, onset_max)["harm_eval_r2_test"]
                                 - _cell(seed, onset_min)["harm_eval_r2_test"])
        per_seed_immature_r2.append(_cell(seed, onset_min)["harm_eval_r2_test"])
        per_seed_gap_delta.append(_cell(seed, onset_min)["harm_eval_gap"]
                                  - _cell(seed, onset_max)["harm_eval_gap"])
        per_seed_rho.append(_spearman([float(o) for o in ONSET_EPISODES], r2s))
        per_seed_iv_delta.append(_cell(seed, onset_max)["world_feat_decode_r2"]
                                 - _cell(seed, onset_min)["world_feat_decode_r2"])
        per_seed_dv_decode_mature.append(_cell(seed, onset_max)["hazard_feat_decode_r2"])
        b2s = [_cell(seed, o)["benefit_eval_r2_test"] for o in ONSET_EPISODES]
        per_seed_benefit_delta.append(_cell(seed, onset_max)["benefit_eval_r2_test"]
                                      - _cell(seed, onset_min)["benefit_eval_r2_test"])
        per_seed_benefit_rho.append(_spearman([float(o) for o in ONSET_EPISODES], b2s))

    def _mean(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _sd(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = _mean(v)
        return float((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5)

    mean_delta_r2 = _mean(per_seed_delta_r2)
    sd_delta_r2 = _sd(per_seed_delta_r2)
    mean_immature_r2 = _mean(per_seed_immature_r2)
    mean_gap_delta = _mean(per_seed_gap_delta)   # diagnostic only
    mean_rho = _mean(per_seed_rho)
    mean_iv_delta = _mean(per_seed_iv_delta)
    mean_dv_decode_mature = _mean(per_seed_dv_decode_mature)
    min_target_std = min(r["hazard_target_std"] for r in rows)

    mean_benefit_delta = _mean(per_seed_benefit_delta)
    mean_benefit_rho = _mean(per_seed_benefit_rho)
    benefit_corroborates = (mean_benefit_delta >= R2_DELTA_FLOOR) and (mean_benefit_rho >= MONOTONE_RHO_MIN)

    # Preconditions (validity).
    pc_iv_moved = (mean_iv_delta >= IV_MOVE_FLOOR) and (mean_iv_delta > 0.0)
    pc_dv_decodable = mean_dv_decode_mature >= DV_DECODABLE_FLOOR
    pc_target_var = min_target_std >= TARGET_STD_FLOOR
    preconditions_met = pc_iv_moved and pc_dv_decodable and pc_target_var

    # Criteria (INV-088 supports) -- on the PRIMARY harm_eval SD-003 DV.
    c1_effect = mean_delta_r2 >= (R2_DELTA_SD_MULT * sd_delta_r2)
    c1 = (mean_delta_r2 >= R2_DELTA_FLOOR) and c1_effect
    c2 = mean_rho >= MONOTONE_RHO_MIN
    c3 = mean_immature_r2 <= NOISE_FIT_CEIL

    return {
        "onset_min": onset_min,
        "onset_max": onset_max,
        "per_seed_delta_r2": per_seed_delta_r2,
        "per_seed_immature_r2": per_seed_immature_r2,
        "per_seed_gap_delta": per_seed_gap_delta,
        "per_seed_spearman_rho": per_seed_rho,
        "per_seed_iv_delta": per_seed_iv_delta,
        "per_seed_dv_decode_mature": per_seed_dv_decode_mature,
        "per_seed_benefit_delta": per_seed_benefit_delta,
        "per_seed_benefit_rho": per_seed_benefit_rho,
        "mean_delta_r2": mean_delta_r2,
        "sd_delta_r2": sd_delta_r2,
        "mean_immature_r2": mean_immature_r2,
        "mean_gap_delta": mean_gap_delta,
        "mean_spearman_rho": mean_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_dv_decode_mature": mean_dv_decode_mature,
        "min_target_std": min_target_std,
        "mean_benefit_delta": mean_benefit_delta,
        "mean_benefit_rho": mean_benefit_rho,
        "benefit_corroborates": benefit_corroborates,
        "PC_iv_moved": pc_iv_moved,
        "PC_dv_decodable": pc_dv_decodable,
        "PC_target_var": pc_target_var,
        "preconditions_met": preconditions_met,
        "C1_quality_gain": c1,
        "C2_monotone": c2,
        "C3_noise_fit_floor": c3,
    }


def _aggregate_by_onset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ONSET_EPISODES:
        cells = [r for r in rows if r["onset_episodes"] == o]
        out[f"onset_{o}"] = {
            "mean_harm_eval_r2_test": float(np.mean([c["harm_eval_r2_test"] for c in cells])),
            "mean_harm_eval_r2_train": float(np.mean([c["harm_eval_r2_train"] for c in cells])),
            "mean_harm_eval_gap": float(np.mean([c["harm_eval_gap"] for c in cells])),
            "mean_benefit_eval_r2_test": float(np.mean([c["benefit_eval_r2_test"] for c in cells])),
            "mean_benefit_eval_gap": float(np.mean([c["benefit_eval_gap"] for c in cells])),
            "mean_world_feat_decode_r2": float(np.mean([c["world_feat_decode_r2"] for c in cells])),
            "mean_hazard_feat_decode_r2": float(np.mean([c["hazard_feat_decode_r2"] for c in cells])),
            "mean_resource_feat_decode_r2": float(np.mean([c["resource_feat_decode_r2"] for c in cells])),
            "mean_e2_forward_r2": float(np.mean([c["e2_forward_r2"] for c in cells])),
            "mean_zworld_eff_rank": float(np.mean([c["zworld_eff_rank"] for c in cells])),
        }
    return out


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [SEEDS[0]]
        onsets = [0, 2]
        steps_per_ep = 20
        collect_eps = 3
        eval_epochs = 3
    else:
        seeds = SEEDS
        onsets = ONSET_EPISODES
        steps_per_ep = STEPS_PER_EP
        collect_eps = COLLECT_EPISODES
        eval_epochs = EVAL_EPOCHS

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for onset in onsets:
            rows.append(_run_cell(seed, onset, steps_per_ep, collect_eps,
                                  eval_epochs, dry_run))

    if dry_run:
        return {
            "outcome": "PASS",
            "dry_run": True,
            "n_cells": len(rows),
            "arm_results": rows,
        }

    criteria = _evaluate(rows)
    by_onset = _aggregate_by_onset(rows)

    preconditions_met = criteria["preconditions_met"]
    claim_pass = criteria["C1_quality_gain"] and criteria["C2_monotone"] and criteria["C3_noise_fit_floor"]

    if not preconditions_met:
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        reasons = []
        if not criteria["PC_iv_moved"]:
            reasons.append(
                f"IV did not move in the predicted direction: mean predictive-decode "
                f"delta {criteria['mean_iv_delta']:.3f} < {IV_MOVE_FLOOR} or <= 0")
        if not criteria["PC_dv_decodable"]:
            reasons.append(
                f"primary DV target not decodable-in-principle: mean mature-anchor "
                f"hazard_feat_decode_r2 {criteria['mean_dv_decode_mature']:.3f} < {DV_DECODABLE_FLOOR}")
        if not criteria["PC_target_var"]:
            reasons.append(
                f"primary DV target near-constant: min target std "
                f"{criteria['min_target_std']:.4f} < {TARGET_STD_FLOOR}")
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


def _write_manifest(result: Dict[str, Any], started_at: float) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    cache_states = [r.get("prefix_cache") for r in result["arm_results"]]
    full_config = {
        "seeds": SEEDS,
        "onset_episodes": ONSET_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "collect_episodes": COLLECT_EPISODES,
        "eval_epochs": EVAL_EPOCHS,
        "eval_batch": EVAL_BATCH,
        "eval_lr": EVAL_LR,
        "heldout_frac": HELDOUT_FRAC,
        "ridge_lambda": RIDGE_LAMBDA,
        "collect_seed_base": COLLECT_SEED_BASE,
        "eval_train_seed": EVAL_TRAIN_SEED,
        "decode_split_seed": DECODE_SPLIT_SEED,
        "env_kwargs": ENV_KWARGS,
        "thresholds": {
            "R2_DELTA_FLOOR": R2_DELTA_FLOOR,
            "R2_DELTA_SD_MULT": R2_DELTA_SD_MULT,
            "MONOTONE_RHO_MIN": MONOTONE_RHO_MIN,
            "NOISE_FIT_CEIL": NOISE_FIT_CEIL,
            "IV_MOVE_FLOOR": IV_MOVE_FLOOR,
            "DV_DECODABLE_FLOOR": DV_DECODABLE_FLOOR,
            "TARGET_STD_FLOOR": TARGET_STD_FLOOR,
        },
    }
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "exp:simulation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "extends": "V3-EXQ-744",
        "outcome": result["outcome"],
        "evidence_direction": result.get("evidence_direction", "unknown"),
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "onset_episodes": ONSET_EPISODES,
        "prefix_cache_states": cache_states,
        "prefix_cache_summary": {
            "hit": sum(1 for s in cache_states if s == "hit"),
            "miss": sum(1 for s in cache_states if s == "miss"),
            "disabled": sum(1 for s in cache_states if s == "disabled"),
        },
        "thresholds": full_config["thresholds"],
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
    # Sanctioned single writer: stamps the Experimental Recording Standard always-core
    # (recording_schema / substrate_hash / machine / machine_class / elapsed_seconds /
    # config / seeds) -- multi-arm, so substrate_hash HOISTS from the per-cell
    # arm_fingerprints -- and enforces the run_id/_v3 + status identity invariants.
    out_path = write_flat_manifest(
        manifest, out_dir, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=started_at,
    )
    print(f"Wrote manifest: {out_path}", flush=True)
    return out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    started_at = time.perf_counter()
    result = run_experiment(dry_run=dry_run)

    if dry_run:
        print(f"DRY_RUN complete: {result['n_cells']} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result, started_at)
    c = result["criteria"]
    print("=== INV-088 world/goal-evaluator DV-coupling result (744a, n=8) ===", flush=True)
    print(f"  preconditions_met: {result['preconditions_met']} "
          f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.3f}; "
          f"dv_decodable={c['PC_dv_decodable']} mature_hazard_decode_r2={c['mean_dv_decode_mature']:.3f}; "
          f"target_var={c['PC_target_var']} min_std={c['min_target_std']:.4f})", flush=True)
    print(f"  C1 quality_gain: {c['C1_quality_gain']} "
          f"(mean_delta_r2={c['mean_delta_r2']:.3f}, sd={c['sd_delta_r2']:.3f}, "
          f"2*sd={2.0 * c['sd_delta_r2']:.3f})", flush=True)
    print(f"  C2 monotone:     {c['C2_monotone']} (mean_rho={c['mean_spearman_rho']:.3f})", flush=True)
    print(f"  C3 noise_fit:    {c['C3_noise_fit_floor']} (mean_immature_r2={c['mean_immature_r2']:.3f} <= {NOISE_FIT_CEIL})",
          flush=True)
    print(f"  [secondary] benefit_eval corroborates: {c['benefit_corroborates']} "
          f"(delta={c['mean_benefit_delta']:.3f}, rho={c['mean_benefit_rho']:.3f})", flush=True)
    cs = result["arm_results"]
    print(f"  prefix_cache: {sum(1 for r in cs if r.get('prefix_cache') == 'hit')} hit / "
          f"{sum(1 for r in cs if r.get('prefix_cache') == 'miss')} miss of {len(cs)} cells", flush=True)
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
