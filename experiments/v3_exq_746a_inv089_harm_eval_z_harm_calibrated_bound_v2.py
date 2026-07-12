#!/opt/local/bin/python3
"""V3-EXQ-746a: INV-089 harm-evaluator z_harm CALIBRATED BOUND test -- CORRECTED re-run of
V3-EXQ-746, built on the shared maturation-curriculum module + frozen-prefix tensor cache.
SUPERSEDES V3-EXQ-746. First WILD CONSUMER of the module's z_harm (harm) leg.

WHY THIS RUN (the 746 result it responds to):
    V3-EXQ-746 RAN and STARVED -- outcome FAIL but evidence_direction="unknown",
    non_degenerate=False, preconditions_met=False (NON_CONTRIBUTORY, does NOT weigh against
    INV-089). Its calibrated bound therefore never actually landed; INV-089 is still only
    provisional and owes exactly this test. Two coupled defects starved 746:
      (a) SPARSE TARGET. 746's DV/IV target was the single-cell ground-truth hazard at the
          agent, Ystate = clip(hazard_field[agent], 0, 1). A random-walk agent almost never
          sits on a hot cell, so the RAW target barely varied (ystate_raw_std 0.004-0.052;
          near-constant for seeds 3, 23, 47). 746's PC_target_var gate checked the NORMALISED
          std (which min-max amplification inflated to ~0.07, passing) instead of the RAW std,
          so it did not catch the degeneracy.
      (b) LABEL != TARGET, so the IV never moved. The HarmEncoder matures on the NORMALISED
          proxy view harm_obs[12] (z_harm decodes it at prox_decode_r2 0.48-0.90), but the RAW
          single-cell target did NOT become more decodable with maturation:
          zharm_state_decode_r2 was flat-to-FALLING across onset (mean IV delta -0.184) ->
          PC_iv_moved False -> the bound test was STARVED. Near-constant-target seeds (3, 47)
          also blew the trained head up (dv_test -1e8, ridge R2 -2.55).

THE FIXES (this run):
    FIX 1 -- DENSER PRIMARY TARGET. The load-bearing (primary) target is a NEIGHBOURHOOD
      hazard density: the mean of the RAW hazard_field over the 5x5 neighbourhood of the agent
      EXCLUDING the centre cell. hazard_field is smooth/autocorrelated (env comment), so the
      neighbourhood mean varies smoothly under a random walk where the single centre cell
      collapses to ~0. It is STATE-DETERMINED (a deterministic function of s_t, read BEFORE the
      action) and DISTINCT from the maturation label harm_obs[12] (the RAW field, not the
      normalised proxy view) -> decoding it is a genuine differentiation test, not a read-back.
    FIX 2 -- RAW-variance precondition. PC_target_var now gates on the RAW target std (>= 0.01),
      the quantity whose collapse actually starves the test -- so a genuinely near-constant
      target routes non_contributory instead of masquerading as passable.
    FIX 3 -- DV numerical guard. The harm_eval_z_harm head training is gradient-clipped and a
      non-finite held-out R2 is floored, so a single pathological cell cannot produce a -1e8
      seed mean (746's seed-3/47 explosion).
    FIX 4 -- THREE targets, one shared prefix (user request). The expensive maturation prefix is
      collected ONCE via the shared module; all three state-determined targets are cheap no-RNG
      reads of the same frozen trajectory, so one run tells us which target definition yields a
      valid, non-degenerate bound. PRIMARY (gating) = local density. SECONDARY (reported,
      each with its OWN precondition gate so a starved secondary is non_contributory-for-that-leg,
      NEVER a false weakens): at_agent (746's original single cell) and next_step (mean over the
      4 orthogonally-reachable cells, a predictive variant).

WHY THE SHARED MODULE (the reuse lever this run exercises):
    743 (INV-089 positive control), 746 (INV-089 bound, starved) each RE-INLINED the identical
    HarmEncoder maturation + frozen-dataset collection. This run builds its frozen z_harm prefix
    via experiments/_lib/baselines/maturation_curriculum.mature_and_collect_harm, which
    (a) makes the whole harm leg share ONE recipe so sibling arm_fingerprints match by
    construction, and (b) mints a frozen-prefix TENSOR cache. The module's collect_harm_dataset
    was extended (2026-07-12) to also emit the three raw-hazard_field targets as no-RNG reads,
    so the shared Zharm/Y/Prox stream stays bit-identical to 743's inline path (the module's
    caller-contract). Cells are emitted reuse-ELIGIBLE (include_driver_script_in_hash=False),
    the harm leg's first minted-and-consumed cells.

DESIGN (measurement-only, commitment-free frozen-representation curriculum-ORDER contrast;
        methodological sibling of INV-088's 744a):
    Per (seed, onset) cell:
      1. Build agent (056c SD-010 wiring: world_dim = z_harm_dim = 32) + a standalone
         HarmEncoder(51, 32); snapshot the FRESH harm_eval_z_harm_head init (module-captured).
      2. MATURE the HarmEncoder for `onset` episodes on harm_obs[12] with a throwaway head
         (discarded). onset=0 -> fresh encoder = the genuinely-immature anchor (743-identical
         recipe -> reproduces 743's z_harm differentiation).
      3. FREEZE; collect the SHARED frozen dataset (fixed action seq, fixed seeded env,
         maturity-independent trajectory -> the only per-arm variable is the frozen z_harm).
         Per step: z_harm[t]; Ydens/Yat/Ynext (state-determined raw targets); Yreal (743
         realized-harm cross-check); Prox = harm_obs[12] (maturation label, reported only).
      4. Per target: IV = ridge held-out R2 z_harm -> target (rises with maturation if the
         encoder differentiates that target); DV = harm_eval_z_harm held-out R2 predicting the
         target (re-init head to the fixed shared init, FIXED gradient-clipped budget).

    IV per arm = onset episodes {0, 1, 4, 12, 30}.  SEEDS: 8 (mirror 744a / 746).

REGIME (unchanged -- 537b / 740a / 743 lesson): scheduled_external_hazard OFF so harm is
    predictable-from-state and z_harm differentiation is the binding constraint.

PRE-REGISTERED PASS CRITERIA (LOAD-BEARING; SCALE-FREE; on the PRIMARY local-density DV):
    C1 dv_monotone:     mean_seed Spearman(onset, harm_eval_r2_test across arms) >= 0.80
    C2 bound_coupling:  mean_seed(DV[onset_max] - DV[onset_min]) > 0
                        AND mean_seed Spearman(IV_arm, DV_arm) >= 0.80
    C3 dv_reliable:     mean_seed(DV_delta) >= 2.0 * SD_seed(DV_delta)
    PASS = C1 and C2 and C3 on the PRIMARY target.

PRECONDITIONS (validity; on the PRIMARY target; unmet -> non_degenerate=False, direction unknown):
    PC_iv_moved:     mean_seed(IV[onset_max] - IV[onset_min]) > 0 AND mean_seed Spearman(onset,IV) > 0
    PC_dv_decodable: mean_seed(IV[onset_max]) >= 0.05
    PC_target_var:   min over cells of the RAW primary-target std >= 0.008  (the 746 fix: RAW, not
                     normalised -- a genuinely near-constant target STARVES, not falsifies)

WHAT A PASS DOES: moves INV-089 provisional -> toward stable (the calibrated bound the
    provisional evidence_quality_note names as the missing test). A met-precondition C1/C2/C3
    FAIL is a genuine WEAKENS; only a starved IV / undecodable target / near-constant RAW target
    routes non_contributory.

RE-DERIVE BRAKE (INV-089 substrate_ceiling autopsies: 0) HONOURED: reads the z_harm stream,
    tests INV-089, does NOT decode scalar harm from z_world -> NOT a 740b. SUPERSEDES V3-EXQ-746
    (which never validly tested the bound). NO SUBSTRATE BUILD OWED (HarmEncoder + harm_eval_z_harm
    are SD-010 IMPLEMENTED).
"""

from __future__ import annotations

import argparse
import copy
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
from experiments._lib.baselines.maturation_curriculum import mature_and_collect_harm  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_746a_inv089_harm_eval_z_harm_calibrated_bound_v2"
QUEUE_ID = "V3-EXQ-746a"
SUPERSEDES = "V3-EXQ-746"      # 746 starved (non_contributory); this is the valid calibrated bound
CLAIM_IDS = ["INV-089"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants (743 maturation/collection recipe; seeds mirror 744a/746) ---
SEEDS = [42, 7, 19, 3, 11, 23, 47, 101]
ONSET_EPISODES = [0, 1, 4, 12, 30]            # onset_0 = fresh (untrained) HarmEncoder anchor
MATURE_PROGRESS_DENOM = max(ONSET_EPISODES)   # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared harm-dataset collection (identical per arm)
DV_EPOCHS = 40                       # FIXED harm_eval_z_harm head training budget (all arms/targets)
DV_BATCH = 64
DV_LR = 1e-3
DV_GRAD_CLIP = 1.0                   # 746 fix: gradient-clip the DV head (prevents the -1e8 blow-up)
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear decode probes
DV_TRAIN_SEED = 90002                # fixed DV optimisation RNG (all arms/targets identical)
DECODE_SPLIT_SEED = 90003            # fixed decode-probe train/test split (all arms identical)
COLLECT_SEED_BASE = 70000            # collection env + action RNG base (per-seed, arm-independent)
MATURE_SEED_BASE = 60000             # maturation env + action RNG base (per-seed, arm-independent)

HARM_OBS_DIM = 51
Z_HARM_DIM = 32
WORLD_DIM = 32                       # = Z_HARM_DIM so harm_eval_z_harm_head (world_dim fallback) matches
HARM_OBS_CENTER = 12                 # harm_obs[12] = normalised hazard proximity at agent cell (SD-010 label)
HAZARD_NEIGHBOURHOOD_RADIUS = 2      # 5x5 neighbourhood for the local-density primary target

# Three STATE-DETERMINED raw-hazard_field targets collected by the shared module.
# key -> (dataset field, human label, is_primary)
TARGETS: List[Dict[str, Any]] = [
    {"key": "dens", "field": "Ydens", "label": "local_density", "primary": True},
    {"key": "at", "field": "Yat", "label": "at_agent", "primary": False},
    {"key": "next", "field": "Ynext", "label": "next_step", "primary": False},
]
PRIMARY_KEY = "dens"

ENV_KWARGS = {
    "size": 12,
    "num_hazards": 5,   # denser predictable harm -> enough hazard variation per collected step
    "num_resources": 5,
    "use_proxy_fields": True,   # SD-010: required for harm_obs in obs_dict
    "env_drift_prob": 0.3,
    "env_drift_interval": 1,
    "limb_damage_enabled": True,
    "harm_history_len": 10,
    "reef_enabled": True,
    "n_reef_patches": 3,
    "reef_patch_radius": 2,
    "hazard_food_attraction": 0.7,
    # 537b / 740a / 743 lesson: OFF so harm is predictable-from-state and z_harm
    # differentiation is the binding constraint (not a by-design-unpredictable ceiling).
    "scheduled_external_hazard_enabled": False,
}

# --- pre-registered thresholds (CALIBRATED: SCALE-FREE; NO 740a magnitude floors) ---
MONOTONE_RHO_MIN = 0.80        # C1: DV monotone in onset (rank-based -> narrow gradient OK)
COUPLING_RHO_MIN = 0.80        # C2: DV tracks measured z_harm differentiation (IV<->DV rank)
DV_DELTA_SD_MULT = 2.0         # C3: reliability effect-size (scale-free t-stat-like)
# Preconditions (validity).
DV_DECODABLE_FLOOR = 0.05      # PC_dv_decodable: mature-anchor target decodability positive control
RAW_TARGET_STD_FLOOR = 0.008   # PC_target_var: RAW target std floor (the 746 fix; RAW not normalised).
#   Calibrated 2026-07-12 from a real-fidelity probe (seeds 3,47,42,7 x onset {0,30}): the dense
#   PRIMARY target's raw std was 0.0119-0.0237 (clears), while the degenerate single-cell `at`
#   target that starved 746 was 0.0038-0.0065 (correctly excluded). 0.008 sits between them.
DV_NONFINITE_FLOOR = -1.0      # 746 fix: non-finite held-out R2 floored here (R2=-1 = 2x mean-error fit)
# Reported-only reference (continuity with 743 / 746 / 740a; NOT gating).
ZWORLD_HARM_DECODE_740A_REF = 0.034  # 740a mature-z_world realized-harm decode (cited contrast)


def _r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _eff_rank(Z: torch.Tensor) -> float:
    """Participation ratio of the z_harm covariance eigenspectrum (diagnosis only)."""
    if Z.shape[0] < 3:
        return 0.0
    Zc = Z - Z.mean(dim=0, keepdim=True)
    cov = (Zc.t() @ Zc) / (Z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    denom = (eig.pow(2).sum() + 1e-12).item()
    return float((eig.sum().item() ** 2) / denom)


def _ridge_heldout_r2(Z: torch.Tensor, T: torch.Tensor, split_seed: int,
                      lam: float = RIDGE_LAMBDA) -> float:
    """Held-out R^2 of a closed-form ridge linear probe Z -> T (fixed split identical across
    arms; baseline is the TRAIN-set per-column mean). Pure read-out of the FROZEN z_harm."""
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
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return r2 if np.isfinite(r2) else DV_NONFINITE_FLOOR


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


def _train_dv_and_eval(agent: REEAgent, Z: torch.Tensor, T: torch.Tensor,
                       head_init_state: Dict[str, Any], dv_epochs: int) -> Dict[str, float]:
    """Re-init harm_eval_z_harm_head to the fixed shared init, train for a FIXED budget on the
    frozen (z_harm, target) tensors, return train/test R^2 + gap. Fixed, bit-identical init +
    optimisation RNG + split across all arms/targets. Gradient-clipped (746 blow-up fix); a
    non-finite held-out R2 is floored to DV_NONFINITE_FLOOR.
    """
    n = Z.shape[0]
    n_test = max(1, int(n * HELDOUT_FRAC))
    split_perm = np.random.default_rng(DV_TRAIN_SEED + 1).permutation(n)
    te = torch.as_tensor(split_perm[:n_test], dtype=torch.long)
    tr = torch.as_tensor(split_perm[n_test:], dtype=torch.long)
    Ztr, Ttr = Z[tr], T[tr]
    Zte, Tte = Z[te], T[te]

    agent.e3.harm_eval_z_harm_head.load_state_dict(copy.deepcopy(head_init_state))
    for p in agent.e3.harm_eval_z_harm_head.parameters():
        p.requires_grad_(True)
    torch.manual_seed(DV_TRAIN_SEED)
    opt = torch.optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=DV_LR)
    batch_rng = np.random.default_rng(DV_TRAIN_SEED)

    ntr = Ztr.shape[0]
    for epoch in range(dv_epochs):
        perm = batch_rng.permutation(ntr)
        for start in range(0, ntr, DV_BATCH):
            bidx = torch.as_tensor(perm[start:start + DV_BATCH], dtype=torch.long)
            pred = agent.e3.harm_eval_z_harm(Ztr[bidx])
            loss = F.mse_loss(pred, Ttr[bidx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_z_harm_head.parameters(), DV_GRAD_CLIP)
            opt.step()

    agent.eval()
    with torch.no_grad():
        r2_train = _r2(agent.e3.harm_eval_z_harm(Ztr), Ttr)
        r2_test = _r2(agent.e3.harm_eval_z_harm(Zte), Tte)
    if not np.isfinite(r2_test):
        r2_test = DV_NONFINITE_FLOOR
    if not np.isfinite(r2_train):
        r2_train = DV_NONFINITE_FLOOR
    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "gap": r2_train - r2_test,
    }


def _run_cell(seed: int, onset: int, steps_per_ep: int, collect_eps: int,
              dv_epochs: int, dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition onset_{onset}", flush=True)
    config_slice = {
        "onset_episodes": onset,
        "steps_per_ep": steps_per_ep,
        "collect_episodes": collect_eps,
        "dv_epochs": dv_epochs,
        "env_kwargs": ENV_KWARGS,
        "z_harm_dim": Z_HARM_DIM,
        "world_dim": WORLD_DIM,
        "dv_batch": DV_BATCH,
        "dv_lr": DV_LR,
        "dv_grad_clip": DV_GRAD_CLIP,
        "ridge_lambda": RIDGE_LAMBDA,
        "hazard_neighbourhood_radius": HAZARD_NEIGHBOURHOOD_RADIUS,
        "targets": [t["label"] for t in TARGETS],
    }
    # Cells ARE reuse-eligible: the frozen z_harm prefix is a DETERMINISTIC pure function of
    # (substrate, config_slice, seed) collected by the shared maturation_curriculum module.
    # include_driver_script_in_hash=False anchors the fingerprint on that module (in the _lib
    # substrate glob) so a differently-driven harm-leg sibling can cite these cells.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
        # Frozen z_harm prefix via the shared module (mint-as-you-go into the frozen-prefix
        # tensor cache). Called FIRST inside arm_cell (per the module's bit-identity contract),
        # so the fresh harm_eval_z_harm_head init + dataset match 743's inline path exactly.
        agent, _harm_enc, fresh_heads, data, n_mature_steps, prov = mature_and_collect_harm(
            seed, onset,
            env_kwargs=ENV_KWARGS,
            steps_per_ep=steps_per_ep,
            collect_episodes=collect_eps,
            world_dim=WORLD_DIM,
            z_harm_dim=Z_HARM_DIM,
            harm_obs_dim=HARM_OBS_DIM,
            harm_obs_center=HARM_OBS_CENTER,
            mature_lr=1e-3,
            mature_hidden=32,
            mature_seed_base=MATURE_SEED_BASE,
            collect_seed_base=COLLECT_SEED_BASE,
            hazard_neighbourhood_radius=HAZARD_NEIGHBOURHOOD_RADIUS,
            mature_progress_denom=MATURE_PROGRESS_DENOM,
        )
        Zharm = data["Zharm"]
        head_init = fresh_heads["harm_eval_z_harm_head"]

        # Differentiation mediators at freeze.
        zharm_var = float(Zharm.var(dim=0).mean().item())
        zharm_eff_rank = _eff_rank(Zharm)
        # Reported cross-checks (NOT gating): 743 realized-harm decode + maturation-label decode.
        zharm_realized_decode_r2 = _ridge_heldout_r2(Zharm, data["Y"], DECODE_SPLIT_SEED)
        zharm_prox_decode_r2 = _ridge_heldout_r2(Zharm, data["Prox"], DECODE_SPLIT_SEED)

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "n_mature_steps": n_mature_steps,
            "zharm_var": zharm_var,
            "zharm_eff_rank": zharm_eff_rank,
            "zharm_realized_decode_r2": zharm_realized_decode_r2,  # 743 cross-check (reported)
            "zharm_prox_decode_r2": zharm_prox_decode_r2,          # maturation-label decode (reported)
            "n_samples": int(Zharm.shape[0]),
            "prefix_cache": prov["cache"],
        }
        # Per target: IV (ridge held-out R2 z_harm -> target), DV (trained harm_eval_z_harm head
        # held-out R2), and the RAW target std (per-seed; same across the onset arms of a seed).
        for t in TARGETS:
            T = data[t["field"]]                                   # raw-clipped [0,1] target
            iv = _ridge_heldout_r2(Zharm, T, DECODE_SPLIT_SEED)
            dv = _train_dv_and_eval(agent, Zharm, T, head_init, dv_epochs)
            k = t["key"]
            row[f"iv_{k}_decode_r2"] = iv
            row[f"dv_{k}_r2_test"] = dv["r2_test"]
            row[f"dv_{k}_r2_train"] = dv["r2_train"]
            row[f"dv_{k}_gap"] = dv["gap"]
            row[f"target_{k}_raw_std"] = float(T.std().item())
        cell.stamp(row)

    print(f"verdict: {'PASS' if np.isfinite(row['dv_dens_r2_test']) else 'FAIL'}",
          flush=True)  # NaN-guard: cell completed
    return row


def _evaluate_target(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Scale-free calibrated-bound evaluation for ONE target (identical to 746's criteria)."""
    onset_min = min(ONSET_EPISODES)
    onset_max = max(ONSET_EPISODES)

    def _cell(seed: int, onset: int) -> Dict[str, Any]:
        return next(r for r in rows if r["seed"] == seed and r["onset_episodes"] == onset)

    dv_delta: List[float] = []
    dv_rho: List[float] = []          # C1: DV monotone in onset
    iv_delta: List[float] = []
    iv_rho: List[float] = []          # PC_iv_moved: IV monotone in onset
    couple_rho: List[float] = []      # C2: DV tracks measured IV (across arms)
    iv_mature: List[float] = []       # PC_dv_decodable: mature-anchor IV
    for seed in SEEDS:
        onsets = [float(o) for o in ONSET_EPISODES]
        dvs = [_cell(seed, o)[f"dv_{key}_r2_test"] for o in ONSET_EPISODES]
        ivs = [_cell(seed, o)[f"iv_{key}_decode_r2"] for o in ONSET_EPISODES]
        dv_delta.append(_cell(seed, onset_max)[f"dv_{key}_r2_test"]
                        - _cell(seed, onset_min)[f"dv_{key}_r2_test"])
        dv_rho.append(_spearman(onsets, dvs))
        iv_delta.append(_cell(seed, onset_max)[f"iv_{key}_decode_r2"]
                        - _cell(seed, onset_min)[f"iv_{key}_decode_r2"])
        iv_rho.append(_spearman(onsets, ivs))
        couple_rho.append(_spearman(ivs, dvs))
        iv_mature.append(_cell(seed, onset_max)[f"iv_{key}_decode_r2"])

    def _mean(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _sd(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = _mean(v)
        return float((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5)

    mean_dv_delta = _mean(dv_delta)
    sd_dv_delta = _sd(dv_delta)
    mean_dv_rho = _mean(dv_rho)
    mean_iv_delta = _mean(iv_delta)
    mean_iv_rho = _mean(iv_rho)
    mean_couple_rho = _mean(couple_rho)
    mean_iv_mature = _mean(iv_mature)
    min_raw_std = min(r[f"target_{key}_raw_std"] for r in rows)

    # Preconditions (validity). RAW-std gate (the 746 fix).
    pc_iv_moved = (mean_iv_delta > 0.0) and (mean_iv_rho > 0.0)
    pc_dv_decodable = mean_iv_mature >= DV_DECODABLE_FLOOR
    pc_target_var = min_raw_std >= RAW_TARGET_STD_FLOOR
    preconditions_met = pc_iv_moved and pc_dv_decodable and pc_target_var

    # LOAD-BEARING PASS criteria (scale-free).
    c1 = mean_dv_rho >= MONOTONE_RHO_MIN
    c2 = (mean_dv_delta > 0.0) and (mean_couple_rho >= COUPLING_RHO_MIN)
    c3 = mean_dv_delta >= (DV_DELTA_SD_MULT * sd_dv_delta)
    claim_pass = c1 and c2 and c3

    return {
        "target_key": key,
        "per_seed_dv_delta": dv_delta,
        "per_seed_dv_rho": dv_rho,
        "per_seed_iv_delta": iv_delta,
        "per_seed_iv_rho": iv_rho,
        "per_seed_couple_rho": couple_rho,
        "per_seed_iv_mature": iv_mature,
        "mean_dv_delta": mean_dv_delta,
        "sd_dv_delta": sd_dv_delta,
        "mean_dv_rho": mean_dv_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_iv_rho": mean_iv_rho,
        "mean_couple_rho": mean_couple_rho,
        "mean_iv_mature": mean_iv_mature,
        "min_raw_std": min_raw_std,
        "PC_iv_moved": pc_iv_moved,
        "PC_dv_decodable": pc_dv_decodable,
        "PC_target_var": pc_target_var,
        "preconditions_met": preconditions_met,
        "C1_dv_monotone": c1,
        "C2_bound_coupling": c2,
        "C3_dv_reliable": c3,
        "claim_pass": claim_pass,
    }


def _aggregate_by_onset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ONSET_EPISODES:
        cells = [r for r in rows if r["onset_episodes"] == o]
        entry: Dict[str, float] = {
            "mean_zharm_realized_decode_r2": float(np.mean([c["zharm_realized_decode_r2"] for c in cells])),
            "mean_zharm_prox_decode_r2": float(np.mean([c["zharm_prox_decode_r2"] for c in cells])),
            "mean_zharm_var": float(np.mean([c["zharm_var"] for c in cells])),
            "mean_zharm_eff_rank": float(np.mean([c["zharm_eff_rank"] for c in cells])),
        }
        for t in TARGETS:
            k = t["key"]
            entry[f"mean_dv_{k}_r2_test"] = float(np.mean([c[f"dv_{k}_r2_test"] for c in cells]))
            entry[f"mean_iv_{k}_decode_r2"] = float(np.mean([c[f"iv_{k}_decode_r2"] for c in cells]))
        out[f"onset_{o}"] = entry
    return out


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [SEEDS[0]]
        onsets = [0, 2]
        steps_per_ep = 20
        collect_eps = 3
        dv_epochs = 3
    else:
        seeds = SEEDS
        onsets = ONSET_EPISODES
        steps_per_ep = STEPS_PER_EP
        collect_eps = COLLECT_EPISODES
        dv_epochs = DV_EPOCHS

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for onset in onsets:
            rows.append(_run_cell(seed, onset, steps_per_ep, collect_eps, dv_epochs, dry_run))

    if dry_run:
        return {
            "outcome": "PASS",
            "dry_run": True,
            "n_cells": len(rows),
            "arm_results": rows,
        }

    per_target = {t["key"]: _evaluate_target(rows, t["key"]) for t in TARGETS}
    by_onset = _aggregate_by_onset(rows)
    primary = per_target[PRIMARY_KEY]

    preconditions_met = primary["preconditions_met"]
    claim_pass = primary["claim_pass"]

    if not preconditions_met:
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        reasons = []
        if not primary["PC_iv_moved"]:
            reasons.append(
                f"primary (local_density) z_harm differentiation gradient did not move: "
                f"mean IV delta {primary['mean_iv_delta']:.4f} <= 0 or mean IV rank rho "
                f"{primary['mean_iv_rho']:.3f} <= 0 (bound test starved, not falsified)")
        if not primary["PC_dv_decodable"]:
            reasons.append(
                f"primary target not decodable-in-principle from mature z_harm: mean "
                f"mature-anchor IV {primary['mean_iv_mature']:.3f} < {DV_DECODABLE_FLOOR}")
        if not primary["PC_target_var"]:
            reasons.append(
                f"primary target near-constant: min RAW std {primary['min_raw_std']:.4f} "
                f"< {RAW_TARGET_STD_FLOOR}")
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
        "per_target": per_target,
        "primary_target": PRIMARY_KEY,
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
        "dv_epochs": DV_EPOCHS,
        "dv_batch": DV_BATCH,
        "dv_lr": DV_LR,
        "dv_grad_clip": DV_GRAD_CLIP,
        "heldout_frac": HELDOUT_FRAC,
        "ridge_lambda": RIDGE_LAMBDA,
        "collect_seed_base": COLLECT_SEED_BASE,
        "mature_seed_base": MATURE_SEED_BASE,
        "dv_train_seed": DV_TRAIN_SEED,
        "decode_split_seed": DECODE_SPLIT_SEED,
        "hazard_neighbourhood_radius": HAZARD_NEIGHBOURHOOD_RADIUS,
        "env_kwargs": ENV_KWARGS,
        "targets": {t["key"]: t["label"] for t in TARGETS},
        "primary_target": PRIMARY_KEY,
        "thresholds": {
            "MONOTONE_RHO_MIN": MONOTONE_RHO_MIN,
            "COUPLING_RHO_MIN": COUPLING_RHO_MIN,
            "DV_DELTA_SD_MULT": DV_DELTA_SD_MULT,
            "DV_DECODABLE_FLOOR": DV_DECODABLE_FLOOR,
            "RAW_TARGET_STD_FLOOR": RAW_TARGET_STD_FLOOR,
            "ZWORLD_HARM_DECODE_740A_REF": ZWORLD_HARM_DECODE_740A_REF,
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
        "extends": "V3-EXQ-743",
        "supersedes": SUPERSEDES,
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
        "primary_target": PRIMARY_KEY,
        "per_target": result["per_target"],
        "by_onset": result["by_onset"],
        "arm_results": result["arm_results"],
        "deliverable_note": (
            "CORRECTED calibrated bound test for INV-089, SUPERSEDING V3-EXQ-746 which ran but "
            "STARVED (non_contributory: PC_iv_moved/PC_dv_decodable failed on a SPARSE single-cell "
            "target the encoder's maturation label did not track; seed-3/47 DV blow-ups). Three "
            "state-determined raw-hazard_field targets share one frozen prefix (collected via the "
            "shared maturation_curriculum.mature_and_collect_harm module -- the harm leg's first "
            "wild consumer). PRIMARY (gating) = local neighbourhood density (dense -> high raw "
            "variance under a random walk). Fixes over 746: (1) denser primary target, (2) RAW-std "
            "precondition (not the normalised std 746 mis-gated on), (3) gradient-clipped DV head + "
            "non-finite floor. LOAD-BEARING PASS = harm_eval_z_harm held-out quality on the PRIMARY "
            "target rises monotonically with maturation (C1), tracks measured z_harm differentiation "
            "(C2), reliably across seeds (C3). A met-precondition C1/C2/C3 FAIL is a genuine WEAKENS; "
            "a starved IV / undecodable / near-constant primary target routes non_contributory. "
            "Secondary targets (at_agent, next_step) are reported with their own precondition gates, "
            "non-gating. A PASS moves INV-089 provisional -> toward stable."
        ),
        "cross_stream_contrast_note": (
            "Extends V3-EXQ-743 (INV-089 positive control), supersedes V3-EXQ-746 (starved bound). "
            "Reuses 743's exact z_harm maturation + collection recipe (same MATURE/COLLECT seed "
            "bases) via the shared module. zharm_realized_decode_r2 (743's positive-control metric) "
            "and zharm_prox_decode_r2 are reported per cell for continuity. Re-derive brake "
            "(INV-089 substrate_ceiling autopsies: 0) HONOURED: z_harm observable, INV-089 claim, no "
            "z_world decode -- NOT a 740b."
        ),
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
    print("=== INV-089 harm-evaluator z_harm CALIBRATED BOUND result (746a, n=8, 3 targets) ===",
          flush=True)
    for t in TARGETS:
        c = result["per_target"][t["key"]]
        tag = "PRIMARY" if t["primary"] else "secondary"
        print(f"  [{tag}:{t['label']}] pre_met={c['preconditions_met']} "
              f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.4f}; "
              f"dv_decodable={c['PC_dv_decodable']} mature_iv={c['mean_iv_mature']:.3f}; "
              f"raw_std={c['min_raw_std']:.4f}) "
              f"C1={c['C1_dv_monotone']}(rho={c['mean_dv_rho']:.3f}) "
              f"C2={c['C2_bound_coupling']}(dvd={c['mean_dv_delta']:.4f},cpl={c['mean_couple_rho']:.3f}) "
              f"C3={c['C3_dv_reliable']}(2sd={DV_DELTA_SD_MULT * c['sd_dv_delta']:.4f}) "
              f"-> claim_pass={c['claim_pass']}", flush=True)
    print(f"  OUTCOME: {result['outcome']} (direction={result['evidence_direction']}) "
          f"on PRIMARY={PRIMARY_KEY}", flush=True)
    cs = result["arm_results"]
    print(f"  prefix_cache: {sum(1 for r in cs if r.get('prefix_cache') == 'hit')} hit / "
          f"{sum(1 for r in cs if r.get('prefix_cache') == 'miss')} miss of {len(cs)} cells", flush=True)

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
