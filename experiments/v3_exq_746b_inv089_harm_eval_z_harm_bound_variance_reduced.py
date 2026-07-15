#!/opt/local/bin/python3
"""V3-EXQ-746b: INV-089 harm-evaluator z_harm CALIBRATED BOUND -- VARIANCE-REDUCED re-estimate
of V3-EXQ-746a. EXTENDS V3-EXQ-743 (positive control) and re-estimates 746a's DV. Built on the
same shared maturation-curriculum module + frozen-prefix tensor cache, so the frozen z_harm prefix
and the IV probe are BIT-IDENTICAL to 746a and reuse its cached cells; ONLY the DV estimator changes.

WHY THIS RUN (the 746a result it responds to):
    V3-EXQ-746a ran with ALL preconditions met (non_degenerate=True) and WEAKENED INV-089's
    strict-bound reading: C1 dv_monotone rho=-0.53, C2 couple rho=-0.08, mean_dv_delta=-34.5.
    But that "weakens" is a MEASUREMENT ARTIFACT, not a real null. Diagnosis from 746a's own
    per-cell numbers:
      - The IV is CLEAN and MOVING. Local-density decodes from the frozen z_harm via a linear
        ridge at ~0.37-0.40 (mature-anchor iv_mature ~0.38-0.54); PC_iv_moved True; the z_harm
        differentiation gradient is real. Nothing wrong with the representation or the target.
      - The DV is MEASUREMENT-PATHOLOGICAL. The trained harm_eval_z_harm MLP head
        (Linear(32,hidden)->ReLU->Linear(hidden,1)) was trained UNREGULARISED, for a fixed 40
        epochs, evaluated on a SINGLE 30% held-out split. Its held-out R2 was -26..-69 per onset
        with per-seed dv_delta swinging -4.7..-166 (one seed -166) -- i.e. the head OVERFITS ~490
        points and DIVERGES on held-out, producing garbage R2 whose magnitude dwarfs any signal.
        The rank-based C1/C2 read that divergence (which WORSENS with onset as z_harm differentiates
        and the head overfits harder) as an anti-coupling; the magnitude C2/C3 are polluted by the
        -166. A LINEAR RIDGE on the IDENTICAL (z_harm, target) gets +0.38 -- so the target is easily
        decodable and the failure is the DV ESTIMATOR, not the bound.
    Conclusion: 746a never validly measured "harm_eval_z_harm held-out quality". This run measures
    it correctly. The bound-coupling question INV-089 owes is therefore still OPEN.

THE FIXES (DV estimator + data power -- the maturation recipe and the IV probe are unchanged;
diagnosed empirically 2026-07-15 on real cells, seed 42, onset {0,30}):
    FIX 1 -- STANDARDISE THE DV TRAINING TARGET (the load-bearing fix). Ydens has tiny magnitude
      (raw std ~0.02); a fresh harm_eval_z_harm head outputs O(1), so MSE on the raw target is
      badly conditioned and the SGD head lands WORSE than predicting the mean (held-out R2 < -1)
      while the scale-robust closed-form ridge on the SAME data gets +0.38. Standardising the target
      with the FOLD-TRAIN mean/std (R2 is affine-invariant, so this does NOT change what is measured)
      fixes it: measured DV rose from -1.000 (raw) to +0.33 (standardised) at the ORIGINAL data size,
      and to +0.47/+0.52 at collect=60 -- a valid, low-variance estimator whose R2 is directly
      comparable to the IV ridge R2.
    FIX 2 -- K-FOLD AVERAGED DV. Replace 746a's single fixed 30% split with K=5 repeated random
      held-out splits (fixed seeds); average the held-out R2 over folds. Kills the single-split luck
      behind 746a's -166 vs -4.7 per-seed swing (measured per-fold spread now ~0.05-0.10).
    FIX 3 -- REGULARISED + EARLY-STOPPED HEAD. The head Adam carries L2 weight decay, and each fold
      trains with EARLY STOPPING on an inner-validation split (patience) -- halting at the best
      generalisation point instead of training into divergence.
    FIX 4 -- FINITE R2 FLOOR. Each fold's held-out (and train) R2 is floored at a FINITE bound
      (R2 < -1 = uninformative noise). 746a only floored NON-finite R2, so its finite -60/-166 values
      polluted every mean/SD; with a finite floor the magnitude criteria (C2 clause 1, C3) are
      meaningful again.
    FIX 5 -- MORE DATA FOR POWER. collect_episodes 14 -> 60 (~195 -> ~780 samples/cell). At n~195 the
      standardised DV is valid but per-fold-noisy relative to the ~0.07 IV gradient; n~780 tightens
      the folds and lifts the DV dynamic range (0.47->0.52 across onset for seed 42), giving the
      bound-coupling test genuine power. This means 746b does NOT reuse 746a's frozen prefix (the
      collection changed); it mints its OWN reuse-eligible prefix for future harm-leg siblings.
    FIX 6 -- DV-ESTIMATOR READINESS PRECONDITION (NEW). PC_dv_estimator_ok: the mean-seed
      MATURE-ANCHOR DV (k-fold mean, floored) must clear DV_ESTIMATOR_FLOOR (>= 0.0 -- beat the mean
      on held-out) GIVEN the target is ridge-decodable at ~0.4. If the standardised + regularised +
      early-stopped + fold-averaged head STILL cannot beat the mean at maturity, the DV measurement
      is not viable -> route non_contributory (substrate_not_ready), NOT weakens. The rail that keeps
      a broken measurement from ever masquerading as evidence (the standing INV-064/740a lesson).

    SEEDS: same 8 as 746a. Per-cell DV estimation (fixed by FIX 1-5), NOT seed count, was the
    bottleneck -- 746a was already 8-seed.

DESIGN (measurement-only, commitment-free frozen-representation curriculum-ORDER contrast;
        maturation recipe + IV probe unchanged from 743/746a, collection widened for power):
    Per (seed, onset) cell:
      1. Frozen z_harm prefix via mature_and_collect_harm (shared module; mints its own prefix).
      2. IV = ridge held-out R2 z_harm -> target (same probe as 746a; the clean, moving leg).
      3. DV = harm_eval_z_harm k-fold averaged held-out R2 predicting the STANDARDISED target,
         re-init to the fixed shared head init per fold, weight-decayed + early-stopped, per-fold
         floored.
    IV per arm = onset episodes {0, 1, 4, 12, 30}. PRIMARY target = local neighbourhood density
    (dense -> high raw variance under a random walk; the 746a-validated primary). Secondaries
    (at_agent, next_step) reported with their own precondition gates, non-gating.

REGIME (unchanged -- 537b / 740a / 743 lesson): scheduled_external_hazard OFF so harm is
    predictable-from-state and z_harm differentiation is the binding constraint.

PRE-REGISTERED PASS CRITERIA (LOAD-BEARING; SCALE-FREE; on the PRIMARY local-density DV):
    C1 dv_monotone:     mean_seed Spearman(onset, harm_eval_r2_test across arms) >= 0.80
    C2 bound_coupling:  mean_seed(DV[onset_max] - DV[onset_min]) > 0
                        AND mean_seed Spearman(IV_arm, DV_arm) >= 0.80
    C3 dv_reliable:     mean_seed(DV_delta) >= 2.0 * SD_seed(DV_delta)
    PASS = C1 and C2 and C3 on the PRIMARY target.

PRECONDITIONS (validity; on the PRIMARY target; unmet -> non_degenerate=False, direction unknown):
    PC_iv_moved:        mean_seed(IV[onset_max] - IV[onset_min]) > 0 AND mean_seed Spearman(onset,IV) > 0
    PC_dv_decodable:    mean_seed(IV[onset_max]) >= 0.05
    PC_target_var:      min over cells of the RAW primary-target std >= 0.008 (746a RAW-std gate)
    PC_dv_estimator_ok: mean_seed(DV[onset_max], k-fold, floored) >= 0.0 (NEW -- the DV estimator
                        must be viable; a still-broken estimator routes non_contributory not weakens)

WHAT A PASS DOES: moves INV-089 provisional -> toward stable (the calibrated bound the provisional
    evidence_quality_note names as the missing test). A met-precondition C1/C2/C3 FAIL is a genuine,
    now-TRUSTWORTHY WEAKENS (if 746b weakens with a clean DV, 746a's weakens stands, reinforced). A
    PASS instead reframes 746a's weakens as the DV-measurement artifact diagnosed above -> governance
    may then mark 746a superseded. Either way the verdict is finally on a valid DV.

RE-DERIVE BRAKE (INV-089 substrate_ceiling autopsies: 0) HONOURED: reads the z_harm stream, tests
    INV-089, does NOT decode scalar harm from z_world -> NOT a 740b. NO SUBSTRATE BUILD OWED
    (HarmEncoder + harm_eval_z_harm are SD-010 IMPLEMENTED). This is a DV-estimator refinement of a
    same-question test (letter b), not a new mechanism.
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

EXPERIMENT_TYPE = "v3_exq_746b_inv089_harm_eval_z_harm_bound_variance_reduced"
QUEUE_ID = "V3-EXQ-746b"
EXTENDS = "V3-EXQ-746a"        # re-estimates 746a's DV (maturation/collection/IV bit-identical)
CLAIM_IDS = ["INV-089"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants: BIT-IDENTICAL to 746a (frozen prefix + IV reuse) ---
SEEDS = [42, 7, 19, 3, 11, 23, 47, 101]
ONSET_EPISODES = [0, 1, 4, 12, 30]            # onset_0 = fresh (untrained) HarmEncoder anchor
MATURE_PROGRESS_DENOM = max(ONSET_EPISODES)   # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 60                # FIX 5: widened 14 -> 60 (~195 -> ~780 samples/cell) for DV power
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear IV decode probe (= 746a)
DECODE_SPLIT_SEED = 90003            # fixed IV decode-probe split (all arms identical; = 746a)
COLLECT_SEED_BASE = 70000            # collection env + action RNG base (= 746a; keeps prefix cache hit)
MATURE_SEED_BASE = 60000             # maturation env + action RNG base (= 746a; keeps prefix cache hit)

# --- DV estimator constants: the 746b VARIANCE-REDUCTION changes (these differ from 746a) ---
DV_KFOLD = 5                         # FIX 1: K repeated random held-out splits, DV R2 averaged
DV_MAX_EPOCHS = 80                   # FIX 2: max budget; early stopping halts well before this
DV_BATCH = 64
DV_LR = 1e-3
DV_WEIGHT_DECAY = 1e-3               # FIX 2: L2 on the harm_eval_z_harm head Adam (anti-divergence)
DV_EARLY_STOP_PATIENCE = 10         # FIX 2: epochs of no inner-val improvement before stopping
DV_EARLY_STOP_MIN_DELTA = 1e-5      # FIX 2: min inner-val MSE improvement to reset patience
DV_INNER_VAL_FRAC = 0.2             # FIX 2: inner-validation fraction of the fold train set
DV_GRAD_CLIP = 1.0                  # retained 746a numerical guard
DV_R2_FLOOR = -1.0                  # FIX 4: FINITE per-fold R2 floor (R2 < -1 = uninformative noise)
DV_STANDARDIZE_TARGET = True        # FIX 1: standardise the DV training target (fold-train mean/std)
DV_TARGET_STD_FLOOR = 1e-6          # guard for a degenerate (near-constant) fold-train target std
DV_TRAIN_SEED = 90002               # DV optimisation RNG base (varied per fold, deterministic)

HARM_OBS_DIM = 51
Z_HARM_DIM = 32
WORLD_DIM = 32                       # = Z_HARM_DIM so harm_eval_z_harm_head (world_dim fallback) matches
HARM_OBS_CENTER = 12                 # harm_obs[12] = normalised hazard proximity at agent (SD-010 label)
HAZARD_NEIGHBOURHOOD_RADIUS = 2      # 5x5 neighbourhood for the local-density primary target

# Three STATE-DETERMINED raw-hazard_field targets collected by the shared module (= 746a).
TARGETS: List[Dict[str, Any]] = [
    {"key": "dens", "field": "Ydens", "label": "local_density", "primary": True},
    {"key": "at", "field": "Yat", "label": "at_agent", "primary": False},
    {"key": "next", "field": "Ynext", "label": "next_step", "primary": False},
]
PRIMARY_KEY = "dens"

ENV_KWARGS = {
    "size": 12,
    "num_hazards": 5,
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
RAW_TARGET_STD_FLOOR = 0.008   # PC_target_var: RAW target std floor (the 746 fix; RAW not normalised)
DV_ESTIMATOR_FLOOR = 0.0       # PC_dv_estimator_ok (NEW): mature-anchor DV must beat the mean (>=0)
#   Rationale: the target is ridge-decodable from z_harm at ~0.38 (746a IV). A viable trained MLP
#   evaluator (which contains a linear map as a special case) must therefore at least beat predicting
#   the mean on held-out at maturity. If the regularised + early-stopped + fold-averaged DV still
#   cannot (< 0), the DV MEASUREMENT is not viable -> non_contributory, never a false weakens.
# Reported-only reference (continuity with 743 / 746 / 746a / 740a; NOT gating).
ZWORLD_HARM_DECODE_740A_REF = 0.034  # 740a mature-z_world realized-harm decode (cited contrast)


def _r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _floor_r2(r2: float) -> float:
    """Finite per-fold R2 floor: NaN/inf and any value below DV_R2_FLOOR clamp to the floor
    (R2 < -1 = worse than predicting the mean = uninformative noise; the 746a -60/-166 fix)."""
    if not np.isfinite(r2):
        return DV_R2_FLOOR
    return max(r2, DV_R2_FLOOR)


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
    """Held-out R^2 of a closed-form ridge linear probe Z -> T (fixed split identical across arms;
    baseline is the TRAIN-set per-column mean). UNCHANGED from 746a -- the clean IV leg."""
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
    return r2 if np.isfinite(r2) else DV_R2_FLOOR


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


def _train_one_fold(agent: REEAgent, Ztr: torch.Tensor, Ttr: torch.Tensor,
                    Zval: torch.Tensor, Tval: torch.Tensor,
                    head_init_state: Dict[str, Any], opt_seed: int) -> None:
    """Train harm_eval_z_harm_head from the fixed shared init on (Ztr, Ttr) with weight decay and
    EARLY STOPPING on inner-validation MSE (patience). Restores the best-val state in place. This
    is the 746a divergence fix: the head halts at its best generalisation point, never trains into
    the held-out R2=-60 regime."""
    agent.e3.harm_eval_z_harm_head.load_state_dict(copy.deepcopy(head_init_state))
    for p in agent.e3.harm_eval_z_harm_head.parameters():
        p.requires_grad_(True)
    torch.manual_seed(opt_seed)
    opt = torch.optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(),
                           lr=DV_LR, weight_decay=DV_WEIGHT_DECAY)
    batch_rng = np.random.default_rng(opt_seed)

    ntr = Ztr.shape[0]
    best_val = float("inf")
    best_state = copy.deepcopy(agent.e3.harm_eval_z_harm_head.state_dict())
    patience = 0
    for _epoch in range(DV_MAX_EPOCHS):
        agent.e3.harm_eval_z_harm_head.train()
        perm = batch_rng.permutation(ntr)
        for start in range(0, ntr, DV_BATCH):
            bidx = torch.as_tensor(perm[start:start + DV_BATCH], dtype=torch.long)
            pred = agent.e3.harm_eval_z_harm(Ztr[bidx])
            loss = F.mse_loss(pred, Ttr[bidx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_z_harm_head.parameters(), DV_GRAD_CLIP)
            opt.step()
        agent.e3.harm_eval_z_harm_head.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(agent.e3.harm_eval_z_harm(Zval), Tval).item()
        if np.isfinite(val_mse) and val_mse < best_val - DV_EARLY_STOP_MIN_DELTA:
            best_val = val_mse
            best_state = copy.deepcopy(agent.e3.harm_eval_z_harm_head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= DV_EARLY_STOP_PATIENCE:
                break
    agent.e3.harm_eval_z_harm_head.load_state_dict(best_state)


def _train_dv_kfold(agent: REEAgent, Z: torch.Tensor, T: torch.Tensor,
                    head_init_state: Dict[str, Any]) -> Dict[str, Any]:
    """K-FOLD averaged DV (the 746b variance reducer). For each of DV_KFOLD random outer splits:
    standardise the target with the FOLD-TRAIN mean/std (FIX 1 -- R2 is affine-invariant, so this
    does not change what is measured; it only conditions the MSE optimisation for the tiny-magnitude
    target), carve an inner-validation set, train the head with weight decay + early stopping,
    evaluate held-out R2 on the standardised scale, floor it at DV_R2_FLOOR. Return the fold-averaged
    train/test R2 + gap + per-fold lists. Fixed, deterministic split + optimisation RNG per fold."""
    n = Z.shape[0]
    n_test = max(1, int(n * HELDOUT_FRAC))
    fold_r2_test: List[float] = []
    fold_r2_train: List[float] = []
    for fold in range(DV_KFOLD):
        split_perm = np.random.default_rng(DV_TRAIN_SEED + 1 + 100 * fold).permutation(n)
        te = torch.as_tensor(split_perm[:n_test], dtype=torch.long)
        tr_all = split_perm[n_test:]
        n_val = max(1, int(len(tr_all) * DV_INNER_VAL_FRAC))
        val = torch.as_tensor(tr_all[:n_val], dtype=torch.long)
        tr = torch.as_tensor(tr_all[n_val:], dtype=torch.long)
        # FIX 1: standardise the target with the FOLD-TRAIN statistics (affine -> R2-invariant).
        if DV_STANDARDIZE_TARGET:
            mu = T[tr].mean()
            sd = T[tr].std().clamp(min=DV_TARGET_STD_FLOOR)
        else:
            mu = torch.zeros(())
            sd = torch.ones(())
        Ts = (T - mu) / sd
        Ztr, Ttr = Z[tr], Ts[tr]
        Zval, Tval = Z[val], Ts[val]
        Zte, Tte = Z[te], Ts[te]
        _train_one_fold(agent, Ztr, Ttr, Zval, Tval, head_init_state,
                        opt_seed=DV_TRAIN_SEED + fold)
        agent.e3.harm_eval_z_harm_head.eval()
        with torch.no_grad():
            r2_test = _floor_r2(_r2(agent.e3.harm_eval_z_harm(Zte), Tte))
            r2_train = _floor_r2(_r2(agent.e3.harm_eval_z_harm(Ztr), Ttr))
        fold_r2_test.append(r2_test)
        fold_r2_train.append(r2_train)

    mean_test = float(np.mean(fold_r2_test))
    mean_train = float(np.mean(fold_r2_train))
    return {
        "r2_test": mean_test,
        "r2_train": mean_train,
        "gap": mean_train - mean_test,
        "per_fold_r2_test": fold_r2_test,
        "per_fold_r2_train": fold_r2_train,
        "fold_r2_test_sd": float(np.std(fold_r2_test)) if len(fold_r2_test) > 1 else 0.0,
    }


def _run_cell(seed: int, onset: int, steps_per_ep: int, collect_eps: int,
              dry_run: bool) -> Dict[str, Any]:
    print(f"Seed {seed} Condition onset_{onset}", flush=True)
    # config_slice carries the DV-estimator params so the arm_fingerprint correctly DIFFERS from
    # 746a (a different DV output). The expensive frozen z_harm prefix still cache-HITs off 746a
    # because the module's prefix-cache key is keyed ONLY on the maturation/collection params below,
    # which are unchanged -- the DV params never enter that key.
    config_slice = {
        "onset_episodes": onset,
        "steps_per_ep": steps_per_ep,
        "collect_episodes": collect_eps,
        "env_kwargs": ENV_KWARGS,
        "z_harm_dim": Z_HARM_DIM,
        "world_dim": WORLD_DIM,
        "ridge_lambda": RIDGE_LAMBDA,
        "hazard_neighbourhood_radius": HAZARD_NEIGHBOURHOOD_RADIUS,
        "targets": [t["label"] for t in TARGETS],
        # DV-estimator params (differ from 746a; enter the arm_fingerprint, NOT the prefix cache):
        "dv_kfold": DV_KFOLD,
        "dv_max_epochs": DV_MAX_EPOCHS,
        "dv_batch": DV_BATCH,
        "dv_lr": DV_LR,
        "dv_weight_decay": DV_WEIGHT_DECAY,
        "dv_early_stop_patience": DV_EARLY_STOP_PATIENCE,
        "dv_inner_val_frac": DV_INNER_VAL_FRAC,
        "dv_grad_clip": DV_GRAD_CLIP,
        "dv_r2_floor": DV_R2_FLOOR,
        "dv_standardize_target": DV_STANDARDIZE_TARGET,
    }
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
    ) as cell:
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

        zharm_var = float(Zharm.var(dim=0).mean().item())
        zharm_eff_rank = _eff_rank(Zharm)
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
        for t in TARGETS:
            T = data[t["field"]]                                   # raw-clipped [0,1] target
            iv = _ridge_heldout_r2(Zharm, T, DECODE_SPLIT_SEED)    # UNCHANGED clean IV
            dv = _train_dv_kfold(agent, Zharm, T, head_init)       # 746b k-fold variance-reduced DV
            k = t["key"]
            row[f"iv_{k}_decode_r2"] = iv
            row[f"dv_{k}_r2_test"] = dv["r2_test"]
            row[f"dv_{k}_r2_train"] = dv["r2_train"]
            row[f"dv_{k}_gap"] = dv["gap"]
            row[f"dv_{k}_fold_sd"] = dv["fold_r2_test_sd"]
            row[f"dv_{k}_per_fold_r2_test"] = dv["per_fold_r2_test"]
            row[f"target_{k}_raw_std"] = float(T.std().item())
        cell.stamp(row)

    print(f"verdict: {'PASS' if np.isfinite(row['dv_dens_r2_test']) else 'FAIL'}",
          flush=True)  # per-cell completion tick (aggregate verdict decided after all cells)
    return row


def _evaluate_target(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Scale-free calibrated-bound evaluation for ONE target. Criteria identical to 746a; adds the
    PC_dv_estimator_ok readiness precondition (mature-anchor DV must beat the mean)."""
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
    dv_mature: List[float] = []       # PC_dv_estimator_ok: mature-anchor DV
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
        dv_mature.append(_cell(seed, onset_max)[f"dv_{key}_r2_test"])

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
    mean_dv_mature = _mean(dv_mature)
    min_raw_std = min(r[f"target_{key}_raw_std"] for r in rows)

    # Preconditions (validity).
    pc_iv_moved = (mean_iv_delta > 0.0) and (mean_iv_rho > 0.0)
    pc_dv_decodable = mean_iv_mature >= DV_DECODABLE_FLOOR
    pc_target_var = min_raw_std >= RAW_TARGET_STD_FLOOR
    pc_dv_estimator_ok = mean_dv_mature >= DV_ESTIMATOR_FLOOR   # NEW (746b)
    preconditions_met = pc_iv_moved and pc_dv_decodable and pc_target_var and pc_dv_estimator_ok

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
        "per_seed_dv_mature": dv_mature,
        "mean_dv_delta": mean_dv_delta,
        "sd_dv_delta": sd_dv_delta,
        "mean_dv_rho": mean_dv_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_iv_rho": mean_iv_rho,
        "mean_couple_rho": mean_couple_rho,
        "mean_iv_mature": mean_iv_mature,
        "mean_dv_mature": mean_dv_mature,
        "min_raw_std": min_raw_std,
        "PC_iv_moved": pc_iv_moved,
        "PC_dv_decodable": pc_dv_decodable,
        "PC_target_var": pc_target_var,
        "PC_dv_estimator_ok": pc_dv_estimator_ok,
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
    else:
        seeds = SEEDS
        onsets = ONSET_EPISODES
        steps_per_ep = STEPS_PER_EP
        collect_eps = COLLECT_EPISODES

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for onset in onsets:
            rows.append(_run_cell(seed, onset, steps_per_ep, collect_eps, dry_run))

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
        if not primary["PC_dv_estimator_ok"]:
            reasons.append(
                f"DV estimator not viable: mean mature-anchor DV (k-fold, floored) "
                f"{primary['mean_dv_mature']:.3f} < {DV_ESTIMATOR_FLOOR} despite the target being "
                f"ridge-decodable at {primary['mean_iv_mature']:.3f} -- the trained harm_eval_z_harm "
                f"head cannot beat the mean on held-out at maturity, so the bound is unmeasurable on "
                f"this substrate (non_contributory, NOT a weakens)")
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
        "dv_kfold": DV_KFOLD,
        "dv_max_epochs": DV_MAX_EPOCHS,
        "dv_batch": DV_BATCH,
        "dv_lr": DV_LR,
        "dv_weight_decay": DV_WEIGHT_DECAY,
        "dv_early_stop_patience": DV_EARLY_STOP_PATIENCE,
        "dv_early_stop_min_delta": DV_EARLY_STOP_MIN_DELTA,
        "dv_inner_val_frac": DV_INNER_VAL_FRAC,
        "dv_grad_clip": DV_GRAD_CLIP,
        "dv_r2_floor": DV_R2_FLOOR,
        "dv_standardize_target": DV_STANDARDIZE_TARGET,
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
            "DV_ESTIMATOR_FLOOR": DV_ESTIMATOR_FLOOR,
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
        "extends": EXTENDS,
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
            "VARIANCE-REDUCED, VALID-DV re-estimate of V3-EXQ-746a's INV-089 harm-evaluator z_harm "
            "calibrated bound. 746a met all preconditions and WEAKENED (C1 rho=-0.53, C2 couple "
            "rho=-0.08, mean_dv_delta=-34.5), but that weakens is a DV MEASUREMENT ARTIFACT: its "
            "IV (local-density ridge-decode from frozen z_harm) is clean and moving (~0.37-0.40, "
            "PC_iv_moved True), while its DV (trained harm_eval_z_harm MLP head) produced held-out R2 "
            "-26..-69 with per-seed dv_delta -4.7..-166. Real-cell diagnosis (2026-07-15): the root "
            "cause is the tiny-magnitude Ydens target (raw std ~0.02) making MSE ill-conditioned for a "
            "fresh O(1)-output MLP -- the scale-robust closed-form ridge on the SAME (z_harm, target) "
            "gets +0.38 while the SGD MLP lands worse than the mean. This run fixes the DV estimator: "
            "(1) STANDARDISE the DV training target with fold-train mean/std (R2 affine-invariant -> "
            "measures the same quantity; measured DV -1.000 raw -> +0.33 standardised at n~195, "
            "+0.47/+0.52 at n~780), (2) K=5-fold averaged held-out R2 (kills single-split luck), "
            "(3) weight decay + inner-validation EARLY STOPPING, (4) FINITE per-fold R2 floor at -1, "
            "(5) collect_episodes 14->60 (~780 samples) for bound-test POWER. Because the collection "
            "widened, 746b does NOT reuse 746a's frozen prefix -- it MINTS its own reuse-eligible "
            "prefix (include_driver_script_in_hash=False) for future harm-leg siblings. NEW "
            "PC_dv_estimator_ok precondition: the mature-anchor DV (k-fold, floored, standardised) "
            "must beat the mean (>=0) given the target is ridge-decodable at ~0.4; if the fixed "
            "estimator still cannot, the DV is not viable and the run routes non_contributory (NOT "
            "weakens) -- so a broken measurement can never masquerade as evidence. LOAD-BEARING PASS = "
            "harm_eval_z_harm held-out quality on the PRIMARY (local_density) target rises monotonically "
            "with maturation (C1), tracks measured z_harm differentiation (C2), reliably across seeds "
            "(C3). A met-precondition C1/C2/C3 FAIL is now a TRUSTWORTHY weakens (reinforcing 746a for "
            "the right reason); a PASS reframes 746a's weakens as the DV artifact diagnosed here "
            "(governance may then mark 746a superseded). A PASS moves INV-089 provisional -> toward "
            "stable. Same 8 seeds as 746a (DV estimation, not seed count, was the bottleneck)."
        ),
        "cross_stream_contrast_note": (
            "Extends V3-EXQ-743 (INV-089 positive control) and re-estimates V3-EXQ-746a's DV. Same "
            "z_harm maturation recipe + IV ridge probe as 743/746a via the shared "
            "maturation_curriculum module; the collection is widened (collect_episodes 14->60) for DV "
            "power, so the frozen prefix is freshly MINTED (reuse-eligible for future harm-leg "
            "siblings), not reused from 746a. zharm_realized_decode_r2 (743's positive-control metric) "
            "and zharm_prox_decode_r2 are reported per cell for continuity. Re-derive brake "
            "(INV-089 substrate_ceiling autopsies: 0) HONOURED: z_harm observable, INV-089 claim, no "
            "z_world decode -- NOT a 740b. No substrate build owed (HarmEncoder + harm_eval_z_harm are "
            "SD-010 IMPLEMENTED); this is a DV-estimator refinement (letter b) of a same-question test."
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
    # config / seeds) -- multi-arm, so substrate_hash HOISTS from the per-cell arm_fingerprints.
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
    print("=== INV-089 harm-evaluator z_harm VARIANCE-REDUCED BOUND result (746b, n=8, 3 targets) ===",
          flush=True)
    for t in TARGETS:
        c = result["per_target"][t["key"]]
        tag = "PRIMARY" if t["primary"] else "secondary"
        print(f"  [{tag}:{t['label']}] pre_met={c['preconditions_met']} "
              f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.4f}; "
              f"dv_decodable={c['PC_dv_decodable']} mature_iv={c['mean_iv_mature']:.3f}; "
              f"raw_std={c['min_raw_std']:.4f}; "
              f"dv_estimator_ok={c['PC_dv_estimator_ok']} mature_dv={c['mean_dv_mature']:.3f}) "
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
