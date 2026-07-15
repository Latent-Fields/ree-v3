#!/opt/local/bin/python3
"""V3-EXQ-746c: INV-089 harm-evaluator z_harm CALIBRATED BOUND -- TARGET-CORRECTED +
BUDGET-DECOUPLED redesign of V3-EXQ-746b. Same scientific question (alphabetic suffix);
retains 746b's validated DV-estimator fix VERBATIM and fixes the two coupled defects that
starved 746b: (1) the gating target and (2) the shared IV/DV data budget.

WHY THIS RUN (the 746b result it responds to):
    V3-EXQ-746b FAILED STARVED on PC_iv_moved (mean_iv_delta +0.0020, mean_iv_rho -0.037):
    the local-density z_harm differentiation IV did not vary across onset, so the
    bound-coupling test could not run -- NOT a falsification, no weight on INV-089's bound.
    Full diagnosis: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-746b_2026-07-15.md.

    ROOT CAUSE (autopsy, now empirically confirmed from the 746a + 746b manifests):
    IV-power (onset differentiation contrast) and DV-power (harm_eval_z_harm estimator
    generalisation) were ANTI-COUPLED in a shared collect_episodes budget.
      - collect=14 (746a, ~195 samples): IV moved (dens delta +0.032, rho +0.51) but the DV
        estimator was broken (held-out R2 -26..-166 -- overfit).
      - collect=60 (746b, ~780 samples): the DV-estimator fix worked (mean_dv_mature 0.45)
        but the IV onset gradient WASHED OUT (dens delta +0.002) -- the mature-anchor IV rose
        so the implied onset-0 IV climbed to ~mature level and the delta collapsed to ~0.
    So local-density's onset gradient is a SMALL-SAMPLE UNDERFITTING ARTIFACT of the sparse
    collect=14 IV probe (adequate data lifts the onset-0 floor as much as the mature ceiling);
    z_harm decodes local-density about equally well immature vs mature once the probe is fed
    enough data. The IV that INV-089's bound rides on is therefore fragile for that target.

THE TWO FIXES (this run):
    FIX A -- TARGET CORRECTION (user-confirmed 2026-07-15, grounded in the two prior manifests'
      by_onset cross-checks). PRIMARY (gating) target is now `prox` = harm_obs[center]
      (the SD-010 hazard-proximity label; dataset field data["Prox"]). Among ALL collected
      targets it is the ONLY one whose z_harm decodability GENUINELY grows with maturation AND
      SURVIVES adequate sampling:
        prox IV onset 0->30:   0.705 -> 0.829 at collect=14 (746a)
                               0.839 -> 0.882 at collect=60 (746b)  <- gradient PERSISTS at large n
        dens IV onset 0->30:   0.366 -> 0.398 (746a, delta +0.032)
                               0.420 -> 0.422 (746b, delta +0.002)  <- washes out (the artifact)
        realized-harm Y IV:    ~0.03 flat (746a) / 0.064 -> 0.063 flat (746b)  <- no gradient
      prox is high-magnitude (~0.84) and is the canonical harm-relevant signal the harm stream
      exists to encode.

      READ-BACK CAVEAT (stated plainly): prox is the scalar the standalone HarmEncoder is
      MATURED on (via a throwaway nonlinear temp_head, discarded). 746a deliberately avoided it
      as a "read-back". The naive circularity is BROKEN here because the IV probe is a LINEAR
      ridge and the DV is a FRESH-INIT harm_eval_z_harm MLP trained on FROZEN z_harm -- the bound
      tests whether maturation makes a *fresh* evaluator better, which is exactly INV-089's
      claim. But a prox-based PASS is a VALID-BUT-SOFTER test of the bound than a fully-distinct
      target would be; governance should read a prox PASS as a lower-bound existence proof of
      the coupling, not the strongest possible form. dens (the confirmed artifact target) and
      realized-harm Y (the flat target) are RETAINED as reported, NON-gating secondaries so the
      manifest keeps the artifact/flat contrast explicit and auditable.

    FIX B -- DECOUPLE THE IV / DV DATA BUDGETS (the core mechanical fix). Collect ONE large pool
      per cell (COLLECT_EPISODES = 100, ~1300 samples), then PARTITION the sample indices into
      two DISJOINT sets via a fixed deterministic RNG (IV_POOL_FRAC = 0.4): an IV-pool
      (~520 samples) and a DV-pool (~780 samples, matching 746b's proven-adequate DV size). The
      IV ridge probe runs ENTIRELY within the IV-pool (its own held-out split); the DV k-fold
      estimator runs ENTIRELY within the DV-pool. Each leg thus gets adequate, independently
      sized sampling so NEITHER underfits (killing the artifact) NOR overfits, and the C2 IV<->DV
      coupling becomes a genuinely INDEPENDENT-SAMPLE coupling (no shared-sample inflation).

    FALSIFIER READOUT (rule out the small-sample artifact BEFORE trusting any bound result).
      The GATING IV (PC_iv_moved + the C-criteria coupling) is computed at FULL (adequate)
      IV-pool sampling -- so an artifact gradient that only exists at small-n correctly STARVES
      rather than sneaking a pass. ADDITIONALLY a diagnostic small-sample IV (iv_{key}_small) is
      computed on a ~150-sample subsample of the IV-pool (~746a collect=14 scale). Per target we
      emit BOTH mean_iv_delta_full and mean_iv_delta_small plus an explicit per-seed-averaged
      onset-0-vs-mature IV gap (mean_iv_onset0_vs_mature_gap) at full sampling. For the primary
      this is the direct falsifier: a genuine target (prox) keeps its gradient at full-n; an
      artifact target (dens) shows a gradient at small-n only. Reported prominently.

    DV-ESTIMATOR FIX (RETAINED VERBATIM from 746b -- the validated part). standardise the DV
      training target with the FOLD-TRAIN mean/std (R2 affine-invariant); K=5-fold averaged
      held-out R2; L2 weight-decay + inner-validation EARLY STOPPING; FINITE per-fold R2 floor
      at -1; and the PC_dv_estimator_ok rail (mean mature-anchor DV, k-fold, floored, must be
      >= 0.0 else the DV measurement is not viable -> route non_contributory, NOT weakens).

DESIGN (measurement-only, commitment-free frozen-representation curriculum-ORDER contrast;
        maturation recipe unchanged from 743/746a/746b, collection widened, budgets decoupled):
    Per (seed, onset) cell:
      1. Frozen z_harm prefix via mature_and_collect_harm (shared module; mints its own prefix).
      2. PARTITION sample indices -> disjoint IV-pool / DV-pool (fixed PARTITION_SEED).
      3. Per target: IV = ridge held-out R2 z_harm -> target within the IV-pool (full + a
         small-n diagnostic subsample). DV = harm_eval_z_harm k-fold averaged held-out R2
         predicting the STANDARDISED target within the DV-pool (fresh head per fold,
         weight-decayed + early-stopped, per-fold floored).
    IV per arm = onset episodes {0, 1, 4, 12, 30}. PRIMARY target = prox (SD-010 harm-proximity
    label). Secondaries (dens, real, at, next) reported with their own precondition gates,
    non-gating.

REGIME (unchanged -- 537b / 740a / 743 lesson): scheduled_external_hazard OFF so harm is
    predictable-from-state and z_harm differentiation is the binding constraint.

PRE-REGISTERED PASS CRITERIA (LOAD-BEARING; SCALE-FREE; on the PRIMARY prox DV):
    C1 dv_monotone:     mean_seed Spearman(onset, harm_eval_r2_test across arms) >= 0.80
    C2 bound_coupling:  mean_seed(DV[onset_max] - DV[onset_min]) > 0
                        AND mean_seed Spearman(IV_arm, DV_arm) >= 0.80
    C3 dv_reliable:     mean_seed(DV_delta) >= 2.0 * SD_seed(DV_delta)
    PASS = C1 and C2 and C3 on the PRIMARY target.

PRECONDITIONS (validity; on the PRIMARY prox target; unmet -> non_degenerate=False, unknown):
    PC_iv_moved:        FULL-pool IV: mean_seed(IV[onset_max]-IV[onset_min]) > 0 AND
                        mean_seed Spearman(onset,IV) > 0
    PC_dv_decodable:    mean_seed(IV[onset_max]) >= 0.05
    PC_target_var:      min over cells of the RAW primary-target std >= 0.008
    PC_dv_estimator_ok: mean_seed(DV[onset_max], k-fold, floored) >= 0.0

WHAT A PASS DOES: moves INV-089 provisional -> toward stable (the calibrated bound the
    provisional evidence_quality_note names as the missing test), on a valid, adequately-sampled,
    genuinely-maturation-dependent target -- caveated by the read-back note above. A
    met-precondition C1/C2/C3 FAIL is a genuine, now-TRUSTWORTHY WEAKENS. A starved IV /
    undecodable / near-constant / non-viable-DV primary routes non_contributory.

RE-DERIVE BRAKE (INV-089 substrate_ceiling autopsies: 1 -- the 746b starve; threshold 2, NOT
    braked) HONOURED: reads the z_harm stream, tests INV-089, does NOT decode scalar harm from
    z_world -> NOT a 740b. NO SUBSTRATE BUILD OWED (HarmEncoder + harm_eval_z_harm are SD-010
    IMPLEMENTED). This is a target-correction + budget-decoupling refinement of a same-question
    test (letter c), NOT a new mechanism and NOT a naive sparse-IV restore of collect=14.
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

EXPERIMENT_TYPE = "v3_exq_746c_inv089_harm_eval_z_harm_bound_target_corrected"
QUEUE_ID = "V3-EXQ-746c"
EXTENDS = "V3-EXQ-746b"        # target-corrected + budget-decoupled redesign of 746b's starved bound
CLAIM_IDS = ["INV-089"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants: maturation recipe BIT-IDENTICAL to 743/746a/746b ---
SEEDS = [42, 7, 19, 3, 11, 23, 47, 101]
ONSET_EPISODES = [0, 1, 4, 12, 30]            # onset_0 = fresh (untrained) HarmEncoder anchor
MATURE_PROGRESS_DENOM = max(ONSET_EPISODES)   # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 100              # FIX B: widened 60 -> 100 (~780 -> ~1300 samples/cell) so the
#                                    disjoint IV/DV partition leaves BOTH legs adequately sampled.
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear IV decode probe (= 746a/b)
DECODE_SPLIT_SEED = 90003            # fixed IV decode-probe split (all arms identical; = 746a/b)
COLLECT_SEED_BASE = 70000            # collection env + action RNG base (= 746a/b; module contract)
MATURE_SEED_BASE = 60000             # maturation env + action RNG base (= 746a/b; module contract)

# --- FIX B: disjoint IV / DV data-budget partition (the 746c decoupling) ---
PARTITION_SEED = 91001              # fixed, deterministic index partition (identical across arms)
IV_POOL_FRAC = 0.4                  # ~0.4 of the pool -> IV ridge probe; the rest -> DV k-fold
IV_SMALL_N = 150                    # diagnostic small-sample IV subsample (~746a collect=14 scale)

# --- DV estimator constants: RETAINED VERBATIM from 746b (the validated variance-reduced fix) ---
DV_KFOLD = 5                         # K repeated random held-out splits, DV R2 averaged
DV_MAX_EPOCHS = 80                   # max budget; early stopping halts well before this
DV_BATCH = 64
DV_LR = 1e-3
DV_WEIGHT_DECAY = 1e-3               # L2 on the harm_eval_z_harm head Adam (anti-divergence)
DV_EARLY_STOP_PATIENCE = 10         # epochs of no inner-val improvement before stopping
DV_EARLY_STOP_MIN_DELTA = 1e-5      # min inner-val MSE improvement to reset patience
DV_INNER_VAL_FRAC = 0.2             # inner-validation fraction of the fold train set
DV_GRAD_CLIP = 1.0                  # retained 746a numerical guard
DV_R2_FLOOR = -1.0                  # FINITE per-fold R2 floor (R2 < -1 = uninformative noise)
DV_STANDARDIZE_TARGET = True        # standardise the DV training target (fold-train mean/std)
DV_TARGET_STD_FLOOR = 1e-6          # guard for a degenerate (near-constant) fold-train target std
DV_TRAIN_SEED = 90002               # DV optimisation RNG base (varied per fold, deterministic)

HARM_OBS_DIM = 51
Z_HARM_DIM = 32
WORLD_DIM = 32                       # = Z_HARM_DIM so harm_eval_z_harm_head (world_dim fallback) matches
HARM_OBS_CENTER = 12                 # harm_obs[12] = normalised hazard proximity at agent (SD-010 label)
HAZARD_NEIGHBOURHOOD_RADIUS = 2      # 5x5 neighbourhood for the local-density secondary target

# Targets. PRIMARY = prox (the SD-010 harm-proximity label; genuine + sampling-robust gradient).
# Secondaries reported with their own precondition gates, NON-gating. dens (confirmed artifact)
# and real (flat) retained so the manifest keeps the artifact/flat contrast explicit.
TARGETS: List[Dict[str, Any]] = [
    {"key": "prox", "field": "Prox", "label": "harm_proximity_label", "primary": True},
    {"key": "dens", "field": "Ydens", "label": "local_density", "primary": False},
    {"key": "real", "field": "Y", "label": "realized_harm", "primary": False},
    {"key": "at", "field": "Yat", "label": "at_agent", "primary": False},
    {"key": "next", "field": "Ynext", "label": "next_step", "primary": False},
]
PRIMARY_KEY = "prox"

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
DV_ESTIMATOR_FLOOR = 0.0       # PC_dv_estimator_ok: mature-anchor DV must beat the mean (>=0)
# Reported-only reference (continuity with 743 / 746 / 746a / 746b / 740a; NOT gating).
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
    baseline is the TRAIN-set per-column mean). UNCHANGED from 746a/b -- the clean IV leg. Runs on
    whatever (Z, T) slice it is given (746c hands it the IV-pool, or the small-n IV subsample)."""
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


def _partition_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """FIX B: split the collected sample indices into a DISJOINT IV-pool and DV-pool via a fixed
    deterministic RNG (identical partition across all arms of a cell). n_iv = round(n*IV_POOL_FRAC),
    clamped so both pools keep >= 4 samples (the ridge/kfold minimum) when the pool is tiny (dry-run
    guard). Returns (iv_idx, dv_idx) as long tensors."""
    perm = np.random.default_rng(PARTITION_SEED).permutation(n)
    n_iv = int(round(n * IV_POOL_FRAC))
    n_iv = max(4, min(n_iv, n - 4)) if n >= 8 else max(1, n // 2)
    iv_idx = torch.as_tensor(perm[:n_iv], dtype=torch.long)
    dv_idx = torch.as_tensor(perm[n_iv:], dtype=torch.long)
    return iv_idx, dv_idx


def _train_one_fold(agent: REEAgent, Ztr: torch.Tensor, Ttr: torch.Tensor,
                    Zval: torch.Tensor, Tval: torch.Tensor,
                    head_init_state: Dict[str, Any], opt_seed: int) -> None:
    """Train harm_eval_z_harm_head from the fixed shared init on (Ztr, Ttr) with weight decay and
    EARLY STOPPING on inner-validation MSE (patience). Restores the best-val state in place. The
    746a divergence fix retained verbatim: the head halts at its best generalisation point, never
    trains into the held-out R2=-60 regime."""
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
    """K-FOLD averaged DV (746b variance reducer, retained verbatim). Runs on the DV-pool slice
    (FIX B). For each of DV_KFOLD random outer splits: standardise the target with the FOLD-TRAIN
    mean/std (R2 affine-invariant), carve an inner-validation set, train the head with weight decay
    + early stopping, evaluate held-out R2 on the standardised scale, floor at DV_R2_FLOOR. Return
    the fold-averaged train/test R2 + gap + per-fold lists."""
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
        # standardise the target with the FOLD-TRAIN statistics (affine -> R2-invariant).
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
    # config_slice carries the target set + the FIX B partition params + the 746b DV-estimator
    # params so the arm_fingerprint correctly DIFFERS from 746b (different targets, budgets, DV
    # output). The frozen z_harm prefix cache-MISSES 746b (collect_episodes changed 60->100), so
    # 746c MINTS its own reuse-eligible prefix for future harm-leg siblings.
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
        "primary_key": PRIMARY_KEY,
        # FIX B: disjoint IV/DV budget partition (enters the fingerprint):
        "partition_seed": PARTITION_SEED,
        "iv_pool_frac": IV_POOL_FRAC,
        "iv_small_n": IV_SMALL_N,
        # DV-estimator params (retained from 746b; enter the arm_fingerprint, NOT the prefix cache):
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

        # FIX B: disjoint IV-pool / DV-pool partition (identical split for every target in a cell).
        iv_idx, dv_idx = _partition_indices(int(Zharm.shape[0]))
        iv_small_idx = iv_idx[:max(4, min(IV_SMALL_N, iv_idx.shape[0]))]
        Z_iv = Zharm[iv_idx]
        Z_iv_small = Zharm[iv_small_idx]
        Z_dv = Zharm[dv_idx]

        zharm_var = float(Zharm.var(dim=0).mean().item())
        zharm_eff_rank = _eff_rank(Zharm)
        zharm_realized_decode_r2 = _ridge_heldout_r2(Z_iv, data["Y"][iv_idx], DECODE_SPLIT_SEED)
        zharm_prox_decode_r2 = _ridge_heldout_r2(Z_iv, data["Prox"][iv_idx], DECODE_SPLIT_SEED)

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
            "n_iv_pool": int(iv_idx.shape[0]),
            "n_iv_small": int(iv_small_idx.shape[0]),
            "n_dv_pool": int(dv_idx.shape[0]),
            "prefix_cache": prov["cache"],
        }
        for t in TARGETS:
            T = data[t["field"]]                                   # raw target (Prox / clipped [0,1])
            iv = _ridge_heldout_r2(Z_iv, T[iv_idx], DECODE_SPLIT_SEED)          # FULL IV-pool (gating)
            iv_small = _ridge_heldout_r2(Z_iv_small, T[iv_small_idx], DECODE_SPLIT_SEED)  # diagnostic
            dv = _train_dv_kfold(agent, Z_dv, T[dv_idx], head_init)            # DV-pool k-fold
            k = t["key"]
            row[f"iv_{k}_decode_r2"] = iv
            row[f"iv_{k}_decode_r2_small"] = iv_small
            row[f"dv_{k}_r2_test"] = dv["r2_test"]
            row[f"dv_{k}_r2_train"] = dv["r2_train"]
            row[f"dv_{k}_gap"] = dv["gap"]
            row[f"dv_{k}_fold_sd"] = dv["fold_r2_test_sd"]
            row[f"dv_{k}_per_fold_r2_test"] = dv["per_fold_r2_test"]
            row[f"target_{k}_raw_std"] = float(T.std().item())
        cell.stamp(row)

    print(f"verdict: {'PASS' if np.isfinite(row['dv_prox_r2_test']) else 'FAIL'}",
          flush=True)  # per-cell completion tick (aggregate verdict decided after all cells)
    return row


def _evaluate_target(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Scale-free calibrated-bound evaluation for ONE target. Criteria identical to 746b; adds the
    small-sample IV falsifier contrast (mean_iv_delta_small vs mean_iv_delta_full) and the explicit
    onset-0-vs-mature IV gap. The gating IV metrics use the FULL IV-pool (FIX B)."""
    onset_min = min(ONSET_EPISODES)
    onset_max = max(ONSET_EPISODES)

    def _cell(seed: int, onset: int) -> Dict[str, Any]:
        return next(r for r in rows if r["seed"] == seed and r["onset_episodes"] == onset)

    dv_delta: List[float] = []
    dv_rho: List[float] = []          # C1: DV monotone in onset
    iv_delta: List[float] = []        # PC_iv_moved: FULL-pool IV onset gap
    iv_rho: List[float] = []          # PC_iv_moved: FULL-pool IV monotone in onset
    iv_delta_small: List[float] = []  # diagnostic: small-n IV onset gap (the falsifier contrast)
    couple_rho: List[float] = []      # C2: DV tracks measured IV (across arms, independent samples)
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
        iv_delta_small.append(_cell(seed, onset_max)[f"iv_{key}_decode_r2_small"]
                              - _cell(seed, onset_min)[f"iv_{key}_decode_r2_small"])
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
    mean_iv_delta_small = _mean(iv_delta_small)
    mean_couple_rho = _mean(couple_rho)
    mean_iv_mature = _mean(iv_mature)
    mean_dv_mature = _mean(dv_mature)
    min_raw_std = min(r[f"target_{key}_raw_std"] for r in rows)

    # Preconditions (validity). PC_iv_moved uses the FULL IV-pool gradient (FIX B) so a
    # small-sample artifact gradient correctly STARVES rather than sneaking a pass.
    pc_iv_moved = (mean_iv_delta > 0.0) and (mean_iv_rho > 0.0)
    pc_dv_decodable = mean_iv_mature >= DV_DECODABLE_FLOOR
    pc_target_var = min_raw_std >= RAW_TARGET_STD_FLOOR
    pc_dv_estimator_ok = mean_dv_mature >= DV_ESTIMATOR_FLOOR
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
        "per_seed_iv_delta_small": iv_delta_small,
        "per_seed_couple_rho": couple_rho,
        "per_seed_iv_mature": iv_mature,
        "per_seed_dv_mature": dv_mature,
        "mean_dv_delta": mean_dv_delta,
        "sd_dv_delta": sd_dv_delta,
        "mean_dv_rho": mean_dv_rho,
        "mean_iv_delta": mean_iv_delta,                      # FULL-pool onset gap (gating)
        "mean_iv_delta_full": mean_iv_delta,                 # explicit alias (falsifier readout)
        "mean_iv_delta_small": mean_iv_delta_small,          # small-n onset gap (falsifier contrast)
        "mean_iv_onset0_vs_mature_gap": mean_iv_delta,       # onset_0 == onset_min -> == full gap
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
            entry[f"mean_iv_{k}_decode_r2_small"] = float(np.mean([c[f"iv_{k}_decode_r2_small"] for c in cells]))
        out[f"onset_{o}"] = entry
    return out


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [SEEDS[0]]
        onsets = [0, 2]
        steps_per_ep = 20
        collect_eps = 4
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
                f"primary (harm_proximity_label) z_harm differentiation gradient did not move at "
                f"FULL sampling: mean IV delta {primary['mean_iv_delta']:.4f} <= 0 or mean IV rank "
                f"rho {primary['mean_iv_rho']:.3f} <= 0 (bound test starved, not falsified; "
                f"small-n IV delta {primary['mean_iv_delta_small']:.4f} for the artifact contrast)")
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
        "partition_seed": PARTITION_SEED,
        "iv_pool_frac": IV_POOL_FRAC,
        "iv_small_n": IV_SMALL_N,
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
            "TARGET-CORRECTED + BUDGET-DECOUPLED re-estimate of V3-EXQ-746b's INV-089 harm-evaluator "
            "z_harm calibrated bound. 746b FAILED STARVED on PC_iv_moved (local-density IV onset "
            "gradient washed out: mean_iv_delta +0.002) -- NOT a falsification. Root cause: IV-power "
            "and DV-power were anti-coupled in a shared collect budget, and local-density's onset "
            "gradient is a SMALL-SAMPLE UNDERFITTING ARTIFACT (it exists at collect=14 but vanishes "
            "at collect=60 once the probe has enough data). TWO fixes: (A) TARGET CORRECTION -- the "
            "PRIMARY gating target is now prox = harm_obs[center] (the SD-010 hazard-proximity "
            "label), the ONLY collected target whose z_harm decodability genuinely grows with "
            "maturation AND survives adequate sampling (prox IV onset 0->30: 0.705->0.829 at "
            "collect=14 AND 0.839->0.882 at collect=60 -- gradient PERSISTS at large n; dens washes "
            "out; realized-harm Y is flat). dens + realized-harm retained as reported NON-gating "
            "secondaries so the artifact/flat contrast stays auditable. READ-BACK CAVEAT: prox is the "
            "scalar the HarmEncoder is matured on, so a prox PASS is a VALID-BUT-SOFTER existence "
            "proof of the bound (the naive circularity is broken by the LINEAR IV probe + FRESH-INIT "
            "MLP DV on FROZEN z_harm, which tests whether maturation makes a fresh evaluator better -- "
            "exactly INV-089's claim). (B) DECOUPLE IV/DV DATA BUDGETS -- collect one large pool "
            "(collect_episodes 60->100, ~1300 samples) then PARTITION indices into a DISJOINT IV-pool "
            "(~520) and DV-pool (~780) via a fixed RNG, so the IV ridge probe and DV k-fold each get "
            "adequate independently-sized sampling (IV can't underfit into a false gradient; DV can't "
            "overfit) and the C2 coupling is an independent-sample coupling. FALSIFIER READOUT: the "
            "gating IV is computed at FULL sampling (an artifact gradient starves rather than passes), "
            "and a diagnostic small-n IV (mean_iv_delta_small) is emitted per target as the direct "
            "artifact contrast. RETAINED VERBATIM from 746b: the validated DV-estimator fix "
            "(standardise target with fold-train mean/std; K=5-fold averaged held-out R2; weight-decay "
            "+ early-stopping; finite R2 floor at -1) and the PC_dv_estimator_ok rail (a still-broken "
            "DV routes non_contributory, never a false weakens). Because collect_episodes changed, "
            "746c does NOT reuse 746b's frozen prefix -- it MINTS its own reuse-eligible prefix "
            "(include_driver_script_in_hash=False) for future harm-leg siblings. LOAD-BEARING PASS = "
            "harm_eval_z_harm held-out quality on the PRIMARY (prox) target rises monotonically with "
            "maturation (C1), tracks measured z_harm differentiation (C2), reliably across seeds (C3). "
            "A met-precondition C1/C2/C3 FAIL is a TRUSTWORTHY weakens; a PASS moves INV-089 "
            "provisional -> toward stable (caveated by the read-back note). Same 8 seeds as 746a/b."
        ),
        "cross_stream_contrast_note": (
            "Extends V3-EXQ-746b (starved bound) and the 743 (INV-089 positive control) / 746a "
            "lineage. Same z_harm maturation recipe + IV ridge probe as 743/746a/746b via the shared "
            "maturation_curriculum module; the collection is widened (collect_episodes 60->100) for "
            "the disjoint IV/DV budget partition, so the frozen prefix is freshly MINTED "
            "(reuse-eligible for future harm-leg siblings), not reused from 746b. "
            "zharm_realized_decode_r2 (743's positive-control metric) and zharm_prox_decode_r2 are "
            "reported per cell for continuity (both computed on the IV-pool). Re-derive brake "
            "(INV-089 substrate_ceiling/non_contributory autopsies: 1 -- the 746b starve; threshold "
            "2, NOT braked) HONOURED: z_harm observable, INV-089 claim, no z_world decode -- NOT a "
            "740b. No substrate build owed (HarmEncoder + harm_eval_z_harm are SD-010 IMPLEMENTED); "
            "this is a target-correction + budget-decoupling refinement (letter c) of a same-question "
            "test, NOT a naive sparse-IV restore of collect=14."
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
    print("=== INV-089 harm-evaluator z_harm TARGET-CORRECTED BOUND result (746c, n=8, 5 targets) ===",
          flush=True)
    for t in TARGETS:
        c = result["per_target"][t["key"]]
        tag = "PRIMARY" if t["primary"] else "secondary"
        print(f"  [{tag}:{t['label']}] pre_met={c['preconditions_met']} "
              f"(iv_moved={c['PC_iv_moved']} iv_delta_full={c['mean_iv_delta']:.4f} "
              f"iv_delta_small={c['mean_iv_delta_small']:.4f}; "
              f"dv_decodable={c['PC_dv_decodable']} mature_iv={c['mean_iv_mature']:.3f}; "
              f"raw_std={c['min_raw_std']:.4f}; "
              f"dv_estimator_ok={c['PC_dv_estimator_ok']} mature_dv={c['mean_dv_mature']:.3f}) "
              f"C1={c['C1_dv_monotone']}(rho={c['mean_dv_rho']:.3f}) "
              f"C2={c['C2_bound_coupling']}(dvd={c['mean_dv_delta']:.4f},cpl={c['mean_couple_rho']:.3f}) "
              f"C3={c['C3_dv_reliable']}(2sd={DV_DELTA_SD_MULT * c['sd_dv_delta']:.4f}) "
              f"-> claim_pass={c['claim_pass']}", flush=True)
    prim = result["per_target"][PRIMARY_KEY]
    print(f"  FALSIFIER (primary={PRIMARY_KEY}): onset0-vs-mature IV gap FULL={prim['mean_iv_delta_full']:.4f} "
          f"vs SMALL-n={prim['mean_iv_delta_small']:.4f} "
          f"(genuine target keeps its gradient at full-n; an artifact target shows it at small-n only)",
          flush=True)
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
