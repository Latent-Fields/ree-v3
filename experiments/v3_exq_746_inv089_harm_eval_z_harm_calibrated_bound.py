#!/opt/local/bin/python3
"""V3-EXQ-746: INV-089 harm-evaluator z_harm CALIBRATED BOUND test -- the DEFERRED
core-coupling follow-up to V3-EXQ-743's positive control.

INV-089 (harm_evaluator_bounded_by_z_harm_differentiation, invariant / emergent from
ARC-003/019/027; PROVISIONAL as of 2026-07-12) is child (ii) of INV-064. Its CORE
assertion:

    E3's NOCICEPTIVE harm-evaluation QUALITY -- harm_eval_z_harm (e3_selector.py:660),
    which reads z_harm = HarmEncoder(harm_obs) (stack.py:119) -- is strictly bounded by
    z_harm representational differentiation. Productive harm-evaluator training cannot
    precede sufficient z_harm differentiation.

WHY THIS RUN (what V3-EXQ-743 left DEFERRED):
    743 PASSED, but its PASS gate was only the LOAD-BEARING POSITIVE CONTROL: realized
    harm is decodable-in-principle from the dedicated z_harm stream (mature
    zharm_decode_r2=0.124 >= 0.05 floor, C1) and RISES with HarmEncoder maturation (C2),
    dispositively crossing the 0.034-flat z_world decode of mis-streamed 740a. On that
    strength INV-089 was promoted candidate->provisional. But the CLAIM'S CORE ASSERTION
    -- harm_eval_z_harm HELD-OUT EVALUATOR QUALITY bounded by z_harm differentiation (the
    743 SEC_* coupling) -- was REPORTED-NOT-GATING (SEC_quality_gain / SEC_monotone both
    False) and its proper test was DEFERRED, for TWO reasons the 743 spot-check exposed:
      (a) 743's DV target was REALIZED post-random-action scalar harm Y=abs(harm_signal<0),
          which is UNDER-DETERMINED even from harm_obs (740a's flagged confound): harm at
          t+1 depends on the RANDOM action taken, so z_harm(harm_obs[t]) has irreducible
          action-noise variance w.r.t. Y -> the trained head sticks ~0.05, non-monotone.
      (b) The z_harm harm-decodability gradient is NARROW (delta ~0.018, high JL floor),
          so 740a's transplanted 0.03/0.15 MAGNITUDE thresholds are mis-calibrated for it.

THE TWO CALIBRATION FIXES (this run):
    FIX (a) STATE-DETERMINED DV TARGET. The DV/IV target is the GROUND-TRUTH current-cell
      hazard exposure Y* = clip(hazard_field[agent_x, agent_y], 0, 1), read from the env
      BEFORE the action and aligned to the SAME state s_t that z_harm encodes. This is a
      DETERMINISTIC function of the current state -- the true harm exposure a nociceptive
      evaluator ought to output -- with the action-selection noise integrated out. It is
      DISTINCT from the maturation supervision label harm_obs[12] (the NORMALISED proxy
      VIEW of proximity under use_proxy_fields), so decoding the RAW ground-truth harm
      from z_harm is a genuine (non-vacuous) differentiation test, not a proxy read-back.
      NOT realized post-random-action harm -> fixes 743's under-determination.
    FIX (b) z_harm-SCALED (SCALE-FREE) THRESHOLDS. The load-bearing gates are RANK-based
      (Spearman monotonicity) + a reliability effect-size (mean_delta >= 2*SD_seed) +
      an IV<->DV COUPLING rank test -- NONE impose an absolute 740a-scaled magnitude floor
      (0.03/0.15). A narrow-but-monotone-and-reliable gradient PASSES; the criteria are
      immune to the high-JL-floor magnitude mis-calibration.

DESIGN (measurement-only, commitment-free frozen-representation curriculum-ORDER
        contrast -- 743's exact maturation + collection trajectory, RE-TARGETED onto the
        state-determined hazard exposure; methodological sibling of INV-088's 744a):
    Per (seed, onset) cell:
      1. Build agent (056c SD-010 wiring: world_dim = z_harm_dim = 32 so
         agent.e3.harm_eval_z_harm_head matches the 32-dim standalone HarmEncoder) +
         a STANDALONE HarmEncoder(51, 32). Snapshot the fresh harm_eval_z_harm_head init.
      2. MATURE the HarmEncoder for `onset` episodes on the canonical SD-010 label
         harm_obs[12] with a THROWAWAY proximity head (discarded). onset=0 -> fresh
         (untrained) encoder = the genuinely-immature anchor. (743-identical: same
         MATURE_SEED_BASE, same recipe -> reproduces 743's z_harm differentiation.)
      3. FREEZE the HarmEncoder.
      4. Collect a SHARED frozen dataset (fixed action seq, fixed seeded env,
         maturity-independent -> bit-identical raw trajectory across the onset arms of a
         seed; the ONLY per-arm variable is the frozen z_harm). Per step record:
           z_harm[t]  = HarmEncoder(harm_obs[t])            (frozen; per-arm variable)
           Ystate[t]  = clip(hazard_field[agent], 0, 1)     (STATE-DETERMINED DV/IV target)
           Yreal[t]   = abs(harm_signal<0)                  (743 realized-harm cross-check)
           Prox[t]    = harm_obs[t][12]                     (maturation label; reported only)
         Ystate is min-max normalised over the (shared) trajectory -> [0,1], identical
         across arms.
      5. IV per arm = z_harm differentiation for the STATE target = ridge held-out R^2 of
         z_harm[t] -> Ystate[t]. Rises with maturation (JL-safe: a random HarmEncoder does
         not make the ground-truth hazard linearly readable from the normalised proxy
         view; training does).
      6. DV per arm = harm_eval_z_harm held-out R^2 predicting Ystate (re-init head to the
         fixed shared init, FIXED budget on the frozen (z_harm, Ystate) tensors).

    IV (per arm): onset episodes {0, 1, 4, 12, 30}  (HarmEncoder maturity at DV onset).
    SEEDS: 8 (mirror 744a) -- 743/744 showed 3 seeds under-power the effect-size gate.

REGIME (unchanged -- 537b / 740a / 743 lesson): scheduled_external_hazard OFF so harm is
    predictable-from-state and z_harm differentiation is the binding constraint (not a
    by-design-unpredictable ceiling), the regime in which INV-089 is falsifiable.

PRE-REGISTERED PASS CRITERIA (LOAD-BEARING -- the CALIBRATED bound; SCALE-FREE, on the
                             STATE-DETERMINED harm_eval_z_harm DV):
    C1 dv_monotone:     mean_seed Spearman(onset, harm_eval_r2_test across arms) >= 0.80
                        -- harm_eval_z_harm held-out quality RISES MONOTONICALLY with
                        HarmEncoder maturation (the task's literal ask; rank-based ->
                        immune to the narrow-magnitude mis-calibration).
    C2 bound_coupling:  mean_seed(DV[onset_max] - DV[onset_min]) > 0
                        AND mean_seed Spearman(IV_arm, DV_arm) >= 0.80
                        -- evaluator quality TRACKS (is bounded by) measured z_harm
                        differentiation, the CORE INV-089 assertion, scored by COUPLING
                        not absolute magnitude.
    C3 dv_reliable:     mean_seed(DV_delta) >= 2.0 * SD_seed(DV_delta)
                        -- the rise is RELIABLE across seeds (scale-free t-stat-like
                        effect-size; keeps 744a's reliability discipline WITHOUT the
                        mis-calibrated 0.15 absolute floor).
    PASS = C1 and C2 and C3.

PRECONDITIONS (validity -- only genuine measurement/starvation failure routes
               non_contributory; a met-precondition C1/C2/C3 failure is a real WEAKENS):
    PC_iv_moved:     mean_seed(IV[onset_max] - IV[onset_min]) > 0 AND
                     mean_seed Spearman(onset, IV) > 0
                     -- the z_harm differentiation gradient EXISTS and is directional.
                     NO magnitude floor (narrow gradient allowed). If the IV does not
                     move, the bound test is STARVED, not falsified -> non_contributory.
    PC_dv_decodable: mean_seed(IV[onset_max]) >= 0.05
                     -- Ystate decodable-in-principle from the MATURE z_harm (positive
                     control that the stream carries the state harm signal at all).
    PC_target_var:   min over cells of Ystate std >= 0.02  (target not near-constant).
    Unmet -> non_degenerate=False, evidence_direction="unknown".

WHAT A PASS DOES: moves INV-089 provisional -> toward stable (the calibrated bound the
    provisional evidence_quality_note names as the missing test). A met-precondition FAIL
    is a genuine WEAKENS (or, like 744->744a, an under-powered near-miss -> re-estimate).

RE-DERIVE BRAKE (INV-064 740-cluster) HONOURED: reads the z_harm stream (a DIFFERENT
    observable than the z_world 740-cluster), tests INV-089 (a distinct claim), and does
    NOT decode scalar harm from z_world -> NOT a same-observable 740b. Supersedes NOTHING
    (743's positive-control evidence stands; this ADDS the deferred core coupling).

NO SUBSTRATE BUILD OWED: HarmEncoder (stack.py:119) + harm_eval_z_harm (e3_selector.py:660)
    exist and are SD-010 IMPLEMENTED. Testable-now.
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
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.latent.stack import HarmEncoder  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_746_inv089_harm_eval_z_harm_calibrated_bound"
QUEUE_ID = "V3-EXQ-746"
SUPERSEDES = None            # different target + calibrated criteria; 743's pos-ctrl stands
CLAIM_IDS = ["INV-089"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants (743 maturation/collection recipe; seeds 3 -> 8 like 744a) ---
SEEDS = [42, 7, 19, 3, 11, 23, 47, 101]
# onset_0 = fresh (untrained) HarmEncoder = the genuinely-immature anchor.
ONSET_EPISODES = [0, 1, 4, 12, 30]
MATURE_PROGRESS_DENOM = max(ONSET_EPISODES)   # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared harm-dataset collection (identical per arm)
DV_EPOCHS = 40                       # FIXED harm_eval_z_harm head training budget (all arms)
DV_BATCH = 64
DV_LR = 1e-3
MATURE_LR = 1e-3                     # HarmEncoder + throwaway proximity head co-train LR
MATURE_HIDDEN = 32                  # throwaway proximity-head hidden width (discarded)
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear decode probes
DV_TRAIN_SEED = 90002                # fixed DV optimisation RNG (all arms identical)
DECODE_SPLIT_SEED = 90003            # fixed decode-probe train/test split (all arms identical)
COLLECT_SEED_BASE = 70000            # collection env + action RNG base (per-seed, arm-independent)
MATURE_SEED_BASE = 60000             # maturation env + action RNG base (per-seed, arm-independent)

HARM_OBS_DIM = 51
Z_HARM_DIM = 32
WORLD_DIM = 32                       # = Z_HARM_DIM so harm_eval_z_harm_head (world_dim fallback) matches
HARM_OBS_CENTER = 12                 # harm_obs[12] = normalised hazard proximity at agent cell (SD-010 label)

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
# Load-bearing PASS gates.
MONOTONE_RHO_MIN = 0.80        # C1: DV monotone in onset (rank-based -> narrow gradient OK)
COUPLING_RHO_MIN = 0.80        # C2: DV tracks measured z_harm differentiation (IV<->DV rank)
DV_DELTA_SD_MULT = 2.0         # C3: reliability effect-size (scale-free t-stat-like)
# Preconditions (validity).
DV_DECODABLE_FLOOR = 0.05      # PC_dv_decodable: mature-anchor Ystate decodability positive control
TARGET_STD_FLOOR = 0.02        # PC_target_var: state target must not be near-constant
# Reported-only reference (continuity with 743 / 740a; NOT gating).
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
    """Held-out R^2 of a closed-form ridge linear probe Z -> T.

    Fixed train/test split (identical indices across arms -- depends only on n, which is
    identical across the onset arms of a seed because the collection trajectory is shared).
    Baseline for R^2 is the TRAIN-set per-column mean (a proper held-out R^2, so an
    uninformative representation scores <= 0). Pure read-out of the FROZEN z_harm -- no
    gradients, distinct from the trained harm_eval_z_harm MLP (the DV).
    """
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


def _build_agent(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    """056c SD-010 wiring (743-identical): world_dim = z_harm_dim = 32 so the agent's
    harm_eval_z_harm_head (which falls back to world_dim when the E3 config carries no
    z_harm_dim) matches the 32-dim standalone HarmEncoder output.
    """
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(cfg)
    return agent, env


def _row_vec(x: Any) -> torch.Tensor:
    """Coerce an obs harm field to a [1, D] float32 row tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return t.reshape(1, -1)


def _hazard_at_agent(env: CausalGridWorldV2) -> float:
    """Ground-truth hazard exposure at the agent's CURRENT cell, clipped to [0, 1].

    Read from the env BEFORE the action so it aligns with the state s_t that z_harm
    encodes (info['hazard_field_at_agent'] is emitted post-step, reflecting s_{t+1}). This
    is the STATE-DETERMINED harm target: a deterministic function of the current state,
    distinct from the normalised proxy view harm_obs[12] the encoder trains on, with the
    action-selection noise of 743's realized-harm target integrated out.
    """
    ax, ay = int(env.agent_x), int(env.agent_y)
    return float(np.clip(env.hazard_field[ax, ay], 0.0, 1.0))


def _mature_harm_encoder(harm_enc: HarmEncoder, seed: int, onset: int,
                         steps_per_ep: int, denom: int) -> int:
    """Mature the HarmEncoder for `onset` episodes on the SD-010 supervision label
    harm_obs[12], co-trained with a THROWAWAY proximity head (discarded). onset=0 -> no
    training (fresh encoder, the immature anchor). 743-identical (same MATURE_SEED_BASE
    and recipe, so the z_harm differentiation gradient reproduces 743's).
    """
    if onset <= 0:
        return 0
    mature_env = CausalGridWorldV2(seed=MATURE_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(MATURE_SEED_BASE + seed)
    action_dim = mature_env.action_dim
    temp_head = nn.Sequential(
        nn.Linear(Z_HARM_DIM, MATURE_HIDDEN),
        nn.ReLU(),
        nn.Linear(MATURE_HIDDEN, 1),
    )
    opt = torch.optim.Adam(
        list(harm_enc.parameters()) + list(temp_head.parameters()), lr=MATURE_LR)
    harm_enc.train()
    temp_head.train()
    n_steps = 0
    last_loss = float("nan")
    for ep in range(onset):
        _, obs_dict = mature_env.reset()
        for _step in range(steps_per_ep):
            harm_obs_t = _row_vec(obs_dict.get("harm_obs"))
            label = harm_obs_t[:, HARM_OBS_CENTER:HARM_OBS_CENTER + 1]  # [1,1] in [0,1]
            z_harm = harm_enc(harm_obs_t)
            pred = temp_head(z_harm)
            loss = F.mse_loss(pred, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
            opt.step()
            last_loss = float(loss.item())
            n_steps += 1

            action = int(act_rng.integers(0, action_dim))
            _, _harm_signal, done, _info, obs_dict = mature_env.step(action)
            if done:
                break
        if (ep + 1) % 5 == 0 or ep + 1 == onset:
            print(f"  [train] mat seed={seed} onset={onset} ep {ep + 1}/{denom} "
                  f"loss={last_loss:.4f}", flush=True)
    return n_steps


def _collect_frozen_dataset(harm_enc: HarmEncoder, seed: int, steps_per_ep: int,
                            n_episodes: int) -> Dict[str, torch.Tensor]:
    """Replay a FIXED action sequence through a FIXED collection env, encoding z_harm with
    the (frozen) HarmEncoder. Identical raw trajectory + targets across onset arms; only
    the frozen encoder differs.

    Per step:
      Zharm  = HarmEncoder(harm_obs[t])          (frozen encoding, the only per-arm var)
      Ystate = clip(hazard_field[agent], 0, 1)   (STATE-DETERMINED DV/IV target, aligned
                                                   to s_t -- read BEFORE the action)
      Yreal  = abs(harm_signal) if < 0 else 0    (743 realized-harm cross-check; reported)
      Prox   = harm_obs[t][12]                    (SD-010 maturation label; reported only)
    """
    collect_env = CausalGridWorldV2(seed=COLLECT_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(COLLECT_SEED_BASE + seed)
    action_dim = collect_env.action_dim

    zh_list: List[torch.Tensor] = []
    ystate_list: List[float] = []
    yreal_list: List[float] = []
    prox_list: List[float] = []

    harm_enc.eval()
    with torch.no_grad():
        for _ep in range(n_episodes):
            _, obs_dict = collect_env.reset()
            for _step in range(steps_per_ep):
                harm_obs_t = _row_vec(obs_dict.get("harm_obs"))
                z_harm = harm_enc(harm_obs_t).detach().cpu()
                prox = float(harm_obs_t[0, HARM_OBS_CENTER].item())
                # STATE-DETERMINED target aligned to the CURRENT state (pre-action).
                ystate = _hazard_at_agent(collect_env)

                action = int(act_rng.integers(0, action_dim))
                _, harm_signal, done, _info, obs_next = collect_env.step(action)
                yreal = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

                zh_list.append(z_harm)
                ystate_list.append(ystate)
                yreal_list.append(yreal)
                prox_list.append(prox)

                obs_dict = obs_next
                if done:
                    break

    Zharm = torch.cat(zh_list, dim=0)
    Ystate_raw = torch.tensor(ystate_list, dtype=torch.float32).unsqueeze(1)
    # Min-max normalise Ystate over the (shared) trajectory -> [0,1]; identical across the
    # onset arms of a seed because the raw target is trajectory-determined, not encoder-
    # determined (mirrors 744a's dataset-normalised target).
    ymin = Ystate_raw.min()
    ymax = Ystate_raw.max()
    Ystate = (Ystate_raw - ymin) / (ymax - ymin).clamp(min=1e-6)
    Yreal = torch.tensor(yreal_list, dtype=torch.float32).unsqueeze(1)
    Prox = torch.tensor(prox_list, dtype=torch.float32).unsqueeze(1)
    return {"Zharm": Zharm, "Ystate": Ystate, "Ystate_raw": Ystate_raw,
            "Yreal": Yreal, "Prox": Prox}


def _freeze(harm_enc: HarmEncoder) -> None:
    for p in harm_enc.parameters():
        p.requires_grad_(False)


def _train_dv_and_eval(agent: REEAgent, Z: torch.Tensor, T: torch.Tensor,
                       head_init_state: Dict[str, Any], dv_epochs: int) -> Dict[str, float]:
    """Re-init harm_eval_z_harm_head to the fixed shared init, train for a FIXED budget on
    the frozen (z_harm, Ystate) tensors, return train/test R^2 + gap. Fixed, bit-identical
    init + optimisation RNG + split across all arms.
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
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == dv_epochs:
            print(f"  [dv-fit] epoch {epoch + 1}/{dv_epochs}", flush=True)

    agent.eval()
    with torch.no_grad():
        r2_train = _r2(agent.e3.harm_eval_z_harm(Ztr), Ttr)
        r2_test = _r2(agent.e3.harm_eval_z_harm(Zte), Tte)
    return {
        "harm_eval_r2_train": r2_train,
        "harm_eval_r2_test": r2_test,
        "harm_eval_gap": r2_train - r2_test,
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
        "mature_lr": MATURE_LR,
        "ridge_lambda": RIDGE_LAMBDA,
        "target": "state_determined_hazard_at_agent",
    }
    # Cells are reuse-INELIGIBLE: the frozen z_harm + state target are functions of the
    # shared maturation+collection trajectory, produced by THIS driver's inline code (not
    # a hashed _lib module), so the fingerprint must fold in the driver script to stay
    # sound. Fingerprint emitted (validator requirement) but flagged ineligible. (A future
    # z_harm sibling wanting cross-driver reuse should factor a state-determined harm
    # collect+mature into experiments/_lib/baselines/maturation_curriculum.py first.)
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=["state_determined_target_inline_driver_collection"],
    ) as cell:
        agent, _env = _build_agent(seed)
        harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
        # Snapshot the fresh harm_eval_z_harm_head init BEFORE any training so every arm
        # starts the DV head from bit-identical weights.
        head_init_state = copy.deepcopy(agent.e3.harm_eval_z_harm_head.state_dict())

        # 1. Mature the HarmEncoder (onset=0 -> fresh encoder anchor).
        n_mature_steps = _mature_harm_encoder(
            harm_enc, seed, onset, steps_per_ep, MATURE_PROGRESS_DENOM)
        # 2. Freeze.
        _freeze(harm_enc)
        # 3. Collect the shared frozen dataset (identical trajectory across arms).
        data = _collect_frozen_dataset(harm_enc, seed, steps_per_ep, collect_eps)

        # Differentiation mediators at freeze.
        zharm_var = float(data["Zharm"].var(dim=0).mean().item())
        zharm_eff_rank = _eff_rank(data["Zharm"])
        # IV: linear z_harm differentiation for the STATE-DETERMINED target.
        zharm_state_decode_r2 = _ridge_heldout_r2(data["Zharm"], data["Ystate"], DECODE_SPLIT_SEED)
        # Reported cross-checks (NOT the IV): 743 realized-harm decode + proximity decode.
        zharm_realized_decode_r2 = _ridge_heldout_r2(data["Zharm"], data["Yreal"], DECODE_SPLIT_SEED)
        zharm_prox_decode_r2 = _ridge_heldout_r2(data["Zharm"], data["Prox"], DECODE_SPLIT_SEED)

        # 4. Controlled frozen harm_eval_z_harm training + readout (the DV) on Ystate.
        dv = _train_dv_and_eval(agent, data["Zharm"], data["Ystate"], head_init_state, dv_epochs)

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "n_mature_steps": n_mature_steps,
            "zharm_var": zharm_var,
            "zharm_eff_rank": zharm_eff_rank,
            "zharm_state_decode_r2": zharm_state_decode_r2,       # IV (differentiation for Ystate)
            "zharm_realized_decode_r2": zharm_realized_decode_r2,  # 743 cross-check (reported)
            "zharm_prox_decode_r2": zharm_prox_decode_r2,          # maturation-label decode (reported)
            "ystate_std": float(data["Ystate"].std().item()),
            "ystate_raw_std": float(data["Ystate_raw"].std().item()),
            "n_samples": int(data["Zharm"].shape[0]),
            **dv,
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

    per_seed_dv_delta: List[float] = []
    per_seed_dv_rho: List[float] = []         # C1: DV monotone in onset
    per_seed_iv_delta: List[float] = []
    per_seed_iv_rho: List[float] = []         # PC_iv_moved: IV monotone in onset
    per_seed_couple_rho: List[float] = []     # C2: DV tracks measured IV (across arms)
    per_seed_iv_mature: List[float] = []      # PC_dv_decodable: mature-anchor IV
    for seed in SEEDS:
        onsets = [float(o) for o in ONSET_EPISODES]
        dvs = [_cell(seed, o)["harm_eval_r2_test"] for o in ONSET_EPISODES]
        ivs = [_cell(seed, o)["zharm_state_decode_r2"] for o in ONSET_EPISODES]
        per_seed_dv_delta.append(_cell(seed, onset_max)["harm_eval_r2_test"]
                                 - _cell(seed, onset_min)["harm_eval_r2_test"])
        per_seed_dv_rho.append(_spearman(onsets, dvs))
        per_seed_iv_delta.append(_cell(seed, onset_max)["zharm_state_decode_r2"]
                                 - _cell(seed, onset_min)["zharm_state_decode_r2"])
        per_seed_iv_rho.append(_spearman(onsets, ivs))
        per_seed_couple_rho.append(_spearman(ivs, dvs))
        per_seed_iv_mature.append(_cell(seed, onset_max)["zharm_state_decode_r2"])

    def _mean(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _sd(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = _mean(v)
        return float((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5)

    mean_dv_delta = _mean(per_seed_dv_delta)
    sd_dv_delta = _sd(per_seed_dv_delta)
    mean_dv_rho = _mean(per_seed_dv_rho)
    mean_iv_delta = _mean(per_seed_iv_delta)
    mean_iv_rho = _mean(per_seed_iv_rho)
    mean_couple_rho = _mean(per_seed_couple_rho)
    mean_iv_mature = _mean(per_seed_iv_mature)
    min_target_std = min(r["ystate_std"] for r in rows)

    # Preconditions (validity). Narrow gradient allowed -> IV move gate is directional
    # (>0 + positive rank), NO absolute magnitude floor.
    pc_iv_moved = (mean_iv_delta > 0.0) and (mean_iv_rho > 0.0)
    pc_dv_decodable = mean_iv_mature >= DV_DECODABLE_FLOOR
    pc_target_var = min_target_std >= TARGET_STD_FLOOR
    preconditions_met = pc_iv_moved and pc_dv_decodable and pc_target_var

    # LOAD-BEARING PASS criteria (the CALIBRATED bound; scale-free).
    c1_dv_monotone = mean_dv_rho >= MONOTONE_RHO_MIN
    c2_bound_coupling = (mean_dv_delta > 0.0) and (mean_couple_rho >= COUPLING_RHO_MIN)
    c3_dv_reliable = mean_dv_delta >= (DV_DELTA_SD_MULT * sd_dv_delta)

    return {
        "onset_min": onset_min,
        "onset_max": onset_max,
        "per_seed_dv_delta": per_seed_dv_delta,
        "per_seed_dv_rho": per_seed_dv_rho,
        "per_seed_iv_delta": per_seed_iv_delta,
        "per_seed_iv_rho": per_seed_iv_rho,
        "per_seed_couple_rho": per_seed_couple_rho,
        "per_seed_iv_mature": per_seed_iv_mature,
        "mean_dv_delta": mean_dv_delta,
        "sd_dv_delta": sd_dv_delta,
        "mean_dv_rho": mean_dv_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_iv_rho": mean_iv_rho,
        "mean_couple_rho": mean_couple_rho,
        "mean_iv_mature": mean_iv_mature,
        "min_target_std": min_target_std,
        "zworld_harm_decode_740a_ref": ZWORLD_HARM_DECODE_740A_REF,
        "PC_iv_moved": pc_iv_moved,
        "PC_dv_decodable": pc_dv_decodable,
        "PC_target_var": pc_target_var,
        "preconditions_met": preconditions_met,
        "C1_dv_monotone": c1_dv_monotone,
        "C2_bound_coupling": c2_bound_coupling,
        "C3_dv_reliable": c3_dv_reliable,
    }


def _aggregate_by_onset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ONSET_EPISODES:
        cells = [r for r in rows if r["onset_episodes"] == o]
        out[f"onset_{o}"] = {
            "mean_harm_eval_r2_test": float(np.mean([c["harm_eval_r2_test"] for c in cells])),
            "mean_harm_eval_r2_train": float(np.mean([c["harm_eval_r2_train"] for c in cells])),
            "mean_harm_eval_gap": float(np.mean([c["harm_eval_gap"] for c in cells])),
            "mean_zharm_state_decode_r2": float(np.mean([c["zharm_state_decode_r2"] for c in cells])),
            "mean_zharm_realized_decode_r2": float(np.mean([c["zharm_realized_decode_r2"] for c in cells])),
            "mean_zharm_prox_decode_r2": float(np.mean([c["zharm_prox_decode_r2"] for c in cells])),
            "mean_zharm_var": float(np.mean([c["zharm_var"] for c in cells])),
            "mean_zharm_eff_rank": float(np.mean([c["zharm_eff_rank"] for c in cells])),
        }
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
            rows.append(_run_cell(seed, onset, steps_per_ep, collect_eps,
                                  dv_epochs, dry_run))

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
    # PASS = the CALIBRATED bound: harm_eval_z_harm held-out quality rises monotonically
    # with maturation (C1), tracks measured z_harm differentiation (C2), reliably (C3).
    claim_pass = (criteria["C1_dv_monotone"]
                  and criteria["C2_bound_coupling"]
                  and criteria["C3_dv_reliable"])

    if not preconditions_met:
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        reasons = []
        if not criteria["PC_iv_moved"]:
            reasons.append(
                f"z_harm differentiation gradient did not move: mean IV delta "
                f"{criteria['mean_iv_delta']:.4f} <= 0 or mean IV rank rho "
                f"{criteria['mean_iv_rho']:.3f} <= 0 (bound test starved, not falsified)")
        if not criteria["PC_dv_decodable"]:
            reasons.append(
                f"state target not decodable-in-principle from mature z_harm: mean "
                f"mature-anchor zharm_state_decode_r2 {criteria['mean_iv_mature']:.3f} "
                f"< {DV_DECODABLE_FLOOR}")
        if not criteria["PC_target_var"]:
            reasons.append(
                f"state target near-constant: min Ystate std "
                f"{criteria['min_target_std']:.4f} < {TARGET_STD_FLOOR}")
        degeneracy_reason = "; ".join(reasons)
    else:
        outcome = "PASS" if claim_pass else "FAIL"
        # supports = harm_eval_z_harm quality reliably rises with, and tracks, z_harm
        # differentiation (the calibrated bound INV-089 asserts); weakens = it is
        # flat / non-monotone / does-not-track across a validly-moving gradient.
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
    full_config = {
        "seeds": SEEDS,
        "onset_episodes": ONSET_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "collect_episodes": COLLECT_EPISODES,
        "dv_epochs": DV_EPOCHS,
        "dv_batch": DV_BATCH,
        "dv_lr": DV_LR,
        "mature_lr": MATURE_LR,
        "heldout_frac": HELDOUT_FRAC,
        "ridge_lambda": RIDGE_LAMBDA,
        "collect_seed_base": COLLECT_SEED_BASE,
        "mature_seed_base": MATURE_SEED_BASE,
        "dv_train_seed": DV_TRAIN_SEED,
        "decode_split_seed": DECODE_SPLIT_SEED,
        "env_kwargs": ENV_KWARGS,
        "dv_target": "state_determined_hazard_at_agent",
        "thresholds": {
            "MONOTONE_RHO_MIN": MONOTONE_RHO_MIN,
            "COUPLING_RHO_MIN": COUPLING_RHO_MIN,
            "DV_DELTA_SD_MULT": DV_DELTA_SD_MULT,
            "DV_DECODABLE_FLOOR": DV_DECODABLE_FLOOR,
            "TARGET_STD_FLOOR": TARGET_STD_FLOOR,
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
        "outcome": result["outcome"],
        "evidence_direction": result.get("evidence_direction", "unknown"),
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "onset_episodes": ONSET_EPISODES,
        "thresholds": full_config["thresholds"],
        "criteria": result["criteria"],
        "by_onset": result["by_onset"],
        "arm_results": result["arm_results"],
        "deliverable_note": (
            "The DEFERRED CALIBRATED BOUND test for INV-089, the core coupling V3-EXQ-743 "
            "reported-not-gated. LOAD-BEARING PASS = harm_eval_z_harm held-out evaluator "
            "quality (a) rises MONOTONICALLY with HarmEncoder maturation (C1), (b) TRACKS "
            "measured z_harm differentiation (C2 IV<->DV coupling), (c) RELIABLY across "
            "seeds (C3 effect-size). Two calibration fixes over 743: (a) STATE-DETERMINED "
            "DV target = ground-truth clip(hazard_field[agent],0,1) aligned to the same "
            "state z_harm encodes (NOT realized post-random-action harm, 743's "
            "under-determined confound); (b) SCALE-FREE (rank/coupling/effect-size) "
            "thresholds (NO 740a 0.03/0.15 magnitude floors, mis-calibrated for the "
            "narrow/high-JL z_harm gradient). A met-precondition C1/C2/C3 FAIL is a genuine "
            "WEAKENS; only a starved IV / undecodable target / near-constant target routes "
            "non_contributory. A PASS moves INV-089 provisional -> toward stable."
        ),
        "cross_stream_contrast_note": (
            "Extends V3-EXQ-743 (INV-089 positive control) with the deferred core coupling. "
            "Reuses 743's exact z_harm maturation + collection recipe (same MATURE/COLLECT "
            "seed bases), re-targeted from realized harm onto the state-determined "
            "ground-truth hazard exposure. zharm_realized_decode_r2 (743's positive-control "
            "metric) is reported per cell for continuity. Re-derive brake (INV-064 "
            "740-cluster) HONOURED: z_harm observable, INV-089 claim, no z_world decode -- "
            "NOT a 740b. Supersedes NOTHING."
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
    c = result["criteria"]
    print("=== INV-089 harm-evaluator z_harm CALIBRATED BOUND result (746, n=8) ===", flush=True)
    print(f"  preconditions_met: {result['preconditions_met']} "
          f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.4f} iv_rho={c['mean_iv_rho']:.3f}; "
          f"dv_decodable={c['PC_dv_decodable']} mature_iv={c['mean_iv_mature']:.3f}; "
          f"target_var={c['PC_target_var']} min_std={c['min_target_std']:.4f})", flush=True)
    print(f"  C1 dv_monotone:    {c['C1_dv_monotone']} (mean_dv_rho={c['mean_dv_rho']:.3f} >= {MONOTONE_RHO_MIN})",
          flush=True)
    print(f"  C2 bound_coupling: {c['C2_bound_coupling']} "
          f"(mean_dv_delta={c['mean_dv_delta']:.4f}>0, mean_couple_rho={c['mean_couple_rho']:.3f} >= {COUPLING_RHO_MIN})",
          flush=True)
    print(f"  C3 dv_reliable:    {c['C3_dv_reliable']} "
          f"(mean_dv_delta={c['mean_dv_delta']:.4f} >= {DV_DELTA_SD_MULT}*SD={DV_DELTA_SD_MULT * c['sd_dv_delta']:.4f})",
          flush=True)
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
