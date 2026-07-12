#!/opt/local/bin/python3
"""V3-EXQ-743: INV-089 harm-evaluator quality bounded by z_harm differentiation --
the re-posed, RE-STREAMED successor to the mis-streamed harm leg of V3-EXQ-740a.

INV-089 (harm_evaluator_bounded_by_z_harm_differentiation, invariant / emergent
from ARC-003/019/027; candidate, pending_substrate_reconfirmation) is child (ii) of
INV-064's granularity decomposition (claim_synthesis_INV-064_2026-07-12). It asserts:

    E3's NOCICEPTIVE harm-evaluation quality -- harm_eval_z_harm (e3_selector.py:660),
    which reads z_harm = HarmEncoder(harm_obs) (stack.py:119) -- is strictly bounded by
    z_harm representational differentiation, a maturation trajectory DISTINCT from
    z_world's. Under SD-010 the HarmEncoder is instantiated OUTSIDE the z_world encoder
    and is exempt from reafference correction (the EXQ-027b over-correction fix). So
    harm-evaluator quality is bounded by z_harm differentiation, NOT z_world
    differentiation. Productive harm-evaluator training cannot precede sufficient
    z_harm differentiation.

WHY THIS IS A NEW OBSERVABLE, NOT A 740b (re-derive brake FIRED, honoured):
    The INV-064 740-cluster re-derive brake (2 non_contributory autopsies: 740
    IV-backwards, 740a DV-undecodable) REFUSES a same-observable 740b that decodes
    scalar harm from z_world. 740a's dispositive finding: from a FROZEN z_world, harm
    WORLD-FEATURES decode fine and rise (world_feat_decode_r2 0.048 -> 0.245) while
    realized SCALAR harm does not (harm_decode_r2 0.034, FLAT) -- because SD-010 routes
    harm through a dedicated z_harm stream, not z_world. This experiment probes the
    stream E3 ACTUALLY uses for harm (z_harm), the DIFFERENT observable the brake
    explicitly permits (claim_synthesis_INV-064_2026-07-12 sec 4; INV-089 claim text).
    It tests INV-089 (not INV-064): new claim, new EXQ NUMBER, supersedes NOTHING.

THE DISPOSITIVE CROSS-STREAM CONTRAST (this run vs 740a, SAME realized-harm target):
    Target Y = realized per-step harm = abs(harm_signal) when harm_signal < 0 -- the
    IDENTICAL target 740a decoded at 0.034 FLAT from frozen z_world. Here it is decoded
    from frozen z_harm across HarmEncoder maturation. INV-089's prediction: z_harm carries
    realized harm (positive control clears the floor at the mature anchor) AND that
    decodability RISES with HarmEncoder maturation -- the leg z_world could not express.

DESIGN (measurement-only, commitment-free frozen-representation curriculum-ORDER
        contrast -- structurally identical to 740a, re-streamed onto z_harm):
    Per (seed, onset) cell:
      1. Build agent (056c SD-010 wiring: world_dim = z_harm_dim = 32, use_harm_stream
         so agent.e3.harm_eval_z_harm_head is 32-dim) + a STANDALONE HarmEncoder(51,32).
         z_world maturation is IRRELEVANT (harm routes through harm_obs, not z_world);
         only HarmEncoder maturity varies across arms.
      2. Snapshot the fresh harm_eval_z_harm_head init (bit-identical across arms --
         arm_cell resets all RNG to `seed` at cell entry, so every onset arm of a seed
         builds an identical agent + encoder before maturation diverges).
      3. MATURE the HarmEncoder for `onset` episodes on the canonical SD-010 supervision
         label harm_obs[12] (normalised hazard proximity at the agent cell, [0,1]; the
         056c-proven target), co-trained with a THROWAWAY proximity head that is
         discarded. onset=0 -> a FRESH (untrained) HarmEncoder, the genuinely-immature
         anchor. Maturing on the SD-010 proximity signal (NOT the realized-harm decode
         target Y) keeps the positive control NON-circular -- exactly as 740a matured the
         z_world encoder on its native REE losses, then decoded harm_target.
      4. FREEZE the HarmEncoder.
      5. Collect a SHARED frozen dataset by replaying a FIXED action sequence through a
         FIXED collection env (seeded, maturity-independent). Identical raw trajectory +
         labels across the onset arms of a seed; the ONLY per-arm variable is the frozen
         z_harm. Per step: z_harm[t] (frozen), Y[t]=realized harm, prox[t]=harm_obs[12].
      6. Re-init agent.e3.harm_eval_z_harm_head to the fixed shared init and train ONLY
         that head for a FIXED budget on the frozen (z_harm, Y) tensors.
      7. Read out: DV = harm_eval_z_harm held-out R^2 (+ train R^2, gap); IV = linear
         z_harm differentiation; positive control = linear z_harm harm-decodability.

    IV (per arm): onset episodes {0, 1, 4, 12, 30}  (HarmEncoder maturity at DV onset).
    IV OBSERVABLE: zharm_decode_r2 = held-out R^2 of a ridge LINEAR probe z_harm[t] ->
                   Y[t] (realized harm). Linear legibility = z_harm differentiation.
                   Rises with maturation (JL-safe: a random HarmEncoder does not make
                   the nonlinear realized-harm signal linearly readable; training does).
    DV: harm_eval_z_harm held-out R^2 (trained nonlinear head, re-init + fixed budget),
        and gap = train_R2 - test_R2.
    POSITIVE CTRL: zharm_decode_r2 at onset_max clears HARM_DECODABLE_FLOOR (harm is
                   decodable-in-principle from the stream E3 uses) -- the leg 740a
                   FAILED on z_world (0.034 < 0.05). This is the load-bearing contrast.

REGIME (unchanged -- 537b / 740a lesson).
    scheduled_external_hazard is OFF: injecting by-design-unpredictable ext events floors
    harm decodability regardless of z_harm maturity, making the contrast uninterpretable.
    OFF -> harm is predictable-from-state and z_harm differentiation is the binding
    constraint -- the regime in which INV-089's prediction is falsifiable.

DELIVERABLE = THE POSITIVE CONTROL (design decision 2026-07-12, user-selected at the
    queue-experiment gate after a spot-check showed the literal 740a-transplanted "bound"
    test degenerates on this stream). A real-budget spot check (seed 42, onset {0,1,12,30})
    found: (a) the positive control PASSES -- realized harm decodes from frozen z_harm at
    0.065->0.083 (rising, clears the 0.05 floor) where 740a's mature z_world was 0.034 FLAT;
    but (b) the harm_eval_z_harm trained-head "bound" DV is stuck ~0.05 and non-monotone,
    because realized post-random-action scalar harm is UNDER-DETERMINED even from harm_obs
    (740a's own flagged confound), and (c) the z_harm harm-decodability gradient is NARROW
    (delta ~0.018, high JL floor), so the 740a-scaled 0.03/0.15 magnitude thresholds are
    mis-calibrated for this stream. So the PASS gate is the positive-control contrast (which
    is the dispositive INV-089 premise), and the trained-head bound coupling is REPORTED but
    NOT gating, with a calibrated bound test DEFERRED to a follow-up.

PRE-REGISTERED PASS CRITERIA (LOAD-BEARING -- evidence supports INV-089's premise):
    C1 harm_decodable_from_zharm: mean_seed(zharm_decode_r2[onset_max]) >= 0.05
                      -- realized harm is decodable-in-principle from the z_harm stream E3
                      actually uses (the dispositive contrast with 740a's z_world 0.034).
    C2 decodability_rises:       mean_seed(zharm_decode_r2[onset_max] -
                      zharm_decode_r2[onset_min]) > 0  AND  mean_seed Spearman(onset,
                      zharm_decode_r2) > 0  -- directional rise with HarmEncoder maturation
                      (NO z_world-scaled magnitude floor; the gradient is narrow by JL).
    PASS = C1 and C2.
    Also reported: crosses_zworld_reference = mean_seed(zharm_decode_r2[onset_max]) > 0.034
                   (machine-readable cross-stream contrast vs the landed 740a value).

PRECONDITION (validity -- only genuine measurement failure routes non_contributory):
    PC_events: min over cells of harm_event_frac >= 0.03 (enough harm to measure
    decodability). If unmet the run is non_degenerate=False / evidence_direction="unknown".

SECONDARY / EXPLORATORY (REPORTED, NOT a PASS gate): harm_eval_z_harm trained-head bound
    coupling -- SEC_quality_gain (delta R2 >= 0.15 + effect size), SEC_monotone (rho >=
    0.80), SEC_noise_signature (gap delta >= 0.08). Recorded for continuity only; the
    under-determined realized-harm DV makes these an UNFAIR bound test on this stream.

FALSIFIABILITY: PASS = INV-089's first positive evidence (harm carried + rising on the
    z_harm stream). It CAN fail as a genuine WEAKENS (non_degenerate=True): if realized
    harm is NOT decodable from z_harm at maturity (C1 unmet -- harm-eval cannot be bounded
    by a stream that does not carry harm) or does not rise (C2 unmet). Only a harm-event
    shortage (PC_events) routes non_contributory.

NO SUBSTRATE BUILD OWED: HarmEncoder (stack.py:119) + harm_eval_z_harm (e3_selector.py:660)
    already exist and are SD-010 IMPLEMENTED (ree-v3/CLAUDE.md). This is testable-now.
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
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.latent.stack import HarmEncoder  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_743_inv089_harm_evaluator_z_harm_bounded"
QUEUE_ID = "V3-EXQ-743"
SUPERSEDES = None            # different observable (z_harm), NOT a fix of 740a
CLAIM_IDS = ["INV-089"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants ---
SEEDS = [42, 7, 19]
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
DV_INIT_SEED = 90001                 # (documentation) fixed head init is arm_cell-seeded
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
    "num_hazards": 5,   # denser predictable harm -> enough harm events per collected step
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
    # 537b / 740a lesson: OFF so harm is predictable-from-state and z_harm
    # differentiation is the binding constraint (not a by-design-unpredictable ceiling).
    "scheduled_external_hazard_enabled": False,
}

# --- pre-registered thresholds ---
# LOAD-BEARING (the positive-control deliverable: harm is decodable-in-principle from
# the z_harm stream E3 actually uses, and that decodability RISES with maturation).
HARM_DECODABLE_FLOOR = 0.05       # C1: mature-anchor linear harm-decodability from z_harm
ZWORLD_HARM_DECODE_740A_REF = 0.034  # 740a mature-z_world realized-harm decode (cited contrast)
# C2 uses a DIRECTIONAL rise (delta > 0 AND positive monotone), NOT a z_world-scaled
# magnitude floor: the z_harm harm-decodability gradient is narrow (high JL floor --
# harm is near-linearly present in harm_obs), so a 740a-style 0.03/0.15 magnitude floor
# is mis-calibrated for this stream. Deferred: a properly-calibrated bound test needs a
# state-determined DV target (realized post-random-action harm is under-determined) +
# z_harm-scaled thresholds (see secondary block below).
HARM_EVENT_FRAC_FLOOR = 0.03      # validity precondition: enough harm to fit

# SECONDARY / EXPLORATORY (REPORTED, NOT a PASS gate). The harm_eval_z_harm "bound"
# coupling. These 740a-transplanted thresholds are recorded for continuity but do NOT
# gate the outcome -- the realized-harm DV is under-determined (740a's own flagged
# confound) so the trained-head coupling is not a fair bound test on this stream.
SEC_R2_DELTA_FLOOR = 0.15
SEC_R2_DELTA_SD_MULT = 2.0
SEC_MONOTONE_RHO_MIN = 0.80
SEC_GAP_DELTA_FLOOR = 0.08


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

    Fixed train/test split (identical indices across arms -- depends only on n, which
    is identical across the onset arms of a seed because the collection trajectory is
    shared). Baseline for R^2 is the TRAIN-set per-column mean (a proper held-out R^2,
    so an uninformative representation scores <= 0). Pure read-out of the FROZEN z_harm
    -- no gradients, distinct from the trained harm_eval_z_harm MLP (the DV).
    """
    n = Z.shape[0]
    if n < 4:
        return float("nan")
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
    """056c SD-010 wiring: world_dim = z_harm_dim = 32 so the agent's
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


def _mature_harm_encoder(harm_enc: HarmEncoder, seed: int, onset: int,
                         steps_per_ep: int, denom: int) -> int:
    """Mature the HarmEncoder for `onset` episodes on the SD-010 supervision label
    harm_obs[12], co-trained with a THROWAWAY proximity head (discarded). onset=0 ->
    no training (fresh encoder, the immature anchor). Returns the training-step count.

    Uses a per-seed maturation env + action RNG (MATURE_SEED_BASE) that is INDEPENDENT
    of the collection trajectory, so the encoder never sees the collection data and the
    maturation RNG consumption does not perturb downstream (collection/DV) determinism.
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
    """Replay a FIXED action sequence through a FIXED collection env, encoding z_harm
    with the (frozen) HarmEncoder. Identical raw trajectory + labels across onset arms;
    only the frozen encoder differs.

    Per step:
      Zharm = HarmEncoder(harm_obs[t])          (frozen encoding, the only per-arm var)
      Y     = realized harm_target[t]            (DV + positive-control target; == 740a)
      Prox  = harm_obs[t][12]                    (SD-010 proximity label; secondary target)
    """
    collect_env = CausalGridWorldV2(seed=COLLECT_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(COLLECT_SEED_BASE + seed)
    action_dim = collect_env.action_dim

    zh_list: List[torch.Tensor] = []
    y_list: List[float] = []
    prox_list: List[float] = []

    harm_enc.eval()
    with torch.no_grad():
        for _ep in range(n_episodes):
            _, obs_dict = collect_env.reset()
            for _step in range(steps_per_ep):
                harm_obs_t = _row_vec(obs_dict.get("harm_obs"))
                z_harm = harm_enc(harm_obs_t).detach().cpu()
                prox = float(harm_obs_t[0, HARM_OBS_CENTER].item())

                action = int(act_rng.integers(0, action_dim))
                _, harm_signal, done, _info, obs_next = collect_env.step(action)
                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

                zh_list.append(z_harm)
                y_list.append(harm_target)
                prox_list.append(prox)

                obs_dict = obs_next
                if done:
                    break

    Zharm = torch.cat(zh_list, dim=0)
    Y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    Prox = torch.tensor(prox_list, dtype=torch.float32).unsqueeze(1)
    return {"Zharm": Zharm, "Y": Y, "Prox": Prox}


def _freeze(harm_enc: HarmEncoder) -> None:
    for p in harm_enc.parameters():
        p.requires_grad_(False)


def _train_dv_and_eval(agent: REEAgent, data: Dict[str, torch.Tensor],
                       head_init_state: Dict[str, Any],
                       dv_epochs: int) -> Dict[str, float]:
    """Re-init harm_eval_z_harm_head to the fixed shared init, train for a FIXED budget
    on the frozen (z_harm, harm_target) tensors, return train/test R^2.
    """
    Z, Y = data["Zharm"], data["Y"]
    n = Z.shape[0]
    n_test = max(1, int(n * HELDOUT_FRAC))
    # Fixed shuffled split (identical indices across all arms -- depends only on n,
    # which is identical across arms because the collection trajectory is shared).
    split_perm = np.random.default_rng(DV_TRAIN_SEED + 1).permutation(n)
    te = torch.as_tensor(split_perm[:n_test], dtype=torch.long)
    tr = torch.as_tensor(split_perm[n_test:], dtype=torch.long)
    Ztr, Ytr = Z[tr], Y[tr]
    Zte, Yte = Z[te], Y[te]

    # Fixed, bit-identical head init + optimisation RNG across all arms.
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
            loss = F.mse_loss(pred, Ytr[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == dv_epochs:
            print(f"  [dv-fit] epoch {epoch + 1}/{dv_epochs}", flush=True)

    agent.eval()
    with torch.no_grad():
        r2_train = _r2(agent.e3.harm_eval_z_harm(Ztr), Ytr)
        r2_test = _r2(agent.e3.harm_eval_z_harm(Zte), Yte)
    harm_event_frac = float((Y > 1e-6).float().mean().item())
    return {
        "harm_r2_train": r2_train,
        "harm_r2_test": r2_test,
        "harm_gap": r2_train - r2_test,
        "harm_event_frac": harm_event_frac,
        "n_samples": int(n),
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
    }
    # Cells are NOT reuse-eligible: the frozen z_harm is a function of the shared
    # maturation trajectory, not reproducible from config_slice alone (there is no OFF
    # baseline arm -- every cell is an onset level). Fingerprint emitted (validator
    # requirement) but flagged ineligible.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=["frozen_representation_from_maturation_trajectory"],
    ) as cell:
        agent, _env = _build_agent(seed)
        # Standalone HarmEncoder -- the module we mature (bit-identical init across the
        # onset arms of a seed because arm_cell reset all RNG to `seed` on entry).
        harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
        # Snapshot the fresh harm_eval_z_harm_head init BEFORE any training so every arm
        # starts the DV head from bit-identical weights.
        head_init_state = copy.deepcopy(agent.e3.harm_eval_z_harm_head.state_dict())

        # 1. Mature the HarmEncoder (onset=0 -> fresh encoder anchor). The throwaway
        #    proximity head is discarded; agent.e3.harm_eval_z_harm_head is untouched
        #    here and re-loaded from the snapshot in step 4.
        n_mature_steps = _mature_harm_encoder(
            harm_enc, seed, onset, steps_per_ep, MATURE_PROGRESS_DENOM)

        # 2. Freeze the HarmEncoder.
        _freeze(harm_enc)

        # 3. Collect the shared frozen dataset (identical trajectory across arms).
        data = _collect_frozen_dataset(harm_enc, seed, steps_per_ep, collect_eps)

        # Differentiation mediators at freeze.
        zharm_var = float(data["Zharm"].var(dim=0).mean().item())
        zharm_eff_rank = _eff_rank(data["Zharm"])
        # IV / positive control: linear z_harm harm-decodability (realized harm Y).
        zharm_decode_r2 = _ridge_heldout_r2(data["Zharm"], data["Y"], DECODE_SPLIT_SEED)
        # Secondary diagnostic: linear z_harm proximity-decodability (SD-010 label).
        zharm_prox_decode_r2 = _ridge_heldout_r2(data["Zharm"], data["Prox"], DECODE_SPLIT_SEED)

        # 4-5. Controlled frozen harm_eval_z_harm training + readout (the DV).
        dv = _train_dv_and_eval(agent, data, head_init_state, dv_epochs)

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "n_mature_steps": n_mature_steps,
            "zharm_var": zharm_var,
            "zharm_eff_rank": zharm_eff_rank,
            "zharm_decode_r2": zharm_decode_r2,
            "zharm_prox_decode_r2": zharm_prox_decode_r2,
            **dv,
        }
        cell.stamp(row)

    print(f"verdict: {'PASS' if dv['harm_r2_test'] == dv['harm_r2_test'] else 'FAIL'}",
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
    per_seed_iv_delta: List[float] = []
    per_seed_harm_decode_mature: List[float] = []
    for seed in SEEDS:
        r2s = [_cell(seed, o)["harm_r2_test"] for o in ONSET_EPISODES]
        per_seed_delta_r2.append(_cell(seed, onset_max)["harm_r2_test"]
                                 - _cell(seed, onset_min)["harm_r2_test"])
        per_seed_gap_delta.append(_cell(seed, onset_min)["harm_gap"]
                                  - _cell(seed, onset_max)["harm_gap"])
        per_seed_rho.append(_spearman([float(o) for o in ONSET_EPISODES], r2s))
        # z_harm-differentiation movement (predicted +).
        per_seed_iv_delta.append(_cell(seed, onset_max)["zharm_decode_r2"]
                                 - _cell(seed, onset_min)["zharm_decode_r2"])
        # mature-anchor positive control.
        per_seed_harm_decode_mature.append(_cell(seed, onset_max)["zharm_decode_r2"])

    def _mean(v: List[float]) -> float:
        return float(sum(v) / len(v)) if v else 0.0

    def _sd(v: List[float]) -> float:
        if len(v) < 2:
            return 0.0
        m = _mean(v)
        return float((sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5)

    # Per-seed z_harm harm-decodability spearman across onset (directional-rise check).
    per_seed_decode_rho: List[float] = []
    for seed in SEEDS:
        decodes = [_cell(seed, o)["zharm_decode_r2"] for o in ONSET_EPISODES]
        per_seed_decode_rho.append(_spearman([float(o) for o in ONSET_EPISODES], decodes))

    mean_delta_r2 = _mean(per_seed_delta_r2)
    sd_delta_r2 = _sd(per_seed_delta_r2)
    mean_gap_delta = _mean(per_seed_gap_delta)
    mean_rho = _mean(per_seed_rho)
    mean_iv_delta = _mean(per_seed_iv_delta)
    mean_decode_rho = _mean(per_seed_decode_rho)
    mean_harm_decode_mature = _mean(per_seed_harm_decode_mature)
    min_harm_event_frac = min(r["harm_event_frac"] for r in rows)

    # Validity precondition (only genuine measurement failure routes non_contributory).
    pc_events = min_harm_event_frac >= HARM_EVENT_FRAC_FLOOR
    preconditions_met = pc_events

    # LOAD-BEARING PASS criteria (the positive-control deliverable for INV-089).
    # C1: realized harm decodable-in-principle from the z_harm stream E3 uses (mature
    #     anchor clears the floor) -- the dispositive contrast with 740a's z_world 0.034.
    c1_harm_decodable = mean_harm_decode_mature >= HARM_DECODABLE_FLOOR
    # C2: that decodability RISES with HarmEncoder maturation (directional -- delta > 0
    #     AND positive monotone; NO z_world-scaled magnitude floor).
    c2_decodability_rises = (mean_iv_delta > 0.0) and (mean_decode_rho > 0.0)
    # Machine-readable cross-stream contrast (vs the landed 740a mature-z_world value).
    crosses_zworld_reference = mean_harm_decode_mature > ZWORLD_HARM_DECODE_740A_REF

    # SECONDARY / EXPLORATORY -- REPORTED, NOT gating (harm_eval_z_harm bound coupling).
    sec_c1_effect = mean_delta_r2 >= (SEC_R2_DELTA_SD_MULT * sd_delta_r2)
    sec_quality_gain = (mean_delta_r2 >= SEC_R2_DELTA_FLOOR) and sec_c1_effect
    sec_monotone = mean_rho >= SEC_MONOTONE_RHO_MIN
    sec_noise_signature = mean_gap_delta >= SEC_GAP_DELTA_FLOOR

    return {
        "onset_min": onset_min,
        "onset_max": onset_max,
        "per_seed_delta_r2": per_seed_delta_r2,
        "per_seed_gap_delta": per_seed_gap_delta,
        "per_seed_spearman_rho": per_seed_rho,
        "per_seed_iv_delta": per_seed_iv_delta,
        "per_seed_decode_rho": per_seed_decode_rho,
        "per_seed_harm_decode_mature": per_seed_harm_decode_mature,
        "mean_delta_r2": mean_delta_r2,
        "sd_delta_r2": sd_delta_r2,
        "mean_gap_delta": mean_gap_delta,
        "mean_spearman_rho": mean_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_decode_rho": mean_decode_rho,
        "mean_harm_decode_mature": mean_harm_decode_mature,
        "zworld_harm_decode_740a_ref": ZWORLD_HARM_DECODE_740A_REF,
        "min_harm_event_frac": min_harm_event_frac,
        "PC_events": pc_events,
        "preconditions_met": preconditions_met,
        # load-bearing
        "C1_harm_decodable_from_zharm": c1_harm_decodable,
        "C2_decodability_rises": c2_decodability_rises,
        "crosses_zworld_reference": crosses_zworld_reference,
        # secondary / exploratory (non-gating)
        "SEC_quality_gain": sec_quality_gain,
        "SEC_monotone": sec_monotone,
        "SEC_noise_signature": sec_noise_signature,
    }


def _aggregate_by_onset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ONSET_EPISODES:
        cells = [r for r in rows if r["onset_episodes"] == o]
        out[f"onset_{o}"] = {
            "mean_harm_r2_test": float(np.mean([c["harm_r2_test"] for c in cells])),
            "mean_harm_r2_train": float(np.mean([c["harm_r2_train"] for c in cells])),
            "mean_harm_gap": float(np.mean([c["harm_gap"] for c in cells])),
            "mean_zharm_var": float(np.mean([c["zharm_var"] for c in cells])),
            "mean_zharm_eff_rank": float(np.mean([c["zharm_eff_rank"] for c in cells])),
            "mean_zharm_decode_r2": float(np.mean([c["zharm_decode_r2"] for c in cells])),
            "mean_zharm_prox_decode_r2": float(
                np.mean([c["zharm_prox_decode_r2"] for c in cells])),
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
    # PASS = the load-bearing positive control: harm decodable-in-principle from z_harm
    # (C1) AND that decodability rises with maturation (C2). The harm_eval_z_harm bound
    # coupling (SEC_*) is reported but does NOT gate the outcome.
    claim_pass = (criteria["C1_harm_decodable_from_zharm"]
                  and criteria["C2_decodability_rises"])

    if not preconditions_met:
        outcome = "FAIL"
        evidence_direction = "unknown"
        non_degenerate = False
        reasons = []
        if not criteria["PC_events"]:
            reasons.append(
                f"insufficient harm events: min frac {criteria['min_harm_event_frac']:.4f} "
                f"< {HARM_EVENT_FRAC_FLOOR} (harm-decodability cannot be validly measured)")
        degeneracy_reason = "; ".join(reasons)
    else:
        outcome = "PASS" if claim_pass else "FAIL"
        # supports = z_harm carries realized harm and it rises with maturation (first
        # positive evidence for INV-089's premise); weakens = harm NOT decodable from
        # z_harm at maturity or does not rise (a real result, not a degeneracy).
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
            "HARM_DECODABLE_FLOOR": HARM_DECODABLE_FLOOR,
            "ZWORLD_HARM_DECODE_740A_REF": ZWORLD_HARM_DECODE_740A_REF,
            "HARM_EVENT_FRAC_FLOOR": HARM_EVENT_FRAC_FLOOR,
            "SEC_R2_DELTA_FLOOR": SEC_R2_DELTA_FLOOR,
            "SEC_R2_DELTA_SD_MULT": SEC_R2_DELTA_SD_MULT,
            "SEC_MONOTONE_RHO_MIN": SEC_MONOTONE_RHO_MIN,
            "SEC_GAP_DELTA_FLOOR": SEC_GAP_DELTA_FLOOR,
        },
        "criteria": result["criteria"],
        "by_onset": result["by_onset"],
        "arm_results": result["arm_results"],
        "deliverable_note": (
            "PASS gate is the POSITIVE CONTROL (load-bearing): realized harm is "
            "decodable-in-principle from the dedicated z_harm stream E3 actually uses "
            "(C1: mature-anchor zharm_decode_r2 >= 0.05) AND that decodability RISES with "
            "HarmEncoder maturation (C2: directional). The harm_eval_z_harm 'bound' "
            "coupling (SEC_*) is REPORTED but NOT gating: the realized post-random-action "
            "harm DV is under-determined (740a's flagged confound) so it is not a fair "
            "bound test on this stream. A calibrated bound test (state-determined DV "
            "target + z_harm-scaled thresholds) is DEFERRED to a follow-up."
        ),
        "cross_stream_contrast_note": (
            "Re-streamed successor to V3-EXQ-740a's mis-streamed harm leg. Same realized "
            "harm target Y=abs(harm_signal<0); 740a decoded it from frozen z_world at "
            "0.034 FLAT (positive control FAILED, < 0.05); this decodes it from frozen "
            "z_harm across HarmEncoder maturation. crosses_zworld_reference = mature "
            "zharm_decode_r2 > 0.034. Re-derive brake (INV-064 740-cluster) HONOURED: "
            "different observable (z_harm), different claim (INV-089), NOT a z_world 740b."
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_path}", flush=True)
    return out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    """Run the experiment; return (outcome, manifest_path). The emit_outcome sentinel
    is written by the __main__ block (runner-conformance contract).
    """
    result = run_experiment(dry_run=dry_run)

    if dry_run:
        print(f"DRY_RUN complete: {result['n_cells']} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result)
    c = result["criteria"]
    print("=== INV-089 harm-evaluator z_harm positive-control result (743) ===", flush=True)
    print(f"  precondition (events): {c['PC_events']} (min_frac={c['min_harm_event_frac']:.4f})",
          flush=True)
    print(f"  C1 harm_decodable_from_zharm: {c['C1_harm_decodable_from_zharm']} "
          f"(mature_zharm_decode_r2={c['mean_harm_decode_mature']:.3f} vs floor {HARM_DECODABLE_FLOOR} "
          f"vs 740a z_world {c['zworld_harm_decode_740a_ref']}; crosses_zworld={c['crosses_zworld_reference']})",
          flush=True)
    print(f"  C2 decodability_rises: {c['C2_decodability_rises']} "
          f"(mean_delta={c['mean_iv_delta']:.3f}, mean_rho={c['mean_decode_rho']:.3f})", flush=True)
    print(f"  [secondary/non-gating] bound-coupling: quality_gain={c['SEC_quality_gain']} "
          f"(delta_r2={c['mean_delta_r2']:.3f}) monotone={c['SEC_monotone']} "
          f"(rho={c['mean_spearman_rho']:.3f}) noise_sig={c['SEC_noise_signature']} "
          f"(gap_delta={c['mean_gap_delta']:.3f})", flush=True)
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
