#!/opt/local/bin/python3
"""V3-EXQ-744: INV-088 world/goal-evaluator bounded by z_world differentiation --
the DV-COUPLING half of the world/goal leg. EXTENDS (does not supersede)
V3-EXQ-740a.

CLAIM UNDER TEST (INV-088, world_goal_evaluator_bounded_by_z_world_differentiation;
child (i) of INV-064, registered 2026-07-12 via /claim-synthesis):
    "E3's goal and world-feature evaluation quality -- the z_world-reading
    evaluators: harm_eval(z_world) (SD-003 attribution head, e3_selector.py:593),
    benefit_eval, and goal/trajectory scoring -- is strictly bounded by z_world (E1)
    representational differentiation. Poorly-differentiated z_world yields a
    noise-fitted goal/world-feature evaluator, so productive training of the
    z_world-reading evaluators cannot precede sufficient E1 schema differentiation."

WHAT THIS TESTS (the remaining half; the IV half is already positive):
    V3-EXQ-740a already validly established the IV: task-relevant differentiation of
    z_world (world_feat_decode_r2 = held-out ridge R2 of z_world[t] -> harm_obs[t+1])
    rises MONOTONICALLY 0.048 -> 0.245 across onset {0,1,4,12,30} with a genuine
    immature anchor (onset_0 = fresh encoder); PC_iv_moved passed (mean delta 0.197).
    That was the first valid demonstration that E1/E2 task-relevant differentiation
    increases with maturation on the V3 substrate.

    This experiment extends that curriculum-order contrast to the DV COUPLING: that the
    held-out QUALITY of a re-initialised z_world-reading E3 evaluator (fixed training
    budget on FROZEN z_world) improves MONOTONICALLY with the rising z_world
    differentiation. Falsified if evaluator quality is flat / non-monotone across a
    validly-moving z_world-differentiation gradient.

THE CORRECTED DV TARGET (this is the whole point of the world/goal leg vs 740a's
harm leg):
    740a's harm_eval_head was trained to predict REALIZED SCALAR HARM
    (abs(harm_signal) when < 0) from z_world. The 740a failure autopsy
    (failure_autopsy_V3-EXQ-740a_2026-07-12) adjudicated that target MIS-STREAMED and
    STRUCTURALLY UNDER-DETERMINED:
      (a) SD-010 routes NOCICEPTIVE harm through a DEDICATED z_harm = HarmEncoder(harm_obs)
          stream (harm_eval_z_harm), OUTSIDE the z_world encoder -- so realized scalar
          harm is not a function of z_world (that is INV-089's leg, tested separately in
          V3-EXQ-743 on z_harm);
      (b) realized post-random-action scalar harm = f(world-state, body/z_self, action) --
          body + random-action variance is noise w.r.t. z_world.
    The DISPOSITIVE contrast the autopsy drew: from the SAME frozen z_world, harm
    WORLD-FEATURES (harm_obs) decode fine and RISE with maturation (0.048 -> 0.245),
    while realized scalar harm does not (0.034, flat). So the world/goal leg's evaluator
    must read a WORLD/GOAL-FEATURE target that IS decodable from z_world, NOT realized
    scalar harm.

    harm_obs [51] = hazard_field_view[25] (harm-relevant proximity world features,
    the SD-003 harm_eval attribution domain) + resource_field_view[25]
    (goal/benefit-relevant proximity world features, the benefit_eval / goal-scoring
    domain) + harm_exposure[1] (a body-exposure scalar, EXCLUDED -- not a pure world
    feature). Both proximity-field blocks are components of the harm_obs[t+1] vector
    740a proved decodable-and-rising. The DV evaluators predict the NEXT-STEP
    world/goal-feature target (predictive -> JL-safe: a fresh random-projection encoder
    CANNOT predict next-step world structure, so evaluator quality genuinely varies with
    E1/E2 maturation, unlike a current-step target a random projection would already
    recover).

    PRIMARY DV (load-bearing) -- the REAL substrate SD-003 head:
        agent.e3.harm_eval_head (e3_selector.py:247/593), re-init to a bit-identical
        fixed init, trained a FIXED budget on FROZEN z_world[t] -> next-step
        hazard-proximity world feature (mean hazard_field_view of harm_obs[t+1],
        dataset-normalised to [0,1] for the Sigmoid head). This is the SAME head 740a
        used, with the CORRECTED (world-feature, decodable) target.
    SECONDARY DV (corroborating, reported, NOT load-bearing) -- the REAL benefit_eval
    head:
        agent.e3.benefit_eval_head (e3_selector.py:259/611) -> next-step
        resource-proximity world feature (mean resource_field_view of harm_obs[t+1]).
        The goal-leg cross-target check: does a DIFFERENT z_world-reading evaluator on a
        DIFFERENT world/goal-feature target also couple to the same differentiation
        gradient? Its positive control is ADVISORY (a weakly-decodable resource target
        WARNs but does NOT degenerate the run -- only the primary harm_eval SD-003
        decodability gates validity).

MEASUREMENT-ONLY, COMMITMENT-FREE DESIGN (inherited from 740/740a).
    Controlled curriculum-ORDER contrast on a FROZEN representation; NO sustained
    multi-step action-commitment layer is exercised (no basal-ganglia / F-dominance
    path), so a PASS/FAIL is a real verdict on INV-088's DV coupling, not a
    re-derivation of the conversion ceiling.

    Per (seed, onset) cell:
      1. Mature E1 + E2.world_forward + encoder (latent_stack) for `onset` episodes via
         the canonical goal_pipeline warmup_train (onset=0 -> fresh encoder, the immature
         anchor). Heads trained incidentally here are DISCARDED (re-initialised in step 5).
      2. FREEZE E1, E2.world_transition, E2.world_action_encoder, latent_stack.
      3. Collect a shared frozen dataset by replaying a FIXED action sequence (seeded
         RNG, independent of maturity) through a FIXED collection env. The raw trajectory
         + world/goal-feature labels are IDENTICAL across the onset arms of a seed, so the
         ONLY per-arm variable is the frozen encoder's z_world.
      4. Read the IV (world_feat_decode_r2 = ridge z_world[t] -> harm_obs[t+1], reused
         from 740a) + the DV-target positive controls (ridge z_world[t] -> next-step
         hazard/resource world-feature scalar).
      5. Re-init agent.e3.harm_eval_head (PRIMARY) + agent.e3.benefit_eval_head
         (SECONDARY) to FIXED bit-identical inits and train ONLY each head for a FIXED
         budget on the frozen-encoded (z_world, next-step world-feature) tensors.
      6. Read out DV quality (held-out R2, train R2, gap = train - test) per head.

    IV  (per arm): onset episodes {0, 1, 4, 12, 30}  (E1/E2 maturity at evaluator onset).
    IV OBSERVABLE (reused from 740a): world_feat_decode_r2 = held-out ridge R2
                   z_world[t] -> harm_obs[t+1] (predictive, JL-safe, INCREASES with maturation).
    DV (primary):  harm_eval SD-003 held-out R2 on next-step hazard-proximity world
                   feature, + gap = train_R2 - test_R2.
    POS. CTRL:     hazard_feat_decode_r2 = held-out ridge R2 z_world[t] -> next-step
                   hazard-proximity scalar (decodable-in-principle), gated at onset_max
                   (this is the corrected positive control that 740a's harm target FAILED
                   -- here it reads the proven-decodable world-feature leg, so it should
                   PASS).

REGIME (unchanged -- 537b lesson).
    scheduled_external_hazard is OFF: by-design-unpredictable ext events would floor
    world-feature predictability regardless of E1 maturity, making the maturity contrast
    uninterpretable. OFF -> world features are predictable-from-state, so E1/E2
    differentiation is the binding constraint (the regime in which INV-088's prediction
    is falsifiable).

PRE-REGISTERED PASS CRITERIA (evidence supports INV-088; on the PRIMARY harm_eval SD-003
DV):
    C1 quality-gain:  mean_seed(R2_test[onset_max] - R2_test[onset_min]) >= 0.15
                      AND that mean delta >= 2.0 * SD_seed(delta)   (effect-size gate)
    C2 monotone:      mean_seed Spearman(onset, R2_test across arms) >= 0.80
    C3 noise-fit floor: mean_seed(R2_test[onset_min]) <= 0.05
                      (the IMMATURE-anchor evaluator is noise-fitted -- no better than
                      predicting the target mean, held-out R2 ~ 0. This directly
                      operationalises INV-088's "poorly-differentiated z_world yields a
                      NOISE-FITTED goal/world-feature evaluator, so productive training
                      cannot begin until sufficient differentiation". It is the
                      load-bearing NON-TAUTOLOGICAL criterion vs the ridge IV: the IV is a
                      slope; C3 asserts the evaluator is USELESS at immaturity, not merely
                      worse. NOTE the train-test GAP is NOT the signature here -- with a
                      smooth world-feature regression target a poorly-differentiated
                      z_world makes the small head UNDER-fit (train ~ test ~ 0), not
                      over-fit; the noise-fit manifests as R2_test ~ 0 at the anchor. The
                      gap is still reported as a diagnostic.
    PASS = C1 and C2 and C3.

PRECONDITIONS (validity -- if unmet the contrast is vacuous, not a real FAIL):
    PC_iv_moved:       mean_seed(world_feat_decode_r2[onset_max]
                                - world_feat_decode_r2[onset_min]) >= 0.03  AND > 0
                       (the z_world-differentiation gradient actually moved -- reuses
                       740a's validated IV; without it the DV coupling has no gradient to
                       track).
    PC_dv_decodable:   mean_seed(hazard_feat_decode_r2[onset_max]) >= 0.05
                       (mature-anchor positive control on the PRIMARY DV target: the
                       next-step world/goal feature is decodable-in-principle from
                       z_world, so a null DV gradient is not confounded by an undecodable
                       target -- the corrected 740a control, reading the proven world-
                       feature leg).
    PC_target_var:     min over cells of the primary DV target std >= 0.02
                       (the target is not degenerate/constant, so held-out R2 is
                       well-defined).
    If a precondition is unmet the run is marked non_degenerate=False (excluded from
    governance scoring) and evidence_direction="unknown".

RE-DERIVE BRAKE / SCOPE NOTE:
    INV-088 is a NEW claim (0 prior autopsies) -- the re-derive brake does NOT fire. The
    INV-064 brake (2 non_contributory autopsies: 740, 740a) refuses a SAME-OBSERVABLE
    z_world 740b that re-decodes REALIZED SCALAR HARM from z_world. This experiment does
    the opposite: it reads the WORLD/GOAL-FEATURE target the 740a autopsy PROVED is
    decodable-and-rising from z_world, which the /claim-synthesis explicitly routes as
    child (i)'s "testable-now" DV-coupling extension of the validated IV. It is the
    sanctioned path, not the refused loop. The z_harm harm leg (realized nociception) is
    INV-089, tested separately in V3-EXQ-743 on z_harm.
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

EXPERIMENT_TYPE = "v3_exq_744_inv088_world_goal_evaluator_dv_coupling"
QUEUE_ID = "V3-EXQ-744"
SUPERSEDES = None  # EXTENDS 740a (different claim: INV-088 vs INV-064); does not supersede
CLAIM_IDS = ["INV-088"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants (mirror 740a so the IV gradient is identical) ---
SEEDS = [42, 7, 19]
ONSET_EPISODES = [0, 1, 4, 12, 30]        # onset_0 = fresh encoder (immature anchor)
WARMUP_PROGRESS_DENOM = max(ONSET_EPISODES)  # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared frozen-dataset collection (identical per arm)
EVAL_EPOCHS = 40                     # FIXED evaluator-head training budget (all arms/heads)
EVAL_BATCH = 64
EVAL_LR = 1e-3
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear decode probes
EVAL_INIT_SEED = 90001              # fixed evaluator-head init (all arms/heads identical)
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

# --- pre-registered thresholds ---
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
    """Participation ratio of the z_world covariance eigenspectrum. Diagnostic only
    (ran backwards in 740; retained for continuity)."""
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
    output dims). Pure read-out of the FROZEN representation (no gradients), distinct
    from the trained evaluator head (the DV)."""
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
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    arm = ArmSpec(arm_id="maturation", gap4_operating=False)
    cfg = build_config(env, arm)  # from_dims path -> alpha_world=0.9 (SD-008)
    agent = REEAgent(cfg)
    return agent, env


def _row_vec(x: Any) -> torch.Tensor:
    """Coerce an obs harm field to a [1, D] float32 row tensor."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    return t.reshape(1, -1)


def _collect_frozen_dataset(agent: REEAgent, seed: int, steps_per_ep: int,
                            n_episodes: int) -> Dict[str, torch.Tensor]:
    """Replay a FIXED action sequence through a FIXED collection env, encoding z_world
    with the (frozen) agent. Identical raw trajectory across onset arms; only the encoder
    differs. Collects z_world[t], harm_obs[t], harm_obs[t+1], and E2 transition triples.
    """
    collect_env = CausalGridWorldV2(seed=COLLECT_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(COLLECT_SEED_BASE + seed)
    action_dim = collect_env.action_dim

    z_list: List[torch.Tensor] = []
    hcur_list: List[torch.Tensor] = []
    hnext_list: List[torch.Tensor] = []
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

                _, _harm_signal, done, _info, obs_next = collect_env.step(action)

                z_list.append(z_world)
                hcur_list.append(_row_vec(obs_dict.get("harm_obs")))
                hnext_list.append(_row_vec(obs_next.get("harm_obs")))

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
    Hcur = torch.cat(hcur_list, dim=0)
    Hnext = torch.cat(hnext_list, dim=0)
    out = {"Z": Z, "Hcur": Hcur, "Hnext": Hnext}
    if zprev_list:
        out["Zprev"] = torch.cat(zprev_list, dim=0)
        out["A"] = torch.cat(a_list, dim=0)
        out["Zcurr"] = torch.cat(zcurr_list, dim=0)
    return out


def _next_step_feature(Hnext: torch.Tensor, slc: Tuple[int, int]) -> torch.Tensor:
    """Mean proximity over a world-feature block of harm_obs[t+1] -> [n, 1] scalar target.

    NEXT-STEP (predictive) -> JL-safe: a fresh random-projection encoder cannot predict
    next-step world structure, so evaluator quality genuinely varies with maturation.
    Dataset-normalised to [0,1] (min-max over the shared trajectory, identical across the
    onset arms of a seed) so the Sigmoid substrate heads (harm_eval / benefit_eval, output
    [0,1]) can reach the target.
    """
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


def _freeze(agent: REEAgent) -> None:
    params = (
        list(agent.e1.parameters())
        + list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
        + list(agent.latent_stack.parameters())
    )
    for p in params:
        p.requires_grad_(False)


def _split_indices(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed shuffled train/test split -- identical indices across all arms/heads
    (depends only on n, which is identical across arms because the trajectory is shared).
    Shuffling avoids a feature-sparse contiguous tail making held-out R^2 degenerate."""
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

    Fixed, bit-identical init + optimisation RNG + split across all arms/heads, so the
    ONLY per-arm variable is the frozen z_world (E1/E2 differentiation).
    """
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
    # Cells are NOT reuse-eligible: the frozen representation is a function of the shared
    # maturation trajectory, not reproducible from config_slice alone. The fingerprint is
    # emitted (validator requirement) but flagged ineligible -- no baseline mint applies.
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=["frozen_representation_from_maturation_trajectory"],
    ) as cell:
        agent, env = _build_agent(seed)
        # Snapshot the fresh head inits BEFORE any training so every arm starts each
        # evaluator from bit-identical weights.
        harm_init = copy.deepcopy(agent.e3.harm_eval_head.state_dict())
        benefit_init = copy.deepcopy(agent.e3.benefit_eval_head.state_dict())

        # 1. Mature E1 + E2.world_forward + encoder (heads trained here are discarded via
        #    the re-init in step 5). onset=0 -> fresh encoder anchor.
        warmup_train(
            agent, env,
            num_episodes=onset,
            steps_per_episode=steps_per_ep,
            label=f"mat seed={seed} onset={onset}",
            progress_total_episodes=WARMUP_PROGRESS_DENOM,
        )

        # 2. Freeze E1/E2/encoder.
        _freeze(agent)

        # 3. Collect shared frozen dataset (identical trajectory across arms).
        data = _collect_frozen_dataset(agent, seed, steps_per_ep, collect_eps)
        Z = data["Z"]

        # DV targets: NEXT-STEP world/goal-feature scalars (JL-safe, decodable leg).
        hazard_next = _next_step_feature(data["Hnext"], HAZARD_SLICE)       # PRIMARY (SD-003)
        resource_next = _next_step_feature(data["Hnext"], RESOURCE_SLICE)   # SECONDARY (goal)

        # Differentiation mediators at freeze.
        zworld_var = float(Z.var(dim=0).mean().item())
        zworld_eff_rank = _eff_rank(Z)                 # diagnostic only
        e2_r2 = _e2_forward_r2(agent, data)
        # IV (reused from 740a): predictive full harm_obs decodability -- the z_world
        # differentiation gradient.
        world_feat_decode_r2 = _ridge_heldout_r2(Z, data["Hnext"], DECODE_SPLIT_SEED)
        # Positive controls: are the DV targets decodable-in-principle from z_world?
        hazard_feat_decode_r2 = _ridge_heldout_r2(Z, hazard_next, DECODE_SPLIT_SEED)   # PRIMARY PC
        resource_feat_decode_r2 = _ridge_heldout_r2(Z, resource_next, DECODE_SPLIT_SEED)  # advisory

        # 5-6. Controlled frozen evaluator training + readout (both real substrate heads).
        harm_eval = _train_eval_head(agent.e3.harm_eval_head, harm_init, Z, hazard_next,
                                     eval_epochs, "harm_sd003")
        benefit_eval = _train_eval_head(agent.e3.benefit_eval_head, benefit_init, Z,
                                        resource_next, eval_epochs, "benefit")

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
    # secondary (benefit) corroboration
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
        # secondary
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

    # secondary (benefit) corroboration -- reported, NOT load-bearing
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
    c3 = mean_immature_r2 <= NOISE_FIT_CEIL   # immature evaluator is noise-fitted (~chance)

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
                f"delta {criteria['mean_iv_delta']:.3f} < {IV_MOVE_FLOOR} or <= 0 "
                f"(z_world differentiation gradient did not increase across arms)")
        if not criteria["PC_dv_decodable"]:
            reasons.append(
                f"primary DV target not decodable-in-principle: mean mature-anchor "
                f"hazard_feat_decode_r2 {criteria['mean_dv_decode_mature']:.3f} "
                f"< {DV_DECODABLE_FLOOR} (a null DV gradient would be confounded by an "
                f"undecodable world-feature target)")
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
        "extends": "V3-EXQ-740a",
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
            "NOISE_FIT_CEIL": NOISE_FIT_CEIL,
            "IV_MOVE_FLOOR": IV_MOVE_FLOOR,
            "DV_DECODABLE_FLOOR": DV_DECODABLE_FLOOR,
            "TARGET_STD_FLOOR": TARGET_STD_FLOOR,
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
    result = run_experiment(dry_run=dry_run)

    if dry_run:
        print(f"DRY_RUN complete: {result['n_cells']} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result)
    c = result["criteria"]
    print("=== INV-088 world/goal-evaluator DV-coupling result (744) ===", flush=True)
    print(f"  preconditions_met: {result['preconditions_met']} "
          f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.3f}; "
          f"dv_decodable={c['PC_dv_decodable']} mature_hazard_decode_r2={c['mean_dv_decode_mature']:.3f}; "
          f"target_var={c['PC_target_var']} min_std={c['min_target_std']:.4f})", flush=True)
    print(f"  C1 quality_gain: {c['C1_quality_gain']} "
          f"(mean_delta_r2={c['mean_delta_r2']:.3f}, sd={c['sd_delta_r2']:.3f})", flush=True)
    print(f"  C2 monotone:     {c['C2_monotone']} (mean_rho={c['mean_spearman_rho']:.3f})", flush=True)
    print(f"  C3 noise_fit:    {c['C3_noise_fit_floor']} (mean_immature_r2={c['mean_immature_r2']:.3f} <= {NOISE_FIT_CEIL}; "
          f"diag mean_gap_delta={c['mean_gap_delta']:.3f})", flush=True)
    print(f"  [secondary] benefit_eval corroborates: {c['benefit_corroborates']} "
          f"(delta={c['mean_benefit_delta']:.3f}, rho={c['mean_benefit_rho']:.3f})", flush=True)
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
