#!/opt/local/bin/python3
"""V3-EXQ-740a: INV-064 maturational-sequence necessity -- CORRECTED successor to
V3-EXQ-740. E3 evaluator quality is bounded by E1/E2 representational
differentiation (curriculum-order diagnostic).

SUPERSEDES V3-EXQ-740 (FAIL / non_contributory / measurement_degeneracy). See
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-740_2026-07-11.{md,json}.
740 FAILed on its VALIDITY PRECONDITION only (PC_iv_moved): the differentiation
observable it used -- participation-ratio effective rank of the alpha_world=0.9 EMA
z_world sampled over a random-action replay trajectory -- ran BACKWARDS (mean
onset_max/onset_min ratio 0.770 < 1.15; per-seed [0.855, 0.493, 0.962]) because a
trained EMA world-latent specialises onto a low-dim task manifold, so its
trajectory-covariance rank DECREASES with maturation -- opposite to INV-064's
"differentiation" (task-relevant discriminability, which INCREASES). Two co-dominant
root causes: (1) MEASUREMENT (eff_rank of a smoothed EMA over a drifting replay is
the wrong observable); (2) WINDOW (onset 2..22 sat entirely post-saturation --
E2 forward R^2 ~ 0.99 at onset_2 -- so no genuinely-immature anchor arm existed).

CLAIM UNDER TEST (INV-064, e1_e2_e3_maturational_sequence_necessity):
    "E3's primary inputs are z_world from E1 and action_objects from E2. If these
    inputs carry only poorly-differentiated, low-information representations, E3
    training produces a noise-fitted harm/goal evaluator. The information quality
    available to E3 is strictly bounded by E1/E2 representational differentiation.
    Therefore productive E3 training cannot begin until E1/E2 have reached
    sufficient schema differentiation."

WHAT CHANGED vs 740 (the four autopsy-mandated fixes):
    FIX 1 -- TASK-RELEVANT DIFFERENTIATION OBSERVABLE THAT INCREASES WITH MATURATION.
        Replace the raw eff_rank (a covariance-SPREAD metric, which random/untrained
        projections inflate -- JL) with a DECODABILITY metric: how well a frozen
        z_world linearly PREDICTS the NEXT-step harm-relevant world-feature vector
        (harm_obs[t+1] from z_world[t]). Predictive decodability is JL-SAFE -- a
        random encoder cannot predict next-step world structure (that needs the
        learned dynamics), so this measure genuinely INCREASES as E1/E2 mature. This
        is INV-064's "task-relevant discriminability", not covariance drift. (The old
        trajectory eff_rank is still recorded for continuity/diagnosis -- expect it
        to keep running backwards, documenting the fix.)
    FIX 2 -- A GENUINELY-IMMATURE ANCHOR ARM. Onset now spans 0..30 episodes:
        onset_0 is a FRESH (untrained) encoder, onset_1 is barely-trained. 740's
        window (2..22) was entirely past E2-forward saturation; extending down to
        0/1 creates a real immature->mature gradient for the IV to move across.
    FIX 3 -- A MATURE-ANCHOR HARM-DECODABILITY POSITIVE CONTROL. Before reading the
        gradient, verify realized harm is decodable-IN-PRINCIPLE from the mature
        z_world (a closed-form LINEAR probe z_world[t]->harm_target[t] held-out R^2
        clears a floor at onset_max). 740's secondary signal was a NEGATIVE held-out
        harm R^2 in ALL arms while E2 forward R^2 ~ 0.99 -- so a null DV gradient
        could be confounded by harm being intrinsically undecodable from z_world.
        If this positive control fails, the run is marked degenerate (confounded),
        NOT a weakens.
    FIX 4 -- PC_iv_moved GUARD ON THE CORRECTED OBSERVABLE, CHECKING THE PREDICTED
        DIRECTION. The precondition now requires the predictive-decodability DELTA
        (onset_max - onset_min) to clear a floor AND be positive (increase with
        maturation), rather than a ratio on a metric that can move either way.

WHY DECODABILITY OVER THE SHARED TRAJECTORY (not a controlled probe-set):
    740's covariance-drift confound was specific to eff_rank (a SPREAD statistic) over
    a drifting EMA replay. A DECODABILITY statistic measures information/legibility,
    not spread, so it is not faked by drift -- the shared fixed-trajectory design
    (identical raw states + harm labels across the onset arms of a seed; ONLY the
    frozen encoder differs) is retained, keeping the contrast clean. The autopsy's
    controlled-probe-set option was offered specifically to rescue eff_rank; switching
    the primary observable to predictive decodability makes it unnecessary.

MEASUREMENT-ONLY, COMMITMENT-FREE DESIGN (unchanged from 740).
    This probe requires NO sustained multi-step action-commitment layer (no
    basal-ganglia / F-dominance path is exercised). It is a controlled
    curriculum-ORDER contrast on a frozen representation, so a PASS/FAIL is a real
    verdict on INV-064's central mechanism, not a re-derivation of the conversion
    ceiling.

    Per (seed, onset) cell:
      1. Mature E1 + E2.world_forward + the encoder (latent_stack) for `onset`
         episodes via the canonical goal_pipeline warmup_train (onset=0 -> fresh
         encoder, the immature anchor). The harm_eval head trained incidentally here
         is DISCARDED (re-initialised in step 4), so E3's training budget is
         controlled independently of maturity.
      2. FREEZE E1, E2.world_transition, E2.world_action_encoder, latent_stack.
      3. Collect a harm dataset by replaying a FIXED action sequence (seeded RNG,
         independent of maturity) through a FIXED collection env. Because the raw
         trajectory + harm labels are identical across the onset arms of a seed, the
         ONLY thing that varies across arms is the frozen encoder's z_world -- so any
         difference in E3 quality (or decodability) is attributable to E1/E2
         differentiation, not to data quantity or the harm-label distribution.
      4. Re-initialise agent.e3.harm_eval_head to a FIXED init (bit-identical across
         arms) and train ONLY that head for a FIXED budget (E3_EPOCHS) on the
         frozen-encoded (z_world, harm_target) tensors.
      5. Read out: E3 quality (held-out harm R^2, train R^2, gap = train-test); the
         corrected IV (predictive harm-feature decodability); and the positive-control
         linear harm decodability.

    IV  (per arm): onset episodes {0, 1, 4, 12, 30}  (E1/E2 maturity at E3 onset).
    IV OBSERVABLE (corrected): world_feat_decode_r2 = held-out R^2 of a ridge linear
                   probe z_world[t] -> harm_obs[t+1] (predictive, JL-safe, INCREASES
                   with maturation).
    DV:            harm_eval held-out R^2, and gap = train_R2 - test_R2.
    POSITIVE CTRL: harm_decode_r2 = held-out R^2 of a ridge linear probe
                   z_world[t] -> harm_target[t] (decodable-in-principle), gated at
                   onset_max.

REGIME (unchanged -- 537b lesson).
    scheduled_external_hazard is OFF: injecting by-design-unpredictable ext events
    floors harm R^2 regardless of E1 maturity, which would make the maturity contrast
    uninterpretable. In the OFF regime harm is predictable-from-state, so E1/E2
    differentiation is the binding constraint -- the regime in which INV-064's
    prediction is falsifiable.

PRE-REGISTERED PASS CRITERIA (evidence supports INV-064):
    C1 quality-gain:  mean_seed(R2_test[onset_max] - R2_test[onset_min]) >= 0.15
                      AND that mean delta >= 2.0 * SD_seed(delta)   (effect-size gate)
    C2 monotone:      mean_seed Spearman(onset, R2_test across arms) >= 0.80
    C3 noise-signature: mean_seed(gap[onset_min] - gap[onset_max]) >= 0.08
    PASS = C1 and C2 and C3.

PRECONDITIONS (validity -- if unmet the contrast is vacuous, not a real FAIL):
    PC_iv_moved:      mean_seed(world_feat_decode_r2[onset_max]
                               - world_feat_decode_r2[onset_min]) >= 0.03  AND > 0
                      (task-relevant differentiation actually INCREASED across arms).
    PC_harm_decodable: mean_seed(harm_decode_r2[onset_max]) >= 0.05
                      (mature-anchor positive control: realized harm is decodable-
                      in-principle from z_world, so a null DV gradient is not
                      confounded by an undecodable harm target).
    PC_events:        min over cells of harm_event_frac >= 0.03  (enough harm to fit).
    If a precondition is unmet the run is marked non_degenerate=False (excluded from
    governance scoring) and evidence_direction="unknown".

RE-DERIVE NOTE (autopsy pre-registration): this is the corrected FIRST successor.
    A SECOND non_contributory / degenerate INV-064 result would implicate the
    OPERATIONALISATION itself -> route to a test-bed / measurement redesign, NOT a
    third lettered iteration circling the same observable.
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

EXPERIMENT_TYPE = "v3_exq_740a_inv064_maturational_sequence_e3_bounded"
QUEUE_ID = "V3-EXQ-740a"
SUPERSEDES = "V3-EXQ-740"
CLAIM_IDS = ["INV-064"]
EXPERIMENT_PURPOSE = "evidence"

# --- IV / design constants ---
SEEDS = [42, 7, 19]
# FIX 2: extend down to a genuinely-immature anchor (onset_0 = fresh encoder,
# onset_1 = barely-trained). 740's [2,5,11,22] was entirely post-saturation.
ONSET_EPISODES = [0, 1, 4, 12, 30]
WARMUP_PROGRESS_DENOM = max(ONSET_EPISODES)   # stable [train] ep N/M denominator across arms
STEPS_PER_EP = 50
COLLECT_EPISODES = 14                # shared harm-dataset collection (identical per arm)
E3_EPOCHS = 40                       # FIXED harm_eval training budget (all arms)
E3_BATCH = 64
E3_LR = 1e-3
HELDOUT_FRAC = 0.3
RIDGE_LAMBDA = 1.0                   # ridge regularisation for the linear decode probes
E3_INIT_SEED = 90001                 # fixed harm_eval_head init (all arms identical)
E3_TRAIN_SEED = 90002                # fixed E3 optimisation RNG (all arms identical)
DECODE_SPLIT_SEED = 90003            # fixed decode-probe train/test split (all arms identical)
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
IV_MOVE_FLOOR = 0.03            # min predictive-decodability increase onset_min->onset_max
HARM_DECODABLE_FLOOR = 0.05    # mature-anchor linear harm-decodability positive-control floor
HARM_EVENT_FRAC_FLOOR = 0.03


def _r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    ss_res = ((pred - tgt) ** 2).sum().item()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _eff_rank(Z: torch.Tensor) -> float:
    """Participation ratio of the z_world covariance eigenspectrum.

    RETAINED FROM 740 FOR CONTINUITY/DIAGNOSIS ONLY -- this is the metric that ran
    BACKWARDS in 740 (a spread statistic over a drifting EMA replay). It is NOT a
    precondition in 740a; the corrected IV is world_feat_decode_r2. Expect this to
    keep decreasing with maturation, documenting the fix.
    """
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
    so an uninformative representation scores <= 0). Pooled over output dims for
    multi-output T (harm_obs is 51-dim). This is a pure read-out of the FROZEN
    representation -- no gradients, distinct from the trained harm_eval MLP (the DV).
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
    # Augment a bias column.
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
    """Replay a FIXED action sequence through a FIXED collection env, encoding
    z_world with the (frozen) agent. Identical raw trajectory across onset arms;
    only the encoder differs.

    Collects, per step:
      Z      = z_world[t]                       (frozen encoding, the only per-arm var)
      Y      = harm_target[t]                    (DV target + positive-control target)
      Hcur   = harm_obs[t]                       (current harm-relevant world features)
      Hnext  = harm_obs[t+1]                     (FIX 1: predictive IV decode target)
    plus the E2 forward-model transition triples (Zprev, A, Zcurr).
    """
    collect_env = CausalGridWorldV2(seed=COLLECT_SEED_BASE + seed, **ENV_KWARGS)
    act_rng = np.random.default_rng(COLLECT_SEED_BASE + seed)
    action_dim = collect_env.action_dim

    z_list: List[torch.Tensor] = []
    y_list: List[float] = []
    hcur_list: List[torch.Tensor] = []
    hnext_list: List[torch.Tensor] = []
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
    Y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    Hcur = torch.cat(hcur_list, dim=0)
    Hnext = torch.cat(hnext_list, dim=0)
    out = {"Z": Z, "Y": Y, "Hcur": Hcur, "Hnext": Hnext}
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
        "ridge_lambda": RIDGE_LAMBDA,
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
        #    discarded via the re-init in step 4). onset=0 -> fresh encoder anchor.
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

        # Differentiation mediators at freeze.
        zworld_var = float(data["Z"].var(dim=0).mean().item())
        zworld_eff_rank = _eff_rank(data["Z"])          # OLD metric (diagnosis only)
        e2_r2 = _e2_forward_r2(agent, data)
        # FIX 1: corrected IV -- predictive harm-feature decodability (JL-safe, rises
        # with maturation). Also the current-step decode for diagnosis.
        world_feat_decode_r2 = _ridge_heldout_r2(data["Z"], data["Hnext"], DECODE_SPLIT_SEED)
        world_feat_decode_r2_current = _ridge_heldout_r2(data["Z"], data["Hcur"], DECODE_SPLIT_SEED)
        # FIX 3: positive-control -- linear harm decodability (decodable-in-principle).
        harm_decode_r2 = _ridge_heldout_r2(data["Z"], data["Y"], DECODE_SPLIT_SEED)

        # 4-5. Controlled frozen E3 training + readout.
        e3 = _train_e3_and_eval(agent, data, e3_init_state, e3_epochs)

        row: Dict[str, Any] = {
            "arm_id": f"onset_{onset}",
            "onset_episodes": onset,
            "seed": seed,
            "zworld_var": zworld_var,
            "zworld_eff_rank": zworld_eff_rank,
            "e2_forward_r2": e2_r2,
            "world_feat_decode_r2": world_feat_decode_r2,
            "world_feat_decode_r2_current": world_feat_decode_r2_current,
            "harm_decode_r2": harm_decode_r2,
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
    per_seed_iv_delta: List[float] = []
    per_seed_harm_decode_mature: List[float] = []
    for seed in SEEDS:
        r2s = [_cell(seed, o)["harm_r2_test"] for o in ONSET_EPISODES]
        per_seed_delta_r2.append(_cell(seed, onset_max)["harm_r2_test"]
                                 - _cell(seed, onset_min)["harm_r2_test"])
        per_seed_gap_delta.append(_cell(seed, onset_min)["harm_gap"]
                                  - _cell(seed, onset_max)["harm_gap"])
        per_seed_rho.append(_spearman([float(o) for o in ONSET_EPISODES], r2s))
        # FIX 4: corrected IV movement -- predictive-decodability DELTA (predicted +).
        per_seed_iv_delta.append(_cell(seed, onset_max)["world_feat_decode_r2"]
                                 - _cell(seed, onset_min)["world_feat_decode_r2"])
        # FIX 3: mature-anchor positive control.
        per_seed_harm_decode_mature.append(_cell(seed, onset_max)["harm_decode_r2"])

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
    mean_iv_delta = _mean(per_seed_iv_delta)
    mean_harm_decode_mature = _mean(per_seed_harm_decode_mature)
    min_harm_event_frac = min(r["harm_event_frac"] for r in rows)

    # Preconditions (validity).
    pc_iv_moved = (mean_iv_delta >= IV_MOVE_FLOOR) and (mean_iv_delta > 0.0)
    pc_harm_decodable = mean_harm_decode_mature >= HARM_DECODABLE_FLOOR
    pc_events = min_harm_event_frac >= HARM_EVENT_FRAC_FLOOR
    preconditions_met = pc_iv_moved and pc_harm_decodable and pc_events

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
        "per_seed_iv_delta": per_seed_iv_delta,
        "per_seed_harm_decode_mature": per_seed_harm_decode_mature,
        "mean_delta_r2": mean_delta_r2,
        "sd_delta_r2": sd_delta_r2,
        "mean_gap_delta": mean_gap_delta,
        "mean_spearman_rho": mean_rho,
        "mean_iv_delta": mean_iv_delta,
        "mean_harm_decode_mature": mean_harm_decode_mature,
        "min_harm_event_frac": min_harm_event_frac,
        "PC_iv_moved": pc_iv_moved,
        "PC_harm_decodable": pc_harm_decodable,
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
            "mean_world_feat_decode_r2": float(np.mean([c["world_feat_decode_r2"] for c in cells])),
            "mean_world_feat_decode_r2_current": float(
                np.mean([c["world_feat_decode_r2_current"] for c in cells])),
            "mean_harm_decode_r2": float(np.mean([c["harm_decode_r2"] for c in cells])),
        }
    return out


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [SEEDS[0]]
        onsets = [0, 2]
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
                f"IV did not move in the predicted direction: mean predictive-decode "
                f"delta {criteria['mean_iv_delta']:.3f} < {IV_MOVE_FLOOR} or <= 0 "
                f"(E1/E2 task-relevant differentiation did not increase across arms)")
        if not criteria["PC_harm_decodable"]:
            reasons.append(
                f"mature-anchor harm not decodable-in-principle: mean linear harm "
                f"decode R2 {criteria['mean_harm_decode_mature']:.3f} "
                f"< {HARM_DECODABLE_FLOOR} (DV null would be confounded by an "
                f"undecodable harm target)")
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
        "supersedes": SUPERSEDES,
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
            "IV_MOVE_FLOOR": IV_MOVE_FLOOR,
            "HARM_DECODABLE_FLOOR": HARM_DECODABLE_FLOOR,
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
    print("=== INV-064 maturational-sequence result (740a, corrected) ===", flush=True)
    print(f"  preconditions_met: {result['preconditions_met']} "
          f"(iv_moved={c['PC_iv_moved']} iv_delta={c['mean_iv_delta']:.3f}; "
          f"harm_decodable={c['PC_harm_decodable']} mature_harm_decode_r2={c['mean_harm_decode_mature']:.3f}; "
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
