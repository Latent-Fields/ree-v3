"""V3-EXQ-805: ARC-016 -- recalibrating the precision-to-commitment threshold from
EVAL-time variance. Closes the one open item the claim's own notes name.

Authored from EXP-0397 (REE_assembly evidence/planning/manual_proposals.v1.json,
minted by /thought-digestion 2026-07-21). claim_ids = [ARC-016] ONLY.

=== THIS IS NOT A DISCOVERY EXPERIMENT ===
ARC-016's core circuit -- E3-derived variance -> relative commit threshold ->
BetaGate -> action_selection -- is ALREADY VALIDATED:

    V3-EXQ-018b  PASS 5/5 (2026-03-20). commit_threshold = 2 x
                 training_baseline_variance. Stable env: commit_rate 0.90,
                 precision 718. Perturbed: commit_rate 0.50, precision 426. A 40%
                 precision drop produced a proportional 40-point commit-rate drop.
    V3-EXQ-060   PASS 4/5 (2026-03-21). committed_step_count 5980,
                 hold_rate_during_committed 0.936.

What this run closes is the SINGLE open item named verbatim in the claim's own
notes: **"Pending recalibration with eval-derived threshold."**

=== THE 396-SERIES DEATH, AND WHY IT IS A CALIBRATION ARTEFACT ===
EXQ-396a (x3 runs) and EXQ-396b were ALL reclassified non_contributory. Cause: a
TRAIN/EVAL VARIANCE GAP. Training variance settles at ~0.000032-0.000064 while
eval-time variance runs at ~0.001 -- roughly 15-30x. A threshold pinned at
2 x training_baseline_variance therefore sits ~15-30x BELOW the variance it must
discriminate, the gate `committed = running_variance < effective_threshold`
(e3_selector.py:2792) never fires in eval, and EVERY arm is identical. Nobody has
since re-run the design with the threshold derived from EVAL-time variance.

That is the entire manipulation here: **the baseline-variance SOURCE is the only
manipulated factor.** Everything else reproduces the 396 / 018b design.

=== ARMS (threshold calibration source) ===
  A0_TRAIN_BASELINE   threshold = 2 x mean rv over the last BASELINE_WINDOW
                      TRAINING episodes. Reproduces the 396 configuration as the
                      pre-registered NEGATIVE REFERENCE. It is EXPECTED not to
                      engage; that non-engagement is its informative content.
  A1_EVAL_DERIVED     threshold = 2 x mean rv over an EVAL-time calibration window
                      run in the STABLE reference condition under the SAME frozen
                      /reset protocol the scored evals use. The proposed fix.
  A2_ONLINE_ADAPTED   threshold_t = 2 x a SLOW trailing EMA of rv, seeded at the
                      eval-derived baseline and updated every tick.

THE THRESHOLD IS DERIVED ONCE FROM THE REFERENCE CONDITION AND HELD FIXED across
the manipulated precision levels (A0/A1). Deriving it per-condition would ABSORB
the manipulation and measure nothing -- this is the same reason 018b calibrated on
the stable env and then evaluated stable-vs-perturbed.

=== PRECISION MANIPULATION (as EXQ-018b) ===
  stable    num_hazards=2,  env_drift_interval=200, env_drift_prob=0.0
  perturbed num_hazards=25, env_drift_interval=1,   env_drift_prob=1.0
Maximal variance contrast, the same extreme pairing 018b used to get its 5/5.

=== SCOPE GUARD -- STRUCTURAL ONLY (do NOT add a behavioural harm DV) ===
Post-split (2026-03-22) ARC-016 covers ONLY the structural/mechanistic circuit. The
BEHAVIOURAL consequence layer (committed vs uncommitted -> measurably distinct harm
outcomes) is **ARC-029** and is NOT tested here. Attaching a behavioural harm DV
would re-import the V_s MONOSTRATEGY-LOCK confound that voided EXQ-454, where
threshold-adaptation behaviour was dominated by the policy lock-in rather than by
the precision-to-commitment circuit under test.

The design excludes that confound STRUCTURALLY, not by promise: the driver takes
RANDOM actions throughout (as 018b/396 did) and reads commitment off
`agent.e3.select(...).committed`. With no learned policy there is no V_s
monostrategy lock to dominate anything. No harm/benefit quantity is a DV anywhere
in this run.

Calling e3.select() directly also means every call is one genuine, independent
selection: the E3 cadence (heartbeat.e3_steps_per_tick, default 10) and the
commitment latch inside agent.select_action are not in the path, so the V3-EXQ-785
~9x diagnostics pseudo-replication defect is structurally impossible here.
n_latched_ticks is 0 by construction and is emitted as 0.

=== NON-DEGENERACY: THE THRESHOLD MUST ACTUALLY ENGAGE ===
This is the acceptance check that voided EXQ-396a/b, and it is a PRECONDITION here,
never a criterion -- a threshold that never fires makes all arms identical and must
self-route substrate_not_ready_requeue, NOT a verdict on ARC-016.

  P2 THRESHOLD ENGAGEMENT HEADROOM. `min(commit_rate, 1 - commit_rate)` above a
     floor, taken over the WORST of the arm's two conditions. Expressed as a
     single-bound FLOOR on a worst-cell statistic so the indexer's recompute is
     exact (a two-sided band cannot be declared through PreconditionSpec, and a
     half-declared band is the V3-EXQ-779b defect: the undeclared leg silently
     passes). Note this guard covers BOTH conditions of the arm, not just the
     reference one -- V3-EXQ-779b/777's saturation guard inspected only the
     baseline partition, i.e. the arm LEAST likely to saturate.
     SCOPED OUT of A0_TRAIN_BASELINE (disposition (a), applies_to): A0 is the
     pre-registered NEGATIVE REFERENCE whose designated role is to reproduce the
     396 non-engagement, so asserting engagement there would make it structurally
     un-passable and would collapse the three-arm design. A0's headroom is still
     MEASURED and emitted as a NON-GATING diagnostic -- if A0 unexpectedly engages,
     that is visible rather than hidden by the scoping.

  P1 PRECISION MANIPULATION TOOK. The RELATIVE variance lift
     (rv(perturbed) - rv(stable)) / rv(stable) above a floor. If the perturbation
     does not raise prediction-error variance, nothing downstream can track it and
     the run measures the env, not the circuit. RELATIVE and not absolute: rv's
     operating scale is itself what this experiment recalibrates against, so no
     absolute floor is meaningful across arms -- and an over-permissive absolute
     floor would additionally let a ~0 denominator blow up A2's tracking ratio
     (a shift-over-shift quotient). EXQ-018b's C1 was relative for the same reason.
  P3 COMMIT-DECISION COUNT floor.
  P4 CALIBRATED THRESHOLD FINITE AND POSITIVE (018b's C0).

Preconditions are evaluated PER ARM (experiments/_lib/precondition_gate.py) and
aggregated with aggregate_arm_gates, so a red arm cannot vacate a green one.

=== DV-SYMMETRY DECLARATION (mandatory, per arm) ===
DV = commit_rate = the fraction of selections with
`running_variance < commit_threshold`. Its symmetry group is every transform that
preserves the SIGN of (threshold - rv) at each tick: any monotone transform applied
JOINTLY to rv and the threshold, and any threshold change that stays on the same
side of the rv distribution's mass.

  A1_EVAL_DERIVED -- NOT INVARIANT, and this is the load-bearing point. The train-
    and eval-derived thresholds sit on OPPOSITE SIDES of the eval rv distribution's
    mass (train baseline ~3e-5 against eval rv ~1e-3, ~15-30x), so the sign of
    (threshold - rv) flips for the great majority of ticks. The realised ratio is
    measured and emitted as `threshold_ratio_eval_over_train`, so the claim that
    the manipulation moved the gate is a NUMBER, not an assertion.
  A0_TRAIN_BASELINE -- REFERENCE ARM, declared. Its DV is expected pinned; it is
    not cited as a measured effect and does not route a verdict on its own.
  A2_ONLINE_ADAPTED -- INVARIANCE HAZARD, GUARDED. If the threshold tracked rv
    FAST, then `rv < 2 x EMA(rv)` would be true almost always whatever the
    condition: the manipulation would be ABSORBED and the DV fixed at ~1.0 BY
    ARITHMETIC, not measured. The EMA is therefore deliberately SLOW
    (A2_EMA_ALPHA), and the realised absorption is MEASURED as
    `a2_threshold_tracking_ratio` (how far the arm's threshold moved between the
    stable and perturbed phases, relative to how far rv moved). If it exceeds
    A2_TRACKING_CEILING the arm's threshold followed the variance and its DV is an
    identity -- C3 is then marked criteria_non_degenerate:false and routes nothing.

=== PRE-REGISTERED CRITERIA ===
  C1 TRACKING (LOAD-BEARING): in A1_EVAL_DERIVED, commit_rate tracks manipulated
     precision in the EXQ-018b direction -- commit_rate(stable) - commit_rate
     (perturbed) exceeds max(COMMIT_DELTA_FLOOR, 2 x SD of the per-seed delta).
  C2 396-REPRODUCTION (LOAD-BEARING): A0_TRAIN_BASELINE reproduces the 396
     non-engagement -- its engagement headroom is BELOW the P2 floor (the gate
     never fires) AND its commit delta is below the same margin. Without this the
     run cannot say the 396 result was a CALIBRATION ARTEFACT; it could only say
     "some configuration works".
  C3 ONLINE VARIANT (NOT load-bearing): A2_ONLINE_ADAPTED also tracks. Its null is
     declared: an online baseline that ABSORBS the perturbation is an informative
     result about adaptation TIMESCALE, NOT evidence against ARC-016.

INTERPRETATION GRID:
  C1 pass + C2 pass -> PASS / supports          396_series_was_calibration_artefact
                                                (ARC-016's validation stands
                                                 unqualified; the 396 runs stay
                                                 non_contributory on the merits)
  C1 pass + C2 FAIL -> FAIL / mixed             tracking_holds_but_396_config_not_reproduced
                                                (A1 tracks, but the negative
                                                 reference did not behave as the
                                                 diagnosis requires -- the
                                                 calibration-artefact story is
                                                 unconfirmed)
  C1 FAIL + C2 pass -> FAIL / weakens           commit_rate_does_not_track_precision
                                                DECISION-FLIPPING. With a threshold
                                                that demonstrably ENGAGES (P2
                                                green), commit_rate still fails to
                                                track precision. The 15 prior FAILs
                                                would need re-reading as right for
                                                the wrong reasons and ARC-016
                                                demotes from provisional.
  C1 FAIL + C2 FAIL -> FAIL / non_contributory  calibration_contrast_uninformative
  gate RED          -> FAIL / non_contributory  substrate_not_ready_requeue
                                                NEVER a substrate verdict.

GOVERNANCE FLAG, deliberately NOT acted on by this run: ARC-016's exp_conf 0.53 is
an AGGREGATION ARTEFACT -- 15 FAILs of which nearly all are already stamped
non_contributory or substrate-version-stale (see the claim's heterogeneity_note)
still score it as half-refuted. Re-scoring needs a governance eye, because
stripping non-contributory FAILs can expose that the remaining supports are thinner
than the headline number suggested. This run neither promotes nor re-scores.

GOV-REUSE-1: decisive readout is commit_rate under an EVAL-DERIVED commit threshold
across a manipulated precision level. No recorded manifest carries it -- 018b and
396a/b all derive the threshold from TRAINING variance by construction, which is
precisely the configuration under test as the negative reference, and none records
an eval-time calibration window. Not recoverable -> must run.

MINT-AS-YOU-GO: every cell is stamped with include_driver_script_in_hash=False, so
the A0_TRAIN_BASELINE reference cells are cross-driver reusable by any successor.

TRAINING PROTOCOL: P0 substrate warmup (E1 + E2.self + world-decoder reconstruction
+ E2.world_forward one-step MSE) on the stable env, then FROZEN eval phases under
agent.eval() / no_grad. NO downstream head is trained on a latent, so the P1
freeze-and-detach phase does not apply -- there is no head to chase a moving target.

Run:
  /opt/local/bin/python3 experiments/v3_exq_805_arc016_eval_derived_commit_threshold.py --dry-run
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_805_arc016_eval_derived_commit_threshold"
QUEUE_ID = "V3-EXQ-805"
EXPERIMENT_PURPOSE = "evidence"
# ONLY the claim this implementation directly tests. ARC-029 (the behavioural
# consequence layer, split off 2026-03-22) is deliberately NOT tagged -- no
# behavioural DV exists in this run. The proposal's other related_claims
# (ARC-005 / MECH-025 / MECH-039 / MECH-059 / SD-019 / SD-020 / SD-021 / Q-041 /
# Q-042) are not exercised either.
CLAIM_IDS: List[str] = ["ARC-016"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2]
TRAIN_EPISODES = 250
CAL_EPISODES = 30          # eval-time calibration window, STABLE condition
EVAL_EPISODES = 40         # per scored condition
STEPS_PER_EPISODE = 100
BASELINE_WINDOW = 60       # episodes averaged for a baseline estimate
CAL_TAIL_EPISODES = 20     # calibration episodes averaged, AFTER the reset transient

DRY_TRAIN_EPISODES = 6
DRY_CAL_EPISODES = 3
DRY_EVAL_EPISODES = 3
DRY_STEPS = 25

ARM_TRAIN = "A0_TRAIN_BASELINE"
ARM_EVAL = "A1_EVAL_DERIVED"
ARM_ONLINE = "A2_ONLINE_ADAPTED"
ARMS: List[str] = [ARM_TRAIN, ARM_EVAL, ARM_ONLINE]

CALIBRATION_FACTOR = 2.0   # 018b's value, retained unchanged
E3_DECISION_INTERVAL = 10  # steps between E3 selections (018b's value)
NUM_CANDIDATES = 16
CANDIDATE_HORIZON = 5
ALPHA_WORLD = 0.9          # the 0.3 default is a known SD-008 root cause
SELECT_TEMPERATURE = 1.0
LR = 1e-3
E2W_BUF_MAX = 500
E2W_BATCH = 32

# A2: deliberately SLOW so the threshold cannot absorb the within-phase variance
# shift. A fast EMA would make `rv < 2 x EMA(rv)` true by arithmetic (see the
# DV-symmetry declaration) and the arm would measure nothing.
A2_EMA_ALPHA = 0.002
A2_TRACKING_CEILING = 0.5  # absorbed-fraction above which C3 is declared degenerate

# Env conditions (EXQ-018b's extreme contrast).
ENV_STABLE = dict(size=12, num_hazards=2, num_resources=5,
                  env_drift_interval=200, env_drift_prob=0.0)
ENV_PERTURBED = dict(size=12, num_hazards=25, num_resources=5,
                     env_drift_interval=1, env_drift_prob=1.0)
HAZARD_HARM = 0.02

# Gate floors (FLOORS unless marked upper).
# P1 is RELATIVE, not absolute: (rv_perturbed - rv_stable) / rv_stable. An absolute
# floor is unusable here because rv's operating scale is itself what the experiment
# recalibrates against -- a 1e-6 absolute floor passes a variance shift of
# essentially zero, which (a) means the precision manipulation did not take and (b)
# makes A2's tracking ratio (a shift-over-shift quotient) explode on a ~0
# denominator. EXQ-018b's own C1 was likewise relative (var_diff > 0.5 x baseline);
# 0.25 sits below the ~38% shift 018b measured (stable 0.0026 -> perturbed 0.0036).
VAR_DIFF_REL_FLOOR = 0.25
ENGAGEMENT_HEADROOM_FLOOR = 0.02   # P2: min(cr, 1-cr), WORST of the arm's conditions
MIN_COMMIT_DECISIONS = 200.0       # P3
THRESHOLD_FLOOR = 0.0              # P4: calibrated threshold strictly positive

# PASS criteria.
COMMIT_DELTA_FLOOR = 0.05          # absolute floor on commit_rate(stable-perturbed)
COMMIT_DELTA_SD_MULTIPLIER = 2.0   # effect-size floor scaled on the SD of the DELTA


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _make_world_decoder(world_dim: int, world_obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, 64), nn.ReLU(), nn.Linear(64, world_obs_dim))


def _headroom(rate: float) -> float:
    """Engagement headroom: distance from the nearer of the 0.0 / 1.0 pins."""
    return float(min(rate, 1.0 - rate))


def _config_slice(env: CausalGridWorldV2, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        body_obs_dim=int(obs_dict["body_state"].shape[-1]),
        world_obs_dim=int(obs_dict["world_state"].shape[-1]),
        action_dim=int(env.action_dim),
        alpha_world=ALPHA_WORLD,
    )


def _build(seed: int) -> Tuple[REEAgent, CausalGridWorldV2, CausalGridWorldV2,
                               nn.Module, Dict[str, Any]]:
    # Each env carries its OWN np.random.default_rng(seed); omitting seed= makes
    # the run non-reproducible. The perturbed env is offset so the two conditions
    # do not share a layout stream.
    env_stable = CausalGridWorldV2(seed=seed, use_proxy_fields=True,
                                   hazard_harm=HAZARD_HARM, **ENV_STABLE)
    env_perturbed = CausalGridWorldV2(seed=seed + 100, use_proxy_fields=True,
                                      hazard_harm=HAZARD_HARM, **ENV_PERTURBED)
    _obs, obs_dict = env_stable.reset()
    slice_ = _config_slice(env_stable, obs_dict)
    cfg = REEConfig.from_dims(**slice_)
    agent = REEAgent(cfg)
    decoder = _make_world_decoder(cfg.latent.world_dim, int(slice_["world_obs_dim"]))
    return agent, env_stable, env_perturbed, decoder, slice_


def _run_phase(agent: REEAgent, env: CausalGridWorldV2, decoder: nn.Module,
               opt: Optional[optim.Optimizer], n_episodes: int, steps: int,
               train: bool, phase: str, rng: np.random.Generator,
               arm_id: str, seed: int,
               online_threshold: bool = False,
               online_seed_value: float = 0.0,
               ep_offset: int = 0, ep_total: int = 0) -> Dict[str, Any]:
    """One phase. Random actions throughout -- no learned policy, so there is no
    V_s monostrategy lock to dominate the threshold behaviour (the EXQ-454
    confound this design excludes STRUCTURALLY rather than by promise)."""
    if train:
        agent.train()
        decoder.train()
    else:
        agent.eval()
        decoder.eval()

    rv_traj: List[float] = []
    precision_traj: List[float] = []
    commits: List[bool] = []
    thresholds_used: List[float] = []
    ep_mean_rv: List[float] = []
    buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    ema = float(online_seed_value)
    step_counter = 0

    for ep in range(n_episodes):
        _obs, obs_dict = env.reset()
        agent.reset()
        zw_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None
        ep_rv: List[float] = []

        for _step in range(steps):
            obs_b = obs_dict["body_state"].unsqueeze(0)
            obs_w = obs_dict["world_state"].unsqueeze(0)
            latent = agent.sense(obs_b, obs_w)
            agent.clock.advance()
            zw_cur = latent.z_world.detach()
            zs_cur = latent.z_self.detach()

            act_idx = int(rng.integers(0, env.action_dim))
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, act_idx] = 1.0
            agent._last_action = action

            # V3-EXQ-396a: update_running_variance() has NO CALLER anywhere in
            # ree_core, so the driver MUST drive the EMA or _running_variance stays
            # pinned at precision_init and the commit gate never fires at all
            # (measured commit rate 0.000 in that state).
            if zw_prev is not None and a_prev is not None:
                with torch.no_grad():
                    pred = agent.e2.world_forward(zw_prev, a_prev)
                    agent.e3.update_running_variance(zw_cur - pred.detach())

            rv = float(agent.e3._running_variance)
            rv_traj.append(rv)
            ep_rv.append(rv)
            precision_traj.append(float(agent.e3.current_precision))

            # A2: slow trailing EMA of rv drives the threshold every tick.
            if online_threshold:
                ema = (1.0 - A2_EMA_ALPHA) * ema + A2_EMA_ALPHA * rv
                agent.e3.config.commitment_threshold = CALIBRATION_FACTOR * ema

            if step_counter % E3_DECISION_INTERVAL == 0:
                with torch.no_grad():
                    try:
                        cands = agent.e2.generate_candidates_random(
                            initial_z_self=zs_cur, initial_z_world=zw_cur,
                            num_candidates=NUM_CANDIDATES,
                            horizon=CANDIDATE_HORIZON,
                            compute_action_objects=True)
                        # DIRECT select(): one genuine selection per call, so the
                        # E3 cadence / commitment latch inside agent.select_action
                        # is not in the path and no diagnostic can be re-read from
                        # a stale tick (the V3-EXQ-785 ~9x defect).
                        res = agent.e3.select(cands, temperature=SELECT_TEMPERATURE)
                        commits.append(bool(res.committed))
                        thresholds_used.append(float(agent.e3.commit_threshold))
                    except Exception:
                        pass

            if train and opt is not None:
                if zw_prev is not None and a_prev is not None:
                    buf.append((zw_prev, a_prev, zw_cur))
                    if len(buf) > E2W_BUF_MAX:
                        buf = buf[-E2W_BUF_MAX:]
                e1_loss = agent.compute_prediction_loss()
                e2_self_loss = agent.compute_e2_loss()
                z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
                recon_loss = F.mse_loss(decoder(z_w), obs_w)
                e2w_loss = torch.zeros((), device=agent.device)
                if len(buf) >= 16:
                    k = min(E2W_BATCH, len(buf))
                    idxs = torch.randperm(len(buf))[:k].tolist()
                    zw_t = torch.cat([buf[i][0] for i in idxs], dim=0)
                    a_t = torch.cat([buf[i][1] for i in idxs], dim=0)
                    zw_t1 = torch.cat([buf[i][2] for i in idxs], dim=0)
                    e2w_loss = F.mse_loss(agent.e2.world_forward(zw_t, a_t), zw_t1)
                total = e1_loss + e2_self_loss + recon_loss + e2w_loss
                if total.requires_grad:
                    opt.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    opt.step()

            zw_prev = zw_cur
            a_prev = action.detach()
            step_counter += 1
            _obs, _r, done, _info, obs_dict = env.step(act_idx)
            if done:
                _obs, obs_dict = env.reset()

        if ep_rv:
            ep_mean_rv.append(float(np.mean(ep_rv)))
        # The episode counter is CUMULATIVE across all four phases of the cell, and
        # the denominator is the cell's TOTAL episode count. The runner overwrites
        # episodes_per_run with whatever denominator it last parsed from an
        # `ep N/M` line, so per-phase denominators (250 / 30 / 40 / 40) would make
        # the progress bar and ETA meaningless. One denominator per cell, matching
        # the queue entry's episodes_per_run.
        if (ep + 1) % 25 == 0 or ep == n_episodes - 1:
            print(f"  [train] arc016 {arm_id} seed={seed} phase={phase} "
                  f"ep {ep_offset + ep + 1}/{ep_total} "
                  f"rv={rv_traj[-1] if rv_traj else 0.0:.8f} "
                  f"commits={sum(commits)}/{len(commits)}", flush=True)

    return {
        "phase": phase,
        "mean_rv": float(np.mean(rv_traj)) if rv_traj else 0.0,
        "mean_precision": float(np.mean(precision_traj)) if precision_traj else 0.0,
        "commit_rate": float(np.mean(commits)) if commits else 0.0,
        "n_commit_decisions": float(len(commits)),
        "ep_mean_rv": ep_mean_rv,
        "threshold_used_mean": (
            float(np.mean(thresholds_used)) if thresholds_used else 0.0),
        "final_ema": ema,
    }


def _collect_cell(arm_id: str, seed: int, n_train: int, n_cal: int, n_eval: int,
                  steps: int, rng: np.random.Generator) -> Dict[str, Any]:
    agent, env_stable, env_perturbed, decoder, _slice = _build(seed)
    print(f"Seed {seed} Condition {arm_id}", flush=True)
    cfg_precision_init = float(agent.e3.config.precision_init)
    opt = optim.Adam(list(agent.parameters()) + list(decoder.parameters()), lr=LR)

    # ---- P0: TRAIN on the stable reference condition ---- #
    ep_total = n_train + n_cal + 2 * n_eval
    train_out = _run_phase(agent, env_stable, decoder, opt, n_train, steps,
                           train=True, phase="train", rng=rng,
                           arm_id=arm_id, seed=seed,
                           ep_offset=0, ep_total=ep_total)
    train_baseline = (float(np.mean(train_out["ep_mean_rv"][-BASELINE_WINDOW:]))
                      if train_out["ep_mean_rv"] else 0.0)

    # ---- Eval-time calibration window (STABLE), same reset protocol as scoring.
    # The reset is what makes this comparable: every scored phase starts from
    # precision_init and lets the EMA settle, so the calibration window carries the
    # SAME reset transient the scored phases do -- the 396 threshold was derived
    # from training, which carries no such transient at all.
    agent.e3._running_variance = cfg_precision_init
    cal_out = _run_phase(agent, env_stable, decoder, None, n_cal, steps,
                         train=False, phase="calibrate", rng=rng,
                         arm_id=arm_id, seed=seed,
                         ep_offset=n_train, ep_total=ep_total)
    eval_baseline = (float(np.mean(cal_out["ep_mean_rv"][-CAL_TAIL_EPISODES:]))
                     if cal_out["ep_mean_rv"] else 0.0)

    # ---- Threshold assignment: the ONLY manipulated factor ---- #
    online = (arm_id == ARM_ONLINE)
    if arm_id == ARM_TRAIN:
        threshold = CALIBRATION_FACTOR * train_baseline
    else:
        threshold = CALIBRATION_FACTOR * eval_baseline
    agent.e3.config.commitment_threshold = threshold

    # ---- Scored evals ---- #
    agent.e3._running_variance = cfg_precision_init
    stable_out = _run_phase(agent, env_stable, decoder, None, n_eval, steps,
                            train=False, phase="eval_stable", rng=rng,
                            arm_id=arm_id, seed=seed,
                            online_threshold=online, online_seed_value=eval_baseline,
                            ep_offset=n_train + n_cal, ep_total=ep_total)
    agent.e3.config.commitment_threshold = threshold
    agent.e3._running_variance = cfg_precision_init
    perturbed_out = _run_phase(agent, env_perturbed, decoder, None, n_eval, steps,
                               train=False, phase="eval_perturbed", rng=rng,
                               arm_id=arm_id, seed=seed,
                               online_threshold=online,
                               online_seed_value=(stable_out["final_ema"] if online
                                                  else eval_baseline),
                               ep_offset=n_train + n_cal + n_eval,
                               ep_total=ep_total)

    cr_s = stable_out["commit_rate"]
    cr_p = perturbed_out["commit_rate"]
    rv_s = stable_out["mean_rv"]
    rv_p = perturbed_out["mean_rv"]

    # A2 absorption: how much of the rv shift the threshold followed. ~1.0 means the
    # threshold tracked the variance and the DV is an identity (see DV-symmetry).
    thr_shift = abs(perturbed_out["threshold_used_mean"]
                    - stable_out["threshold_used_mean"])
    rv_shift = abs(rv_p - rv_s)
    tracking_ratio = float(thr_shift / rv_shift) if rv_shift > 1e-12 else 0.0

    return {
        "arm_id": arm_id,
        "seed": seed,
        # Structurally 0: e3.select() is called directly, so no call can re-read a
        # previous selection's latched diagnostics.
        "n_latched_ticks": 0,
        "train_baseline_variance": train_baseline,
        "eval_baseline_variance": eval_baseline,
        "threshold_ratio_eval_over_train": (
            float(eval_baseline / train_baseline) if train_baseline > 0 else 0.0),
        "calibrated_threshold": threshold,
        "commit_rate_stable": cr_s,
        "commit_rate_perturbed": cr_p,
        "commit_delta_stable_minus_perturbed": cr_s - cr_p,
        "engagement_headroom_stable": _headroom(cr_s),
        "engagement_headroom_perturbed": _headroom(cr_p),
        "engagement_headroom_worst": min(_headroom(cr_s), _headroom(cr_p)),
        "engagement_headroom_worst_condition": (
            "eval_stable" if _headroom(cr_s) <= _headroom(cr_p) else "eval_perturbed"),
        "mean_rv_stable": rv_s,
        "mean_rv_perturbed": rv_p,
        "rv_diff_perturbed_minus_stable": rv_p - rv_s,
        "rv_diff_relative": float((rv_p - rv_s) / rv_s) if rv_s > 1e-12 else 0.0,
        "mean_precision_stable": stable_out["mean_precision"],
        "mean_precision_perturbed": perturbed_out["mean_precision"],
        "n_commit_decisions_stable": stable_out["n_commit_decisions"],
        "n_commit_decisions_perturbed": perturbed_out["n_commit_decisions"],
        "n_commit_decisions_min": min(stable_out["n_commit_decisions"],
                                      perturbed_out["n_commit_decisions"]),
        "threshold_used_mean_stable": stable_out["threshold_used_mean"],
        "threshold_used_mean_perturbed": perturbed_out["threshold_used_mean"],
        "a2_threshold_tracking_ratio": tracking_ratio,
        "phase_train": {k: v for k, v in train_out.items() if k != "ep_mean_rv"},
        "phase_calibrate": {k: v for k, v in cal_out.items() if k != "ep_mean_rv"},
        "phase_eval_stable": {k: v for k, v in stable_out.items()
                              if k != "ep_mean_rv"},
        "phase_eval_perturbed": {k: v for k, v in perturbed_out.items()
                                 if k != "ep_mean_rv"},
        "per_episode_rv_train_tail": train_out["ep_mean_rv"][-BASELINE_WINDOW:],
        "per_episode_rv_calibrate": cal_out["ep_mean_rv"],
        "per_episode_rv_eval_stable": stable_out["ep_mean_rv"],
        "per_episode_rv_eval_perturbed": perturbed_out["ep_mean_rv"],
    }


# ------------------------------------------------------------------ #
# Gate                                                               #
# ------------------------------------------------------------------ #
def _specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="precision_manipulation_took",
            description=(
                "RELATIVE variance lift: (mean rv(perturbed) - mean rv(stable)) / "
                "mean rv(stable), worst cell. If the perturbation does not raise "
                "prediction-error variance, nothing downstream can track it and the "
                "run measures the environment rather than the "
                "precision-to-commitment circuit. Relative rather than absolute "
                "because rv's operating scale is itself what this experiment "
                "recalibrates against, so no absolute floor is meaningful across "
                "arms; EXQ-018b's C1 was relative for the same reason"),
            control="EXQ-018b extreme env contrast (2 hazards / static vs 25 / "
                    "drift every step at p=1.0), which measured a ~38% lift",
            threshold=VAR_DIFF_REL_FLOOR),
        PreconditionSpec(
            name="threshold_engagement_headroom",
            description=(
                "min(commit_rate, 1 - commit_rate), taken over the WORST of the "
                "arm's two conditions (not just the reference one -- the "
                "V3-EXQ-779b/777 saturation-guard defect). THE THRESHOLD MUST "
                "ACTUALLY ENGAGE: a rate pinned at 0.0 or 1.0 makes all arms "
                "identical and is exactly what voided EXQ-396a/b. Declared as a "
                "single-bound FLOOR on a worst-cell statistic so the indexer's "
                "recompute is exact, rather than a half-declared two-sided band"),
            control="eval-derived calibration on the stable reference condition",
            threshold=ENGAGEMENT_HEADROOM_FLOOR,
            applies_to=lambda ctx: ctx["arm_id"] != ARM_TRAIN,
            applies_note=(
                "not meaningful for A0_TRAIN_BASELINE: it is the pre-registered "
                "NEGATIVE REFERENCE whose designated role is to REPRODUCE the 396 "
                "non-engagement, so asserting engagement there would make the arm "
                "structurally un-passable and collapse the three-arm design. A0's "
                "headroom is still measured and emitted as a NON-GATING diagnostic "
                "(diagnostics.engagement_headroom_per_arm), so an unexpected "
                "engagement is visible rather than hidden by the scoping")),
        PreconditionSpec(
            name="commit_decision_count",
            description="genuine E3 commit decisions banked in the leaner condition",
            control="one e3.select() every E3_DECISION_INTERVAL steps",
            threshold=MIN_COMMIT_DECISIONS),
        PreconditionSpec(
            name="calibrated_threshold_positive",
            description=(
                "the calibrated commit threshold is finite and strictly positive "
                "(EXQ-018b's C0 -- a zero baseline means calibration never ran)"),
            control="CALIBRATION_FACTOR x a measured baseline variance",
            threshold=THRESHOLD_FLOOR),
    ]


def _measured(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        # WORST cell throughout, never a mean: `met` is a claim about every cell,
        # and an in-band mean can mask a single out-of-band cell (the V3-EXQ-779b
        # mean-vs-quantifier defect).
        "precision_manipulation_took": float(np.min(
            [r["rv_diff_relative"] for r in rows])),
        "threshold_engagement_headroom": float(np.min(
            [r["engagement_headroom_worst"] for r in rows])),
        "commit_decision_count": float(np.min(
            [r["n_commit_decisions_min"] for r in rows])),
        "calibrated_threshold_positive": float(np.min(
            [r["calibrated_threshold"] for r in rows])),
    }


def _headroom_diagnostic(by_arm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """NON-GATING per-arm engagement headroom, worst cell + the offending seed.

    Emitted on PASS runs too: a diagnostic that appears only when something already
    looks wrong cannot establish that anything was ever right.
    """
    out: Dict[str, Any] = {"floor_reference": ENGAGEMENT_HEADROOM_FLOOR, "arms": []}
    saturating: List[str] = []
    for arm_id, rows in by_arm.items():
        worst = min(rows, key=lambda r: r["engagement_headroom_worst"])
        entry = {
            "arm": arm_id,
            "worst_headroom": worst["engagement_headroom_worst"],
            "offending_seed": worst["seed"],
            "offending_condition": worst["engagement_headroom_worst_condition"],
            "commit_rate_stable": worst["commit_rate_stable"],
            "commit_rate_perturbed": worst["commit_rate_perturbed"],
            "gating": arm_id != ARM_TRAIN,
        }
        out["arms"].append(entry)
        if worst["engagement_headroom_worst"] < ENGAGEMENT_HEADROOM_FLOOR:
            saturating.append(arm_id)
    out["saturating_arms"] = saturating
    return out


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_train = DRY_TRAIN_EPISODES if dry_run else TRAIN_EPISODES
    n_cal = DRY_CAL_EPISODES if dry_run else CAL_EPISODES
    n_eval = DRY_EVAL_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_STEPS if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    specs = _specs()
    arm_ctxs = [{"arm_id": a} for a in ARMS]
    assert_no_structurally_unsatisfiable_gate(specs, arm_ctxs)

    arm_results: List[Dict[str, Any]] = []
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}

    for arm_id in ARMS:
        for seed in seeds:
            probe_slice = dict(_build(seed)[4], arm_id=arm_id,
                               train_episodes=n_train, cal_episodes=n_cal,
                               eval_episodes=n_eval, steps=steps,
                               calibration_factor=CALIBRATION_FACTOR,
                               e3_decision_interval=E3_DECISION_INTERVAL,
                               env_stable=ENV_STABLE, env_perturbed=ENV_PERTURBED)
            with arm_cell(
                seed,
                config_slice=probe_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                # MINT-AS-YOU-GO: cross-driver reusable by any successor.
                include_driver_script_in_hash=False,
            ) as cell:
                rng = np.random.default_rng(30_000 + 131 * seed)
                row = _collect_cell(arm_id, seed, n_train, n_cal, n_eval, steps, rng)
                cell.stamp(row)
            by_arm[arm_id].append(row)
            arm_results.append(row)
            print(f"verdict: "
                  f"{'PASS' if row['n_commit_decisions_min'] > 0 else 'FAIL'}",
                  flush=True)

    arm_gates = [
        evaluate_arm_gate(arm_id, {"arm_id": arm_id}, specs, _measured(by_arm[arm_id]))
        for arm_id in ARMS
    ]
    agg = aggregate_arm_gates(arm_gates)
    green = set(agg["green_arms"])

    # ---------------- criteria ---------------- #
    def _delta(arm: str) -> float:
        return float(np.mean(
            [r["commit_delta_stable_minus_perturbed"] for r in by_arm[arm]]))

    def _margin(arm: str) -> float:
        ds = [r["commit_delta_stable_minus_perturbed"] for r in by_arm[arm]]
        sd = float(np.std(ds)) if len(ds) >= 2 else 0.0
        return max(COMMIT_DELTA_FLOOR, COMMIT_DELTA_SD_MULTIPLIER * sd)

    d_eval, m_eval = _delta(ARM_EVAL), _margin(ARM_EVAL)
    d_train, m_train = _delta(ARM_TRAIN), _margin(ARM_TRAIN)
    d_online, m_online = _delta(ARM_ONLINE), _margin(ARM_ONLINE)

    c1_scorable = ARM_EVAL in green
    c1_pass = bool(c1_scorable and d_eval > m_eval)

    train_headroom_worst = float(np.min(
        [r["engagement_headroom_worst"] for r in by_arm[ARM_TRAIN]]))
    c2_scorable = ARM_TRAIN in green
    c2_pass = bool(c2_scorable
                   and train_headroom_worst < ENGAGEMENT_HEADROOM_FLOOR
                   and d_train <= m_train)

    tracking = float(np.max(
        [r["a2_threshold_tracking_ratio"] for r in by_arm[ARM_ONLINE]]))
    c3_scorable = bool(ARM_ONLINE in green and tracking < A2_TRACKING_CEILING)
    c3_pass = bool(c3_scorable and d_online > m_online)

    if not agg["any_green"] or not (c1_scorable or c2_scorable):
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = (
            "ARC-016 recalibration gate RED: " + agg["degeneracy_reason"]
            + ". A threshold that never engages, or a perturbation that did not "
              "raise variance, makes every arm identical -- exactly the state that "
              "voided EXQ-396a/b. This run is NOT scored and is NOT evidence "
              "against ARC-016.")
    else:
        non_degenerate, degeneracy_reason = True, ""
        if c1_pass and c2_pass:
            outcome, evidence_direction = "PASS", "supports"
            label = "396_series_was_calibration_artefact"
        elif c1_pass and not c2_pass:
            outcome, evidence_direction = "FAIL", "mixed"
            label = "tracking_holds_but_396_config_not_reproduced"
        elif c2_pass and not c1_pass:
            outcome, evidence_direction = "FAIL", "weakens"
            label = "commit_rate_does_not_track_precision"
        else:
            outcome, evidence_direction = "FAIL", "non_contributory"
            label = "calibration_contrast_uninformative"
        if not c1_scorable:
            outcome, evidence_direction = "FAIL", "non_contributory"
            label = "eval_derived_arm_red_tracking_unscorable"

    thr_ratio = float(np.mean(
        [r["threshold_ratio_eval_over_train"] for r in arm_results]))

    criteria = [
        {"name": "C1_commit_rate_tracks_precision_under_eval_derived_threshold",
         "load_bearing": True, "passed": c1_pass, "scorable": c1_scorable,
         "measured": d_eval, "threshold": m_eval,
         "commit_rate_stable": float(np.mean(
             [r["commit_rate_stable"] for r in by_arm[ARM_EVAL]])),
         "commit_rate_perturbed": float(np.mean(
             [r["commit_rate_perturbed"] for r in by_arm[ARM_EVAL]])),
         "null_note": (
             "a null here, with P2 GREEN (the threshold demonstrably engages), is "
             "the FIRST genuine evidence against the precision-to-commitment "
             "circuit: the 15 prior FAILs would need re-reading as right for the "
             "wrong reasons and ARC-016 demotes from provisional. It says nothing "
             "about ARC-029 (the behavioural consequence layer), which is not "
             "tested here")},
        {"name": "C2_train_baseline_arm_reproduces_396_non_engagement",
         "load_bearing": True, "passed": c2_pass, "scorable": c2_scorable,
         "measured_headroom": train_headroom_worst,
         "threshold_headroom": ENGAGEMENT_HEADROOM_FLOOR,
         "measured_delta": d_train, "threshold_delta": m_train,
         "null_note": (
             "without this the run cannot say the 396 result was a CALIBRATION "
             "ARTEFACT -- only that some configuration works. If the "
             "train-baseline threshold DOES engage here, the 396 diagnosis does "
             "not reproduce on this substrate and the whole framing needs revisiting")},
        {"name": "C3_online_adapted_threshold_also_tracks",
         "load_bearing": False, "passed": c3_pass, "scorable": c3_scorable,
         "measured": d_online, "threshold": m_online,
         "a2_threshold_tracking_ratio": tracking,
         "a2_tracking_ceiling": A2_TRACKING_CEILING,
         "null_note": (
             "NOT load-bearing. An online baseline that ABSORBS the perturbation "
             "(flat commit_rate) is an informative result about adaptation "
             "TIMESCALE, NOT evidence against ARC-016. Marked non-degenerate:false "
             "when a2_threshold_tracking_ratio exceeds the ceiling, because a "
             "threshold that followed the variance makes this DV an arithmetic "
             "identity rather than a measurement")},
    ]
    criteria_non_degenerate = arm_criteria_non_degenerate(
        {ARM_EVAL: ["C1_commit_rate_tracks_precision_under_eval_derived_threshold"],
         ARM_TRAIN: ["C2_train_baseline_arm_reproduces_396_non_engagement"],
         ARM_ONLINE: ["C3_online_adapted_threshold_also_tracks"]},
        agg,
        extra={"C3_online_adapted_threshold_also_tracks":
               bool(tracking < A2_TRACKING_CEILING)})

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "arms": ARMS,
        "train_episodes": n_train,
        "cal_episodes": n_cal,
        "eval_episodes_per_condition": n_eval,
        "steps_per_episode": steps,
        "baseline_window": BASELINE_WINDOW,
        "cal_tail_episodes": CAL_TAIL_EPISODES,
        "calibration_factor": CALIBRATION_FACTOR,
        "e3_decision_interval": E3_DECISION_INTERVAL,
        "num_candidates": NUM_CANDIDATES,
        "candidate_horizon": CANDIDATE_HORIZON,
        "alpha_world": ALPHA_WORLD,
        "select_temperature": SELECT_TEMPERATURE,
        "a2_ema_alpha": A2_EMA_ALPHA,
        "lr": LR,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True,
                "hazard_harm": HAZARD_HARM,
                "stable": ENV_STABLE, "perturbed": ENV_PERTURBED},
        "thresholds": {
            "VAR_DIFF_REL_FLOOR": VAR_DIFF_REL_FLOOR,
            "ENGAGEMENT_HEADROOM_FLOOR": ENGAGEMENT_HEADROOM_FLOOR,
            "MIN_COMMIT_DECISIONS": MIN_COMMIT_DECISIONS,
            "THRESHOLD_FLOOR": THRESHOLD_FLOOR,
            "COMMIT_DELTA_FLOOR": COMMIT_DELTA_FLOOR,
            "COMMIT_DELTA_SD_MULTIPLIER": COMMIT_DELTA_SD_MULTIPLIER,
            "A2_TRACKING_CEILING": A2_TRACKING_CEILING,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "config": full_config,
        "seeds": seeds,
        "arm_results": arm_results,
        "per_arm_gate": agg["per_arm_gate"],
        "diagnostics": {
            # Emitted on PASS runs too, by design.
            "engagement_headroom_per_arm": _headroom_diagnostic(by_arm),
        },
        "metrics": {
            "commit_delta_eval_derived": d_eval,
            "commit_delta_train_baseline": d_train,
            "commit_delta_online_adapted": d_online,
            "commit_delta_margin_eval_derived": m_eval,
            "train_baseline_headroom_worst": train_headroom_worst,
            "a2_threshold_tracking_ratio_max": tracking,
            "threshold_ratio_eval_over_train_mean": thr_ratio,
            "mean_rv_stable_eval_arm": float(np.mean(
                [r["mean_rv_stable"] for r in by_arm[ARM_EVAL]])),
            "mean_rv_perturbed_eval_arm": float(np.mean(
                [r["mean_rv_perturbed"] for r in by_arm[ARM_EVAL]])),
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
            "not_a_discovery_experiment_note": (
                "ARC-016's core circuit (E3-derived variance -> relative commit "
                "threshold -> BetaGate -> action_selection) is ALREADY VALIDATED by "
                "V3-EXQ-018b (PASS 5/5) and V3-EXQ-060 (PASS 4/5). This run closes "
                "the single open item named in the claim's own notes: 'pending "
                "recalibration with eval-derived threshold'. A PASS confirms the "
                "396-series death was a CALIBRATION ARTEFACT and leaves the "
                "existing validation standing unqualified; it PROMOTES NOTHING."),
            "scope_guard_note": (
                "STRUCTURAL ONLY. Post-split (2026-03-22) ARC-016 covers just the "
                "mechanistic circuit; the behavioural consequence layer is ARC-029 "
                "and is NOT tagged or tested. No behavioural harm DV is attached, "
                "because that would re-import the V_s MONOSTRATEGY-LOCK confound "
                "that voided EXQ-454. The exclusion is STRUCTURAL, not a promise: "
                "the driver takes RANDOM actions throughout, so there is no learned "
                "policy that could lock into a single strategy, and no harm or "
                "benefit quantity is a DV anywhere in the run."),
            "engagement_is_a_precondition_note": (
                "'The threshold must actually engage' is a PRECONDITION (P2), never "
                "a criterion. A threshold that never fires makes every arm "
                "identical -- exactly what voided EXQ-396a/b -- and must self-route "
                "substrate_not_ready_requeue rather than deliver a verdict on "
                "ARC-016. It is SCOPED OUT of A0_TRAIN_BASELINE, whose designated "
                "role is to reproduce that non-engagement (disposition (a): not "
                "meaningful for the regime, arm stays scorable); A0's headroom is "
                "still measured and emitted non-gating."),
            "dv_symmetry_note": (
                "DV = commit_rate = fraction of selections with running_variance < "
                "commit_threshold. Symmetry group: any transform preserving the "
                "SIGN of (threshold - rv) per tick -- in particular any monotone "
                "transform applied JOINTLY to rv and threshold, and any threshold "
                "move that stays on the same side of the rv distribution's mass. "
                "A1_EVAL_DERIVED is NOT invariant: the train- and eval-derived "
                "thresholds sit on OPPOSITE sides of the eval rv mass (~15-30x "
                "apart), so the sign flips for most ticks; the realised ratio is "
                "emitted as threshold_ratio_eval_over_train. A0_TRAIN_BASELINE is a "
                "declared REFERENCE arm with an expected-pinned DV. "
                "A2_ONLINE_ADAPTED carries a REAL invariance hazard -- a fast-"
                "adapting baseline makes rv < 2 x EMA(rv) true by arithmetic -- so "
                "its EMA is deliberately slow and the realised absorption is "
                "measured as a2_threshold_tracking_ratio; above A2_TRACKING_CEILING "
                "C3 is declared degenerate and routes nothing."),
            "threshold_held_fixed_note": (
                "The threshold is derived ONCE from the STABLE reference condition "
                "and held FIXED across the manipulated precision levels (A0/A1). "
                "Deriving it per-condition would ABSORB the manipulation and "
                "measure nothing -- the same reason EXQ-018b calibrated on stable "
                "and then evaluated stable-vs-perturbed."),
            "reset_protocol_note": (
                "Every scored phase, AND the eval-time calibration window, starts "
                "from precision_init and lets the EMA settle. That matching is the "
                "point: the 396 threshold came from TRAINING, which carries no such "
                "reset transient at all, so it was calibrated against a variance "
                "regime the eval phases never occupy."),
            "driver_constraint_note": (
                "V3-EXQ-396a: update_running_variance() has NO CALLER anywhere in "
                "ree_core, so the driver must drive the EMA itself or "
                "_running_variance stays pinned at precision_init and the commit "
                "gate never fires (measured commit rate 0.000 in that state). Do "
                "not 'simplify' this away."),
            "governance_flag_not_acted_on": (
                "ARC-016's exp_conf 0.53 is an AGGREGATION ARTEFACT: 15 FAILs, "
                "nearly all already stamped non_contributory or substrate-version-"
                "stale (heterogeneity_note), still score it as half-refuted. "
                "Re-scoring needs a governance eye rather than an experiment, "
                "because stripping non-contributory FAILs can expose that the "
                "remaining supports are thinner than the headline number suggested. "
                "This run neither promotes nor re-scores."),
        },
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
        "custom_information": {
            "proposal_id": "EXP-0397",
            "gov_reuse_1_check": (
                "Decisive readout: commit_rate under an EVAL-DERIVED commit "
                "threshold across a manipulated precision level. Checked the "
                "ARC-016 corpus: V3-EXQ-018b, 018, 396, 396a (x3) and 396b all "
                "derive the threshold from TRAINING variance BY CONSTRUCTION -- "
                "which is precisely the configuration reproduced here as the "
                "negative reference -- and none records an eval-time calibration "
                "window, so the readout cannot be recovered or derived post hoc. "
                "Not recoverable -> must run."),
            "supersedes_context": (
                "Does NOT set `supersedes`: V3-EXQ-396a/396b are already "
                "reclassified non_contributory on the merits, so there is no live "
                "scoring to displace. This is a NEW EXQ number rather than a 396 "
                "letter because the manipulated factor (baseline-variance SOURCE) "
                "is a different scientific question from 396's hazard_harm sweep."),
        },
    }

    stamp_recording_core(manifest, config=full_config, seeds=seeds,
                         script_path=Path(__file__), started_at=t0)
    return manifest


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("V3-EXQ-805: ARC-016 eval-derived commit-threshold recalibration",
          flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    t_start = time.perf_counter()
    manifest = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run,
        config=manifest.get("config"), seeds=manifest.get("seeds"),
        script_path=Path(__file__), started_at=t_start,
    )

    m = manifest["metrics"]
    print(f"  outcome={manifest['outcome']} "
          f"direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    print(f"  C1 delta_eval={m['commit_delta_eval_derived']:.4f} "
          f"margin={m['commit_delta_margin_eval_derived']:.4f}", flush=True)
    print(f"  C2 train_headroom_worst={m['train_baseline_headroom_worst']:.4f} "
          f"delta_train={m['commit_delta_train_baseline']:.4f}", flush=True)
    print(f"  C3 delta_online={m['commit_delta_online_adapted']:.4f} "
          f"tracking={m['a2_threshold_tracking_ratio_max']:.4f}", flush=True)
    print(f"  eval/train baseline ratio="
          f"{m['threshold_ratio_eval_over_train_mean']:.3f} "
          f"rv_stable={m['mean_rv_stable_eval_arm']:.8f} "
          f"rv_perturbed={m['mean_rv_perturbed_eval_arm']:.8f}", flush=True)
    _pag = manifest["per_arm_gate"]
    for g in list(_pag.get("green", [])) + list(_pag.get("red", [])):
        print(f"  [{g.get('arm')}] gate="
              f"{'GREEN' if g.get('gate_green') else 'RED'} "
              f"failed={g.get('failed_preconditions')}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
