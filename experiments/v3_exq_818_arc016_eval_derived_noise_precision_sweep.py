"""V3-EXQ-818: ARC-016 -- the residual validation gate. Eval-derived commit
threshold + a GRADED-NOISE precision sweep that tests whether commit_rate tracks
E3-derived precision MONOTONICALLY during eval, once the threshold genuinely engages.

claim_ids = [ARC-016] ONLY. STRUCTURAL circuit test. NO behavioural harm DV.

=== WHAT THIS CLOSES (and what it does NOT) ===
ARC-016's core circuit -- E3-derived variance -> relative commit threshold ->
BetaGate -> action_selection -- is ALREADY VALIDATED at TRAIN-TIME / structurally:
    V3-EXQ-018b  PASS 5/5 (2026-03-20). commit_threshold = 2 x
                 training_baseline_variance; stable commit_rate 0.90 / precision 718,
                 perturbed 0.50 / 426 -- a 40% precision drop -> ~40-pt commit drop.
    V3-EXQ-060   PASS 4/5 (2026-03-21). committed_step_count 5980, hold_rate 0.936.
The 2026-07-25 /claim-synthesis rescore (claim_synthesis_arc-016_2026-07-25.md)
confirmed exp_conf 0.775, quadrant confirmed_established, ZERO genuine refutations --
but ALL 5 surviving supports are TRAIN-TIME / structural. The ONE open item is
EVAL-TIME engagement: does the commit threshold actually FIRE during eval and does
commit_rate TRACK a manipulated precision level? That is the single gate between
provisional and `shown`. This run is that gate. A PASS confirms the eval-time
circuit engages and tracks; it PROMOTES NOTHING on its own (governance owns that).

=== WHY THE PRIOR EVAL-TIME ATTEMPTS DIED, AND THE FIX ===
Every eval-time attempt is non_contributory:
  EXQ-396a(x3)/396b  threshold pinned to TRAINING variance (~3e-5) while eval
                     variance ran ~1e-3 (15-30x): the gate never fired in eval.
  EXQ-454/454a       V_s monostrategy lock dominated threshold-adaptation.
  EXQ-530/530c       already non_contributory.
  EXQ-805            got the eval-derived threshold RIGHT, but its precision
                     manipulation DID NOT TAKE: the env-difficulty lever
                     (num_hazards 2 vs 25, drift 0 vs 1) moved eval running-variance
                     only ~5.6% (rv_diff_relative 0.0558 << the 0.25 floor), far
                     below 018b's ~38%. A well-trained world_forward predicts the
                     "perturbed" env almost as well as the stable one, so
                     prediction-error variance barely separates. It correctly
                     self-routed substrate_not_ready_requeue on precision_manipulation
                     _took, and is NOT evidence against ARC-016.

FIX (user-selected 2026-07-25: "graded noise in E3 PE path"): stop relying on
env-difficulty to move the E3 running variance INDIRECTLY. Instead inject a
controlled, graded Gaussian noise of std sigma_L directly into the world_forward
prediction error that feeds e3.update_running_variance(), during eval only, and
sweep sigma across N_LEVELS pre-registered levels. update_running_variance computes
error_var = prediction_error.pow(2).mean() (e3_selector.py:602), so additive noise
n ~ N(0, sigma^2) raises per-tick error_var by ~sigma^2 in expectation -- the
E3 running variance therefore sweeps a MONOTONIC range that is E3-derived (still the
EMA of a genuine prediction error, now with a controlled surprise component) and,
crucially, CALIBRATED to bracket the threshold BY CONSTRUCTION. The manipulation is
guaranteed to take (805's failure) and the sweep enables the MONOTONIC-tracking test
what_would_answer literally asks for (a graded sweep, not a binary delta).

=== PRECISION MANIPULATION -- graded noise, per-seed-calibrated to bracket ===
sigma_L is NOT an absolute constant (that is exactly the 396/805 scale trap). After
a frozen, noise-free eval calibration window measures eval_baseline (the natural eval
running variance), the threshold is set = CALIBRATION_FACTOR x eval_baseline (2.0,
018b's value), and the sweep is defined by pre-registered TARGET RV MULTIPLES of
eval_baseline: RV_TARGET_MULTIPLES = [1.0, 1.5, 2.5, 4.0, 7.0]. For target multiple
k_L the injected std is sigma_L = sqrt(max(0, k_L - 1) * eval_baseline), so the
realised running variance settles near k_L x eval_baseline. With the threshold at
2 x eval_baseline the sweep straddles it: level 0-1 (rv ~1-1.5x < thr) commit, levels
2-4 (rv ~2.5-7x > thr) do not. commit_rate should sweep ~1 -> ~0 through the
transition -- IF the circuit tracks precision. The schedule is pre-registered; only
its scaling to physical sigma uses the measured eval_baseline (that scaling IS the
eval-derived calibration under test).

=== NON-DEGENERACY IS A PRECONDITION, NEVER A VERDICT (the 396a/b / 805 lesson) ===
A threshold that never fires, or a manipulation that does not move rv, makes all
levels identical and MUST self-route substrate_not_ready_requeue, not a verdict.
Preconditions (worst seed, single-bound FLOORs so the indexer recompute is exact):
  P1 MANIPULATION TOOK. Relative rv lift across the sweep,
     (rv[top level] - rv[bottom level]) / rv[bottom level], worst seed, above
     MANIP_TOOK_REL_FLOOR. By construction ~6x; asserted in case EMA/reset washes it.
  P2 COMMIT-RATE ENGAGES (THE non-degeneracy the residual gate mandates). The
     commit_rate RANGE across levels (max - min), worst seed, above COMMIT_RANGE_FLOOR
     -- i.e. commit_rate is NOT pinned at 0.0 or 1.0 across the whole sweep; the
     threshold demonstrably transitions the gate. This is the sweep-appropriate form
     of "commit_rate strictly between floor and ceiling": a degenerate sweep has
     range ~0.
  P3 COMMIT-DECISION COUNT floor per level (worst seed x level).
  P4 CALIBRATED THRESHOLD finite and strictly positive (018b's C0).

=== PRE-REGISTERED LOAD-BEARING CRITERION ===
  C1 MONOTONIC TRACKING (LOAD-BEARING): with the threshold GREEN (P2 met),
     commit_rate tracks the E3-derived precision monotonically. Operationalised as
     Spearman rho between the MEASURED mean running variance per level and commit_rate
     per level, per seed. Higher rv = lower precision -> fewer commits, so a tracking
     circuit gives a STRONG NEGATIVE rho. C1 passes iff mean rho <= -RHO_FLOOR AND
     every seed's rho <= -RHO_SEED_FLOOR (sign-consistent), with the commit_rate range
     already guaranteed non-trivial by P2.

INTERPRETATION GRID:
  gate GREEN + C1 pass -> PASS / supports
      "eval_time_threshold_engages_and_tracks_precision_monotonically" -- the eval-time
      circuit engages and tracks; the 396-series death was a calibration artefact and
      ARC-016's validation stands. Promotes nothing (governance owns status).
  gate GREEN + C1 FAIL -> FAIL / weakens
      "commit_rate_does_not_track_precision_under_engaged_eval_threshold" --
      DECISION-FLIPPING. With a threshold that demonstrably ENGAGES (P2 green),
      commit_rate still fails to track precision. This is the FIRST genuine evidence
      against the precision-to-commitment circuit; the 15 prior FAILs would need
      re-reading and ARC-016 loses its provisional standing. Says NOTHING about
      ARC-029 (behavioural consequence layer), which is not tested here.
  gate RED -> FAIL / non_contributory  "substrate_not_ready_requeue" (NEVER a verdict).

=== SCOPE GUARD -- STRUCTURAL ONLY (do NOT add a behavioural harm DV) ===
Post-split (2026-03-22) ARC-016 covers ONLY the structural circuit; the behavioural
consequence layer (committed vs uncommitted -> distinct harm outcomes) is ARC-029.
Attaching a harm DV would re-import the V_s MONOSTRATEGY-LOCK confound that voided
EXQ-454. The exclusion is STRUCTURAL, not a promise: the driver takes RANDOM actions
throughout, so there is no learned policy to lock into a strategy, and no harm/benefit
quantity is a DV anywhere in this run.

=== DV-SYMMETRY DECLARATION (mandatory) ===
DV = commit_rate = fraction of selections with running_variance < commit_threshold.
Symmetry group: any transform preserving the SIGN of (threshold - rv) per tick -- any
monotone transform applied JOINTLY to rv and threshold, or a threshold move that stays
on the same side of the rv mass. The manipulation (additive noise into the PE that
feeds rv) is NOT invariant under this group: it shifts rv ACROSS the FIXED threshold
(the threshold is derived once from the noise-free baseline and held constant over the
sweep), so the sign of (threshold - rv) flips as sigma rises -- that sign flip is the
measured effect, emitted as the per-level rv-vs-threshold crossing. There is no
broadcast-scalar / rank-invariance / permutation degeneracy: rv is a scalar the noise
genuinely moves, and commit_rate is a fraction, not an argmax.

=== E3 LATCH (V3-EXQ-785 pseudo-replication defect) -- structurally impossible ===
Commitment is read by calling agent.e3.select(...) DIRECTLY, one genuine independent
selection per call; the E3 cadence (heartbeat.e3_steps_per_tick) and the commitment
latch inside agent.select_action are not in the path, so no diagnostic can be re-read
from a stale tick. n_latched_ticks is 0 by construction and emitted as 0.

TRAINING PROTOCOL (phased): P0 substrate warmup (E1 + E2.self + world-decoder recon +
E2.world_forward one-step MSE) with grad, noise-free. Then FROZEN eval (agent.eval() /
no_grad) for the calibration window and every scored sweep level. NO downstream head is
trained on a latent, so the P1 freeze-and-detach phase is inapplicable (no head to
chase a moving target); noise is injected in eval ONLY, so the world model never learns
to expect it.

GOV-REUSE-1: decisive readout is commit_rate under an EVAL-DERIVED, held-fixed commit
threshold across a NOISE-swept E3 running-variance level. No recorded manifest carries
it: 018b/396a/b derive the threshold from TRAINING variance by construction; 805 used
an env-difficulty manipulation that did not take and records no per-level noise sweep.
Not recoverable -> must run.

Run:
  /opt/local/bin/python3 experiments/v3_exq_818_arc016_eval_derived_noise_precision_sweep.py --dry-run
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
from experiments._lib.arm_fingerprint import reset_all_rng  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_818_arc016_eval_derived_noise_precision_sweep"
QUEUE_ID = "V3-EXQ-818"
EXPERIMENT_PURPOSE = "evidence"
# ONLY the claim this implementation directly tests. ARC-029 (behavioural
# consequence layer, split off 2026-03-22) is deliberately NOT tagged -- no
# behavioural DV exists in this run.
CLAIM_IDS: List[str] = ["ARC-016"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3]
TRAIN_EPISODES = 250
CAL_EPISODES = 30           # eval-time, noise-free calibration window
EVAL_EPISODES_PER_LEVEL = 40
EVAL_SETTLE_EPISODES = 10   # discarded per level: rv resets to precision_init (0.5),
                            # far above eval_baseline (~1e-3), so the EMA settling
                            # transient sits above threshold and would depress the
                            # low-sigma commit_rate. Score only the settled regime.
STEPS_PER_EPISODE = 100
CAL_TAIL_EPISODES = 20      # calibration episodes averaged, after the reset transient

DRY_TRAIN_EPISODES = 6
DRY_CAL_EPISODES = 3
DRY_EVAL_EPISODES = 4
DRY_SETTLE_EPISODES = 1
DRY_STEPS = 25

# Target running-variance multiples of the (per-seed) eval_baseline. sigma_L is
# derived from these so the swept rv brackets the CALIBRATION_FACTOR x eval_baseline
# threshold by construction. Level 0 (multiple 1.0) is the natural noise-free eval.
RV_TARGET_MULTIPLES: List[float] = [1.0, 1.5, 2.5, 4.0, 7.0]
N_LEVELS = len(RV_TARGET_MULTIPLES)

CALIBRATION_FACTOR = 2.0    # 018b's value, unchanged
E3_DECISION_INTERVAL = 10   # steps between E3 selections (018b's value)
NUM_CANDIDATES = 16
CANDIDATE_HORIZON = 5
ALPHA_WORLD = 0.9           # the 0.3 default is a known SD-008 root cause
SELECT_TEMPERATURE = 1.0
LR = 1e-3
E2W_BUF_MAX = 500
E2W_BATCH = 32

# One static env family throughout: the manipulation is the injected noise, not the
# env. Static (drift_prob=0.0) lets E2.world_forward converge to a low, stable
# baseline, matching 018b (~0.001) so the noise sweep has a clean reference.
ENV = dict(size=12, num_hazards=4, num_resources=5,
           env_drift_interval=200, env_drift_prob=0.0)
HAZARD_HARM = 0.02

# Precondition floors (single-bound FLOORs on the WORST seed).
MANIP_TOOK_REL_FLOOR = 1.0   # P1: rv[top]/rv[bot] - 1 >= 1.0 (rv at least doubled)
COMMIT_RANGE_FLOOR = 0.30    # P2: max-min commit_rate across levels (threshold engages)
MIN_COMMIT_DECISIONS = 200.0 # P3
THRESHOLD_FLOOR = 0.0        # P4: calibrated threshold strictly positive

# C1 pre-registered tracking thresholds.
RHO_FLOOR = 0.80             # mean Spearman rho(rv, commit_rate) <= -0.80
RHO_SEED_FLOOR = 0.50        # every seed's rho <= -0.50 (sign-consistent)


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _make_world_decoder(world_dim: int, world_obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, 64), nn.ReLU(), nn.Linear(64, world_obs_dim))


def _config_slice(env: CausalGridWorldV2, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        body_obs_dim=int(obs_dict["body_state"].shape[-1]),
        world_obs_dim=int(obs_dict["world_state"].shape[-1]),
        action_dim=int(env.action_dim),
        alpha_world=ALPHA_WORLD,
    )


def _build(seed: int) -> Tuple[REEAgent, CausalGridWorldV2, nn.Module, Dict[str, Any]]:
    env = CausalGridWorldV2(seed=seed, use_proxy_fields=True,
                            hazard_harm=HAZARD_HARM, **ENV)
    _obs, obs_dict = env.reset()
    slice_ = _config_slice(env, obs_dict)
    cfg = REEConfig.from_dims(**slice_)
    agent = REEAgent(cfg)
    decoder = _make_world_decoder(cfg.latent.world_dim, int(slice_["world_obs_dim"]))
    return agent, env, decoder, slice_


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rho = Pearson correlation of ranks. Stdlib/numpy only (no scipy).
    Average ranks for ties. Returns 0.0 for a degenerate (zero-variance) input."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def _rank(v: List[float]) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        # average ties
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _run_phase(agent: REEAgent, env: CausalGridWorldV2, decoder: nn.Module,
               opt: Optional[optim.Optimizer], n_episodes: int, steps: int,
               train: bool, phase: str, rng: np.random.Generator,
               seed: int, noise_std: float,
               ep_offset: int, ep_total: int,
               warmup_episodes: int = 0) -> Dict[str, Any]:
    """One phase. Random actions throughout -- no learned policy, so there is no
    V_s monostrategy lock to dominate threshold behaviour (the EXQ-454 confound this
    design excludes STRUCTURALLY). noise_std > 0 injects graded Gaussian noise into
    the world_forward prediction error before update_running_variance -- the ONLY
    precision manipulation, applied in eval phases only. The first warmup_episodes
    drive the EMA (so it settles from precision_init) but are EXCLUDED from all scored
    aggregates -- mean_rv / commit_rate / precision reflect the settled regime only."""
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
    step_counter = 0

    for ep in range(n_episodes):
        _obs, obs_dict = env.reset()
        agent.reset()
        zw_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None
        ep_rv: List[float] = []
        scored = ep >= warmup_episodes  # settle prefix excluded from all aggregates

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

            # update_running_variance() has NO CALLER in ree_core, so the driver
            # MUST drive the EMA or _running_variance stays pinned at precision_init
            # and the commit gate never fires. noise_std > 0 adds a controlled
            # surprise term to the PE (the graded precision manipulation).
            if zw_prev is not None and a_prev is not None:
                with torch.no_grad():
                    pred = agent.e2.world_forward(zw_prev, a_prev)
                    err = zw_cur - pred.detach()
                    if noise_std > 0.0:
                        err = err + torch.randn_like(err) * noise_std
                    agent.e3.update_running_variance(err)

            rv = float(agent.e3._running_variance)
            if scored:
                rv_traj.append(rv)
                ep_rv.append(rv)
                precision_traj.append(float(agent.e3.current_precision))

            if step_counter % E3_DECISION_INTERVAL == 0:
                with torch.no_grad():
                    try:
                        cands = agent.e2.generate_candidates_random(
                            initial_z_self=zs_cur, initial_z_world=zw_cur,
                            num_candidates=NUM_CANDIDATES,
                            horizon=CANDIDATE_HORIZON,
                            compute_action_objects=True)
                        # DIRECT select(): one genuine selection per call; the E3
                        # cadence / commitment latch is not in the path (V3-EXQ-785
                        # ~9x pseudo-replication defect is structurally impossible).
                        # select() runs every eligible tick (settling included) so the
                        # dynamics are identical; only SCORED ticks feed the aggregate.
                        res = agent.e3.select(cands, temperature=SELECT_TEMPERATURE)
                        if scored:
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
        # Cumulative episode counter across all phases of the seed; denominator is
        # the seed's TOTAL episode count (matches queue episodes_per_run). Per-phase
        # denominators would make the runner's progress bar / ETA meaningless.
        if (ep + 1) % 25 == 0 or ep == n_episodes - 1:
            print(f"  [train] arc016 seed={seed} phase={phase} "
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
    }


def _collect_seed(seed: int, n_train: int, n_cal: int, n_eval: int, n_settle: int,
                  steps: int, rng: np.random.Generator) -> Dict[str, Any]:
    reset_all_rng(seed)  # torch+cuda+numpy+random reset for reproducibility
    agent, env, decoder, _slice = _build(seed)
    print(f"Seed {seed} Condition noise_sweep", flush=True)
    cfg_precision_init = float(agent.e3.config.precision_init)
    opt = optim.Adam(list(agent.parameters()) + list(decoder.parameters()), lr=LR)

    ep_total = n_train + n_cal + N_LEVELS * n_eval

    # ---- P0: TRAIN (grad, noise-free) ---- #
    _train_out = _run_phase(agent, env, decoder, opt, n_train, steps,
                            train=True, phase="train", rng=rng, seed=seed,
                            noise_std=0.0, ep_offset=0, ep_total=ep_total)

    # ---- Eval-time calibration window (frozen, noise-free): eval_baseline ---- #
    # Same reset protocol every scored level uses: start from precision_init, let the
    # EMA settle. The 396 threshold came from TRAINING, which carries no reset
    # transient -- calibrating on the eval regime is the whole point.
    agent.e3._running_variance = cfg_precision_init
    cal_out = _run_phase(agent, env, decoder, None, n_cal, steps,
                         train=False, phase="calibrate", rng=rng, seed=seed,
                         noise_std=0.0, ep_offset=n_train, ep_total=ep_total)
    eval_baseline = (float(np.mean(cal_out["ep_mean_rv"][-CAL_TAIL_EPISODES:]))
                     if cal_out["ep_mean_rv"] else 0.0)

    # ---- Eval-derived threshold: derived ONCE, held FIXED across the sweep ---- #
    threshold = CALIBRATION_FACTOR * eval_baseline
    agent.e3.config.commitment_threshold = threshold

    # ---- Graded-noise precision sweep (frozen), sigma_L calibrated to bracket ---- #
    levels: List[Dict[str, Any]] = []
    for li, k_L in enumerate(RV_TARGET_MULTIPLES):
        sigma_L = float(np.sqrt(max(0.0, k_L - 1.0) * eval_baseline))
        agent.e3.config.commitment_threshold = threshold  # re-assert (fixed)
        agent.e3._running_variance = cfg_precision_init
        out = _run_phase(agent, env, decoder, None, n_eval, steps,
                         train=False, phase=f"eval_L{li}", rng=rng, seed=seed,
                         noise_std=sigma_L,
                         ep_offset=n_train + n_cal + li * n_eval, ep_total=ep_total,
                         warmup_episodes=n_settle)
        levels.append({
            "level_index": li,
            "rv_target_multiple": k_L,
            "sigma_injected": sigma_L,
            "mean_rv": out["mean_rv"],
            "mean_precision": out["mean_precision"],
            "commit_rate": out["commit_rate"],
            "n_commit_decisions": out["n_commit_decisions"],
            "threshold_used_mean": out["threshold_used_mean"],
            "rv_over_threshold": (float(out["mean_rv"] / threshold)
                                  if threshold > 1e-12 else 0.0),
        })

    rv_by_level = [lv["mean_rv"] for lv in levels]
    cr_by_level = [lv["commit_rate"] for lv in levels]
    rv_bot, rv_top = rv_by_level[0], rv_by_level[-1]
    manip_rel_lift = float((rv_top - rv_bot) / rv_bot) if rv_bot > 1e-12 else 0.0
    commit_range = float(max(cr_by_level) - min(cr_by_level))
    # Tracking: higher rv (lower precision) -> fewer commits -> strong NEGATIVE rho.
    rho = _spearman(rv_by_level, cr_by_level)

    return {
        "seed": seed,
        "n_latched_ticks": 0,  # e3.select() called directly; no stale-tick re-read
        "eval_baseline_variance": eval_baseline,
        "calibrated_threshold": threshold,
        "levels": levels,
        "manip_rel_lift": manip_rel_lift,
        "commit_range": commit_range,
        "rho_rv_vs_commit": rho,
        "min_commit_decisions": float(min(lv["n_commit_decisions"] for lv in levels)),
        "per_level_mean_rv": rv_by_level,
        "per_level_commit_rate": cr_by_level,
        "per_episode_rv_calibrate": cal_out["ep_mean_rv"],
    }


def run_experiment(dry_run: bool):
    t0 = time.perf_counter()
    n_train = DRY_TRAIN_EPISODES if dry_run else TRAIN_EPISODES
    n_cal = DRY_CAL_EPISODES if dry_run else CAL_EPISODES
    n_eval = DRY_EVAL_EPISODES if dry_run else EVAL_EPISODES_PER_LEVEL
    n_settle = DRY_SETTLE_EPISODES if dry_run else EVAL_SETTLE_EPISODES
    steps = DRY_STEPS if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(30_000 + 131 * seed)
        row = _collect_seed(seed, n_train, n_cal, n_eval, n_settle, steps, rng)
        per_seed.append(row)
        print(f"verdict: {'PASS' if row['min_commit_decisions'] > 0 else 'FAIL'}",
              flush=True)

    # ---------------- preconditions (worst seed) ---------------- #
    manip_took_measured = float(np.min([r["manip_rel_lift"] for r in per_seed]))
    commit_range_measured = float(np.min([r["commit_range"] for r in per_seed]))
    min_decisions_measured = float(np.min([r["min_commit_decisions"] for r in per_seed]))
    threshold_measured = float(np.min([r["calibrated_threshold"] for r in per_seed]))

    p1 = manip_took_measured >= MANIP_TOOK_REL_FLOOR
    p2 = commit_range_measured >= COMMIT_RANGE_FLOOR
    p3 = min_decisions_measured >= MIN_COMMIT_DECISIONS
    p4 = threshold_measured > THRESHOLD_FLOOR
    gate_green = bool(p1 and p2 and p3 and p4)

    preconditions = [
        {"name": "precision_manipulation_took",
         "description": ("relative rv lift across the noise sweep, "
                         "(rv[top]-rv[bottom])/rv[bottom], worst seed. Additive PE "
                         "noise is calibrated to bracket the threshold, so a failure "
                         "here means the EMA/reset washed the manipulation out and the "
                         "run measures nothing"),
         "control": "sigma_L = sqrt((k_L-1) x eval_baseline); top multiple 7x",
         "measured": manip_took_measured, "threshold": MANIP_TOOK_REL_FLOOR,
         "direction": "lower", "met": bool(p1)},
        {"name": "commit_rate_engages",
         "description": ("commit_rate RANGE (max-min) across levels, worst seed -- the "
                         "residual gate's non-degeneracy mandate. A rate pinned at 0.0 "
                         "or 1.0 across the whole sweep means the threshold never "
                         "engaged (the 396a/b state) and routes substrate_not_ready, "
                         "NOT a verdict"),
         "control": "threshold = 2 x eval_baseline sits inside the swept rv range",
         "measured": commit_range_measured, "threshold": COMMIT_RANGE_FLOOR,
         "direction": "lower", "met": bool(p2)},
        {"name": "commit_decision_count",
         "description": "genuine E3 commit decisions banked per level, worst seed x level",
         "control": "one e3.select() every E3_DECISION_INTERVAL steps",
         "measured": min_decisions_measured, "threshold": MIN_COMMIT_DECISIONS,
         "direction": "lower", "met": bool(p3)},
        {"name": "calibrated_threshold_positive",
         "description": ("the eval-derived commit threshold is finite and strictly "
                         "positive (018b's C0 -- a zero baseline means calibration "
                         "never ran)"),
         "control": "CALIBRATION_FACTOR x measured eval_baseline",
         "measured": threshold_measured, "threshold": THRESHOLD_FLOOR,
         "direction": "lower", "comparator": ">", "met": bool(p4)},
    ]

    # ---------------- C1 tracking criterion ---------------- #
    rhos = [r["rho_rv_vs_commit"] for r in per_seed]
    mean_rho = float(np.mean(rhos))
    all_seeds_negative = bool(all(rh <= -RHO_SEED_FLOOR for rh in rhos))
    c1_pass = bool(gate_green and mean_rho <= -RHO_FLOOR and all_seeds_negative)

    if not gate_green:
        outcome, evidence_direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        unmet = [p["name"] for p in preconditions if not p["met"]]
        degeneracy_reason = (
            "ARC-016 residual gate RED: precondition(s) "
            f"{unmet} not met (manip_took={manip_took_measured:.3f}/"
            f"{MANIP_TOOK_REL_FLOOR}, commit_range={commit_range_measured:.3f}/"
            f"{COMMIT_RANGE_FLOOR}). A manipulation that did not move rv, or a "
            "threshold that never engaged, makes every level identical -- exactly the "
            "state that voided EXQ-396a/b. NOT scored, NOT evidence against ARC-016.")
    else:
        non_degenerate, degeneracy_reason = True, ""
        if c1_pass:
            outcome, evidence_direction = "PASS", "supports"
            label = "eval_time_threshold_engages_and_tracks_precision_monotonically"
        else:
            outcome, evidence_direction = "FAIL", "weakens"
            label = "commit_rate_does_not_track_precision_under_engaged_eval_threshold"

    criteria = [
        {"name": "C1_commit_rate_tracks_precision_monotonically",
         "load_bearing": True, "passed": c1_pass, "scorable": gate_green,
         "mean_rho_rv_vs_commit": mean_rho, "rho_floor": -RHO_FLOOR,
         "per_seed_rho": rhos, "all_seeds_negative": all_seeds_negative,
         "commit_range_worst_seed": commit_range_measured,
         "null_note": (
             "a null here, with the gate GREEN (the eval-derived threshold "
             "demonstrably engages -- commit_range above floor), is the FIRST genuine "
             "evidence against the precision-to-commitment circuit: commit_rate fails "
             "to track E3-derived precision even when the threshold fires. The 15 "
             "prior FAILs would need re-reading and ARC-016 loses provisional "
             "standing. Says nothing about ARC-029 (behavioural layer), not tested here")},
    ]
    criteria_non_degenerate = {
        "C1_commit_rate_tracks_precision_monotonically": bool(gate_green)}

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "train_episodes": n_train,
        "cal_episodes": n_cal,
        "eval_episodes_per_level": n_eval,
        "eval_settle_episodes": n_settle,
        "steps_per_episode": steps,
        "cal_tail_episodes": CAL_TAIL_EPISODES,
        "rv_target_multiples": RV_TARGET_MULTIPLES,
        "n_levels": N_LEVELS,
        "calibration_factor": CALIBRATION_FACTOR,
        "e3_decision_interval": E3_DECISION_INTERVAL,
        "num_candidates": NUM_CANDIDATES,
        "candidate_horizon": CANDIDATE_HORIZON,
        "alpha_world": ALPHA_WORLD,
        "select_temperature": SELECT_TEMPERATURE,
        "lr": LR,
        "env": {"cls": "CausalGridWorldV2", "use_proxy_fields": True,
                "hazard_harm": HAZARD_HARM, "env": ENV},
        "thresholds": {
            "MANIP_TOOK_REL_FLOOR": MANIP_TOOK_REL_FLOOR,
            "COMMIT_RANGE_FLOOR": COMMIT_RANGE_FLOOR,
            "MIN_COMMIT_DECISIONS": MIN_COMMIT_DECISIONS,
            "THRESHOLD_FLOOR": THRESHOLD_FLOOR,
            "RHO_FLOOR": RHO_FLOOR,
            "RHO_SEED_FLOOR": RHO_SEED_FLOOR,
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
        "per_seed_results": per_seed,
        "metrics": {
            "mean_rho_rv_vs_commit": mean_rho,
            "manip_took_worst_seed": manip_took_measured,
            "commit_range_worst_seed": commit_range_measured,
            "min_commit_decisions_worst": min_decisions_measured,
            "calibrated_threshold_worst_seed": threshold_measured,
            "gate_green": gate_green,
        },
        "criteria": criteria,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "not_a_discovery_experiment_note": (
                "ARC-016's core circuit is ALREADY VALIDATED at train-time by "
                "V3-EXQ-018b (PASS 5/5) and V3-EXQ-060 (PASS 4/5); the 2026-07-25 "
                "rescore confirmed exp_conf 0.775 / confirmed_established / zero "
                "genuine refutations. This run closes the SINGLE residual open item: "
                "eval-time threshold engagement + monotonic tracking. A PASS confirms "
                "the eval-time circuit engages and tracks; it PROMOTES NOTHING "
                "(governance owns status)."),
            "manipulation_note": (
                "The precision manipulation is graded Gaussian noise injected into the "
                "world_forward prediction error that feeds e3.update_running_variance "
                "(error_var = PE.pow(2).mean(), so additive N(0,sigma^2) raises rv by "
                "~sigma^2). sigma_L is calibrated per-seed from the measured eval "
                "baseline so the swept rv brackets the eval-derived threshold BY "
                "CONSTRUCTION -- the fix for V3-EXQ-805, whose env-difficulty lever "
                "moved rv only ~5.6%. The variance stays E3-derived (EMA of a genuine "
                "PE, now with a controlled surprise term)."),
            "scope_guard_note": (
                "STRUCTURAL ONLY. Post-split (2026-03-22) ARC-016 covers just the "
                "mechanistic circuit; the behavioural consequence layer is ARC-029 and "
                "is NOT tagged or tested. No behavioural harm DV is attached (that "
                "would re-import the V_s MONOSTRATEGY-LOCK confound that voided "
                "EXQ-454). The exclusion is STRUCTURAL: the driver takes RANDOM actions "
                "throughout, so there is no learned policy to lock, and no harm/benefit "
                "quantity is a DV anywhere."),
            "engagement_is_a_precondition_note": (
                "'The threshold must actually engage' is a PRECONDITION (commit_rate "
                "range above floor), never a criterion. A threshold that never fires "
                "makes every level identical -- exactly what voided EXQ-396a/b -- and "
                "self-routes substrate_not_ready_requeue rather than a verdict."),
            "dv_symmetry_note": (
                "DV = commit_rate = fraction of selections with running_variance < "
                "commit_threshold. Symmetry group: transforms preserving sign(threshold "
                "- rv) per tick. The additive-noise manipulation is NOT invariant: it "
                "moves rv across a FIXED threshold (derived once from the noise-free "
                "baseline, held constant over the sweep), flipping the sign as sigma "
                "rises -- the measured effect. No broadcast-scalar / rank / permutation "
                "degeneracy: rv is a scalar the noise genuinely moves, commit_rate is a "
                "fraction not an argmax."),
            "threshold_held_fixed_note": (
                "The threshold is derived ONCE from the noise-free eval calibration "
                "window and held FIXED across the sweep. Deriving it per-level would "
                "ABSORB the manipulation and measure nothing."),
            "latch_note": (
                "n_latched_ticks = 0 by construction: agent.e3.select() is called "
                "directly, so the E3 cadence / commitment latch is not in the path and "
                "no diagnostic can be re-read from a stale tick (V3-EXQ-785 defect "
                "structurally impossible)."),
            "governance_flag_not_acted_on": (
                "This run neither promotes nor re-scores ARC-016. The 2026-07-25 "
                "rescore already corrected the exp_conf aggregation artefact (0.521 -> "
                "0.775); a PASS here is the evidence a future /governance cycle would "
                "weigh for provisional -> shown, not a status change this run applies."),
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
            "proposal_id": None,
            "residual_gate_context": (
                "Named in ARC-016 what_would_answer and the 2026-07-25 "
                "/claim-synthesis rescore (claim_synthesis_arc-016_2026-07-25.md) as "
                "the single run standing between provisional and shown."),
            "gov_reuse_1_check": (
                "Decisive readout: commit_rate under an EVAL-DERIVED, held-fixed commit "
                "threshold across a NOISE-swept E3 running-variance level. Checked the "
                "ARC-016 corpus (018/018b/031/096a supports; 396/396a/396b/454/530/805 "
                "non_contributory): 018b/396a/b derive the threshold from TRAINING "
                "variance by construction; 805 used an env-difficulty manipulation that "
                "did not take (rv_diff_relative 0.0558) and records no per-level noise "
                "sweep. Readout not recoverable/derivable -> must run."),
            "supersedes_context": (
                "Does NOT set `supersedes`: 396a/b/454/530/805 are already "
                "non_contributory, so there is no live scoring to displace. NEW EXQ "
                "number (not an 805 letter) because the manipulated factor -- graded "
                "PE noise swept to bracket the threshold, testing MONOTONIC tracking -- "
                "is a different design from 805's binary env-difficulty delta."),
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

    print("V3-EXQ-818: ARC-016 eval-derived threshold + graded-noise precision sweep",
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
    print(f"  gate_green={m['gate_green']} "
          f"manip_took={m['manip_took_worst_seed']:.3f} "
          f"commit_range={m['commit_range_worst_seed']:.3f}", flush=True)
    print(f"  C1 mean_rho={m['mean_rho_rv_vs_commit']:.4f} "
          f"(floor {-RHO_FLOOR})", flush=True)
    for r in manifest["per_seed_results"]:
        print(f"  [seed {r['seed']}] rho={r['rho_rv_vs_commit']:.4f} "
              f"commit_range={r['commit_range']:.3f} "
              f"threshold={r['calibrated_threshold']:.8f} "
              f"cr_by_level={[round(c, 3) for c in r['per_level_commit_rate']]}",
              flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
