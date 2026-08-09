#!/opt/local/bin/python3
"""
V3-EXQ-903a -- MECH-075 Ventral VTA-RPE Probe -- INSTRUMENTED RE-RUN of V3-EXQ-903

EXPERIMENT_PURPOSE = "evidence"

V3-EXQ-903a CHANGE (per-seed episode-length / termination-cause recording)
--------------------------------------------------------------------------------
V3-EXQ-903 (run_id v3_exq_903_mech075_ventral_vta_rpe_probe_20260808T222748Z_v3,
FAIL, substrate_not_ready_requeue) self-routed correctly per the 2026-08-09
cluster autopsy (failure_autopsy_mech075-vta-lc-cluster_2026-08-09): P0 readiness
passed in only 1/3 seeds (seed 7: grounding_corr +0.674, |dt| 0.0187), while
seeds 42/123 showed NEGATIVE grounding correlation (-0.042, -0.085). The
seed-majority gate (summary.n_p0a_pass = 1/3) correctly FAILED even though the
top-level manifest preconditions[].met read `true` (a seed-mean statistic
dominated by the single passing seed -- authoritative gate is the majority
count, not the mean).

THE CONFOUND THIS ITERATION INSTRUMENTS: seed 7's apparent P0 pass coincided
with ~25x FEWER eval samples than the other seeds (n_dt_samples 292 vs 6774-7956;
n_basin_samples 248 vs 1349-1778), consistent with near-immediate episode
TERMINATION (agent dying / health_depleted early, so far fewer steps per
episode). A grounding correlation computed over a near-empty episode is
small-sample-volatile -- seed 7's +0.674 may be a sampling artifact of a
qualitatively different, hazard-dominated short-horizon regime rather than
genuine valuation-head grounding. 903 recorded n_dt_samples but NOT episode
length or termination cause, so the confound could only be inferred, not
confirmed.

FIX (this iteration), per the autopsy's routing:
  (1) RECORD per-seed, per-condition (and per-positive-control) episode LENGTH
      and TERMINATION CAUSE. The env surfaces info["done_cause"]
      ("health_depleted" | "step_limit" | "") and info["episode_steps"]. In this
      experiment (inner loop capped at STEPS_PER_EP=200 < the env's own 500-step
      cap) an episode ends either by health_depleted (done=True, an early death)
      or by exhausting the experiment's own step budget (loop runs all `steps`,
      done never fires -> recorded "step_budget_exhausted"). Each phase now emits
      episode_lengths[], a termination_cause count dict, and mean_episode_length,
      so a future P0 read can be trusted (or discarded) against the actual
      per-seed sample regime rather than an inferred one.
  (2) INCREASE seed count (3 -> 5: [42, 7, 123, 5, 17], majority 2/3 -> 3/5) and
      POSCTRL episodes (5 -> 10) to reduce the small-sample correlation
      volatility that made a single anomalous seed dominate the majority gate.
      (Seed 44 deliberately avoided per the CLAUDE.md reef-config caution;
      irrelevant to this grid-world env but harmless to respect.)
Nothing about the mechanism, signal, terrain metric, phased training, or the
P0/C1/C2 criteria and thresholds changes -- this is a same-question re-run with
better instrumentation and a more robust readiness gate. `supersedes` = the
V3-EXQ-903 run_id (its P0 read was uninformative; do not double-count it).

MECHANISM / PURPOSE
--------------------
MECH-075 bifurcates into a DORSAL leg (LC-mediated arousal gain on spatial/
planning terrain -- V3-EXQ-192/192a/209/230, all FAIL/weakens) and a VENTRAL
leg (VTA-mediated reward-prediction-error gain on motivational/affective
terrain), which has NEVER been experimentally attempted: every prior attempt
instrumented dorsal-HPC terrain with a VTA/RPE-shaped signal -- the wrong axis
pairing. This script is the first attempt at the ventral leg on its own terms.

DIRECTION IS OPPOSITE THE DORSAL SIBLING. The dorsal script's hypothesis is
that LC-arousal WIDENS attractor basin width (broadens exploration under
arousal). This script's hypothesis is that VTA-RPE reinforcement NARROWS
attractor basin width (reinforcement sharpens commitment onto a valenced
attractor -- a Go/No-Go-style convergence, not a broadening). The CEM
noise-scaling formula below is deliberately the inverse shape of the dorsal
script's (dividing by (1 + gain * signal) instead of multiplying by it).

Signal used: the REAL, live-computed, SIGNED VTA-RPE analog already wired
under the ARC-108 "unified dopamine substrate" in
ree_core/predictors/e3_selector.py -- E3TrajectorySelector.post_action_update()
computes, when config.use_habenula_decommit=True:
    R_t     = benefit_eval_head(z_world).mean() - harm_eval_head(z_world).mean()
    delta_t = R_t - v_hat        (v_hat = self._lcg_value_baseline, a slow EMA)
and surfaces it via REEAgent.update_residue()'s return dict as
metrics["e3_habenula_delta_t"]. This is explicitly NOT the unsigned ARC-016
prediction-error-variance channel (MECH-043's dependency) -- delta_t is SIGNED,
which is what a reinforcement-gain leg needs (MECH-043's unsigned precision
channel is excluded by the claim's own what_would_answer text).

CORRECTION TO THE ORCHESTRATING SESSION'S ARC-108 READING (see also the
report filed with this script): benefit_eval_head / harm_eval_head are NOT
pre-trained "already-trained submodules" in general -- REEAgent.
actor_critic_reward()'s own docstring warns they "start random-init behind
the ARC-030 warmup gate" and are "only meaningful once those heads are
grounded ... a validation using it MUST first assert grounding on a positive
control." agent.py has compute_benefit_eval_loss() (ARC-030/MECH-112,
supervised MSE against info["benefit_exposure"]) but NO harm-side
counterpart -- this script adds a local _compute_harm_eval_loss() that
mirrors compute_benefit_eval_loss()'s exact pattern, supervising
E3.harm_eval_head against info["harm_exposure"] (CausalGridWorldV2
use_proxy_fields=True nociceptive EMA; the harm-side sibling of ARC-030's
hedonic EMA). Both heads are trained during the shared warmup phase, and P0
below explicitly asserts grounding on a positive-control episode before the
main seed grid is trusted (the positive control agent.py's own docstring
requires).

A first calibration run training ONLY the two valuation heads (no encoder-
side supervision, joint with E1/E2 throughout) showed corr(pred_harm,
true_harm) = -0.15 after 200 episodes: harm_eval_head converged to a
near-constant prediction that minimises aggregate MSE without tracking
instantaneous hazard proximity. TWO compounding causes, both fixed:
  (1) ENCODER-representation gap: z_world itself carries near-zero
      hazard/resource-discriminative structure from E1/E2 reconstruction
      losses alone (compute_resource_proximity_loss's own docstring names
      this exact failure mode: R2=-0.004, EXQ-085m, absent SD-018
      supervision). Fixed by ALSO training the z_world encoder via the same
      SD-018 (resource-proximity regression, compute_resource_proximity_loss)
      and SD-009 (event-contrastive CE, compute_event_contrastive_loss)
      auxiliary losses MECH-073's own V3-EXQ-375 uses
      (use_resource_proximity_head=True, use_event_classifier=True) --
      existing agent.py methods, not new machinery.
  (2) MOVING-TARGET COLLAPSE (queue-experiment skill "Training protocol"
      rule): training the encoder (e1/e2/aux losses) and the valuation
      heads JOINTLY on every step makes z_world a moving target for the
      heads -- the same failure mode documented elsewhere (EXQ-166b/c/d,
      EXQ-085l, EXQ-194). Fixed by explicit PHASED training:
      TRAIN_ENCODER_WARMUP (e1/e2/SD-018/SD-009 only, no valuation loss) ->
      TRAIN_HEAD_GROUNDING (benefit_loss + harm_loss only; both already read
      z_world .detach()ed, and with no encoder-loss term in that phase's
      `total`, Adam never updates the encoder -- no explicit
      requires_grad=False needed).
Together these flipped corr(pred_harm, true_harm) from -0.15 to +0.43 in a
follow-up smoke test (see the filed report for the full before/after
numbers); whether it clears the pre-registered floor at the full training
budget is exactly what P0 measures.

CEM noise-scaling mechanism: rather than monkey-patching HippocampalModule's
CEM sampling internals (the April v3_exq_192a pattern), this script routes
through the REAL, already-wired MECH-267 mode-conditioned proposal mechanism
(HippocampalConfig.mode_conditioning_enabled / mode_noise_scale) with a
synthetic "rpe_gated" mode whose probability weight is always 1.0 -- i.e.
mode_noise_scale["rpe_gated"] directly becomes the CEM proposal std
multiplier, updated live from the ventral RPE EMA. _e3_tick() never passes
operating_mode to propose_trajectories() on the ordinary waking path, so a
thin instance-level wrapper on agent.hippocampal.propose_trajectories injects
operating_mode={"rpe_gated": 1.0} on every call; the actual multiplier is
written to config.mode_noise_scale["rpe_gated"] before each tick.

Terrain metric ("ventral/motivational attractor basin width"): the SPREAD
(population std) of predicted VALENCE (benefit_eval_head - harm_eval_head)
across the K proposed candidates' first-step predicted z_world, computed the
same way agent.py's own _candidate_world_summaries() builds action-conditional
z_world predictions (z0 = current z_world, a_i = each candidate's first
action, agent.e2.world_forward(z0_K, actions_K) -> [K, world_dim]). This
directly reuses MECH-073's own validated valence-geometry methodology
(benefit_eval_head / harm_eval_head on z_world; V3-EXQ-375) rather than a
spatial-position spread (which would be the dorsal metric).

Conditions (single agent per seed, shared warmup, two eval conditions)
------------------------------------------------------------------------
VTA_RPE_GATED   -- EMA of |delta_t| feeds mode_noise_scale["rpe_gated"] =
                   1 / (1 + RPE_GAIN * rpe_abs_ema) each step (narrows CEM
                   search width as reinforcement-relevant signal grows).
VTA_RPE_ABLATED -- mode_noise_scale["rpe_gated"] held at 1.0 (fixed baseline
                   CEM noise) throughout. The TRUE delta_t is still computed
                   and logged every step (use_habenula_decommit stays True in
                   BOTH conditions) rather than disabling the flag -- this
                   reports what the "would-be" RPE looked like in the ablated
                   arm too, which is more informative than a fully-disabled
                   config for interpreting a null C2 result.

Pre-registered criteria
------------------------
P0 (readiness, positive control, evaluated BEFORE the main seed grid trusts
    any result): TWO preconditions on a dedicated positive-control episode
    per seed (genuine harm/benefit-differentiated transitions):
      P0a valuation_heads_grounded: Pearson correlation between predicted
          valence (benefit_eval_head - harm_eval_head on the live z_world)
          and the env's true (benefit_exposure - harm_exposure) clears
          GROUNDING_CORR_FLOOR. Directly asserts the grounding
          actor_critic_reward()'s docstring requires.
      P0b rpe_signal_detectable: mean |delta_t| over the positive control
          clears RPE_FLOOR (the floor is NOT pre-established for this signal
          per the claim's own text; calibrated empirically via --dry-run,
          see the module-level constants below and the filed report).
    P0 fails on either sub-check -> substrate_not_ready_requeue (uninformative,
    not a claim verdict -- this is the FIRST attempt at this leg).
C1 (manipulation check): mean |delta_t| in GATED clears RPE_FLOOR, >= 2/3
    seeds. NOT load_bearing (manipulation check, not the behavioral claim).
C2 (behavioral, matches the claim's CONFIRMING criterion): GATED
    ventral-valence-spread (basin width) is LOWER than ABLATED by
    NARROW_REL_THRESH (relative decrease), >= 2/3 seeds. load_bearing: true.

PASS requires P0 + C1 + C2. P0 or C1 fail -> substrate_not_ready_requeue.
P0+C1 pass, C2 fails -> genuine FALSIFYING result (evidence_direction
"weakens", per the claim's own FALSIFYING text: "RPE signal clears its
detection floor but produces no measurable ventral attractor-width
modulation"). All pass -> evidence_direction "supports".

Claims: MECH-075 (single claim; the dorsal leg is a sibling script, not this
one). Seeds [42, 7, 123] match V3-EXQ-230's set for continuity with the
dorsal sibling.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_903a_mech075_ventral_vta_rpe_probe"
CLAIM_IDS = ["MECH-075"]
EXPERIMENT_PURPOSE = "evidence"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
# Superseded predecessor: its P0 read was uninformative (1/3 seed-majority FAIL,
# the passing seed confounded by ~25x fewer eval samples). See docstring
# "V3-EXQ-903a CHANGE" and the 2026-08-09 cluster autopsy.
SUPERSEDES_RUN_ID = "v3_exq_903_mech075_ventral_vta_rpe_probe_20260808T222748Z_v3"

# 903a: seed count raised 3 -> 5 (majority 2/3 -> 3/5) to reduce small-sample
# correlation volatility in the P0 readiness gate. Seed 44 avoided (reef-config
# caution; irrelevant here but harmless).
SEEDS = [42, 7, 123, 5, 17]
MAJORITY = 3  # of 5 seeds

# ---------------------------------------------------------------------------
# Dims (CausalGridWorldV2 / use_proxy_fields=True convention -- matches
# V3-EXQ-230's env dims for continuity with the dorsal sibling)
# ---------------------------------------------------------------------------
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 5
WORLD_DIM = 32
SELF_DIM = 32

TRAIN_EPISODES = 150
POSCTRL_EPISODES = 10  # 903a: raised 5 -> 10 to stabilise the P0a grounding correlation
EVAL_EPISODES = 40
STEPS_PER_EP = 200
LR = 1e-3

# ---------------------------------------------------------------------------
# Pre-registered thresholds -- calibrated empirically against several
# throwaway probe runs on this exact substrate/config (see the filed report
# for the full before/after numbers). The claim's own what_would_answer text
# guessed the |delta_t| natural scale would land closer to O(0.01-1) "since
# it's built from valuation-head outputs rather than an MSE" -- empirically
# WRONG: observed mean |delta_t| across every probe (5 to 200 warmup
# episodes, with and without encoder-auxiliary losses, with and without
# phased training) fell in O(0.0002-0.02), because v_hat (the slow EMA
# baseline delta_t is measured AGAINST) tracks R_t's own distribution and
# keeps the RESIDUAL small even once R_t itself is well-grounded. RPE_FLOOR
# below is set against that observed range, not the claim text's guess.
# ---------------------------------------------------------------------------
RPE_GAIN = 50.0              # CEM noise narrowing gain: scale = 1/(1+RPE_GAIN*ema).
                              # At the observed rpe_abs_ema range (~0.002-0.02) this
                              # gives cem_noise_scale ~0.5-0.91 -- a decisive
                              # manipulation range, not the ~0.86-0.99 an
                              # RPE_GAIN=8 (dorsal-script-scale gain) would give at
                              # this substrate's much smaller delta_t magnitude.
RPE_EMA_ALPHA = 0.1           # EMA alpha for |delta_t| (matches dorsal novelty-EMA convention)
RPE_FLOOR = 0.001             # P0b / C1: mean |delta_t| detection floor. ~5-10x
                              # above the lowest genuine (non-degenerate, trained)
                              # values observed (~0.0002 at 5 warmup episodes),
                              # comfortably above pure numerical noise (the other
                              # V3 signals in this codebase floor around 1e-4 to
                              # 1e-5), well below the ~0.002-0.02 range observed at
                              # 60-200 warmup episodes.
GROUNDING_CORR_FLOOR = 0.15   # P0a: predicted-vs-true valence correlation floor.
                              # Observed 0.43 with phased training + encoder
                              # auxiliary losses even at a 10-episode (5+5) smoke
                              # budget, vs -0.15/-0.28 without either fix -- 0.15
                              # is a real, achieved-in-practice bar, not a token one.
NARROW_REL_THRESH = 0.15      # C2: GATED must be >=15% narrower than ABLATED


# ---------------------------------------------------------------------------
# Env / config factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
        use_proxy_fields=True,
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        use_habenula_decommit=True,
        # SD-018/SD-009 z_world-encoder auxiliary supervision (MECH-073's own
        # V3-EXQ-375 validated methodology): WITHOUT these, benefit_eval_head/
        # harm_eval_head can only discriminate whatever hazard/resource structure
        # z_world happens to carry for free from E1/E2 reconstruction losses,
        # which is near-zero (compute_resource_proximity_loss's own docstring
        # cites R2=-0.004, EXQ-085m, absent this). Confirmed empirically here
        # too: a calibration run WITHOUT these (60-200 warmup episodes, no
        # encoder-side supervision) showed corr(pred_harm, true_harm) = -0.15,
        # i.e. harm_eval_head converges to a near-constant prediction that
        # minimises aggregate MSE rather than tracking instantaneous hazard
        # proximity -- an encoder-representation problem, not a valuation-head
        # problem. See the filed report for the full before/after numbers.
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
    )


# ---------------------------------------------------------------------------
# Harm-eval grounding loss (LOCAL: agent.py has compute_benefit_eval_loss but
# no harm-side counterpart -- this mirrors its exact pattern for harm_eval_head).
# ---------------------------------------------------------------------------

def _compute_harm_eval_loss(agent: REEAgent, harm_exposure: torch.Tensor) -> torch.Tensor:
    """Supervise E3.harm_eval_head to predict hazard proximity from
    info["harm_exposure"] (CausalGridWorldV2 use_proxy_fields=True nociceptive
    EMA) -- the harm-side sibling of ARC-030/MECH-112's compute_benefit_eval_loss
    (agent.py:9513), which trains benefit_eval_head from benefit_exposure but has
    no harm-side counterpart in agent.py. Gradient flows through
    E3.harm_eval_head only (z_world detached), exactly mirroring
    compute_benefit_eval_loss's own shape.
    """
    zero_loss = next(agent.e3.parameters()).sum() * 0.0
    if agent._current_latent is None:
        return zero_loss
    z_world = agent._current_latent.z_world.detach()
    harm_pred = agent.e3.harm_eval(z_world)  # [batch, 1]
    target = harm_exposure.to(z_world.device).float()
    if target.dim() == 0:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 1:
        target = target.unsqueeze(-1)
    return F.mse_loss(harm_pred, target.expand_as(harm_pred))


# ---------------------------------------------------------------------------
# Ventral valence-spread ("attractor basin width") over CEM candidates
# ---------------------------------------------------------------------------

def _candidate_valence_spread(agent: REEAgent, candidates: list) -> Optional[float]:
    """Population std of predicted valence (benefit_eval_head - harm_eval_head)
    across the K candidates' first-step predicted z_world -- the ventral analog
    of a dorsal spatial-spread metric. Mirrors agent._candidate_world_summaries()'s
    exact z0_K / actions_K / e2.world_forward calling convention (agent.py
    around line 5462), but computed locally so it runs regardless of
    config.candidate_summary_source (which gates that method's use in
    selection, not diagnostics). Returns None when unavailable (< 2 candidates
    or no current latent).
    """
    if agent._current_latent is None or not candidates:
        return None
    z0 = agent._current_latent.z_world.detach()
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0[:1]
    adim = int(candidates[0].actions.shape[-1])
    act_rows = [c.actions[:, 0, :].detach().reshape(adim) for c in candidates]
    actions_K = torch.stack(act_rows, dim=0).to(device=z0.device, dtype=z0.dtype)
    if actions_K.shape[0] < 2:
        return None
    z0_K = z0.expand(actions_K.shape[0], -1)
    with torch.no_grad():
        preds = agent.e2.world_forward(z0_K, actions_K)  # [K, world_dim]
        benefit = agent.e3.benefit_eval_head(preds).squeeze(-1)
        harm = agent.e3.harm_eval_head(preds).squeeze(-1)
        valence = benefit - harm
    return float(valence.std(unbiased=False).item())


def _cem_noise_scale(rpe_abs_ema: float, gain: float) -> float:
    """Ventral (narrowing) shape -- the inverse of the dorsal script's widening
    (1 + gain*signal) formula: reinforcement-relevant signal SHRINKS CEM search
    width instead of expanding it."""
    return 1.0 / (1.0 + gain * rpe_abs_ema)


def _term_cause_counts(causes: List[str]) -> Dict[str, int]:
    """903a: tally episode termination causes (e.g. health_depleted vs
    step_budget_exhausted) so a per-seed sample-size confound is legible."""
    counts: Dict[str, int] = {}
    for c in causes:
        counts[c] = counts.get(c, 0) + 1
    return counts


def _episode_length_stats(lengths: List[int], causes: List[str]) -> Dict[str, Any]:
    """903a: per-phase episode-length + termination-cause summary. mean_episode_
    length near STEPS_PER_EP means episodes ran to the step budget; well below it
    means early deaths (the seed-7 short-horizon confound the autopsy named)."""
    n = len(lengths)
    return {
        "n_episodes": n,
        "mean_episode_length": (sum(lengths) / n) if n else 0.0,
        "min_episode_length": min(lengths) if lengths else 0,
        "max_episode_length": max(lengths) if lengths else 0,
        "episode_lengths": list(lengths),
        "termination_causes": _term_cause_counts(causes),
    }


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom < 1e-12:
        return 0.0
    return cov / denom


def _install_rpe_gated_proposer(agent: REEAgent):
    """Route CEM proposal noise through the REAL, already-wired MECH-267
    mode-conditioned mechanism (HippocampalConfig.mode_conditioning_enabled /
    mode_noise_scale; ree_core/hippocampal/module.py propose_trajectories /
    _compute_mode_noise_scale) instead of reimplementing CEM noise generation
    by hand (unlike the April v3_exq_192a script's direct monkey-patch of the
    CEM sampling internals). A synthetic "rpe_gated" mode with probability
    weight 1.0 means mode_noise_scale["rpe_gated"] IS the noise multiplier,
    updated live from the ventral RPE EMA each step. _e3_tick() never passes
    operating_mode to propose_trajectories on the ordinary waking path, so this
    installs a thin instance-level wrapper that injects it on every call.
    Returns the live HippocampalConfig so the caller can write
    config.mode_noise_scale["rpe_gated"] per step.
    """
    hip = agent.hippocampal
    hip.config.mode_conditioning_enabled = True
    hip.config.mode_noise_scale["rpe_gated"] = 1.0
    _orig_propose = hip.propose_trajectories

    def _wrapped_propose(*args, **kwargs):
        kwargs["operating_mode"] = {"rpe_gated": 1.0}
        return _orig_propose(*args, **kwargs)

    hip.propose_trajectories = _wrapped_propose
    return hip.config


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    n_train = 5 if dry_run else TRAIN_EPISODES
    n_posctrl = 2 if dry_run else POSCTRL_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES
    steps = 20 if dry_run else STEPS_PER_EP

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    config = _make_config()
    agent = REEAgent(config)
    opt = optim.Adam(agent.parameters(), lr=LR)

    print(
        f"  [EXQ-903] seed={seed} train={n_train} posctrl={n_posctrl}"
        f" eval_per_cond={n_eval} steps={steps}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # PHASED training (queue-experiment skill "Training protocol" rule: any
    # head trained on z_world must use P0 encoder-warmup -> P1 frozen-encoder
    # head-training, never joint training -- joint training makes z_world a
    # MOVING TARGET for the heads and silently produces moving-target
    # collapse, confirmed repeatedly elsewhere (EXQ-166b/c/d, EXQ-085l,
    # EXQ-194)). A first calibration WITHOUT phasing (joint e1/e2/benefit/harm
    # training throughout) showed corr(pred_harm, true_harm) = -0.15 after
    # 200 episodes even WITH the SD-018/SD-009 encoder-auxiliary losses added
    # -- consistent with moving-target collapse, not just an under-trained
    # encoder. So training is split into two explicit phases:
    #   TRAIN_ENCODER_WARMUP -- e1_loss + e2_loss + the SD-018 resource-
    #     proximity regression + SD-009 event-contrastive CE (MECH-073's own
    #     V3-EXQ-375 methodology for forcing z_world to carry hazard/resource-
    #     discriminative structure; compute_resource_proximity_loss's own
    #     docstring cites R2=-0.004 absent this). NO valuation-head loss yet.
    #   TRAIN_HEAD_GROUNDING -- encoder losses STOP (no e1/e2/aux terms in
    #     `total`), so no gradient reaches the encoder even indirectly (Adam
    #     skips a param whose .grad is None after zero_grad()+backward() on a
    #     total that never touched it). Only benefit_loss + harm_loss train,
    #     and both already read z_world .detach()ed (compute_benefit_eval_loss
    #     / _compute_harm_eval_loss), so the heads converge against a FIXED
    #     target representation instead of a shifting one.
    # -----------------------------------------------------------------------
    n_encoder_warmup = max(1, n_train // 2)
    n_head_train = max(1, n_train - n_encoder_warmup)

    print(f"Seed {seed} Condition TRAIN_ENCODER_WARMUP", flush=True)
    agent.train()
    prev_ttype = "none"
    for ep in range(n_encoder_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)
            _, reward, done, info, obs_dict_next = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ttype = info.get("transition_type", "none")

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss

            # SD-018: resource-proximity encoder supervision, from the obs
            # that PRODUCED `latent` (obs_dict -- NOT obs_dict_next).
            rfv = obs_dict.get("resource_field_view")
            if rfv is not None:
                total = total + agent.compute_resource_proximity_loss(
                    float(rfv.max().item()), latent
                )
            # SD-009: event-contrastive encoder supervision. `latent` (obs_t)
            # is labelled with prev_ttype (the transition that PRODUCED
            # obs_t, i.e. this step's info BEFORE it is rolled into
            # prev_ttype below) -- not this step's own ttype.
            total = total + agent.compute_event_contrastive_loss(prev_ttype, latent)

            if total.requires_grad:
                opt.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                opt.step()
            agent.update_residue(harm_signal)
            prev_ttype = ttype
            obs_dict = obs_dict_next
            if done:
                break
        if (ep + 1) % 50 == 0:
            print(f"    [train] seed={seed} ep {ep + 1}/{n_encoder_warmup}", flush=True)

    print(f"Seed {seed} Condition TRAIN_HEAD_GROUNDING", flush=True)
    for ep in range(n_head_train):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)
            _, reward, done, info, obs_dict_next = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            # Encoder is NOT touched this phase: total is built ONLY from the
            # two (already-detached-z_world) valuation losses.
            benefit_t = torch.tensor([max(0.0, float(info.get("benefit_exposure", 0.0)))])
            harm_t = torch.tensor([max(0.0, float(info.get("harm_exposure", 0.0)))])
            benefit_loss = agent.compute_benefit_eval_loss(benefit_t)
            harm_loss = _compute_harm_eval_loss(agent, harm_t)
            total = benefit_loss + harm_loss
            if total.requires_grad:
                opt.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                opt.step()
            agent.e3.record_benefit_sample()
            agent.update_residue(harm_signal)
            obs_dict = obs_dict_next
            if done:
                break
        if (ep + 1) % 50 == 0:
            print(f"    [train] seed={seed} ep {ep + 1}/{n_head_train}", flush=True)

    # -----------------------------------------------------------------------
    # P0: positive-control episodes -- assert valuation-head grounding AND
    # raw |delta_t| detectability BEFORE trusting the main seed grid.
    # -----------------------------------------------------------------------
    print(f"Seed {seed} Condition POSITIVE_CONTROL", flush=True)
    agent.eval()
    posctrl_abs_dt: List[float] = []
    posctrl_pred_valence: List[float] = []
    posctrl_true_valence: List[float] = []
    posctrl_ep_lengths: List[int] = []       # 903a: per-episode length
    posctrl_term_causes: List[str] = []      # 903a: per-episode termination cause
    for ep in range(n_posctrl):
        _, obs_dict = env.reset()
        agent.reset()
        ep_len = steps
        term_cause = "step_budget_exhausted"
        for _step in range(steps):
            obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, WORLD_DIM, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                z_now = agent._current_latent.z_world.detach()
                pred_val = float(
                    (agent.e3.benefit_eval_head(z_now) - agent.e3.harm_eval_head(z_now))
                    .mean().item()
                )
            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            residue_metrics = agent.update_residue(harm_signal)
            dt = residue_metrics.get("e3_habenula_delta_t")
            if dt is not None:
                posctrl_abs_dt.append(abs(float(dt.item())))
            true_val = float(info.get("benefit_exposure", 0.0)) - float(info.get("harm_exposure", 0.0))
            posctrl_pred_valence.append(pred_val)
            posctrl_true_valence.append(true_val)
            if done:
                ep_len = _step + 1
                term_cause = str(info.get("done_cause", "") or "unknown_done")
                break
        posctrl_ep_lengths.append(ep_len)
        posctrl_term_causes.append(term_cause)

    p0b_mean_abs_dt = sum(posctrl_abs_dt) / max(1, len(posctrl_abs_dt))
    p0a_grounding_corr = _pearson(posctrl_pred_valence, posctrl_true_valence)
    p0a_pass = p0a_grounding_corr >= GROUNDING_CORR_FLOOR
    p0b_pass = p0b_mean_abs_dt >= RPE_FLOOR
    p0_pass = p0a_pass and p0b_pass
    # 903a: episode-length / termination-cause of the positive control. The P0a
    # grounding correlation is computed over these steps, so a short-horizon
    # posctrl (few samples) is exactly what makes that correlation volatile --
    # n_posctrl_valence_samples is the true denominator behind the Pearson r.
    posctrl_ep_stats = _episode_length_stats(posctrl_ep_lengths, posctrl_term_causes)
    print(
        f"  [POSITIVE_CONTROL] seed={seed} mean|delta_t|={p0b_mean_abs_dt:.5f}"
        f" grounding_corr={p0a_grounding_corr:.4f} n_samples={len(posctrl_pred_valence)}"
        f" mean_ep_len={posctrl_ep_stats['mean_episode_length']:.1f}"
        f" term={posctrl_ep_stats['termination_causes']}"
        f" P0a={'P' if p0a_pass else 'F'} P0b={'P' if p0b_pass else 'F'}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Eval phase: VTA_RPE_GATED then VTA_RPE_ABLATED.
    # -----------------------------------------------------------------------
    hip_config = _install_rpe_gated_proposer(agent)
    condition_stats: Dict[str, Dict[str, float]] = {}

    for condition in ["VTA_RPE_GATED", "VTA_RPE_ABLATED"]:
        print(f"Seed {seed} Condition {condition}", flush=True)
        agent.e3._lcg_value_baseline = 0.0
        rpe_abs_ema = 0.0
        dt_history: List[float] = []
        basin_widths: List[float] = []
        eval_ep_lengths: List[int] = []      # 903a: per-episode length
        eval_term_causes: List[str] = []     # 903a: per-episode termination cause

        for ep in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()
            ep_len = steps
            term_cause = "step_budget_exhausted"
            for _step in range(steps):
                obs_body = torch.tensor(obs_dict["body_state"], dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

                cem_scale = (
                    _cem_noise_scale(rpe_abs_ema, RPE_GAIN)
                    if condition == "VTA_RPE_GATED" else 1.0
                )
                hip_config.mode_noise_scale["rpe_gated"] = cem_scale

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks["e1_tick"]
                        else torch.zeros(1, WORLD_DIM, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    if ticks["e3_tick"]:
                        spread = _candidate_valence_spread(agent, candidates)
                        if spread is not None:
                            basin_widths.append(spread)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                _, reward, done, info, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0
                residue_metrics = agent.update_residue(harm_signal)
                dt = residue_metrics.get("e3_habenula_delta_t")
                if dt is not None:
                    dt_abs = abs(float(dt.item()))
                    dt_history.append(dt_abs)
                    if condition == "VTA_RPE_GATED":
                        rpe_abs_ema = (
                            (1 - RPE_EMA_ALPHA) * rpe_abs_ema + RPE_EMA_ALPHA * dt_abs
                        )
                    else:
                        # ABLATED: EMA held at 0 (the noise scale stays fixed at
                        # 1.0 above regardless); the TRUE dt is still logged in
                        # dt_history for diagnostic "what would it have looked
                        # like" reporting.
                        rpe_abs_ema = 0.0

                if done:
                    ep_len = _step + 1
                    term_cause = str(info.get("done_cause", "") or "unknown_done")
                    break
            eval_ep_lengths.append(ep_len)
            eval_term_causes.append(term_cause)

        mean_abs_dt = sum(dt_history) / max(1, len(dt_history))
        mean_basin = sum(basin_widths) / max(1, len(basin_widths))
        ep_stats = _episode_length_stats(eval_ep_lengths, eval_term_causes)
        condition_stats[condition] = {
            "mean_abs_delta_t": mean_abs_dt,
            "mean_basin_width": mean_basin,
            "n_dt_samples": len(dt_history),
            "n_basin_samples": len(basin_widths),
            # 903a: episode-length / termination-cause instrumentation -- the
            # confound the autopsy named (seed 7 = ~25x fewer samples via early
            # deaths) is now recorded, not inferred.
            "mean_episode_length": ep_stats["mean_episode_length"],
            "min_episode_length": ep_stats["min_episode_length"],
            "max_episode_length": ep_stats["max_episode_length"],
            "episode_lengths": ep_stats["episode_lengths"],
            "termination_causes": ep_stats["termination_causes"],
        }
        print(
            f"  [{condition}] seed={seed} mean|delta_t|={mean_abs_dt:.5f}"
            f" basin_width={mean_basin:.5f}"
            f" n_dt={len(dt_history)} n_basin={len(basin_widths)}"
            f" mean_ep_len={ep_stats['mean_episode_length']:.1f}"
            f" term={ep_stats['termination_causes']}",
            flush=True,
        )

    c1_pass = condition_stats["VTA_RPE_GATED"]["mean_abs_delta_t"] >= RPE_FLOOR
    gated_basin = condition_stats["VTA_RPE_GATED"]["mean_basin_width"]
    ablated_basin = condition_stats["VTA_RPE_ABLATED"]["mean_basin_width"]
    relative_narrowing = (
        (ablated_basin - gated_basin) / ablated_basin if ablated_basin > 1e-9 else 0.0
    )
    c2_pass = relative_narrowing >= NARROW_REL_THRESH

    seed_pass = p0_pass and c1_pass and c2_pass
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "p0a_grounding_corr": p0a_grounding_corr,
        "p0a_pass": p0a_pass,
        "p0b_mean_abs_delta_t_posctrl": p0b_mean_abs_dt,
        "p0b_pass": p0b_pass,
        "p0_pass": p0_pass,
        "c1_pass": c1_pass,
        "c1_mean_abs_delta_t_gated": condition_stats["VTA_RPE_GATED"]["mean_abs_delta_t"],
        "c2_pass": c2_pass,
        "gated_basin_width": gated_basin,
        "ablated_basin_width": ablated_basin,
        "relative_narrowing": relative_narrowing,
        "condition_stats": condition_stats,
        # 903a: P0a's true sample count + the positive-control episode-length /
        # termination-cause regime, so a future P0 read is trustable against the
        # actual per-seed sampling regime (the seed-7 confound the autopsy named).
        "n_posctrl_valence_samples": len(posctrl_pred_valence),
        "posctrl_episode_stats": posctrl_ep_stats,
        "seed_pass": seed_pass,
    }


# ---------------------------------------------------------------------------
# Aggregate run
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], str]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    print(f"\n[V3-EXQ-903] MECH-075 Ventral VTA-RPE Probe (dry_run={dry_run})", flush=True)

    seed_records = [_run_seed(s, dry_run) for s in seeds]

    n = len(seed_records)
    majority = 1 if dry_run else MAJORITY
    n_p0 = sum(1 for r in seed_records if r["p0_pass"])
    n_p0a = sum(1 for r in seed_records if r["p0a_pass"])
    n_p0b = sum(1 for r in seed_records if r["p0b_pass"])
    n_c1 = sum(1 for r in seed_records if r["c1_pass"])
    n_c2 = sum(1 for r in seed_records if r["c2_pass"])
    p0_pass = n_p0 >= majority
    p0a_pass_agg = n_p0a >= majority
    p0b_pass_agg = n_p0b >= majority
    c1_pass = n_c1 >= majority
    c2_pass = n_c2 >= majority
    mean_p0a_corr = sum(r["p0a_grounding_corr"] for r in seed_records) / n
    mean_p0b_dt = sum(r["p0b_mean_abs_delta_t_posctrl"] for r in seed_records) / n

    if not (p0_pass and c1_pass):
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "inconclusive"
        interpretation_text = (
            "P0 (readiness) or C1 (manipulation check) failed in a majority of"
            " seeds. This is an uninformative substrate-readiness result, not a"
            " claim verdict -- the ventral leg's own detection floor was not"
            " established prior to this, its first attempt."
        )
    elif not c2_pass:
        label = "ventral_rpe_no_basin_narrowing"
        outcome = "FAIL"
        evidence_direction = "weakens"
        interpretation_text = (
            "P0 and C1 passed: the signed VTA-RPE delta_t signal clears its"
            " detection floor and is grounded against true env valence. But"
            " C2 failed: no measurable narrowing of ventral/motivational"
            " attractor-basin width under high-RPE-reinforcement gain vs low."
            " Per the claim's own FALSIFYING text, this is a genuine"
            " falsifying result for the ventral leg."
        )
    else:
        label = "ventral_rpe_narrows_basin"
        outcome = "PASS"
        evidence_direction = "supports"
        interpretation_text = (
            "P0, C1 and C2 all passed: the signed VTA-RPE delta_t signal is"
            " detectable and grounded, and gating CEM proposal noise on it"
            " narrows ventral/motivational attractor-basin width relative to"
            " the ablated baseline, exceeding the pre-registered floor."
        )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config = {
        "seeds": seeds,
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "train_episodes": TRAIN_EPISODES,
        "posctrl_episodes": POSCTRL_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EP,
        "lr": LR,
        "rpe_gain": RPE_GAIN,
        "rpe_ema_alpha": RPE_EMA_ALPHA,
        "rpe_floor": RPE_FLOOR,
        "grounding_corr_floor": GROUNDING_CORR_FLOOR,
        "narrow_rel_thresh": NARROW_REL_THRESH,
        "majority_required": majority,
        "dry_run": dry_run,
        "env": {
            "size": 10, "num_hazards": 2, "num_resources": 3,
            "hazard_harm": 0.02, "env_drift_interval": 10, "env_drift_prob": 0.2,
            "use_proxy_fields": True,
        },
        "use_habenula_decommit": True,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES_RUN_ID,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": evidence_direction,
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            "preconditions": [
                {
                    "name": "valuation_heads_grounded",
                    "description": (
                        "Pearson correlation between predicted valence"
                        " (benefit_eval_head - harm_eval_head on live z_world)"
                        " and the env's true (benefit_exposure - harm_exposure)"
                        " on a positive-control episode clears the grounding"
                        " floor -- asserts the grounding"
                        " REEAgent.actor_critic_reward()'s own docstring"
                        " requires before trusting benefit_eval/harm_eval."
                        " (Seed-majority gate used for the actual P0 verdict is"
                        " reported separately under summary.n_p0a_pass.)"
                    ),
                    "measured": mean_p0a_corr,
                    "threshold": GROUNDING_CORR_FLOOR,
                    "direction": "floor",
                    "control": "positive-control episode with genuine harm/benefit transitions",
                    "met": bool(mean_p0a_corr >= GROUNDING_CORR_FLOOR),
                },
                {
                    "name": "rpe_signal_detectable",
                    "description": (
                        "Mean |delta_t| over the positive-control episode"
                        " clears the RPE detection floor (not pre-established"
                        " for this signal; calibrated empirically, see the"
                        " filed report for the observed distribution)."
                        " (Seed-majority gate used for the actual P0 verdict is"
                        " reported separately under summary.n_p0b_pass.)"
                    ),
                    "measured": mean_p0b_dt,
                    "threshold": RPE_FLOOR,
                    "direction": "floor",
                    "control": "positive-control episode with genuine harm/benefit transitions",
                    "met": bool(mean_p0b_dt >= RPE_FLOOR),
                },
            ],
            "criteria_non_degenerate": {
                "C1_manipulation_check": True,
                "C2_ventral_basin_narrowing": True,
            },
            "criteria": [
                {
                    "name": "C1_manipulation_check",
                    "load_bearing": False,
                    "passed": bool(c1_pass),
                    "detail": f"{n_c1}/{n} seeds: mean|delta_t| in GATED >= {RPE_FLOOR}",
                },
                {
                    "name": "C2_ventral_basin_narrowing",
                    "load_bearing": True,
                    "passed": bool(c2_pass),
                    "detail": (
                        f"{n_c2}/{n} seeds: GATED basin width narrower than"
                        f" ABLATED by >= {NARROW_REL_THRESH:.0%}"
                    ),
                },
            ],
        },
        "summary": {
            "n_seeds": n,
            "n_p0_pass": n_p0,
            "n_p0a_pass": n_p0a,
            "n_p0b_pass": n_p0b,
            "n_c1_pass": n_c1,
            "n_c2_pass": n_c2,
            "majority_required": majority,
            "p0_pass": p0_pass,
            "p0a_pass": p0a_pass_agg,
            "p0b_pass": p0b_pass_agg,
            "c1_pass": c1_pass,
            "c2_pass": c2_pass,
        },
        "seed_records": seed_records,
        "config": full_config,
        "notes": (
            "903a: instrumented re-run of V3-EXQ-903 (supersedes it). Adds"
            " per-seed / per-condition / positive-control episode-length +"
            " termination-cause recording (condition_stats[*].mean_episode_length"
            " / .termination_causes, seed_records[*].posctrl_episode_stats,"
            " .n_posctrl_valence_samples) to surface the seed-7 short-horizon"
            " sampling confound the 2026-08-09 cluster autopsy named (seed 7's P0"
            " pass coincided with ~25x fewer eval samples). Seed count 3->5"
            " (majority 3/5), POSCTRL 5->10, to reduce P0a correlation volatility."
            " Mechanism/signal/terrain/criteria unchanged. --- "
            "First attempt at the MECH-075 VENTRAL leg (VTA-mediated RPE gain on"
            " motivational terrain). Uses the real signed ARC-108 delta_t signal"
            " (e3_selector.post_action_update, use_habenula_decommit=True) --"
            " NOT the unsigned ARC-016/MECH-043 precision channel. CEM noise"
            " routed through the real MECH-267 mode-conditioning mechanism"
            " (HippocampalConfig.mode_noise_scale), not a hand-rolled patch of"
            " the CEM sampling internals. Direction is OPPOSITE the dorsal"
            " sibling: narrowing, not widening. Single-arm two-condition"
            " design (VTA_RPE_GATED vs VTA_RPE_ABLATED), one trained agent per"
            " seed, shared warmup grounds benefit_eval_head/harm_eval_head via"
            " compute_benefit_eval_loss (agent.py, existing) and a local"
            " harm-side analog (_compute_harm_eval_loss, this script -- no"
            " agent.py counterpart exists)."
        ),
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
    )

    print("\n[V3-EXQ-903] Results", flush=True)
    for r in seed_records:
        print(
            f"  seed={r['seed']}: P0={'P' if r['p0_pass'] else 'F'}"
            f" C1={'P' if r['c1_pass'] else 'F'} C2={'P' if r['c2_pass'] else 'F'}"
            f" gated_basin={r['gated_basin_width']:.5f}"
            f" ablated_basin={r['ablated_basin_width']:.5f}"
            f" narrowing={r['relative_narrowing']:+.4f}",
            flush=True,
        )
    print(f"  outcome={outcome} label={label} evidence_direction={evidence_direction}", flush=True)

    return manifest, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, run_id = run_experiment(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        out_dir=None,
        dry_run=args.dry_run,
        script_path=Path(__file__),
        stamp=False,  # already stamped above
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {manifest['outcome']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        run_id=run_id,
        dry_run=args.dry_run,
    )
