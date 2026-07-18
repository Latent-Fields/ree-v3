"""V3-EXQ-777a: MECH-063 orthogonal control-plane axes -- behavioural rank-2 falsifier.

SUPERSEDES V3-EXQ-777 (same scientific question; implementation fix only).

Claim:    MECH-063 (control plane retains orthogonal axes rather than collapsing
          into one scalar).
Proposal: EVB-0204 (IGW-20260717-203).
Purpose:  evidence (tests the MECH-063 hypothesis directly).

WHY 777a EXISTS -- what went wrong in 777 (see
REE_assembly/evidence/planning/failure_autopsy_MECH-063-777-779-cluster_2026-07-18.md,
adjudicated ARTIFACT; 777's `weakens` reclassified `non_contributory`):

  777 self-routed `control_axes_collinear_toward_one_scalar` / `weakens`, but the
  numbers do not say that. mean_sin_angle was 0.530, EXCEEDING the 0.500 margin.
  C1 fell on seed-count (3 of 5 vs min_seeds 4) and on dispersion
  (mean_sin - sd_sin = 0.177), not on collinearity. Underneath that:

  (1) The score axis's dependent variable D_action_mass_mean was SATURATED --
      pinned at ceiling on seeds 11/29/37 (0.94-0.9997) and at floor on seed 23
      (0.002-0.013). dD_score tracks D-headroom (largest, 0.054, on seed 17, the
      only mid-range seed). With no headroom the score axis can only move
      entropy, i.e. it degenerates into a weak copy of the temperature axis --
      which is exactly what a low sin_angle reports. The apparent "collinearity"
      was a CEILING/FLOOR artifact on the readout, not a property of the control
      plane. 777 guarded saturation on the TEMPERATURE axis's readout
      (E_SAT_CEIL) and had no equivalent guard on the SCORE axis's; R4's
      `D_action_mass_std > 0.0` is a non-degeneracy test, not a saturation test.
  (2) R5 axis_authority aggregated norm_v_score by MEAN across seeds
      (0.1046 > 0.05 floor) while seeds 23 (0.0423) and 29 (0.0346) were
      individually BELOW the floor. The gate built to catch the false-collapse
      trap was defeated by its own aggregator and waved the run through.
  (3) 3 of 5 seeds ran at 2-35% of the fixed 900-step budget (untrained agent,
      hazard-terminating env, `if r.done: break` with no continuation). The worst
      cell sat EXACTLY at MIN_SELECTS = 20. Episode-denominated budgets do not
      control sample size here: yield varied 40x across one seed set and
      reproduced almost exactly in V3-EXQ-779.

THE FOUR FIXES (this script):

  F1 SAMPLE-DRIVEN STOPPING. Each cell runs until it has collected
     TARGET_SELECTS fresh E3 selections, auto-resetting across episodes, bounded
     by MAX_ENV_STEPS_PER_CELL / MAX_EPISODES_PER_CELL. Realised env-step and
     episode counts are recorded per cell. The budget is now denominated in the
     same unit as the gates (selections), which is what 777 lacked.
  F2 SATURATION GUARD ON THE SCORE AXIS'S DV. D_SAT_LOW / D_SAT_HIGH bracket
     D_action_mass_mean, mirroring E_SAT_CEIL on entropy. A seed whose D is
     pinned at ceiling or floor is reported NON-INFORMATIVE for the score axis
     and contributes NO angle to C1, rather than contributing a spurious ~0.
  F3 PER-SEED AUTHORITY GATING. R5 tests ||v_temp|| and ||v_score|| > EFFECT_FLOOR
     PER SEED (never a mean across seeds), and reports which seeds pass.
  F4 RE-EXAMINED SEED BAR, AND AN EXPANDED SEED POOL. The fixed
     `min_seeds = 4 of 5` was arithmetically out of reach once 3 seeds ran short.
     C1 is now evaluated over the INFORMATIVE pool (unsaturated AND
     authority-met) with a fractional bar (C1_SEED_FRAC of the pool, min
     MIN_INFORMATIVE_SEEDS in the pool for the run to be scoreable at all), and
     per-seed saturation state is recorded alongside sin_angle. Critically, once
     F2 excludes saturated seeds the binding constraint becomes POOL SIZE, not
     the bar: on 777's evidence only ~2 of its 5 seeds had score-axis headroom,
     which would self-route dv_saturation_requeue and test nothing a second time.
     So the seed pool is expanded 5 -> 14 (see SEEDS). Growing the pool is the
     correct lever because saturation is a per-seed property of the untrained
     agent's candidate distribution; LOOSENING the guard instead would re-admit
     the ceiling-pinned cells that produced 777's false collinearity signature.

  Consequence for routing: `weakens` is now reachable ONLY from a pool of seeds
  where the score axis demonstrably had room to move and demonstrably had
  authority. A saturated or starved run self-routes to a distinct requeue label
  (`dv_saturation_requeue` / `sample_starvation_requeue`) instead of a claim
  verdict -- the 777 failure mode is structurally unreachable.

VALIDATION NOTE (777 autopsy learning 5): a SINGLE-SEED inline pre-queue check is
NOT valid validation for this probe family. 777's seed-11 inline PASS
(sin_angle 0.93) and its seed-11 cloud FAIL (0.019) came from IDENTICAL code
(one commit, 3b270a1) and IDENTICAL substrate; they differ only in which
saturation regime seed 11 landed in on that machine class (darwin-arm64 vs
linux-x86_64). The verdict is knife-edge on per-seed saturation regime, and that
regime is machine-class sensitive. Validation must run the full seed set and
report the DV's saturation state per seed -- which is what F2 now records.

MECHANISM UNDER TEST (unchanged from 777): two independently-toggleable REAL
production regulators that route to two DIFFERENT parameters of the same E3
softmax at the select() call site (probs = softmax(-(scores + score_bias) / T)):

  SCORE AXIS  -- MECH-320 tonic_vigor: an additive per-candidate directional
                 bias (dacc_score_bias) favouring action over no-op. Shifts the
                 LOCATION (mode) of the E3 softmax over candidates.
  TEMPERATURE -- MECH-313 noise_floor: lifts the E3 softmax TEMPERATURE
     AXIS        (effective_temperature). Scales the SPREAD of the softmax.

WHY A RANK-2 TEST, NOT A SCALAR DOUBLE-DISSOCIATION: with a (bias, temperature)
softmax, NO single scalar summary is cleanly owned by one axis -- both knobs
perturb any marginal statistic. The identifiable, faithful statement of "not one
scalar" is that the two axes move a 2-D policy-readout vector in linearly
independent directions. So we measure a 2-vector readout per arm and test the
rank of the 2x2 main-effect matrix.

DESIGN: 2x2 factorial telemetry probe, NO gradient training (the regulators act
at selection time; tonic_vigor is forced live with tonic_vigor_v_t_floor so the
score axis does not depend on EWMA charging -- the V3-EXQ-563 / 650 pattern).
  Factor A (SCORE axis)      : use_tonic_vigor in {OFF, ON}
  Factor B (TEMPERATURE axis): use_noise_floor  in {OFF, ON}
  4 arms x SEEDS seeds. Shared substrate settings across ALL arms:
    use_control_vector_logging=True (read-only telemetry; bit-identical
    behaviour -- V3-EXQ-650 C3), hippocampal.use_action_class_scaffold_candidates
    =True (a no-op-class candidate is always present for the score axis to act
    on), same env.

READOUTS (per fresh E3 selection; candidates captured via a read-only wrapper on
agent.generate_trajectories, aligned with agent.e3.last_precommit_probs):
  E = normalised Shannon entropy of last_precommit_probs (spread; nats / ln K).
  D = precommit probability MASS on non-no-op candidates (action preference).

AGGREGATION (per seed): cell means (E, D) for the 4 arms; standardise E and D by
their pooled across-cell SD -- computed over the UNSATURATED cells only, so a
floor-pinned or ceiling-pinned seed cannot inflate the pooled SD and thereby
suppress every other seed's standardised effect (a latent 777 defect). Then form
  v_temp  = mean_B1(E,D) - mean_B0(E,D)      (noise_floor ON - OFF)
  v_score = mean_A1(E,D) - mean_A0(E,D)      (tonic_vigor ON - OFF)
  s = |det[v_temp, v_score]| / (||v_temp|| * ||v_score||)  = |sin(angle)| in [0,1].
s ~ 1 => orthogonal axes (rank-2, supports MECH-063).
s ~ 0 => parallel effect vectors (one scalar, weakens MECH-063).

P0 READINESS (self-routes a REQUEUE label -- NEVER a claim verdict):
  R1 score axis live    : A1 arms mean v_t > 0 and mean n_noop_candidates >= 1.
  R2 temperature live   : B1 arms mean noise_floor_temp_lift >= TEMP_LIFT_FLOOR.
  R3 sample sufficiency : every cell has >= MIN_SELECTS fresh E3 selections
                          (F1; offending cells reported by seed+arm).
  R4 entropy headroom   : baseline arm E < E_SAT_CEIL.
  R4b D headroom        : >= MIN_INFORMATIVE_SEEDS seeds have D_seed_mean strictly
                          inside (D_SAT_LOW, D_SAT_HIGH) (F2).
  R5 axis authority     : >= MIN_INFORMATIVE_SEEDS unsaturated seeds have BOTH
                          ||v_temp|| and ||v_score|| > EFFECT_FLOOR, tested
                          PER SEED, reported per seed (F3).

VERDICT (pre-registered constants; not derived post-hoc):
  readiness unmet                    -> FAIL / non_contributory (distinct requeue
                                        label naming the actual binding cause).
  readiness met AND C1 (s > SIN_MARGIN on >= ceil(C1_SEED_FRAC * n_informative)
     informative seeds, and robust: mean_s - SD_s > SIN_MARGIN over that pool)
                                     -> PASS / supports (rank-2: NOT one scalar).
  readiness met AND C1 unmet         -> FAIL / weakens (effect vectors parallel
                                        on a pool with demonstrated headroom and
                                        demonstrated authority).

SHARED _lib ROLLOUT HELPER: F1's stopping rule is NOT reimplemented here. It is
delegated to experiments/_lib/sample_driven_rollout.run_cell_until_samples --
routing (c) of the same autopsy, landed on ree-v3 main as aae239b -- so this
probe and V3-EXQ-779a share one implementation and the whole 2x2 read-only
telemetry-probe family inherits the fix. `_collect_cell_samples` below keeps only
the probe-specific concerns the helper deliberately does not own: what to
accumulate per tick (the E/D readouts) and progress reporting.

MECH-094: trains nothing, writes no memory, no replay -- N/A (waking select only).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.sample_driven_rollout import (  # noqa: E402
    RolloutBudget,
    TickContext,
    run_cell_until_samples,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_777a_mech063_orthogonal_control_axes_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-063"]
SUPERSEDES = "V3-EXQ-777"

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----
# F4: the SEED POOL is expanded from 777's 5 seeds to 14. Rationale: once F2
# excludes saturated seeds, the binding constraint on C1 stops being the bar and
# becomes the POOL SIZE. On 777's evidence only seeds 17 (D 0.45-0.53) and
# arguably 37 (0.82-0.95) had score-axis headroom -- an informative pool of ~2,
# below MIN_INFORMATIVE_SEEDS, which would self-route dv_saturation_requeue and
# test nothing for a second time. Saturation is a per-seed property of the
# untrained agent's candidate distribution, so the fix is to SAMPLE MORE SEEDS,
# not to loosen the guard (loosening it would re-admit exactly the ceiling-pinned
# cells that produced 777's false collinearity signature). At the ~2-of-5
# headroom rate observed in 777, 14 seeds gives an expected pool of ~5-6.
# Affordable only because F1 made each cell far cheaper than 777's 900-step
# budget. The first five entries are 777's seeds, kept in order so per-seed
# comparison against the 777 manifest is direct.
SEEDS = [11, 17, 23, 29, 37, 3, 5, 13, 19, 41, 53, 61, 71, 83]

# F1: sample-driven stopping. The budget is denominated in E3 SELECTIONS (the
# same unit the gates use), not episodes. 777's fixed 3 x 300 episode budget
# yielded 20-900 steps depending on seed (40x spread).
TARGET_SELECTS = 250            # collect this many fresh E3 selections per cell
MIN_SELECTS = 200               # R3 readiness floor per cell (was 20 in 777)
MAX_ENV_STEPS_PER_CELL = 4000   # runtime bound (starved seeds reset repeatedly)
MAX_EPISODES_PER_CELL = 400     # runtime bound (seed 23 died in ~7 steps/episode)
STEPS_PER_EPISODE = 300         # per-episode cap before a forced reset

# Progress reporting: sample-driven stopping has no fixed episode denominator,
# so progress is reported in PROGRESS_TICKS checkpoints of TARGET_SELECTS. This
# is the loop-bound denominator the runner reads (`ep k/PROGRESS_TICKS`).
PROGRESS_TICKS = 10

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
NOOP_CLASS = 4                # CausalGridWorldV2 ACTIONS[4] = (0,0) = stay/no-op
ACTION_DIM = 5                # CausalGridWorldV2 ACTIONS = {0..3 moves, 4 stay}

# Score axis (MECH-320 tonic_vigor) -- forced live via v_t_floor.
TV_V_T_FLOOR = 0.5
TV_W_ACTION = 0.5
TV_W_PASSIVE = 0.5
TV_BIAS_SCALE = 1.0
# Temperature axis (MECH-313 noise_floor). Baseline E3 temperature is 1.0
# (StepHarness passes temperature=1.0); effective = max(1.0 + alpha, min_T) = 2.0.
NF_ALPHA = 1.0
NF_MIN_T = 2.0

# Readiness thresholds.
TEMP_LIFT_FLOOR = 0.5        # R2: effective_T - baseline_T in noise_floor arms
E_SAT_CEIL = 0.98            # R4: baseline normalised entropy must leave headroom
# F2: the guard 777 lacked. D_action_mass_mean pinned at ceiling/floor leaves the
# score axis no room to express itself. Calibrated against the 777 per-seed table:
# seed 11 (0.997) and 29 (0.94-0.98) are excluded as ceiling-pinned, seed 23
# (0.002-0.013) as floor-pinned; seed 17 (0.45-0.53) and 37 (0.82-0.95) retained.
D_SAT_LOW = 0.05
D_SAT_HIGH = 0.95
EFFECT_FLOOR = 0.05          # R5: min standardised main-effect norm, PER SEED
MIN_INFORMATIVE_SEEDS = 3    # R4b/R5: pool size below which the run is unscoreable

# Verdict thresholds.
SIN_MARGIN = 0.5             # C1: |sin(angle)| between effect vectors (~30 deg)
C1_SEED_FRAC = 0.75          # F4: fraction of the INFORMATIVE pool C1 must hold on

ARMS = ["A0B0", "A1B0", "A0B1", "A1B1"]  # (tonic_vigor, noise_floor)
_ARM_FLAGS = {
    "A0B0": (False, False),
    "A1B0": (True, False),
    "A0B1": (False, True),
    "A1B1": (True, True),
}


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE, num_hazards=ENV_HAZARDS, num_resources=ENV_RESOURCES
    )


def _build_config(arm: str, env: CausalGridWorldV2) -> REEConfig:
    use_tv, use_nf = _ARM_FLAGS[arm]
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    # Shared substrate operating settings (identical across all four arms).
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    # Score axis.
    cfg.use_tonic_vigor = use_tv
    if use_tv:
        cfg.tonic_vigor_v_t_floor = TV_V_T_FLOOR
        cfg.tonic_vigor_w_action = TV_W_ACTION
        cfg.tonic_vigor_w_passive = TV_W_PASSIVE
        cfg.tonic_vigor_bias_scale = TV_BIAS_SCALE
        cfg.tonic_vigor_noop_class = NOOP_CLASS
        cfg.tonic_vigor_form = "additive"
    # Temperature axis.
    cfg.use_noise_floor = use_nf
    if use_nf:
        cfg.noise_floor_alpha = NF_ALPHA
        cfg.noise_floor_min_temperature = NF_MIN_T
    return cfg


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + this arm's
    control flags + the sample-driven stopping rule (which determines the
    cell's computation and therefore MUST be declared -- it differs from 777,
    so 777's mints correctly refuse to match)."""
    use_tv, use_nf = _ARM_FLAGS[arm]
    sl: Dict[str, Any] = {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "action_dim": ACTION_DIM,
        "stopping_rule": "sample_driven_selects",
        "target_selects": TARGET_SELECTS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_control_vector_logging": True,
        "use_action_class_scaffold_candidates": True,
        "use_tonic_vigor": use_tv,
        "use_noise_floor": use_nf,
    }
    if use_tv:
        sl.update(
            tonic_vigor_v_t_floor=TV_V_T_FLOOR,
            tonic_vigor_w_action=TV_W_ACTION,
            tonic_vigor_w_passive=TV_W_PASSIVE,
            tonic_vigor_bias_scale=TV_BIAS_SCALE,
            tonic_vigor_noop_class=NOOP_CLASS,
            tonic_vigor_form="additive",
        )
    if use_nf:
        sl.update(noise_floor_alpha=NF_ALPHA, noise_floor_min_temperature=NF_MIN_T)
    return sl


def _norm_entropy(probs: torch.Tensor) -> Optional[float]:
    """Normalised Shannon entropy (nats / ln K) of a candidate softmax."""
    p = probs.detach().reshape(-1).float()
    k = int(p.numel())
    if k < 2:
        return None
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return h / math.log(k)


def _candidate_noop_mask(candidates: Any) -> Optional[torch.Tensor]:
    """Per-candidate boolean mask: True where the first-action class == NOOP.
    First-action class = argmax over action_dim of candidate.actions[:, 0, :]."""
    classes: List[int] = []
    for c in candidates:
        acts = getattr(c, "actions", None)
        if acts is None:
            return None
        a = acts.reshape(-1, acts.shape[-1])[0]  # first-step action vector
        classes.append(int(a.argmax().item()))
    if not classes:
        return None
    return torch.tensor(classes) == NOOP_CLASS


def _collect_cell_samples(
    agent: Any,
    env: Any,
    captured: Dict[str, Any],
    seed: int,
    arm: str,
    target_selects: int,
    max_env_steps: int,
    max_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """F1: SAMPLE-DRIVEN cell rollout.

    Delegates the stopping rule to the shared family helper
    `experiments/_lib/sample_driven_rollout.run_cell_until_samples` (routing (c)
    of the 777/779 autopsy, landed on ree-v3 main as aae239b). The helper runs
    until the declared sample floor -- denominated in fresh E3 SELECTIONS, the
    same unit as this probe's readiness gates -- is met, AUTO-RESETTING the env
    across episodes (the fix for 777's `if r.done: break` with no continuation),
    bounded by `max_env_steps` / `max_episodes`, and returns the REALISED step,
    episode and per-readout counts plus an explicit stop reason.

    This function keeps only the probe-specific concerns the helper deliberately
    does NOT own: what to accumulate per tick, and progress reporting.
    """
    harness = StepHarness(agent, env, train_mode=True, seed=seed)

    e_vals: List[float] = []
    d_vals: List[float] = []
    vt_vals: List[float] = []
    templift_vals: List[float] = []
    nnoop_vals: List[int] = []
    progress = {"next": 1}

    def _observe(ctx: TickContext) -> Optional[Dict[str, int]]:
        """Accumulate this probe's readouts for one tick; count selections."""
        probs = ctx.probs
        cands = captured.get("cands")
        if not (ctx.fresh and cands is not None and probs is not None):
            return None
        if len(cands) != int(probs.numel()):
            return None
        ent = _norm_entropy(probs)
        mask = _candidate_noop_mask(cands)
        if ent is None or mask is None:
            return None

        e_vals.append(ent)
        pp = probs.detach().reshape(-1).float()
        d_vals.append(float(pp[~mask].sum().item()))  # action mass
        nnoop_vals.append(int(mask.sum().item()))
        cv = agent._last_control_vector or {}
        sh = cv.get("shared", {})
        gv = cv.get("G_vigor", {})
        vt_vals.append(float(sh.get("tonic_vigor_v_t", 0.0)))
        templift_vals.append(float(gv.get("noise_floor_temp_lift", 0.0)))

        # Progress checkpoint (loop-bound denominator PROGRESS_TICKS).
        n_sel = len(e_vals)
        while (
            progress["next"] <= PROGRESS_TICKS
            and n_sel >= (target_selects * progress["next"]) // PROGRESS_TICKS
        ):
            print(
                f"  [train] {arm} seed={seed} ep {progress['next']}/{PROGRESS_TICKS} "
                f"selects={n_sel}/{target_selects} env_steps={ctx.n_env_steps} "
                f"episodes={ctx.episode_index + 1}",
                flush=True,
            )
            progress["next"] += 1
        return {"selections": 1}

    outcome = run_cell_until_samples(
        env=env,
        agent=agent,
        harness=harness,
        budget=RolloutBudget(
            sample_floors={"selections": target_selects},
            max_env_steps=max_env_steps,
            steps_per_episode=steps_per_episode,
            max_episodes=max_episodes,
        ),
        observe=_observe,
        progress_label=f"{arm} seed={seed}",
    )

    # Emit any remaining progress checkpoints so the runner's bar completes even
    # when a cell stopped on the step/episode cap rather than on the floor.
    while progress["next"] <= PROGRESS_TICKS:
        print(
            f"  [train] {arm} seed={seed} ep {progress['next']}/{PROGRESS_TICKS} "
            f"selects={len(e_vals)}/{target_selects} env_steps={outcome.n_env_steps} "
            f"episodes={outcome.n_episodes} (stopped on {outcome.stop_reason})",
            flush=True,
        )
        progress["next"] += 1

    return {
        "e_vals": e_vals,
        "d_vals": d_vals,
        "vt_vals": vt_vals,
        "templift_vals": templift_vals,
        "nnoop_vals": nnoop_vals,
        "n_e3_selects": len(e_vals),
        "n_env_steps": outcome.n_env_steps,
        "n_episodes": outcome.n_episodes,
        "target_selects": target_selects,
        "stopped_on": outcome.stop_reason,
        "rollout_fields": outcome.as_manifest_fields(),
    }


def _run_cell(arm: str, seed: int, target_selects: int, max_env_steps: int,
              max_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    """One (arm, seed) cell: sample-driven telemetry rollout, no gradient training."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,  # cross-driver reusable mint
    ) as cell:
        env = _mk_env()
        cfg = _build_config(arm, env)
        agent = REEAgent(cfg)

        # Read-only wrapper to capture the candidate list E3 selects over, so the
        # per-candidate no-op mask aligns with agent.e3.last_precommit_probs.
        captured: Dict[str, Any] = {"cands": None}
        _orig_gen = agent.generate_trajectories

        def _gen_capture(*a: Any, **k: Any) -> Any:
            cands = _orig_gen(*a, **k)
            captured["cands"] = cands
            return cands

        agent.generate_trajectories = _gen_capture  # type: ignore[assignment]

        coll = _collect_cell_samples(
            agent, env, captured, seed, arm,
            target_selects=target_selects,
            max_env_steps=max_env_steps,
            max_episodes=max_episodes,
            steps_per_episode=steps_per_episode,
        )

        def _mean(xs: List[float]) -> float:
            return float(statistics.fmean(xs)) if xs else 0.0

        d_vals = coll["d_vals"]
        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "use_tonic_vigor": _ARM_FLAGS[arm][0],
            "use_noise_floor": _ARM_FLAGS[arm][1],
            # F1: realised counts recorded per cell (the field that made the 777
            # diagnosis possible without a re-run -- now with episodes too).
            "n_env_steps": coll["n_env_steps"],
            "n_episodes_realised": coll["n_episodes"],
            "n_e3_selects": coll["n_e3_selects"],
            "stopped_on": coll["stopped_on"],
            # Realised-yield block as returned by the shared _lib rollout helper.
            "rollout": coll["rollout_fields"],
            "E_norm_entropy_mean": _mean(coll["e_vals"]),
            "D_action_mass_mean": _mean(d_vals),
            "D_action_mass_std": float(statistics.pstdev(d_vals)) if len(d_vals) > 1 else 0.0,
            "vt_mean": _mean(coll["vt_vals"]),
            "noise_floor_temp_lift_mean": _mean(coll["templift_vals"]),
            "n_noop_candidates_mean": _mean([float(x) for x in coll["nnoop_vals"]]),
            # Per-selection series retained for generous recording / reanalysis.
            "E_series": coll["e_vals"],
            "D_series": d_vals,
        }
        cell.stamp(row)
    return row


# ----------------------------------------------------------------------
# Aggregation: 2x2 rank-2 non-collinearity of the control-axis effect vectors.
# ----------------------------------------------------------------------
def _pooled_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _saturation_regime(d_seed_mean: float) -> str:
    if d_seed_mean >= D_SAT_HIGH:
        return "ceiling"
    if d_seed_mean <= D_SAT_LOW:
        return "floor"
    return "headroom"


def _seed_analysis(rows_by_arm: Dict[str, Dict[str, Any]], e_sd: float, d_sd: float) -> Dict[str, Any]:
    """Standardise (E, D), form the two main-effect vectors, compute |sin angle|."""
    def _std_ed(arm: str) -> Tuple[float, float]:
        r = rows_by_arm[arm]
        e = r["E_norm_entropy_mean"] / e_sd if e_sd > 0 else 0.0
        d = r["D_action_mass_mean"] / d_sd if d_sd > 0 else 0.0
        return (e, d)

    e00, d00 = _std_ed("A0B0")
    e10, d10 = _std_ed("A1B0")
    e01, d01 = _std_ed("A0B1")
    e11, d11 = _std_ed("A1B1")

    # Temperature main effect (noise_floor ON - OFF), averaged over factor A.
    vt_e = ((e01 + e11) / 2.0) - ((e00 + e10) / 2.0)
    vt_d = ((d01 + d11) / 2.0) - ((d00 + d10) / 2.0)
    # Score main effect (tonic_vigor ON - OFF), averaged over factor B.
    vs_e = ((e10 + e11) / 2.0) - ((e00 + e01) / 2.0)
    vs_d = ((d10 + d11) / 2.0) - ((d00 + d01) / 2.0)

    norm_t = math.hypot(vt_e, vt_d)
    norm_s = math.hypot(vs_e, vs_d)
    det = vt_e * vs_d - vt_d * vs_e
    sin_angle = abs(det) / (norm_t * norm_s) if (norm_t > 0 and norm_s > 0) else 0.0

    d_seed_mean = float(statistics.fmean(
        [rows_by_arm[a]["D_action_mass_mean"] for a in ARMS]
    ))
    regime = _saturation_regime(d_seed_mean)

    return {
        "v_temp": [vt_e, vt_d],
        "v_score": [vs_e, vs_d],
        "norm_v_temp": norm_t,
        "norm_v_score": norm_s,
        "sin_angle": sin_angle,
        # F2: per-seed DV saturation state, recorded alongside sin_angle.
        "D_seed_mean": d_seed_mean,
        "D_arm_means": {a: rows_by_arm[a]["D_action_mass_mean"] for a in ARMS},
        "saturation_regime": regime,
        "d_saturated": bool(regime != "headroom"),
        # F3: per-seed authority, never a mean across seeds.
        "authority_met": bool(norm_t > EFFECT_FLOOR and norm_s > EFFECT_FLOOR),
        "min_n_e3_selects": min(rows_by_arm[a]["n_e3_selects"] for a in ARMS),
        # Raw (unstandardised) marginal deltas for interpretability.
        "dE_temp": ((rows_by_arm["A0B1"]["E_norm_entropy_mean"] + rows_by_arm["A1B1"]["E_norm_entropy_mean"]) / 2.0)
        - ((rows_by_arm["A0B0"]["E_norm_entropy_mean"] + rows_by_arm["A1B0"]["E_norm_entropy_mean"]) / 2.0),
        "dD_score": ((rows_by_arm["A1B0"]["D_action_mass_mean"] + rows_by_arm["A1B1"]["D_action_mass_mean"]) / 2.0)
        - ((rows_by_arm["A0B0"]["D_action_mass_mean"] + rows_by_arm["A0B1"]["D_action_mass_mean"]) / 2.0),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    target_selects = 12 if dry_run else TARGET_SELECTS
    max_env_steps = 200 if dry_run else MAX_ENV_STEPS_PER_CELL
    max_episodes = 20 if dry_run else MAX_EPISODES_PER_CELL
    min_selects = 8 if dry_run else MIN_SELECTS

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed, target_selects, max_env_steps,
                            max_episodes, STEPS_PER_EPISODE)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_e3_selects'] > 0 else 'FAIL'}", flush=True)

    # ---- F2 step 1: determine saturation from RAW D, before any standardisation.
    # Ordering matters: saturation depends only on raw D_action_mass_mean, so it
    # breaks the circularity (saturation -> pooled SD -> authority -> pool).
    d_by_seed: Dict[int, float] = {}
    for seed in seeds:
        ds = [r["D_action_mass_mean"] for r in rows if r["seed"] == seed]
        if ds:
            d_by_seed[seed] = float(statistics.fmean(ds))
    unsaturated_seeds = [s for s, dm in d_by_seed.items()
                         if _saturation_regime(dm) == "headroom"]

    # ---- F2 step 2: pooled SDs over UNSATURATED cells only. A floor- or
    # ceiling-pinned seed must not inflate the pooled SD and thereby suppress
    # every other seed's standardised effect (a latent 777 defect).
    sd_rows = [r for r in rows if r["seed"] in unsaturated_seeds] or rows
    e_sd = _pooled_std([r["E_norm_entropy_mean"] for r in sd_rows])
    d_sd = _pooled_std([r["D_action_mass_mean"] for r in sd_rows])

    # Per-seed rank-2 analysis (computed for ALL seeds; recorded for all).
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        if len(by_arm) != len(ARMS):
            continue
        a = _seed_analysis(by_arm, e_sd, d_sd)
        a["seed"] = seed
        per_seed.append(a)

    # ---- F2 + F3: the INFORMATIVE pool -- unsaturated AND per-seed authority met.
    informative = [a for a in per_seed if (not a["d_saturated"]) and a["authority_met"]]
    informative_seeds = [a["seed"] for a in informative]
    n_informative = len(informative)

    # ---- Readiness preconditions (self-route a REQUEUE label if unmet) ----
    a1_rows = [r for r in rows if r["use_tonic_vigor"]]
    b1_rows = [r for r in rows if r["use_noise_floor"]]
    baseline_rows = [r for r in rows if r["arm"] == "A0B0"]

    r1_score_live = bool(a1_rows) and all(
        r["vt_mean"] > 0.0 and r["n_noop_candidates_mean"] >= 1.0 for r in a1_rows
    )
    r2_temp_live = bool(b1_rows) and all(
        r["noise_floor_temp_lift_mean"] >= TEMP_LIFT_FLOOR for r in b1_rows
    )
    # F1/R3: sample sufficiency, with the offending cells named (777 reported
    # only a bare min, which sent the 779 reader hunting a substrate fault).
    starved_cells = [
        {"seed": r["seed"], "arm": r["arm"], "n_e3_selects": r["n_e3_selects"],
         "n_env_steps": r["n_env_steps"], "n_episodes_realised": r["n_episodes_realised"],
         "stopped_on": r["stopped_on"]}
        for r in rows if r["n_e3_selects"] < min_selects
    ]
    r3_samples = bool(rows) and not starved_cells
    r4_headroom = bool(baseline_rows) and all(
        r["E_norm_entropy_mean"] < E_SAT_CEIL for r in baseline_rows
    )
    # F2/R4b: enough seeds with score-axis DV headroom.
    r4b_d_headroom = len(unsaturated_seeds) >= min(MIN_INFORMATIVE_SEEDS, len(seeds))
    # F3/R5: per-seed authority, counted over unsaturated seeds.
    r5_authority = n_informative >= min(MIN_INFORMATIVE_SEEDS, len(seeds))

    readiness_met = bool(
        r1_score_live and r2_temp_live and r3_samples
        and r4_headroom and r4b_d_headroom and r5_authority
    )

    # ---- F4 + load-bearing criterion C1: rank-2 non-collinearity, on the pool ----
    sins = [a["sin_angle"] for a in informative]
    seeds_c1 = sum(1 for s in sins if s > SIN_MARGIN)
    mean_sin = statistics.fmean(sins) if sins else 0.0
    sd_sin = _pooled_std(sins)
    min_seeds_required = (
        max(MIN_INFORMATIVE_SEEDS, math.ceil(C1_SEED_FRAC * n_informative))
        if n_informative else 0
    )
    c1_seed_count = bool(n_informative and seeds_c1 >= min_seeds_required)
    c1_robust = (mean_sin - sd_sin) > SIN_MARGIN  # effect exceeds its own noise
    c1 = bool(c1_seed_count and c1_robust)

    non_degenerate = True
    degeneracy_reason = None

    if not (r1_score_live and r2_temp_live):
        outcome, direction = "FAIL", "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = "a control axis was not exercised (R1/R2 unmet)"
    elif not r3_samples:
        # Distinct from a substrate-capability failure -- the 779 mislabel lesson.
        outcome, direction = "FAIL", "non_contributory"
        label = "sample_starvation_requeue"
        non_degenerate = False
        degeneracy_reason = (
            f"{len(starved_cells)} cell(s) below MIN_SELECTS={min_selects} "
            "(see interpretation.starved_cells)"
        )
    elif not r4_headroom:
        outcome, direction = "FAIL", "non_contributory"
        label = "entropy_saturation_requeue"
        non_degenerate = False
        degeneracy_reason = "baseline normalised entropy at ceiling (R4 unmet)"
    elif not r4b_d_headroom:
        # F2: the 777 failure mode, now a requeue rather than a claim verdict.
        outcome, direction = "FAIL", "non_contributory"
        label = "dv_saturation_requeue"
        non_degenerate = False
        degeneracy_reason = (
            f"only {len(unsaturated_seeds)} seed(s) have D_action_mass in "
            f"({D_SAT_LOW}, {D_SAT_HIGH}); the score axis had no headroom"
        )
    elif not r5_authority:
        # F3: per-seed authority, not a mean that masks per-seed axis death.
        outcome, direction = "FAIL", "non_contributory"
        label = "axis_authority_starvation_requeue"
        non_degenerate = False
        degeneracy_reason = (
            f"only {n_informative} unsaturated seed(s) met per-seed authority "
            f"(EFFECT_FLOOR={EFFECT_FLOOR})"
        )
    elif c1:
        outcome, direction = "PASS", "supports"
        label = "control_axes_orthogonal_rank2"
    else:
        outcome, direction = "FAIL", "weakens"
        label = "control_axes_collinear_toward_one_scalar"

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "score_axis_live", "control": "A1 arms: mean v_t and no-op candidate count",
             "measured": (statistics.fmean([r["vt_mean"] for r in a1_rows]) if a1_rows else 0.0),
             "threshold": 0.0, "met": bool(r1_score_live)},
            {"name": "temperature_axis_live", "control": "B1 arms: effective_T - baseline_T",
             "measured": (statistics.fmean([r["noise_floor_temp_lift_mean"] for r in b1_rows]) if b1_rows else 0.0),
             "threshold": TEMP_LIFT_FLOOR, "met": bool(r2_temp_live)},
            {"name": "sample_sufficiency", "control": "min fresh E3 selections over cells (sample-driven stopping)",
             "measured": (min(r["n_e3_selects"] for r in rows) if rows else 0),
             "threshold": min_selects, "met": bool(r3_samples)},
            {"name": "baseline_entropy_headroom", "control": "A0B0 normalised entropy below ceiling",
             "measured": (statistics.fmean([r["E_norm_entropy_mean"] for r in baseline_rows]) if baseline_rows else 1.0),
             "threshold": E_SAT_CEIL, "direction": "upper", "met": bool(r4_headroom)},
            {"name": "score_dv_headroom_seeds",
             "control": ("seeds whose D_action_mass_mean is strictly inside "
                         f"({D_SAT_LOW}, {D_SAT_HIGH}) -- the guard V3-EXQ-777 lacked"),
             "measured": len(unsaturated_seeds),
             "threshold": min(MIN_INFORMATIVE_SEEDS, len(seeds)), "met": bool(r4b_d_headroom)},
            {"name": "axis_authority_seeds_per_seed",
             "control": ("unsaturated seeds with BOTH ||v_temp|| and ||v_score|| > EFFECT_FLOOR, "
                         "tested PER SEED (V3-EXQ-777 used a mean across seeds)"),
             "measured": n_informative,
             "threshold": min(MIN_INFORMATIVE_SEEDS, len(seeds)), "met": bool(r5_authority)},
        ],
        "criteria": [
            {"name": "C1_rank2_non_collinearity", "load_bearing": True, "passed": bool(c1)},
        ],
        "criteria_non_degenerate": {
            "C1_rank2_non_collinearity": bool(n_informative >= min(MIN_INFORMATIVE_SEEDS, len(seeds))),
        },
        "starved_cells": starved_cells,
        "informative_seeds": informative_seeds,
        "saturation_by_seed": {
            str(a["seed"]): {"D_seed_mean": a["D_seed_mean"],
                             "regime": a["saturation_regime"],
                             "authority_met": a["authority_met"],
                             "norm_v_score": a["norm_v_score"],
                             "norm_v_temp": a["norm_v_temp"],
                             "sin_angle": a["sin_angle"]}
            for a in per_seed
        },
        "summary": (
            "MECH-063 rank-2 test on the production control plane (V3-EXQ-777a, "
            "supersedes V3-EXQ-777). Two independently-toggled regulators route to "
            "two E3-softmax parameters (score-bias vs temperature). PASS = their "
            "standardised behavioural effect vectors are non-collinear on a pool of "
            "seeds with demonstrated DV headroom and demonstrated per-seed axis "
            "authority (rank-2, control is NOT one scalar). FAIL/weakens = parallel "
            "on that same clean pool. FAIL/non_contributory = the run could not "
            "test the claim, with the binding cause named by the label "
            "(sample_starvation / dv_saturation / axis_authority_starvation / "
            "substrate_not_ready). The 777 artifact -- a saturated score-axis DV "
            "read as collinearity -- is structurally unreachable here."
        ),
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": direction,
        "supersedes": SUPERSEDES,
        "backlog_id": "EVB-0204",
        "proposal_id": "EVB-0204",
        "dry_run": bool(dry_run),
        "non_degenerate": non_degenerate,
        "interpretation": interpretation,
        "acceptance": {
            "readiness_met": readiness_met,
            "C1_rank2_non_collinearity": c1,
            "c1_seed_count": seeds_c1,
            "c1_seed_count_required": min_seeds_required,
            "c1_robust": c1_robust,
            "n_informative_seeds": n_informative,
            "informative_seeds": informative_seeds,
            "unsaturated_seeds": sorted(unsaturated_seeds),
            "mean_sin_angle": mean_sin,
            "sd_sin_angle": sd_sin,
            "SIN_MARGIN": SIN_MARGIN,
            "C1_SEED_FRAC": C1_SEED_FRAC,
            "MIN_INFORMATIVE_SEEDS": min(MIN_INFORMATIVE_SEEDS, len(seeds)),
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "readout_standardisation": {
            "E_pooled_sd": e_sd, "D_pooled_sd": d_sd,
            "pooled_over": "unsaturated_cells_only",
            "n_cells_in_pool": len(sd_rows),
        },
        "sample_yield": {
            "per_cell": [
                {"seed": r["seed"], "arm": r["arm"], "n_e3_selects": r["n_e3_selects"],
                 "n_env_steps": r["n_env_steps"], "n_episodes_realised": r["n_episodes_realised"],
                 "stopped_on": r["stopped_on"]}
                for r in rows
            ],
            "target_selects": target_selects,
            "min_selects": min_selects,
        },
        "notes": (
            "MECH-063 orthogonal control-axes rank-2 falsifier, iteration a. "
            "SUPERSEDES V3-EXQ-777, whose weakens self-route was adjudicated an "
            "ARTIFACT by failure_autopsy_MECH-063-777-779-cluster_2026-07-18 "
            "(mean_sin_angle 0.530 EXCEEDED the 0.500 margin; C1 fell on seed-count "
            "3/5 and dispersion, while the score axis's DV was pinned at ceiling on "
            "seeds 11/29/37 and at floor on seed 23). Four fixes: (F1) sample-driven "
            "stopping in E3 SELECTIONS with auto-reset across episodes and recorded "
            "realised step/episode counts, replacing the fixed 3x300 episode budget "
            "that yielded a 40x per-seed spread; (F2) D_SAT_LOW/D_SAT_HIGH saturation "
            "guard on D_action_mass_mean mirroring E_SAT_CEIL, with saturated seeds "
            "excluded from the angle pool and from the pooled-SD standardisation; "
            "(F3) per-seed R5 authority gating replacing the mean-across-seeds test "
            "that masked per-seed axis death on seeds 23/29; (F4) C1 seed bar "
            "re-expressed as a fraction of the INFORMATIVE pool rather than a fixed "
            "4-of-5 that was arithmetically unreachable, WITH the seed pool expanded "
            "5 -> 14 because after F2 excludes saturated seeds the binding constraint "
            "becomes pool size (777 had only ~2 seeds with score-axis headroom, which "
            "would self-route dv_saturation_requeue and test nothing twice); growing "
            "the pool is the right lever since saturation is a per-seed property of "
            "the untrained agent, whereas loosening D_SAT would re-admit the very "
            "ceiling-pinned cells that produced the false collinearity signature. "
            "Requeue labels now name the "
            "binding cause (sample_starvation / dv_saturation / axis_authority_"
            "starvation) rather than conflating sampling with substrate capability "
            "-- the V3-EXQ-779 mislabel lesson. GOV-REUSE-1: the decisive readout "
            "(behavioural rank of the 2-axis effect matrix, on an UNSATURATED pool) "
            "is absent from all recorded manifests -- 777's own manifest records the "
            "readout only on a saturated pool and is the run being superseded; "
            "V3-EXQ-650 measured the WITHIN-MECH-320 C_time/G_vigor collapse "
            "(claim_ids=[]), a different question -> run. Re-derive brake: 1 "
            "non_contributory autopsy on MECH-063, below the threshold of 2 -> not "
            "braked (and this iteration fixes the sampling model rather than nudging "
            "a threshold, per the autopsy's explicit flag)."
        ),
    }
    if non_degenerate is False and degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)

    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "seeds": SEEDS,
        "stopping_rule": "sample_driven_selects",
        "target_selects": TARGET_SELECTS,
        "min_selects": MIN_SELECTS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "progress_ticks": PROGRESS_TICKS,
        "env": {"size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES},
        "noop_class": NOOP_CLASS,
        "tonic_vigor": {"v_t_floor": TV_V_T_FLOOR, "w_action": TV_W_ACTION,
                        "w_passive": TV_W_PASSIVE, "bias_scale": TV_BIAS_SCALE, "form": "additive"},
        "noise_floor": {"alpha": NF_ALPHA, "min_temperature": NF_MIN_T},
        "thresholds": {"MIN_SELECTS": MIN_SELECTS, "TEMP_LIFT_FLOOR": TEMP_LIFT_FLOOR,
                       "E_SAT_CEIL": E_SAT_CEIL, "D_SAT_LOW": D_SAT_LOW,
                       "D_SAT_HIGH": D_SAT_HIGH, "EFFECT_FLOOR": EFFECT_FLOOR,
                       "MIN_INFORMATIVE_SEEDS": MIN_INFORMATIVE_SEEDS,
                       "SIN_MARGIN": SIN_MARGIN, "C1_SEED_FRAC": C1_SEED_FRAC},
        "supersedes": SUPERSEDES,
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=_THIS,
        started_at=t_start,
    )

    # NOTE: use a non-"verdict:" prefix here -- the runner counts "verdict:" lines
    # as completed runs, and the per-cell prints already emit exactly
    # seeds x conditions of them.
    print(f"FINAL_OUTCOME: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  readiness_met: {manifest['acceptance']['readiness_met']}", flush=True)
    print(f"  informative_seeds: {manifest['acceptance']['informative_seeds']}", flush=True)
    print(f"  mean_sin_angle: {manifest['acceptance']['mean_sin_angle']:.3f} "
          f"(margin {SIN_MARGIN})", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    emit_outcome(
        outcome=manifest["outcome"] if manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
