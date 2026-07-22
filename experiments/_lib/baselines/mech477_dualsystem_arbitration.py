"""Canonical cell for the MECH-477 / SD-081 dual-system arbitration lineage.

Arm-reuse Phase 0/1. Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 7b, 9).

WHAT THIS MODULE IS
-------------------
The single source of truth for ONE (seed, arm) cell of the SD-081 arbitrator
OFF-vs-ON falsifier lineage (V3-EXQ-811 -> 811a -> ...). Both arms are built and
run from here, so the OFF arm is a canonical baseline a later, DIFFERENT-DRIVER
iteration can reuse by citing this run's mint (per the /queue-experiment
"Saving a baseline for reuse" recipe: the first experiment of a lineage mints
in-line, with include_driver_script_in_hash=False, and no separate mint job).

WHY THE MEASUREMENT LIVES HERE AND NOT IN THE DRIVER
----------------------------------------------------
The cell's returned values must be a pure function of (substrate, config_slice,
seed) for a cross-driver reuse to be sound. `experiments/_lib/**/*.py` is inside
the arm-fingerprint substrate glob, so everything in this file is content-hashed
and any edit correctly REFUSES a stale reuse. If the probe loop lived in the
driver instead, a consumer with its own driver could silently change how a
"reused" cell's numbers were produced while the fingerprint still matched --
exactly the hazard include_driver_script_in_hash=False otherwise opens. So the
driver here owns only aggregation, thresholds, gating and routing (none of which
enter a cell's values); this module owns everything a cell computes.

THE TWO ARMS
------------
  arb_off -- cfg.e3.use_dualsystem_arbitration = False. The pre-SD-081 selection
             path: an unconditional full-horizon J(zeta), which is what
             V3-EXQ-786a ran. NOT 786a as-run, though -- see HABIT DEPTH below.
  arb_on  -- the same, with the SD-081 arbitrator live at its REGISTERED
             DEFAULTS (gain 4.0, bias 0.0, uncertainty EMA alpha 0.05, habit
             depth 2). The defaults are deliberately not re-tuned: the claim
             under test is about the registered substrate.

Everything else is held identical between the arms, including curiosity (see
below), so the arms differ in exactly one flag.

HABIT DEPTH 2 -- THE 786a DEGENERACY, AND WHY BOTH ARMS MUST BE RE-RUN
-----------------------------------------------------------------------
786a read the myopic pathway as evaluate_trajectory(world_seq[:, :1, :]). Index 0
of the z_world sequence is the CURRENT state, shared by every candidate, so that
vector is CONSTANT -- measured n_unique = 1 across all 32 candidates on every
scored tick under 786a's own config. Its _spearman degeneracy guard could not
fire (it tests the std of the RANKS, and double-argsort of a constant vector is a
permutation of 0..K-1), so the DV measured stable-sort tie-break noise: simulated
mean 1.0173 sd 0.1871 against the manifest's reported 1.01725. That is why the
OFF arm CANNOT be 786a as-run and both arms are freshly measured here at
HABIT_DEPTH = 2 -- the same floor SD-081 enforces inside _arbitrate_dual_system.

Measured on this config after training: depth-2 habit scores are fully distinct
across candidates (distinct-fraction 1.000, cross-candidate range ~1.3e-2), and
recruitment reads ~0.56 familiar / ~0.81 novel rather than ~1.02.

CURIOSITY IS ON IN BOTH ARMS, AND ITS SCORER LEVER IS MEASURABLY INERT
-----------------------------------------------------------------------
cfg.hippocampal.curiosity_weight > 0 is REQUIRED for the arbitrator's PREFERRED
u_habit path: it is the switch that builds hippocampal.familiarity_tracker, and
without that tracker REEAgent.select_action falls back to the E1 novelty EMA.
That fallback was measured on this config and is DEGENERATE here -- u_habit_raw
reads exactly 0.0 on every tick, so u_habit_norm is a constant and the novelty
channel the claim is about never varies. The familiarity path is therefore not a
preference but a precondition for testing MECH-477 at all.

786a deliberately refused curiosity, on the grounds that curiosity_weight > 0
also makes the CEM scorer novelty-sensitive (score -= curiosity_weight *
mean(density * (1 - familiarity)), SD-025) and would make a single-arm
"recruitment is higher in novel contexts" claim circular. Two things resolve that
here rather than dismissing it:

  1. The lever is MEASURED, not assumed. The bonus is proportional to the SD-024
     representational density, and benefit_terrain_live_producer defaults False,
     so the benefit RBF map is unpopulated and the bonus is identically zero.
     Measured directly on this config: cross-candidate curiosity-bonus range =
     0.0 exactly (V3-EXQ-795 independently pins the OFF-producer path at zero
     benefit centers). The driver gates on that range as a precondition with a
     CEILING, so if a future substrate populates the terrain the run self-routes
     substrate_not_ready_requeue instead of quietly measuring a circular DV.
  2. Even were it live, it is COMMON-MODE. MECH-477's scored quantity is a
     difference-in-differences (the novel-minus-familiar delta ON versus OFF),
     and curiosity is held identical across the arms, so it cancels from that
     contrast. It would only threaten the secondary within-arm MECH-163 leg-(1)
     reading -- which is why the gate exists anyway.

FAMILIARITY INSTRUMENT (the manipulation check) IS SEPARATE AND PINNED
-----------------------------------------------------------------------
The manipulation check is 786a's, unchanged: rank discriminability (AUC) of
practiced vs held-out layout familiarity, read from an EXPERIMENT-OWNED
FamiliarityTracker that is wired into nothing, at the pre-registered query
bandwidth 0.20 with update bandwidth 1.0. Both are pinned here and NOT read from
config, for 786a's reason: config.familiarity_bandwidth was moved 1.0 -> 0.20 on
2026-07-20, and since the same constant is update()'s association threshold
(thresh_sq = bw*bw), inheriting it would allocate far more anchors and push the
clamped sum back toward the saturation that made V3-EXQ-786 unmeasurable.

This is a SECOND, independent tracker from the one the arbitrator reads. The
substrate's own hippocampal.familiarity_tracker (config bandwidth) drives
u_habit; this one is the ruler the manipulation check is measured with. Keeping
them separate is what stops the gate from moving when the substrate default does.

ASCII-only output (repo rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from _lib.goal_pipeline_tier1 import ENV_FISHTANK_KWARGS, ArmSpec, build_config, warmup_train
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.curiosity import FamiliarityTracker

LINEAGE = "mech477_dualsystem_arbitration"

ARM_OFF = "arb_off"
ARM_ON = "arb_on"
ARMS: Tuple[str, str] = (ARM_OFF, ARM_ON)

# --- Schedule (inherited from V3-EXQ-786a unchanged) --------------------------
FAMILIAR_ENV_SEEDS = [1000, 1001, 1002, 1003]
NOVEL_ENV_SEEDS = [2000, 2001, 2002, 2003]
PRACTICE_EPISODES_PER_LAYOUT = 30
STEPS_PER_EPISODE = 120
PROBE_EPISODES_PER_LAYOUT = 4
ACCUM_EPISODES_PER_LAYOUT = 10
TOTAL_TRAINING_EPISODES = PRACTICE_EPISODES_PER_LAYOUT * len(FAMILIAR_ENV_SEEDS)

# --- Familiarity manipulation (786a, unchanged) -------------------------------
# All three varied fields are invisible to world_obs_dim (causal_grid_world.py
# 1152-1176) and all three move 5x5 LOCAL field statistics, which is what this
# learner (egocentric local view) can actually observe. n_landmarks_b stays >= 1
# in both: dropping it to 0 would shrink world_obs_dim 300 -> 250.
FAMILIAR_ENV_KWARGS: Dict[str, Any] = dict(
    ENV_FISHTANK_KWARGS,
    env_drift_prob=0.0,
    size=10,
    num_hazards=3,
    num_resources=5,
    n_landmarks_b=2,
)
NOVEL_ENV_KWARGS: Dict[str, Any] = dict(
    ENV_FISHTANK_KWARGS,
    env_drift_prob=0.0,
    size=14,
    num_hazards=7,
    num_resources=2,
    n_landmarks_b=5,
)

# --- Measurement constants (enter the cell's values -> enter the slice) -------
# HABIT_DEPTH mirrors E3Config.dualsystem_habit_depth. 2, never 1: index 0 is the
# shared current state, so a depth-1 vector is constant by construction.
HABIT_DEPTH = 2
CURIOSITY_WEIGHT = 0.05
# SD-081 registered defaults, pinned so a later default change refuses reuse
# rather than silently re-basing the arm.
ARB_GAIN = 4.0
ARB_BIAS = 0.0
ARB_UNCERTAINTY_EMA_ALPHA = 0.05
# Experiment-owned manipulation-check instrument (786a's calibrated pair).
FAMILIARITY_QUERY_BANDWIDTH = 0.20
FAMILIARITY_UPDATE_BANDWIDTH = 1.0
FAMILIARITY_EMA_ALPHA = 0.01
FAMILIARITY_SATURATION_EPS = 0.01
# A reading below this cross-candidate range is treated as no ranking at all.
SCORE_RANGE_FLOOR = 1e-6


def arm_config_slice(arm_id: str) -> Dict[str, Any]:
    """The fingerprint slice for one arm: exactly what its computation reads.

    Declares env kwargs + schedule + the substrate-operating config + the arm's
    own flag + every measurement constant that enters a returned value. It does
    NOT declare acceptance thresholds, gate floors, seed lists or arm labels --
    none of those change what a cell computes, and folding them in would refuse
    reuse across an iteration that only moved a threshold (plan 7b constraint 2).
    """
    if arm_id not in ARMS:
        raise ValueError(f"unknown arm_id {arm_id!r}; expected one of {ARMS}")
    return {
        "lineage": LINEAGE,
        "arm_id": arm_id,
        "use_dualsystem_arbitration": bool(arm_id == ARM_ON),
        "familiar_env_kwargs": FAMILIAR_ENV_KWARGS,
        "novel_env_kwargs": NOVEL_ENV_KWARGS,
        "familiar_env_seeds": FAMILIAR_ENV_SEEDS,
        "novel_env_seeds": NOVEL_ENV_SEEDS,
        "practice_episodes_per_layout": PRACTICE_EPISODES_PER_LAYOUT,
        "probe_episodes_per_layout": PROBE_EPISODES_PER_LAYOUT,
        "accum_episodes_per_layout": ACCUM_EPISODES_PER_LAYOUT,
        "steps_per_episode": STEPS_PER_EPISODE,
        "build_config_arm_id": "mech163_recruitment",
        "curiosity_weight": CURIOSITY_WEIGHT,
        "habit_depth": HABIT_DEPTH,
        "dualsystem_arbitration_gain": ARB_GAIN,
        "dualsystem_arbitration_bias": ARB_BIAS,
        "dualsystem_uncertainty_ema_alpha": ARB_UNCERTAINTY_EMA_ALPHA,
        "familiarity_query_bandwidth": FAMILIARITY_QUERY_BANDWIDTH,
        "familiarity_update_bandwidth": FAMILIARITY_UPDATE_BANDWIDTH,
        "familiarity_ema_alpha": FAMILIARITY_EMA_ALPHA,
    }


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical OFF/baseline slice -- what a consumer cites to reuse."""
    return arm_config_slice(ARM_OFF)


def assert_dims_match() -> Dict[str, int]:
    """Refuse the run if the two conditions do not share observation dims.

    The novelty axis is chosen precisely because world_obs_dim is independent of
    size / num_hazards / num_resources. That is a property of the CURRENT env,
    not a law. One agent per (seed, arm) scores both conditions, so a mismatch
    would be a crash or -- worse -- a silent shape coercion. Assert, do not trust
    the comment.
    """
    fam = CausalGridWorldV2(seed=FAMILIAR_ENV_SEEDS[0], **FAMILIAR_ENV_KWARGS)
    nov = CausalGridWorldV2(seed=NOVEL_ENV_SEEDS[0], **NOVEL_ENV_KWARGS)
    if (fam.world_obs_dim, fam.body_obs_dim) != (nov.world_obs_dim, nov.body_obs_dim):
        raise RuntimeError(
            "familiar/novel observation dims differ "
            f"(familiar world={fam.world_obs_dim} body={fam.body_obs_dim}; "
            f"novel world={nov.world_obs_dim} body={nov.body_obs_dim})."
        )
    if fam.action_dim != nov.action_dim:
        raise RuntimeError(
            f"familiar/novel action dims differ ({fam.action_dim} vs {nov.action_dim})."
        )
    return {
        "world_obs_dim": int(fam.world_obs_dim),
        "body_obs_dim": int(fam.body_obs_dim),
        "action_dim": int(fam.action_dim),
    }


def build_arm_agent(arm_id: str) -> Tuple[REEAgent, Any]:
    """Build the (config, agent) pair for one arm.

    The SD-081 params live on E3Config, NOT REEConfig: E3Selector.config IS the
    E3Config, so a REEConfig-level field reads as a missing attribute in the
    selector and defaults to False -- the silently-unreachable-flag hazard one
    level below the documented from_dims one, and the one this build tripped
    during authoring (arbitrator wired, 45 select() calls, zero arbitrations).
    Set on cfg.e3 explicitly, before REEAgent construction.
    """
    proto_env = CausalGridWorldV2(seed=FAMILIAR_ENV_SEEDS[0], **FAMILIAR_ENV_KWARGS)
    cfg = build_config(proto_env, ArmSpec(arm_id="mech163_recruitment"))
    # from_dims path -> alpha_world=0.9 (SD-008).
    cfg.hippocampal.curiosity_weight = CURIOSITY_WEIGHT
    cfg.e3.use_dualsystem_arbitration = bool(arm_id == ARM_ON)
    cfg.e3.dualsystem_arbitration_gain = ARB_GAIN
    cfg.e3.dualsystem_arbitration_bias = ARB_BIAS
    cfg.e3.dualsystem_uncertainty_ema_alpha = ARB_UNCERTAINTY_EMA_ALPHA
    cfg.e3.dualsystem_habit_depth = HABIT_DEPTH
    return REEAgent(cfg), cfg


def _obs_field(obs_dict: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    val = obs_dict.get(key)
    if val is None:
        return None
    val = val.float()
    return val.unsqueeze(0) if val.dim() == 1 else val


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman rank correlation as Pearson over ranks (no scipy dep).

    Returns None when either vector has zero RANK variance. NOTE that guard is
    NOT sufficient on its own -- double-argsort of a CONSTANT vector is a
    permutation of 0..K-1 with large rank std, which is exactly why 786a's guard
    could not fire on its constant depth-1 vector. The caller must additionally
    gate on the raw cross-candidate range / distinct-value fraction, which the
    driver does as a precondition.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_greater(pos: List[float], neg: List[float]) -> Optional[float]:
    """P(a random `pos` exceeds a random `neg`), ties 0.5. None if either empty.

    Rank-based, hence invariant under any monotone transform of the familiarity
    readout -- the point of 786a's CHANGE 2: the readout's scale is an instrument
    parameter, so a raw-difference threshold moves when the instrument is
    re-tuned. 0.5 is the principled null.
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def _depth_scores(agent: REEAgent, candidates: List[Any]) -> Tuple[List[float], List[float]]:
    """Score every candidate at FULL horizon and at HABIT_DEPTH.

    Mirrors HippocampalModule.score_trajectory (which sums
    residue_field.evaluate_trajectory over the sequence) at two depths of the
    SAME machinery -- so the habit read is a depth-limited view of the planned
    read, not a second scorer that could drift from it. Lower = better.
    Returns ([], []) when any candidate lacks a usable world-state sequence.
    """
    full: List[float] = []
    habit: List[float] = []
    with torch.no_grad():
        for traj in candidates:
            world_seq = traj.get_world_state_sequence()   # [batch, horizon+1, world_dim]
            if world_seq is None or world_seq.shape[1] <= HABIT_DEPTH:
                return [], []
            full.append(float(agent.residue_field.evaluate_trajectory(world_seq)[0].item()))
            habit.append(
                float(
                    agent.residue_field.evaluate_trajectory(
                        world_seq[:, :HABIT_DEPTH, :]
                    )[0].item()
                )
            )
    return full, habit


def _curiosity_bonus_range(agent: REEAgent, candidates: List[Any]) -> float:
    """Cross-candidate RANGE of the SD-025 curiosity bonus.

    RANGE, not magnitude, for two reasons that point the same way: (a) the
    V3-EXQ-643 same-statistic lesson, and (b) a UNIFORM bonus is invariant under
    the CEM argmin, so only the cross-candidate spread could reorder anything.
    Fail-closed: an unmeasurable bonus returns +inf so the driver's CEILING
    precondition refuses the run rather than assuming inertness.
    """
    hipp = getattr(agent, "hippocampal", None)
    if hipp is None:
        return 0.0
    vals: List[float] = []
    try:
        with torch.no_grad():
            for traj in candidates:
                seq = traj.get_world_state_sequence()
                if seq is None:
                    return float("inf")
                vals.append(float(hipp._curiosity_bonus(seq).item()))
    except Exception:
        return float("inf")
    if not vals or not all(np.isfinite(vals)):
        return float("inf")
    return float(max(vals) - min(vals))


def _familiarity(
    tracker: FamiliarityTracker,
    z_world: torch.Tensor,
    bandwidth: float = FAMILIARITY_QUERY_BANDWIDTH,
) -> Optional[float]:
    """Query the EXPERIMENT-OWNED manipulation-check instrument."""
    with torch.no_grad():
        fam = tracker.query(z_world, bandwidth=bandwidth)
    if fam is None or fam.numel() == 0:
        return None
    return float(fam.mean().item())


def probe_layout(
    agent: REEAgent,
    tracker: FamiliarityTracker,
    env_seed: int,
    env_kwargs: Dict[str, Any],
    n_episodes: int,
    arm_id: str,
    update_tracker: bool = False,
) -> Dict[str, Any]:
    """Run probe episodes on ONE layout, measuring recruitment per E3 tick.

    RECRUITMENT = 1 - spearman(full_horizon_scores, habit_depth_scores): the rate
    at which deep rollout REORDERS the candidate set relative to a myopic read of
    the same machinery. 0 = lookahead changes nothing (the habit path suffices);
    higher = planning is doing work.

    LATCH DISCIPLINE. Everything read here is sampled ONLY on a real E3 tick.
    Between ticks generate_trajectories returns the CACHED candidate set (MECH-057a
    gate) and e3.last_arbitration retains its previous value, so a per-env-step
    read would re-record one selection as many independent observations -- the
    E3 cadence defaults to 10 steps/tick, so ~10x pseudo-replication with nothing
    erroring. last_arbitration is additionally cleared immediately before
    select_action, and a None reading on an E3 tick in the ON arm is counted as
    n_latched rather than recorded.

    No reward, no competence, forward-only.
    """
    is_on = bool(arm_id == ARM_ON)
    ticks_scored = 0
    n_latched = 0
    recruitments: List[float] = []
    full_ranges: List[float] = []
    habit_ranges: List[float] = []
    habit_distinct_fracs: List[float] = []
    familiarities: List[float] = []
    arb_rows: List[Dict[str, Any]] = []
    arb_degenerate = 0
    arb_sources: Dict[str, int] = {}
    curiosity_range_max = 0.0
    curiosity_sampled = 0

    for _ep in range(n_episodes):
        # Fresh instance at a FIXED seed -> identical layout every time.
        # CausalGridWorldV2 seeds self._rng ONCE in __init__ and reset() advances
        # that stream, so layouts differ episode to episode within one instance;
        # re-instantiating pins the layout.
        env = CausalGridWorldV2(seed=env_seed, **env_kwargs)
        _flat, obs_dict = env.reset()
        agent.reset()

        for _step in range(STEPS_PER_EPISODE):
            body = _obs_field(obs_dict, "body_state")
            world = _obs_field(obs_dict, "world_state")
            if body is None or world is None:
                break
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=_obs_field(obs_dict, "harm_obs"),
                obs_harm_a=_obs_field(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs_field(obs_dict, "harm_history"),
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            is_e3 = bool(ticks.get("e3_tick", False))

            if is_e3 and candidates and len(candidates) >= 2:
                full, habit = _depth_scores(agent, candidates)
                if full and habit:
                    k = len(habit)
                    f_rng = float(max(full) - min(full))
                    h_rng = float(max(habit) - min(habit))
                    full_ranges.append(f_rng)
                    habit_ranges.append(h_rng)
                    # The direct 786a check: a constant vector has distinct-frac
                    # 1/K, and no rank-variance guard can see it.
                    habit_distinct_fracs.append(float(len(set(habit))) / float(k))
                    if f_rng > SCORE_RANGE_FLOOR and h_rng > SCORE_RANGE_FLOOR:
                        rho = spearman(full, habit)
                        if rho is not None:
                            ticks_scored += 1
                            recruitments.append(1.0 - rho)
                    fam = _familiarity(tracker, latent.z_world)
                    if fam is not None:
                        familiarities.append(fam)
                    # Sampled once per layout: the SD-025 lever is structurally
                    # zero or not, so a per-tick recompute over 32 candidates
                    # would double the probe cost for no extra information.
                    if curiosity_sampled == 0:
                        curiosity_range_max = max(
                            curiosity_range_max, _curiosity_bonus_range(agent, candidates)
                        )
                        curiosity_sampled += 1

            if update_tracker:
                # Familiarity is advanced ONLY on the accumulation pass, and only
                # on real visited states -- never on CEM-internal rollout states.
                with torch.no_grad():
                    tracker.update(latent.z_world.detach())

            agent.e3.last_arbitration = None
            action = agent.select_action(candidates, ticks)
            if is_on and is_e3:
                arb = agent.e3.last_arbitration
                if arb is None:
                    n_latched += 1
                else:
                    src = str(arb.get("habit_uncertainty_source") or "none")
                    arb_sources[src] = arb_sources.get(src, 0) + 1
                    if bool(arb.get("degenerate")):
                        arb_degenerate += 1
                    else:
                        arb_rows.append({
                            "w_planned": float(arb["w_planned"]),
                            "u_habit_norm": float(arb["u_habit_norm"]),
                            "u_planned_norm": float(arb["u_planned_norm"]),
                            "u_habit_raw": float(arb["u_habit_raw"]),
                            "u_planned_raw": float(arb["u_planned_raw"]),
                            "habit_score_range": float(arb["habit_score_range"]),
                            "planned_score_range": float(arb["planned_score_range"]),
                            "source": src,
                        })

            if action is None or not torch.isfinite(action).all():
                act_idx = int(np.random.randint(0, int(env.action_dim)))
            else:
                act_idx = int(action[0].argmax().item())

            _flat, _harm, done, info, obs_dict = env.step(act_idx)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(info.get("harm_signal", 0.0)) if isinstance(info, dict) else 0.0,
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            if done:
                break

    if familiarities:
        n_f = float(len(familiarities))
        pinned_hi = sum(1 for f in familiarities
                        if f >= 1.0 - FAMILIARITY_SATURATION_EPS) / n_f
        pinned_lo = sum(1 for f in familiarities
                        if f <= FAMILIARITY_SATURATION_EPS) / n_f
    else:
        pinned_hi = pinned_lo = 0.0

    return {
        "env_seed": env_seed,
        "ticks_scored": ticks_scored,
        "n_latched_ticks": n_latched,
        "recruitment_rate": float(np.mean(recruitments)) if recruitments else None,
        "recruitment_sd": float(np.std(recruitments)) if recruitments else None,
        "mean_full_score_range": float(np.mean(full_ranges)) if full_ranges else 0.0,
        # WORST tick, never a mean: the degeneracy this exists to catch is a
        # single pinned tick, which a mean hides (and the indexer recomputes
        # `met` from the reported number, so it must be the quantified one).
        "min_full_score_range": float(min(full_ranges)) if full_ranges else 0.0,
        "min_habit_score_range": float(min(habit_ranges)) if habit_ranges else 0.0,
        "mean_habit_score_range": float(np.mean(habit_ranges)) if habit_ranges else 0.0,
        "min_habit_distinct_frac": (
            float(min(habit_distinct_fracs)) if habit_distinct_fracs else 0.0
        ),
        "mean_familiarity": float(np.mean(familiarities)) if familiarities else None,
        "familiarity_pinned_high_frac": pinned_hi,
        "familiarity_pinned_low_frac": pinned_lo,
        "curiosity_bonus_range_max": float(curiosity_range_max),
        "arb_rows": arb_rows,
        "arb_degenerate": arb_degenerate,
        "arb_sources": arb_sources,
    }


def run_cell(
    seed: int,
    arm_id: str,
    *,
    practice_episodes: int = PRACTICE_EPISODES_PER_LAYOUT,
    probe_episodes: int = PROBE_EPISODES_PER_LAYOUT,
    accum_episodes: int = ACCUM_EPISODES_PER_LAYOUT,
) -> Dict[str, Any]:
    """One (seed, arm) cell: practice on FAMILIAR, then probe BOTH conditions.

    The caller is responsible for the per-cell RNG reset and fingerprint stamp
    (use experiments/_lib/arm_fingerprint.arm_cell, which does both).
    """
    print(f"Seed {seed} Condition {arm_id}_practice", flush=True)
    agent, cfg = build_arm_agent(arm_id)

    tracker = FamiliarityTracker(
        world_dim=int(cfg.hippocampal.world_dim),
        ema_alpha=FAMILIARITY_EMA_ALPHA,
        bandwidth=FAMILIARITY_UPDATE_BANDWIDTH,
    )

    total_eps = practice_episodes * len(FAMILIAR_ENV_SEEDS)
    for env_seed in FAMILIAR_ENV_SEEDS:
        env = CausalGridWorldV2(seed=env_seed, **FAMILIAR_ENV_KWARGS)
        warmup_train(
            agent,
            env,
            num_episodes=practice_episodes,
            steps_per_episode=STEPS_PER_EPISODE,
            label=f"seed{seed}_{arm_id}_layout{env_seed}",
            progress_total_episodes=total_eps,
        )

    # Forward-only familiarity accumulation over the PRACTICED layouts, after
    # training so the instrument reads the same encoder the probes will use.
    for env_seed in FAMILIAR_ENV_SEEDS:
        probe_layout(agent, tracker, env_seed, FAMILIAR_ENV_KWARGS, accum_episodes,
                     arm_id, update_tracker=True)

    per_condition: Dict[str, Any] = {}
    for cond, env_seeds, env_kwargs in (
        ("familiar", FAMILIAR_ENV_SEEDS, FAMILIAR_ENV_KWARGS),
        ("novel", NOVEL_ENV_SEEDS, NOVEL_ENV_KWARGS),
    ):
        print(f"Seed {seed} Condition {arm_id}_{cond}", flush=True)
        rows = [probe_layout(agent, tracker, es, env_kwargs, probe_episodes, arm_id)
                for es in env_seeds]
        rates = [r["recruitment_rate"] for r in rows if r["recruitment_rate"] is not None]
        fams = [r["mean_familiarity"] for r in rows if r["mean_familiarity"] is not None]
        arb_rows: List[Dict[str, Any]] = []
        sources: Dict[str, int] = {}
        for r in rows:
            arb_rows.extend(r["arb_rows"])
            for k, v in r["arb_sources"].items():
                sources[k] = sources.get(k, 0) + v
        per_condition[cond] = {
            "per_layout": [{k: v for k, v in r.items() if k != "arb_rows"} for r in rows],
            "layout_familiarities": fams,
            "recruitment_rate": float(np.mean(rates)) if rates else None,
            "mean_familiarity": float(np.mean(fams)) if fams else None,
            "ticks_scored": int(sum(r["ticks_scored"] for r in rows)),
            "n_latched_ticks": int(sum(r["n_latched_ticks"] for r in rows)),
            "min_full_score_range": float(min(r["min_full_score_range"] for r in rows)),
            "min_habit_score_range": float(min(r["min_habit_score_range"] for r in rows)),
            "min_habit_distinct_frac": float(min(r["min_habit_distinct_frac"] for r in rows)),
            "curiosity_bonus_range_max": float(
                max(r["curiosity_bonus_range_max"] for r in rows)
            ),
            "worst_pinned_high_frac": max(r["familiarity_pinned_high_frac"] for r in rows),
            "worst_pinned_low_frac": max(r["familiarity_pinned_low_frac"] for r in rows),
            "arb_n_degenerate": int(sum(r["arb_degenerate"] for r in rows)),
            "arb_sources": sources,
            "arb_w_mean": (
                float(np.mean([a["w_planned"] for a in arb_rows])) if arb_rows else None
            ),
            "arb_n_rows": len(arb_rows),
            "_arb_rows": arb_rows,
        }

    fam_rate = per_condition["familiar"]["recruitment_rate"]
    nov_rate = per_condition["novel"]["recruitment_rate"]
    delta = (nov_rate - fam_rate) if (fam_rate is not None and nov_rate is not None) else None

    # MANIPULATION CHECK: rank discriminability of the two layout populations.
    # A layout that logged no scored E3 tick contributes no familiarity value,
    # and AUC over a truncated population is not the pre-registered statistic.
    fam_pop = per_condition["familiar"]["layout_familiarities"]
    nov_pop = per_condition["novel"]["layout_familiarities"]
    auc = (
        auc_greater(fam_pop, nov_pop)
        if (len(fam_pop) >= 3 and len(nov_pop) >= 3)
        else None
    )

    fam_f = per_condition["familiar"]["mean_familiarity"]
    nov_f = per_condition["novel"]["mean_familiarity"]

    return {
        "seed": seed,
        "arm_id": arm_id,
        "conditions": per_condition,
        "recruitment_delta": delta,
        "familiarity_auc": auc,
        # NON-GATING, kept so a successor can compare like with like against
        # V3-EXQ-786's recorded 0.049365 achievable separation.
        "familiarity_separation_raw": (
            (fam_f - nov_f) if (fam_f is not None and nov_f is not None) else None
        ),
        "min_full_score_range": float(min(
            per_condition["familiar"]["min_full_score_range"],
            per_condition["novel"]["min_full_score_range"],
        )),
        "min_habit_score_range": float(min(
            per_condition["familiar"]["min_habit_score_range"],
            per_condition["novel"]["min_habit_score_range"],
        )),
        "min_habit_distinct_frac": float(min(
            per_condition["familiar"]["min_habit_distinct_frac"],
            per_condition["novel"]["min_habit_distinct_frac"],
        )),
        "curiosity_bonus_range_max": float(max(
            per_condition["familiar"]["curiosity_bonus_range_max"],
            per_condition["novel"]["curiosity_bonus_range_max"],
        )),
        "worst_pinned_high_frac": max(
            per_condition["familiar"]["worst_pinned_high_frac"],
            per_condition["novel"]["worst_pinned_high_frac"],
        ),
        "worst_pinned_low_frac": max(
            per_condition["familiar"]["worst_pinned_low_frac"],
            per_condition["novel"]["worst_pinned_low_frac"],
        ),
    }


__all__ = [
    "LINEAGE", "ARM_OFF", "ARM_ON", "ARMS",
    "FAMILIAR_ENV_SEEDS", "NOVEL_ENV_SEEDS", "FAMILIAR_ENV_KWARGS", "NOVEL_ENV_KWARGS",
    "PRACTICE_EPISODES_PER_LAYOUT", "PROBE_EPISODES_PER_LAYOUT",
    "ACCUM_EPISODES_PER_LAYOUT", "STEPS_PER_EPISODE", "TOTAL_TRAINING_EPISODES",
    "HABIT_DEPTH", "CURIOSITY_WEIGHT", "ARB_GAIN", "ARB_BIAS",
    "ARB_UNCERTAINTY_EMA_ALPHA", "FAMILIARITY_QUERY_BANDWIDTH",
    "FAMILIARITY_UPDATE_BANDWIDTH", "FAMILIARITY_EMA_ALPHA", "SCORE_RANGE_FLOOR",
    "arm_config_slice", "off_path_config_slice", "assert_dims_match",
    "build_arm_agent", "probe_layout", "run_cell", "spearman", "auc_greater",
]
