"""Canonical CONTROL-arm baseline for the MECH-321 SCALE-RESOLVED ROLLOUT
BOUNDARY sub-lineage (V3-EXQ-830 and successors).

WHY A NEW MODULE (and not a reuse of mech321_policy_decomposition_harshened_v2)
    The 816 campaign's OFF arm is `use_policy_decomposition=False` -- an arm in
    which MECH-321 never runs at all. This sub-lineage asks a question INSIDE a
    running MECH-321: which MECH-288 scale drives the boundary half of the R1
    OR trigger, and at which rollout position. So BOTH arms here must have
    decomposition ON, and the manipulation is
    `use_decomposition_scale_resolved_probe` alone. That is a different OFF
    closure from 816d's, so per the arm-fingerprint reuse model it needs its own
    canonical module -- editing the v2 harshened module in place would silently
    invalidate the reuse contract 816d minted for its own OFF arm.

ENV DOSE: inherited unchanged from the 816d dose-escalated harshened module
    (env_drift_interval=1, world_rule_shift_interval=12, depth=2). That is the
    configuration in which the motivating measurement was taken --
    `decomp_fired_frac_arm1 = 1.0` (816d) with `vs_trigger_fires_total = 0`,
    `pe_trigger_fires_total = 13`, `cofire_total = 0` (816c). Reading the scale
    composition of the ALWAYS-FIRES end requires standing in the configuration
    where it saturates, so the env is held fixed and only the probe flag moves.

z_goal_enabled=True IS PART OF THIS BASELINE, IN BOTH ARMS
    `REEConfig.from_dims(z_goal_enabled=...)` defaults to False, so
    `REEAgent.goal_state` is None and `_current_z_goal` is None on every
    proposal tick. The scale-resolved probe adds `z_goal` to the rollout
    latent_signature -- but MECH-288's slow BOCPD detector returns
    (False, 0.0, []) when NONE of its declared streams resolves to a tensor
    (event_segmenter.py `if not sources`), and a None value is skipped exactly
    like an absent key. So with the substrate default the probe flag would be
    INERT: the slow scale stays structurally dead, and the run would report a
    clean "slow never fires on rollout" -- spike outcome 2, closing the design
    question -- when the true state is that the instrument was never switched
    on. That is the same failure shape the probe's own C18b contract caught in
    the substrate wiring (ree-v3 9a6e7f3976), arriving one layer up.

    z_goal_enabled is therefore held ON in BOTH arms: it is a precondition of
    the measurement, not the manipulation. Both arms pay its cost identically,
    so the probe-flag contrast is unconfounded by it. This is also why the ON
    arm carries a z_goal-VARIATION readiness precondition rather than a mere
    presence check -- see the script's precondition block.

benefit_threshold=0.05 IS ALSO A PRECONDITION, AND IT WAS MEASURED, NOT GUESSED
    z_goal_enabled alone is NOT sufficient. GoalState.update() only pulls z_goal
    toward z_world when `benefit_exposure > benefit_threshold`, and until that
    first pull z_goal is the zero vector, so `GoalState.is_active()` is False and
    REEAgent hands `current_z_goal=None` to the rollout sweep -- leaving the slow
    scale dead for the second time, one layer further down.

    `REEConfig.from_dims` defaults benefit_threshold to 0.1 (config.py:5211).
    Measured on THIS env dose (3 seeds x 60 episodes x 24 steps, random policy,
    2941 steps): benefit_exposure is a hedonic EMA whose distribution is
    p50=0.014, p90=0.045, p95=0.059, MAX=0.1018. It exceeds 0.1 on 0.03% of steps
    -- 1 step in 2941 -- so at the from_dims default z_goal would essentially
    never seed and the probe would spend a full cloud run to report a confident
    "slow never fires on rollout" that is entirely a wiring artefact. The
    V3-EXQ-830 dry-run smoke confirmed this end-to-end before the value was
    changed: zgoal_present_frac = 0.0, gate red, label
    substrate_not_ready_requeue.

    0.05 is NOT a tuned number invented to force a result. It is the substrate's
    OWN default in `REEConfig.enable_goal_stream(benefit_threshold=0.05)`
    (config.py:5095) -- i.e. the value the substrate uses whenever the goal
    stream is deliberately switched on, which is precisely what this experiment
    does. `from_dims`'s 0.1 is the outlier. On the measured distribution 0.05
    seeds on 8.1% of steps: INTERMITTENT, which is what the slow BOCPD needs --
    z_goal is pulled toward z_world on those steps and decays (decay_goal=0.005)
    between them, so its norm genuinely moves and carries change-points. A
    threshold that fired on every step would saturate the norm and be just as
    dead as one that never fires.

    COST, STATED PLAINLY: at 0.05 the goal is seeded by high-benefit EXPOSURE
    (a proximity-weighted hedonic EMA) rather than only by resource CONTACT. A
    slow-scale fire therefore means "the goal latent shifted", not specifically
    "a resource was consumed". That is the correct object for this question --
    MECH-288's slow scale is a goal-SHIFT detector, not a consumption detector --
    but it is a real change in what z_goal indexes and is recorded here so no
    downstream reader has to infer it. Held IDENTICAL in both arms, so it cannot
    confound the probe-flag contrast; and the ON arm's readiness gate still
    asserts the z_goal norm actually VARIED, so if 0.05 turns out to be
    insufficient at run time the probe self-routes to substrate_not_ready_requeue
    rather than mislabelling the null.

MINT / REUSE
    V3-EXQ-830 is the FIRST experiment of this sub-lineage: its
    ARM_PROBE_OFF cells are emitted reuse-ELIGIBLE
    (include_driver_script_in_hash=False) and constructed from THIS module, so
    a future same-config successor matches the fingerprint BY CONSTRUCTION.
"""
from __future__ import annotations

from typing import Any, Dict, List

from experiments._lib.baselines import (
    mech321_policy_decomposition_harshened_v2 as base_v2,
)

# --- Env shape / schedule / representation / chunk: inherited unchanged from
# the 816d dose-escalated harshened module -- single source of truth. ---
ENV_SIZE = base_v2.ENV_SIZE
ENV_NUM_HAZARDS = base_v2.ENV_NUM_HAZARDS
ENV_NUM_RESOURCES = base_v2.ENV_NUM_RESOURCES
ENV_USE_PROXY_FIELDS = base_v2.ENV_USE_PROXY_FIELDS
HARSH_ENV_DRIFT_INTERVAL = base_v2.HARSH_ENV_DRIFT_INTERVAL
HARSH_WORLD_RULE_SHIFT_ENABLED = base_v2.HARSH_WORLD_RULE_SHIFT_ENABLED
HARSH_WORLD_RULE_SHIFT_INTERVAL = base_v2.HARSH_WORLD_RULE_SHIFT_INTERVAL
HARSH_WORLD_RULE_SHIFT_DEPTH = base_v2.HARSH_WORLD_RULE_SHIFT_DEPTH
WARMUP_EPISODES = base_v2.WARMUP_EPISODES
MEASURE_EPISODES = base_v2.MEASURE_EPISODES
STEPS_PER_EPISODE = base_v2.STEPS_PER_EPISODE
ALPHA_WORLD = base_v2.ALPHA_WORLD
SEEDED_CHUNK_SEQUENCE = base_v2.SEEDED_CHUNK_SEQUENCE

# --- Decomposition parameters, held identical in BOTH arms. Threshold 0.5 and
# depth_cap 3 are the values the whole 816 ladder used. Per the 2026-07-27
# scoping spike section 1a, depth_cap is a binary switch at this seeded-chunk
# configuration (every chunk is minted at depth 1), so 3 here behaves exactly
# as 2 would -- it is carried unchanged for ladder comparability, not because
# the value is load-bearing. ---
DECOMPOSITION_VS_THRESHOLD = 0.5
DECOMPOSITION_DEPTH_CAP = 3

# --- z_goal seeding threshold. The substrate's own enable_goal_stream() default
# (config.py:5095), NOT from_dims's 0.1 (config.py:5211), which is unreachable on
# this env dose (measured: benefit_exposure max 0.1018, exceeds 0.1 on 0.03% of
# steps). At 0.05 z_goal seeds on ~8.1% of steps -- intermittent, so its norm
# genuinely moves and the slow BOCPD has change-points to find. See module
# docstring for the full measurement and the stated cost. ---
GOAL_BENEFIT_THRESHOLD = 0.05


def total_episodes() -> int:
    """Denominator M for the runner progress contract (warmup + measure)."""
    return WARMUP_EPISODES + MEASURE_EPISODES


def env_kwargs(seed: int) -> Dict[str, Any]:
    """Dose-escalated harshened CausalGridWorldV2 kwargs (816d's env, verbatim)."""
    return dict(base_v2.env_kwargs(seed))


def substrate_stack_flags() -> Dict[str, Any]:
    """Substrate stack shared by BOTH arms of this sub-lineage.

    The 816 stack, plus the two additions this question requires:
      use_policy_decomposition=True -- MECH-321 must actually RUN in both arms
        (the manipulation is which SCALES it consumes, not whether it runs).
      z_goal_enabled=True           -- gives GoalState existence so
        _current_z_goal can be non-None; without it the probe flag is inert
        (see module docstring).
      benefit_threshold=0.05        -- lets z_goal actually SEED on this env
        dose, so is_active() becomes True and _current_z_goal stops being None.
        z_goal_enabled alone is not sufficient (see module docstring).
    """
    flags = dict(base_v2.substrate_stack_flags())
    flags.update({
        "z_goal_enabled": True,
        "benefit_threshold": GOAL_BENEFIT_THRESHOLD,
        "use_policy_decomposition": True,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
    })
    return flags


def off_arm_flags() -> Dict[str, Any]:
    """ARM_PROBE_OFF: full stack, scale-resolved probe OFF.

    Reproduces the current single-scale behaviour exactly -- the slow scale has
    no z_goal on the rollout side, so it is structurally dead and the per-scale
    counters read (n_boundary_fires, 0, 0) by construction.
    """
    flags = substrate_stack_flags()
    flags["use_decomposition_scale_resolved_probe"] = False
    return flags


def on_arm_flags() -> Dict[str, Any]:
    """ARM_PROBE_ON: full stack, scale-resolved probe ON (the manipulation)."""
    flags = substrate_stack_flags()
    flags["use_decomposition_scale_resolved_probe"] = True
    return flags


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical ARM_PROBE_OFF fingerprint config slice (producer AND consumer).

    MUST be identical byte-for-byte between V3-EXQ-830's ARM_PROBE_OFF emit and
    any successor's try_reuse_cell(...) call, or reuse is (correctly) refused.
    Declares only what the control computation reads.
    """
    slice_: Dict[str, Any] = {
        "env": {
            "size": ENV_SIZE,
            "num_hazards": ENV_NUM_HAZARDS,
            "num_resources": ENV_NUM_RESOURCES,
            "use_proxy_fields": ENV_USE_PROXY_FIELDS,
            "env_drift_interval": HARSH_ENV_DRIFT_INTERVAL,
            "world_rule_shift_enabled": HARSH_WORLD_RULE_SHIFT_ENABLED,
            "world_rule_shift_interval": HARSH_WORLD_RULE_SHIFT_INTERVAL,
            "world_rule_shift_depth": HARSH_WORLD_RULE_SHIFT_DEPTH,
        },
        "schedule": {
            "warmup_episodes": WARMUP_EPISODES,
            "measure_episodes": MEASURE_EPISODES,
            "steps": STEPS_PER_EPISODE,
        },
        "alpha_world": ALPHA_WORLD,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
    }
    slice_.update(off_arm_flags())
    return slice_


def off_needed_keys() -> List[str]:
    """Control-arm metrics a consumer reads back from a reused cell."""
    return [
        "n_sweeps",
        "n_sweeps_fast_only",
        "n_sweeps_slow_only",
        "n_sweeps_cofire",
        "decomp_n_boundary_fires",
        "decomp_n_boundary_fires_fast",
        "decomp_n_boundary_fires_slow",
        "decomp_n_evaluated_precommit",
        "decomp_n_evaluated_midexec",
        "n_measure_steps",
    ]
