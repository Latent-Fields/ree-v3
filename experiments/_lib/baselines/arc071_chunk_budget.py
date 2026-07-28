"""Canonical OFF-path baseline for the ARC-071 / MECH-323 chunk-budget lineage.

WHAT THIS IS. The shared construction of the lineage's baseline arm -- chunking
ON, both growable ceilings OFF, so chunk_max_size and chunk_max_depth are hard
lifetime caps exactly as they were before the 2026-07-27 growable-ceiling
landings. Every experiment in this lineage builds that arm from HERE rather than
from its own literals, which is what makes the arm fingerprint match ACROSS
DRIVERS by construction (arm_reuse_fingerprint_plan.md section 7b): a later
sibling with a different driver script constructs a bit-identical OFF cell and
can cite the mint instead of re-training it.

WHY THE BASELINE IS CHUNKING-ON, NOT CHUNKING-OFF. The question this lineage
asks is what the growable ceilings do, so the reference condition is the agent
with the operators running under FIXED ceilings. A chunking-OFF agent never
constructs the accumulator at all and so has no ceiling readout to compare
against -- it is an inertness control, not a baseline.

MINT CONTRACT. Consumers -- including the first experiment of the lineage --
must emit this arm's fingerprint with include_driver_script_in_hash=False. With
the default (True) each driver folds its own script into substrate_hash and no
sibling can ever match the mint (plan sections 9.4 / 9.7).

Lineage mint run: V3-EXQ-834
(v3_exq_834_arc071_mech323_budget_coupled_ceilings). Cite it as
reuse_baseline_from once its manifest lands.
"""
from __future__ import annotations

from typing import Any, Dict

LINEAGE = "arc071_chunk_budget"
CANONICAL_BASELINE_ID = "arc071_chunk_budget_fixed_ceilings_v1"

# --- Environment -----------------------------------------------------------
# num_hazards > 0 deliberately (V3-EXQ-810 post-mortem, ree-v3/CLAUDE.md): harm
# events are the main source of the MECH-091 salient-event E3 phase_reset that
# breaks the perfectly-periodic E3 tick. With num_hazards=0 and no
# update_residue call the tick is exactly periodic, the symbol buffer is exactly
# steps_per_episode / e3_steps_per_tick on every seed, and the behavioural
# repertoire is too thin to chunk.
ENV_KWARGS: Dict[str, Any] = {
    "size": 8,
    "num_hazards": 2,
    "num_resources": 6,
    "use_proxy_fields": True,
}

# --- Schedule --------------------------------------------------------------
# STEPS_PER_EPISODE is load-bearing and is NOT a free parameter. record_step is
# reached only on the E3 deliberation path and E3 ticks every
# e3_steps_per_tick = 10 env steps, so the accumulator sees
# steps_per_episode / 10 symbols per episode. note_outcome breaks out at
# len(actions) < size, so a sequence length above that count is STRUCTURALLY
# unreachable and every ceiling at or above it is inert. At V3-EXQ-810's 24
# steps that was 3 symbols and sizes 4-5 could never form.
#
# 60 steps is set from a MEASUREMENT, not from steps/e3_steps_per_tick. The
# V3-EXQ-834 smoke measured 5.50 symbols per 20-step episode = 0.275 symbols per
# env step, well ABOVE the 0.1 a perfectly-periodic E3 tick would give. That gap
# is the fix working: with the canonical StepHarness loop calling update_residue
# every step and num_hazards > 0, MECH-091 salient events fire phase_reset() and
# shorten the E3 interval below its nominal cadence. (810 measured exactly 3.00
# at 24 steps -- the periodic value -- precisely because it did neither.)
#
# So 60 steps predicts ~16 symbols per episode, roughly a 2x margin over the
# largest ceiling this lineage derives (8, at deliberation horizon 50). That
# margin is deliberate and bounded on BOTH sides: too few symbols makes long
# sequences structurally unreachable, but too many raises key DIVERSITY, and a
# specific n-gram then recurs across trials less often, which is what
# min_repetitions actually requires. The buffer is a GATED precondition, so an
# under-delivery self-routes substrate_not_ready_requeue rather than producing a
# false verdict on the ceilings.
N_EPISODES = 100
STEPS_PER_EPISODE = 60

# --- Chunking parameters ---------------------------------------------------
# Scaled down from the registered defaults (R_min 20 / W 100 / C_min 5) to fit a
# 100-episode probe, exactly as V3-EXQ-810 did: at 100 trials a 20-repetition
# bar inside a 100-trial window is structurally near-unsatisfiable. Ratios and
# every structural property are preserved; only absolute counts are reduced.
CHUNK_MIN_REPETITIONS = 5
CHUNK_WINDOW_TRIALS = 60
CHUNK_CRYSTALLISATION_MIN = 2

# SD-008: z_world fidelity. The default alpha_world=0.3 is a known root cause of
# degraded z_world, which the chunk outcome stream depends on only indirectly
# but which every agent-level probe in this substrate sets to 0.9.
ALPHA_WORLD = 0.9

# The all-position credit rule is ON in every arm of this lineage and is held
# CONSTANT -- it is not the independent variable. Trailing-only credit tallies at
# most one key per size per outcome and never tallies a leading sub-sequence,
# which starves the tally the growth rule has to read (ree-v3/CLAUDE.md, the
# credit-rule entry; V3-EXQ-810 readiness FAIL).
USE_ALL_POSITION_CREDIT = True

# Proposal injection OFF in every arm. E3 never sees a chunk, so chunk formation
# cannot perturb the action stream -- which makes the per-seed behavioural
# trajectory identical across every arm and isolates the ceiling manipulation
# completely. A probe must not perturb the behaviour a later run would measure.
USE_CHUNK_PROPOSAL_INJECTION = False

# MECH-322 sleep-replay carve-out OFF everywhere. The strict MECH-094 gate is an
# asserted safety property of every cell, never a manipulated one.
USE_CHUNK_REPLAY_ORIGIN_PATH = False

# MECH-324 maintenance ON and held constant. Nothing here contrasts it, so this
# lineage does not tag MECH-324.
USE_CHUNK_MAINTENANCE = True

# The inherited fiat constants the growable ceilings start from.
CHUNK_MIN_SIZE = 2
CHUNK_MAX_SIZE = 5
CHUNK_MAX_DEPTH = 3

# The baseline's deliberation budget. 50 rather than the anchor 30 on purpose:
# the baseline must sit at the SAME budget as the treatment arms so that the only
# difference between it and them is the growth flags. At horizon 30 the
# derivations return exactly the starting constants, so a horizon-30 baseline
# would be confounded with the anchor control.
CHUNK_DELIBERATION_HORIZON = 50


def env_kwargs() -> Dict[str, Any]:
    """Constructor kwargs for the lineage environment (seed added by caller)."""
    return dict(ENV_KWARGS)


def agent_kwargs(
    *,
    use_growable_chunk_ceiling: bool = False,
    use_growable_chunk_depth: bool = False,
    chunk_deliberation_horizon: int = CHUNK_DELIBERATION_HORIZON,
) -> Dict[str, Any]:
    """REEConfig.from_dims kwargs for one cell of this lineage.

    Defaults are the OFF path (both growable ceilings off). A treatment arm flips
    only the two flags and/or the shared budget; everything else is identical by
    construction, which is the property the fingerprint match depends on.
    """
    return {
        "alpha_world": ALPHA_WORLD,
        "use_policy_chunking": True,
        "use_chunk_maintenance": USE_CHUNK_MAINTENANCE,
        "use_chunk_proposal_injection": USE_CHUNK_PROPOSAL_INJECTION,
        "use_chunk_replay_origin_path": USE_CHUNK_REPLAY_ORIGIN_PATH,
        "use_chunk_all_position_credit": USE_ALL_POSITION_CREDIT,
        "chunk_min_repetitions": CHUNK_MIN_REPETITIONS,
        "chunk_window_trials": CHUNK_WINDOW_TRIALS,
        "chunk_crystallisation_min": CHUNK_CRYSTALLISATION_MIN,
        "chunk_min_size": CHUNK_MIN_SIZE,
        "chunk_max_size": CHUNK_MAX_SIZE,
        "chunk_max_depth": CHUNK_MAX_DEPTH,
        "use_growable_chunk_ceiling": bool(use_growable_chunk_ceiling),
        "use_growable_chunk_depth": bool(use_growable_chunk_depth),
        "chunk_deliberation_horizon": int(chunk_deliberation_horizon),
    }


def config_slice(
    *,
    use_growable_chunk_ceiling: bool = False,
    use_growable_chunk_depth: bool = False,
    chunk_deliberation_horizon: int = CHUNK_DELIBERATION_HORIZON,
    n_episodes: int = N_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    """Declared config slice for one cell's arm fingerprint.

    Only what the cell's computation READS: env kwargs, schedule, and the
    substrate-operating config. No arm labels, no acceptance thresholds, no
    seed list -- none of those change the draw, and excluding them is what lets a
    later sibling's OFF cell match this lineage's mint.
    """
    return {
        "baseline_id": CANONICAL_BASELINE_ID,
        "lineage": LINEAGE,
        "env_kwargs": env_kwargs(),
        "schedule": {
            "n_episodes": int(n_episodes),
            "steps_per_episode": int(steps_per_episode),
        },
        "agent": agent_kwargs(
            use_growable_chunk_ceiling=use_growable_chunk_ceiling,
            use_growable_chunk_depth=use_growable_chunk_depth,
            chunk_deliberation_horizon=chunk_deliberation_horizon,
        ),
    }


def off_path_config_slice(
    n_episodes: int = N_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
) -> Dict[str, Any]:
    """The declared OFF-path config slice -- both growable ceilings off."""
    return config_slice(
        use_growable_chunk_ceiling=False,
        use_growable_chunk_depth=False,
        chunk_deliberation_horizon=CHUNK_DELIBERATION_HORIZON,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
    )
