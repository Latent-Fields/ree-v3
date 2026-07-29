"""Canonical CONTROL-arm baseline for the SD-084 / MECH-321-R4 MID-EXECUTION
REACHABILITY lineage (V3-EXQ-839 and successors).

WHY A NEW MODULE, AND NOT A REUSE OF THE mech321_* FAMILY
    The existing MECH-321 baselines (`mech321_policy_decomposition*`,
    `mech321_scale_resolved_probe`) all stand in the 816 dose-escalated
    harshened env, and their OFF arms differ from this one in the manipulation
    AND in the closure. This lineage asks a strictly narrower question -- is
    MECH-321's R4 mid-execution hook REACHABLE AT ALL under a standard
    `select_action -> update_residue` driver loop -- whose answer must not
    depend on env dose. So the closure here is the MINIMAL one in which the
    hook has ever been observed to fire: the configuration pinned by
    `tests/contracts/test_mech321_midexec_natural_reachability.py`, carried
    over verbatim. Editing an 816-family module in place to reach it would
    silently invalidate the reuse contract those lineages already minted.

THE ONE MANIPULATION
    `use_persistent_committed_program_handle` (SD-084), and nothing else.
    OFF reproduces the V3-EXQ-830 structural zero; ON makes gate (4) of the
    mid-execution conjunction satisfiable for the first time. Every other flag
    is held identical across both arms.

    NOTE, and it is why this lineage exists at all: the ON arm is NOT a pure
    diagnostic. A reachable mid-execution hook can newly reach `boundary.fired`,
    which feeds MECH-321's R1 OR trigger, and a mid-execution fire RELEASES THE
    COMMIT LATCH, aborting the remaining macro. The two arms therefore take
    different action sequences by design. See the SD doc's "NOT a pure
    diagnostic" section.

WHY A CRYSTALLISED CHUNK MUST BE REGISTERED, IN BOTH ARMS
    Gate (6) of the hook is `len(remaining) > 1`: the hook only fires on a tick
    FOLLOWING the commit of a MULTI-ACTION program. E3 commits a multi-action
    ARC-071 chunk only when that chunk beats the CEM-optimised candidates on raw
    score, so without a chunk in the library there is nothing multi-action to
    commit and the DV is a structural zero for a reason that has nothing to do
    with SD-084. Registering `SEEDED_CHUNK_SEQUENCE` is ordinary ARC-071 usage,
    is done identically in both arms, and is a PRECONDITION of the measurement
    rather than the manipulation. Length 3 is load-bearing: `_committed_step_idx`
    is 1 on the tick after a commit, so a 3-action program leaves exactly 2
    unexecuted actions -- one above gate (6)'s floor.

ENV SEEDING: THE ONE DEPARTURE FROM THE CONTRACT, AND IT IS LOAD-BEARING
    `CausalGridWorldV2.__init__` takes `seed: Optional[int] = None`, and the
    reachability contract constructs it as `CausalGridWorldV2()` -- UNSEEDED. For
    a contract that is fine: its assertions are existential and anti-vacuity
    guarded (`midexec > 0`, `multi_commits > 0`), so run-to-run variation cannot
    make it pass for the wrong reason.

    For an EXPERIMENT it is not fine, and measuring it is what caught it.
    `reset_all_rng(seed)` does NOT make an unseeded env reproducible. Measured
    2026-07-29 on darwin-arm64, running the OFF arm TWICE at the same seed
    (same flag, same code path -- a pure replicate):

        seed 11: 6/24 actions differed, decomp_n_evaluated_precommit 40 vs 84
        seed 23: 9/24 differed, 48 vs 39
        seed 47: 3/24 differed, 46 vs 56
        seed 71: 11/24 differed, 72 vs 73

    With `seed=seed` passed to the env, the same replicate gives **0/24 differing
    actions and identical counts on all four seeds**.

    WHY THIS IS NOT COSMETIC. Two things silently break without it:
      1. A PAIRED OFF-vs-ON behavioural delta measures NOTHING. Baseline
         run-to-run noise (3-11 ticks of 24) is the same size as, and
         indistinguishable from, any effect of the flag -- so an
         "action_divergence_frac" between arms would be attributed to SD-084
         while being mostly RNG. The same-arm replicate above is the control
         that makes that visible, and it is the control the design originally
         lacked.
      2. The ARM-REUSE FINGERPRINT would be a LIE. A cell emitted
         `reuse_eligible=True` promises it is a pure function of
         (substrate, config, seed); an unseeded env breaks exactly that, so a
         successor citing this mint would silently substitute a DIFFERENT
         baseline computation for its own.

    So the env is seeded per cell, and `env_seeded_per_cell` is declared in the
    fingerprint config slice (the SEED VALUE is not -- `arm_cell(seed, ...)`
    already keys the fingerprint on the seed, so putting it in the slice too
    would be redundant). Consequence to remember: the seed tiers below were
    RE-MEASURED under the seeded env, because seeding changes the dynamics and
    therefore changes which seeds commit multi-action programs.

TWO PARAMETER TRAPS, BOTH MEASURED 2026-07-29, BOTH RESOLVED HERE
    `DECOMPOSITION_DEPTH_CAP = 3` is the config DEFAULT and is non-degenerate.
    Do NOT lower it to 1 to make chunks survive decomposition: the substrate
    itself emits a UserWarning that depth_cap=1 is degenerate (every ARC-071
    chunk is minted at depth >= 1, so every triggering chunk is marked
    unreliable rather than re-tiled, and decomposition never fires at all).
    That would make the whole lineage read green for the wrong reason.

    `DECOMPOSITION_VS_THRESHOLD = 0.99` is the value the shipped MECH-321
    contracts use, and the sign convention is the opposite of the naive reading:
    `vs_trigger` is `region_vs < threshold`, so a HIGH threshold means
    NEAR-CERTAIN decomposition. The stated cost, carried openly: at 0.99 a
    triggering chunk is decomposed to SINGLE actions, leaving no remainder, so
    gate (6) fails on exactly those ticks even though gate (4) passes. It still
    works at depth_cap=3 because decomposition does not fire on every tick -- but
    the margin is thin, which is precisely why the consuming script instruments
    the multi-action-commit count PER CELL rather than assuming it.

SEED-DEPENDENCE IS A PROPERTY OF THIS BASELINE, NOT A DEFECT
    There is no chunk-selection-bias knob in the substrate, so whether a given
    seed ever commits a multi-action program is an emergent property of the
    CEM-vs-chunk score comparison. The seed tiers below are therefore NOT free
    parameters -- they are measurements, and the consuming script gates on them
    rather than trusting them.

    AUTHORITATIVE MEASUREMENT (under the SEEDED env -- see ENV SEEDING above;
    pre-seeding numbers are NOT comparable and were discarded).
    darwin-arm64-py3.13-torch2.12.0, 2 episodes x 60 steps, 2026-07-29,
    `--seed-scan` over ten candidates. `multi` = multi_action_commits:

        seed   OFF multi / ON multi   OFF midexec / ON midexec   precommit
        47         30 / 30                  0 / 29               343 / 343
        71         18 / 26                  0 / 26               279 / 350
         3         12 / 14                  0 / 14               130 / 144
        89          8 / 12                  0 / 12                95 / 120
        11          0 /  0                  0 /  0               121 / 121
        23          0 /  0                  0 /  0               146 / 146
        97          0 /  0                  0 /  0               156 / 156
        17          0 /  0                  0 /  0               106 / 106
        29          0 /  0                  0 /  0                86 /  86
        53          0 /  0                  0 /  0               202 / 202

    THREE THINGS THIS TABLE ESTABLISHES, none of which was visible before the
    env was seeded:
      1. OFF midexec is ZERO on all ten seeds -- the structural teardown, with
         no exceptions, including on seeds committing 30 multi-action programs.
      2. ON midexec is NON-ZERO on exactly the seeds that commit multi-action
         programs, and seed 47 is the cleanest case in the set: OFF and ON have
         IDENTICAL multi (30), total commits (72) and precommit (343), so the
         only thing that differs is whether the hook could see the program.
      3. On every NON-attributable seed the two arms are BIT-IDENTICAL in every
         recorded field. That is the negative-control prediction holding
         exactly: no multi-action commit -> gate (6) fails -> no latch release
         -> same behaviour. It is also why those seeds make a usable control
         rather than noise.

    MACHINE-CLASS CAVEAT, AND ONE PARTIAL CROSS-CLASS CONFIRMATION.
    The table is darwin-arm64. `torch.multinomial` is not cross-class
    reproducible and E3 samples that way in the live selection path, so which
    candidate wins a commit need not carry to `linux-x86_64` a priori; seeding
    the env buys determinism WITHIN a class, not across them.

    Measured on `linux-x86_64-py3.10-torch2.12.0+cpu` 2026-07-29, one cell before
    the scan was stopped to free a saturated fleet: seed 3 ARM_HANDLE_OFF gave
    multi=12, total_commits=27, midexec=0, precommit=130 -- IDENTICAL in all four
    fields to the darwin row above. Four exact matches in one cell is not
    coincidence, and it suggests the commit decision at this config is driven by
    the deterministic chunk-vs-CEM score comparison rather than by a multinomial
    draw. It is ONE cell of twelve, so it is encouraging rather than conclusive:
    the consuming script's readiness gate remains the backstop, and
    `--seed-scan` re-measures the whole pool in minutes on whichever box runs it.

TWO SEED TIERS, AND THE SECOND ONE IS A CONTROL, NOT PADDING
    SEED_POOL_ATTRIBUTABLE -- seeds measured to commit multi-action programs, so
        the mid-execution hook actually has something to re-evaluate. These
        carry the load-bearing criterion.
    SEED_POOL_NEGATIVE_CONTROL -- seeds measured to commit NONE, with pre-commit
        decomposition nonetheless live. These are included deliberately so the
        run DEMONSTRATES the attributability instrument instead of merely
        asserting it: on these seeds the DV must be zero in BOTH arms, and the
        two arms should not even diverge behaviourally (no mid-execution fire ->
        no latch release -> same action sequence). A reader can then see, inside
        one manifest, both that the handle makes the hook fire where there is
        something to re-evaluate AND that a zero is correctly attributable where
        there is not.

MINT / REUSE
    V3-EXQ-839 is the FIRST experiment of this lineage: its ARM_HANDLE_OFF
    cells are emitted reuse-ELIGIBLE (`include_driver_script_in_hash=False`)
    and constructed from THIS module, so a future same-config successor matches
    the fingerprint BY CONSTRUCTION and can skip re-running the control.
    Mint run_id: the V3-EXQ-839 run (cite via `reuse_baseline_from`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# --- Environment. The contract's default CausalGridWorldV2 dose, with ONE
# deliberate departure: the env is SEEDED PER CELL. See ENV SEEDING below.
# Otherwise plain default rather than an 816-family harshened dose -- a
# reachability answer that needed a particular env dose would not be the
# structural answer the autopsy asked for. ---
ENV_KWARGS: Dict[str, Any] = {}
ENV_SEEDED_PER_CELL = True

# --- Representation. self_dim / world_dim as pinned by the contract. ---
SELF_DIM = 32
WORLD_DIM = 32

# --- Schedule. Scaled up from the contract's 2 x 40 (which was sized to keep a
# unit test fast) so a per-cell count has statistical body rather than the
# contract's 1-to-8 evaluations. ---
EPISODES = 12
STEPS_PER_EPISODE = 60

# --- The seeded ARC-071 chunk. Length 3: see module docstring, gate (6). ---
SEEDED_CHUNK_SEQUENCE: Tuple[int, ...] = (0, 1, 2)
SEEDED_CHUNK_DEPTH = 1
SEEDED_CHUNK_SELECTION_WEIGHT = 1.0

# --- Decomposition parameters, held IDENTICAL in both arms. See the two traps
# in the module docstring before changing either. ---
DECOMPOSITION_VS_THRESHOLD = 0.99
DECOMPOSITION_DEPTH_CAP = 3

# --- Seed tiers. Re-measured per machine class by the consuming script's
# --seed-scan mode; the readiness gate refuses the run rather than trusting these
# lists, so a stale entry costs a refusal, never a false verdict. See the module
# docstring for the measurements behind each tier and why the second is a
# control rather than padding. ---
SEED_POOL_ATTRIBUTABLE: Tuple[int, ...] = (3, 47, 71, 89)
SEED_POOL_NEGATIVE_CONTROL: Tuple[int, ...] = (23, 53)
SEED_POOL: Tuple[int, ...] = SEED_POOL_ATTRIBUTABLE + SEED_POOL_NEGATIVE_CONTROL


def total_episodes() -> int:
    """Denominator M for the runner progress contract."""
    return EPISODES


def env_kwargs(seed: int) -> Dict[str, Any]:
    """Default CausalGridWorldV2 dose, SEEDED (see ENV SEEDING in the docstring)."""
    kwargs = dict(ENV_KWARGS)
    if ENV_SEEDED_PER_CELL:
        kwargs["seed"] = int(seed)
    return kwargs


def substrate_stack_flags() -> Dict[str, Any]:
    """The substrate stack shared by BOTH arms.

    All four of these are preconditions of the hook's gate conjunction, not the
    manipulation:
      use_policy_decomposition   -- gate (1); MECH-321 must actually run.
      use_event_segmenter        -- gate (3); the hook reads
                                    hippocampal.event_segmenter.
      use_policy_chunking        -- mints the ARC-071 chunk whose metadata
                                    satisfies gate (5) ("arc071_chunk").
      use_chunk_proposal_injection -- puts that chunk into the candidate set so
                                    it can be committed at all.
    """
    return {
        "use_event_segmenter": True,
        "use_policy_chunking": True,
        "use_chunk_proposal_injection": True,
        "use_policy_decomposition": True,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
    }


def off_arm_flags() -> Dict[str, Any]:
    """ARM_HANDLE_OFF: the legacy path.

    Gate (4) reads `e3._committed_trajectory` alone, which
    `post_action_update` destroys as its last statement on every tick -- so the
    mid-execution hook is structurally unreachable and the DV is exactly the
    V3-EXQ-830 zero. This is the negative control, and it is what gives the ON
    arm's reading content.
    """
    flags = substrate_stack_flags()
    flags["use_persistent_committed_program_handle"] = False
    return flags


def on_arm_flags() -> Dict[str, Any]:
    """ARM_HANDLE_ON: the SD-084 manipulation, and the only difference."""
    flags = substrate_stack_flags()
    flags["use_persistent_committed_program_handle"] = True
    return flags


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical ARM_HANDLE_OFF fingerprint config slice.

    Used by BOTH the V3-EXQ-839 mint and any successor's `try_reuse_cell(...)`,
    so the fingerprints match by construction. Declares only what the control
    computation reads -- never the ON-arm flag, never acceptance thresholds.
    """
    slice_: Dict[str, Any] = {
        "env": dict(ENV_KWARGS),
        # The FACT of per-cell env seeding is part of the closure; the seed VALUE
        # is not -- arm_cell(seed, ...) already keys the fingerprint on it.
        "env_seeded_per_cell": ENV_SEEDED_PER_CELL,
        "schedule": {
            "episodes": EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "seeded_chunk_depth": SEEDED_CHUNK_DEPTH,
        "seeded_chunk_selection_weight": SEEDED_CHUNK_SELECTION_WEIGHT,
    }
    slice_.update(off_arm_flags())
    return slice_


def off_needed_keys() -> List[str]:
    """Control-arm metrics a consumer reads back from a reused cell."""
    return [
        "decomp_n_evaluated_midexec",
        "decomp_n_evaluated_precommit",
        "decomp_n_decomposed_midexec",
        "decomp_n_decomposed_precommit",
        "multi_action_commits",
        "total_commits",
        "n_ticks",
        "net_harm_per_step",
        "beta_elevated_frac",
        "committed_run_len_mean",
        "action_sequence",
    ]
