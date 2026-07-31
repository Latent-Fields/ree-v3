"""Standing default eval-phase episode/step budget for the INV-091 cross-stream-
similarity-band driver family (v3_exq_827 -> v3_exq_827a -> v3_exq_828 lineage).

WHY THIS MODULE EXISTS
-----------------------
All three runs to date were executed with WARMUP_EPISODES=40, EVAL_EPISODES=10,
STEPS_PER_EPISODE=150 (1500 eval-phase steps per arm), and every one of them found
`null_validation.checked=False` in its manifest: the constrained-realisation
surrogate null (`q081_surrogate.plan_blocks`, this package) requires the eval-phase
array to support blocks of at least `safety_factor * tau_max` steps (see
`q081_surrogate.py` "CHOOSING W"), and none of the three runs supplied enough steps
for that bound to clear. The recorded deficits, read from each manifest's
`null_validation.reason`:

    827  (pre phase-sync fix):        1500 supplied,  12000 needed
         -- period_max-dominated; driven by the un-fixed lockstep clock-collapse
         bug that 827a's redesign fixed (see 827a's module docstring). Not used
         as the sizing basis below -- it is a bug artefact, not a property of a
         correctly-built driver.
    827a (phase-sync redesign):       1099 supplied,   2576 needed  (tau_max=161)
    828  (remaining-ablations sweep): 1500 supplied, 2464-2848 needed (tau_max=178,
         worst of 6 arms)

Registered as `substrate_queue.json` sd_id `INV091-NULL-VALIDATION-RUN-LENGTH`
("bump the default run length for this driver family to >=2848 steps as a
standing default so future runs both compare arms AND establish significance
against chance"). 2848 -- the worst POST-phase-sync-fix deficit observed (828)
-- is the sizing basis; 827's 12000 is excluded per the note above.

WHAT THIS DOES AND DOES NOT CHANGE
-----------------------------------
This bumps the CALLER's run length so the existing q081_surrogate null-validation
threshold (`DEFAULT_SAFETY_FACTOR`, `DEFAULT_MIN_BLOCKS` in `q081_surrogate.py`,
both untouched here) can actually be met. It does not touch `q081_surrogate.py`,
any `ree_core/` substrate, or any already-run experiment's recorded result --
827/827a/828 stay exactly as they were reviewed and scored in claims.yaml.

USAGE
-----
Any future successor script in this driver family (a new `v3_exq_82Xz` or
`v3_exq_NNN_inv091_*` script, authored via `/queue-experiment`) should import
these constants instead of re-deriving WARMUP_EPISODES / EVAL_EPISODES /
STEPS_PER_EPISODE from scratch (827/827a/828 each hardcoded their own copy of
the same undersized numbers -- this module exists so the next one does not
repeat that):

    from experiments._lib.inv091_driver_defaults import (
        INV091_WARMUP_EPISODES, INV091_EVAL_EPISODES, INV091_STEPS_PER_EPISODE,
        INV091_MIN_EVAL_STEPS_REQUIRED,
    )

`dry_run` paths in existing scripts (DRY_WARMUP_EPISODES=2 / DRY_EVAL_EPISODES=2 /
DRY_STEPS_PER_EPISODE=20) are unaffected -- a smoke test does not need to clear
the null-validation bound, only to confirm the arms execute.
"""

from __future__ import annotations

# Eval-phase episode/step budget -- bumped 2026-07-31 (was EVAL_EPISODES=10 /
# 1500 total eval-phase steps in 827/827a/828).
INV091_WARMUP_EPISODES = 40            # unchanged from 827/827a/828
INV091_EVAL_EPISODES = 24              # was 10
INV091_STEPS_PER_EPISODE = 150         # unchanged from 827/827a/828

# Worst POST-phase-sync-fix deficit recorded across 827a/828 (see module
# docstring). The standing default above must clear this with margin.
INV091_MIN_EVAL_STEPS_REQUIRED = 2848

INV091_EVAL_STEPS_TOTAL = INV091_EVAL_EPISODES * INV091_STEPS_PER_EPISODE

assert INV091_EVAL_STEPS_TOTAL >= INV091_MIN_EVAL_STEPS_REQUIRED, (
    "INV091_EVAL_EPISODES * INV091_STEPS_PER_EPISODE (%d) fell below the "
    "recorded worst-case null-validation requirement (%d) -- see this module's "
    "docstring before lowering either constant."
    % (INV091_EVAL_STEPS_TOTAL, INV091_MIN_EVAL_STEPS_REQUIRED)
)
