"""
Contract: the INV-091 driver-family standing default clears its own recorded floor.

substrate_queue.json sd_id INV091-NULL-VALIDATION-RUN-LENGTH: all three INV-091
cross-stream-similarity-band runs to date (V3-EXQ-827, 827a, 828) were executed at
EVAL_EPISODES=10 / STEPS_PER_EPISODE=150 (1500 eval-phase steps) and every one of
them found `null_validation.checked=False` -- the eval-phase array was too short
for `q081_surrogate.plan_blocks` to build a valid constrained-realisation null.
The worst POST-phase-sync-fix deficit recorded (V3-EXQ-828) was 2848 steps needed
against 1500 supplied (827's 12000-needed reading is excluded from the sizing
basis -- it was driven by the pre-fix lockstep clock-collapse bug that 827a's
redesign corrected, not a property of a correctly-built driver; see
`experiments/_lib/inv091_driver_defaults.py`'s module docstring for the full
per-run trace).

`experiments/_lib/inv091_driver_defaults.py` is the fix: a standing default eval
budget for any future successor script in this driver family, sized to clear the
worst recorded deficit with margin. This file pins two things:

  1. POSITIVE -- the shipped constants actually clear the recorded 2848-step floor
     (this is what future successor scripts inherit; a silent regression here
     would reintroduce the exact defect the substrate entry exists to fix).
  2. NEGATIVE CONTROL -- the guard is real, not vacuous: an eval budget sized
     to the OLD (undersized) 827/827a/828 numbers is correctly rejected by the
     same arithmetic check, proving the assertion in the defaults module would
     have caught the original defect had it existed before 827 was run.

ASCII-only output (repo rule).
"""

import importlib

import pytest

from experiments._lib import inv091_driver_defaults as defaults


def test_standing_default_clears_the_recorded_floor():
    """The shipped INV-091 driver defaults clear the worst recorded deficit (828)."""
    eval_steps_total = defaults.INV091_EVAL_EPISODES * defaults.INV091_STEPS_PER_EPISODE
    assert eval_steps_total == defaults.INV091_EVAL_STEPS_TOTAL
    assert defaults.INV091_MIN_EVAL_STEPS_REQUIRED == 2848
    assert eval_steps_total >= defaults.INV091_MIN_EVAL_STEPS_REQUIRED, (
        "INV091_EVAL_EPISODES * INV091_STEPS_PER_EPISODE (%d) no longer clears "
        "the recorded null-validation floor (%d) -- this reintroduces the "
        "V3-EXQ-827/827a/828 defect (null_validation.checked=False)."
        % (eval_steps_total, defaults.INV091_MIN_EVAL_STEPS_REQUIRED)
    )
    # A comfortable margin, not a bare pass -- 828's deficit varied 2464-2848
    # across its 6 arms, so the default should clear the top of that range with
    # room, not sit exactly on it.
    margin = eval_steps_total / defaults.INV091_MIN_EVAL_STEPS_REQUIRED - 1.0
    assert margin >= 0.10, (
        "standing default only clears the recorded floor by %.1f%% -- too thin "
        "a margin against arm-to-arm tau variance (828 ranged 2464-2848)."
        % (margin * 100.0)
    )


def test_warmup_and_step_length_unchanged_from_the_original_driver():
    """Only the EVAL budget was bumped -- warmup/step-length match 827/827a/828."""
    assert defaults.INV091_WARMUP_EPISODES == 40
    assert defaults.INV091_STEPS_PER_EPISODE == 150


def test_the_original_undersized_budget_would_fail_this_same_check():
    """NEGATIVE CONTROL: 827/827a/828's actual EVAL_EPISODES=10 fails the guard.

    Proves the check above is not vacuously true for any positive integer --
    it correctly rejects the exact undersized configuration that produced the
    confirmed V3-EXQ-827/827a/828 null_validation.checked=False defect.
    """
    old_eval_episodes = 10
    old_steps_per_episode = 150
    old_eval_steps_total = old_eval_episodes * old_steps_per_episode
    assert old_eval_steps_total == 1500
    assert old_eval_steps_total < defaults.INV091_MIN_EVAL_STEPS_REQUIRED


def test_module_self_check_fires_on_a_lowered_constant():
    """The module's own top-level assert is not dead code: executing a copy of
    its source with the eval budget forced back down to 827/827a/828's actual
    (undersized) EVAL_EPISODES=1 raises, rather than silently accepting a
    future edit that lowers the constant back down."""
    src_text = open(defaults.__file__).read()
    assert "INV091_EVAL_EPISODES = %d" % defaults.INV091_EVAL_EPISODES in src_text
    sabotaged_src = src_text.replace(
        "INV091_EVAL_EPISODES = %d" % defaults.INV091_EVAL_EPISODES,
        "INV091_EVAL_EPISODES = 1",
    )
    assert "INV091_EVAL_EPISODES = 1" in sabotaged_src
    sabotaged_code = compile(sabotaged_src, defaults.__file__ + "<sabotaged>", "exec")
    with pytest.raises(AssertionError):
        exec(sabotaged_code, {"__name__": "inv091_driver_defaults_sabotaged"})


def test_importable_as_documented_in_the_module_docstring():
    """The exact import statement the docstring tells future authors to use."""
    from experiments._lib.inv091_driver_defaults import (  # noqa: F401
        INV091_WARMUP_EPISODES,
        INV091_EVAL_EPISODES,
        INV091_STEPS_PER_EPISODE,
        INV091_MIN_EVAL_STEPS_REQUIRED,
    )
    # reload is cheap and confirms no import-time side effects beyond the assert
    importlib.reload(defaults)
