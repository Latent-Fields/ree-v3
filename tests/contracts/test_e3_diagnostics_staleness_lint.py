"""Contracts for the stale-E3-diagnostics lint.

Surfaces under test:
  (1) validate_experiments.e3_diagnostics_staleness_lint -- flags a driver that reads
      an E3 `last_*` diagnostic inside a per-env-step loop without clearing the latch,
      which silently pseudo-replicates one selection into many "independent" rows.
  (2) validate_experiments.py --checks e3_diagnostics_staleness -- the selector, and
      the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. `ree_core/predictors/e3_selector.py` sets
last_score_diagnostics / last_score_decomp / last_channel_terms / last_scores /
last_precommit_probs ONLY inside `select()`. The attributes LATCH: on a tick where
`select()` did not run they still hold the PREVIOUS selection. A driver reading them
once per env step therefore records the same selection repeatedly. Nothing raises --
the run just reports a sample size it does not have. Confirmed 2026-07-19 on the
V3-EXQ-785 config: 600 recorded rows behind 67 genuine `select()` calls (~9.0x).

MECHANISM -- the widely-assumed cause is WRONG, and pinning the correction is half the
point of this file. The skip is NOT `beta_gate.is_elevated`. `ree_core/agent.py`
returns the held/stepped action on `if not ticks["e3_tick"] and self._last_action is
not None:` BEFORE the only `e3.select()` call site; `beta_gate.is_elevated` merely picks
step-vs-hold WITHIN an already-skipped tick. The real driver is the E3 CADENCE
(`heartbeat.e3_steps_per_tick`, default 10). So "commitment was effectively disabled for
this run" does NOT exculpate a driver -- see test_e3s_message_corrects_the_beta_gate_myth.

Reference implementation (the idiom this gate asks for):
experiments/v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py -- clears before
every select_action(...), records only on repopulation, and emits the skipped-tick
count as `n_latched_ticks` (real run: 1757 genuine selections from 15000 ticks).

SCOPE. This gates NEW scripts. The landed corpus carries a large backlog whose runs are
already complete, and per the 2026-07-19 decision a completed run's pre-registered
emission is not rewritten -- hence WARN-only, and hence the fire-rate pin below is a
BACKLOG SIZE, not a target of zero.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.e3_diagnostics_staleness_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) the defect shape -----------------------------------------------------------

_DEFECTIVE = '''
def main():
    for step in range(100):
        agent.select_action(obs)
        diag = agent.e3.last_score_diagnostics
        rows.append(diag)

if __name__ == "__main__":
    main()
'''

_DEFECTIVE_GETATTR = '''
def main():
    for step in range(100):
        agent.select_action(obs)
        diag = getattr(agent.e3, "last_score_diagnostics", {}) or {}
        rows.append(diag)

if __name__ == "__main__":
    main()
'''


def test_e3s_bare_loop_read_is_flagged():
    out = _lint_src(_DEFECTIVE)
    assert out is not None and "STALE E3 DIAGNOSTICS" in out, out
    assert "last_score_diagnostics" in out, out


def test_e3s_getattr_form_is_flagged():
    """The confirmed real-world spelling (V3-EXQ-722 line 1019) is the getattr form."""
    out = _lint_src(_DEFECTIVE_GETATTR)
    assert out is not None and "STALE E3 DIAGNOSTICS" in out, out


def test_e3s_reports_the_offending_line_numbers():
    out = _lint_src(_DEFECTIVE)
    assert "line(s)" in out, out


# ---- (2) the three exemptions -------------------------------------------------------

_CLEARED = '''
def main():
    for step in range(100):
        agent.e3.last_score_diagnostics = None
        agent.select_action(obs)
        diag = agent.e3.last_score_diagnostics
        if diag is None:
            n_latched += 1
            continue
        rows.append(diag)

if __name__ == "__main__":
    main()
'''

_TICK_GUARDED = '''
def main():
    for step in range(100):
        agent.select_action(obs)
        if not ticks["e3_tick"]:
            continue
        rows.append(agent.e3.last_score_diagnostics)

if __name__ == "__main__":
    main()
'''

_DIRECT_SELECT = '''
def main():
    for step in range(100):
        agent.e3.select(candidates)
        rows.append(agent.e3.last_score_diagnostics)
        agent.select_action(obs)

if __name__ == "__main__":
    main()
'''


def test_e3s_clear_before_select_is_exempt():
    """(a) the reference idiom -- the fix the WARN actually asks for."""
    assert _lint_src(_CLEARED) is None


def test_e3s_tick_guard_is_exempt():
    """(b) the driver already knows about the E3 cadence."""
    assert _lint_src(_TICK_GUARDED) is None


def test_e3s_direct_select_call_is_exempt():
    """(c) the driver drives selection itself, so each read follows a fresh select."""
    assert _lint_src(_DIRECT_SELECT) is None


_ONE_SHOT = '''
def main():
    agent.select_action(obs)
    diag = agent.e3.last_score_diagnostics
    record(diag)

if __name__ == "__main__":
    main()
'''

_NO_DRIVER = '''
def summarise(manifest):
    return manifest["e3"]["last_score_diagnostics"]
'''


def test_e3s_one_shot_read_outside_a_loop_is_exempt():
    """No loop -> no pseudo-replication. Only the driver-loop shape inflates n."""
    assert _lint_src(_ONE_SHOT) is None


def test_e3s_non_driver_script_is_exempt():
    """A script that never calls select_action is not driving the agent."""
    assert _lint_src(_NO_DRIVER) is None


def test_e3s_explicit_opt_out_is_honoured():
    src = 'E3_DIAGNOSTICS_STALENESS_EXEMPT = "reads once per episode"\n' + _DEFECTIVE
    assert _lint_src(src) is None


# ---- (3) the mechanism correction ---------------------------------------------------

def test_e3s_message_corrects_the_beta_gate_myth():
    """REGRESSION GUARD for the correction itself.

    The widely-assumed cause is beta_gate.is_elevated. It is not -- agent.py returns
    early on `not ticks["e3_tick"]` BEFORE select() is reached. If someone later
    "simplifies" this lint to key on the beta gate, or softens the message to imply
    that disabled commitment makes a driver safe, that is the defect coming back.
    """
    out = _lint_src(_DEFECTIVE)
    assert "e3_steps_per_tick" in out, out
    assert "beta_gate.is_elevated" in out, "message must name and refute the myth"
    assert "does NOT exculpate" in out, out


def test_e3s_message_names_the_reference_and_the_telemetry_convention():
    out = _lint_src(_DEFECTIVE)
    assert "n_latched_ticks" in out, out
    assert "785a" in out, out


# ---- (4) real-corpus witnesses ------------------------------------------------------

def test_e3s_real_785_is_the_detection_witness():
    """Pins the confirmed defect against the real file, not just a synthetic shape."""
    real = EXPERIMENTS_DIR / "v3_exq_785_mech463_arousal_variance_amplifier_decomp.py"
    if not real.exists():
        return
    out = V.e3_diagnostics_staleness_lint(real)
    assert out is not None and "STALE E3 DIAGNOSTICS" in out, out


def test_e3s_real_785a_reference_impl_is_clean():
    """The fixed sibling must NOT fire -- otherwise the WARN is unactionable."""
    real = EXPERIMENTS_DIR / "v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py"
    if not real.exists():
        return
    assert V.e3_diagnostics_staleness_lint(real) is None


# ---- (5) invariants: WARN-only, and the backlog does not silently grow ---------------

def test_e3s_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE)
        name = f.name
    try:
        r = _run("--checks", "e3_diagnostics_staleness", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "STALE E3 DIAGNOSTICS" in r.stdout
    finally:
        os.unlink(name)


# Pinned 2026-07-19 against the v3_exq_*.py corpus (66 at first measurement, 65 after
# fixing V3-EXQ-722 -- the one carrier that had never run, so fixing it cost nothing).
# This is a BACKLOG SIZE, not a
# target -- the landed scripts have all run and are deliberately NOT retro-edited
# (a completed run's pre-registered emission is not rewritten). The pin exists so a
# NEW script carrying the defect shows up as a rise, and so a later widening of the
# rule announces its own blast radius instead of drifting silently.
_PINNED_CORPUS_FIRE_COUNT = 65


def test_e3s_corpus_fire_rate_is_pinned():
    fired = [p for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
             if V.e3_diagnostics_staleness_lint(p) is not None]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"stale-E3-diagnostics fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(clear-before-select) rather than re-pinning. If you deliberately widened or "
        f"narrowed the rule, re-pin and say so in the commit message. "
        f"Fired: {[p.name for p in fired]}")
