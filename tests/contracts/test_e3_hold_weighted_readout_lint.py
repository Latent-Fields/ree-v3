"""Contracts for the hold-weighted-E3-readout lint -- pseudo-replication defect FORM 2.

Surfaces under test:
  (1) validate_experiments.e3_hold_weighted_readout_lint -- flags a driver that
      accumulates a per-env-step statistic from the select_action RETURN VALUE, from
      agent.e3._last_selected_trajectory, or from the e3_tick-gated candidate list.
  (2) validate_experiments.py --checks e3_hold_weighted_readout -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES.

WHY THIS GATE EXISTS, AND WHY IT IS SEPARATE FROM test_e3_diagnostics_staleness_lint.
`ree_core/agent.py:5430` returns the HELD action on
`if not ticks["e3_tick"] and self._last_action is not None:` -- BEFORE the only
`e3.select()` call site is reached. `agent.generate_trajectories` (`agent.py:4812`)
likewise returns CACHED candidates on a non-E3 tick (the MECH-057a gate). So the value a
driver gets back from `select_action(...)` is UNCHANGED across a whole hold, and any
per-step statistic accumulated from it is weighted by HOLD DURATION rather than counted
per selection. Cadence defaults to 10 (`ree_core/utils/config.py:2017`) and varies 5-20
under MECH-093 arousal modulation (`ree_core/heartbeat/clock.py:52-70`).

CRUCIALLY, this touches NO diagnostics latch, so the form-1 gate is structurally blind to
it. Established by the V3-EXQ-699 re-adjudication (REE_assembly `ac2fb64028`): form 1
fired on `:929` (`last_score_diagnostics`, incidental) and was SILENT on `:882` -- the
run's PRIMARY DV, and the site that forced withdrawal of the `levers_compound` finding.
Empirical confirmation that these are one defect: 699's `selected_class_entropy_nats`
equalled `committed_class_entropy_nats` to 6dp on all 12 arm-seeds.

The two gates are kept SEPARATE, with separate pins, on purpose. Freshness and
replication are independent defects and conflating them mis-adjudicates in BOTH
directions: 699's `active_frac == 1.0` is INFORMATIVE precisely because its diagnostics
are fresh, where 708's identical 1.0 was vacuous (autopsy 699 sec 11.2).

SCOPE. This gates NEW scripts. Completed runs are re-adjudicated via `/failure-autopsy`,
never rewritten -- hence WARN-only, and hence the fire-rate pin below is a BACKLOG SIZE,
not a target of zero.
"""
import ast
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
        return V.e3_hold_weighted_readout_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) the three defect shapes ----------------------------------------------------

_DEFECT_RETURN_VALUE = '''
def main():
    counts = {}
    for step in range(100):
        action = agent.select_action(candidates, ticks)
        committed_class = int(action[0].argmax().item())
        counts[committed_class] = counts.get(committed_class, 0) + 1

if __name__ == "__main__":
    main()
'''

_DEFECT_SELECTED_TRAJECTORY = '''
def main():
    for step in range(100):
        action = agent.select_action(candidates, ticks)
        sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
        if sel_traj is not None:
            rows.append(sel_traj)

if __name__ == "__main__":
    main()
'''

_DEFECT_CANDIDATES = '''
def main():
    n_pre_ge2 = 0
    for step in range(100):
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        pre_classes = sorted({_class_of(t) for t in candidates})
        if len(pre_classes) >= 2:
            n_pre_ge2 += 1
        action = agent.select_action(candidates, ticks)

if __name__ == "__main__":
    main()
'''


def test_hw_select_action_return_value_is_flagged():
    """(a) the shape that carried V3-EXQ-699's PRIMARY DV."""
    out = _lint_src(_DEFECT_RETURN_VALUE)
    assert out is not None and "HOLD-WEIGHTED E3 READOUT" in out, out
    assert "select_action() return value" in out, out


def test_hw_selected_trajectory_read_is_flagged():
    """(b) `_last_selected_trajectory` latches like the form-1 six -- the READ is enough."""
    out = _lint_src(_DEFECT_SELECTED_TRAJECTORY)
    assert out is not None and "_last_selected_trajectory" in out, out


def test_hw_e3_tick_gated_candidates_are_flagged():
    """(c) generate_trajectories returns CACHED candidates between E3 ticks."""
    out = _lint_src(_DEFECT_CANDIDATES)
    assert out is not None and "candidate list" in out, out


def test_hw_reports_the_offending_line_numbers():
    out = _lint_src(_DEFECT_RETURN_VALUE)
    assert "line(s)" in out, out


# ---- (2) PRECISION: the shapes that must NOT fire ------------------------------------
# These are the reason the rule requires a SCALAR REDUCTION rather than a mere mention.
# A first cut propagated taint through any mention of the action and fired on most of the
# corpus for a non-defect. Both guards below pin that correction.

_REPLAY_BUFFER = '''
def main():
    for step in range(100):
        action = agent.select_action(candidates, ticks)
        obs_dict, r, done, info = env.step(action)
        transition_buffer.append((z0_prev, action, z1_obs))
        agent.record_transition(z_self_prev, action, latent.z_self)

if __name__ == "__main__":
    main()
'''


def test_hw_replay_buffer_of_the_action_does_not_fire():
    """STORING the action every env step is CORRECT -- the held action IS the action taken.

    Only a per-step STATISTIC derived from it is hold-weighted. If this ever starts
    firing, the rule has lost the distinction it exists to draw and will flag nearly the
    whole corpus for a non-defect.
    """
    assert _lint_src(_REPLAY_BUFFER) is None


def test_hw_env_step_unpacking_does_not_propagate_taint():
    """REGRESSION GUARD for the over-propagation bug found while building this lint.

    `obs, r, done, info = env.step(action)` mentions the tainted action, so a
    mention-based rule tainted the ENTIRE driver -- measured on v3_exq_785: `agent`,
    `cfg`, `done`, `info` and `latent` all marked, and the lint fired on unrelated helper
    lines. The env's response to a held action is a GENUINE per-step measurement, not a
    replicated one, so taint must not cross tuple unpacking.
    """
    tree = ast.parse(_REPLAY_BUFFER)
    tainted, _ = V._e3_cadence_gated_sources(tree)
    for name in ("done", "info", "obs_dict", "r"):
        assert name not in tainted, (
            f"taint crossed tuple unpacking into `{name}` -- env.step()'s return is a "
            f"fresh per-step observation, not a replicated one")


def test_hw_scalar_ness_is_a_property_of_the_variable_not_the_use_site():
    """699 reduces at :882 and accumulates at :899 -- seventeen lines apart.

    A rule demanding a reduction AT the accumulation site missed the run's primary DV
    entirely. `action` (a tensor) must not be scalar; `committed_class` must be.
    """
    tree = ast.parse(_DEFECT_RETURN_VALUE)
    tainted, scalars = V._e3_cadence_gated_sources(tree)
    assert "action" in tainted and "action" not in scalars
    assert "committed_class" in scalars


# ---- (3) exemptions ------------------------------------------------------------------

_CLEARED = '''
def main():
    counts = {}
    for step in range(100):
        agent.e3.last_score_diagnostics = None
        action = agent.select_action(candidates, ticks)
        committed_class = int(action[0].argmax().item())
        counts[committed_class] = counts.get(committed_class, 0) + 1

if __name__ == "__main__":
    main()
'''

_TICK_GUARDED = '''
def main():
    counts = {}
    for step in range(100):
        action = agent.select_action(candidates, ticks)
        if not ticks["e3_tick"]:
            continue
        committed_class = int(action[0].argmax().item())
        counts[committed_class] = counts.get(committed_class, 0) + 1

if __name__ == "__main__":
    main()
'''

_DIRECT_SELECT = '''
def main():
    counts = {}
    for step in range(100):
        agent.e3.select(candidates)
        action = agent.select_action(candidates, ticks)
        committed_class = int(action[0].argmax().item())
        counts[committed_class] = counts.get(committed_class, 0) + 1

if __name__ == "__main__":
    main()
'''

_TRAJ_CLEARED = '''
def main():
    for step in range(100):
        agent.e3._last_selected_trajectory = None
        action = agent.select_action(candidates, ticks)
        sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
        if sel_traj is not None:
            rows.append(sel_traj)

if __name__ == "__main__":
    main()
'''


def test_hw_clear_before_select_is_exempt():
    assert _lint_src(_CLEARED) is None


def test_hw_tick_guard_is_exempt():
    assert _lint_src(_TICK_GUARDED) is None


def test_hw_direct_select_call_is_exempt():
    assert _lint_src(_DIRECT_SELECT) is None


def test_hw_clearing_the_trajectory_latch_is_exempt():
    """The 699b repair idiom: `_last_selected_trajectory = None` before every select."""
    assert _lint_src(_TRAJ_CLEARED) is None


def test_hw_no_loop_is_exempt():
    """No loop -> no pseudo-replication, exactly as in form 1."""
    src = _DEFECT_RETURN_VALUE.replace("for step in range(100):", "if True:")
    assert _lint_src(src) is None


def test_hw_non_driver_script_is_exempt():
    assert _lint_src('def summarise(m):\n    return m["committed_class_entropy"]\n') is None


def test_hw_explicit_opt_out_is_honoured():
    src = 'E3_HOLD_WEIGHTED_READOUT_EXEMPT = "reads once per selection"\n' + _DEFECT_RETURN_VALUE
    assert _lint_src(src) is None


# ---- (4) the substrate premises, asserted rather than assumed -------------------------

def test_hw_selected_trajectory_is_select_only_in_the_substrate():
    """`_last_selected_trajectory` must be ASSIGNED only inside E3Selector.select().

    That is what makes it latch. If a refactor adds a reset path it stops latching and
    these fires become false positives -- this fails first and says so. (`:3224` reads it
    in post_action_update; a READ is fine, only assignment matters.)
    """
    sel = REPO_ROOT / "ree_core" / "predictors" / "e3_selector.py"
    tree = ast.parse(sel.read_text(encoding="utf-8"))
    funcs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            last = max((getattr(x, "lineno", n.lineno) for x in ast.walk(n)),
                       default=n.lineno)
            funcs.append((n.lineno, last, n.name))

    def _owner(lineno):
        c = [f for f in funcs if f[0] <= lineno <= f[1]]
        return min(c, key=lambda f: f[1] - f[0])[2] if c else "<module>"

    owners = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and t.attr == "_last_selected_trajectory":
                    owners.append(_owner(t.lineno))
    assert owners, "_last_selected_trajectory is never assigned -- stale lint entry?"
    assert set(owners) <= {"select", "__init__"}, (
        f"_last_selected_trajectory is assigned outside select() ({sorted(set(owners))}). "
        f"It no longer latches unconditionally, so the lint's premise fails for it.")


def test_hw_agent_returns_held_action_before_select_is_reached():
    """THE mechanism. `select_action` must return early on `not ticks["e3_tick"]`.

    If this guard moves or is removed, the whole defect class changes shape and every
    verdict derived from it needs revisiting. Pinned against the substrate so a refactor
    cannot silently invalidate the lint, the 699/708 autopsies, and the corpus triage.
    """
    src = (REPO_ROOT / "ree_core" / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "select_action"), None)
    assert fn is not None, "agent.select_action no longer exists"
    end = max(getattr(x, "lineno", fn.lineno) for x in ast.walk(fn))
    body = "\n".join(src.splitlines()[fn.lineno - 1:end])
    assert 'not ticks["e3_tick"]' in body, (
        "select_action no longer guards on `not ticks[\"e3_tick\"]` -- the hold-weighting "
        "mechanism this lint encodes may no longer hold")


def test_hw_generate_trajectories_caches_between_ticks():
    """The (c) premise: candidates are e3_tick-gated (MECH-057a)."""
    src = (REPO_ROOT / "ree_core" / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "generate_trajectories"), None)
    assert fn is not None, "agent.generate_trajectories no longer exists"
    end = max(getattr(x, "lineno", fn.lineno) for x in ast.walk(fn))
    body = "\n".join(src.splitlines()[fn.lineno - 1:end])
    assert 'not ticks["e3_tick"]' in body, (
        "generate_trajectories no longer caches between E3 ticks -- exposure (c) may no "
        "longer hold")


# ---- (5) the message must carry the TRIAGE TEST, not just a flag ---------------------

def test_hw_message_carries_the_triage_test():
    """REGRESSION GUARD. An inflated n is NOT sufficient for contamination.

    The 699 and 708 autopsies both turned on this: 699's PASS SURVIVED on
    threshold-invariant readiness gates while its FINDING was withdrawn. A message that
    said only "this is contaminated" would have mis-adjudicated the PASS. If someone
    later trims this message to a bare flag, that distinction is lost and the gate starts
    manufacturing false withdrawals.
    """
    out = _lint_src(_DEFECT_RETURN_VALUE)
    assert "TRIAGE" in out, out
    assert "threshold-invariant" in out, out
    assert "DISQUALIFYING" in out and "entropy" in out, out
    assert "saturated at 1.0" in out, out


def test_hw_message_states_the_limits_of_the_663_calibration():
    """The <1% sign-varying 663 replay must never be cited as a general bound.

    It measured the artifact WHERE IT CANCELS (arm-symmetric, continuous-magnitude DV).
    699 sits where it does not: an entropy DV, arms differing in hold duration, and a
    flip needing only 0.115-0.134 nats against 0.51-1.10 -- 1-2 orders of magnitude ABOVE
    the 663 artifact scale (autopsy sec 4d). Quoting the reassuring half alone is the
    specific misreading this guards.
    """
    out = _lint_src(_DEFECT_RETURN_VALUE)
    assert "663" in out, out
    assert "hold duration" in out, out


def test_hw_message_names_the_repair_and_the_telemetry_convention():
    out = _lint_src(_DEFECT_RETURN_VALUE)
    assert "n_fresh_select" in out, out
    assert "785a" in out, out


# ---- (6) form 1 and form 2 are INDEPENDENT gates --------------------------------------

def test_hw_form1_is_blind_to_the_return_value_shape():
    """The defining property: form 1 CANNOT see this, which is why form 2 exists.

    If form 1 ever starts firing on this shape, these two pins are measuring overlapping
    things and the separation argument (autopsy sec 11.2) needs revisiting.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECT_RETURN_VALUE)
        name = f.name
    try:
        assert V.e3_diagnostics_staleness_lint(Path(name)) is None, (
            "form 1 now sees the return-value shape -- re-examine the two-gate split")
        assert V.e3_hold_weighted_readout_lint(Path(name)) is not None
    finally:
        os.unlink(name)


def test_hw_real_699_is_the_detection_witness():
    """The worked instance. All THREE exposure roots must be named.

    Autopsy sec 2 table: :882/:899 (primary DV), :913 (_last_selected_trajectory),
    :856 (pre_e3_classes). Only :929 was visible to form 1.
    """
    real = EXPERIMENTS_DIR / "v3_exq_699_pcomp_demotion_x_gonogo_composition.py"
    if not real.exists():
        return
    out = V.e3_hold_weighted_readout_lint(real)
    assert out is not None and "HOLD-WEIGHTED E3 READOUT" in out, out
    for root in ("select_action() return value", "_last_selected_trajectory",
                 "candidate list"):
        assert root in out, f"missing exposure root {root!r} on the 699 witness: {out}"


def test_hw_real_785a_reference_impl_is_clean():
    """The clear-before-select reference must not fire, or the WARN is unactionable."""
    real = EXPERIMENTS_DIR / "v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py"
    if not real.exists():
        return
    assert V.e3_hold_weighted_readout_lint(real) is None


# ---- (7) invariants: WARN-only, and the backlog does not silently grow ---------------

def test_hw_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECT_RETURN_VALUE)
        name = f.name
    try:
        r = _run("--checks", "e3_hold_weighted_readout", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "HOLD-WEIGHTED E3 READOUT" in r.stdout
    finally:
        os.unlink(name)


# Pinned 2026-07-20 against the TRACKED v3_exq_*.py corpus (1093 scripts): 150 fire, of
# which 109 have a landed manifest (completed runs, re-adjudication candidates) and 41
# have not run. 91 of the 150 are INVISIBLE to the form-1 gate -- that gap is the whole
# reason this lint exists.
#
# This is a BACKLOG SIZE, not a target of zero. Completed runs are deliberately NOT
# retro-edited (a completed run's pre-registered emission is not rewritten; it is
# re-adjudicated via /failure-autopsy). The pin exists so a NEW script carrying the
# defect shows up as a rise, and so a later widening of the rule announces its own blast
# radius instead of drifting silently.
#
# NOTE ON LOCAL FAILURES. This test globs the DIRECTORY, so an UNTRACKED script in a
# working tree counts, and both this pin and the form-1 one can read high on a machine
# with in-flight work. Observed live while pinning: another session's untracked
# `v3_exq_699b_pcomp_demotion_x_gonogo_fresh_select.py` (the corrected 699 successor)
# fired BOTH gates at 05:34Z, taking form-1 to 64 against its pin of 63, and was repaired
# by its owning session at 05:46Z -- after which both gates read clean and both pins hold.
# So: if this test reports 151 where a worker reports 150, run
# `git status --porcelain experiments/` FIRST. An untracked in-flight script is not a
# corpus change and must not be re-pinned around. (Deliberately not filtered through
# `git ls-files`: `scripts/remote_pytest.sh` excludes `.git/`, so a git-based check fails
# CLOSED on a worker -- the documented phantom-failure trap that bit `validate_queue`.)
_PINNED_CORPUS_FIRE_COUNT = 150


def test_hw_corpus_fire_rate_is_pinned():
    fired = [p for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
             if V.e3_hold_weighted_readout_lint(p) is not None]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"hold-weighted-readout fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(gate the accumulation on a fresh selection) rather than re-pinning. If an "
        f"UNTRACKED in-flight script is present, that is not a corpus change -- see the "
        f"note above. If you deliberately widened or narrowed the rule, re-pin and say "
        f"so in the commit message. Fired: {[p.name for p in fired]}")
