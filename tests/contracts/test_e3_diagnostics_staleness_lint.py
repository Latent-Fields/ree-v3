"""Contracts for the stale-E3-diagnostics lint.

Surfaces under test:
  (1) validate_experiments.e3_diagnostics_staleness_lint -- flags a driver that reads
      an E3 `last_*` diagnostic inside a per-env-step loop without clearing the latch,
      which silently pseudo-replicates one selection into many "independent" rows.
  (2) validate_experiments.py --checks e3_diagnostics_staleness -- the selector, and
      the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).

WHY THIS GATE EXISTS. `ree_core/predictors/e3_selector.py` sets all SIX of
last_score_diagnostics / last_score_decomp / last_channel_terms / last_scores /
last_precommit_probs / last_raw_scores ONLY inside `select()`. The attributes LATCH: on a tick where
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


_IDENTITY_GUARDED = '''
def main():
    prev_probs_id = None
    for step in range(100):
        agent.select_action(obs)
        probs = getattr(agent.e3, "last_precommit_probs", None)
        pid = id(probs) if probs is not None else None
        fresh = probs is not None and pid != prev_probs_id
        prev_probs_id = pid
        if fresh:
            rows.append(probs)

if __name__ == "__main__":
    main()
'''

_DEFECTIVE_RAW_SCORES = '''
def main():
    for step in range(100):
        agent.select_action(obs)
        raw = getattr(agent.e3, "last_raw_scores", None)
        rows.append(raw)

if __name__ == "__main__":
    main()
'''


def test_e3s_identity_freshness_guard_is_exempt():
    """Exemption (d): dedupe by object identity discharges the same obligation as (a).

    A latched read hands back the SAME object on a skipped tick; a real select()
    allocates a new one. Gating the append on id(...) changing therefore admits exactly
    the fresh selections. This is the shape v3_exq_777 uses, and it was the ONE false
    positive in the 2026-07-19 landed corpus of 65.
    """
    assert _lint_src(_IDENTITY_GUARDED) is None


def test_e3s_identity_guard_does_not_blanket_exempt_unrelated_id_calls():
    """Guard against (d) being over-broad: id() on a NON-latched value must not exempt.

    23 of the landed 65 call id() for unrelated reasons -- 708 and 709 among them, both
    CONFIRMED carriers. If (d) keyed on the mere presence of id() it would silently
    exempt them and the backlog would collapse for a bookkeeping reason.
    """
    src = _DEFECTIVE.replace("rows.append(diag)",
                             "prev = id(obs)\n        rows.append((diag, prev != 0))")
    out = _lint_src(src)
    assert out is not None and "STALE E3 DIAGNOSTICS" in out, out


def test_e3s_covers_last_raw_scores():
    """last_raw_scores latches like the other five (e3_selector.py:2103, inside select()).

    It was absent from _E3_LATCHED_ATTRS until 2026-07-19. Not academic: V3-EXQ-722
    carried TWO latched reads and this was the second, with a comment asserting exactly
    the assumption the defect refutes.
    """
    assert "last_raw_scores" in V._E3_LATCHED_ATTRS
    out = _lint_src(_DEFECTIVE_RAW_SCORES)
    assert out is not None and "last_raw_scores" in out, out


def test_e3s_all_six_latched_attrs_are_select_only_in_the_substrate():
    """The lint's premise, asserted against the substrate rather than assumed.

    Every name in _E3_LATCHED_ATTRS must be assigned ONLY inside E3Selector.select().
    If a future refactor adds an __init__ default or a reset path, that attribute stops
    latching and its fires become false positives -- this fails first and says so.
    """
    import ast as _ast
    sel = REPO_ROOT / "ree_core" / "predictors" / "e3_selector.py"
    tree = _ast.parse(sel.read_text(encoding="utf-8"))
    funcs = []
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            last = max((getattr(x, "lineno", n.lineno) for x in _ast.walk(n)),
                       default=n.lineno)
            funcs.append((n.lineno, last, n.name))

    def _owner(lineno):
        c = [f for f in funcs if f[0] <= lineno <= f[1]]
        return min(c, key=lambda f: f[1] - f[0])[2] if c else "<module>"

    seen = {a: [] for a in V._E3_LATCHED_ATTRS}
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Assign):
            for t in n.targets:
                if isinstance(t, _ast.Attribute) and t.attr in seen:
                    seen[t.attr].append(_owner(t.lineno))
    for attr, owners in seen.items():
        assert owners, f"{attr} is never assigned in e3_selector.py -- stale lint entry?"
        assert set(owners) == {"select"}, (
            f"{attr} is assigned outside select() ({sorted(set(owners))}). It no longer "
            f"latches unconditionally, so the lint's premise does not hold for it.")


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
# Re-pinned 65 -> 64 the same day; see RECONCILIATION below. This is a BACKLOG SIZE, not
# a target -- the landed scripts have all run and are deliberately NOT retro-edited
# (a completed run's pre-registered emission is not rewritten). The pin exists so a
# NEW script carrying the defect shows up as a rise, and so a later widening of the
# rule announces its own blast radius instead of drifting silently.
#
# ---- RECONCILIATION: why 65, not the ~52 the commissioning brief predicted ----------
# Both numbers were right about different questions, and NEITHER was a mis-measurement.
# The ~52 came from the 2026-07-19 corpus audit (session `upbeat-gauss-918164`, recorded
# in WORKSPACE_STATE.md 15:30Z): a grep returning "61 grep hits triaged", of which "~9
# cleanly UNAFFECTED" were set aside -- 61 - 9 = 52. That audit's unit of measurement
# differs from this lint's in THREE compounding ways, each verified by re-running the
# landed lint against the corpus as it stood at the audit-era commit (fa52889, 1086
# scripts):
#   (1) ATTRIBUTE SET -- dominant. The audit grepped the ONE confirmed attribute,
#       `last_score_diagnostics`. The lint covers every attribute that latches. A
#       diagnostics-only variant of this lint fires on 46 of 1086 at that commit, where
#       the full rule fires on 66 -- so the attribute set alone accounts for ~20 files.
#       On today's corpus: 45 of the fires read last_score_diagnostics, 20 fire on a
#       DIFFERENT latching attribute only. Those 20 are true positives, not slack: all
#       six attributes are assigned exclusively inside select() (asserted against the
#       substrate by test_e3s_all_six_latched_attrs_are_select_only_in_the_substrate),
#       so a driver latching last_scores pseudo-replicates exactly as one latching
#       last_score_diagnostics does.
#   (2) GLOB. The lint's corpus is `v3_exq_*.py`; the brief said `experiments/*.py`. At
#       the audit commit the single-attribute grep hit 60 files under the lint's glob but
#       63 under the broader one. The audit's own `v4_exq_001`/`v4_exq_003` exemptions
#       are outside this lint's corpus entirely -- correctly, they drive a bare selector.
#   (3) COUNTING UNIT. This lint reports one finding per FILE; the audit triaged grep
#       HITS, and a grep hit includes docstring and comment mentions that are not reads
#       at all. That is why ~52 sits ABOVE the 46 an attribute-matched AST scan yields.
# The audit's ~9 hand-triaged exemptions (the 485h-m + 696 family, 689g/689h) need no
# special handling -- they call `e3.select()` directly and are already discharged by
# exemption (c). The audit and the lint agree about them.
# CONCLUSION: 65 was the better measure of the defect and is kept as the basis. The
# brief's ~52 was a single-attribute, hand-triaged lower bound, not a competing count.
#
# ---- What actually changed at the 65 -> 64 re-pin ------------------------------------
# Reconciling the two definitions surfaced one genuine error in each direction. Both were
# fixed on their merits; neither was chosen to move the number toward ~52 (and note they
# nearly cancel, so fitting to ~52 was never available anyway):
#   NARROWED, -1: exemption (d), the identity-freshness guard. v3_exq_777 dedupes by
#     `id(last_precommit_probs)` and records only on change -- semantically equivalent to
#     clear-before-select, and the ONLY false positive in the 65. Checked precisely: 23
#     other fires call id() for unrelated reasons and all still fire, 708 and 709 among
#     them (both CONFIRMED carriers per the 708 re-adjudication).
#   WIDENED, +0: `last_raw_scores` added to _E3_LATCHED_ATTRS -- a coverage hole, since
#     V3-EXQ-722's SECOND latched read was exactly that attribute. Measured net effect on
#     the corpus: ZERO. Future coverage at no backlog cost.
_PINNED_CORPUS_FIRE_COUNT = 64


def test_e3s_corpus_fire_rate_is_pinned():
    fired = [p for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
             if V.e3_diagnostics_staleness_lint(p) is not None]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"stale-E3-diagnostics fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(clear-before-select) rather than re-pinning. If you deliberately widened or "
        f"narrowed the rule, re-pin and say so in the commit message. "
        f"Fired: {[p.name for p in fired]}")
