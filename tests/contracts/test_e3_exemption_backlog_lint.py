"""Contracts for the e3-exemption-backlog counter.

Surfaces under test:
  (1) validate_experiments.e3_exemption_backlog_lint -- flags a script whose source still
      contains either E3 blanket-opt-out marker (E3_DIAGNOSTICS_STALENESS_EXEMPT /
      E3_HOLD_WEIGHTED_READOUT_EXEMPT), regardless of whether the lint the marker opts out
      of would still fire without it.
  (2) validate_experiments.py --checks e3_exemption_backlog -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths).

WHY THIS GATE EXISTS. Both `e3_diagnostics_staleness_lint` and
`e3_hold_weighted_readout_lint` return None UNCONDITIONALLY once their marker is present
in source -- before checking whether the read pattern the marker opts out of would even
have fired. So a script that no longer needs the marker (already migrated onto a
discharge path, or one that never had the driver-loop shape either rule targets) reports
"OK, 0 exempt" identically to a script that genuinely still needs it. The exemption goes
INERT-BUT-PRESENT and neither lint's own output says so -- see
REE_assembly/evidence/planning/e3_fresh_select_migration_plan.md sec 4, which recommended
this counter as the structural fix rather than relying on a planning doc to be
remembered. Mirrors the three sibling backlog counters already in this file:
arm-fingerprint-backlog, degeneracy-self-report-backlog, manifest-writer-backlog.

SCOPE. This is a plain substring match over raw source -- deliberately, since that is how
both E3 lints themselves detect their marker (`_E3_STALENESS_EXEMPT_MARKER in src`), and
per the migration plan's own caution, naming a marker string anywhere (even a comment)
silently re-arms the blanket exemption for both lints. So this counter has to be
sensitive to the identical thing to stay honest with them. WARN-only in BOTH modes: this
is a visibility counter over existing markers, not a new gate on new code.
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
        return V.e3_exemption_backlog_lint(Path(name))
    finally:
        os.unlink(name)


_STALENESS_MARKER_ONLY = '''
E3_DIAGNOSTICS_STALENESS_EXEMPT = "reads once per episode"

def main():
    pass
'''

_HOLD_WEIGHTED_MARKER_ONLY = '''
E3_HOLD_WEIGHTED_READOUT_EXEMPT = "threshold-invariant floor"

def main():
    pass
'''

_BOTH_MARKERS = '''
E3_DIAGNOSTICS_STALENESS_EXEMPT = "reads once per episode"
E3_HOLD_WEIGHTED_READOUT_EXEMPT = "threshold-invariant floor"

def main():
    pass
'''

_NO_MARKER = '''
def main():
    for step in range(100):
        agent.select_action(obs)
        diag = agent.e3.last_score_diagnostics
        rows.append(diag)

if __name__ == "__main__":
    main()
'''


def test_e3eb_flags_staleness_marker():
    out = _lint_src(_STALENESS_MARKER_ONLY)
    assert out is not None and "E3 EXEMPTION BACKLOG" in out, out
    assert "E3_DIAGNOSTICS_STALENESS_EXEMPT" in out, out


def test_e3eb_flags_hold_weighted_marker():
    out = _lint_src(_HOLD_WEIGHTED_MARKER_ONLY)
    assert out is not None and "E3 EXEMPTION BACKLOG" in out, out
    assert "E3_HOLD_WEIGHTED_READOUT_EXEMPT" in out, out


def test_e3eb_flags_both_markers_together():
    out = _lint_src(_BOTH_MARKERS)
    assert out is not None
    assert "E3_DIAGNOSTICS_STALENESS_EXEMPT" in out, out
    assert "E3_HOLD_WEIGHTED_READOUT_EXEMPT" in out, out


def test_e3eb_fires_even_when_underlying_lint_would_not():
    """The whole point: a marker with nothing to discharge is still counted.

    `_NO_MARKER`-shaped source WOULD fire e3_diagnostics_staleness_lint (per that
    lint's own contract, test_e3s_bare_loop_read_is_flagged). Add the marker and the
    underlying lint goes silent (test_e3s_explicit_opt_out_is_honoured) -- but this
    counter must still flag it, precisely because the marker is doing real work here
    and has NOT yet been removed as part of a migration.
    """
    src = 'E3_DIAGNOSTICS_STALENESS_EXEMPT = "x"\n' + _NO_MARKER
    out = _lint_src(src)
    assert out is not None and "E3_DIAGNOSTICS_STALENESS_EXEMPT" in out, out


def test_e3eb_clean_script_is_exempt():
    """No marker anywhere in source -> nothing to count."""
    assert _lint_src(_NO_MARKER) is None


def test_e3eb_migrated_script_is_exempt():
    """A script that uses the shared helper and carries no marker text is clean."""
    src = '''
from experiments._lib.fresh_select import FreshSelectProbe

def main():
    fs = FreshSelectProbe(agent, "ns")
    for step in range(100):
        agent.select_action(obs)
        fs.flush()
'''
    assert _lint_src(src) is None


def test_e3eb_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other branch of this gate family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_STALENESS_MARKER_ONLY)
        name = f.name
    try:
        r = _run("--checks", "e3_exemption_backlog", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "E3 EXEMPTION BACKLOG" in r.stdout
    finally:
        os.unlink(name)


# ---- real-corpus witnesses ------------------------------------------------------------

def test_e3eb_699b_and_689i_are_clean():
    """Both call sites named in e3_fresh_select_migration_plan.md were migrated onto the
    shared helper 2026-08-10 -- both exemption markers removed. If either reappears here,
    the migration regressed."""
    for name in ("v3_exq_699b_pcomp_demotion_x_gonogo_fresh_select.py",
                 "v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair.py"):
        real = EXPERIMENTS_DIR / name
        if not real.exists():
            continue
        assert V.e3_exemption_backlog_lint(real) is None, name


# ---- corpus fire-rate pin --------------------------------------------------------------
# Pinned 2026-08-10 against the v3_exq_*.py corpus at the counter's own introduction.
# This is a BACKLOG SIZE, not a target of zero -- the ten scripts below carry a marker
# for reasons unrelated to the fresh_select migration (each still has a genuine driver
# shape the corresponding E3 lint would otherwise flag). A NEW script that adds either
# marker without discharging the pattern properly will move this count; fix the script
# (or migrate it) rather than re-pinning, unless the change is a deliberate migration --
# in which case re-pin down and say so in the commit message.
_PINNED_CORPUS_FIRE_COUNT = 10


def test_e3eb_corpus_fire_rate_is_pinned(corpus_scan):
    fired = corpus_scan["e3_exemption_backlog_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"e3-exemption-backlog fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list because it added an "
        f"exemption marker, migrate it onto the shared helper (or another discharge path) "
        f"rather than re-pinning. If you deliberately migrated an EXISTING carrier off its "
        f"marker, re-pin down and say so in the commit message. "
        f"Fired: {[p.name for p in fired]}")
