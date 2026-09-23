"""Contract for `validate_experiments.py`'s --paths state discrimination.

THE DEFECT (negative-instrument audit, REE_assembly/evidence/planning/
negative_instrument_audit_20260922.md, finding 5, commit 79e796789c). Before
the fix, `validate_experiments.py` collapsed three genuinely different states
into the same "exit 0, looks clean" outcome:

  1. validated-clean   -- real scripts named, none non-conforming.
  2. validated-nothing -- --paths given but the shell expansion produced zero
     tokens (nargs='*' cannot tell "omitted" from "given empty" without a
     sentinel default). This silently fell back to `_candidate_paths([])`'s
     full-glob branch -- a run over all 1504 drivers with all four hard gates
     (arm-fingerprint, degeneracy self-report, manifest-writer, use-before-def)
     downgraded from hard-under-`--paths` to advisory-in-full-glob. A caller
     asking "did MY new script conform" got "1504 scripts advisory-passed"
     instead.
  3. could-not-validate -- a --paths entry that does not exist or cannot be
     read. `Path(p).resolve()` performed no existence check, and 7 of 8
     per-path lints catch the resulting OSError on `read_text()` and return
     None (= clean), so an unreadable path validated as silently green;
     only `check_script` (the base conformance gate) failed closed.

Reproduction, verbatim from the audit: `--strict --quiet --checks
manifest_writer --paths <bogus>` used to exit 0 with
"checked 1 scripts: 0 OK, 0 exempt, 0 non-conforming".

THE FIX. `--paths` now defaults to `None` (not `[]`), which is what makes
"omitted" and "given but empty" distinguishable at all under argparse's
`nargs='*'`. An explicitly-empty expansion, and any --paths entry that does
not resolve to a readable file, both now exit 3 with a printed reason --
never 0, and never silently promoted into a full-glob advisory run.

THE TEST HALF. These tests exercise the REAL CLI subprocess and the REAL
filesystem/shell-expansion boundary -- not a stubbed `_candidate_paths` --
because the defect lives in what the tool does with its actual input, per
the chip brief: "A test that stubs the path expansion and asserts the
comparison logic does NOT satisfy this." Each of the three states must
produce a DIFFERENT (exit code, message) pair; see
`test_the_three_states_are_pairwise_distinguishable` for the direct
assertion, and each state's own test for the old-defect blind-spot
measurement recorded in its docstring (checked by hand against
`git show HEAD:validate_experiments.py`, the pre-fix content, during
authoring -- all three returned exit 0 there; see the chip's closing note
for the full transcript).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# A real, currently-clean driver -- used as the validated-clean control.
# If this script stops being clean, swap it for another that currently is;
# the point is a REAL file, not a synthetic fixture, since check_script and
# the manifest_writer lint both read real source.
CLEAN_DRIVER = "v3_exq_1075_sdppb5_action_sensitivity_validation.py"


def _run(args):
    return subprocess.run(
        [sys.executable, "validate_experiments.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_validated_clean_a_real_existing_script_exits_0():
    """State 1: a real, conforming --paths entry validates clean."""
    path = EXPERIMENTS_DIR / CLEAN_DRIVER
    assert path.exists(), f"{CLEAN_DRIVER} missing -- pick another currently-clean driver"
    out = _run(["--strict", "--quiet", "--checks", "manifest_writer",
                "--paths", f"experiments/{CLEAN_DRIVER}"])
    assert out.returncode == 0, f"expected clean-validate exit 0, got {out.returncode}\n{out.stdout}"
    assert "checked 1 scripts" in out.stdout


def test_paths_given_but_shell_expansion_is_empty_is_not_silently_clean():
    """State 2: --paths present with zero tokens must NOT read as validated-clean
    or silently fall back to a full-glob advisory sweep.

    OLD DEFECT (measured against the pre-fix content): `args.paths` defaulted
    to `[]`, identical to the omitted case, so `_candidate_paths([])` took the
    full-glob branch (`sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))`, ~1500
    scripts) and every hard gate's `bool(args.paths)` guard evaluated False --
    exit 0, "checked 1500+ scripts", with the four hard gates silently
    downgraded to advisory. A caller who ran `--paths $(some_glob_that_matched_nothing)`
    believing they were checking nothing (or one script) got a full advisory
    sweep instead, indistinguishable from a deliberate full-glob invocation.
    """
    out = _run(["--strict", "--quiet", "--checks", "manifest_writer", "--paths"])
    assert out.returncode == 3, (
        f"empty --paths expansion must be a distinct could-not-validate exit, "
        f"not the full-glob fallback; got {out.returncode}\n{out.stdout}")
    assert "checked 1" not in out.stdout and "checked 15" not in out.stdout, (
        "must not have silently run the full-glob sweep")
    assert "expanded to zero paths" in out.stdout


def test_unreadable_paths_entry_is_not_silently_clean():
    """State 3: a nonexistent/unreadable --paths entry must not validate clean.

    OLD DEFECT (measured against the pre-fix content, the audit's own
    reproduction): `Path(p).resolve()` performed no existence check, and 7 of
    8 per-path lints catch OSError-on-read and return None (clean). Exact old
    repro: `--strict --quiet --checks manifest_writer --paths <bogus>` exited
    0 with "checked 1 scripts: 0 OK, 0 exempt, 0 non-conforming" -- a fully
    green report about a file that was never actually read.
    """
    bogus = "/tmp/does_not_exist_validate_experiments_contract_probe_zzz.py"
    assert not Path(bogus).exists()
    out = _run(["--strict", "--quiet", "--checks", "manifest_writer", "--paths", bogus])
    assert out.returncode == 3, (
        f"an unreadable --paths entry must be could-not-validate, not clean; "
        f"got {out.returncode}\n{out.stdout}")
    assert "0 OK, 0 exempt, 0 non-conforming" not in out.stdout, (
        "must not print the old silently-clean summary for an unread path")
    assert "does not exist" in out.stdout


def test_the_three_states_are_pairwise_distinguishable():
    """Direct assertion of the audit's own framing: nothing may collapse.

    Runs all three invocations and asserts their (returncode, stdout) pairs
    are pairwise distinct -- the property the old code violated (clean and
    could-not-validate were both exit 0 with a "0 non-conforming" summary).
    """
    clean = _run(["--strict", "--quiet", "--checks", "manifest_writer",
                  "--paths", f"experiments/{CLEAN_DRIVER}"])
    empty = _run(["--strict", "--quiet", "--checks", "manifest_writer", "--paths"])
    missing = _run(["--strict", "--quiet", "--checks", "manifest_writer",
                    "--paths", "/tmp/does_not_exist_validate_experiments_contract_probe_zzz.py"])

    codes = {clean.returncode, empty.returncode, missing.returncode}
    assert clean.returncode == 0
    assert empty.returncode != 0 and missing.returncode != 0
    assert len(codes) >= 2, "clean must be distinguishable from both failure states by exit code"
    # empty-expansion and unreadable-path are both could-not-validate (exit 3
    # by design -- neither is a conformance verdict), but their printed
    # reasons must still differ so a human reading the log is not left
    # guessing which failure mode occurred.
    assert "expanded to zero paths" in empty.stdout
    assert "does not exist" in missing.stdout
    assert empty.stdout != missing.stdout


def test_omitted_paths_still_runs_the_full_glob_sweep_advisory():
    """Sanity: omitting --paths entirely is still the deliberate full-glob
    entry point and must remain unaffected -- only the explicit-but-empty
    case (previous test) is new behaviour. Restricted to a single fast check
    so this stays a contract test, not a ~1500-script sweep.
    """
    out = _run(["--quiet", "--checks", "manifest_writer"])
    assert out.returncode == 0
    assert "checked 1 scripts" not in out.stdout  # full-glob, not a single file
    assert "no scripts found" not in out.stdout


def test_precommit_hook_shape_a_real_staged_experiment_still_validates():
    """The hook-path hard constraint: precommit_contracts.sh runs
    `validate_experiments.py --strict --quiet --paths $STAGED_EXPERIMENTS`
    with an unquoted, real, non-empty variable. This must behave exactly as
    before the fix for a normal commit -- confirmed here with a real script
    exactly as the hook would invoke it (word-split, not list-quoted).
    """
    out = _run(["--strict", "--quiet", "--checks", "manifest_writer",
                "--paths", f"experiments/{CLEAN_DRIVER}"])
    assert out.returncode == 0, (
        f"a normal staged experiment must still pass the hook path; "
        f"got {out.returncode}\n{out.stdout}")
