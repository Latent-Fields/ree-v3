"""Contract: the CI gate's collection roots must cover EVERY test in the tree.

WHAT THIS GUARDS. `ree-v3` declares no pytest `testpaths` -- there is no
pytest.ini, pyproject.toml, setup.cfg or tox.ini anywhere in the tree -- so
pytest collects purely by the paths it is given. A test file that no root
reaches is not failed, not skipped and not reported: it is silently never run.
That defect class has already bitten this repo twice, both on 2026-07-27:

  (a) a default of `tests/` alone ran 2499 tests and none of coordinator's 264,
      the Phase-3 writer-race guards. Two regressions sat on trunk undetected.
  (b) fixing (a) by naming the second root left dispatch/test_dispatch.py,
      test_runner_reclassify_cmd.py, test_validate_recording.py and
      experiments/test_infant_curriculum_phase3.py still uncollected.

and a third time in .github/workflows/contract-tests.yml itself, which ran
`tests/` alone until 2026-09-18 -- 1026 tests, 919 of them coordinator/, with no
post-merge check at all.

WHY THIS ENUMERATES THE TREE RATHER THAN PINNING A LIST. (b) is the whole
argument. A check that asserts "the roots are these six names" is green the
moment it is written and stays green while a seventh root goes unrun -- (a)'s
pin was already green while (b) was live. So this asserts the CONVERSE: every
pytest-collectable file in the tree must be reachable from some declared root,
and a new stray file fails this test naming the exact path to add. It is the
same contract `scripts/remote_pytest.sh --selftest` (`_selftest_default_args`)
holds over its own DEFAULT_PYTEST_ARGS, applied to the CI gate's copy -- the two
lists live in different repos and nothing else keeps them honest.

This matters MORE now that the gate runs under pytest-xdist than it did before.
xdist removes the exhaustiveness hazard for DISTRIBUTION (there is no partition
to get wrong; pytest collects and xdist distributes), which is exactly why xdist
was chosen over a runner matrix. It does nothing for COLLECTION: xdist faithfully
distributes whatever it was handed, so an unnamed root is still silently unrun.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "contract-tests.yml"

# Mirrors _selftest_default_args' sweep exclusions, and for the same reasons:
#   .git, nested worktrees under .claude (full repo copies -- collecting those
#   would double-run everything), virtualenvs/node_modules, and the
#   experiments/v3_exq_* DRIVERS, which end `..._test.py` by convention and are
#   experiment entry points rather than tests (see the workflow's own note on
#   why `experiments/` is not passed as a bare directory).
_SKIP_PREFIXES = (".git/", ".claude/", ".venv/", "venv/", "node_modules/")
_SKIP_CONTAINS = ("/.git/", "/.claude/", "/.venv/", "/node_modules/", "/site-packages/")


def _declared_roots() -> list[str]:
    """The roots the CI job actually passes to pytest, read from the workflow.

    Parsed as TEXT, not YAML: this test must run in the CI job's own
    environment, which installs numpy/pytest/torch and makes no promise about
    PyYAML being importable. The workflow keeps PYTEST_ROOTS on one line for
    exactly this reason.
    """
    assert WORKFLOW.is_file(), f"CI workflow not found at {WORKFLOW}"
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r'^\s*PYTEST_ROOTS:\s*"([^"]+)"\s*$', text, re.MULTILINE)
    assert m, (
        "no single-line `PYTEST_ROOTS: \"...\"` mapping found in "
        f"{WORKFLOW.relative_to(REPO_ROOT)}.\n"
        "This test reads that line as the gate's source of truth. If the run "
        "step was reworked, keep the roots on one double-quoted line (or update "
        "this parser deliberately) -- do not let the gate's collected set stop "
        "being checkable."
    )
    roots = m.group(1).split()
    assert roots, "PYTEST_ROOTS is empty -- the gate would collect nothing"
    return roots


def _collectable_files() -> list[str]:
    """Every file pytest would collect, as repo-relative posix paths."""
    out = []
    for pat in ("test_*.py", "*_test.py"):
        for abs_path in REPO_ROOT.rglob(pat):
            rel = abs_path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith(_SKIP_PREFIXES) or any(c in rel for c in _SKIP_CONTAINS):
                continue
            if rel.startswith("experiments/v3_exq_"):
                continue  # experiment drivers, not tests
            out.append(rel)
    return sorted(set(out))


def _covered(rel: str, roots: list[str]) -> bool:
    for want in roots:
        if want.endswith("/"):
            if rel.startswith(want):
                return True
        elif rel == want:
            return True
    return False


def test_every_declared_root_still_exists():
    """A rename or consolidation must speak up, not leave a root collecting nothing."""
    missing = [r for r in _declared_roots() if not (REPO_ROOT / r).exists()]
    assert not missing, (
        "PYTEST_ROOTS names path(s) that no longer exist, so the CI gate "
        f"silently collects nothing from them: {missing}"
    )


def test_ci_roots_reach_every_collectable_test_file():
    """The converse, and the part that catches the NEXT stray file."""
    roots = _declared_roots()
    files = _collectable_files()
    assert files, "enumeration found no test files at all -- the sweep is broken"

    uncovered = [rel for rel in files if not _covered(rel, roots)]
    assert not uncovered, (
        f"{len(uncovered)} pytest-collectable file(s) are NOT reached by any "
        "root in the CI gate's PYTEST_ROOTS, so they are silently never run "
        "post-merge:\n  "
        + "\n  ".join(uncovered)
        + "\n\nFIX: add the path to PYTEST_ROOTS in "
        ".github/workflows/contract-tests.yml (and to DEFAULT_PYTEST_ARGS in "
        "REE_Working/scripts/remote_pytest.sh, which holds the same contract "
        "for local/worker runs). Do NOT add it here."
    )


@pytest.mark.parametrize(
    "root", ["tests/", "coordinator/", "dispatch/"]
)
def test_the_three_directory_roots_are_named(root):
    """Regression pin for the two 2026-07-27 misses and the 2026-09-18 CI one.

    Deliberately NOT the whole list: pinning every root is the degenerate check
    argued against in this module's docstring. These three are pinned because
    each was a MEASURED silent-skip, and coordinator/ in particular is the
    phase3 writer-race corpus whose absence went unnoticed for months.
    """
    assert root in _declared_roots(), (
        f"{root} was dropped from the CI gate's PYTEST_ROOTS -- this exact "
        "omission is a recorded incident, not a hypothetical"
    )
