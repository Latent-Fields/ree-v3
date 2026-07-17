"""Shared pytest config for the regression suite.

Adds ree-v3 repo root to sys.path so `from ree_core import ...` works
regardless of pytest invocation cwd.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Environment variables git uses to LOCATE the repository / index / work tree.
# A parent `git` process -- most importantly the pre-commit contract hook, which
# runs this suite as a child of `git commit` -- exports these into its children.
# Several contract tests create throwaway git repos via `subprocess.run(["git",
# ...], cwd=<tmp>)` WITHOUT passing an explicit env, so an inherited GIT_DIR /
# GIT_INDEX_FILE / GIT_WORK_TREE / GIT_PREFIX silently redirects those git calls
# at the REAL repository instead of the tmp repo. That produces spurious,
# hook-only failures (e.g. `fatal: unable to read <sha>`, an --ff-only pull that
# should refuse but fast-forwards, a tracked script reported untracked) while the
# same tests pass when pytest is run standalone (clean env). Strip these for the
# whole session so the suite is hermetic whether run standalone or inside a git
# hook. Identity vars (GIT_AUTHOR_*/GIT_COMMITTER_*) are intentionally left
# alone -- they do not affect repo resolution and a tmp-repo commit can safely
# inherit them.
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_PREFIX",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_INDEX_VERSION",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
)


@pytest.fixture(autouse=True, scope="session")
def _isolate_git_repo_env():
    """Strip inherited git repo-location env vars for the whole test session.

    Makes the git-subprocess contract tests hermetic against being run inside a
    git hook (the pre-commit contract gate), where `git commit` exports GIT_DIR
    / GIT_INDEX_FILE / GIT_PREFIX into the pytest process. See the note above
    _GIT_LOCATION_ENV_VARS for the full failure mode.
    """
    saved = {k: os.environ.pop(k) for k in _GIT_LOCATION_ENV_VARS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)
