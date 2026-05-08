"""Contract tests for runner -> subprocess env isolation.

The runner spawns each experiment as a subprocess and passes REE_QUEUE_ID
and REE_RUNNER_SIGNAL_DIR via the env. emit_outcome() in the child reads
these env vars to decide where to write the run-conformance sentinel.

If the runner's OWN shell env has either var set (left over from a prior
debug run, a wrapper, manual `export`, etc.) and the runner's env-build
forgets to overwrite, the child silently inherits the stale value and
emit_outcome writes the sentinel under <stale_dir>/<stale_id>.json --
wrong file, wrong directory, masking real sentinels with phantom ones.

Contracts:
  C1. queue_id truthy -> env["REE_QUEUE_ID"] == queue_id, regardless of
      what the parent shell had.
  C2. queue_id falsy ("" / None-like) -> env["REE_QUEUE_ID"] == "",
      explicitly overriding any inherited value (no leak).
  C3. signal_dir is a Path -> env["REE_RUNNER_SIGNAL_DIR"] == str(path),
      regardless of parent shell.
  C4. signal_dir is None -> env["REE_RUNNER_SIGNAL_DIR"] == "",
      explicitly overriding any inherited value (no leak).
  C5. Other env vars (PATH, etc.) ARE inherited -- only the two runner
      vars are forced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_runner import _build_subprocess_env  # noqa: E402


@pytest.fixture
def stale_parent_env(monkeypatch):
    """Simulate a runner process whose own shell env has stale values."""
    monkeypatch.setenv("REE_QUEUE_ID", "STALE-EXQ-999")
    monkeypatch.setenv("REE_RUNNER_SIGNAL_DIR", "/tmp/stale_signal_dir_xyz")


def test_c1_queue_id_truthy_overrides_parent(stale_parent_env):
    env = _build_subprocess_env("V3-EXQ-042", Path("/tmp/sig"))
    assert env["REE_QUEUE_ID"] == "V3-EXQ-042"


def test_c2_queue_id_empty_clears_parent(stale_parent_env):
    env = _build_subprocess_env("", Path("/tmp/sig"))
    assert env["REE_QUEUE_ID"] == "", (
        "empty queue_id must explicitly clear the env var, not inherit "
        f"the parent's stale value (got {env['REE_QUEUE_ID']!r})"
    )


def test_c3_signal_dir_path_overrides_parent(stale_parent_env, tmp_path):
    env = _build_subprocess_env("V3-EXQ-042", tmp_path)
    assert env["REE_RUNNER_SIGNAL_DIR"] == str(tmp_path)


def test_c4_signal_dir_none_clears_parent(stale_parent_env):
    env = _build_subprocess_env("V3-EXQ-042", None)
    assert env["REE_RUNNER_SIGNAL_DIR"] == "", (
        "signal_dir=None must explicitly clear the env var, not inherit "
        f"the parent's stale value (got {env['REE_RUNNER_SIGNAL_DIR']!r})"
    )


def test_c5_other_env_vars_inherited(monkeypatch):
    monkeypatch.setenv("REE_TEST_SENTINEL_VAR", "preserve-me")
    env = _build_subprocess_env("V3-EXQ-042", Path("/tmp/sig"))
    assert env.get("REE_TEST_SENTINEL_VAR") == "preserve-me"
    # PATH should also pass through (sanity check that we're copying os.environ)
    assert "PATH" in env


def test_c2_c4_falsy_does_not_unset_keys(stale_parent_env):
    """Falsy queue_id / signal_dir must SET the key to '', not delete it.

    Setting to '' is what guarantees the child sees an explicit override.
    Deleting (env.pop) would still leak via os.environ inheritance from
    the parent in some subprocess wrappers.
    """
    env = _build_subprocess_env("", None)
    assert "REE_QUEUE_ID" in env
    assert "REE_RUNNER_SIGNAL_DIR" in env
    assert env["REE_QUEUE_ID"] == ""
    assert env["REE_RUNNER_SIGNAL_DIR"] == ""
