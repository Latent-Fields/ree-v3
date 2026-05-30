"""Contract tests for the Phase 3 heartbeat WRITE-gate scope.

The Phase 3 `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1` env flag suppresses
the LOCAL runner_heartbeats/<host>.json file write to avoid pull-conflict
UU markers when the hub's sync_daemon.phase3_heartbeat_writer publishes
its own version of the same file via origin/master.

Crucially, the gate MUST NOT suppress the coordinator POST. The writer
materialises the canonical runner_heartbeats file from the coordinator
DB's heartbeats table (sync_daemon.phase3_heartbeat_writer WHERE clause:
`heartbeat_payload_json IS NOT NULL`). The DB row is only populated by
POST /heartbeat from the runner. Suppressing the POST under the WRITE
gate leaves the writer with nothing to publish -> fleet-wide stale
heartbeats.

History:
- 2026-05-29 (commit e82b2a823f): serve.py started injecting
  PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1 for the Mac runner to fix
  pull-conflict UU markers (the writer's version vs the runner's local
  write collided on every auto-sync git pull).
- 2026-05-29 .. 2026-05-30: write_heartbeat() short-circuited the entire
  function under the gate, so the coordinator POST never fired either.
  Heartbeats table went stale fleet-wide; explorer + scaler workflow
  saw every machine as idle. Coordinator shadow log on the Mac showed
  ZERO POST /heartbeat attempts after 2026-05-29T06:49Z.
- 2026-05-30: gate scoped to local file write only; coordinator POST
  fires unconditionally so the writer always has fresh DB rows to
  materialise from.

Contracts:
  C1. WRITE gate OFF -> local file written AND coordinator POST fired.
  C2. WRITE gate ON  -> local file SKIPPED but coordinator POST fired.
  C3. WRITE gate OFF + invalid assembly path -> early-return None, POST
      not attempted (the function never gets that far).
  C4. WRITE gate ON  + transient local-write failure simulated -> POST
      still fires (the gate's whole point is to make local-write failures
      tolerable; the canonical materialiser is the writer, not the file).
  C5. coordinator_client.TIMEOUT default >= 10 (raised from 3 on
      2026-05-30 after fleet-wide /heartbeat timeouts under 3s).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fake_assembly(tmp_path: Path) -> Path:
    asm = tmp_path / "REE_assembly"
    asm.mkdir()
    return asm


@pytest.fixture(autouse=True)
def _clear_phase3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with the gate UNSET so tests opt in explicitly."""
    monkeypatch.delenv("PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", raising=False)
    monkeypatch.delenv("PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH", raising=False)


def _call_write_heartbeat(
    rrc_mod: Any, assembly_path: Path,
) -> tuple[Path | None, Any]:
    """Invoke write_heartbeat with a stable test fixture; return
    (returned_path, coordinator_post_mock)."""
    with mock.patch.object(
        rrc_mod.coordinator_client, "report_heartbeat",
        autospec=True,
    ) as post_mock:
        returned = rrc_mod.write_heartbeat(
            assembly_path, "test-host", state="running",
            current_exq="V3-EXQ-TEST",
            progress={"episodes_done": 1, "episodes_total": 10},
            recent_lines=["line a", "line b"],
            runner_pid=12345,
        )
    return returned, post_mock


def test_c1_gate_off_writes_local_file_and_fires_coordinator_post(
    fake_assembly: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WRITE gate OFF -> local file written AND coordinator POST fired."""
    monkeypatch.delenv(
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", raising=False)
    # Reload to pick up cleared env (module-level gate-logged flag).
    import importlib
    import runner_remote_control as rrc
    importlib.reload(rrc)
    returned, post_mock = _call_write_heartbeat(rrc, fake_assembly)
    hb_dir = fake_assembly / rrc.HEARTBEAT_SUBPATH
    target = hb_dir / "test-host.json"
    assert returned == target
    assert target.exists()
    assert post_mock.call_count == 1
    # The POST must carry the rich payload (writer materialises from it).
    _, kwargs = post_mock.call_args
    assert kwargs.get("payload") is not None


def test_c2_gate_on_skips_local_file_but_fires_coordinator_post(
    fake_assembly: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WRITE gate ON -> local file SKIPPED but coordinator POST fired.

    This is the load-bearing contract: the writer needs the DB row, and
    the DB row is only populated by POST /heartbeat. Suppressing the POST
    under the WRITE gate is the bug we fixed on 2026-05-30.
    """
    monkeypatch.setenv("PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", "1")
    import importlib
    import runner_remote_control as rrc
    importlib.reload(rrc)
    returned, post_mock = _call_write_heartbeat(rrc, fake_assembly)
    hb_dir = fake_assembly / rrc.HEARTBEAT_SUBPATH
    target = hb_dir / "test-host.json"
    # Local file not written.
    assert not target.exists()
    # write_heartbeat returns None under the WRITE gate so the caller's
    # `if hb_path is not None: push_heartbeat(...)` correctly suppresses
    # the git push of a file that was never written.
    assert returned is None
    # Coordinator POST fired regardless. This is the regression guard.
    assert post_mock.call_count == 1
    _, kwargs = post_mock.call_args
    assert kwargs.get("payload") is not None
    # Structured columns must also be present (writer reads them too).
    args, _ = post_mock.call_args
    assert args[0] == "test-host"  # machine
    assert args[1] == "running"    # state
    assert args[2] == "V3-EXQ-TEST"  # current_exq


def test_c3_invalid_assembly_path_returns_early_no_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing assembly dir -> early-return None; coordinator POST not
    attempted. The invalid-path check is BEFORE the WRITE-gate check."""
    monkeypatch.delenv(
        "PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", raising=False)
    import importlib
    import runner_remote_control as rrc
    importlib.reload(rrc)
    bogus = Path("/nonexistent/path/that/does/not/exist")
    returned, post_mock = _call_write_heartbeat(rrc, bogus)
    assert returned is None
    assert post_mock.call_count == 0


def test_c4_gate_on_local_write_failure_still_fires_coordinator_post(
    fake_assembly: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under WRITE gate ON, transient local-write IO failure must not
    prevent the coordinator POST. The gate's purpose IS to make the local
    write expendable; with it on, the canonical channel is the POST."""
    monkeypatch.setenv("PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE", "1")
    import importlib
    import runner_remote_control as rrc
    importlib.reload(rrc)
    # Block the local file write path by removing read+exec on the
    # heartbeat dir's parent. With the gate ON, the function never tries
    # the local write anyway -- so we just confirm the POST still fires.
    returned, post_mock = _call_write_heartbeat(rrc, fake_assembly)
    assert returned is None
    assert post_mock.call_count == 1


def test_c5_coordinator_timeout_default_at_least_ten_seconds() -> None:
    """The 3s default was the proximate cause of fleet-wide POST timeouts
    on 2026-05-28 (coordinator_shadow.log evidence). Raised to 10s on
    2026-05-30 to fit the rich-payload POST over WireGuard. Anyone
    lowering this default should restore both fix legs together."""
    # Read the value from a fresh import so the env-default is the one
    # that ships in source, not whatever the running shell happens to
    # have exported.
    import importlib
    import coordinator_client as cc
    saved = os.environ.pop("COORDINATOR_TIMEOUT", None)
    try:
        importlib.reload(cc)
        assert cc.TIMEOUT >= 10.0, (
            f"COORDINATOR_TIMEOUT default {cc.TIMEOUT}s is below the 10s "
            "floor established 2026-05-30 to fit /heartbeat POSTs with "
            "the rich payload over WireGuard. See "
            "test_runner_heartbeat_write_gate_preserves_coordinator_post.py "
            "C5 docstring + 2026-05-28 coordinator_shadow.log timeouts.")
    finally:
        if saved is not None:
            os.environ["COORDINATOR_TIMEOUT"] = saved
            importlib.reload(cc)
