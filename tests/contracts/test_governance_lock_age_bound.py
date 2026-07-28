"""Contract tests for the shape-aware governance-lock age bound.

`REE_assembly/scripts/governance.sh` opens a `governance-sh-<host>` TASK_CLAIMS
claim over evidence/ for the duration of a governance regen (REE_assembly
49d5c87922), so the runner heartbeat's per-minute `git pull --rebase
--autostash` cannot sweep a half-written ~1050-1190-file artifact set. The lock
is released by an exit trap -- which cannot fire on SIGKILL or power loss. The
residual hole this file pins: such an orphaned lock used to gate the heartbeat
push FOREVER, until a human noticed and cleared it by hand.

The fix is deliberately NOT a flat age bound. The whole contract is the
ASYMMETRY between two claim shapes:

  * `governance-sh-*` is MACHINE-owned, has no human behind it, and wraps a
    minutes-long derive-only pipeline -- so past 2h it is abandoned by
    construction and stops gating.
  * A SESSION claim stays UNBOUNDED at any age. Bounding it would convert a
    LOUD failure (the heartbeat visibly stops pushing, someone asks why) into a
    QUIET one (autostash silently sweeps live uncommitted work with nobody
    told) -- the exact trade this guard exists to refuse. Measured 2026-07-28,
    a flat 6h bound would have de-protected two sessions that were holding real
    uncommitted work at that instant.

So a test that only checked "aged governance lock stops gating" would pass just
as happily against a flat bound, which is the wrong implementation. Every
direction below is pinned WITH its counterpart.

Contracts:
  G1. Aged `governance-sh-*` (> 2h) does NOT gate.
  G2. Equally-aged SESSION claim DOES gate. (G1+G2 together are the asymmetry;
      this is the test a flat bound fails.)
  G3. Fresh `governance-sh-*` (< 2h) DOES gate -- a running regen is never
      aged out, which is why the threshold sits ~an order of magnitude above a
      real regen rather than at its typical duration.
  G4. Boundary: just-under stays gating, just-over stops.
  G5. Undatable `governance-sh-*` does NOT gate (fail-open, matching
      `_active_claim_on_paths`'s documented convention).
  G6. Undatable SESSION claim still gates -- the fail-open in G5 is scoped to
      the machine shape, not a general relaxation.
  G7. A `done` governance lock never gates regardless of age (status check runs
      first; the bound does not resurrect closed entries).
  G8. The bound is unconditional across CALLERS: it applies on the ree-v3 path
      too, which passes its own `max_age_hours`.
  G9. A non-string / absent `session_id` neither crashes nor expires.
  G10. Threshold + prefix stay in sync with the auditor that ANNOUNCES the
      reap (`scripts/audit_stale_claims.py`, bucket G). The aging-out here is
      only non-silent because those two numbers agree.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """tmp REE_Working/REE_assembly layout, as the evidence guard expects.

    The helper resolves TASK_CLAIMS.json as ree_assembly_path.parent /
    "TASK_CLAIMS.json", so the assembly subdir is what gets passed in.
    """
    assembly = tmp_path / "REE_assembly"
    assembly.mkdir()
    return assembly


def _aged(hours: float) -> str:
    """A `claimed_at` stamp `hours` in the past, in task_claim.py's format."""
    stamp = datetime.now(timezone.utc) - timedelta(hours=hours)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_claims(repo: Path, claims: list[dict]) -> None:
    payload = {
        "schema_version": "v1",
        "stale_after_hours": 6,
        "claims": claims,
    }
    (repo.parent / "TASK_CLAIMS.json").write_text(json.dumps(payload))


def _claim(session_id: str, *, age_hours: float | None,
           resources: list[str] | None = None,
           status: str = "active") -> dict:
    entry: dict = {
        "session_id": session_id,
        "status": status,
        "resources": resources or ["REE_assembly/evidence/experiments/x.json"],
    }
    if age_hours is not None:
        entry["claimed_at"] = _aged(age_hours)
    return entry


# --------------------------------------------------------------------------
# G1 + G2 -- the asymmetry. These two are the point of the whole change.
# --------------------------------------------------------------------------

def test_g1_aged_governance_lock_does_not_gate(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("governance-sh-DLAPTOP-4", age_hours=5)])
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_g2_equally_aged_session_claim_still_gates(fake_repo: Path) -> None:
    """The counterpart to G1, at the SAME age, and the flat-bound killer.

    A flat bound (of any value <= 5h) passes G1 and fails here.
    """
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("zealous-merkle-f5dfc8", age_hours=5)])
    assert _active_claim_on_evidence_dir(fake_repo) is True


def test_g2c_shipped_guard_is_demonstrably_not_a_flat_bound(
        fake_repo: Path) -> None:
    """Non-vacuity: show the flat form and the shipped form DIVERGE.

    Same aged session claim, two evaluations. `_active_claim_on_paths` with an
    explicit `max_age_hours=2.0` is the flat bound that was rejected; the
    evidence guard is what actually ships. If someone later "simplifies" the
    guard into a flat bound, these two stop diverging and this test fails --
    which G1 alone would not catch, since G1 passes under either design.
    """
    from runner_remote_control import (
        _EVIDENCE_CLAIM_PREFIXES,
        _active_claim_on_evidence_dir,
        _active_claim_on_paths,
    )

    _write_claims(fake_repo, [_claim("a-live-session", age_hours=5)])

    flat = _active_claim_on_paths(
        fake_repo.parent, _EVIDENCE_CLAIM_PREFIXES, max_age_hours=2.0)
    shipped = _active_claim_on_evidence_dir(fake_repo)

    assert flat is False, "flat bound should drop a 5h session claim"
    assert shipped is True, "shipped guard must keep protecting it"
    assert flat != shipped


def test_g2b_very_old_session_claim_still_gates(fake_repo: Path) -> None:
    """Unbounded means unbounded -- 30 days does not relax a session claim."""
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("some-session-abc123", age_hours=24 * 30)])
    assert _active_claim_on_evidence_dir(fake_repo) is True


# --------------------------------------------------------------------------
# G3 + G4 -- a LIVE regen must never be aged out.
# --------------------------------------------------------------------------

def test_g3_fresh_governance_lock_gates(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("governance-sh-ree-cloud-1", age_hours=0.1)])
    assert _active_claim_on_evidence_dir(fake_repo) is True


@pytest.mark.parametrize(
    "age_hours,expected_gate",
    [(1.9, True), (2.1, False)],
    ids=["just-under-2h-gates", "just-over-2h-does-not"],
)
def test_g4_boundary(fake_repo: Path, age_hours: float,
                     expected_gate: bool) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("governance-sh-x", age_hours=age_hours)])
    assert _active_claim_on_evidence_dir(fake_repo) is expected_gate


# --------------------------------------------------------------------------
# G5 + G6 -- undatable entries, and the scope of the fail-open.
# --------------------------------------------------------------------------

def test_g5_undatable_governance_lock_does_not_gate(fake_repo: Path) -> None:
    """Fail-open, matching `_active_claim_on_paths`'s stated convention.

    Diverges on purpose from audit_stale_claims.py, which leaves an undatable
    lock unreaped: that script MUTATES TASK_CLAIMS.json and needs certainty,
    whereas this only defers one heartbeat push and self-corrects next tick.
    """
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("governance-sh-x", age_hours=None)])
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_g5b_unparseable_claimed_at_treated_as_undatable(
        fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    entry = _claim("governance-sh-x", age_hours=None)
    entry["claimed_at"] = "not-a-timestamp"
    _write_claims(fake_repo, [entry])
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_g6_undatable_session_claim_still_gates(fake_repo: Path) -> None:
    """The G5 fail-open is scoped to the machine shape, not general."""
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(fake_repo, [_claim("human-session-xyz", age_hours=None)])
    assert _active_claim_on_evidence_dir(fake_repo) is True


# --------------------------------------------------------------------------
# G7 / G8 / G9 -- interaction with status, other callers, and junk input.
# --------------------------------------------------------------------------

def test_g7_done_governance_lock_never_gates(fake_repo: Path) -> None:
    from runner_remote_control import _active_claim_on_evidence_dir

    _write_claims(
        fake_repo,
        [_claim("governance-sh-x", age_hours=0.1, status="done")],
    )
    assert _active_claim_on_evidence_dir(fake_repo) is False


def test_g8_bound_applies_on_the_ree_v3_caller_too(tmp_path: Path) -> None:
    """Unconditional across callers, including one passing max_age_hours.

    Paired directions again: the aged governance lock stops gating the ree-v3
    pull, while a same-aged session claim inside that caller's own 6h window
    still does.
    """
    from runner_remote_control import _active_claim_on_ree_v3_code

    ree_v3 = tmp_path / "ree-v3"
    ree_v3.mkdir()
    code_res = ["ree-v3/ree_core/agent.py"]

    _write_claims(
        ree_v3,
        [_claim("governance-sh-x", age_hours=3, resources=code_res)],
    )
    assert _active_claim_on_ree_v3_code(ree_v3) is False

    _write_claims(
        ree_v3,
        [_claim("a-session", age_hours=3, resources=code_res)],
    )
    assert _active_claim_on_ree_v3_code(ree_v3) is True


@pytest.mark.parametrize("session_id", [None, 123, {"a": 1}, []])
def test_g9_non_string_session_id_is_inert(fake_repo: Path,
                                           session_id) -> None:
    """Junk session_id must neither crash nor be treated as expired."""
    from runner_remote_control import _active_claim_on_evidence_dir

    entry = _claim("placeholder", age_hours=99)
    if session_id is None:
        entry.pop("session_id")
    else:
        entry["session_id"] = session_id
    _write_claims(fake_repo, [entry])
    # Not a governance lock -> unbounded session semantics -> still gates.
    assert _active_claim_on_evidence_dir(fake_repo) is True


def test_g9b_expired_helper_is_directly_falsifiable() -> None:
    """Unit-level both-directions check on the predicate itself."""
    from runner_remote_control import _expired_governance_lock

    assert _expired_governance_lock(
        {"session_id": "governance-sh-h", "claimed_at": _aged(9)}) is True
    assert _expired_governance_lock(
        {"session_id": "governance-sh-h", "claimed_at": _aged(0.5)}) is False
    assert _expired_governance_lock(
        {"session_id": "a-session", "claimed_at": _aged(9)}) is False
    assert _expired_governance_lock({}) is False


# --------------------------------------------------------------------------
# G10 -- drift guard against the auditor that announces the reap.
# --------------------------------------------------------------------------

def _audit_stale_claims_source() -> str | None:
    """Source of REE_Working/scripts/audit_stale_claims.py, or None.

    Absent on the cloud workers (only the ree-v3 tree is staged there), so the
    check degrades to a skip rather than a false failure.
    """
    for base in (
        Path("/Users/dgolden/REE_Working/scripts/audit_stale_claims.py"),
        Path(__file__).resolve().parents[3] / "scripts"
        / "audit_stale_claims.py",
    ):
        try:
            if base.is_file():
                return base.read_text(encoding="utf-8")
        except Exception:
            continue
    return None


def test_g10_threshold_and_prefix_stay_in_sync_with_the_auditor() -> None:
    """The aging-out is only non-silent because these two agree.

    This guard stops honouring a governance lock at exactly the age at which
    audit_stale_claims.py (bucket G) reaps and ANNOUNCES it. If the numbers
    drift, a lock can stop gating while nothing tells anyone -- which is the
    silent-lapse failure this whole design refuses.
    """
    from runner_remote_control import (
        _GOVERNANCE_LOCK_MAX_AGE_HOURS,
        _GOVERNANCE_LOCK_PREFIX,
    )

    src = _audit_stale_claims_source()
    if src is None:
        pytest.skip("audit_stale_claims.py not present (cloud worker tree)")

    hours = re.search(r"^GOVERNANCE_REAP_HOURS\s*=\s*([0-9.]+)", src,
                      re.M)
    prefix = re.search(r"^GOVERNANCE_SESSION_PREFIX\s*=\s*[\"']([^\"']+)",
                       src, re.M)
    assert hours, "GOVERNANCE_REAP_HOURS not found in audit_stale_claims.py"
    assert prefix, "GOVERNANCE_SESSION_PREFIX not found in audit_stale_claims.py"

    assert float(hours.group(1)) == _GOVERNANCE_LOCK_MAX_AGE_HOURS
    assert prefix.group(1) == _GOVERNANCE_LOCK_PREFIX
