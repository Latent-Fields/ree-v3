"""Contract tests for the claim-aware guard on the ree-v3 `git pull --rebase
--autostash` (experiment_runner._pull_ree_v3 / _ree_v3_pull_blocked, backed by
runner_remote_control._active_claim_on_ree_v3_code).

Why this exists
---------------
The runner pulls the SHARED /Users/dgolden/REE_Working/ree-v3 checkout with
`git pull --rebase --autostash`. That autostashes a concurrent session's
uncommitted substrate work, and when the pop fails NO error reaches the owning
session -- `git status` shows the files unmodified, so the work reads as having
silently vanished. Confirmed 2026-07-27: an in-progress ARC-071 substrate
change was recovered from `stash@{0}` only because the session happened to
notice its own edits were gone, with five older entries behind it unnoticed for
up to eight days. Full triage:
REE_assembly/evidence/planning/ree_v3_orphaned_autostash_triage.md
("Coverage gap", recommendation (a)).

The pre-existing guard (`_active_claim_on_evidence_dir`, 2026-05-01, broadened
2026-05-08 and 2026-06-14) covered only REE_assembly pushes. This widens the
same machinery to ree-v3 substrate paths.

KNOWN LIMITATION, pinned here so it is not later mistaken for full coverage:
the guard fires only for sessions that opened a TASK_CLAIMS entry naming
ree-v3 substrate paths, and substrate sessions do not reliably do that today.
It COMPLEMENTS the detection route (REE_Working/scripts/audit_stashes.py,
REE_Working 4cc9cb9c35), which covers the no-claim case. Neither subsumes the
other -- do not remove or weaken the audit on the strength of this guard.

Failure-direction contracts (the point of the file)
---------------------------------------------------
A guard that fails OPEN (does not skip when it should) silently loses work.
One that fails CLOSED (always skips) starves the runner of code and queue
updates. The directions are therefore chosen per hazard, not uniformly, and
both are pinned:

  C1.  No TASK_CLAIMS.json          -> NOT blocked. This is the cloud-worker
       case: the file does not exist on the hub or the workers at all, so the
       whole guard is inert there and cannot wedge the fleet.
  C2.  Unparseable TASK_CLAIMS.json -> NOT blocked.
  C3.  Active claim with no / empty `resources` -> NOT blocked.
  C4.  Non-active (done) claim on a substrate path -> NOT blocked.
  C5.  Active claim on ree_core/ | experiments/ | tests/ -> guard fires.
  C6.  Claim older than the 6h bound -> NOT blocked. One forgotten `active`
       entry must not disable the runner's sync indefinitely.
  C7.  Claim with missing / unparseable `claimed_at` -> NOT blocked. An
       undatable claim cannot be bounded, and an unbounded gate is the wedge.
  C8.  `REE_assembly/evidence/experiments/...` does NOT fire the ree-v3 guard.
       This is why ree-v3 matching is prefix-ANCHORED rather than substring:
       a substring test for "experiments/" matches every evidence claim, and
       evidence claims are the most common kind, so it would gate the ree-v3
       pull almost permanently for reasons unrelated to ree-v3.
  C9.  `ree-v3/experiment_queue.json` does NOT fire it (DB-authoritative under
       Phase 3; not substrate work the pull could sweep).
  C10. Clean tree + live claim -> NOT blocked. Autostash has nothing to sweep
       when the tree is clean, so skipping then buys no protection and costs
       sync. Measured 2026-07-28: 8 of 10 active claims named ree-v3
       substrate paths, so a claim-only gate would have held the pull nearly
       always.
  C11. Dirty substrate path + live claim -> BLOCKED. The protective case.
  C12. Dirty path OUTSIDE the substrate prefixes + live claim -> NOT blocked.
  C13. Live claim + INDETERMINATE dirty check (git failed/timed out) ->
       BLOCKED. Here a session is known to be at risk and only the tree state
       is unknown; a deferred pull is recoverable and loud, a swept stash is
       neither. Still bounded by the 6h claim age.
  C14. `_rrc` unimportable -> NOT blocked (mirrors the existing REE_assembly
       call sites, which are all gated on `_rrc is not None`).
  C15. An exception raised anywhere inside the guard -> NOT blocked. The
       guard must never be the reason a pull does not happen.
  C16. `_pull_ree_v3` never raises even when git_pull raises.
  C17. The REE_assembly evidence guard is BIT-IDENTICAL after the widening:
       substring matching, no age bound, same two prefixes.
  C18. Every ree-v3 pull call site goes through the guard -- no bare
       `git_pull(REPO_ROOT, "ree-v3")` survives.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402
import runner_remote_control as rrc  # noqa: E402


REPO = Path(__file__).resolve().parents[2]


def _stamp(hours_ago: float = 0.0) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def fake_v3(tmp_path: Path) -> Path:
    """A tmp REE_Working/ree-v3 layout. TASK_CLAIMS.json is resolved as
    ree_v3_path.parent / "TASK_CLAIMS.json", so pass the subdir as the arg."""
    v3 = tmp_path / "ree-v3"
    v3.mkdir()
    return v3


def _write_claims(v3: Path, claims: list[dict]) -> None:
    payload = {"schema_version": "v1", "stale_after_hours": 6, "claims": claims}
    (v3.parent / "TASK_CLAIMS.json").write_text(json.dumps(payload))


def _claim(resources: list[str], *, status: str = "active", age: float = 0.1,
           dated: bool = True) -> dict:
    entry = {"session_id": "s", "status": status, "resources": resources}
    if dated:
        entry["claimed_at"] = _stamp(age)
    return entry


# ---------------------------------------------------------------------------
# The claim matcher itself
# ---------------------------------------------------------------------------


def test_c1_missing_claims_file_does_not_fire(fake_v3: Path) -> None:
    # The cloud-worker case: no TASK_CLAIMS.json on the hub or the workers.
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False


def test_c2_malformed_json_does_not_fire(fake_v3: Path) -> None:
    (fake_v3.parent / "TASK_CLAIMS.json").write_text("{not valid json")
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False


def test_c3_claim_without_resources_does_not_fire(fake_v3: Path) -> None:
    _write_claims(fake_v3, [
        {"session_id": "s", "status": "active", "claimed_at": _stamp()},
        _claim([]),
    ])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False


def test_c4_done_claim_does_not_fire(fake_v3: Path) -> None:
    _write_claims(fake_v3, [_claim(["ree-v3/ree_core/e3.py"], status="done")])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False


@pytest.mark.parametrize("resource", [
    "ree-v3/ree_core/predictors/e3_selector.py",
    "ree-v3/experiments/v3_exq_834_arc071.py",
    "ree-v3/tests/contracts/test_corpus_scan_sharing.py",
    # All three whole-directory spellings occur verbatim in the live
    # TASK_CLAIMS.json; the no-trailing-slash form is the one a plain
    # startswith() silently misses.
    "ree-v3/experiments",
    "ree-v3/experiments/",
    "ree-v3/experiments/diagnostics",
    "ree-v3/ree_core",
    "experiments/_lib/manifest_core.py",   # repo prefix omitted
    "./ree-v3/tests/preflight/test_x.py",  # leading ./
])
def test_c5_substrate_claims_fire(fake_v3: Path, resource: str) -> None:
    _write_claims(fake_v3, [_claim([resource])])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is True, (
        f"C5 FAIL: {resource!r} must fire the ree-v3 substrate guard."
    )


def test_c6_stale_claim_does_not_fire(fake_v3: Path) -> None:
    _write_claims(fake_v3, [_claim(["ree-v3/experiments/foo.py"], age=9.0)])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False, (
        "C6 FAIL: a claim past the 6h bound must not hold the pull -- one "
        "forgotten 'active' entry would otherwise disable sync indefinitely."
    )


def test_c6b_claim_just_inside_the_bound_still_fires(fake_v3: Path) -> None:
    _write_claims(fake_v3, [_claim(["ree-v3/experiments/foo.py"], age=5.5)])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is True


@pytest.mark.parametrize("entry", [
    _claim(["ree-v3/experiments/foo.py"], dated=False),
    {"session_id": "s", "status": "active", "claimed_at": "not-a-date",
     "resources": ["ree-v3/experiments/foo.py"]},
    {"session_id": "s", "status": "active", "claimed_at": None,
     "resources": ["ree-v3/experiments/foo.py"]},
])
def test_c7_undatable_claim_does_not_fire(fake_v3: Path, entry: dict) -> None:
    _write_claims(fake_v3, [entry])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False, (
        "C7 FAIL: an undatable claim cannot be age-bounded, and an unbounded "
        "gate is the wedge failure mode."
    )


@pytest.mark.parametrize("resource", [
    "REE_assembly/evidence/experiments/some_run/manifest.json",
    "REE_assembly/evidence/experiments/claim_evidence.v1.json",
    "evidence/experiments/scripts/build_experiment_indexes.py",
    "REE_assembly/evidence/planning/substrate_queue.json",
    "REE_assembly/docs/claims/claims.yaml",
    # The bare-directory allowance must not leak across repos either.
    "REE_assembly/evidence/experiments",
])
def test_c8_assembly_evidence_claims_do_not_fire_v3_guard(
    fake_v3: Path, resource: str
) -> None:
    """Anchored, not substring: 'experiments/' as a substring would match
    every REE_assembly/evidence/experiments/... claim and gate the ree-v3
    pull almost permanently."""
    _write_claims(fake_v3, [_claim([resource])])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False, (
        f"C8 FAIL: {resource!r} is REE_assembly work and must not gate the "
        "ree-v3 pull. If this fails, the matcher regressed to substring."
    )


@pytest.mark.parametrize("resource", [
    "ree-v3/experiment_queue.json",
    "ree-v3 stash list",              # a real, non-path resource spelling
    "ree-v3/validate_queue.py",
    "ree-v3/experiment_runner.py",
    "WORKSPACE_STATE.md",
])
def test_c9_non_substrate_ree_v3_resources_do_not_fire(
    fake_v3: Path, resource: str
) -> None:
    _write_claims(fake_v3, [_claim([resource])])
    assert rrc._active_claim_on_ree_v3_code(fake_v3) is False


# ---------------------------------------------------------------------------
# The dirty-tree narrowing and the composed decision
# ---------------------------------------------------------------------------


@pytest.fixture
def live_claim(monkeypatch):
    """Force the claim half of the conjunction True."""
    monkeypatch.setattr(rrc, "_active_claim_on_ree_v3_code", lambda _p: True)
    monkeypatch.setattr(experiment_runner, "_rrc", rrc)


def _set_dirty(monkeypatch, value):
    monkeypatch.setattr(experiment_runner, "_ree_v3_code_dirty", lambda: value)


def test_c10_clean_tree_does_not_block(monkeypatch, live_claim) -> None:
    _set_dirty(monkeypatch, False)
    assert experiment_runner._ree_v3_pull_blocked("t") is False, (
        "C10 FAIL: a clean tree has nothing for autostash to sweep, so "
        "skipping buys no protection and only starves the runner of sync."
    )


def test_c11_dirty_substrate_blocks(monkeypatch, live_claim) -> None:
    _set_dirty(monkeypatch, True)
    assert experiment_runner._ree_v3_pull_blocked("t") is True


def test_c13_indeterminate_dirty_check_blocks(monkeypatch, live_claim) -> None:
    _set_dirty(monkeypatch, None)
    assert experiment_runner._ree_v3_pull_blocked("t") is True, (
        "C13 FAIL: with a session known to be at risk and the tree state "
        "unknown, the protective direction is to defer the pull."
    )


def test_c14_rrc_none_does_not_block(monkeypatch) -> None:
    monkeypatch.setattr(experiment_runner, "_rrc", None)
    assert experiment_runner._ree_v3_pull_blocked("t") is False


def test_c15_guard_exception_does_not_block(monkeypatch) -> None:
    def _boom(_p):
        raise RuntimeError("claims read exploded")

    monkeypatch.setattr(rrc, "_active_claim_on_ree_v3_code", _boom)
    monkeypatch.setattr(experiment_runner, "_rrc", rrc)
    pulled = []
    monkeypatch.setattr(
        experiment_runner, "git_pull", lambda r, l: pulled.append(l)
    )
    assert experiment_runner._pull_ree_v3("t") is True
    assert pulled == ["ree-v3"], (
        "C15 FAIL: the guard must never be the reason a pull does not happen."
    )


def test_c16_pull_never_raises(monkeypatch) -> None:
    monkeypatch.setattr(rrc, "_active_claim_on_ree_v3_code", lambda _p: False)
    monkeypatch.setattr(experiment_runner, "_rrc", rrc)

    def _boom(repo, label):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(experiment_runner, "git_pull", _boom)
    assert experiment_runner._pull_ree_v3("t") is True


def test_blocked_pull_does_not_call_git_pull(monkeypatch, live_claim) -> None:
    _set_dirty(monkeypatch, True)
    pulled = []
    monkeypatch.setattr(
        experiment_runner, "git_pull", lambda r, l: pulled.append(l)
    )
    assert experiment_runner._pull_ree_v3("t") is False
    assert pulled == [], "a blocked pull must not reach git_pull"


# ---------------------------------------------------------------------------
# _ree_v3_code_dirty against a real git repo
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "ree-v3"
    (repo / "ree_core").mkdir(parents=True)
    (repo / "experiments").mkdir()
    (repo / "docs").mkdir()
    (repo / "ree_core" / "e3.py").write_text("x = 1\n")
    (repo / "experiments" / "e.py").write_text("y = 1\n")
    (repo / "docs" / "note.md").write_text("hello\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True,
                   env={**dict(__import__("os").environ), **env})
    monkeypatch.setattr(experiment_runner, "REPO_ROOT", repo)
    return repo


def test_dirty_false_on_clean_tree(git_repo: Path) -> None:
    assert experiment_runner._ree_v3_code_dirty() is False


def test_dirty_true_on_modified_substrate(git_repo: Path) -> None:
    (git_repo / "ree_core" / "e3.py").write_text("x = 2\n")
    assert experiment_runner._ree_v3_code_dirty() is True


def test_c12_dirty_outside_substrate_prefixes_is_false(git_repo: Path) -> None:
    (git_repo / "docs" / "note.md").write_text("changed\n")
    assert experiment_runner._ree_v3_code_dirty() is False, (
        "C12 FAIL: only the guarded substrate prefixes count as dirty."
    )


def test_untracked_substrate_file_is_not_dirty(git_repo: Path) -> None:
    """`-uno`: autostash does not touch untracked files, and blocking
    untracked paths are handled by _prepull_stash_blocking_untracked."""
    (git_repo / "experiments" / "brand_new.py").write_text("z = 1\n")
    assert experiment_runner._ree_v3_code_dirty() is False


def test_staged_substrate_change_is_dirty(git_repo: Path) -> None:
    (git_repo / "experiments" / "e.py").write_text("y = 9\n")
    subprocess.run(["git", "add", "experiments/e.py"], cwd=git_repo, check=True)
    assert experiment_runner._ree_v3_code_dirty() is True


def test_dirty_none_when_git_fails(tmp_path: Path, monkeypatch) -> None:
    not_a_repo = tmp_path / "ree-v3"
    not_a_repo.mkdir()
    monkeypatch.setattr(experiment_runner, "REPO_ROOT", not_a_repo)
    assert experiment_runner._ree_v3_code_dirty() is None


# ---------------------------------------------------------------------------
# Non-regression of the REE_assembly guard, and call-site coverage
# ---------------------------------------------------------------------------


def test_c17_evidence_guard_is_bit_identical(tmp_path: Path) -> None:
    """The widening must not change REE_assembly behaviour: substring match,
    no age bound, same two prefixes."""
    asm = tmp_path / "REE_assembly"
    asm.mkdir()

    def w(claims):
        (asm.parent / "TASK_CLAIMS.json").write_text(json.dumps({"claims": claims}))

    # Substring semantics: a nested spelling with no anchoring still fires.
    w([_claim(["some/wrapper/evidence/experiments/x.json"])])
    assert rrc._active_claim_on_evidence_dir(asm) is True

    # No age bound: a claim far past 6h still fires (unlike the ree-v3 guard).
    w([_claim(["REE_assembly/evidence/planning/p.json"], age=99.0)])
    assert rrc._active_claim_on_evidence_dir(asm) is True, (
        "C17 FAIL: the evidence guard must remain unbounded by claim age."
    )

    # Undated claims still fire (the ree-v3 guard skips these; assembly must not).
    w([_claim(["REE_assembly/docs/claims/claims.yaml"], dated=False)])
    assert rrc._active_claim_on_evidence_dir(asm) is True

    # ree-v3 substrate paths still do NOT fire it.
    w([_claim(["ree-v3/experiments/foo.py", "ree-v3/ree_core/e3.py"])])
    assert rrc._active_claim_on_evidence_dir(asm) is False

    assert rrc._EVIDENCE_CLAIM_PREFIXES == ("evidence/", "docs/claims/")


def test_c18_no_unguarded_ree_v3_pull_call_sites() -> None:
    """Every ree-v3 pull must route through _pull_ree_v3. A future call site
    added as a bare git_pull(REPO_ROOT, "ree-v3") would silently reopen the
    autostash hole this guard closes."""
    src = (REPO / "experiment_runner.py").read_text(encoding="utf-8")
    offenders = [
        (n, line) for n, line in enumerate(src.splitlines(), 1)
        if 'git_pull(REPO_ROOT, "ree-v3")' in line
        and not line.lstrip().startswith(('"""', "#", '"'))
    ]
    # The single legitimate occurrence is inside _pull_ree_v3 itself.
    assert len(offenders) == 1, (
        "C18 FAIL: expected exactly one raw ree-v3 pull (inside "
        f"_pull_ree_v3); found {len(offenders)}: {offenders}"
    )
    fn_start = src.index("def _pull_ree_v3(")
    fn_end = src.index("def _check_active_claim_on_file(")
    assert 'git_pull(REPO_ROOT, "ree-v3")' in src[fn_start:fn_end], (
        "C18 FAIL: the one raw ree-v3 pull is not the one inside _pull_ree_v3."
    )


def test_guard_prefixes_are_the_documented_set() -> None:
    assert rrc._REE_V3_CODE_CLAIM_PREFIXES == (
        "ree_core/", "experiments/", "tests/"
    )
    assert rrc._REE_V3_CLAIM_MAX_AGE_HOURS == 6.0


def test_skip_message_is_ascii(monkeypatch, live_claim, capsys) -> None:
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 mojibake)."""
    _set_dirty(monkeypatch, True)
    experiment_runner._ree_v3_pull_blocked("between-pass")
    out = capsys.readouterr().out
    assert out.strip(), "the skip must be announced, not silent"
    out.encode("ascii")  # raises UnicodeEncodeError on any non-ASCII byte
