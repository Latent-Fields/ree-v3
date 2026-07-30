"""Contract tests for prepull-stash safety and untracked-blocked pull recovery
in experiment_runner.git_pull.

Background -- 2026-07-20 ree-cloud-3 wedge:
  REE_assembly's `git pull` aborted every ~60s for an extended period with
  "untracked working tree files would be overwritten by merge". The blocking
  file was NESTED one level inside a run pack
  (`evidence/experiments/v3_exq_664_.../..._episode_log.json`), and the
  prepull stash only ever matched FLAT manifests -- so nothing in the runner
  could clear it and the worker's checkout stayed behind origin indefinitely.

  Two further defects made the same code path unsafe rather than merely
  ineffective:

  - _postpull_restore_prepull_stash inspected only `git stash list -1` and
    returned early unless the prepull stash was on TOP. Any intervening entry
    (typically the heartbeat's own autostash) stranded it. cloud-3 was still
    holding one from ~2026-06-12.
  - On pop failure it ran `git stash drop`, logging "paths likely on origin
    now". That inference is unsound: the stranded cloud-3 stash contained a
    V3-EXQ-673 manifest (FAIL / does_not_support) present at NO path on
    origin/master. A drop there destroys the only copy of a completed run.

Contracts:
  C1. _find_prepull_stash_ref locates the prepull stash by MESSAGE even when
      it is not the top entry.
  C2. A pop failure KEEPS the stash (never drops it) -- the evidence-loss
      guard.
  C3. A successful restore pops the prepull stash even with a foreign entry
      stacked on top of it.
  C4. _untracked_path_is_redundant is TRUE for byte-identical content and for
      an origin JSON superset, and FALSE when the path is absent upstream or
      any shared key's value differs.
  C5. git_pull recovers from an untracked-blocked pull when the blocking file
      is a NESTED run-pack file provably already on origin (the wedge), and
      does NOT delete a blocking file it cannot verify.

Background -- 2026-07-30 orphaned-stash leak (C8/C9/C10 below):
  _postpull_restore_prepull_stash was called from only THREE of git_pull's
  five exit paths. A pull that failed all three retries, or raised, returned
  without ever popping -- so the entry stayed behind permanently and nothing
  reaped it. At roughly one pull per 62s per worker this compounded:
  ree-cloud-3 held 13 orphaned entries, two of which contained run manifests
  present at NO path on origin/master (V3-EXQ-707c / ARC-110, 40.9 hours of
  compute) -- again the only surviving copies. Second-order, the push/pop
  cycle generated ~6 unreachable objects per tick, reaching 20,326 loose
  objects (84 MiB), which tripped git's unreachable-object guard and wrote
  .git/gc.log -- and the mere PRESENCE of gc.log disables automatic gc on
  that repo indefinitely.

  The defect was STRUCTURAL, not a wrong branch: the call was correct
  everywhere it appeared and simply absent from two paths. So C8 pins the
  structure (the restore must be in a `finally` covering every `return`)
  rather than only the behaviour, which is what lets a NEW exit path added
  later fail the suite instead of silently leaking again.

Contracts:
  C8. Structural -- git_pull's restore call is in a `finally` whose `try`
      encloses every return/raise in the function, and appears nowhere else.
  C9. Behavioural -- a pull that fails every retry, and a pull that raises,
      both still restore the prepull stash.
  C10. The restore DRAINS all prepull entries (the reaper for entries already
      stranded on the fleet), still without ever dropping one, and a path
      that keeps being re-stashed without landing is escalated.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys

from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiment_runner  # noqa: E402


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        timeout=30,
    )


@pytest.fixture()
def origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare origin plus a clone, both with an initial commit on master."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "master")
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "README.md").write_text("seed\n")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "clone", "-q", "--bare", str(seed), str(origin))

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)],
                   capture_output=True, timeout=60)
    _git(clone, "config", "user.email", "t@t.t")
    _git(clone, "config", "user.name", "t")
    return origin, clone


def _push_from_origin_side(origin: Path, tmp_path: Path, rel: str,
                           payload: dict) -> None:
    """Commit `rel` onto origin/master via a throwaway working clone."""
    work = tmp_path / f"work_{abs(hash(rel)) % 10000}"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)],
                   capture_output=True, timeout=60)
    _git(work, "config", "user.email", "t@t.t")
    _git(work, "config", "user.name", "t")
    p = work / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    _git(work, "add", rel)
    _git(work, "commit", "-q", "-m", f"add {rel}")
    _git(work, "push", "-q", "origin", "master")


# --- C1 / C2 / C3: prepull stash handling ---------------------------------

def _make_prepull_stash(clone: Path) -> None:
    rel = "evidence/experiments/v3_exq_999_probe.json"
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": "v3_exq_999_probe"}) + "\n")
    _git(clone, "stash", "push", "--include-untracked", "-m",
         experiment_runner._PREPULL_STASH_MESSAGE, "--", rel)


def _make_foreign_stash(clone: Path) -> None:
    (clone / "README.md").write_text("locally modified\n")
    _git(clone, "stash", "push", "-m", "autostash")


def test_c1_find_prepull_stash_by_message_not_position(origin_and_clone):
    _, clone = origin_and_clone
    _make_prepull_stash(clone)
    _make_foreign_stash(clone)   # now on top

    top = _git(clone, "stash", "list", "-1", "--format=%s").stdout
    assert experiment_runner._PREPULL_STASH_MESSAGE not in top, (
        "fixture invalid: prepull stash must NOT be the top entry"
    )
    ref = experiment_runner._find_prepull_stash_ref(clone)
    assert ref == "stash@{1}"


def test_c2_pop_failure_keeps_stash(origin_and_clone, capsys):
    _, clone = origin_and_clone
    _make_prepull_stash(clone)
    # Recreate the stashed path so the pop collides and fails.
    rel = "evidence/experiments/v3_exq_999_probe.json"
    (clone / rel).parent.mkdir(parents=True, exist_ok=True)
    (clone / rel).write_text('{"run_id": "collision"}\n')

    before = _git(clone, "stash", "list").stdout
    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")
    after = _git(clone, "stash", "list").stdout

    assert experiment_runner._PREPULL_STASH_MESSAGE in after, (
        "prepull stash was DROPPED on pop failure -- this is the "
        "evidence-loss defect"
    )
    assert before == after
    assert "KEPT (not dropped)" in capsys.readouterr().out


def test_c3_restore_pops_prepull_stash_under_foreign_entry(origin_and_clone):
    _, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_999_probe.json"
    _make_prepull_stash(clone)
    _make_foreign_stash(clone)

    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")

    assert (clone / rel).exists(), "prepull stash content not restored"
    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout


# --- C4: redundancy proof --------------------------------------------------

def test_c4_redundancy_predicate(origin_and_clone, tmp_path):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    worker_payload = {"run_id": "v3_exq_664", "episodes": [1, 2, 3]}
    origin_payload = dict(worker_payload)
    origin_payload.update({"machine": "ree-cloud-3", "queue_id": "V3-EXQ-664",
                           "evidence_direction": "supports"})
    _push_from_origin_side(origin, tmp_path, rel, origin_payload)
    _git(clone, "fetch", "-q", "origin")

    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)

    # superset -> redundant
    p.write_text(json.dumps(worker_payload, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is True

    # byte-identical -> redundant
    p.write_text(json.dumps(origin_payload, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is True

    # a shared key disagrees -> NOT redundant
    diverged = dict(worker_payload)
    diverged["episodes"] = [9, 9, 9]
    p.write_text(json.dumps(diverged, indent=2) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, rel, "origin/master") is False

    # absent upstream -> NOT redundant
    other = clone / "evidence/experiments/v3_exq_673_missing.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({"run_id": "v3_exq_673"}) + "\n")
    assert experiment_runner._untracked_path_is_redundant(
        clone, "evidence/experiments/v3_exq_673_missing.json",
        "origin/master") is False


# --- C5: end-to-end wedge recovery ----------------------------------------

def test_c5_nested_untracked_wedge_recovers(origin_and_clone, tmp_path,
                                            capsys):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    payload = {"run_id": "v3_exq_664", "episodes": [1, 2, 3]}
    _push_from_origin_side(origin, tmp_path, rel,
                           {**payload, "machine": "ree-cloud-3"})

    # Worker has the same run pack untracked -- the pull cannot proceed.
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")

    blocked = _git(clone, "pull", "--rebase", "--autostash")
    assert blocked.returncode != 0, "fixture invalid: pull should be blocked"
    assert "untracked working tree files would be overwritten" in \
        (blocked.stderr + blocked.stdout).lower()

    experiment_runner.git_pull(clone, "REE_assembly")

    behind = _git(clone, "rev-list", "--count", "HEAD..origin/master")
    assert behind.stdout.strip() == "0", (
        f"pull still wedged: {capsys.readouterr().out}"
    )
    assert json.loads((clone / rel).read_text())["machine"] == "ree-cloud-3"


def test_c5_unverifiable_blocking_file_is_not_deleted(origin_and_clone,
                                                      tmp_path, capsys):
    origin, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_664_pack/episode_log.json"
    _push_from_origin_side(origin, tmp_path, rel,
                           {"run_id": "v3_exq_664", "episodes": [1, 2, 3]})

    # Local content DISAGREES with origin on a shared key -- unprovable.
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    local_text = json.dumps({"run_id": "v3_exq_664",
                             "episodes": [7, 7, 7]}, indent=2) + "\n"
    p.write_text(local_text)

    experiment_runner.git_pull(clone, "REE_assembly")

    assert p.exists(), "unverifiable blocking file was DELETED"
    assert p.read_text() == local_text
    assert "LEFT IN PLACE" in capsys.readouterr().out


# --- C6: prepull stash matching is GENERATION-AGNOSTIC ---------------------
#
# Regression for the 2026-07-20 hub wedge: `_UNTRACKED_FLAT_MANIFEST_RE`
# hardcoded `v3_`, so the three V4 flat manifests never matched, were never
# stashed, and permanently blocked the pull on the runner checkout at
# /home/ree/REE_Working_runner/REE_assembly (2581 commits behind).

@pytest.mark.parametrize("gen", ["v3", "v4", "v5", "v10"])
def test_c6_flat_manifest_re_matches_every_generation(gen):
    rel = f"evidence/experiments/{gen}_exq_001_probe_20260101T000000Z_{gen}.json"
    assert experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(rel), (
        f"{gen} flat manifest not matched -- it will never be stashed and "
        f"will block the pull permanently (the 2026-07-20 hub wedge)"
    )


@pytest.mark.parametrize("gen", ["V3", "V4", "V5", "V10"])
def test_c6_runner_signal_re_matches_every_generation(gen):
    rel = f"evidence/experiments/_runner_signals/{gen}-EXQ-001.json"
    assert experiment_runner._UNTRACKED_RUNNER_SIGNAL_RE.match(rel), (
        f"{gen} runner signal not matched -- same wedge class as C6 flat"
    )


@pytest.mark.parametrize("rel", [
    # Run-pack dirs stay unmatched (nested path -- the NESTED-PATH carve-out).
    "evidence/experiments/v4_exq_001_probe/manifest.json",
    # Non-generation prefixes stay unmatched.
    "evidence/experiments/INDEX.json",
    "evidence/experiments/version_probe.json",
    "evidence/experiments/runner_status/ree-cloud-3.json",
])
def test_c6_generalisation_does_not_over_match(rel):
    assert not experiment_runner._UNTRACKED_FLAT_MANIFEST_RE.match(rel), (
        f"{rel} newly matched by the generation-agnostic regex -- the "
        f"widening swept a path it must not stash"
    )


def test_c6_v4_manifest_reaches_the_stash_list(origin_and_clone):
    """End-to-end: a V4 flat manifest is selected for the prepull stash."""
    _, clone = origin_and_clone
    rel = "evidence/experiments/v4_exq_001_dr12_probe_20260617T105251Z_v4.json"
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": "v4_exq_001_dr12_probe"}) + "\n")

    assert rel in experiment_runner._untracked_paths_for_prepull_stash(clone)


# --- C7: stdout verdict matching is GENERATION-AGNOSTIC --------------------
#
# RE_EXQ_BANNER is anchored by a literal `===\s+`, so its optional generation
# prefix could not be skipped over and `=== V4-EXQ-001 PASS ===` matched
# nothing -- a V4 run emitting the banner form showed no verdict progress.
# (RE_RUN_DONE_PATTERNS and RE_EXQ_DASHED_OUTCOME were already generation-
# agnostic: the former's first pattern is a bare `verdict:` and the latter is
# unanchored, so .search skips the `V4-`. Asserted here to keep it that way.)

@pytest.mark.parametrize("gen", ["V3", "V4", "V5", "V10"])
def test_c7_banner_verdict_matches_every_generation(gen):
    line = f"=== {gen}-EXQ-001 PASS ==="
    assert experiment_runner.RE_EXQ_BANNER.search(line), (
        f"{gen} banner verdict unmatched -- run shows no PASS/FAIL progress"
    )


@pytest.mark.parametrize("gen", ["V3", "V4", "V5", "V10"])
def test_c7_bare_and_dashed_verdicts_match_every_generation(gen):
    assert any(p.search(f"{gen}-EXQ-001 verdict: PASS")
               for p in experiment_runner.RE_RUN_DONE_PATTERNS)
    assert experiment_runner.RE_EXQ_DASHED_OUTCOME.search(
        f"{gen}-EXQ-001 (probe) -- PASS in 3m")


# --- C8: the restore is STRUCTURALLY unmissable ----------------------------
#
# The 2026-07-30 leak was not a wrong branch -- every call site that existed
# was correct. The call was simply MISSING from two of five exit paths. A
# behavioural test only pins the paths someone thought to write a case for, so
# the structure itself is pinned here: any future `return` added to git_pull
# outside the guarded `try` fails this test by construction.

_RESTORE_FN = "_postpull_restore_prepull_stash"


def _git_pull_ast() -> ast.FunctionDef:
    src = Path(experiment_runner.__file__).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "git_pull":
            return node
    raise AssertionError("git_pull not found in experiment_runner")


def _calls_restore(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == _RESTORE_FN
        for n in ast.walk(node)
    )


def _restore_guards(fn: ast.FunctionDef) -> list[ast.Try]:
    return [t for t in ast.walk(fn)
            if isinstance(t, ast.Try)
            and any(_calls_restore(s) for s in t.finalbody)]


def _exits_outside_guard(fn: ast.FunctionDef) -> list[ast.AST]:
    """Return/raise nodes the restore `finally` does NOT cover.

    Shared by the real-code assertion and its negative control, so the
    detector itself is exercised against a known-bad shape rather than only
    against code that already passes.
    """
    guards = _restore_guards(fn)
    if len(guards) != 1:
        # No single guard -- by definition nothing is covered.
        return [n for n in ast.walk(fn)
                if isinstance(n, (ast.Return, ast.Raise))]
    guarded = {id(n) for stmt in guards[0].body for n in ast.walk(stmt)}
    return [n for n in ast.walk(fn)
            if isinstance(n, (ast.Return, ast.Raise)) and id(n) not in guarded]


# The shape of git_pull BEFORE 2026-07-30, reduced to its control flow: the
# restore is called from some returns and not others. The detector must
# reject this; if it does not, the assertion below is vacuous.
_PRE_FIX_SHAPE = '''
def git_pull(repo_path, label):
    _prepull_stash_blocking_untracked(repo_path, label)
    for attempt in range(3):
        try:
            if ok():
                _postpull_restore_prepull_stash(repo_path, label)
                return
            if retryable() and attempt < 2:
                continue
            if recovered():
                _postpull_restore_prepull_stash(repo_path, label)
                return
            print("warn")
            return                      # <-- LEAKS
        except Exception:
            return                      # <-- LEAKS
'''


def test_c8_detector_rejects_the_pre_fix_shape():
    """Negative control: the pre-fix control flow must FAIL this contract."""
    fn = next(n for n in ast.parse(_PRE_FIX_SHAPE).body
              if isinstance(n, ast.FunctionDef))
    assert _restore_guards(fn) == [], (
        "fixture invalid: the pre-fix shape has no restore `finally`"
    )
    escaping = _exits_outside_guard(fn)
    assert escaping, (
        "the structural detector does NOT flag the known-defective shape -- "
        "test_c8_restore_lives_in_a_finally_covering_every_exit is vacuous"
    )
    assert len(escaping) == 4


def test_c8_restore_lives_in_a_finally_covering_every_exit():
    fn = _git_pull_ast()

    guards = _restore_guards(fn)
    assert len(guards) == 1, (
        f"expected exactly one `finally` calling {_RESTORE_FN} in git_pull, "
        f"found {len(guards)} -- the prepull stash must have ONE restore "
        f"point, or exit paths can diverge again"
    )

    # Every exit from git_pull must be inside the guarded `try` body, so the
    # finally runs for it. `raise` counts: git_pull promises never to raise,
    # but a leaked stash on the way out would be the same evidence loss.
    escaping = _exits_outside_guard(fn)
    assert not escaping, (
        f"git_pull has {len(escaping)} exit(s) at line(s) "
        f"{sorted(n.lineno for n in escaping)} outside the try/finally that "
        f"restores the prepull stash. That is EXACTLY the 2026-07-30 leak: "
        f"an exit path with no pop strands the stash permanently, and it may "
        f"hold the only copy of a completed run's manifest."
    )


def test_c8_restore_is_not_also_called_from_individual_exit_paths():
    """The per-exit-path idiom must be GONE, not merely supplemented.

    Leaving the old inline calls in place beside the finally would double-pop
    and, worse, would let a future edit delete the finally while the suite
    still passed on the surviving inline calls.
    """
    fn = _git_pull_ast()
    guard = next(
        t for t in ast.walk(fn)
        if isinstance(t, ast.Try) and any(_calls_restore(s) for s in t.finalbody)
    )
    in_finally = {
        id(n) for stmt in guard.finalbody for n in ast.walk(stmt)
    }
    stray = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == _RESTORE_FN and id(n) not in in_finally
    ]
    assert not stray, (
        f"{_RESTORE_FN} is still called from inside git_pull's body at "
        f"line(s) {sorted(n.lineno for n in stray)} -- remove the per-exit "
        f"calls; the finally is the single restore point"
    )


# --- C9: behaviour on the two paths that used to leak ----------------------

def _seed_untracked_manifest(clone: Path, rel: str) -> None:
    p = clone / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": Path(rel).stem}) + "\n")


def test_c9_total_pull_failure_still_restores_the_stash(origin_and_clone,
                                                        capsys, tmp_path):
    """The leak proper: every retry fails, function returns, stash stranded."""
    _, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_707c_arc110_20260728T010203Z_v3.json"
    _seed_untracked_manifest(clone, rel)
    payload = (clone / rel).read_text()

    # Make the pull fail unrecoverably for every attempt.
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "gone.git"))

    experiment_runner.git_pull(clone, "REE_assembly")

    out = capsys.readouterr().out
    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout, (
            f"prepull stash was STRANDED by a totally failed pull -- the "
            f"2026-07-30 leak. Runner said: {out}"
        )
    assert (clone / rel).exists(), "manifest not restored to the working tree"
    assert (clone / rel).read_text() == payload


def test_c9_leak_reproduces_when_the_restore_is_disabled(origin_and_clone,
                                                         monkeypatch, tmp_path):
    """Fixture-validity control for the test above.

    With the restore neutralised, the SAME scenario must strand the stash --
    otherwise the passing test proves nothing about the leak, only that this
    scenario never stashed anything in the first place.
    """
    _, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_707e_probe_20260728T010203Z_v3.json"
    _seed_untracked_manifest(clone, rel)
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "gone.git"))

    monkeypatch.setattr(experiment_runner,
                        "_postpull_restore_prepull_stash",
                        lambda *a, **k: None)
    experiment_runner.git_pull(clone, "REE_assembly")
    monkeypatch.undo()

    assert experiment_runner._PREPULL_STASH_MESSAGE in \
        _git(clone, "stash", "list").stdout, (
            "the leak did not reproduce with the restore disabled -- the "
            "C9 scenario does not actually exercise the defective path"
        )
    assert not (clone / rel).exists(), "manifest should be inside the stash"

    # ...and the real restore then reaps exactly that entry.
    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")
    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout
    assert (clone / rel).exists()


def test_c9_exception_path_still_restores_the_stash(origin_and_clone,
                                                    monkeypatch):
    """git_pull's `except Exception` return also used to skip the pop."""
    _, clone = origin_and_clone
    rel = "evidence/experiments/v3_exq_707d_probe_20260728T010203Z_v3.json"
    _seed_untracked_manifest(clone, rel)

    real = experiment_runner._git_run

    def boom(cmd, *a, **kw):
        if len(cmd) > 1 and cmd[1] == "pull":
            raise OSError("simulated git explosion")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(experiment_runner, "_git_run", boom)
    experiment_runner.git_pull(clone, "REE_assembly")
    monkeypatch.undo()

    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout, (
            "prepull stash was stranded by the exception exit path"
        )
    assert (clone / rel).exists()


# --- C10: the reaper, and escalation of a manifest that never lands --------

def test_c10_restore_drains_every_stranded_prepull_entry(origin_and_clone,
                                                         capsys):
    """Entries already stranded across the fleet must be collected.

    ree-cloud-3 held 13. Fixing the leak stops new ones; without a drain the
    existing ones stay invisible in a stash list nobody reads, which is where
    two irreplaceable manifests were found.
    """
    _, clone = origin_and_clone
    rels = [f"evidence/experiments/v3_exq_90{i}_probe_2026070{i}T000000Z_v3.json"
            for i in range(1, 6)]
    for rel in rels:
        _seed_untracked_manifest(clone, rel)
        _git(clone, "stash", "push", "--include-untracked", "-m",
             experiment_runner._PREPULL_STASH_MESSAGE, "--", rel)

    listed = _git(clone, "stash", "list").stdout
    assert listed.count(experiment_runner._PREPULL_STASH_MESSAGE) == 5

    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")
    out = capsys.readouterr().out

    assert experiment_runner._PREPULL_STASH_MESSAGE not in \
        _git(clone, "stash", "list").stdout, (
            f"stranded entries were not reaped: {out}"
        )
    for rel in rels:
        assert (clone / rel).exists(), f"{rel} not restored to the tree"
    assert "reaped 5" in out


def test_c10_drain_steps_past_an_unpoppable_entry_without_dropping_it(
        origin_and_clone, capsys):
    """A collision must not stop the reaper -- nor cost the colliding entry.

    The never-drop rule (C2) is absolute, so an entry whose pop fails stays.
    Before the skip-past, that one entry blocked every older entry behind it
    from ever being reaped.
    """
    _, clone = origin_and_clone
    blocked = "evidence/experiments/v3_exq_911_blocked_20260701T000000Z_v3.json"
    reapable = "evidence/experiments/v3_exq_912_ok_20260701T000000Z_v3.json"
    for rel in (reapable, blocked):      # `blocked` ends up newest / on top
        _seed_untracked_manifest(clone, rel)
        _git(clone, "stash", "push", "--include-untracked", "-m",
             experiment_runner._PREPULL_STASH_MESSAGE, "--", rel)

    # Recreate the top entry's path so its pop collides and fails.
    _seed_untracked_manifest(clone, blocked)
    (clone / blocked).write_text('{"run_id": "collision"}\n')

    experiment_runner._postpull_restore_prepull_stash(clone, "REE_assembly")
    out = capsys.readouterr().out

    remaining = _git(clone, "stash", "list").stdout
    assert remaining.count(experiment_runner._PREPULL_STASH_MESSAGE) == 1, (
        "the unpoppable entry must be KEPT (never dropped) and the other "
        f"reaped -- stash list is:\n{remaining}"
    )
    assert (clone / reapable).exists(), (
        f"the reapable entry behind the collision was not restored: {out}"
    )
    assert "KEPT (not dropped)" in out


def test_c10_repeatedly_stashed_path_is_escalated(capsys):
    """A manifest that never lands must stop being re-stashed in silence.

    On ree-cloud-3 the same file was stashed and popped ~1440 times a day for
    days. Re-stashing is correct; doing it silently is what let it run.
    """
    rel = "evidence/experiments/v3_exq_913_stuck_20260701T000000Z_v3.json"
    experiment_runner._PREPULL_STASH_CYCLES.pop(rel, None)
    try:
        for _ in range(experiment_runner._PREPULL_STUCK_WARN_AT - 1):
            experiment_runner._note_prepull_stash_cycle([rel])
        assert "WARN" not in capsys.readouterr().out, (
            "escalated too early -- a transient undelivered manifest is "
            "normal and must not be noisy"
        )

        experiment_runner._note_prepull_stash_cycle([rel])
        out = capsys.readouterr().out
        assert "WARN" in out and rel in out
        assert "has not landed" in out

        # ...then stays quiet until the next escalation interval.
        for _ in range(experiment_runner._PREPULL_STUCK_WARN_EVERY - 1):
            experiment_runner._note_prepull_stash_cycle([rel])
        assert "WARN" not in capsys.readouterr().out
        experiment_runner._note_prepull_stash_cycle([rel])
        assert "WARN" in capsys.readouterr().out
    finally:
        experiment_runner._PREPULL_STASH_CYCLES.pop(rel, None)


def test_c10_runner_stdout_stays_ascii():
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 mojibake).

    Every runner print is prefixed `[runner]`, so the prefix is a reliable
    marker for the lines that actually reach a terminal.
    """
    offenders = [
        (i, ln) for i, ln in
        enumerate(Path(experiment_runner.__file__).read_text().splitlines(), 1)
        if "[runner]" in ln and not ln.isascii()
    ]
    assert not offenders, f"non-ASCII in runner output: {offenders}"
