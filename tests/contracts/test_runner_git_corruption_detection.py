"""Contract: zero-byte (corrupt) loose git objects must surface, not sit silent.

Confirmed 2026-08-01 (ree-cloud-4): a hard VM reboot mid-write left 4 loose
objects in REE_assembly's `.git/objects/` as empty (0-byte) files. This is
NOT a runner-timeout issue -- `graceful_timeout` (SIGTERM before SIGKILL) had
already been live for over a week when it happened, and the corrupt object's
mtime lines up exactly with the end of a `journalctl --list-boots` entry. A
hard reset gives zero chance for any application-level cleanup, graceful or
not, so this is a distinct failure class from the git-timeout contract in
`test_runner_git_timeout_not_fatal.py`.

The gap this closes: `git_pull` already prints a raw git stderr dump on
every failed pull ("object file ... is empty"), but that is indistinguishable
at a glance from an ordinary transient failure, and unlike a timeout (which
is self-resolving) a corrupt object will NOT clear on its own -- it sat for
two days before being noticed by hand. `_scan_zero_byte_loose_objects` /
`_record_git_corruption_scan` detect it on the first tick and print a loud,
distinctly-labeled line; `write_status` surfaces the live count into the
fleet-visibility `runner_status/<host>.json` channel.

Deliberately NOT `git fsck --full`: fsck reads and hashes every reachable
object (minutes on a large repo) and can itself die mid-scan on the first
corrupt object it meets. This is a pure filesystem stat() walk.
"""

import json
from pathlib import Path

import experiment_runner


def _init_bare_objects_dir(repo_path: Path) -> Path:
    objects_dir = repo_path / ".git" / "objects"
    objects_dir.mkdir(parents=True)
    (objects_dir / "pack").mkdir()
    (objects_dir / "info").mkdir()
    return objects_dir


def _write_loose_object(objects_dir: Path, sha: str, content: bytes = b"") -> Path:
    subdir = objects_dir / sha[:2]
    subdir.mkdir(exist_ok=True)
    f = subdir / sha[2:]
    f.write_bytes(content)
    return f


# --- the primitive ---------------------------------------------------------

def test_scan_finds_nothing_in_a_healthy_repo(tmp_path):
    objects_dir = _init_bare_objects_dir(tmp_path)
    _write_loose_object(objects_dir, "de229fabc9644411c0b396b332516433cea638a5",
                         content=b"not actually empty, just fake content")
    assert experiment_runner._scan_zero_byte_loose_objects(tmp_path) == []


def test_scan_finds_a_zero_byte_object(tmp_path):
    objects_dir = _init_bare_objects_dir(tmp_path)
    _write_loose_object(objects_dir, "de229fabc9644411c0b396b332516433cea638a5")
    hits = experiment_runner._scan_zero_byte_loose_objects(tmp_path)
    assert hits == ["de/229fabc9644411c0b396b332516433cea638a5"]


def test_scan_finds_multiple_zero_byte_objects_across_subdirs(tmp_path):
    objects_dir = _init_bare_objects_dir(tmp_path)
    shas = [
        "004154addfd3b294acdf7d5913c93b0d4050e68b",
        "92ab65260df1cdabd7a7832c4d1b6928f24bfa6d",
        "de229fabc9644411c0b396b332516433cea638a5",
        "e109ae44ef85d57bca6b41042c584e1362ff26c7",
    ]
    for sha in shas:
        _write_loose_object(objects_dir, sha)
    hits = experiment_runner._scan_zero_byte_loose_objects(tmp_path)
    assert len(hits) == 4
    assert set(hits) == {f"{s[:2]}/{s[2:]}" for s in shas}


def test_scan_ignores_pack_and_info_directories(tmp_path):
    """`pack/` and `info/` sit directly under objects/ but are not loose-object
    dirs -- a naive walk that doesn't filter by 2-char dirname would either
    error on their contents or misreport them as corrupt loose objects."""
    objects_dir = _init_bare_objects_dir(tmp_path)
    (objects_dir / "pack" / "pack-abc123.pack").write_bytes(b"")  # legitimately 0 bytes mid-write is not our concern here
    (objects_dir / "info" / "commit-graph").write_bytes(b"")
    assert experiment_runner._scan_zero_byte_loose_objects(tmp_path) == []


def test_scan_on_missing_objects_dir_returns_empty_not_raises(tmp_path):
    """A repo_path with no .git/objects at all (e.g. not a git repo, or
    called before init) must degrade quietly -- this is a diagnostic, it
    must never be the thing that breaks a caller."""
    assert experiment_runner._scan_zero_byte_loose_objects(tmp_path) == []
    (tmp_path / ".git").mkdir()
    assert experiment_runner._scan_zero_byte_loose_objects(tmp_path) == []


# --- the recording / logging layer -----------------------------------------

def test_record_scan_logs_loudly_on_first_detection(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {})
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_EVENT_COUNT", 0)
    objects_dir = _init_bare_objects_dir(tmp_path)
    _write_loose_object(objects_dir, "de229fabc9644411c0b396b332516433cea638a5")

    experiment_runner._record_git_corruption_scan(tmp_path, "REE_assembly")

    out = capsys.readouterr().out
    assert "GIT CORRUPTION #1" in out
    assert "1 zero-byte loose object(s) in REE_assembly" in out
    assert "will NOT self-heal" in out
    assert experiment_runner._GIT_CORRUPTION_STATE["REE_assembly"] == 1


def test_record_scan_does_not_reprint_on_unchanged_count(tmp_path, capsys, monkeypatch):
    """Rate-limited on CHANGE, not printed every tick -- mirrors the
    `_on_status_write_error` lesson: an unrated persistent-fault print would
    scroll the experiment's own output away."""
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {})
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_EVENT_COUNT", 0)
    objects_dir = _init_bare_objects_dir(tmp_path)
    _write_loose_object(objects_dir, "de229fabc9644411c0b396b332516433cea638a5")

    experiment_runner._record_git_corruption_scan(tmp_path, "REE_assembly")
    capsys.readouterr()  # drain the first-detection print
    experiment_runner._record_git_corruption_scan(tmp_path, "REE_assembly")

    assert "GIT CORRUPTION" not in capsys.readouterr().out


def test_record_scan_logs_when_cleared(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {"REE_assembly": 1})
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_EVENT_COUNT", 1)
    _init_bare_objects_dir(tmp_path)  # clean this time -- nothing zero-byte

    experiment_runner._record_git_corruption_scan(tmp_path, "REE_assembly")

    out = capsys.readouterr().out
    assert "CLEARED (1 -> 0" in out
    assert experiment_runner._GIT_CORRUPTION_STATE["REE_assembly"] == 0


def test_record_scan_tracks_labels_independently(tmp_path, monkeypatch):
    """ree-v3 and REE_assembly are pulled as separate `git_pull` calls with
    distinct labels -- corruption in one must not be conflated with the other."""
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {})
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_EVENT_COUNT", 0)
    clean_repo = tmp_path / "clean"
    dirty_repo = tmp_path / "dirty"
    clean_repo.mkdir()
    dirty_objects = _init_bare_objects_dir(dirty_repo)
    _write_loose_object(dirty_objects, "de229fabc9644411c0b396b332516433cea638a5")

    experiment_runner._record_git_corruption_scan(clean_repo, "ree-v3")
    experiment_runner._record_git_corruption_scan(dirty_repo, "REE_assembly")

    assert experiment_runner._GIT_CORRUPTION_STATE == {"ree-v3": 0, "REE_assembly": 1}


def test_record_scan_never_raises_on_a_scan_exception(tmp_path, monkeypatch):
    """A diagnostic must not be the thing that breaks the pull it guards."""
    def _boom(repo_path):
        raise OSError("simulated disk error")
    monkeypatch.setattr(experiment_runner, "_scan_zero_byte_loose_objects", _boom)
    experiment_runner._record_git_corruption_scan(tmp_path, "REE_assembly")  # must not raise


# --- integration: git_pull calls the scan on every invocation --------------

def test_git_pull_records_corruption_even_though_pull_itself_fails(tmp_path, monkeypatch):
    """The scan must run and be visible in state regardless of pull outcome
    -- 3 of the 4 objects in the confirmed incident were NOT on the path any
    pull actually needed, so gating the scan on pull failure would have
    missed them for however long nothing else happened to reference them."""
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {})
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_EVENT_COUNT", 0)
    objects_dir = _init_bare_objects_dir(tmp_path)
    _write_loose_object(objects_dir, "de229fabc9644411c0b396b332516433cea638a5")

    experiment_runner.git_pull(tmp_path, "REE_assembly")  # not a real repo; pull fails, must not raise

    assert experiment_runner._GIT_CORRUPTION_STATE.get("REE_assembly") == 1


# --- fleet-visibility surfacing ---------------------------------------------

def test_write_status_surfaces_nonzero_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE",
                         {"REE_assembly": 2, "ree-v3": 0})
    path = tmp_path / "status.json"
    experiment_runner.write_status({"machine": "ree-cloud-4"}, path)
    written = json.loads(path.read_text())
    assert written["git_corruption"] == {"REE_assembly": 2}, (
        "clean labels (count 0) must be filtered out, not persisted as noise"
    )


def test_write_status_omits_git_corruption_key_before_any_scan(tmp_path, monkeypatch):
    """Before the first git_pull has run, _GIT_CORRUPTION_STATE is empty --
    the key must be absent entirely, not present-and-empty, so a reader
    can't mistake "not yet checked" for "checked, found nothing"."""
    monkeypatch.setattr(experiment_runner, "_GIT_CORRUPTION_STATE", {})
    path = tmp_path / "status.json"
    experiment_runner.write_status({"machine": "ree-cloud-4"}, path)
    written = json.loads(path.read_text())
    assert "git_corruption" not in written
