"""Offline tests for daemon_code_drift.

No systemd, no git, no network: `systemctl show` output and `git log` output
are injected, so every case is deterministic and runs on the Mac.

D1  drift detected      -- daemon started BEFORE the commit -> DRIFT
D2  non-vacuity control -- same fixture, daemon started AFTER -> CURRENT
D3  confirmed incident  -- replays 2026-07-18 sync_daemon/b2d2ef1 -> DRIFT
D4  behind-origin       -- origin ahead of checkout -> BEHIND-ORIGIN, and it
                           BEATS a CURRENT verdict (a daemon newer than a
                           stale checkout is still running old code)
D5  inactive / unknown  -- degrade to a status, never a crash
D6  timestamp math      -- monotonic+btime, immune to TZ abbreviations
D7  render always emits -- the all-clear renders a section too
D8  import surface      -- runner spec excludes subprocess-launched code

ASCII-only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import daemon_code_drift as dcd  # noqa: E402


BOOT = 1780033401  # /proc/stat btime from the hub, 2026-06-...


def show(mono_us, active="active", stamp="Wed 2026-06-24 05:49:51 UTC"):
    return ("ExecMainStartTimestampMonotonic=%d\n"
            "ExecMainStartTimestamp=%s\n"
            "ActiveState=%s\n" % (mono_us, stamp, active))


def git_stub(head=None, origin=None):
    """Fake `git log`. head/origin are (sha, epoch, subject) or None."""
    def _run_git(argv):
        # git -C <repo> log -1 --format=<fmt> <rev> -- <paths...>
        rev = argv[argv.index("log") + 3]
        chosen = origin if rev.startswith("refs/remotes/") else head
        if chosen is None:
            return ""
        return "%s\x00%d\x00%s\n" % chosen
    return _run_git


def spec(**kw):
    base = {"unit": "ree-sync-daemon", "repo": "/nonexistent/ree-v3",
            "branch": "main", "paths": ["coordinator/sync_daemon.py"],
            "note": ""}
    base.update(kw)
    return base


def at(epoch):
    """Monotonic usec such that start == epoch, given BOOT."""
    return (epoch - BOOT) * 1000000


def test_d1_drift_detected():
    started = BOOT + 10000
    commit = started + 3600  # landed an hour AFTER the daemon started
    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show=show(at(started)),
                       _run_git=git_stub(head=("abc123def456", commit, "fix")))
    assert f["status"] == dcd.DRIFT, f
    assert "pre-commit bytecode" in f["detail"]


def test_d2_non_vacuity_control():
    """Same fixture as D1 with the start moved after the commit. If this
    also returned DRIFT, D1 would be asserting nothing."""
    commit = BOOT + 10000
    started = commit + 3600
    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show=show(at(started)),
                       _run_git=git_stub(head=("abc123def456", commit, "fix")))
    assert f["status"] == dcd.CURRENT, f


def test_d3_confirmed_incident_2026_07_18():
    """ree-sync-daemon up since 2026-07-09T18:28:38Z; b2d2ef1 (the queue-writer
    self-deadlock fix) landed 2026-07-18T16:31:00Z. The live process ran
    pre-fix bytecode. This check must call that."""
    started = 1783017 * 1000  # 2026-07-09T18:28:38Z (epoch below)
    started = 1783103318
    commit = 1783873860  # 2026-07-18T16:31:00Z
    assert commit > started
    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show=show(at(started)),
                       _run_git=git_stub(
                           head=("b2d2ef1aaaaa", commit,
                                 "phase3: stop the queue writer deadlocking "
                                 "itself on a failed tick")))
    assert f["status"] == dcd.DRIFT, f
    assert f["commit"] == "b2d2ef1aaa"
    # ~8.9 days of drift should be reported in days, not minutes.
    assert "d" in f["detail"].split("later")[0][-6:]


def test_d4_behind_origin_beats_current():
    """Checkout is stale: daemon started after everything in HEAD (so a
    DRIFT-only check would say CURRENT) while origin carries a newer fix."""
    head_commit = BOOT + 1000
    started = head_commit + 500
    origin_commit = started + 9000
    f = dcd.check_unit(
        spec(), boot_epoch=BOOT, _show=show(at(started)),
        _run_git=git_stub(head=("aaaaaaaaaaaa", head_commit, "old"),
                          origin=("bbbbbbbbbbbb", origin_commit, "new fix")))
    assert f["status"] == dcd.BEHIND_ORIGIN, f
    assert f["behind_origin"]["commit"] == "bbbbbbbbbb"
    assert "pull" in f["detail"]


def test_d4b_same_commit_is_not_behind():
    c = BOOT + 1000
    f = dcd.check_unit(
        spec(), boot_epoch=BOOT, _show=show(at(c + 500)),
        _run_git=git_stub(head=("aaaaaaaaaaaa", c, "x"),
                          origin=("aaaaaaaaaaaa", c, "x")))
    assert f["status"] == dcd.CURRENT, f


def test_d5_inactive_and_unknown():
    f = dcd.check_unit(spec(), boot_epoch=BOOT,
                       _show=show(at(BOOT + 1), active="failed"),
                       _run_git=git_stub(head=("a" * 12, BOOT, "x")))
    assert f["status"] == dcd.INACTIVE

    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show="",
                       _run_git=git_stub(head=("a" * 12, BOOT, "x")))
    assert f["status"] == dcd.UNKNOWN

    # systemd reports the unit but git finds no commit for the paths.
    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show=show(at(BOOT + 1)),
                       _run_git=git_stub(head=None))
    assert f["status"] == dcd.UNKNOWN


def test_d6_timestamp_is_tz_abbreviation_independent():
    """The human ExecMainStartTimestamp says BST while the real start is the
    monotonic-derived epoch. Parsing the string would be an hour off (and
    unparseable in the general case); the monotonic path must ignore it."""
    started = BOOT + 12345
    epoch, active, display = dcd.unit_start_epoch(
        "x", boot_epoch=BOOT,
        _show=show(at(started), stamp="Sat 2026-07-18 20:51:52 BST"))
    assert epoch == started
    assert active == "active"
    assert "BST" in display  # carried for display only


def test_d7_render_always_emits_a_section():
    """An all-clear must still render. A section that appears only on failure
    is indistinguishable from one that silently stopped running."""
    ok = dcd.check_unit(
        spec(), boot_epoch=BOOT, _show=show(at(BOOT + 5000)),
        _run_git=git_stub(head=("a" * 12, BOOT + 1000, "x")))
    md = "\n".join(dcd.render_markdown([ok]))
    assert "Daemon code freshness" in md
    assert "all current" in md
    assert "ree-sync-daemon" in md

    bad = dcd.check_unit(
        spec(), boot_epoch=BOOT, _show=show(at(BOOT + 1000)),
        _run_git=git_stub(head=("a" * 12, BOOT + 5000, "x")))
    md = "\n".join(dcd.render_markdown([bad]))
    assert "1 STALE" in md
    assert "OPERATOR_GUIDE" in md
    assert dcd.worst_status([ok, bad]) == dcd.DRIFT
    assert dcd.worst_status([ok]) == dcd.CURRENT

    assert "not checked" in "\n".join(dcd.render_markdown([]))


def test_d10_non_systemd_host_is_a_clean_no_op():
    """On the Mac the repos exist but the units do not. That must yield an
    empty result (rendered as 'not a systemd host'), not four UNKNOWNs an
    operator has to distinguish from a real fault."""
    assert dcd.check_all(boot_epoch=None) == [] or dcd._boot_epoch() is not None
    if dcd._boot_epoch() is None:  # genuinely non-Linux, e.g. the Mac
        assert dcd.check_all() == []
        assert "not a systemd host" in "\n".join(dcd.render_markdown([]))


def test_d8_runner_import_surface_excludes_subprocess_code():
    """The runner shells out to experiment scripts and ree_core per run, so
    those are never stale-bound. Including them would raise a false DRIFT on
    the runner every time any experiment lands -- which is constantly."""
    runner = [u for u in dcd.UNITS if u["unit"] == "ree-runner"][0]
    joined = " ".join(runner["paths"])
    assert "experiments/" not in joined
    assert "ree_core" not in joined
    assert "experiment_runner.py" in joined
    assert "coordinator_client.py" in joined

    sync = [u for u in dcd.UNITS if u["unit"] == "ree-sync-daemon"][0]
    # sync_daemon imports db + manifest_spool at module scope; a fix to
    # either reaches it only on restart, same as a fix to sync_daemon itself.
    assert "coordinator/db.py" in sync["paths"]
    assert "coordinator/manifest_spool.py" in sync["paths"]


def test_d9_ascii_only_output():
    f = dcd.check_unit(spec(), boot_epoch=BOOT, _show=show(at(BOOT + 1000)),
                       _run_git=git_stub(head=("a" * 12, BOOT + 5000, "x")))
    "\n".join(dcd.render_markdown([f])).encode("ascii")


if __name__ == "__main__":
    import traceback
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS %s" % name)
        except Exception:  # noqa: BLE001
            bad += 1
            print("FAIL %s" % name)
            traceback.print_exc()
    print("\n%d/%d passed" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
