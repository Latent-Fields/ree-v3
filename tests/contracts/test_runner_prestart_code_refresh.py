"""Contract tests for the runner's cold-boot code refresh (ExecStartPre).

Background -- 2026-07-30 cold-boot stale-code window:
  The cloud-scaler powers workers on and off continuously, and Python does not
  hot-reload. A worker that powers on loads whatever experiment_runner.py its
  disk held when it was last powered OFF, and then keeps executing that build
  for its ENTIRE session -- even though its own loop pulls newer code to disk
  within ~60s. _refresh_runner_version() already SURFACES that skew as
  "r4530 9077eb7 2026-07-30 (running r4515 e68d52b)", but nothing refreshed the
  code before the process loaded it, so every runner bugfix carried a silent
  one-boot lag.

  Measured that morning:
    - `systemctl cat ree-runner | grep -c ExecStartPre` was 0 on every worker.
    - The only boot-time git refresh was ree-git-sync-repair.timer, which is
      OnBootSec=10min -- it fires roughly TEN MINUTES AFTER the runner has
      already started and claimed work.
    - ree-cloud-3 ran r4515 e68d52b for ~10h after the prepull-stash fix
      (9077eb7073) landed on origin/main at 06:51:52Z.
    - ree-cloud-2 then cold-booted at 07:35:34Z onto r4528 ba79ebb, which is
      OLDER than that fix -- the failure, live, on a fresh boot.

Contracts:
  C1. The tracked unit template has an ExecStartPre that runs the prestart
      refresh script (the fix is present at all).
  C2. That ExecStartPre is FAIL-OPEN (leading '-'), so a boot-time network blip
      cannot turn a hygiene step into a dead runner.
  C3. The script path in the unit is RELATIVE, never an absolute /home/... path.
      This is the hub-safety invariant: systemd applies WorkingDirectory to all
      Exec* lines, and the hub's drop-in repoints it at the ISOLATED
      ~/REE_Working_runner checkout. An absolute path would pull the hub's
      ~/REE_Working instead and wedge every phase3 writer.
  C4. The script exists, is executable, and parses under /bin/sh.
  C5. The script always exits 0 -- the second fail-open layer, independent of
      the unit's leading dash.
  C6. Every git invocation in the script is wrapped in `timeout`. The unit's
      TimeoutStartUSec (1min30s) COVERS ExecStartPre, so an unbounded git would
      burn the start timeout and fail the unit.
  C7. The script clears a partial rebase on failure, so a failed pull degrades
      to "stale code" rather than a wedged checkout that also breaks the
      runner's own first in-loop pull 60s later.
  C8. The script does NOT pull REE_assembly. Scope is deliberately ree-v3 only:
      the defect is code loaded into the Python process, and pulling
      REE_assembly here would reach the hub's writer-adjacent trees.
  C9. Printed output is ASCII-only (journal is read on mixed terminals).
"""

import re
import subprocess
from pathlib import Path

REE_V3 = Path(__file__).resolve().parents[2]
UNIT = REE_V3 / "ree-runner.service"
SCRIPT = REE_V3 / "coordinator" / "deploy" / "runner-prestart-pull.sh"

# Relative path as it must appear in the unit, and as the script's own location.
REL_SCRIPT_PATH = "./coordinator/deploy/runner-prestart-pull.sh"


def _unit_text() -> str:
    assert UNIT.is_file(), f"missing systemd unit template: {UNIT}"
    return UNIT.read_text(encoding="utf-8")


def _script_text() -> str:
    assert SCRIPT.is_file(), f"missing prestart refresh script: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def _exec_start_pre_lines(text: str) -> list[str]:
    """Directive lines only -- comments mentioning ExecStartPre do not count."""
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip().startswith("ExecStartPre=")
    ]


def _script_command_lines(text: str) -> list[str]:
    """Non-comment, non-blank lines of the script."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


# --- C1 --------------------------------------------------------------------

def test_c1_unit_has_execstartpre_running_the_prestart_script():
    lines = _exec_start_pre_lines(_unit_text())
    assert lines, (
        "ree-runner.service has NO ExecStartPre directive. That is exactly the "
        "2026-07-30 cold-boot stale-code defect: the runner loads whatever code "
        "the disk held at last poweroff and never picks up newer code its own "
        "loop pulls, because Python does not hot-reload."
    )
    matching = [ln for ln in lines if REL_SCRIPT_PATH in ln]
    assert matching, (
        f"no ExecStartPre runs {REL_SCRIPT_PATH}; found instead: {lines}"
    )


# --- C2 --------------------------------------------------------------------

def test_c2_execstartpre_is_fail_open():
    lines = [ln for ln in _exec_start_pre_lines(_unit_text())
             if REL_SCRIPT_PATH in ln]
    assert lines, "precondition: C1 must hold"
    for ln in lines:
        value = ln.split("=", 1)[1].strip()
        assert value.startswith("-"), (
            "the prestart ExecStartPre must be fail-open (leading '-') so a "
            "network blip at boot leaves the worker on stale code rather than "
            "with a DEAD runner. A runner on stale code is strictly better "
            f"than no runner. Got: {value!r}"
        )


# --- C3 --------------------------------------------------------------------

def test_c3_script_path_in_unit_is_relative_not_absolute():
    """The hub-safety invariant -- see module docstring C3."""
    lines = [ln for ln in _exec_start_pre_lines(_unit_text())
             if "runner-prestart-pull.sh" in ln]
    assert lines, "precondition: C1 must hold"
    for ln in lines:
        assert "/home/" not in ln, (
            "the prestart script must be referenced by a WorkingDirectory-"
            "relative path, never an absolute /home/... one. systemd applies "
            "WorkingDirectory to every Exec* line, and the hub's drop-in "
            "repoints it at the ISOLATED ~/REE_Working_runner checkout. An "
            "absolute path would refresh the hub's ~/REE_Working tree instead "
            f"and wedge every phase3 writer. Got: {ln!r}"
        )
        assert "REE_Working_runner" not in ln, (
            "do not special-case the hub checkout in the unit template; the "
            "relative path already resolves correctly on every box"
        )
        # Only the first token must be absolute; assert it is an interpreter.
        value = ln.split("=", 1)[1].strip().lstrip("-").strip()
        first = value.split()[0]
        assert first.startswith("/"), (
            f"systemd requires the Exec* executable to be an absolute path; "
            f"got {first!r}"
        )


# --- C4 --------------------------------------------------------------------

def test_c4_script_exists_is_executable_and_parses():
    assert SCRIPT.is_file(), f"missing prestart refresh script: {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & 0o111, (
        f"{SCRIPT.name} should carry the exec bit (git preserves it). Note the "
        "unit invokes it via `/bin/sh <path>` so it works either way, but the "
        "bit should not silently rot."
    )
    r = subprocess.run(
        ["/bin/sh", "-n", str(SCRIPT)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"{SCRIPT.name} is not valid POSIX sh:\n{r.stderr}"
    )


# --- C5 --------------------------------------------------------------------

def test_c5_script_always_exits_zero():
    cmds = _script_command_lines(_script_text())
    assert cmds, "script has no command lines"
    assert cmds[-1] == "exit 0", (
        "the prestart script must END with an unconditional `exit 0`. This is "
        "the second fail-open layer, independent of the unit's leading '-': a "
        "failed code refresh must never prevent the runner from starting. "
        f"Last command line is: {cmds[-1]!r}"
    )


# --- C6 --------------------------------------------------------------------

def test_c6_every_git_call_is_timeout_wrapped():
    """TimeoutStartUSec covers ExecStartPre -- an unbounded git fails the unit.

    Token scan rather than a regex over the raw line: `git` appears behind shell
    keywords (`if`, `!`) and inside command substitution (`$(timeout 5 git ...)`),
    and an alternation-based pattern silently mis-attributes those prefixes.
    Every bare `git` token must be preceded by `timeout <arg>`; matching the
    -2 token with endswith() covers the `$(timeout` substitution form.

    Quoted spans are masked to a PLACEHOLDER token first, so the word "git"
    inside an echo message is not mistaken for a call. Masking rather than
    deleting matters: `timeout "$GIT_TIMEOUT" git pull` must keep its argument
    slot, or the -2 lookback lands on the wrong token.
    """
    offenders = []
    for ln in _script_command_lines(_script_text()):
        masked = re.sub(r"\"[^\"]*\"", "QSTR", ln)
        masked = re.sub(r"'[^']*'", "QSTR", masked)
        toks = masked.split()
        for i, tok in enumerate(toks):
            if tok != "git":
                continue
            if i < 2 or not toks[i - 2].endswith("timeout"):
                offenders.append(ln)
                break
    assert not offenders, (
        "every git call in the prestart script must be wrapped in `timeout`. "
        "The unit's TimeoutStartUSec (1min30s) COVERS ExecStartPre, so a hung "
        "git would burn the start timeout and FAIL the unit -- turning a "
        f"hygiene step into an outage. Unwrapped: {offenders}"
    )


# --- C7 --------------------------------------------------------------------

def test_c7_script_clears_partial_rebase_on_failure():
    text = _script_text()
    assert "git rebase --abort" in text and "git rebase --quit" in text, (
        "on pull failure the script must clear a partial rebase. A failed "
        "`git pull --rebase` can leave .git/rebase-merge behind, which wedges "
        "EVERY later git op on the repo -- including the runner's own first "
        "in-loop pull 60 seconds later. Without this the failure mode escalates "
        "from 'stale code' (self-heals next tick) to 'wedged checkout' (needs "
        "a human)."
    )


# --- C8 --------------------------------------------------------------------

def test_c8_script_does_not_pull_ree_assembly():
    text = _script_text()
    for ln in _script_command_lines(text):
        assert "REE_assembly" not in ln, (
            "the prestart script must not touch REE_assembly. Scope is "
            "deliberately ree-v3 only: the defect is code LOADED INTO THE "
            "PYTHON PROCESS, and all runner code / ree_core / drivers live in "
            "ree-v3. REE_assembly is a data checkout nothing imports at launch "
            "and that the runner's own loop pulls anyway -- refreshing it here "
            "would reach the hub's writer-adjacent trees for no benefit. "
            f"Offending line: {ln!r}"
        )


# --- C9 --------------------------------------------------------------------

def test_c9_script_output_is_ascii_only():
    text = _script_text()
    for i, ln in enumerate(text.splitlines(), start=1):
        try:
            ln.encode("ascii")
        except UnicodeEncodeError as exc:
            raise AssertionError(
                f"{SCRIPT.name}:{i} contains non-ASCII, which renders as "
                f"mojibake on cp1252 terminals reading the journal: {ln!r} "
                f"({exc})"
            ) from None
