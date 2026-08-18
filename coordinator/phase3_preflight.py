"""Phase 3 pre-cutover gate AND post-cutover steady-state health checks.

Run from the Mac (or any host with coordinator.env + optional SSH to hub).

  /opt/local/bin/python3 phase3_preflight.py                 # auto-detect stage
  /opt/local/bin/python3 phase3_preflight.py --post-cutover  # pin steady state
  /opt/local/bin/python3 phase3_preflight.py --pre-cutover   # pin cutover gate
  /opt/local/bin/python3 phase3_preflight.py --dry-run
  /opt/local/bin/python3 phase3_preflight.py --json

DEPLOYMENT STAGE (added 2026-08-18)
-----------------------------------
This tool was built as a PRE-cutover gate (d9ac962). Phase 3 cut over on
2026-05-29 and has been live ever since, so its routine use today is as a
POST-cutover fleet health check -- which the verdict could not express.

One check, `hub_sync_mode_safe`, asserts the hub has NOT yet been flipped to
SYNC_MODE=authoritative. That is correct for a cutover gate and permanently
wrong afterwards: it FAILs precisely BECAUSE the cutover succeeded. Since any
FAIL sets `ok=False`, the top line read

    VERDICT: FAIL -- 2 blocking check(s)
      Do NOT run phase3_cutover.sh or enable authoritative mode.

on every run, for every reason and no reason -- so callers were reduced to
diffing the per-check lines against a hand-captured baseline to notice a NEW
failure (exactly what chip-20260816-coordinator-canonical-identity-deploy's
brief had to instruct). A verdict that is always FAIL carries no signal, and a
genuine regression hides behind it.

The defect was the VERDICT, not the check. Fixed by giving the run a stage:

  auto (default)   stage inferred from the hub's observed SYNC_MODE, falling
                   back to the local PHASE3_GIT_WRITER_READY flag when the hub
                   was not probed (--dry-run/--mock or an SSH failure).
                   `hub_sync_mode_safe` is ADVISORY here: with no expectation
                   declared, the tool reports the hub's posture and does not
                   adjudicate it. Inferring the stage from that same probe and
                   then blocking on it would be circular.
  --pre-cutover    the original gate. `hub_sync_mode_safe` is BLOCKING with the
                   original polarity (authoritative = FAIL). Implied by
                   --cutover-window, so deploy/phase3_cutover.sh -- which always
                   passes --cutover-window -- keeps its exact prior semantics
                   with no change to that script.
  --post-cutover   steady state. `hub_sync_mode_safe` is BLOCKING with the
                   OPPOSITE polarity: SYNC_MODE=coordinator now means the
                   cutover REGRESSED. Same polarity flip already applied to
                   `phase3_writer_ready` (see _import_sync_daemon_ready).

`fleet_lifecycle` stays blocking in every stage -- it is the check that carries
the real signal, and the reason a bare run is worth making legible at all.

Exit codes:
  0  no blocking check FAILed (SKIP/WARN/advisory-FAIL do not block)
  1  one or more BLOCKING checks FAILed
  2  configuration error (missing coordinator.env / URL)

All stdout/stderr text is ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_BASE = Path.home() / "REE_Working"
DEFAULT_ENV = DEFAULT_BASE / "REE_assembly" / "coordinator.env"
SCHEMA_PATH = HERE / "schema.sql"

# machine_identity.py lives in ROOT (ree-v3/), one directory up from here.
sys.path.insert(0, str(ROOT))
from machine_identity import canonical_machine_name  # noqa: E402

CLOUD_HOSTS = ("ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4")
# Expected lifecycle peers on /shadow/status. The Mac (canonical name
# DLAPTOP; also reported as DLAPTOP-4.local/DLAPTOP-5.local depending on
# macOS LocalHostName drift -- see machine_identity.py) is included because
# the operator runs this script on the Mac, so it MUST be heartbeating to
# the coordinator. _evaluate_fleet_lifecycle() below matches this peer
# through canonical_machine_name() so it is found regardless of which
# hostname alias the heartbeat happened to land under. ree-cloud-4 is in
# shutdown-only mode by default and may legitimately be gracefully_offline
# outside a cutover window -- the fleet_lifecycle check accepts that.
EXPECTED_LIFECYCLE_PEERS = (
    "DLAPTOP",
    "ree-cloud-1",
    "ree-cloud-2",
    "ree-cloud-3",
    "ree-cloud-4",
)
HUB_WG = "10.8.0.1"
DEFAULT_SSH_HOSTS = {
    "ree-cloud-1": "91.98.130.117",
    "ree-cloud-2": "116.203.216.181",
    "ree-cloud-3": "46.62.170.133",
    "ree-cloud-4": "91.99.68.94",
}
HUB_REE_ASSEMBLY = "/home/ree/REE_Working/REE_assembly"
HUB_COORDINATOR_DB = (
    "/home/ree/REE_Working/ree-v3/coordinator/coordinator.db"
)
REGENERABLE_PREFIXES = (
    "evidence/experiments/runner_heartbeats/",
    "evidence/experiments/runner_status/",
    "evidence/experiments/runner_commands/",
)

# Deployment stage. See the module docstring for why this exists.
STAGE_PRE = "pre-cutover"
STAGE_POST = "post-cutover"
STAGE_UNKNOWN = "undetermined"
# Date Phase 3 went live, quoted in operator-facing messages so a reader can
# tell "you already cut over" from "the fleet is unhealthy" without context.
CUTOVER_DATE = "2026-05-29"


@dataclass
class CheckResult:
    check_id: str
    category: str
    status: str  # PASS | FAIL | SKIP | WARN
    message: str
    detail: dict = field(default_factory=dict)
    # A non-blocking (advisory) FAIL/WARN is reported but does not set the
    # exit code. Only `hub_sync_mode_safe` under auto-detected stage is
    # advisory today -- see the module docstring on why that is not a
    # weakening of the gate.
    blocking: bool = True


def _load_env_file(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        cfg[k.strip()] = v.strip()
    return cfg


def _ssh_targets(cfg: dict[str, str]) -> dict[str, str]:
    out = dict(DEFAULT_SSH_HOSTS)
    for h in CLOUD_HOSTS:
        key = "SHADOW_SSH_HOST_" + h
        if cfg.get(key):
            out[h] = cfg[key]
    return out


def _ssh_run(host: str, user: str, remote_cmd: str, *, dry_run: bool,
             timeout: int = 20) -> tuple[bool, str, str]:
    if dry_run:
        return True, "", "dry-run skip"
    target = "%s@%s" % (user, host)
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new",
        target,
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, "", "ssh timed out"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, "", err or "ssh exit %d" % proc.returncode
    return True, (proc.stdout or "").strip(), ""


def _http_get(url: str, token: str | None, path: str,
              timeout: float = 8.0) -> tuple[int | None, dict | None, str]:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        headers={"Authorization": "Bearer " + token} if token else {},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, "HTTP %s" % exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, None, repr(exc)


def _run_check_shadow(url: str, token: str) -> tuple[int, str]:
    script = HERE / "check_shadow.py"
    if not script.exists():
        return 2, "check_shadow.py missing"
    cmd = [sys.executable, str(script), "--url", url, "--token", token]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return 2, "check_shadow timeout"
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else "no output"
    return proc.returncode, summary


def _local_schema_ok() -> tuple[bool, str]:
    if not SCHEMA_PATH.exists():
        return False, "schema.sql missing"
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    required = ("experiments", "results", "heartbeats", "commands", "claim_log")
    missing = [t for t in required if t not in text]
    if missing:
        return False, "schema missing tables: " + ", ".join(missing)
    return True, "schema tables present"


def _import_sync_daemon_ready() -> tuple[bool, str]:
    """The writer-ready flag must match the deployment phase.

    POLARITY CHANGED AT CUTOVER, and this check did not follow for ~2 months.
    Before cutover the invariant was "the writer must NOT be half-enabled",
    so `PHASE3_GIT_WRITER_READY is False` was the PASS. `d98f9a5` (2026-05-28,
    "phase3: flip all three writer-ready flags True") turned the writer on,
    and Phase 3 went live 2026-05-29 -- from which point this check could
    only ever FAIL. Because a single FAIL sets exit 1, the whole preflight
    has returned non-zero unconditionally ever since, which silently voided
    the "run phase3_preflight.py before merging coordinator changes" gate in
    CLAUDE.md and reddened three tests on trunk.

    Post-cutover the invariant is the opposite one, and matches
    `phase3_verify.py:_writer_ready`: sync_daemon IS the sole git writer for
    coordination data, so the flag must be True. A False here would mean the
    writers are stubbed while the runner-side pushes are gated off -- results,
    queue snapshot and telemetry would reach nothing at all.
    """
    try:
        import sync_daemon  # noqa: WPS433 -- intentional local import
        ready = bool(getattr(sync_daemon, "PHASE3_GIT_WRITER_READY", False))
        if ready:
            return True, "PHASE3_GIT_WRITER_READY is True (writer live)"
        return False, ("PHASE3_GIT_WRITER_READY is False -- writer still "
                       "stubbed, but Phase 3 cut over 2026-05-29")
    except Exception as exc:  # noqa: BLE001
        return False, "sync_daemon import failed: %r" % exc


def _observed_sync_mode(raw: str | None) -> str:
    """Classify the hub's `SYNC_MODE=` line into a stage-bearing token.

    `raw` is the grep output, or None when the probe did not run or failed.
    Returns one of: authoritative | coordinator | other | unreadable.
    """
    if not raw:
        return "unreadable"
    if "authoritative" in raw:
        return "authoritative"
    if "coordinator" in raw:
        return "coordinator"
    return "other"


def _resolve_stage(explicit: str | None, observed: str | None,
                   writer_ready: bool) -> tuple[str, str]:
    """Decide which deployment stage this run is judging.

    Returns (stage, source). `source` records WHICH signal decided, so an
    operator can tell a pinned run from an inferred one, and a strong
    inference from a weak one:

      flag              the caller pinned it (--pre-cutover/--post-cutover,
                        or --cutover-window, which implies --pre-cutover).
      hub_sync_mode     read off the live hub. Strong.
      writer_ready_flag inferred from the LOCAL PHASE3_GIT_WRITER_READY
                        constant because the hub was not probed (--dry-run,
                        --mock, or an SSH failure). Weak: it says the
                        substrate in this checkout is post-cutover, not that
                        this deployment is.
      undetermined      no signal at all.

    The hub_sync_mode route is why `hub_sync_mode_safe` is ADVISORY under
    auto: a check cannot both supply the expectation and be judged against
    it. Pin the stage with a flag to make it blocking.
    """
    if explicit:
        return explicit, "flag"
    if observed == "authoritative":
        return STAGE_POST, "hub_sync_mode"
    if observed == "coordinator":
        return STAGE_PRE, "hub_sync_mode"
    if writer_ready:
        return STAGE_POST, "writer_ready_flag"
    return STAGE_UNKNOWN, "undetermined"


def _evaluate_hub_sync_mode(observed: str, raw: str, err: str, *,
                            stage: str,
                            pinned: bool) -> tuple[str, str, bool]:
    """Judge the hub's SYNC_MODE for `stage`. Returns (status, msg, blocking).

    Three polarities, and the reason each is right:

      pinned pre-cutover   authoritative = FAIL. The original invariant: you
                           are about to run phase3_cutover.sh, so the hub must
                           not already be flipped.
      pinned post-cutover  coordinator = FAIL. The opposite invariant, and a
                           real regression signal: post-cutover the hub IS the
                           authoritative writer, so finding it back in
                           coordinator mode means results, the queue snapshot
                           and telemetry have no authoritative writer.
      auto                 ADVISORY. The stage was inferred FROM this probe,
                           so blocking on it would be circular -- it could
                           never fail. Report the posture, block nothing.

    An unreadable hub env is advisory under auto rather than silently lost:
    the SSH round-trip it shares with `sync_daemon_active` fails there too,
    and that check IS blocking in every stage.
    """
    if not pinned:
        if observed == "authoritative":
            return ("PASS",
                    "hub SYNC_MODE=authoritative -- Phase 3 post-cutover "
                    "steady state (advisory; --pre-cutover to gate a cutover)",
                    False)
        if observed == "coordinator":
            return ("PASS",
                    "hub SYNC_MODE=coordinator -- pre-cutover posture "
                    "(advisory; --post-cutover to assert steady state)",
                    False)
        if observed == "other":
            return ("WARN",
                    "hub SYNC_MODE unrecognised: %s (advisory)" % raw,
                    False)
        return ("WARN",
                "cannot read hub env: %s (advisory; sync_daemon_active "
                "carries the blocking SSH signal)" % err,
                False)

    if stage == STAGE_POST:
        if observed == "authoritative":
            return ("PASS", "hub SYNC_MODE=authoritative (expected "
                            "post-cutover)", True)
        if observed == "coordinator":
            return ("FAIL",
                    "hub SYNC_MODE=coordinator -- Phase 3 cutover appears "
                    "REGRESSED (cut over %s); sync_daemon is no longer the "
                    "authoritative writer" % CUTOVER_DATE,
                    True)
        if observed == "other":
            return ("WARN",
                    "hub SYNC_MODE unrecognised: %s" % raw, True)
        return ("FAIL", "cannot read hub env: %s" % err, True)

    # STAGE_PRE (and STAGE_UNKNOWN pinned, which argparse cannot produce)
    if observed == "authoritative":
        return ("FAIL", "hub already SYNC_MODE=authoritative", True)
    if observed == "coordinator":
        return ("PASS", "hub SYNC_MODE=coordinator", True)
    if observed == "other":
        return ("WARN", "hub SYNC_MODE not coordinator: %s" % raw, True)
    return ("FAIL", "cannot read hub env: %s" % err, True)


def _evaluate_fleet_lifecycle(machines: list[dict],
                              *, cutover_window: bool) -> tuple[str, str, dict]:
    """Inspect /shadow/status machines list against EXPECTED_LIFECYCLE_PEERS.

    Returns (status, message, detail). status is PASS/FAIL.

    Policy:
      - steady state (cutover_window=False): each expected peer must be
        `live` OR `gracefully_offline`. `stale` or missing rows FAIL --
        a peer not heartbeating and never announced an intentional
        shutdown is exactly the silent-failure case the operator needs
        to investigate.
      - cutover window (cutover_window=True): each expected peer must
        be `live`. `gracefully_offline` is rejected (the wake_fleet
        helper exits when everyone is live; getting here with anyone
        gracefully_offline means wake didn't actually wake them OR they
        re-shut between wake and preflight). `stale` and missing also
        FAIL.
    """
    # Group observed rows by CANONICAL machine name, not the raw reported
    # string, so a peer heartbeating under an unexpected hostname alias
    # (e.g. the Mac drifting DLAPTOP-4.local <-> DLAPTOP-5.local) is still
    # found. If more than one raw name canonicalizes to the same peer and
    # both are present, prefer whichever row is "live" -- a stale entry
    # under one alias must not mask a live one under another.
    by_canonical: dict[str, dict] = {}
    for m in machines:
        canon = canonical_machine_name(m.get("machine"))
        if not canon:
            continue
        prior = by_canonical.get(canon)
        if prior is None or (m.get("lifecycle_state") == "live"
                              and prior.get("lifecycle_state") != "live"):
            by_canonical[canon] = m
    accepted = ("live",) if cutover_window else ("live", "gracefully_offline")
    bad: list[str] = []
    summary: dict[str, str] = {}
    for peer in EXPECTED_LIFECYCLE_PEERS:
        m = by_canonical.get(canonical_machine_name(peer))
        if m is None:
            summary[peer] = "missing"
            bad.append("%s=missing" % peer)
            continue
        state = m.get("lifecycle_state") or "unknown"
        summary[peer] = state
        if state not in accepted:
            bad.append("%s=%s" % (peer, state))
    if bad:
        mode_label = "cutover" if cutover_window else "steady-state"
        return ("FAIL",
                "%s policy violated: %s" % (mode_label, ", ".join(bad)),
                {"per_machine": summary, "cutover_window": cutover_window,
                 "accepted": list(accepted)})
    mode_label = "cutover" if cutover_window else "steady-state"
    return ("PASS",
            "%s policy OK across %d expected peer(s)" % (
                mode_label, len(EXPECTED_LIFECYCLE_PEERS)),
            {"per_machine": summary, "cutover_window": cutover_window,
             "accepted": list(accepted)})


def run_preflight(
    *,
    env_file: Path | None = None,
    dry_run: bool = False,
    mock: bool = False,
    cutover_window: bool = False,
    stage: str | None = None,
    quiet: bool = False,
) -> dict:
    """Run all checks for `stage`. Returns a serialisable summary dict.

    `stage` is STAGE_PRE / STAGE_POST, or None to auto-detect (see
    _resolve_stage). --cutover-window implies STAGE_PRE: that flag exists
    only for the wake-fleet-then-cut sequence, and pinning it here rather
    than in main() keeps deploy/phase3_cutover.sh -- which passes
    --cutover-window and nothing else -- on exactly its prior semantics
    without touching that script.
    """
    explicit_stage = stage or (STAGE_PRE if cutover_window else None)
    resolved_stage: str | None = None
    stage_source: str | None = None
    observed_sync_mode: str | None = None
    env_path = env_file or Path(
        os.environ.get("COORDINATOR_ENV_FILE", str(DEFAULT_ENV)))
    cfg = _load_env_file(env_path)
    url = cfg.get("COORDINATOR_URL") or os.environ.get("COORDINATOR_URL", "")
    token = (cfg.get("COORDINATOR_LOCAL_TOKEN")
             or os.environ.get("COORDINATOR_TOKEN", ""))
    ssh_user = cfg.get("COORDINATOR_SSH_USER", "ree")
    ssh_hosts = _ssh_targets(cfg)
    hub_ssh = ssh_hosts.get("ree-cloud-1", DEFAULT_SSH_HOSTS["ree-cloud-1"])

    checks: list[CheckResult] = []

    def add(cid: str, cat: str, status: str, msg: str,
            *, blocking: bool = True, **detail):
        checks.append(CheckResult(cid, cat, status, msg, detail=detail,
                                  blocking=blocking))

    # A missing URL is only BLOCKING for a live run. --mock and --dry-run both
    # mean "local checks only, no live coordinator required" (main() already
    # exempts --mock from the env-file existence check), so short-circuiting
    # here defeated their whole purpose: every local check below -- including
    # the phase3_writer_ready implementation gate -- was skipped.
    #
    # This mattered because coordinator.env is GITIGNORED and lives at
    # $HOME/REE_Working/REE_assembly/coordinator.env, entirely outside this
    # repo. It exists only on the operator's Mac. So the three subprocess tests
    # in test_phase3_preflight.py passed on the Mac purely because that
    # untracked local file happened to supply a URL, and failed on any other
    # host -- confirmed 2026-07-27 on the hub, where the file does not exist.
    # Host-dependent green is not green.
    if not url:
        if mock or dry_run:
            add("config", "reachability", "SKIP",
                "no COORDINATOR_URL: local checks only (mock/dry-run)")
        else:
            add("config", "reachability", "FAIL",
                "COORDINATOR_URL not set (coordinator.env)")
            # No hub probe happened, so the stage can only come from a flag
            # (or the weak local writer-ready fallback, unavailable this
            # early). Report it honestly rather than implying a stage.
            return _summary(checks, quiet=quiet,
                            stage=explicit_stage or STAGE_UNKNOWN,
                            stage_source="flag" if explicit_stage
                            else "undetermined")

    # --- reachability / hub ---
    # `not url` reaches here only in mock/dry-run (the live path returned
    # above), and _http_get against an empty URL is not a check, it is a crash.
    if mock or not url:
        add("coordinator_api", "reachability", "SKIP",
            "mock: skip HTTP checks" if mock else "no URL: skip HTTP checks")
        add("hub_health", "hub", "SKIP",
            "mock: skip hub health" if mock else "no URL: skip hub health")
    else:
        st, body, err = _http_get(url, None, "/health")
        if st == 200 and body and body.get("ok"):
            mode = body.get("mode", "?")
            if mode == "coordinator":
                add("hub_health", "hub", "PASS",
                    "health ok mode=coordinator")
            else:
                add("hub_health", "hub", "FAIL",
                    "health ok but mode=%r (expect coordinator)" % mode)
        else:
            add("hub_health", "hub", "FAIL",
                "health unreachable: %s" % (err or st))

        st, body, err = _http_get(url, token, "/shadow/status")
        if st == 200 and body:
            machines = body.get("machines") or []
            add("coordinator_api", "reachability", "PASS",
                "shadow/status ok (%d machines)" % len(machines))
            # Lifecycle policy across the expected fleet. Replaces the
            # legacy SSH-based coordination_mode_uniform check, which
            # produced false positives on surge-only workers
            # (cloud-4 SSH_FAIL even when its absence was intentional).
            status, msg, detail = _evaluate_fleet_lifecycle(
                machines, cutover_window=cutover_window)
            add("fleet_lifecycle", "fleet", status, msg, **detail)
        else:
            add("coordinator_api", "reachability", "FAIL",
                "shadow/status failed: %s" % (err or st))
            add("fleet_lifecycle", "fleet", "SKIP",
                "shadow/status unavailable -- cannot evaluate lifecycle")

    # --- soak (Phase 2 metrics) ---
    if mock or not url:
        add("phase2_shadow_metrics", "soak", "SKIP",
            "mock: skip check_shadow" if mock else "no URL: skip check_shadow")
    elif not token:
        add("phase2_shadow_metrics", "soak", "SKIP",
            "no COORDINATOR_LOCAL_TOKEN for check_shadow")
    else:
        code, summary = _run_check_shadow(url, token)
        if code == 0:
            add("phase2_shadow_metrics", "soak", "PASS",
                "check_shadow exit 0: %s" % summary)
        elif code == 2:
            add("phase2_shadow_metrics", "soak", "WARN",
                "check_shadow NO_SIGNAL: %s" % summary)
        else:
            add("phase2_shadow_metrics", "soak", "FAIL",
                "check_shadow exit %d: %s" % (code, summary))

    # --- implementation gate ---
    ok, note = _import_sync_daemon_ready()
    writer_ready = ok
    # Renamed from `phase3_writer_stub` when the polarity flipped at cutover
    # -- the old id asserted the writer was still a stub. See
    # _import_sync_daemon_ready.
    add("phase3_writer_ready", "implementation",
        "PASS" if ok else "FAIL", note)

    ok, note = _local_schema_ok()
    add("db_schema_present", "data", "PASS" if ok else "FAIL", note)

    # --- hub env / services (SSH) ---
    if dry_run or mock:
        add("hub_sync_mode_safe", "hub", "SKIP",
            "dry-run/mock: skip hub env SSH")
        add("sync_daemon_active", "hub", "SKIP",
            "dry-run/mock: skip systemctl")
        add("hub_git_clean", "hub", "SKIP",
            "dry-run/mock: skip hub git status")
        add("orphaned_claims", "data", "SKIP",
            "dry-run/mock: skip hub DB query")
        # fleet_lifecycle is the new lifecycle-aware fleet check; it runs
        # in the HTTP block above (gated by mock). No SSH stand-in here.
        pass
    else:
        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "grep -E '^SYNC_MODE=' /etc/ree-coordinator.env 2>/dev/null || true",
            dry_run=False)
        raw_mode = out if ok else ""
        observed_sync_mode = _observed_sync_mode(raw_mode)
        # Resolve the stage HERE, before judging the probe: under auto this
        # probe IS the stage signal, which is exactly why the check it feeds
        # is advisory rather than blocking (module docstring).
        resolved_stage, stage_source = _resolve_stage(
            explicit_stage, observed_sync_mode, writer_ready)
        st, msg, blocking = _evaluate_hub_sync_mode(
            observed_sync_mode, raw_mode, err,
            stage=resolved_stage, pinned=(stage_source == "flag"))
        add("hub_sync_mode_safe", "hub", st, msg, blocking=blocking,
            observed_sync_mode=observed_sync_mode, stage=resolved_stage)

        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "systemctl is-active ree-sync-daemon 2>/dev/null",
            dry_run=False)
        if ok and out.strip() == "active":
            add("sync_daemon_active", "hub", "PASS",
                "ree-sync-daemon active")
        else:
            add("sync_daemon_active", "hub", "FAIL",
                "ree-sync-daemon not active: %s" % (err or out))

        ok, out, err = _ssh_run(
            hub_ssh, ssh_user,
            "git -C %s status --porcelain --untracked-files=no 2>/dev/null"
            % HUB_REE_ASSEMBLY,
            dry_run=False)
        if not ok:
            add("hub_git_clean", "hub", "WARN",
                "hub git status failed: %s" % err)
        else:
            bad = []
            for line in out.splitlines():
                entry = line[3:].strip()
                if entry and not any(
                        entry.startswith(p) for p in REGENERABLE_PREFIXES):
                    bad.append(entry)
            if bad:
                add("hub_git_clean", "hub", "FAIL",
                    "hub REE_assembly dirty non-telemetry: %s" % bad[:5])
            else:
                add("hub_git_clean", "hub", "PASS",
                    "hub REE_assembly clean or telemetry-only")

        sql = (
            "SELECT e.queue_id, e.claimed_by_machine "
            "FROM experiments e "
            "LEFT JOIN heartbeats h ON h.machine = e.claimed_by_machine "
            "WHERE e.status='claimed' AND ("
            "  h.last_seen IS NULL OR "
            "  h.last_seen < datetime('now', '-900 seconds') OR "
            "  (h.current_exq IS NOT NULL AND h.current_exq != e.queue_id)"
            ");"
        )
        remote = (
            "sqlite3 -batch %s %s 2>/dev/null | head -20"
            % (HUB_COORDINATOR_DB, repr(sql))
        )
        ok, out, err = _ssh_run(hub_ssh, ssh_user, remote, dry_run=False)
        if not ok:
            add("orphaned_claims", "data", "WARN",
                "hub DB query failed: %s" % err)
        elif out.strip():
            rows = out.strip().splitlines()[:10]
            add("orphaned_claims", "data", "FAIL",
                "possible orphaned claims: %s" % rows)
        else:
            add("orphaned_claims", "data", "PASS",
                "no stale claimed rows in hub DB")

        # The legacy SSH-based COORDINATION_MODE probe used to live here.
        # Replaced by fleet_lifecycle, which reads /shadow/status -- a
        # peer that's not in coordinator mode won't heartbeat, so the
        # lifecycle check catches the same failure mode without
        # SSH-pinging surge-only boxes that may legitimately be off.

    # The hub was never probed (mock/dry-run, or the HTTP block returned
    # early): fall back to the weak local signal so --dry-run still gets a
    # stage-specific verdict instead of a bare "undetermined".
    if resolved_stage is None:
        resolved_stage, stage_source = _resolve_stage(
            explicit_stage, observed_sync_mode, writer_ready)

    blocking_fail = [c for c in checks if c.status == "FAIL" and c.blocking]
    advisory_fail = [c for c in checks
                     if c.status == "FAIL" and not c.blocking]
    required_warn = [c for c in checks if c.status == "WARN"]
    ok = len(blocking_fail) == 0

    return {
        "ok": ok,
        "checked_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "mock": mock,
        "coordinator_url": url,
        "stage": resolved_stage,
        "stage_source": stage_source,
        # fail_count counts BLOCKING failures only -- it is what the verdict
        # line prints as "N blocking check(s)" and what the exit code keys
        # on. Advisory failures are reported separately so nothing is hidden.
        "fail_count": len(blocking_fail),
        "advisory_fail_count": len(advisory_fail),
        "warn_count": len(required_warn),
        "checks": [
            {
                "id": c.check_id,
                "category": c.category,
                "status": c.status,
                "message": c.message,
                "blocking": c.blocking,
                "detail": c.detail,
            }
            for c in checks
        ],
    }


def _summary(checks: list[CheckResult], *, quiet: bool,
             stage: str = STAGE_UNKNOWN,
             stage_source: str = "undetermined") -> dict:
    fail = [c for c in checks if c.status == "FAIL" and c.blocking]
    advisory = [c for c in checks if c.status == "FAIL" and not c.blocking]
    return {
        "ok": len(fail) == 0,
        "checked_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": False,
        "mock": False,
        "coordinator_url": "",
        "stage": stage,
        "stage_source": stage_source,
        "fail_count": len(fail),
        "advisory_fail_count": len(advisory),
        "warn_count": 0,
        "checks": [
            {
                "id": c.check_id,
                "category": c.category,
                "status": c.status,
                "message": c.message,
                "blocking": c.blocking,
                "detail": c.detail,
            }
            for c in checks
        ],
    }


_STAGE_SOURCE_LABEL = {
    "flag": "pinned by flag",
    "hub_sync_mode": "detected from hub SYNC_MODE",
    "writer_ready_flag": "inferred from local PHASE3_GIT_WRITER_READY "
                         "(hub not probed)",
    "undetermined": "no signal",
}


def _verdict_lines(summary: dict) -> list[str]:
    """Build the top-line verdict for this run's stage.

    The whole point of the stage machinery: the verdict must distinguish
    "the fleet is unhealthy" from "you already cut over", and the
    "Do NOT run phase3_cutover.sh" advice must print ONLY where it is
    actionable -- i.e. pre-cutover. Post-cutover that sentence is noise
    that trained callers to ignore the verdict entirely.
    """
    stage = summary.get("stage") or STAGE_UNKNOWN
    n = summary.get("fail_count", 0)
    advisory = summary.get("advisory_fail_count", 0)
    lines: list[str] = []
    if summary.get("ok"):
        if stage == STAGE_POST:
            lines.append("VERDICT: PASS -- Phase 3 steady state healthy")
            lines.append("  (cut over %s; sync_daemon is the authoritative "
                         "writer. This is a health check, not a cutover "
                         "gate.)" % CUTOVER_DATE)
        elif stage == STAGE_PRE:
            lines.append(
                "VERDICT: PASS -- safe to schedule Phase 3 cutover prep")
            lines.append("  (implementation still required before "
                         "SYNC_MODE=authoritative)")
        else:
            lines.append("VERDICT: PASS -- all blocking checks green")
            lines.append("  (deployment stage undetermined; pass "
                         "--pre-cutover or --post-cutover to pin it)")
    else:
        if stage == STAGE_POST:
            lines.append("VERDICT: FAIL -- %d blocking check(s) in Phase 3 "
                         "steady state" % n)
            lines.append("  The fleet or hub is UNHEALTHY -- investigate the "
                         "FAIL line(s) above.")
            lines.append("  (Phase 3 cut over %s; this is a health check, "
                         "not a cutover gate.)" % CUTOVER_DATE)
        elif stage == STAGE_PRE:
            lines.append("VERDICT: FAIL -- %d blocking check(s)" % n)
            lines.append("  Do NOT run phase3_cutover.sh or enable "
                         "authoritative mode.")
        else:
            lines.append("VERDICT: FAIL -- %d blocking check(s)" % n)
            lines.append("  Deployment stage undetermined (hub SYNC_MODE not "
                         "read); pass --pre-cutover or --post-cutover for a "
                         "stage-specific verdict.")
    if advisory:
        lines.append("  (%d advisory FAIL(s) reported above; they do not "
                     "block.)" % advisory)
    return lines


def _print_report(summary: dict) -> None:
    print("Phase 3 preflight @ %s" % summary.get("checked_at", "?"))
    if summary.get("dry_run"):
        print("  (dry-run: SSH checks skipped)")
    if summary.get("mock"):
        print("  (mock: network checks skipped)")
    url = summary.get("coordinator_url") or "?"
    print("  coordinator: %s" % url)
    src = summary.get("stage_source") or "undetermined"
    print("  stage: %s [%s]" % (
        summary.get("stage") or STAGE_UNKNOWN,
        _STAGE_SOURCE_LABEL.get(src, src)))
    for c in summary.get("checks", []):
        # Mark advisory checks inline so a FAIL/WARN that does not block is
        # not mistaken for one that does.
        tag = "" if c.get("blocking", True) else " (advisory)"
        print("  [%s] %s/%s%s: %s" % (
            c["status"], c["category"], c["id"], tag, c["message"]))
    print("")
    for line in _verdict_lines(summary):
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3 pre-cutover gate and post-cutover steady-state "
                    "health checks")
    ap.add_argument("--env-file", default=str(DEFAULT_ENV),
                    help="path to REE_assembly/coordinator.env")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip SSH probes; local checks only")
    ap.add_argument("--mock", action="store_true",
                    help="skip live HTTP/SSH (tests)")
    ap.add_argument("--cutover-window", action="store_true",
                    help="strict fleet_lifecycle policy: require live for "
                         "every expected peer (use after phase3_wake_fleet.sh "
                         "has run successfully; rejects gracefully_offline). "
                         "Implies --pre-cutover.")
    stage_group = ap.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--pre-cutover", dest="stage", action="store_const", const=STAGE_PRE,
        help="pin the pre-cutover gate: hub_sync_mode_safe blocks if the hub "
             "is ALREADY SYNC_MODE=authoritative.")
    stage_group.add_argument(
        "--post-cutover", dest="stage", action="store_const",
        const=STAGE_POST,
        help="pin steady state: hub_sync_mode_safe blocks if the hub is NOT "
             "SYNC_MODE=authoritative (i.e. the cutover regressed). Default "
             "is to auto-detect the stage and report it advisorily.")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON summary on stdout")
    args = ap.parse_args()

    if args.cutover_window and args.stage == STAGE_POST:
        ap.error("--cutover-window implies --pre-cutover; it cannot be "
                 "combined with --post-cutover")

    env_path = Path(args.env_file).expanduser()
    if not env_path.exists() and not args.mock:
        sys.stderr.write("ERROR: missing env file: %s\n" % env_path)
        # --json is a machine contract: a consumer that asked for JSON must get
        # JSON on stdout, not an empty stream and a bare exit 2. Falling
        # through still exits 2 via the coordinator_url check at the end, so
        # the exit-code semantics are unchanged -- only the missing body is
        # fixed. (test_dry_run_local_checks documents exactly this intent:
        # "May FAIL without coordinator.env URL; must emit valid JSON.")
        if not args.json:
            return 2

    summary = run_preflight(
        env_file=env_path,
        dry_run=args.dry_run,
        mock=args.mock,
        cutover_window=args.cutover_window,
        stage=args.stage,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_report(summary)

    if not summary.get("coordinator_url") and not args.mock:
        return 2
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
