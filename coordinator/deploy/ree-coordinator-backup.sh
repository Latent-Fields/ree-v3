#!/bin/bash
# REE coordinator DB backup -- runbook R6 of the git-traffic simplification
# sweep (docs/plans/git_traffic_simplification_sweep_20260901.md section 4).
#
# WHY THIS IS PYTHON AND NOT `sqlite3 .backup`: the runbook originally
# prescribed `sqlite3 coordinator.db ".backup ..."`. The sqlite3 CLI is NOT
# INSTALLED on the hub (`sqlite3: command not found`, confirmed 2026-09-01),
# so that command cannot run there. Python's sqlite3 module ships with the
# stdlib and exposes the same online-backup API, which is safe against a
# LIVE database being written by the coordinator -- a plain `cp` is not.
#
# WHY IT EXISTS: the coordinator DB is now the authoritative store for
# task_claims, chip_ledger, workspace_state_entries and recommendation_log
# entries. R3/R4 retired the heartbeat git mirror and the registries are
# rendered FROM the DB, so git is no longer a second copy of this data.
# This is the category-3 durability net under those retirements.
#
# Rotation: one file per ISO weekday (1-7), so 7 days are retained and the
# set is self-limiting -- no cleanup job to forget.
set -uo pipefail

DB="${COORDINATOR_DB:-/home/ree/REE_Working/ree-v3/coordinator/coordinator.db}"
DEST_DIR="${COORDINATOR_BACKUP_DIR:-/home/ree/backups}"
LOG="${COORDINATOR_BACKUP_LOG:-/home/ree/ree_coordinator_backup.log}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

mkdir -p "$DEST_DIR" || { echo "[$(ts)] backup: FAILED mkdir $DEST_DIR" >> "$LOG"; exit 1; }

if [ ! -f "$DB" ]; then
  echo "[$(ts)] backup: FAILED -- no DB at $DB" >> "$LOG"; exit 1
fi

DEST="$DEST_DIR/coordinator-$(date -u +%u).db"

python3 - "$DB" "$DEST" <<'PY' >> "$LOG" 2>&1
import sqlite3, sys, os, datetime
src_path, dst_path = sys.argv[1], sys.argv[2]
ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = dst_path + ".partial"
try:
    src = sqlite3.connect("file:%s?mode=ro" % src_path, uri=True)
    dst = sqlite3.connect(tmp)
    src.backup(dst)          # online backup: safe against concurrent writers
    dst.close(); src.close()
    chk = sqlite3.connect(tmp).execute("PRAGMA integrity_check").fetchone()[0]
    if chk != "ok":
        print("[%s] backup: FAILED integrity_check=%s (kept %s)" % (ts, chk, tmp))
        raise SystemExit(1)
    os.replace(tmp, dst_path)  # only publish a verified-good file
    size = os.path.getsize(dst_path)
    print("[%s] backup: ok %s (%d bytes, integrity ok)" % (ts, dst_path, size))
except Exception as exc:
    print("[%s] backup: FAILED %s: %s" % (ts, type(exc).__name__, exc))
    raise SystemExit(1)
PY
rc=$?
[ $rc -ne 0 ] && echo "[$(ts)] backup: python exited $rc" >> "$LOG"
exit $rc
