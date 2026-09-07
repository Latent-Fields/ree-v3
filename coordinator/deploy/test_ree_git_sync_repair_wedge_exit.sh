#!/bin/bash
# Functional test for the WEDGED-path automated exit in ree-git-sync-repair.sh
# (DEFECT A port, 2026-09-07). Design record:
# REE_assembly/evidence/planning/checkoutdiverged_automated_exit_20260907.md
#
# THE DEFECT: the wedged gate demanded every ahead commit be TELEMETRY-ONLY.
# TELEMETRY_RE matches only evidence/experiments/runner_(heartbeats|status|
# commands)/, and those hold ZERO files on origin since the telemetry git path
# was retired -- so the gate was unsatisfiable and no wedged cloud checkout could
# ever self-repair (60 wedged / 274 NEEDS_HUMAN events on ree-cloud-4).
#
# MOST OF THESE ARE NEGATIVE CONTROLS, on purpose. The widened gate is the only
# thing between a stale checkout and `reset --hard origin/<branch>` discarding
# real commits. These are what a later session has to break to widen it further:
#
#     A2  reconcile REFUSES            -> NEEDS_HUMAN, ahead commit survives
#     A4  the LAND fails               -> NEEDS_HUMAN, nothing discarded
#     A5  no prover at all             -> fail closed (the cloud-1/cloud-3 case)
#     A6  precious uncommitted dirt    -> still blocks even with a clean proof
#
# Run:  bash coordinator/deploy/test_ree_git_sync_repair_wedge_exit.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/ree-git-sync-repair.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   - $*"; }
bad() { echo "FAIL - $*"; fail=1; }
has()     { case "$2" in *"$1"*) return 0;; *) return 1;; esac; }
check()     { if has "$2" "$3"; then ok "$1"; else bad "$1 (missing: $2)"; fi; }
check_not() { if has "$2" "$3"; then bad "$1 (unexpectedly present: $2)"; else ok "$1"; fi; }

# A fake python3 that answers ONLY for reconcile_wedge_content.py, consuming one
# exit code per invocation from a list. Anything else exits 0 untouched.
mk_prover() {  # mk_prover <rc1> [rc2] ...
  PDIR="$T/prover.$RANDOM"; mkdir -p "$PDIR"
  : > "$PDIR/reconcile_wedge_content.py"
  printf '%s\n' "$@" > "$PDIR/rcs"; echo 1 > "$PDIR/n"
  cat > "$PDIR/python3" <<'STUB'
#!/bin/bash
d="$(dirname "$0")"
case "$*" in
  *reconcile_wedge_content.py*)
    n=$(cat "$d/n" 2>/dev/null || echo 1)
    rc=$(sed -n "${n}p" "$d/rcs"); echo $((n+1)) > "$d/n"
    [ -n "$rc" ] || rc=1
    exit "$rc" ;;
  *) exit 0 ;;
esac
STUB
  chmod +x "$PDIR/python3"
}

# A worker checkout that is genuinely WEDGED: ahead>0 AND behind>0, with the
# ahead commit touching a NON-telemetry path (the real shape).
mk_wedged() {
  rm -rf "$T/origin.git" "$T/hub" "$T/worker"
  git init -q --bare "$T/origin.git"
  git clone -q "$T/origin.git" "$T/hub" 2>/dev/null
  ( cd "$T/hub" && git checkout -q -b master \
    && git config user.email t@t && git config user.name t \
    && mkdir -p evidence/planning && echo base > evidence/planning/ledger.md \
    && git add -A && git commit -qm base && git push -q origin master )
  git clone -q "$T/origin.git" "$T/worker" 2>/dev/null
  ( cd "$T/worker" && git config user.email t@t && git config user.name t )
  # origin moves on (worker goes behind)
  ( cd "$T/hub" && echo upstream >> evidence/planning/other.md \
    && git add -A && git commit -qm upstream && git push -q origin master )
  # worker commits locally (goes ahead), touching a non-telemetry path
  ( cd "$T/worker" && git fetch -q origin master \
    && echo local >> evidence/planning/ledger.md \
    && git add -A && git commit -qm "local ledger append" )
}

run_repair() {
  LOG="$T/log.txt"; : > "$LOG"
  REE_SYNC_REPAIR_LOG="$LOG" \
  REE_SYNC_REPAIR_LOCKDIR="$T/lock.$RANDOM" \
  REE_SYNC_REPAIR_ASIDE_ROOT="$T/aside" \
  REE_SYNC_REPAIR_REPOS="$T/worker|master" \
  REE_SYNC_REPAIR_PYTHON="${PDIR:-/nonexistent}/python3" \
  REE_SYNC_REPAIR_SCRIPTS_DIR="${PDIR:-/nonexistent}" \
    bash "$SCRIPT" >/dev/null 2>&1
  cat "$LOG"
}
w_ahead()  { git -C "$T/worker" rev-list --count origin/master..HEAD; }
w_behind() { git -C "$T/worker" rev-list --count HEAD..origin/master; }

echo "== A1. wedged + ahead content PROVEN upstream -> repairs (the fix) =="
mk_prover 0; mk_wedged
[ "$(w_ahead):$(w_behind)" = "1:1" ] && ok "precondition: genuinely wedged" || bad "precondition (got $(w_ahead):$(w_behind))"
out="$(run_repair)"
check     "logs the positive proof" "provably upstream -- discard is lossless" "$out"
check     "repairs"                 "REPAIRING"                                "$out"
[ "$(w_behind)" = "0" ] && ok "converged: not behind" || bad "still behind $(w_behind)"
[ "$(w_ahead)"  = "0" ] && ok "converged: not ahead"  || bad "still ahead $(w_ahead)"

echo
echo "== A2. NEGATIVE CONTROL: reconcile REFUSES (exit 2) -> NEEDS_HUMAN =="
mk_prover 2; mk_wedged
out="$(run_repair)"
check     "logs no-proof"         "no positive proof the ahead content is upstream" "$out"
check     "stays NEEDS_HUMAN"     "NEEDS_HUMAN"                                     "$out"
check_not "did NOT repair"        "REPAIRING"                                       "$out"
[ "$(w_ahead)" = "1" ] && ok "ahead commit survives" || bad "ahead commit was discarded"

echo
echo "== A3. stranded content is LANDED first, then repaired (never discarded) =="
mk_prover 3 0 0; mk_wedged
out="$(run_repair)"
check "lands before discarding" "content IS stranded -- landing it before any discard" "$out"
check "then repairs"            "REPAIRING"                                            "$out"
[ "$(w_behind)" = "0" ] && ok "converged" || bad "still behind $(w_behind)"

echo
echo "== A4. NEGATIVE CONTROL: the LAND fails -> nothing is discarded =="
mk_prover 3 1; mk_wedged
out="$(run_repair)"
check     "logs the failed land" "--apply did not succeed" "$out"
check     "stays NEEDS_HUMAN"    "NEEDS_HUMAN"             "$out"
check_not "did NOT repair"       "REPAIRING"               "$out"
[ "$(w_ahead)" = "1" ] && ok "ahead commit survives" || bad "ahead commit was discarded"

echo
echo "== A5. NEGATIVE CONTROL: no prover -> fail closed (the cloud-1/cloud-3 case) =="
# These boxes have no ~/REE_Working checkout, so reconcile_wedge_content.py is
# absent. Behaviour must be byte-identical to before the port.
PDIR=""; mk_wedged
out="$(run_repair)"
check     "stays NEEDS_HUMAN"     "NEEDS_HUMAN" "$out"
check_not "did NOT repair"        "REPAIRING"   "$out"
[ "$(w_ahead)" = "1" ] && ok "ahead commit survives" || bad "ahead commit was discarded"

echo
echo "== A6. NEGATIVE CONTROL: precious uncommitted dirt still blocks =="
mk_prover 0; mk_wedged
echo "real uncommitted work" >> "$T/worker/evidence/planning/ledger.md"
out="$(run_repair)"
check     "refuses on real dirt" "uncommitted non-telemetry change" "$out"
check     "stays NEEDS_HUMAN"    "NEEDS_HUMAN"                      "$out"
check_not "did NOT repair"       "REPAIRING"                        "$out"

echo
echo "=========================================="
[ "$fail" = 0 ] && echo "  ALL PASS" || echo "  FAILURES PRESENT"
exit "$fail"
