#!/bin/bash
# Functional test for the untracked-collision pre-flight in ree-git-sync-repair.sh
# (2026-09-06, chip-20260903-untracked-manifest-collision-recurring-class, option B).
#
# Builds a bare origin, a "hub" clone that commits manifests (the phase3
# writer's role), and a "worker" clone that is behind and holds UNTRACKED local
# copies of the same paths: one a strict subset (queue_id-poorer, pretty-printed
# -- the measured 2026-09-03 shape), one with a differing value, one non-JSON.
# Expects: the subset copy is removed and the ff proceeds; the other two are
# moved aside (never deleted); an incoming file with no local copy is untouched;
# a run with NO collision behaves exactly as before; tracked dirt still blocks.
#
# Run:  bash coordinator/deploy/test_ree_git_sync_repair_untracked.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/ree-git-sync-repair.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   - $*"; }
bad() { echo "FAIL - $*"; fail=1; }
run_repair() {
  REE_SYNC_REPAIR_LOG="$LOG" REE_SYNC_REPAIR_LOCKDIR="$T/lock" \
    REE_SYNC_REPAIR_ASIDE_ROOT="$ASIDE" REE_SYNC_REPAIR_REPOS="$T/worker|master" \
    bash "$SCRIPT"
}

git init -q --bare "$T/origin.git"
git clone -q "$T/origin.git" "$T/hub" 2>/dev/null
cd "$T/hub" && git checkout -q -b master && git config user.email t@t && git config user.name t
mkdir -p evidence/experiments && echo base > evidence/experiments/base.txt
git add evidence && git commit -qm base && git push -q origin master
git clone -q "$T/origin.git" "$T/worker" 2>/dev/null
cd "$T/worker" && git config user.email t@t && git config user.name t

# hub commits the enriched manifests (queue_id attached by the coordinator)
cd "$T/hub"
printf '{"run_id":"r1","outcome":"FAIL","queue_id":"V3-EXQ-1","nested":{"a":1,"b":[1,2]}}' > evidence/experiments/r1.json
printf '{"run_id":"r2","outcome":"FAIL","queue_id":"V3-EXQ-2","metric":0.5}' > evidence/experiments/r2.json
printf 'hub text\n' > evidence/experiments/r3.txt
printf '{"run_id":"r4","outcome":"PASS"}' > evidence/experiments/r4.json
git add evidence && git commit -qm "phase3: manifests" && git push -q origin master

# worker holds untracked local copies written by its own driver
cd "$T/worker"
cat > evidence/experiments/r1.json <<'J'
{
  "run_id": "r1",
  "outcome": "FAIL",
  "nested": {
    "a": 1,
    "b": [1, 2]
  }
}
J
printf '{"run_id":"r2","outcome":"FAIL","metric":0.7}' > evidence/experiments/r2.json
printf 'worker text\n' > evidence/experiments/r3.txt

LOG="$T/log"; ASIDE="$T/aside"
run_repair

grep -q "UNTRACKED_CLEARED.* evidence/experiments/r1.json" "$LOG" && ok "subset copy cleared" || bad "subset copy not cleared: $(cat "$LOG")"
grep -q "UNTRACKED_ASIDE.*r2.json(DIFFERS)" "$LOG" && ok "differing copy moved aside" || bad "differing copy not aside"
grep -q "UNTRACKED_ASIDE.*r3.txt(NOTJSON)" "$LOG" && ok "non-json copy moved aside" || bad "non-json copy not aside"
grep -q "worker SYNCED (ff-only" "$LOG" && ok "fast-forward succeeded after pre-flight" || bad "no SYNCED line: $(cat "$LOG")"
[ "$(git -C "$T/worker" rev-parse HEAD)" = "$(git -C "$T/hub" rev-parse HEAD)" ] && ok "worker at origin tip" || bad "worker not at tip"
aside_r2="$(find "$ASIDE" -name r2.json | head -1)"
[ -n "$aside_r2" ] && grep -q '"metric":0.7' "$aside_r2" && ok "differing copy preserved byte-for-byte under aside" || bad "differing copy lost"
[ -n "$(find "$ASIDE" -name r3.txt | head -1)" ] && ok "non-json copy preserved" || bad "non-json copy lost"
grep -q '"queue_id":"V3-EXQ-1"' "$T/worker/evidence/experiments/r1.json" && ok "origin's enriched r1 now on disk" || bad "r1 not origin's"
grep -q '"outcome":"PASS"' "$T/worker/evidence/experiments/r4.json" && ok "non-colliding incoming file untouched" || bad "r4 wrong"

# no-collision run: behaves as before (SYNCED, no pre-flight lines)
cd "$T/hub" && echo more > evidence/experiments/more.txt && git add evidence && git commit -qm more && git push -q origin master
: > "$LOG"
run_repair
grep -q "UNTRACKED_" "$LOG" && bad "pre-flight fired with no collision" || ok "no pre-flight lines without a collision"
grep -q "worker SYNCED" "$LOG" && ok "plain sync still works" || bad "plain sync broken: $(cat "$LOG")"

# tracked dirt is still adjudicated exactly as before: a locally modified
# TRACKED file that origin also changes must block, and nothing is touched.
cd "$T/hub" && echo v2 > evidence/experiments/base.txt && git add evidence && git commit -qm base2 && git push -q origin master
cd "$T/worker" && echo local > evidence/experiments/base.txt
: > "$LOG"
run_repair
grep -q "BEHIND_NOT_SYNCED" "$LOG" && grep -q local "$T/worker/evidence/experiments/base.txt" && ok "tracked dirt still blocks, untouched" || bad "tracked-dirt path changed: $(cat "$LOG")"

[ $fail = 0 ] && echo "ALL PASS" || { echo "FAILURES"; exit 1; }
