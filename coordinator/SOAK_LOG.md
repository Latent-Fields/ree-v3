# Phase-1 shadow soak: explained-divergences ledger

`/shadow/divergence` and `check_shadow.py` count **every** `claim_log`
row with `diverged=1`. `claim_log` is an immutable audit trail by design
(no ack/explained column) and `check_shadow.py` exits `1 DIVERGENCE` on
any non-zero count. The Phase-2 go/no-go is therefore a **human review**:
read the raw count *net of the explained classes below* before deciding.

A divergence here is only a blocker if it is an **unexplained
coordinator-mutex fault** -- the coordinator's atomic claim logic
disagreeing with git about who won a contested claim. Measurement-harness
artifacts are expected and do not gate Phase 2 (see deploy/README.md
"Go / no-go" and the Day-1 false-positive in PLAN.md).

## Explained classes (subtract these from the raw count)

### E2 -- sync mirror ahead of shadow report (git=ok, coord=already_claimed)

`sync_daemon` upserts the authoritative git queue into the coordinator
mirror each tick. A shadow reporter may POST `/claim` with
`git_verdict=ok` after the mirror already shows that machine as owner
(same tick ordering, or a harmless re-report). That is **not** a mutex
fault: git and the coordinator agree on ownership.

Signature: `git_verdict=ok`, `coord_verdict=already_claimed`, mirror row
`status=claimed` with `claimed_by_machine` equal to `machine` on the
log row. Fixed in harness: `db.claim_verdicts_diverge` skips logging;
`check_shadow.py` / `/shadow/status` use `adjusted_divergences`
(blocking count only).

### E1 -- non-shadow machine claim (state-reconcile mirror=pending vs git=claimed)

**Historical only (2026-05-20):** Phase-1 `sync_daemon` no longer runs
pre-upsert state-reconcile when `claim_authority=git`; new E1 rows
should not appear from that path. Rows already in `claim_log` remain
until hub purge. Live E1 still applies when Daniel-PC / EWIN-PC claim
in git without shadow reporting.

Any machine still in `COORDINATION_MODE=git` (not a shadow reporter)
claims experiments in git **without** reporting the claim to the
coordinator. `sync_daemon`'s state-reconcile then sees git=claimed while
its own mirror is still pending and logs `diverged=1`. This is a
**coverage gap, not a mutex fault** -- the coordinator never received a
claim to adjudicate, so its logic never actually disagreed with git.

Signature: `coord_verdict` is `pending`/`null` (NOT `already_claimed`),
`detail` starts `state-reconcile mirror=pending`, `machine` is a known
non-shadow box. Recurs once per experiment that any non-shadow machine
claims until that machine is flipped to shadow. Currently non-shadow:
Daniel-PC, EWIN-PC.

Shadow reporters (2026-05-20): Mac, ree-cloud-2, ree-cloud-3, ree-cloud-4.
cloud-3 onboarded via root SSH (ree had no passwordless sudo); hub peer
10.8.0.13/32 added live with `wg set`.

## Logged instances

| logged_at (UTC) | queue_id | machine | git | coord | class | note |
|---|---|---|---|---|---|---|
| 2026-05-19T08:07:57Z | V3-EXQ-590 | ree-cloud-4 | claimed | pending | E1 | cloud-4 git-mode; claimed V3-EXQ-590 right after the phantom-completion purge (~07:00Z) freed it. No claim reported to coordinator -> mirror stayed pending. Not a mutex fault. Snapshot 2026-05-19T18:58Z: total_claims=4, divergences=1 (this row only). |
| 2026-05-20T19:41Z | V3-EXQ-591 | ree-cloud-2 | pending | claimed | E1 | One-shot state-reconcile before harness fix; stale mirror vs git. |
| 2026-05-20T19:41Z | V3-EXQ-514j | ree-cloud-4 | pending | claimed | E1 | Same. |
| 2026-05-20T19:51Z | V3-EXQ-524a | Mac | ok | already_claimed | E2 | Mirror held Mac claim; shadow re-report. |
| 2026-05-20T19:51Z | V3-EXQ-588b | ree-cloud-2 | ok | already_claimed | E2 | Same pattern on cloud-2. |

Fleet snapshot 2026-05-20 after harness deploy: raw_divergences purged on
hub; `adjusted_divergences=0` with Mac + cloud-2/3/4 FRESH in shadow.

## Phase-2 claim cutover (2026-05-21T08:35Z)

Phase-1 soak gate cleared (2+ days HEALTHY, adjusted_divergences=0).
Maintenance cutover executed from Mac:

- Hub `ree-cloud-1`: `/etc/ree-coordinator.env` -> `COORDINATOR_MODE=coordinator`, `SYNC_MODE=coordinator`; health `{"ok": true, "mode": "coordinator"}`.
- Workers Mac + ree-cloud-1..4: `COORDINATION_MODE=coordinator` in `ree-runner.service.d/shadow.conf` (cloud-1 drop-in created; token minted).
- Resume: `POST /api/coordinator/start` (serve.py) + `systemctl start ree-runner` on clouds.
- Mac runner restarted after stale `stop` command drained; claimed via coordinator (e.g. V3-EXQ-606).

Git still authoritative for results/status/queue commits until Phase 3.
Daniel-PC / EWIN-PC remain manual (git-only if started).

## Phase-3 cutover incident (2026-05-28): scaler powered off the hub

During the Phase-3 cutover, `.github/workflows/cloud-scaler.yml` powered
OFF `ree-worker-1` even though that VM is the coordinator hub
(`ree-coordinator` + `ree-sync-daemon` at 10.8.0.1:8787). Cause: cloud-1's
`ree-runner` had been `systemctl disable`d (separate hub/runner
co-location fix), so the scaler's heartbeat/queue gate read the VM as an
idle worker and issued `hcloud server shutdown`. Effect: full outage of
the coordinator + writers. Recovery: `hcloud server poweron ree-worker-1`
+ manually disable the cloud-scaler workflow. Fix: added explicit
`HUB_NAME=ree-worker-1` guard at the top of the per-worker loop in
cloud-scaler.yml so the hub is never considered for power-on/shutdown.
Update HUB_NAME if the hub VM ever moves.

## Go / no-go reading rule

`adjusted_divergences = raw_divergences - (rows matching E1...En)`.
Advance past Phase 1 only when `adjusted_divergences` is at/near 0 over
several days of real multi-machine load AND every non-E* row was
individually root-caused. If a row's signature does not match an
explained class, it is unexplained -- do NOT cut over.
