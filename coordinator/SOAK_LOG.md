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

### E1 -- non-shadow machine claim (state-reconcile mirror=pending vs git=claimed)

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

## Go / no-go reading rule

`adjusted_divergences = raw_divergences - (rows matching E1...En)`.
Advance past Phase 1 only when `adjusted_divergences` is at/near 0 over
several days of real multi-machine load AND every non-E* row was
individually root-caused. If a row's signature does not match an
explained class, it is unexplained -- do NOT cut over.
