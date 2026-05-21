# Coordinator deploy runbook (Phase 0 standup + Phase 1 shadow)

**Operator narrative (read first):** `../OPERATOR_GUIDE.md` (parallel ->
assess -> retire git). **Fleet checklist:** `FLEET_CHECKLIST.md` (per-host
status, public IPs, Mac SSH caveats).

Hub host = **ree-cloud-1** (public IP, always-on). Topology is hub-and-spoke
WireGuard: every worker reaches only the coordinator, not each other (all
coordination traffic is worker -> coordinator, so a full mesh is
unnecessary and a spoke is simpler to reason about and revoke).

Nothing here changes runner behaviour. Workers stay in `COORDINATION_MODE`
unset (= `git`, byte-identical to today) until you explicitly flip a worker
to `shadow` in Phase 1.

## 0. Standup on ree-cloud-1

```
# as the ree user on ree-cloud-1
cd ~/REE_Working/ree-v3/coordinator
python3 seed_from_queue.py        # DB <- experiment_queue.json
python3 -c "import app"           # sanity import (stdlib only, no pip)
```

### 0a. WireGuard hub

```
sudo apt-get install -y wireguard
wg genkey | tee hub.key | wg pubkey > hub.pub      # keep hub.key secret
sudo cp wg0.coordinator.conf.example /etc/wireguard/wg0.conf
# edit /etc/wireguard/wg0.conf: paste hub.key as PrivateKey; add a
# [Peer] block per worker (worker pubkey + its 10.8.0.x/32)
sudo systemctl enable --now wg-quick@wg0
ip addr show wg0          # expect 10.8.0.1
```

Open ONLY UDP 51820 to the world (the WireGuard handshake). The
coordinator HTTP port is never exposed publicly -- it binds to 10.8.0.1.

### 0b. Tokens (one per worker, individually revocable)

```
python3 gen_token.py DLAPTOP-4.local
python3 gen_token.py Daniel-PC
python3 gen_token.py ree-cloud-2
python3 gen_token.py EWIN-PC
# each prints the token + the exact env lines that worker needs.
# tokens.json (chmod 600) is written next to app.py.
```

### 0c. systemd services

```
sudo cp coordinator.service sync_daemon.service /etc/systemd/system/
sudoedit /etc/ree-coordinator.env       # see env template at bottom
sudo systemctl daemon-reload
sudo systemctl enable --now ree-coordinator ree-sync-daemon
curl -s http://10.8.0.1:8787/health     # {"ok": true, "mode": "shadow"}
```

At this point the coordinator is live but **no worker is talking to it**.
Zero risk: nothing references it yet.

## 1. Shadow (per worker, one at a time)

On a worker, add the WireGuard peer (point it at ree-cloud-1's public IP
in `wg0.peer.conf.example`), bring up `wg0`, confirm `ping 10.8.0.1`, then
add to that runner's environment **only**:

```
COORDINATION_MODE=shadow
COORDINATOR_URL=http://10.8.0.1:8787
COORDINATOR_TOKEN=<token from gen_token.py for this machine>
COORDINATOR_LOG=/home/ree/coordinator_shadow.log   # optional, recommended
```

Restart that runner. It still git-claims exactly as before; it now also
fires best-effort reports. If the coordinator is unreachable the shim logs
to `COORDINATOR_LOG` and the runner is entirely unaffected.

Roll out one machine, watch for a day, then add the rest.

## Go / no-go to advance past Phase 1

Watch `GET http://10.8.0.1:8787/shadow/divergence` (auth with any worker
token):

```
curl -s -H "Authorization: Bearer <tok>" \
  http://10.8.0.1:8787/shadow/divergence | python3 -m json.tool
```

- `divergences` should be at or near **0** across several days of real
  multi-machine load. Each divergence row shows `queue_id`, the
  `git_verdict`, and the coordinator's `coord_verdict` -- investigate every
  one before considering Phase 2.
- Non-zero, explainable divergences (e.g. a worker reported a stale
  `git_verdict` due to its own pull lag) are expected to be rare and
  self-consistent. Unexplained divergence = the coordinator's claim logic
  does not yet match git; do NOT cut over.
- The raw `divergences` count includes measurement-harness artifacts
  (e.g. a still-git-mode machine claiming in git without reporting to the
  coordinator). Read the count **net of the explained classes** logged in
  `../SOAK_LOG.md`; advance only when `adjusted_divergences` is ~0 and
  every unexplained row was root-caused.

## 2. Claim cutover (Phase 2, explicit)

**2026-05-21:** Phase 2 is live on Mac + ree-cloud-1..4 (see `../SOAK_LOG.md`
cutover section). Resume after drain: `POST http://127.0.0.1:8000/api/coordinator/start`
(serve.py) or `deploy/phase2_cutover.sh`.

Only after the adjusted shadow divergence count has stayed near zero under
real load, cut over claims while leaving results/status/queue commits on the
existing git path.

Do not run a mixed claim-authority fleet. Before setting any worker to
`COORDINATION_MODE=coordinator`, drain or pause every worker that is still
in `git`/`shadow` mode, then flip the active workers and resume them. A
git-mode worker still sees `experiment_queue.json` as pending work and can
race outside the coordinator.

On ree-cloud-1:

```
sudoedit /etc/ree-coordinator.env
# set:
#   COORDINATOR_MODE=coordinator
#   SYNC_MODE=coordinator
sudo systemctl restart ree-coordinator ree-sync-daemon
curl -s http://10.8.0.1:8787/health     # {"ok": true, "mode": "coordinator"}
```

On each worker, one at a time:

```
COORDINATION_MODE=coordinator
COORDINATOR_URL=http://10.8.0.1:8787
COORDINATOR_TOKEN=<that worker's token>
```

Restart the runner. In this mode the runner claims via `POST /claim`.
If the coordinator is down, unauthorized, or unreachable, the runner does
**not** run the experiment unclaimed; it skips and retries on the next loop.
Clean shutdown / retry cases call `/claim/release`. Completed items still
push status/results/queue removal through git, then notify `/queue/remove`
so another coordinator-mode worker cannot reclaim the completed item while
git catches up.

Rollback is immediate: set each worker back to `COORDINATION_MODE=shadow`
or `git`, and set ree-cloud-1 back to `COORDINATOR_MODE=shadow` /
`SYNC_MODE=shadow`.

Phase 3 (result cutover, sync_daemon as sole git writer) is described in
`../PLAN.md` and is deliberately NOT enabled by this runbook.

## Starting from the explorer (the button)

Instead of starting each runner by hand, the REE explorer has a "Shadow
Coordination" panel (bottom-right). On the Mac running `serve.py`:

```
cd ~/REE_Working/REE_assembly
cp coordinator.env.example coordinator.env     # gitignored
# set COORDINATOR_URL (the WG IP, e.g. http://10.8.0.1:8787),
# COORDINATOR_LOCAL_TOKEN (this Mac's gen_token.py token), and the
# SHADOW_SSH_HOST_ree-cloud-* lines to the boxes' WireGuard IPs --
# bare names like 'ree-cloud-1' do NOT resolve on the Mac.
# restart serve.py
```

The button then starts the Mac runner in shadow, SSH-starts the
ree-cloud-1 coordinator + cloud-2/3/4 runners, and the panel shows the
live verdict. Daniel-PC / EWIN-PC are reported as "start manually" (no
inbound SSH). Unreachable boxes are reported FAILED, never fatal.

## Fleet integrity (compute takeover watch)

Separate from shadow divergence: host-level read-only checks that the
workers still look like REE boxes (not a substitute for Hetzner account
security or SSH hardening).

```bash
# One-time: capture authorized_keys fingerprints (local, gitignored JSON)
/opt/local/bin/python3 deploy/fleet_integrity_check.py --write-baseline

# Manual or cron (exit 0/1/2/3 for alerting)
/opt/local/bin/python3 deploy/fleet_integrity_check.py
```

Reads SSH targets from `REE_assembly/coordinator.env` when present
(`SHADOW_SSH_HOST_ree-cloud-*`). Probes per host:

- `authorized_keys` SHA-256 vs baseline (ALERT on change)
- newly **enabled** systemd units vs baseline (WARN; docker/miner ALERT)
- top CPU processes (ALERT on miner-like command lines)
- `nvidia-smi` compute apps not matching python/ree (ALERT)
- hub `wg show wg0 peers` count (WARN if below threshold)
- coordinator `/shadow/status` machine names (ALERT if unknown)

Cron template: `deploy/fleet_integrity_cron.example`. Baseline template:
`deploy/fleet_integrity_baseline.json.example`.

Science data in public GitHub repos is unchanged; this guards **compute**
and **SSH/control-plane** drift on the Hetzner fleet.

## Daily soak watch

From any mesh machine (e.g. your Mac), once per day during the shadow
period:

```
python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
  --url http://10.8.0.1:8787 --token <any-worker-token>
```

It prints one verdict and sets an exit code you can alert on:

- `0 HEALTHY` -- claims observed, 0 divergence, >=1 live machine. This is
  the state that, sustained over several days of real load, clears
  Phase 2.
- `1 DIVERGENCE` -- the coordinator disagreed with git at least once. Do
  NOT advance; investigate every printed row.
- `2 NO SIGNAL` -- coordinator healthy but no claim traffic / all
  heartbeats stale. The soak is not exercising anything: runners are
  drained or were not flipped to `COORDINATION_MODE=shadow`.
- `3 UNREACHABLE` -- WireGuard down, coordinator down, or token bad.

## Rollback

Remove the four env vars from a worker (or set `COORDINATION_MODE=git`) and
restart it -- instantly back to today's behaviour. Stopping the coordinator
service has no effect on workers (the shim just logs unreachable).

## /etc/ree-coordinator.env template

```
COORDINATOR_DB=/home/ree/REE_Working/ree-v3/coordinator/coordinator.db
COORDINATOR_TOKENS_FILE=/home/ree/REE_Working/ree-v3/coordinator/tokens.json
COORDINATOR_BIND_HOST=10.8.0.1
COORDINATOR_BIND_PORT=8787
COORDINATOR_STALE_HOURS=6
COORDINATOR_MODE=shadow
COORDINATOR_QUEUE_FILE=/home/ree/REE_Working/ree-v3/experiment_queue.json
SYNC_INTERVAL=60
SYNC_MODE=shadow
```
