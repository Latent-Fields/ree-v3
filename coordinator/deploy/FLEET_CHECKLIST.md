# Coordinator fleet checklist (Phase 1 shadow + Phase 2 claims)

**Start here for the story:** `../OPERATOR_GUIDE.md` (parallel run ->
assess -> claim cutover -> result cutover).

Cross-session operator sheet for bringing **ree-cloud-2 / 3 / 4** into the
shadow soak. Hub = **ree-cloud-1** (`ree-worker-1`, public `91.98.130.117`,
WireGuard `10.8.0.1`). Plan-of-record: `../PLAN.md`, runbook: `README.md`.

**2026-05-21 status (Phase 2 claim cutover):**

| Host | Public SSH | WG IP | coordination | Runner | Notes |
|------|------------|-------|--------------|--------|-------|
| Mac | local | 10.8.0.10 | coordinator | active | via serve `POST /api/coordinator/start` |
| ree-cloud-1 | 91.98.130.117 | 10.8.0.1 | coordinator (hub) | active | hub + worker; `_HEARTBEAT_WRITE=1` (runner skips local telemetry files; sync_daemon publishes) |
| ree-cloud-2 | 116.203.216.181 | 10.8.0.12 | coordinator | active | |
| ree-cloud-3 | 46.62.170.133 | 10.8.0.13 | coordinator | active | |
| ree-cloud-4 | 91.99.68.94 | 10.8.0.14 | coordinator | active | |

Hub health: `{"ok": true, "mode": "coordinator"}`. Claims authoritative on
coordinator; git still used for results/status until Phase 3.

---

## Phase-2 soak checklist (gate before Phase 3)

Phase 2 cutover landed **2026-05-21** (hard drain, not ideal). Treat the fleet
as **early Phase 2** until this checklist is green for **several days** of
real queue traffic. Phase 3 is **not** a config flip: `sync_daemon` needs
`SYNC_MODE=authoritative` **and** `--i-understand-phase3` (stubbed off until
then). See `README.md` and `../sync_daemon.py`.

### What Phase 2 means (still running)

| Path | Authority |
|------|-----------|
| **Claims** | Coordinator `POST /claim` (workers `COORDINATION_MODE=coordinator`) |
| **Results / status / queue file** | Runners still **git push** (`--auto-sync`) |
| **Heartbeats** | Still `runner_remote_control` -> `REE_assembly` (autostash risk **remains**) |
| **Hub** | `COORDINATOR_MODE=coordinator`, `SYNC_MODE=coordinator` on ree-cloud-1 |

### Daily checks (Mac, 2 minutes)

```bash
# Hub mode + traffic
curl -s http://10.8.0.1:8787/health
# Expect: {"ok": true, "mode": "coordinator"}

# Divergence / heartbeats (shadow endpoint still valid in Phase 2)
grep '^COORDINATOR_LOCAL_TOKEN=' ~/REE_Working/REE_assembly/coordinator.env | cut -d= -f2- | \
  xargs -I{} /opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
    --url http://10.8.0.1:8787 --token {}

# Explorer panel
curl -s http://127.0.0.1:8000/api/shadow/status | python3 -c \
  "import sys,json; d=json.load(sys.stdin); g=d.get('guide',{}); \
   print(g.get('phase_label'), d.get('verdict'), 'div', d.get('adjusted_divergences'))"
```

| Check | Pass criterion |
|-------|----------------|
| Hub `/health` | `mode` == `coordinator` |
| `check_shadow.py` | Exit **0** (HEALTHY), `adjusted_divergences` == 0 |
| Explorer guide | **Phase 2 -- claim cutover (live)** |
| Machines | Mac + ree-cloud-1..4 **FRESH** (~10 min), each running or idle cleanly |
| No duplicate EXQ | At most **one** machine `current_exq` per queue_id (affinity respected) |
| Coordinator log | No sustained `POST /claim` failure storms on workers |

### Soak period (human gate)

- [ ] **3+ calendar days** after Phase-2 cutover with runners active and queue non-empty
- [ ] At least one **full EXQ** completes end-to-end (claim -> run -> result on git -> `/queue/remove` notified)
- [ ] No unexplained `DIVERGENCE` rows (`../SOAK_LOG.md`); E1/E2 only if classified
- [ ] No mixed fleet: every experiment host on `COORDINATION_MODE=coordinator` (not `git`/`shadow`)
- [ ] Daniel-PC / EWIN-PC either **idle** or flipped; if git-only and claiming, expect E1 noise
- [ ] `machine_affinity` honored (post-rescue: e.g. V3-EXQ-590a only on ree-cloud-3 if partial there)

### Before scheduling Phase 3 (maintenance window)

Same discipline as Phase 2 cutover:

1. **Pause / drain** all workers (no mixed result writers).
2. Hub: `SYNC_MODE=authoritative` + enable Phase-3 sync_daemon flags per runbook (when implemented).
3. Workers: stop pushing results/queue via git; coordinator + sync_daemon own writes.
4. Verify manifests land in git via sync_daemon only; then retire heartbeat push.

### Phase-2 rollback (claims back to git)

Per-worker: set `COORDINATION_MODE=shadow` or remove drop-in; hub:
`COORDINATOR_MODE=shadow`, `SYNC_MODE=shadow`; restart services. See **Rollback** below.

### Known post-cutover watch items (2026-05-21)

- Prefer **pause -> finish current EXQ -> flip** next time; hard `systemctl kill` loses in-flight compute.
- Stale `stop` in `runner_commands/<machine>.json` can exit runners before first claim; clear pending commands.
- Coordinator claim can stick on wrong host until `POST /claim/release` + runner stop on that host.
- Partial checkpoints (e.g. V3-EXQ-590a `_partial/`) are **per-machine**; pin `machine_affinity` before resume.
- **Hub VM must never be powered off by the cloud-scaler.** `ree-worker-1` runs `ree-coordinator` + `ree-sync-daemon`; the 2026-05-28 cutover-day outage was triggered by the scaler shutting down the hub after cloud-1's `ree-runner` was disabled. `.github/workflows/cloud-scaler.yml` now has a `HUB_NAME=ree-worker-1` guard at the top of the per-worker loop -- if the hub VM ever moves (rename, migration, second hub), update `HUB_NAME` in that workflow in the same change. See `coordinator/SOAK_LOG.md` "Phase-3 cutover incident (2026-05-28)".

---

## Fleet map (Hetzner names)

| Logical name | hcloud name | Public IPv4 | WG Address |
|--------------|-------------|-------------|------------|
| ree-cloud-1 | ree-worker-1 | 91.98.130.117 | 10.8.0.1 (hub) |
| ree-cloud-2 | ree-worker-2 | 116.203.216.181 | 10.8.0.12 |
| ree-cloud-3 | ree-worker-3 | 46.62.170.133 | 10.8.0.13 |
| ree-cloud-4 | ree-worker-4 | 91.99.68.94 | 10.8.0.14 |
| Mac | DLAPTOP-4.local | (home IP) | 10.8.0.10 |

**SSH from the Mac:** use **public IPv4** for cloud-2/3/4. The Mac WG tunnel
only routes `10.8.0.1/32` (hub), not worker /32s. `coordinator.env` should set
`SHADOW_SSH_HOST_ree-cloud-*` to the public IPs (see `coordinator.env.example`).

**SSH from ree-cloud-1 (hub):** can use worker WG IPs (`10.8.0.12` etc.) once
peers are registered, but hub->worker SSH may still need the worker's Mac SSH
key in `~ree/.ssh/authorized_keys` on that box.

---

## Hub prerequisites (ree-cloud-1, one-time)

Done if `curl http://10.8.0.1:8787/health` returns `{"ok": true, "mode": "coordinator"}` (Phase 2).

- [ ] WireGuard hub up (`10.8.0.1`, UDP 51820 open)
- [ ] `ree-coordinator` + `ree-sync-daemon` enabled (`/etc/ree-coordinator.env`)
- [ ] `coordinator.db` seeded (`python3 seed_from_queue.py`)
- [ ] `tokens.json` has one bearer token per worker (`deploy/gen_token.py`)
- [ ] After adding a token: `sudo systemctl restart ree-coordinator`
- [ ] sync_daemon on **origin/main** queue read (fix landed 2026-05-19;
      restart `ree-sync-daemon` after `git pull` on ree-v3)

Verify from Mac (WG up):

```bash
curl -s http://10.8.0.1:8787/health
/opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
  --url http://10.8.0.1:8787 --token '<any-worker-token>'
```

---

## Per-worker checklist (repeat for cloud-2, 3, 4)

### A. WireGuard spoke

On the **worker** (as `ree`, needs **passwordless sudo** on that box):

```bash
sudo apt-get install -y wireguard
wg genkey | tee /tmp/wg.key | wg pubkey > /tmp/wg.pub
# Build /etc/wireguard/wg0.conf -- see deploy/wg0.peer.conf.example
# Address = 10.8.0.12|13|14/32  (one per host)
# Peer PublicKey = hub pubkey (on hub: sudo wg show wg0 | head -1)
# Endpoint = 91.98.130.117:51820
# AllowedIPs = 10.8.0.1/32
# PersistentKeepalive = 25
sudo chmod 600 /etc/wireguard/wg0.conf
sudo systemctl enable --now wg-quick@wg0
ping -c1 10.8.0.1
curl -s http://10.8.0.1:8787/health
```

On **ree-cloud-1** (hub), add peer **without restarting wg** (keeps Mac tunnel up):

```bash
sudo wg set wg0 peer <WORKER_PUBKEY> allowed-ips 10.8.0.1X/32
sudo wg show wg0   # confirm latest handshake
```

**ree-cloud-3 blocker (2026-05-20):** `ree@46.62.170.133` does **not** have
passwordless `sudo` (unlike cloud-2/4). WG install requires a one-time root
session (Hetzner web console or `sudo visudo` to grant NOPASSWD for `ree`).
Token for cloud-3 was already minted on the hub; finish WG + shadow.conf after
sudo is fixed.

### B. ree-v3 code at origin/main

```bash
cd ~/REE_Working/ree-v3
git fetch origin main && git checkout main && git pull --ff-only origin main
test -f coordinator_client.py && echo OK
python3 validate_queue.py    # must pass before runner starts
```

If the queue references a script not yet on `origin/main` (e.g. V3-EXQ-543k),
either push the script to `main` or copy it to
`experiments/` on the worker until pushed.

### C. Coordinator bearer token

On **ree-cloud-1** only:

```bash
cd ~/REE_Working/ree-v3/coordinator/deploy
python3 gen_token.py ree-cloud-N    # N = 2, 3, or 4
sudo systemctl restart ree-coordinator
```

Save the printed `COORDINATOR_TOKEN` (shown once).

### D. systemd drop-in (coordinator mode -- Phase 2)

File name remains `shadow.conf`; value is `COORDINATION_MODE=coordinator`.

```bash
sudo mkdir -p /etc/systemd/system/ree-runner.service.d
sudo tee /etc/systemd/system/ree-runner.service.d/shadow.conf <<'EOF'
[Service]
Environment=COORDINATION_MODE=coordinator
Environment=COORDINATOR_URL=http://10.8.0.1:8787
Environment=COORDINATOR_TOKEN=<paste token>
Environment=COORDINATOR_LOG=/home/ree/coordinator_shadow.log
EOF
sudo systemctl daemon-reload
sudo systemctl reset-failed ree-runner
sudo systemctl start ree-runner
systemctl is-active ree-runner
```

Phase 2: coordinator owns claims; git remains result/status transport until Phase 3.

### D-hub. Hub runner only (ree-cloud-1 / `ree-worker-1`, Phase 3)

The hub shares its `REE_assembly` checkout with `ree-sync-daemon`. **Fleet
telemetry on GitHub (`runner_heartbeats/`, `runner_status/`) is owned by
`sync_daemon.phase3_heartbeat_writer`** -- it materialises files from the
coordinator DB (fed by every worker's `POST /heartbeat`). The hub runner
must not write those paths locally.

Template: `shadow.conf.hub.example` in this directory.

Append to the same `shadow.conf` as section D (do **not** copy to cloud-2/3/4):

```bash
# Hub co-tenancy: skip local heartbeat/commands files; coordinator POST +
# sync_daemon.phase3_heartbeat_writer materialise git copies.
Environment="PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1"
```

Also require the three push gates (usually already present after Phase 3
cutover):

```bash
Environment="PHASE3_DISABLE_RUNNER_RESULT_PUSH=1"
Environment="PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1"
Environment="PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH=1"
```

`ree-runner.service` must pass `--machine ree-cloud-1` (hostname is
`ree-worker-1`). After cutover the hub runner was `systemctl disable`-d until
this gate landed; re-enable with `sudo systemctl enable --now ree-runner`.

Expect journal: `phase3 gate active: heartbeat + commands FILE WRITES will be
skipped`. If telemetry-only dirt blocks writers, `sync_daemon` auto-reverts
`runner_heartbeats/` + `runner_status/` when no other paths are dirty
(2026-06-01). Do not remove `_HEARTBEAT_WRITE` to "fix" stale explorer data.

### E. Post-start verification

```bash
tail -5 ~/coordinator_shadow.log    # should NOT be all POST failures
curl -s http://10.8.0.1:8787/health
```

From Mac explorer or API:

```bash
curl -s http://127.0.0.1:8000/api/shadow/status | python3 -m json.tool
```

Expect the worker under `machines[]` with `last_seen` within ~10 minutes,
`state` running/starting, and fleet `divergences: 0` (net of explained E1 rows
in `../SOAK_LOG.md`).

---

## Mac explorer setup

```bash
cd ~/REE_Working/REE_assembly
cp coordinator.env.example coordinator.env   # if missing
# COORDINATOR_URL=http://10.8.0.1:8787
# COORDINATOR_LOCAL_TOKEN=<Mac token from gen_token.py DLAPTOP-4.local>
# SHADOW_SSH_HOST_ree-cloud-2/3/4 = public IPs (not 10.8.0.12-14 from Mac)
# restart serve.py
```

Panel: bottom-right **Shadow Coordination** -> Start (confirm) -> auto-refresh
verdict. Cloud-2/3/4 restarts use SSH to **public** IPs.

---

## Fleet integrity (optional cron on Mac)

```bash
cd ~/REE_Working/ree-v3/coordinator
/opt/local/bin/python3 deploy/fleet_integrity_check.py --write-baseline   # once
/opt/local/bin/python3 deploy/fleet_integrity_check.py                     # probe
```

Uses `coordinator.env` SSH hosts. See `deploy/README.md` for exit codes.
Does not replace `check_shadow.py`; run both on different schedules.

---

## Daily soak (Phase 1 gate -- historical)

Use **Phase-2 soak checklist** above for current operations. Phase 1 gate
(before claim cutover): all hosts in **shadow**, `adjusted_divergences` ~0 for
**days**, then drain and flip per `README.md` section 2.

```bash
/opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
  --url http://10.8.0.1:8787 --token '<token>'
```

| Exit | Meaning |
|------|---------|
| 0 | HEALTHY -- keep soaking |
| 1 | DIVERGENCE -- investigate; do not advance phase |
| 2 | NO SIGNAL -- runners drained or not reporting |
| 3 | UNREACHABLE -- fix WG / coordinator / token |

---

## Rollback (one worker)

```bash
sudo rm /etc/systemd/system/ree-runner.service.d/shadow.conf
sudo systemctl daemon-reload
sudo systemctl restart ree-runner
```

Instant return to git-only coordination.

---

## Common failures

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| Runner `failed` at preflight | `validate_queue.py` or missing script | `git pull`; push missing experiment script to `main` |
| Runner exits after start | `stop` in `runner_commands/<machine>.json` | Clear command or `systemctl start` after drain done |
| Panel HEALTHY but cloud-N stale | Not in shadow or runner down | Drop-in + restart |
| Divergence `state-reconcile` git=pending coord=claimed | Non-shadow machine claimed in git only | Flip that host to shadow (E1) or ignore per SOAK_LOG |
| Mac SSH to 10.8.0.12 times out | Spoke topology | Use public IP in `coordinator.env` |
| cloud-3 `sudo: password required` | No NOPASSWD for ree | Fix sudoers once, then run section A |
