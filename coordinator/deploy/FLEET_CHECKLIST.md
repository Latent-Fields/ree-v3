# Phase-1 shadow fleet checklist

Cross-session operator sheet for bringing **ree-cloud-2 / 3 / 4** into the
shadow soak. Hub = **ree-cloud-1** (`ree-worker-1`, public `91.98.130.117`,
WireGuard `10.8.0.1`). Plan-of-record: `../PLAN.md`, runbook: `README.md`.

**2026-05-20 status (this session):**

| Host | Public SSH | WG IP | shadow.conf | Runner | Coordinator HB |
|------|------------|-------|-------------|--------|----------------|
| Mac | local | 10.8.0.10 | via explorer | running | fresh |
| ree-cloud-1 | 91.98.130.117 | 10.8.0.1 | n/a (hub) | active | n/a |
| ree-cloud-2 | 116.203.216.181 | 10.8.0.12 | yes | **active** (restarted) | fresh |
| ree-cloud-3 | 46.62.170.133 | 10.8.0.13 | **BLOCKED** | inactive | not on mesh |
| ree-cloud-4 | 91.99.68.94 | 10.8.0.14 | yes | **active** (restarted) | fresh |

`GET /shadow/status` on the hub: **HEALTHY**, divergences **0**, machines
Mac + ree-cloud-2 + ree-cloud-4 (cloud-3 pending WG).

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

Done if `curl http://10.8.0.1:8787/health` returns `{"ok": true, "mode": "shadow"}`.

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

### D. systemd drop-in (shadow mode)

```bash
sudo mkdir -p /etc/systemd/system/ree-runner.service.d
sudo tee /etc/systemd/system/ree-runner.service.d/shadow.conf <<'EOF'
[Service]
Environment=COORDINATION_MODE=shadow
Environment=COORDINATOR_URL=http://10.8.0.1:8787
Environment=COORDINATOR_TOKEN=<paste token>
Environment=COORDINATOR_LOG=/home/ree/coordinator_shadow.log
EOF
sudo systemctl daemon-reload
sudo systemctl reset-failed ree-runner
sudo systemctl start ree-runner
systemctl is-active ree-runner
```

Git claiming is **unchanged**; shadow only adds best-effort coordinator reports.

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

## Daily soak (go/no-go for Phase 2)

```bash
/opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
  --url http://10.8.0.1:8787 --token '<token>'
```

| Exit | Meaning |
|------|---------|
| 0 | HEALTHY -- proceed only after **days** of this under real load |
| 1 | DIVERGENCE -- read rows; do not cut over |
| 2 | NO SIGNAL -- runners drained or not in shadow mode |
| 3 | UNREACHABLE -- fix WG / coordinator / token |

Before Phase 2: all experiment hosts in **shadow**, `adjusted_divergences` ~0
(see `../SOAK_LOG.md` E1 class for git-only stragglers), drain fleet, flip to
`COORDINATION_MODE=coordinator` per `README.md` section 2.

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
