# REE phone-dispatch — start chip-spawned Claude sessions from your iPhone

A small, **isolated** job queue so you can launch self-contained Claude Code
sessions (e.g. mirrored `spawn_task` chips) from your phone. It is deliberately
**separate from the experiment coordinator** — its own port, sqlite db, and
token — so it can never destabilise the coordinator (the sole git writer).

```
  iPhone  ──WireGuard──▶  dispatch_service.py  (HUB ree-cloud-1, always-on)
   browser                   │  durable sqlite queue (staged/pending/…)
                             ▼
  Mac     ──WireGuard poll─▶ GET /api/pending ─▶ claim ─▶ `claude -p` in a
   (executor)                                              fresh git worktree
   dispatch_executor.py      POST /api/update ─▶ ntfy push ─▶ 📲 your phone
```

Job lifecycle: `staged` (a chip suggestion awaiting your tap) → `pending`
(you tapped **Launch**) → `claimed` → `running` → `done`/`failed` (or
`cancelled`). The Mac only runs jobs while awake; jobs wait safely in the hub
queue otherwise.

---

## Components

| File | Runs on | Role |
|---|---|---|
| `dispatch_service.py` | hub (ree-cloud-1) | queue + mobile page + API |
| `dispatch_executor.py` | Mac | polls, runs `claude -p` in a worktree |
| `enqueue.py` | anywhere with the token | mirror a chip into the queue |
| `deploy/ree-dispatch.service` | hub | systemd unit |
| `deploy/com.ree.dispatch-executor.plist` | Mac | launchd agent |
| `deploy/gen_token.py` | hub | make a bearer token |

---

## Setup

### 1. Token (on the hub)
```
cd ~/REE_Working/ree-v3/dispatch
python3 deploy/gen_token.py phone     # prints a token; writes dispatch_tokens.json (chmod 600)
```
Keep the token; you'll paste it into the phone page once and into the Mac
executor config.

### 2. Put your iPhone on WireGuard
The fleet uses WireGuard with the hub at `10.8.0.1`. Machine peers use
`.10–.15`; assign the phone a free address, e.g. **`10.8.0.20`**.

- **Hub** `/etc/wireguard/wg0.conf` — add a peer (use the phone's public key from
  the WireGuard iOS app):
  ```
  [Peer]
  # iPhone
  PublicKey = <PHONE_WG_PUBKEY>
  AllowedIPs = 10.8.0.20/32
  ```
  then `sudo wg syncconf wg0 <(wg-quick strip wg0)` (or restart wg-quick@wg0).
- **iPhone** (WireGuard app → add tunnel):
  ```
  [Interface]
  PrivateKey = <generated on the phone>
  Address = 10.8.0.20/32

  [Peer]
  PublicKey = <HUB_WG_PUBKEY>
  Endpoint = <REE_CLOUD_1_PUBLIC_IP>:51820
  AllowedIPs = 10.8.0.1/32
  PersistentKeepalive = 25
  ```
  Only `10.8.0.1/32` is routed, so this doesn't disturb the phone's normal
  networking. (Template: `wg0.peer.conf.example`.)

### 3. Hub service (systemd)
```
sudo cp deploy/ree-dispatch.service /etc/systemd/system/
# edit User/paths and (optionally) DISPATCH_NTFY_TOPIC in the unit
sudo systemctl daemon-reload && sudo systemctl enable --now ree-dispatch
curl -s http://10.8.0.1:8799/health        # {"ok": true, ...}
```
Binds to `10.8.0.1` only — reachable over WireGuard, nowhere else.

### 4. Mac executor (launchd)
```
cp deploy/com.ree.dispatch-executor.plist ~/Library/LaunchAgents/
# edit DISPATCH_TOKEN (+ DISPATCH_CLAUDE_FLAGS, see below) in the plist
launchctl load ~/Library/LaunchAgents/com.ree.dispatch-executor.plist
tail -f /tmp/ree-dispatch-executor.out
```
The Mac must have an **authenticated `claude` CLI** (you already do — it uses
your logged-in subscription; no API key needed). Ensure `claude` is on the
`PATH` set in the plist.

### 5. Phone push (optional, recommended)
Install the **ntfy** app (iOS), subscribe to a long random topic, and set the
same value in `DISPATCH_NTFY_TOPIC` on the hub unit. You'll get a push when a
job finishes. (Anyone who knows the topic can read your notifications — treat it
as a secret.)

---

## Using it

- On the iPhone (WireGuard on), open **`http://10.8.0.1:8799/`**. Tap **Token**,
  paste the token once (stored in the browser). Add it to your Home Screen for a
  one-tap "app".
- **Enqueue + Launch** to run now, or **Stage only** to queue a suggestion you
  Launch later. Watch status live; get a push on completion.

### Mirroring a `spawn_task` chip into the queue
A chip's prompt is already self-contained — pipe it straight in:
```
DISPATCH_URL=http://10.8.0.1:8799 DISPATCH_TOKEN=... \
  python3 enqueue.py --title "Queue sleep GAP-3b run" \
    --cwd /Users/dgolden/REE_Working/REE_assembly --stdin <<'EOF'
In REE_assembly, via the /queue-experiment skill, author + queue the behavioural
promotion experiment for sleep_substrate:GAP-3b (MECH-285/272/275/273) ...
EOF
```
It lands as `staged`; Launch it from the phone when you want it to run. (You can
wire this into a hook so every chip you spawn is mirrored automatically.)

---

## Claude permissions (read before relying on autonomy)

A bare `claude -p` may **stall on a permission prompt** in headless mode. Set
`DISPATCH_CLAUDE_FLAGS` in the executor plist to the autonomy you want, e.g.:

- `--permission-mode acceptEdits` — auto-accept edits, still gated on risky ops.
- `--allowedTools "Edit Bash(git*)"` — allowlist specific tools.
- `--dangerously-skip-permissions` — fully autonomous. **Highest risk**; only for
  prompts you trust, in the isolated worktree. Your call.

Each job runs in a throwaway `dispatch/<id>` branch + worktree under
`.dispatch-worktrees/`, so a session can't touch your live working tree. Nothing
is auto-merged — review the branch, then merge/discard.

---

## Security model

- **Auth:** every API call needs the bearer token; the page shell is the only
  unauthenticated route (it carries no data). Same trust level as the
  coordinator token.
- **Network:** bind on the WireGuard IP only. Do **not** expose `8799` publicly.
- **Secrets:** `dispatch_tokens.json`, `dispatch.db`, logs, and worktrees are
  git-ignored (`.gitignore`).
- **Blast radius:** standalone process; a crash restarts via systemd and never
  touches the coordinator or the git writers.

---

## Test / troubleshoot

```
python3 test_dispatch.py        # 11 tests: API state machine, auth, e2e w/ fake claude
```
- `401` on the phone → tap **Token**, re-paste.
- Jobs stuck `pending` → Mac asleep, executor down, or `claude` not on the
  plist `PATH`. Check `/tmp/ree-dispatch-executor.{out,err}`.
- `failed` immediately → usually a permission stall or bad `cwd`; read the job's
  summary + `dispatch-logs/dispatch-<id>.log` on the Mac.
