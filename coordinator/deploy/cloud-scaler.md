# cloud-scaler (hub-resident systemd timer)

This replaces `ree-v3/.github/workflows/cloud-scaler.yml` with a systemd
timer + oneshot service running on the coordinator hub VM
(`ree-worker-1` / 10.8.0.1 / 91.98.130.117).

## Why

The GHA workflow was scheduled for every 15 minutes via
`cron: '*/15 * * * *'`. Actual runs landed at gaps of 60-273 minutes due
to GitHub's best-effort SLA on scheduled workflows under load. Idle
cloud workers were staying alive for 1-4 hours past `IDLE_GRACE_MIN=20`,
burning Hetzner credits. On 2026-05-31 cloud-2 sat idle 09:31Z to
~13:48Z (4+ hours) before being manually `hcloud server shutdown`.

On the hub we own the scheduler, so the cadence is reliable. Timer fires
every 5 minutes on the wall clock.

## Files

- `cloud-scaler.py` -- the Python script. Mirrors the bash decision matrix
  from the GHA workflow 1:1; preserves the HUB_NAME guard, HELD_BY_SELF
  veto, surge sister-state pre-check, and the >= 35-minute HEARTBEAT_FRESH_MIN
  floor.
- `cloud-scaler.service` -- oneshot systemd unit; `Type=oneshot`, runs
  `python3 cloud-scaler.py` once and exits. Reads
  `/etc/ree-coordinator.env` for `HCLOUD_TOKEN`.
- `cloud-scaler.timer` -- `OnCalendar=*:0/5` (every 5 minutes). Persistent.

## Install (on the hub)

```bash
ssh ree@91.98.130.117
cd ~/REE_Working/ree-v3
git pull origin main

sudo cp coordinator/deploy/cloud-scaler.service /etc/systemd/system/
sudo cp coordinator/deploy/cloud-scaler.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cloud-scaler.timer
```

The service unit runs as user `ree`, `WorkingDirectory=/home/ree/REE_Working/ree-v3/coordinator/deploy`,
with `EnvironmentFile=/etc/ree-coordinator.env` (already provisioned for
`ree-sync-daemon` -- contains `HCLOUD_TOKEN`).

## Verify

```bash
# Timer is armed
systemctl list-timers cloud-scaler.timer

# Wait ~5 min, then inspect the most recent tick
journalctl -u cloud-scaler.service --since "10 minutes ago"

# Should show output like:
# [2026-05-31T12:35:00Z] cloud-scaler tick: queue=... heartbeats=... ...
# [2026-05-31T12:35:00Z] [ree-worker-1 affinity=ree-cloud-1] HUB_NAME match -- scaler skips hub VM ...
# [2026-05-31T12:35:01Z] [ree-worker-2 affinity=ree-cloud-2] claimable=0 held_by_self=1 status=running idle_ok=0 reason=state_running_current_V3-EXQ-XXX
# [2026-05-31T12:35:01Z]   -> worker holds 1 active claim(s), keeping ree-worker-2 running
# ...
```

## Dry-run (no hcloud calls fire)

```bash
sudo -u ree /usr/bin/python3 \
    /home/ree/REE_Working/ree-v3/coordinator/deploy/cloud-scaler.py \
    --dry-run
```

Reads state and logs the decisions it would make, but does NOT call
`hcloud server poweron`, `hcloud server shutdown`, or the
`coordinator_announce_shutdown.sh` announce script. Use this to sanity-
check decisions before enabling the timer, or to triage a misfire after
the fact (re-run with `--dry-run` and compare to the live tick's
journal).

## Disable the GHA workflow

After the hub timer has been running for ~2 hours and the journal looks
clean, downgrade `ree-v3/.github/workflows/cloud-scaler.yml` to a
backstop schedule (or delete it). The hub timer is authoritative;
keeping GHA on the original `*/15` schedule double-fires shutdown
decisions and competes with the hub for the hcloud API.

Recommended backstop: change the cron line in the YAML to
`cron: '0 */6 * * *'` (every 6 hours) so it can still recover a wedged
hub by powering off forgotten workers, but it does not race with the
primary timer.

## Operator gotchas

- **Never run a runner on the hub VM.** This was the 2026-05-28 incident
  the HUB_NAME guard exists to prevent. `cloud-1`'s runner is
  `systemctl disable`-d for this reason. The scaler must never power
  off `ree-worker-1` regardless of what its queue / heartbeat state
  says, because doing so would take the coordinator + sync_daemon
  offline. The HUB_NAME guard in `cloud-scaler.py` enforces this.

- **HELD_BY_SELF veto is load-bearing.** If you ever see a worker mid-
  experiment getting shut down by the scaler, the HELD_BY_SELF check
  has regressed. The 2026-05-30 fleet incident (V3-EXQ-483e on cloud-2,
  V3-EXQ-614a on cloud-3 powered off mid-run) is what motivates it.
  The Python guard mirrors the bash logic exactly: any item with
  `status==claimed` and `claimed_by.machine == affinity` keeps the
  worker alive regardless of heartbeat staleness.

- **HEARTBEAT_FRESH_MIN must stay >= 35.** The phase3 sync_daemon
  heartbeat-writer refreshes the file on state-changes only with a
  5-min debounce and a 30-min liveness floor; an idle-but-alive worker
  legitimately lags 5-30 minutes behind reality. The Python clamps
  values below 35 with a WARN log line; setting it lower regresses to
  the heartbeat_aging idle-stall bug.

- **Surge ordering matters.** `WORKERS` lists cloud-2 and cloud-3
  before cloud-4 specifically so cloud-4's surge decision can read
  sister-worker state populated by the earlier iterations. Don't
  reorder the list.

## Override env vars (rarely needed)

The Python honours the same env vars the GHA workflow used:
`IDLE_GRACE_MIN`, `HEARTBEAT_FRESH_MIN`, `SURGE_QUEUE_THRESHOLD`, `HUB_NAME`.
To experiment, set them in a drop-in:

```bash
sudo systemctl edit cloud-scaler.service
# Add:
# [Service]
# Environment="IDLE_GRACE_MIN=30"
sudo systemctl daemon-reload
```

`HEARTBEAT_FRESH_MIN` is clamped to a 35-minute floor inside the script;
attempts to set it lower log a WARN and are silently raised.
