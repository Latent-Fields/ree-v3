# Shadow coordination -- operator guide (Daniel)

> **Phase 3 LIVE as of 2026-05-29.** The rest of this guide describes the
> Phase-1 shadow workflow and remains accurate for historical / rollback
> contexts, but is no longer the operational doc-of-record. For the live
> coordinator + writer architecture, the cutover history, the known
> follow-ups, and the operator gotchas, read:
> - `PHASE3_CUTOVER.md` Status section (what's true on origin now)
> - `REE_Working/CLAUDE.md` Coordinator (Phase 3 -- live) section
>   (per-session orientation, writer-health verification snippet)
>
> Three rules in force until the follow-up chips land:
> 1. **Never run a runner on the hub VM** (`ree-cloud-1`). Its
>    `ree-runner` is `systemctl disable`-d. Re-enable only after a
>    `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE` flag lands.
> 2. **Watch the cloud-scaler workflow.** It treats the hub like a
>    regular worker and can power it off. Currently
>    `disabled_manually`. Re-enable only after a hub-protection
>    guard lands.
> 3. **The hub repo will sometimes diverge from origin** because the
>    writer doesn't `git pull --rebase` by design. When operator-side
>    commits land between writer ticks, reconcile by hand:
>    `git -C ~/REE_Working/REE_assembly pull --rebase origin master`
>    (and similar for `ree-v3`). Writers resume on the next tick.

Read this instead of scattered runbook fragments. Technical plan:
`PLAN.md`. Machine checklist: `deploy/FLEET_CHECKLIST.md`.

---

## Your mental model (correct)

You expected three simple stages:

1. **Two systems in parallel** -- git keeps doing what it does today; the
   coordinator runs alongside and watches.
2. **Assess** -- the panel shows whether the new system agrees with git.
3. **Retire the old path** -- only after assessment passes, turn off git-as-mutex
   (and later git heartbeats), with explicit instructions.

That **is** the design (Phases 1-3 below). What made it feel hard was:

- Git sync problems were already painful **before** shadow (heartbeats,
  autostash, five machines pushing `REE_assembly`).
- Phase 1 still runs **both** systems; load does not drop until Phase 3.
- One-time **host setup** (WireGuard, tokens, `shadow.conf` per cloud) was
  mixed into "click Start" -- the button only **restarts** services that are
  already configured.
- The explorer button had sharp edges (Mac runner already in git mode, SSH to
  wrong addresses, pending `stop` commands on clouds).

None of that means you operated it wrong.

---

## The three systems (what runs where)

| System | What it does today | When it goes away |
|--------|-------------------|-------------------|
| **A. Git claiming** | `experiment_queue.json` + push race = who runs which EXQ | Phase 2: claims move to coordinator |
| **B. Git heartbeats** | `runner_heartbeats/*.json` pushed to `REE_assembly` every ~60s | Phase 3: sync_daemon writes derived files; heartbeats stop |
| **C. Coordinator** | SQLite mutex + shadow compare + (later) sole writer | Stays; becomes authoritative |

During **Phase 1 shadow**, A and B are still **authoritative**. C only **observes**
and logs disagreements. Experiments must keep working if C is down.

---

## Phases (what "over the line" means)

| Phase | Name | You see | Git claiming | Coordinator | Safe to "shut down old"? |
|-------|------|---------|--------------|-------------|-------------------------|
| **0** | Standup | Hub health OK, panel grey | on | idle | No |
| **1** | **Shadow soak** (you are here) | Panel **HEALTHY**, div **0**, 4 clouds | **on** (authority) | **shadow** (assess) | **No** -- only assess |
| **2** | Claim cutover | Same panel; claims via C | **off** for claims | **coordinator** mode | Claims only -- results still git |
| **3** | Result cutover | Less `REE_assembly` push fighting | queue/results via sync_daemon | sole git writer | **Yes** -- retire heartbeat war |
| **4** | Cleanup | Code delete, optional UI | gone | normal | Yes |

**"Over the line" for your vision = end of Phase 1** (trust the assessor), then
**Phase 2-3** with a **drained, single-authority fleet**, then retire A/B.

---

## Phase 1 -- what you do (daily)

### One-time (already done on Mac + cloud 1-4)

- Hub: `ree-coordinator` + `ree-sync-daemon` on ree-cloud-1.
- Each experiment cloud: WireGuard to hub, `shadow.conf` on `ree-runner`,
  token in coordinator.
- Mac: `REE_assembly/coordinator.env` (URL, Mac token, **public SSH IPs**
  for cloud-2/3/4).

See `deploy/FLEET_CHECKLIST.md` if a new machine is added.

### Starting the parallel soak (explorer)

1. Optional: clear stale **`stop`** commands on clouds you will restart
   (`REE_assembly/evidence/experiments/runner_commands/ree-cloud-*.json`).
2. In the explorer **Shadow Coordination** panel (bottom-right), read the
   **phase line** (should say Phase 1).
3. Click **Start shadow soak** and confirm.
   - Restarts the **Mac** runner in shadow mode (drains first if serve.py
     started it).
   - SSH: hub coordinator + restart cloud runners.
4. Confirm panel shows **four** machines **FRESH** and verdict **HEALTHY**.

### While soak runs (days, not hours)

- **Keep using the normal runner / queue** -- git still owns claims.
- Glance at the panel or run daily:

```bash
grep '^COORDINATOR_LOCAL_TOKEN=' ~/REE_Working/REE_assembly/coordinator.env | cut -d= -f2- | \
  xargs -I{} /opt/local/bin/python3 ~/REE_Working/ree-v3/coordinator/check_shadow.py \
    --url http://10.8.0.1:8787 --token {}
```

| Verdict | Meaning | You |
|---------|---------|-----|
| **HEALTHY** | Assessor matches git; soak working | Keep parallel run |
| **DIVERGENCE** | Real or harness mismatch | Do **not** advance; read rows |
| **NO_SIGNAL** | Nothing reporting | Fix shadow mode / restart runners |
| **NOT_CONFIGURED** | `coordinator.env` | Fill URL + token |
| **UNREACHABLE** | WG or hub down | Fix network / services |

### What you must NOT do in Phase 1

- Do **not** set `COORDINATION_MODE=coordinator` (Phase 2) because the panel
  looks green once.
- Do **not** stop git pushing or disable `--auto-sync` to "help" shadow --
  you would starve the authority git still uses.
- Do **not** treat **DIVERGENCE** as "shut down git" -- fix or explain rows
  first (`SOAK_LOG.md` for known harness classes).

---

## When the superseded system can be shut down (explicit)

### After Phase 1 (assessment passed)

**Gate:** `check_shadow.py` exit **0** on **HEALTHY** with **0 unexplained**
divergences for **several days** with all experiment hosts in **shadow**, real
queue traffic.

**You still do not shut down git claiming.** Instruction:

> Phase 1 complete. Schedule a maintenance window for Phase 2 claim cutover
> (drain fleet, flip workers to `COORDINATION_MODE=coordinator`, no mixed fleet).

### After Phase 2 (claims on coordinator)

**You may shut down:** git-based **claim pushes** to `experiment_queue.json`
for mutex purposes (claims are on C).

**You still keep:** git pushes for **results / manifests** until Phase 3.

### After Phase 3 (sync_daemon sole writer)

**You may shut down:** per-minute **`runner_remote_control` heartbeat
rebase/autostash`** on `REE_assembly` (the silent-revert class).

**You still keep:** git as **evidence archive** -- commits of manifests and
governance; only the *coordination IPC* moved off git.

### After Phase 4

**You may shut down:** dead code paths, repair cron (optional), mixed-mode docs.

---

## Why it felt harder than "doubled services"

1. **Git was doing three jobs** (evidence + mutex + telemetry), not one.
2. **Phase 1 adds load** (extra HTTP reports) before it removes git push fights.
3. **Host provisioning != the button** -- WG/tokens/`shadow.conf` are separate
   from "Start soak".
4. **Mixed fleet** (Mac in git, clouds in shadow) looks like "shadow is broken"
   but is operator/setup ordering.
5. **Incidents were visible** during soak (good) but felt like regressions.

The panel and this guide exist to keep the story at your three-step level.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Start soak, Mac still git-only | Runner was already running | Panel now drains/restarts Mac; or Stop runner then Start soak |
| Cloud SSH FAILED in result | Wrong SSH host from Mac | Public IPs in `coordinator.env` |
| Cloud runner exits right away | Pending `stop` in runner_commands | Clear/supersede command file |
| HEALTHY but only 1-2 machines | Others not in shadow.conf | FLEET_CHECKLIST per host |
| DIVERGENCE state-reconcile | Git-only machine claimed | Flip that host to shadow (E1 in SOAK_LOG) |

---

## File map

| Question | Read |
|----------|------|
| Why does this exist? | `PLAN.md` |
| Per-machine setup | `deploy/FLEET_CHECKLIST.md` |
| Explained false divergences | `SOAK_LOG.md` |
| Deploy hub / WG | `deploy/README.md` |
| This narrative | `OPERATOR_GUIDE.md` (this file) |
