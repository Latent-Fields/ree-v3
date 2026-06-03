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
> 1. **Running a runner on the hub VM (`ree-cloud-1`) is supported via an
>    ISOLATED checkout (2026-06-02).** The old rule ("never run a runner on
>    the hub") was about the runner sharing the sync_daemon writers'
>    `~/REE_Working` checkout: its result-manifest + queue writes dirty the
>    tree and the Phase-3 writers refuse to commit (recurring fleet-wedge,
>    see `reference_hub_writer_wedge.md`). FIX: run the hub runner from a
>    SECOND checkout, `~/REE_Working_runner/{ree-v3,REE_assembly}`, via a
>    `WorkingDirectory=/home/ree/REE_Working_runner/ree-v3` drop-in in the
>    hub `shadow.conf`. All its dirty writes then land in its own tree; the
>    writers' `~/REE_Working` stays clean. Use the standard WORKER gates
>    (telemetry-off-git + `PHASE3_COMMANDS_VIA_COORDINATOR`) -- do NOT set
>    `PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE` (unnecessary with isolation;
>    it also gates command-file writeback). Template:
>    `coordinator/deploy/shadow.conf.hub.example`. VERIFY after any change
>    that `git -C ~/REE_Working/ree-v3 status` and
>    `git -C ~/REE_Working/REE_assembly status` stay clean while the runner
>    works.
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

## Coordination panel (Explorer)

The bottom-right card in the explorer was renamed from "Shadow Coordination"
to "Coordination" on 2026-05-29 (REE_assembly commit `096e01f6`) and gained a
writer-health subsection. Under Phase 3, this is the place to watch the three
`sync_daemon` writers that are now the sole producers of coordination data on
git (`phase3_git_writer` for results, `phase3_queue_writer` for the queue
snapshot, `phase3_heartbeat_writer` for heartbeats + status -- see
`PHASE3_CUTOVER.md` Status section).

Backed by `GET /api/coordinator/phase3/writers` in `REE_assembly/serve.py`
(`run_phase3_writers_summary`). The panel polls every **15s**; the endpoint
caches **60s** and re-runs a single SSH to the hub. Apparent age in the rows
can therefore lag the hub by up to ~75s -- a row that just turned yellow may
already be green on the hub.

### What each row shows

Each of the three writer rows reports:

| Field | Meaning |
|-------|---------|
| `sha10` | First 10 chars of the most recent writer commit on the relevant branch (`origin/master` for git_writer + heartbeat_writer; `origin/main` for queue_writer). Searched by commit-message prefix: `phase3:`, `phase3-queue:`, `phase3-heartbeats:`. |
| `committed_at` / `age_s` | Author timestamp of that commit; age in seconds since now. |
| `color` | **green** = age < 5 min (writer ticked recently). **yellow** = 5-15 min (worth a glance; idle queue is fine here, but heartbeat_writer should never be this old). **red** = > 15 min (something is wrong; see trouble-tree below). |
| `status` | Derived from the last few lines of `sudo journalctl -u ree-sync-daemon -n 3` on the hub. One of: `idle` (nothing to commit this tick), `committing` (mid-tick), `push-rejected` (non-FF push to origin; writer is wedged), `refusing` (clean-tree check tripped -- dirty REE_assembly working tree on the hub), `rebase-conflict` (the autostash failure mode -- there should be no autostash under Phase 3, but operator-side rebases can still leave the tree wedged). |

Outside the three rows:

- `spool_pending` -- count of files in `/home/ree/coordinator-spool/pending/`
  on the hub. **0 or 1 in steady state.** Persistently > 5 means the writer
  is not draining the spool (paired with a stale `git_writer` row this is
  the classic push-rejected loop).
- `journal_tail` -- the last few `sync_daemon` log lines, surfaced in the
  panel's tooltip so you do not need to SSH for the common diagnosis.
- `hub_reachable` -- `false` plus an `error` string when the SSH to
  `ree@91.98.130.117` (over WireGuard 10.8.0.1) failed. The panel renders
  all three rows grey in this case; no writer data is available.

### What to do when a row goes red

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `git_writer` or `queue_writer` SHA stale (red), spool_pending climbing | Writer wedged in **push-rejected loop** -- operator-side IGW commits landed between writer ticks, so the hub's local writer commits no longer fast-forward onto origin. The Phase-3 writer deliberately does **not** `git pull --rebase --autostash`, so it just keeps failing the push. NOTE: with `PHASE3_QUEUE_CONFLICT_RECOVERY=1` (see the subsection below) the `queue_writer` now **self-heals** this and the conflict variant within one tick; a stale `queue_writer` with that flag on means something else (network, dirty tree). | All writers already absorb *non-conflicting* operator races by rebasing writer-authored commits onto refreshed origin (`_sync_to_origin`). If a writer is still stale: SSH to the hub and FF-rebase by hand: `git -C ~/REE_Working/REE_assembly pull --rebase origin master` (and `git -C ~/REE_Working/ree-v3 pull --rebase origin main` for `queue_writer`). Writers resume on the next tick (~60s). |
| Persistent `rebase-conflict` status | Autostash-style wedge -- a manual rebase on the hub hit a conflict (e.g. the runner_heartbeats/*.json collision class CLAUDE.md describes). The writer cannot commit until the working tree is clean. | SSH to the hub, **inspect the conflict** -- do NOT blindly `git checkout --ours` or `--theirs`. The heartbeat-file collision is usually safe to resolve in favour of the newer runner-written snapshot (the runner overwrites it on next tick), but other paths (claims.yaml, planning docs, manifests) can hide real edits. Resolve, `git rebase --continue`, then leave the tree clean. |
| `refusing` status with clean `journal_tail` | Hub's `REE_assembly` working tree is dirty -- something (an interactive session, or a hub runner using the SHARED `~/REE_Working` checkout) left files modified. Phase 3 writer refuses rather than autostashing. | SSH to the hub, `git -C ~/REE_Working/REE_assembly status`. Find the source. If a hub runner is the cause, it must run from the ISOLATED `~/REE_Working_runner` checkout (`WorkingDirectory` drop-in), NOT `~/REE_Working` -- see rule 1 above. Recovery: back up + clean the dirty files (`mv` untracked manifests aside, `git checkout`/`git pull --rebase` the queue); writers resume within ~10s. |
| `spool_pending` elevated but writer SHA is recent | Writer is committing but falling behind -- batch size too small for the inbound rate, or the hub is doing one-commit-per-manifest. | Check `journal_tail`; usually self-corrects. If sustained, raise `PHASE3_BATCH_SIZE` on the hub (default 32). |
| `hub_reachable: false` | WireGuard down on the Mac, or the SSH key is no longer accepted by `ree@91.98.130.117`. | `wg show` on the Mac (`wg-quick up wg0` if down); `ssh ree@10.8.0.1 true` to test the key; check `coordinator.env` for the hub host override. |
| `heartbeat_writer` stale AND `spool_pending` climbing AND no manual operator activity | **Cloud-scaler powered off the hub VM.** The scaler treats `ree-worker-1` as a regular worker and can shut it down when idle. This is the 2026-05-28 incident class. | Confirm with `hcloud server list | grep ree-worker-1`; if off, `hcloud server poweron ree-worker-1`. The `cloud-scaler.yml` workflow should already be `disabled_manually` -- if it is not, disable it via `gh workflow disable`. Re-enable only after the hub-protection guard chip lands. |

The Coordination panel is the operational watchpoint for the cloud-scaler-vs-hub
failure mode until that guard chip is in place: the signature is a stale
`heartbeat_writer` row combined with a growing `spool_pending` counter, even
though the panel itself stays reachable from the Mac (the explorer's SSH does
not transit the hub VM).

### Queue-writer conflict-recovery (`PHASE3_QUEUE_CONFLICT_RECOVERY`)

**What it fixes.** The historic push-rejected/rebase-conflict wedge that took the
fleet down: an operator/IGW/session commit edits `ree-v3/experiment_queue.json`
on `origin/main` while the `queue_writer` is holding a retained (push-rejected)
snapshot commit. The writer's next-tick `_sync_to_origin` rebase **conflicts** on
that file, aborts, and refuses -- and stays wedged until a human runs
`git -C ~/REE_Working/ree-v3 pull --rebase origin main` on the hub. (2026-06-02:
this sat wedged ~4.5h and the Mac git-sync-repair routine, which only knew the
heartbeat variant, did not catch it.)

**What the flag does.** When `PHASE3_QUEUE_CONFLICT_RECOVERY=1`, the `queue_writer`
recovers from that conflict **within one tick**: it `git reset --hard origin/main`
(dropping only its own writer-authored snapshot commit -- never operator work, see
safety below) and re-materialises `experiment_queue.json` from the coordinator DB.
This is **lossless** because the queue is DB-authoritative: `reconcile_once` has
already absorbed the operator's file edit into the DB *before* the writer
materialises, so the regenerated snapshot includes it and pushes fast-forward.

**Safety.** The reset only runs after `_check_ahead_writer_authored` confirms every
ahead commit is writer-authored (`phase3-queue: ` prefix). A foreign (operator)
unpushed commit on the hub still **refuses** the tick -- the flag never drops
operator work. The clean-tree precondition still holds (no autostash). The flag is
**scoped to the queue writer only**; the result + heartbeat writers keep the
conservative env-fallback policy (their paths are writer-exclusive and do not
conflict).

**Additive over the legacy env.** The flag is *additive* on top of the older blunt
`PHASE3_AUTO_RESET_ON_REBASE_CONFLICT=1` env (a global stopgap that enables conflict
recovery for *all three* writers and is currently set on the hub via the
`auto-reset.conf` drop-in). When the scoped flag is set, the queue writer forces
recovery on; when it is unset, the queue writer falls back to that env -- so the
scoped flag never *disables* recovery the env already provides. Recovery is only
fully off when **both** are unset (then: refuse-on-conflict, manual
`git pull --rebase` to unwedge). Either path records `n_conflict_recoveries`, so the
canary observable works regardless of which one enabled it. As of 2026-06-03 the hub
has both set: the scoped flag (`queue-conflict-recovery.conf`) is the explicit,
intended control; the legacy env remains as the result/heartbeat-writer fallback.

**Enable / verify / rollback (hub `ree-sync-daemon`):**

```bash
# Enable: add to the unit env (e.g. coordinator.env or an Environment= line in
# the ree-sync-daemon drop-in -- read at process start), then restart.
#   Environment=PHASE3_QUEUE_CONFLICT_RECOVERY=1
sudo systemctl restart ree-sync-daemon

# Verify it fired (the canary observable) -- from the Mac:
curl -s http://10.8.0.1:8787/writer-health | python3 -c \
  'import sys,json; w=json.load(sys.stdin)["writers"]["queue_writer"]; \
   print("recoveries:", w["n_conflict_recoveries"], "at", w["last_conflict_recovery_at"])'

# Rollback: unset the env line (or set =0) and restart -- behaviour returns
# to the pre-flag refuse-on-conflict path exactly.
sudo systemctl restart ree-sync-daemon
```

A rising `queue_writer.n_conflict_recoveries` on `/writer-health` is the proof the
self-heal is doing its job (each increment = one operator-race conflict absorbed
without operator intervention). It is informational, not an alarm.

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
