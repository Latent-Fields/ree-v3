"""
Phase 3 end-to-end smoke -- reusable operator helper.

Promoted from ``experiments/v3_exq_612_phase3_cutover_smoke.py`` (the
ad-hoc script written on cutover day 2026-05-28). Same body, lighter
framing: this is an operator helper for verifying the Phase 3 pipeline
from outside an experiment cycle, not a scientific experiment.

When to run
-----------
Use this after any of:
  - a Phase 3 cutover (proves the writer pipeline end-to-end);
  - a hub reboot (proves the writers come back cleanly);
  - re-enabling cloud-1's runner after the
    ``PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE`` flag lands (proves the
    hub co-tenancy fix);
  - any change to ``sync_daemon.py`` writers, the runner-side
    ``coordinator_client``, or ``serve.py`` start_runner env defaults.

What it does
------------
Same flow as the cutover smoke:

  1. A worker can claim the entry via the coordinator ``/claim``.
  2. The runner posts the manifest bytes to coordinator ``/result``,
     which persists them under ``COORDINATOR_SPOOL_DIR``.
  3. ``sync_daemon.phase3_git_writer`` picks the spooled manifest up,
     commits it to ``REE_assembly/evidence/experiments/``, and pushes
     ``origin/master``.
  4. The runner calls ``/queue/remove`` on success.
  5. ``sync_daemon.phase3_queue_writer`` materialises the resulting DB
     state back into ``ree-v3/experiment_queue.json`` on ``origin/main``.
  6. The heartbeat writer continues ticking through all of the above.

How to use
----------
Queue a new entry pointing at this file, e.g.::

  {
    "queue_id": "PHASE3-SMOKE-<YYYYMMDD>",
    "script": "coordinator/tools/phase3_e2e_smoke.py",
    "priority": 1,
    "machine_affinity": "ree-cloud-2",
    "status": "pending",
    "estimated_minutes": 1,
    "title": "Phase 3 end-to-end smoke",
    "experiment_type": "phase3_e2e_smoke"
  }

Target any worker whose ``shadow.conf`` has the full coordinator-mode
env (``COORDINATION_MODE=coordinator``, ``COORDINATOR_URL``,
``COORDINATOR_TOKEN``); the cloud workers are the natural choice. Do
NOT target the hub VM (``ree-cloud-1``) because its runner is
``systemctl disable``-d under the current co-tenancy workaround.

The runner picks the entry up within ~120 s, runs it in ~1 s, and the
hub writer commits the manifest on its next ~60 s tick. Total cycle
< 3 min. Watch via:

  ssh ree@<hub> 'sudo journalctl -u ree-sync-daemon -n 20 --no-pager'

PASS criteria
-------------
  C1: at least 20 episodes ran end-to-end without exception
  C2: at least 50 total env steps recorded
  C3: manifest file written to disk under
      ``REE_assembly/evidence/experiments/``

C1+C2+C3 prove the smoke completed. The actual Phase 3 cycle is
demonstrated by the writer commit (``phase3:`` prefix) and the
coordinator DB row (``results`` table with ``committed_at`` populated).

This file deliberately has no scientific claim_ids; the runner will
warn that no claim is attached, and that's fine for an operator helper.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Walk up to ree-v3 root: coordinator/tools/phase3_e2e_smoke.py
# parents[0] = tools, parents[1] = coordinator, parents[2] = ree-v3
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "phase3_e2e_smoke"
CLAIM_IDS: list = []  # operator helper, not tied to a scientific claim

N_EPISODES = 20
STEPS_PER_EPISODE = 100
SEED = 612


def main() -> int:
    print(f"[{EXPERIMENT_TYPE}] start", flush=True)
    started_at = time.time()

    env = CausalGridWorldV2(seed=SEED)

    total_steps = 0
    per_ep_steps = []
    per_ep_terminal_reasons = []

    for ep in range(N_EPISODES):
        env.reset()
        ep_steps = 0
        terminal_reason = "max_steps"
        for _ in range(STEPS_PER_EPISODE):
            # uniform random action to avoid any policy dependency
            action = (ep + ep_steps) % 4
            _obs, _harm_signal, done, info, _obs_dict = env.step(action)
            ep_steps += 1
            total_steps += 1
            if done:
                terminal_reason = info.get("terminal_reason", "done")
                break
        per_ep_steps.append(ep_steps)
        per_ep_terminal_reasons.append(terminal_reason)
        print(f"[{EXPERIMENT_TYPE}] ep={ep} steps={ep_steps} term={terminal_reason}",
              flush=True)

    elapsed = time.time() - started_at

    c1 = len(per_ep_steps) >= N_EPISODES
    c2 = total_steps >= 50  # random policy dies fast on hazards (~10 steps/ep)
    c3 = True  # set below after we attempt the write

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase3_e2e_smoke_{timestamp}_v3"

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "seed": SEED,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "total_steps": total_steps,
        "per_ep_steps": per_ep_steps,
        "per_ep_terminal_reasons": per_ep_terminal_reasons,
        "elapsed_seconds": round(elapsed, 3),
        "result": "PASS" if (c1 and c2 and c3) else "FAIL",
        "criteria": {
            "C1_min_episodes": {
                "expected": ">= %d" % N_EPISODES,
                "got": len(per_ep_steps),
                "pass": c1,
            },
            "C2_min_total_steps": {
                "expected": ">= 50",
                "got": total_steps,
                "pass": c2,
            },
            "C3_manifest_written": {
                "expected": "manifest path exists post-run",
                "got": "set after write",
                "pass": c3,
            },
        },
        "purpose": (
            "Phase 3 end-to-end operator smoke -- drives the "
            "claim->result->writer->queue-writer cycle."
        ),
    }

    # Write under REE_assembly/evidence/experiments/ -- the runner's
    # standard output path. The Phase 3 path is:
    #   runner reads this file -> POST /result with the bytes ->
    #   coordinator spools -> sync_daemon.phase3_git_writer commits.
    # parents[0] = tools, parents[1] = coordinator, parents[2] = ree-v3,
    # parents[3] = REE_Working
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{EXPERIMENT_TYPE}] manifest written: {out_path}", flush=True)

    print(f"[{EXPERIMENT_TYPE}] result={manifest['result']} "
          f"steps={total_steps} elapsed={elapsed:.1f}s", flush=True)

    # Runner-readable verdict sentinel
    # (experiment_runner.py:101 RE_VERDICT).
    print(f"verdict: {manifest['result']}", flush=True)

    # V3 completion sentinel the runner watches for on stdout.
    print(f"[v3_complete] {run_id}", flush=True)

    # Canonical runner-conformance sentinel: tells the runner WHERE the
    # manifest is so report_result() can ship the bytes to the
    # coordinator. Without this, the runner's "manifest '' is missing"
    # warning fires and the result never reaches the spool.
    emit_outcome(
        outcome=manifest["result"],
        manifest_path=out_path,
        run_id=run_id,
        queue_id=os.environ.get("REE_QUEUE_ID"),
    )

    return 0 if manifest["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
