#!/opt/local/bin/python3
"""
V3-EXQ-1026 -- MECH-423 R3 sleep-INTEGRATED E2 (world-forward) consolidation validation.

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a
              dedicated N_CYCLES wake-sleep-test loop)

WHY THIS RUN (the gap it closes)
---------------------------------
MECH-423's R3 CrossModuleConsolidator is already validated (V3-EXQ-680e PASS, cloud
class) -- but ONLY via a DIRECT `CrossModuleConsolidator.consolidate()` call
(680e's own manifest: `sleep_driver_pattern = "N/A ... no sleep cycle"`). The
SEPARATE call site that actually wires this consolidator into the live SD-017
sleep cycle -- `SleepLoopManager._run_cycle` (ree_core/sleep/phase_manager.py
lines ~669-693), reached via `agent.sleep_loop.force_cycle(agent)` /
`REEAgent.run_sleep_cycle()` -- had never been exercised in any LANDED experiment
manifest (confirmed 2026-09-08 by grep over evidence/experiments/*.json: the only
hit for "cross_module_consolidation" predates this run and used the direct-call
driver). The contract suite's C7 (tests/contracts/test_mech423_cross_module_consolidation.py)
exercises the same integrated call site and asserts the merged-metrics readout,
but does not measure an actual E2 weight delta, and a contract pass is not a
landed experiment manifest -- this run supplies both.

Consequence this closes: INV-063 (across-sleep world-forward prediction-error
improvement on a frozen held-out battery) is blocked_substrate because ree_core's
sleep call graph (run_sleep_cycle / run_sws_schema_pass / run_rem_attribution_pass)
contains zero backward()/optimizer/world_forward-training tokens EXCEPT this one
hook -- so until the hook is shown to fire a real E2 update through a real sleep
cycle, INV-063's leg B DV is structurally 0.0 by construction, not by measurement.

WHY THE E2 LEG (not E1)
------------------------
The hook (phase_manager.py ~669-693) always builds module_losses for BOTH "e1"
(agent.compute_prediction_loss) and "e2" (agent.compute_e2_loss) -- that is
existing, unmodified substrate; this run does not choose which legs fire. But the
VALIDATION FOCUS here is E2 specifically: compute_e2_loss (agent.py:12137) reads
only `_e2_transition_buffer` + `e2.predict_next_self`, and
ree_core/predictors/e2_fast.py has zero references to ContextMemory /
context_memory / e1_deep -- so an E2 weight-delta reading is clean of the open
ContextMemory write-path-addressing-degeneracy defect that confounds the E1 leg
(compute_prediction_loss reads ContextMemory via e1_deep.py). C3 below (the
load-bearing criterion) is scoped to agent.e2.parameters() only for this reason.

NUMERICAL STABILITY (680b/680c history -- why it does NOT transfer here)
--------------------------------------------------------------------------
V3-EXQ-680b/680c's divergence was in the SHARED ENCODER (LatentStack) co-training
under two head losses with LIVE gradient flow into latent_stack at an
under-damped LR -- fixed in 680d/680e with a lower encoder LR, warmup, and grad
clipping. That mechanism is NOT present here: CrossModuleConsolidator.consolidate()
builds LOCAL, per-module Adam optimizers scoped ONLY to agent.e1.parameters() /
agent.e2.parameters() (ree_core/sleep/cross_module_consolidation.py) -- no shared
encoder is touched, and E2's loss is a plain MSE regression on its own replay
buffer. As a defensive measure anyway (the hook itself has no internal grad
clip), this script asserts every e2/e1 parameter is finite after each cycle and
FAILs cleanly (never crashes past a manifest write) if not.

EXPERIMENT_PURPOSE = "diagnostic": this is substrate-readiness validation of an
already-implemented wiring path, not new evidence for the MECH-423 hypothesis
itself (that evidence is 680e's). claim_ids=["MECH-423"] for context/traceability
only; diagnostic runs are excluded from governance confidence/conflict scoring.

TWO ARMS (seed-matched; ON is the load-bearing arm, OFF is the negative control)
----------------------------------------------------------------------------------
  ARM_SLEEP_INTEGRATED_ON  -- use_cross_module_consolidation=True, steps>0. The
                              hook fires inside force_cycle(); C1-C3 measure it.
  ARM_SLEEP_INTEGRATED_OFF -- use_cross_module_consolidation=False (consolidator
                              is None; agent.py's own guard skips the hook
                              entirely). Same seed, same waking rollout, same
                              sleep cycle otherwise. C4 asserts agent.e2's
                              parameters are BIT-IDENTICAL before/after -- so any
                              ON-arm delta is attributable to this hook alone,
                              not to SWS/REM/writeback or any other sleep pass.
                              The two other offline weight-update sites in the
                              sleep call graph both leave agent.e2 untouched:
                              self_model_aggregator.py:275 steps e2_harm_s ONLY
                              (a distinct module from agent.e2), and
                              agent.offline_integration()'s e1.integrate_experience
                              call steps agent.e1 ONLY (confirmed empirically: the
                              smoke's OFF-arm e1_delta=0.240718 with
                              consolidator=None is this second site firing --
                              E1-side only, so the E2 negative control is unaffected).

DV-SYMMETRY / red-team disposition (opus, model diversity from this session's
Sonnet 5): CONTESTED, two findings, both fixed below (never iterated further
per the red-team-pass rule -- one re-spawn budget, not used):
  - Finding 2: the consolidator's exactly-zero-loss sentinel (no replay content)
    is indistinguishable, on C2/C3 alone, from "the hook did not fire" -- fixed
    by a second P0 precondition asserting compute_e2_loss() > 0 on real data
    before trusting C2/C3, so a degenerate-transitions cell routes to
    substrate_not_ready_requeue instead of a misleading not-confirmed FAIL.
  - Finding 3: C4's "bit-identical" reading was not itself proof the OFF arm's
    sleep cycle ran (a short-circuited force_cycle would look identical) --
    fixed by asserting an unconditionally-merged sleep-cycle key
    (post_sleep_z_goal_retention, set at the end of _run_cycle regardless of
    any gate) is present in BOTH arms' returned metrics.
  Docstring correction (Finding 1, non-blocking): the "only other offline
  writer" claim above was wrong before this edit; corrected per the finding.
  Per-arm DV symmetry: neither arm's DV (max|delta| over a specific parameter
  set, a genuine per-run measurement) is a max/argmax, monotone-rescaling, or
  set-aggregate statistic invariant under a uniform additive constant,
  monotone transform, or unit permutation -- the ON/OFF manipulation is a
  binary code-path gate (consolidator None vs not), not a value transform of
  the DV, so none of the three DV-symmetry-invariance classes applies to either
  arm.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1026_mech423_sleep_integrated_e2_consolidation.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1026_mech423_sleep_integrated_e2_consolidation.py
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1026_mech423_sleep_integrated_e2_consolidation"
QUEUE_ID = "V3-EXQ-1026"
CLAIM_IDS: List[str] = ["MECH-423"]
EXPERIMENT_PURPOSE = "diagnostic"

ARMS = ("ARM_SLEEP_INTEGRATED_ON", "ARM_SLEEP_INTEGRATED_OFF")
SEEDS = (42, 123, 456)

GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16

WAKING_EPISODES = 3          # the [train] denominator (M in "ep N/M")
STEPS_PER_EPISODE = 60       # -> ~180 transitions/cell, well above CMC_BATCH

CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

E2_BUFFER_FLOOR = float(CMC_BATCH)  # precondition: enough real transitions to draw a batch
EPS = 1e-8

# The e2_transition_buffer_populated anchor is reachable by construction, not a
# hand-tuned degeneracy definition: WAKING_EPISODES(3) x STEPS_PER_EPISODE(60)
# ~= 180 real record_transition() calls per cell against a floor of CMC_BATCH
# (16) -- more than 10x headroom under the fixed, pre-registered rollout length.
ANCHOR_REACHABILITY_EXEMPT = (
    "e2_transition_buffer_populated floor (16) is >10x cleared by the fixed "
    "waking rollout (~180 transitions/cell); not a hand-tuned degeneracy definition."
)


def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        # A huge K keeps notify_episode_end's automatic K-episode cadence from
        # ever firing during the waking rollout below -- this run wants exactly
        # ONE deliberate, explicit force_cycle() firing per cell, not an
        # accidental extra auto-fired cycle interleaved with it.
        sleep_loop_episodes_K=1_000_000,
        use_cross_module_consolidation=on,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS if on else 0,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
    )
    return REEAgent(cfg)


def _param_snapshot(module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _max_abs_delta(before: List[torch.Tensor], after: List[torch.Tensor]) -> float:
    worst = 0.0
    for b, a in zip(before, after):
        d = float((a - b).abs().max().item()) if b.numel() > 0 else 0.0
        worst = max(worst, d)
    return worst


def _all_finite(module) -> bool:
    return all(torch.isfinite(p).all().item() for p in module.parameters())


def run_cell(arm: str, seed: int, waking_episodes: int, steps: int) -> Dict:
    """Build one (arm, seed) cell: populate a real waking E2 transition buffer,
    fire ONE real sleep-integrated cycle, and measure the E2 weight delta."""
    print(f"Seed {seed} Condition {arm}", flush=True)
    on = arm == "ARM_SLEEP_INTEGRATED_ON"

    env = _make_env(seed)
    agent = _make_agent(env, on)
    device = agent.device
    assert agent.sleep_loop is not None, "use_sleep_loop=True must build sleep_loop"

    rng = torch.Generator(device="cpu").manual_seed(seed)

    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    # ---------------- waking rollout: populate REAL E2 transitions ----------
    for ep in range(waking_episodes):
        print(f"  [train] {arm} seed={seed} ep {ep+1}/{waking_episodes}", flush=True)
        for _step in range(steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)

            action_idx = int(torch.randint(0, env.action_dim, (1,), generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0
            if prev_z_self is not None:
                agent.record_transition(prev_z_self, action, latent.z_self.detach())

            _, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset()
                agent.e1.reset_hidden_state()
        # NOTE: no agent.reset() at episode boundaries -- see sleep_loop_episodes_K
        # comment in _make_agent. This keeps the ONE deliberate force_cycle() call
        # below the only sleep cycle fired this cell.

    n_e2_buffer = len(agent._e2_transition_buffer)

    # ---------------- readiness preconditions: real, non-degenerate transitions --
    # p0_readiness_gate RAISES P0NotReady when unmet (it exists to abort an
    # expensive P1/P2 phase before it runs) -- there is no such phase here, so
    # catch it and let main()'s FORK check decide the run's outcome; this call
    # is only for the preconditions[] record.
    try:
        e2_loss_probe = float(agent.compute_e2_loss(batch_size=CMC_BATCH).detach().item())
    except (RuntimeError, ValueError):
        e2_loss_probe = 0.0
    try:
        preconditions = p0_readiness_gate([
            {"name": "e2_transition_buffer_populated", "measured": float(n_e2_buffer),
             "threshold": E2_BUFFER_FLOOR, "direction": "lower",
             "control": "real waking rollout via record_transition, not synthetic"},
            # red-team (opus) Finding 2: an exactly-zero e2 loss (degenerate
            # transitions, e.g. z_self never actually changing) is
            # indistinguishable from "the hook did not fire" on C2/C3 alone --
            # this precondition catches that case and routes it to
            # substrate_not_ready_requeue instead of a misleading FAIL.
            {"name": "e2_loss_nonzero_on_real_transitions", "measured": e2_loss_probe,
             "threshold": 0.0, "direction": "lower", "comparator": ">",
             "control": "probed on the same real buffer the sleep cycle draws from"},
        ])
    except P0NotReady as e:
        preconditions = e.preconditions

    # ---------------- fire ONE real sleep-integrated cycle -------------------
    e2_before = _param_snapshot(agent.e2)
    e1_before = _param_snapshot(agent.e1)
    metrics = agent.sleep_loop.force_cycle(agent) or {}
    e2_after = _param_snapshot(agent.e2)
    e1_after = _param_snapshot(agent.e1)

    # red-team (opus) Finding 3: prove the cycle actually RAN (rather than
    # short-circuiting silently, which would make a bit-identical OFF-arm E2
    # delta meaningless) via a key that phase_manager._run_cycle merges
    # unconditionally at the end of every completed cycle, independent of any
    # internal gate.
    sleep_cycle_fired = "post_sleep_z_goal_retention" in metrics
    preconditions = list(preconditions) + [{
        "name": "sleep_cycle_fired", "measured": float(sleep_cycle_fired),
        "threshold": 1.0, "direction": "lower",
        "control": "post_sleep_z_goal_retention is merged unconditionally at the end of _run_cycle",
        "met": sleep_cycle_fired,
    }]

    e2_finite = _all_finite(agent.e2)
    e1_finite = _all_finite(agent.e1)
    e2_delta = _max_abs_delta(e2_before, e2_after) if e2_finite else float("nan")
    e1_delta = _max_abs_delta(e1_before, e1_after) if e1_finite else float("nan")

    cmc_keys = sorted(k for k in metrics if k.startswith("cross_module_consolidation_"))
    updates_e2 = float(metrics.get("cross_module_consolidation_updates_e2", 0.0))
    updates_e1 = float(metrics.get("cross_module_consolidation_updates_e1", 0.0))

    print(
        f"  {arm} seed={seed} n_e2_buffer={n_e2_buffer} e2_loss_probe={e2_loss_probe:.6g} "
        f"sleep_cycle_fired={sleep_cycle_fired} cmc_keys={len(cmc_keys)} "
        f"updates_e2={updates_e2:.0f} e2_delta={e2_delta:.6g} e1_delta={e1_delta:.6g} "
        f"e2_finite={e2_finite} e1_finite={e1_finite}",
        flush=True,
    )
    ready = bool(
        n_e2_buffer >= E2_BUFFER_FLOOR and e2_loss_probe > 0.0 and sleep_cycle_fired
    )
    cell_ok = bool(ready and e2_finite and e1_finite)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "n_e2_buffer": n_e2_buffer,
        "e2_loss_probe": e2_loss_probe,
        "sleep_cycle_fired": sleep_cycle_fired,
        "ready": ready,
        "cmc_metric_keys": cmc_keys,
        "cmc_metrics_merged": bool(cmc_keys),
        "updates_e2": updates_e2,
        "updates_e1": updates_e1,
        "e2_max_abs_delta": e2_delta,
        "e1_max_abs_delta": e1_delta,
        "e2_finite": e2_finite,
        "e1_finite": e1_finite,
        "readiness_preconditions": preconditions,
        "agent": agent,
    }


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    waking_episodes = 1 if dry_run else WAKING_EPISODES
    # dry-run still needs to clear E2_BUFFER_FLOOR (CMC_BATCH=16) so the smoke
    # exercises the real PASS/FAIL criteria path, not just the FORK path.
    steps = 30 if dry_run else STEPS_PER_EPISODE
    seeds = (SEEDS[0],) if dry_run else SEEDS

    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) arms={ARMS} seeds={seeds} "
          f"waking_episodes={waking_episodes} steps={steps} cmc_steps={CMC_STEPS}", flush=True)
    t0 = time.time()

    rows: Dict[tuple, Dict] = {}
    arm_results: List[Dict] = []
    agents_seen: List[REEAgent] = []
    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "waking_episodes": waking_episodes,
                "steps_per_episode": steps,
                "cmc_steps": CMC_STEPS,
                "cmc_lr": CMC_LR,
                "cmc_batch": CMC_BATCH,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          extra_ineligible_reasons=["diagnostic_substrate_validation_no_reuse"]) as cell:
                row = run_cell(arm, seed, waking_episodes, steps)
                agent_obj = row.pop("agent")
                cell.stamp(row)  # mutates row in place: sets row["arm_fingerprint"]
            agents_seen.append(agent_obj)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    on_rows = [rows[("ARM_SLEEP_INTEGRATED_ON", s)] for s in seeds]
    off_rows = [rows[("ARM_SLEEP_INTEGRATED_OFF", s)] for s in seeds]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    # ---- FORK: any cell's readiness precondition unmet, or a non-finite
    #      parameter appeared after a cycle -> substrate_not_ready_requeue.
    unready_cells = [
        (r["arm"], r["seed"]) for r in on_rows + off_rows
        if not r["ready"] or not r["e2_finite"] or not r["e1_finite"]
    ]

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "manual-cycle-loop (run_sleep_cycle() called once per cycle in a "
            "dedicated N_CYCLES wake-sleep-test loop)"
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
    }

    def _write_manifest(manifest: Dict):
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest,
            dry_run=False,
            config=manifest.get("config"),
            seeds=list(seeds),
            script_path=Path(__file__),
            agent=agents_seen,
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    if unready_cells:
        reason = f"substrate not ready in cell(s): {unready_cells}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Diagnostic substrate-readiness precondition unmet or a non-finite "
                "parameter appeared after a sleep-integrated cycle; not a MECH-423 "
                "finding. " + reason
            ),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": [
                    p for r in (on_rows + off_rows) for p in r["readiness_preconditions"]
                ],
                "criteria_non_degenerate": {},
            },
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write_manifest(manifest)

    # ---- load-bearing criteria (ON arm) --------------------------------
    min_updates_e2 = min(r["updates_e2"] for r in on_rows)
    min_e2_delta_on = min(r["e2_max_abs_delta"] for r in on_rows)
    all_merged = all(r["cmc_metrics_merged"] for r in on_rows)

    c1_metrics_merged = bool(all_merged)
    c2_updates_e2_positive = bool(min_updates_e2 >= 1.0)
    c3_e2_delta_positive = bool(min_e2_delta_on > 0.0)

    # ---- negative control (OFF arm): e2 must be BIT-IDENTICAL -----------
    max_e2_delta_off = max(r["e2_max_abs_delta"] for r in off_rows)
    c4_off_arm_zero_delta = bool(max_e2_delta_off == 0.0)

    criteria = [
        {"name": "C1_cross_module_consolidation_metrics_merged", "load_bearing": True,
         "passed": c1_metrics_merged, "measured": float(all_merged), "threshold": 1.0},
        {"name": "C2_e2_touched_under_interleaved_schedule", "load_bearing": True,
         "passed": c2_updates_e2_positive, "measured": min_updates_e2, "threshold": 1.0},
        {"name": "C3_e2_weight_delta_positive_sleep_integrated", "load_bearing": True,
         "passed": c3_e2_delta_positive, "measured": min_e2_delta_on, "threshold": 0.0,
         "comparator": ">"},
        {"name": "C4_off_arm_e2_bit_identical_negative_control", "load_bearing": True,
         "passed": c4_off_arm_zero_delta, "measured": max_e2_delta_off, "threshold": 0.0,
         "comparator": "<="},
    ]
    criteria_non_degenerate = {
        "C1_cross_module_consolidation_metrics_merged": c1_metrics_merged,
        "C2_e2_touched_under_interleaved_schedule": c2_updates_e2_positive,
        "C3_e2_weight_delta_positive_sleep_integrated": c3_e2_delta_positive,
        # The negative control's "value" IS non-degenerate exactly because it
        # measures a genuine mechanistic absence (consolidator=None -> hook
        # skipped), not a coincidental tie between two live computations.
        "C4_off_arm_e2_bit_identical_negative_control": True,
    }

    all_pass = all(c["passed"] for c in criteria)
    outcome = "PASS" if all_pass else "FAIL"
    evidence_direction = "supports" if all_pass else "inconclusive"
    label = (
        "sleep_integrated_e2_consolidation_confirmed" if all_pass
        else "sleep_integrated_e2_consolidation_not_confirmed"
    )
    note = (
        f"Sleep-integrated call site (SleepLoopManager.force_cycle -> "
        f"phase_manager.py _run_cycle) fires MECH-423's R3 CrossModuleConsolidator "
        f"and produces a measured, non-zero E2 (world-forward) parameter update: "
        f"min max|delta_e2| across {len(on_rows)} seeds = {min_e2_delta_on:.6g}; "
        f"min updates_e2 = {min_updates_e2:.0f}. Negative control confirms "
        f"attribution: OFF-arm max|delta_e2| across seeds = {max_e2_delta_off:.6g} "
        f"(bit-identical, consolidator=None). Diagnostic substrate-readiness result; "
        f"not new evidence for the MECH-423 super-additivity hypothesis (that is "
        f"V3-EXQ-680e's)."
    ) if all_pass else (
        f"One or more load-bearing criteria failed: {[c['name'] for c in criteria if not c['passed']]}. "
        f"See criteria[] for measured/threshold detail."
    )

    print(f"\n[{EXPERIMENT_TYPE}] verdict:")
    for c in criteria:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']:.6g} "
              f"threshold={c['threshold']}")
    print(f"  -> {label} ({outcome}); elapsed={elapsed:.1f}s")

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": note,
        "non_degenerate": True,
        "interpretation": {
            "label": label,
            "criteria": criteria,
            "combination_rule": "PASS iff ALL of C1..C4 pass (plain AND, no OR/any() branching).",
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": [
                p for r in (on_rows + off_rows) for p in r["readiness_preconditions"]
            ],
        },
        "readout": {
            "c1_metrics_merged": int(c1_metrics_merged),
            "c2_updates_e2_positive": int(c2_updates_e2_positive),
            "c3_e2_delta_positive": int(c3_e2_delta_positive),
            "c4_off_arm_zero_delta": int(c4_off_arm_zero_delta),
            "min_e2_max_abs_delta_on": min_e2_delta_on,
            "min_updates_e2_on": min_updates_e2,
            "max_e2_max_abs_delta_off": max_e2_delta_off,
            "overall_pass": int(all_pass),
        },
        "sleep_integrated_e2_consolidation": {
            "min_e2_max_abs_delta_on": min_e2_delta_on,
            "min_updates_e2_on": min_updates_e2,
            "max_e2_max_abs_delta_off": max_e2_delta_off,
            "cmc_steps": CMC_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
        },
        "config": {
            "arms": list(ARMS),
            "seeds": list(seeds),
            "waking_episodes": waking_episodes,
            "steps_per_episode": steps,
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
            "cmc_steps": CMC_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
        },
        "elapsed_seconds": elapsed,
    })
    return outcome, _write_manifest(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _outcome_clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_outcome_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
