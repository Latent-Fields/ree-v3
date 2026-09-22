"""
V3-EXQ-1072 -- INV-024: offline consolidation / online commitment write-locus isolation audit

!!! NOT QUEUED -- RED-TEAM VERDICT: BLOCKING (fable, 2026-09-22). DO NOT QUEUE AS-IS. !!!
Step 4.5 adversarial design review returned BLOCKING with two findings, BOTH INDEPENDENTLY
VERIFIED AGAINST LIVE SOURCE by the authoring session before being accepted:

  B1. THE CLOSURE DISJUNCT IS STILL DEAD IN ARM_CLOSURE_ON -- the arm does not do its job.
      _closure_committed_trajectory is set at exactly one site (agent.py:9720-9752) under the
      hard conjunct `self.goal_state is not None and self.goal_state.is_active()`. goal_state
      is built only when config.goal.z_goal_enabled is True (agent.py:3427); from_dims
      defaults it False (config.py:7556) and this driver never sets it. MEASURED: the smoke
      records closure_entry_ticks = 0 in CLOSURE_ON despite closure_operator_present = True.
      So the six-flag closure stack instantiates the ClosureOperator but can never ARM the
      latch, and CLOSURE_ON tests the SAME single disjunct as CLOSURE_OFF. This defeats the
      purpose of the two-arm design (attributability of a FAIL), which was a user decision.

  B2. C2's THRESHOLD OF EXACTLY 0.0 IS UNATTAINABLE FOR A REASON THAT IS NOT ISOLATION.
      E3 commits iff _running_variance < commitment_threshold. rv starts at precision_init
      0.5 (config.py:1126), the bar is 0.40 (config.py:1124), the EMA alpha is 0.05, and rv
      updates once per tick inside post_action_update. Best case rv = 0.5 * 0.95^n, so
      0.5*0.95^4 = 0.4073 > 0.40 and 0.5*0.95^5 = 0.3869 < 0.40: NO commit is possible before
      tick 5, in any cell, any seed, any arm. Meanwhile update_residue (agent.py:11048-11058)
      writes durable residue on every harm tick with no commit check. So any hazard-approach
      in a cell's first five ticks is a structurally lineage-less write, and whether a cell
      fails C2 is decided by early random-walk geometry. CONFIRMED by this script's own smoke:
      first_commit_entry_tick = 5 and first_lineage_less_ticks = [0,1,2,3,4] in BOTH arms,
      with lineage_less_after_first_commit = 0.

  Also corrected by the reviewer: the "E3 selects on a cadence (default 10)" attribution in
  the lineage-prefix comment below is WRONG -- the smoke's commit_entry_ticks = 55/60 refutes
  a 10-tick cadence. The real mechanism is the rv warm-up in B2.

  Two further CONTESTED findings, both verified and both to be fixed alongside: (C3) C1/C2
  are max() over ALL rows rather than green-arm rows, so a red arm's cell can still drive a
  "weakens" verdict; (C4) two of the audited "authority stores" (e3_committed_trajectory,
  e3_closure_trajectory) are False pre AND post by construction, so authority_stores_audited
  overstates the discriminating witness count by two.

  DISPOSITION: the fixes are NOT the authoring session's to make -- both change what gets
  measured (B1 changes the substrate configuration; B2 changes a pre-registered criterion),
  which is a consent stop. Raised as decision chip chip-20260922-inv024-c2-predicate-and-
  closure-arm and governance flag; this file is committed UNQUEUED so the verified design work
  and the red-team findings are not lost. Queue only after the user rules on both.

Claims: INV-024 (Offline consolidation and online commitment must remain isolated at
responsibility-bearing write loci.)

EXPERIMENT_PURPOSE = "evidence"

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated
N_CYCLES wake-sleep-test loop) -- concretely, ONE cycle per cell via
REEAgent.force_sleep_cycle_at_eval_boundary(), the sanctioned two-step
(_flush_exploration_episode + sleep_loop.force_cycle) path. agent.reset() is NEVER called,
because reset would clear the residue field and destroy the WWA's own pre-cycle
non-degeneracy precondition. NOTE: agent.run_sleep_cycle() is the SD-017 SWS->REM
convenience path and does NOT reach SleepPhase.WRITEBACK, so it cannot be used here --
measured 2026-09-22 (12 cycle metrics, mech018_residue_integration_fired absent) versus the
phase-manager path (43 metrics, integration_fired=1.0).

WHAT IS UNDER TEST. INV-024's what_would_answer, implemented literally, in two halves:
  OFFLINE half -- across one full sleep cycle the AUTHORITY stores (ResidueField rbf_field
    weights/centres/active_mask, the harm-replay attribution buffer, BetaGate latch state,
    E3 committed/closure trajectory state, ClosureOperator state where present) must be
    BIT-IDENTICAL, while some REPRESENTATIONAL parameter changes (the cycle is live and
    touches only representation).
  ONLINE half -- every durable attribution write (ResidueField.accumulate with
    hypothesis_tag False) must be preceded by a commit event on the same trajectory
    (E3 committed_trajectory set, OR a closure commit intent), so the fraction of durable
    accumulate calls without commit lineage is 0.

EVIDENCE ASYMMETRY -- REGISTERED PER HALF, NOT RUN-WIDE. The two halves carry very
different evidential weight and a reader must not average them:
  * OFFLINE half: a PASS is WEAK evidence for the CLAIM. ResidueField.integrate builds its
    optimizer over self.neural_field.parameters() ONLY, rbf_field is a disjoint submodule,
    and targets are computed under torch.no_grad() (residue/field.py:1122-1170), so
    no-erasure holds BY CONSTRUCTION. A PASS here confirms that the current implementation
    honours the isolation -- NOT that isolation is architecturally necessary. Treat it as a
    contract regression-guard. A FAIL would be very strong evidence (a construction-level
    guarantee breached).
  * ONLINE half: genuinely discriminating, and this is where a real FAIL can come from.
    There are TWO durable-write paths. e3_selector.py:4664 IS commitment-gated by
    construction. But agent.py:11051 (REEAgent.update_residue) is gated only on
    `owned and not hypothesis_tag and self._current_latent is not None`, where
    `owned: bool = True` is a parameter with default True (agent.py:10873) and NO commit
    state is consulted at that site at all. So the online criterion is not construction-
    guaranteed and can fail on live measurement.
This asymmetry is emitted in the manifest under `evidence_asymmetry_per_half` so governance
cannot later read an overall PASS as confirmation of the necessity claim.

LINEAGE READ POINT = COMMIT ENTRY, NOT WRITE TIME (instrumentation correctness, ratified).
`self._committed_trajectory = None` is the LAST statement of E3.post_action_update, and
REEAgent.update_residue calls post_action_update BEFORE its own durable write. So at the
instant of the write the E3 commit latch is ALWAYS None by construction, and a naive
write-time read would report ~100% lineage-less as a pure ordering artifact rather than a
finding. This driver therefore snapshots commit state in the StepHarness `on_action` hook
(after select_action, before env.step and before update_residue) and attributes each durable
write to THAT tick's snapshot.

TWO ARMS, and why (user decision 2026-09-22, via the orchestrator decision lane).
INV-024's online predicate accepts lineage from "E3 committed_trajectory set, OR a closure
commit intent". But agent.closure_operator is None on the configuration the WWA names, so
the SECOND DISJUNCT IS STRUCTURALLY DEAD there (it needs use_closure_operator=True AND
use_lateral_pfc_analog=True; e3_selector.py:562 records that with the closure trajectory flag
off "every consuming union reduces to the bool-latch behaviour"). Testing a disjunctive
predicate with one disjunct dead would inflate the falsification and could commission the
wrong MECH-067 build. The user chose to run BOTH configurations so a FAIL is ATTRIBUTABLE:
  ARM_CLOSURE_OFF -- the WWA-named configuration; closure disjunct structurally dead.
  ARM_CLOSURE_ON  -- closure plane live, so both disjuncts are evaluable.
If lineage-less writes vanish under CLOSURE_ON the gap is a closure-wiring matter; if they
persist it is a genuine isolation violation and MECH-067 is correctly commissioned.

ARM-DIFFERENCE CAVEAT, recorded rather than papered over. The closure plane cannot be
enabled in isolation: agent.py:1538-1560 enforces a mandatory precondition chain, so
ARM_CLOSURE_ON necessarily carries SIX flags (use_closure_operator, use_lateral_pfc_analog,
use_closure_commit_entry, use_closure_commit_beta_coupling, use_natural_commit_latch_hold,
use_closure_commit_entry_trajectory). use_natural_commit_latch_hold in particular changes
commit occupancy dynamics. There is NO configuration in which the closure disjunct is live
without these, so this is a substrate fact, not a design choice. CONSEQUENCE FOR READING: a
CROSS-ARM DIFFERENCE is attributable to "the closure commit plane as a whole", not to the
closure disjunct in isolation. The PRIMARY criteria are deliberately PER-ARM ABSOLUTE (no
authority mutation; lineage-less fraction 0), not a cross-arm delta, so each arm
independently answers "does isolation hold on this configuration" and the verdict does not
depend on the cross-arm contrast being clean.

THE PAIRING TRAP (both flags required, not one). phase_manager.py:679 gates the WRITEBACK
integrate() CALL on use_sleep_residue_integration; whether that call TRAINS is
ResidueConfig.use_offline_integration_gradient_step. With only the latter the offline call
never fires and the WWA's non-degeneracy precondition is unsatisfiable (a vacuous run). Both
are pinned ON here and both are asserted live in the readiness preconditions.

Pre-registered acceptance (NOT derived from this run's statistics):
  C1  offline_authority_stores_unmutated -- every audited authority-store hash is identical
      pre- and post-cycle, for every seed and BOTH arms. (load-bearing)
  C2  online_lineage_complete -- lineage_less_fraction == 0 over all durable accumulate
      calls, for every seed and BOTH arms. (load-bearing)
  PASS iff the per-arm readiness gate is green AND C1 AND C2.

Falsifying per INV-024: one observed authority-store mutation during a cycle, or one durable
online write with no commit lineage. Either refutes the invariant AS IMPLEMENTED and routes
to MECH-067 (the enforcement mechanism), not to a re-run.

Output:
  evidence/experiments/v3_exq_1072_inv024_offline_online_isolation_audit/
    v3_exq_1072_inv024_offline_online_isolation_audit_<ts>.json
"""

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._harness import StepHarness, StepHooks
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_1072_inv024_offline_online_isolation_audit"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["INV-024"]
# z_goal is orthogonal to this run's DV (write-locus isolation), and the audit never calls
# update_z_goal itself -- but StepHarness does on the canonical path and the run steps an
# agent, so record the stream's liveness rather than leaving it unmeasured.
_ZG = ZGoalStreamAccumulator()

SLEEP_DRIVER_PATTERN = (
    "manual-cycle-loop (one forced cycle per cell via "
    "REEAgent.force_sleep_cycle_at_eval_boundary -> SleepLoopManager.force_cycle -> "
    "SleepPhaseManager._run_cycle incl. SleepPhase.WRITEBACK; agent.reset() never called)"
)

# ---- run geometry ----
DEFAULT_SEEDS = [42, 43, 45]   # 44 excluded per CLAUDE.md reef-config instability precedent
WAKING_STEPS = 400             # waking steps per cell -- denominator of the `ep N/M` prints
GRID_SIZE = 8
NUM_HAZARDS = 2
NUM_RESOURCES = 3
MAX_EPISODE_STEPS = 200
ARMS = ["CLOSURE_OFF", "CLOSURE_ON"]

# ---- pre-registered thresholds (constants; NOT derived from this run) ----
MAX_AUTHORITY_MUTATIONS = 0      # C1: zero mutated authority stores
MAX_LINEAGE_LESS_FRACTION = 0.0  # C2: zero durable writes without commit lineage
MIN_ACTIVE_CENTERS = 0.0         # readiness floor (strict >): residue field must be populated
MIN_COMMIT_ENTRIES = 0.0         # readiness floor (strict >): commitment must have occurred
MIN_DURABLE_WRITES = 0.0         # readiness floor (strict >): the online DV must have a denominator
MIN_REPR_DELTA = 0.0             # readiness floor (strict >): the cycle must be LIVE


# --------------------------------------------------------------------------------------
# hashing / snapshots
# --------------------------------------------------------------------------------------
def _hash_tensor(t: Any) -> str:
    """Content hash of an authority-store value. Stable, byte-level, order-preserving."""
    if t is None:
        return "none"
    if isinstance(t, bool):
        return "bool:%s" % t
    if not torch.is_tensor(t):
        try:
            t = torch.as_tensor(t)
        except Exception:
            return "repr:" + hashlib.sha256(repr(t).encode()).hexdigest()[:16]
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _authority_snapshot(agent: REEAgent) -> Dict[str, str]:
    """The stores INV-024 says a sleep cycle must NOT mutate."""
    rf = agent.residue_field
    e3 = agent.e3
    bg = agent.beta_gate
    snap: Dict[str, str] = {
        "residue_rbf_weights": _hash_tensor(rf.rbf_field.weights),
        "residue_rbf_centers": _hash_tensor(rf.rbf_field.centers),
        "residue_rbf_active_mask": _hash_tensor(rf.rbf_field.active_mask),
        "residue_total": _hash_tensor(rf.total_residue),
        "residue_num_harm_events": _hash_tensor(rf.num_harm_events),
        "harm_replay_buffer_len": str(len(rf._harm_history)),
        "harm_replay_buffer": (
            _hash_tensor(torch.stack(rf._harm_history)) if rf._harm_history else "empty"
        ),
        "beta_gate_latch": "bool:%s" % bool(bg.is_elevated),
        "e3_committed_trajectory": "bool:%s" % (e3._committed_trajectory is not None),
        "e3_closure_trajectory": "bool:%s" % (e3._closure_committed_trajectory is not None),
        "e3_persistent_trajectory": (
            "bool:%s" % (e3._persistent_committed_trajectory is not None)
        ),
    }
    bf = getattr(rf, "benefit_rbf_field", None)
    if bf is not None and getattr(rf, "benefit_terrain_enabled", False):
        snap["benefit_rbf_weights"] = _hash_tensor(bf.weights)
        snap["benefit_rbf_centers"] = _hash_tensor(bf.centers)
        snap["benefit_rbf_active_mask"] = _hash_tensor(bf.active_mask)
    co = getattr(agent, "closure_operator", None)
    if co is not None:
        payload = repr(sorted(
            (k, str(v)) for k, v in vars(co).items() if not k.startswith("__")
        ))
        snap["closure_operator_state"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return snap


def _representational_snapshot(agent: REEAgent) -> Dict[str, str]:
    """Stores the cycle IS allowed to change -- the liveness witnesses."""
    out: Dict[str, str] = {}
    rf = agent.residue_field
    nf_params = [p.detach().flatten() for p in rf.neural_field.parameters()]
    if nf_params:
        out["residue_neural_field"] = _hash_tensor(torch.cat(nf_params))
    for name in ("e2_harm_s", "e1", "e2"):
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "parameters"):
            params = [p.detach().flatten() for p in module.parameters()]
            if params:
                out[name] = _hash_tensor(torch.cat(params))
    return out


def _hash_ledger_self_test() -> Dict[str, float]:
    """INSTRUMENT POSITIVE CONTROL -- prove the ledger can DETECT a mutation.

    A bit-identical reading is only meaningful if the ledger could have reported a
    difference. This is the 'gate that certifies its own subject' guard: perturb a CLONE of
    a representative authority tensor by the smallest plausible amount and confirm the hash
    changes. Returns a 1.0/0.0 witness recorded as a readiness precondition.
    """
    base = torch.arange(32, dtype=torch.float32)
    h0 = _hash_tensor(base)
    mutated = base.clone()
    mutated[7] += 1e-6
    h1 = _hash_tensor(mutated)
    flipped = base.clone().to(torch.bool)
    h2 = _hash_tensor(flipped)
    h3 = _hash_tensor(~flipped)
    detects_float = 1.0 if h0 != h1 else 0.0
    detects_mask = 1.0 if h2 != h3 else 0.0
    return {
        "detects_float_perturbation": detects_float,
        "detects_mask_flip": detects_mask,
        "detects_mutation": 1.0 if (detects_float and detects_mask) else 0.0,
    }


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def _arm_config_slice(env: Any, arm: str) -> Dict[str, Any]:
    """The declared config slice for this arm (also the arm_fingerprint slice)."""
    slice_: Dict[str, Any] = {
        "body_obs_dim": int(env.body_obs_dim),
        "world_obs_dim": int(env.world_obs_dim),
        "action_dim": int(env.action_dim),
        # WWA-named instrument configuration
        "use_sleep_aggregation_cluster": True,
        "use_cross_module_consolidation": True,
        # THE PAIRING TRAP -- both required (see module docstring)
        "use_sleep_residue_integration": True,
        "use_offline_integration_gradient_step": True,
        # boundary sleep path unreachable; the driver forces the cycle explicitly
        "sleep_loop_episodes_K": 10_000_000,
        # env geometry
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "waking_steps": WAKING_STEPS,
        # Acceptance constants ride the slice: they affect the recorded cell_pass readout,
        # so a consumer with a different scheme must MISS rather than false-HIT these cells.
        "max_authority_mutations": MAX_AUTHORITY_MUTATIONS,
        "max_lineage_less_fraction": MAX_LINEAGE_LESS_FRACTION,
    }
    if arm == "CLOSURE_ON":
        # Mandatory precondition chain enforced at agent.py:1538-1560 -- all six or none.
        slice_.update({
            "use_closure_operator": True,
            "use_lateral_pfc_analog": True,
            "use_closure_commit_entry": True,
            "use_closure_commit_beta_coupling": True,
            "use_natural_commit_latch_hold": True,
            "use_closure_commit_entry_trajectory": True,
        })
    return slice_


def _build_config(env: Any, arm: str) -> REEConfig:
    kwargs: Dict[str, Any] = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        use_sleep_aggregation_cluster=True,
        use_cross_module_consolidation=True,
        use_sleep_residue_integration=True,
        use_offline_integration_gradient_step=True,
        sleep_loop_episodes_K=10_000_000,
    )
    if arm == "CLOSURE_ON":
        kwargs.update(
            use_closure_operator=True,
            use_lateral_pfc_analog=True,
            use_closure_commit_entry=True,
            use_closure_commit_beta_coupling=True,
            use_natural_commit_latch_hold=True,
            use_closure_commit_entry_trajectory=True,
        )
    return REEConfig.from_dims(**kwargs)


# --------------------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------------------
def _run_cell(arm: str, seed: int, waking_steps: int) -> Dict[str, Any]:
    """One (arm x seed) audit cell: waking lineage audit, then ONE sleep-cycle hash audit."""
    env = CausalGridWorldV2(
        size=GRID_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        max_episode_steps=MAX_EPISODE_STEPS, seed=seed,
    )
    _flat, obs = env.reset()
    agent = REEAgent(_build_config(env, arm)).to(torch.device("cpu"))
    agent.eval()

    print("Seed %d Condition %s" % (seed, arm), flush=True)

    # --- ONLINE half instrumentation -------------------------------------------------
    tick_state: Dict[str, Any] = {"index": 0, "lineage": None}
    writes: List[Dict[str, Any]] = []
    commit_entry_ticks = 0
    closure_entry_ticks = 0

    def _on_action(**_kwargs: Any) -> None:
        """Commit-ENTRY snapshot: after select_action, before env.step/update_residue."""
        e3 = agent.e3
        tick_state["lineage"] = {
            "tick": int(tick_state["index"]),
            "e3_committed": e3._committed_trajectory is not None,
            "closure_committed": e3._closure_committed_trajectory is not None,
            "persistent_committed": e3._persistent_committed_trajectory is not None,
            "beta_elevated": bool(agent.beta_gate.is_elevated),
        }

    residue_field = agent.residue_field
    _original_accumulate = residue_field.accumulate

    def _audited_accumulate(z_world, harm_magnitude=1.0, world_delta=None,
                            hypothesis_tag=False):
        result = _original_accumulate(
            z_world, harm_magnitude=harm_magnitude, world_delta=world_delta,
            hypothesis_tag=hypothesis_tag,
        )
        # MECH-094: a hypothesis_tag write is refused inside accumulate and is NOT durable,
        # so it is deliberately not counted in the online denominator.
        if not hypothesis_tag:
            lineage = dict(tick_state["lineage"] or {"tick": int(tick_state["index"])})
            lineage["phase"] = tick_state.get("phase", "waking")
            writes.append(lineage)
        return result

    residue_field.accumulate = _audited_accumulate

    harness = StepHarness(
        agent, env, train_mode=False, hooks=StepHooks(on_action=_on_action), seed=seed,
    )

    tick_state["phase"] = "waking"
    harm_events_start = float(residue_field.num_harm_events)
    for step_index in range(waking_steps):
        tick_state["index"] = step_index
        result = harness.step(obs)
        lineage = tick_state["lineage"] or {}
        if lineage.get("e3_committed"):
            commit_entry_ticks += 1
        if lineage.get("closure_committed"):
            closure_entry_ticks += 1
        obs = result.next_obs_dict
        if (step_index + 1) % 50 == 0:
            print("  [train] inv024 seed=%d arm=%s ep %d/%d writes=%d"
                  % (seed, arm, step_index + 1, waking_steps, len(writes)), flush=True)
        if result.done:
            _flat, obs = env.reset()
            harness.reset()

    harm_events_end = float(residue_field.num_harm_events)
    durable_writes = len(writes)
    lineage_less = [
        w for w in writes
        if not (w.get("e3_committed") or w.get("closure_committed"))
    ]
    lineage_less_fraction = (len(lineage_less) / durable_writes) if durable_writes else 0.0
    # DECISIVE DIAGNOSTIC: is a lineage-less write a STARTUP artifact or an ONGOING breach?
    # E3 selects on a cadence (heartbeat.e3_steps_per_tick, default 10), so no commitment can
    # exist before the first E3 tick and a harm event in that prefix necessarily writes
    # without lineage. Writes AFTER the first observed commit entry cannot be explained that
    # way. Recording both lets governance separate the two readings without a re-run; the
    # PRE-REGISTERED criterion C2 remains the whole-run fraction either way.
    commit_ticks = [w.get("tick", -1) for w in writes
                    if w.get("e3_committed") or w.get("closure_committed")]
    first_commit_tick = min(commit_ticks) if commit_ticks else None
    if first_commit_tick is None:
        lineage_less_after_first_commit = len(lineage_less)
    else:
        lineage_less_after_first_commit = sum(
            1 for w in lineage_less if int(w.get("tick", -1)) > int(first_commit_tick)
        )
    # Generous recording: a BROADER predicate a later reader might prefer, banked so the
    # alternative need not be re-run. Not the pre-registered criterion.
    lineage_less_broad = [
        w for w in writes
        if not (w.get("e3_committed") or w.get("closure_committed")
                or w.get("persistent_committed") or w.get("beta_elevated"))
    ]

    # --- OFFLINE half: hash audit around ONE forced sleep cycle ----------------------
    active_centers_pre = float(residue_field.rbf_field.active_mask.sum().item())
    pre_authority = _authority_snapshot(agent)
    pre_repr = _representational_snapshot(agent)

    tick_state["phase"] = "sleep"
    writes_before_sleep = len(writes)
    cycle_metrics = agent.force_sleep_cycle_at_eval_boundary() or {}
    offline_durable_writes = len(writes) - writes_before_sleep
    tick_state["phase"] = "post_sleep"

    post_authority = _authority_snapshot(agent)
    post_repr = _representational_snapshot(agent)

    mutated_stores = sorted(
        k for k in pre_authority if pre_authority[k] != post_authority.get(k)
    )
    changed_repr = sorted(k for k in pre_repr if pre_repr[k] != post_repr.get(k))

    integration_fired = float(cycle_metrics.get("mech018_residue_integration_fired", 0.0) or 0.0)
    integration_trains = float(cycle_metrics.get("mech018_residue_trains", 0.0) or 0.0)
    neural_delta = float(cycle_metrics.get("mech018_residue_neural_param_delta_norm", 0.0) or 0.0)
    rbf_abs_sum_delta = float(
        cycle_metrics.get("mech018_residue_rbf_weight_abs_sum_delta", 0.0) or 0.0
    )
    # "SOME representational parameter changed" -- the WWA's own liveness wording.
    repr_delta_witness = float(len(changed_repr))

    residue_field.accumulate = _original_accumulate
    _ZG.observe(agent)

    cell_pass = (
        len(mutated_stores) <= MAX_AUTHORITY_MUTATIONS
        and lineage_less_fraction <= MAX_LINEAGE_LESS_FRACTION
    )
    print("verdict: %s seed=%d arm=%s mutated=%d lineage_less=%.4f"
          % ("PASS" if cell_pass else "FAIL", seed, arm,
             len(mutated_stores), lineage_less_fraction), flush=True)

    return {
        "arm": arm,
        "seed": int(seed),
        "waking_steps": int(waking_steps),
        # online half
        "durable_writes": int(durable_writes),
        "lineage_less_writes": int(len(lineage_less)),
        "lineage_less_fraction": float(lineage_less_fraction),
        "lineage_less_writes_broad_predicate": int(len(lineage_less_broad)),
        "commit_entry_ticks": int(commit_entry_ticks),
        "closure_entry_ticks": int(closure_entry_ticks),
        "first_lineage_less_ticks": [int(w.get("tick", -1)) for w in lineage_less[:20]],
        "first_commit_entry_tick": (int(first_commit_tick) if first_commit_tick is not None else -1),
        "lineage_less_after_first_commit": int(lineage_less_after_first_commit),
        "harm_events_start": float(harm_events_start),
        "harm_events_end": float(harm_events_end),
        "harm_events_delta": float(harm_events_end - harm_events_start),
        # offline half
        "authority_stores_audited": int(len(pre_authority)),
        "authority_stores_mutated": int(len(mutated_stores)),
        "mutated_store_names": mutated_stores,
        "representational_stores_changed": changed_repr,
        "representational_delta_witness": repr_delta_witness,
        "active_centers_pre_cycle": float(active_centers_pre),
        "offline_durable_writes_during_cycle": int(offline_durable_writes),
        "mech018_integration_fired": integration_fired,
        "mech018_residue_trains": integration_trains,
        "mech018_neural_param_delta_norm": neural_delta,
        "mech018_rbf_weight_abs_sum_delta": rbf_abs_sum_delta,
        "sleep_cycle_metric_count": int(len(cycle_metrics)),
        "closure_operator_present": bool(getattr(agent, "closure_operator", None) is not None),
        "cell_pass": bool(cell_pass),
        "authority_snapshot_pre": pre_authority,
        "authority_snapshot_post": post_authority,
    }


# --------------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------------
def _precondition_specs() -> List[PreconditionSpec]:
    """Readiness gate. Every spec is meaningful for BOTH arms, so none is scoped out."""
    return [
        PreconditionSpec(
            name="hash_ledger_detects_mutation",
            description=(
                "INSTRUMENT POSITIVE CONTROL: the authority-store hash ledger flags a "
                "1e-6 float perturbation and a boolean-mask flip on a clone. Without this "
                "a bit-identical reading would certify its own subject"
            ),
            control="clone of a representative authority tensor, perturbed by 1e-6",
            threshold=0.0, direction="lower", kind="readiness",
        ),
        PreconditionSpec(
            name="offline_pathway_live",
            description=(
                "WWA non-degeneracy: the sleep cycle changed SOME representational "
                "parameter (count of representational stores whose hash moved). An "
                "isolation result on an inert sleep pass is vacuous and must not score"
            ),
            control="representational snapshot diffed across the forced cycle",
            threshold=MIN_REPR_DELTA, direction="lower", kind="readiness",
        ),
        PreconditionSpec(
            name="residue_active_centers_pre_cycle",
            description=(
                "WWA non-degeneracy: the residue field holds active centres BEFORE the "
                "cycle, so there is authority state that could have been mutated"
            ),
            control="rbf_field.active_mask sum immediately before the forced cycle",
            threshold=MIN_ACTIVE_CENTERS, direction="lower", kind="readiness",
        ),
        PreconditionSpec(
            name="commitment_events_occurred",
            description=(
                "WWA non-degeneracy: commitment events occurred in waking (n_committed>0), "
                "so the lineage predicate had something to be satisfied BY"
            ),
            control="commit-entry ticks counted in the on_action hook",
            threshold=MIN_COMMIT_ENTRIES, direction="lower", kind="readiness",
        ),
        PreconditionSpec(
            name="durable_writes_occurred",
            description=(
                "WWA non-degeneracy: residue writes occurred in waking, so the online "
                "criterion has a non-zero denominator rather than a vacuous 0/0"
            ),
            control="audited non-hypothesis accumulate calls during the waking phase",
            threshold=MIN_DURABLE_WRITES, direction="lower", kind="readiness",
        ),
        PreconditionSpec(
            name="offline_integration_call_fired_and_trains",
            description=(
                "THE PAIRING TRAP: use_sleep_residue_integration supplied the WRITEBACK "
                "integrate() CALL and use_offline_integration_gradient_step made it TRAIN. "
                "Guards a silently-inert offline pass reading as isolation"
            ),
            control="mech018_residue_integration_fired * mech018_residue_trains from the cycle",
            threshold=0.0, direction="lower", kind="readiness",
        ),
    ]


def _arm_measured(rows: List[Dict[str, Any]], self_test: Dict[str, float]) -> Dict[str, float]:
    """Worst cell per precondition -- `met` is an all-cells claim, so report the extremum."""
    return {
        "hash_ledger_detects_mutation": float(self_test["detects_mutation"]),
        "offline_pathway_live": float(min(r["representational_delta_witness"] for r in rows)),
        "residue_active_centers_pre_cycle": float(min(r["active_centers_pre_cycle"] for r in rows)),
        "commitment_events_occurred": float(min(r["commit_entry_ticks"] for r in rows)),
        "durable_writes_occurred": float(min(r["durable_writes"] for r in rows)),
        "offline_integration_call_fired_and_trains": float(
            min(r["mech018_integration_fired"] * r["mech018_residue_trains"] for r in rows)
        ),
    }


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)
    waking_steps = WAKING_STEPS
    if dry_run:
        seeds = seeds[:1]
        waking_steps = 60   # smoke still exercises BOTH arms end to end

    print("[V3-EXQ-1072] INV-024 offline/online isolation audit", flush=True)
    print("  Arms: %s  Seeds: %s  Waking steps/cell: %d"
          % (ARMS, seeds, waking_steps), flush=True)
    print("  Output: REE_assembly/evidence/experiments/%s/" % EXPERIMENT_TYPE, flush=True)

    self_test = _hash_ledger_self_test()
    print("  [instrument] hash ledger self-test: %s" % self_test, flush=True)

    specs = _precondition_specs()
    arm_contexts = {arm: {"arm": arm, "id": arm, "closure_plane": (arm == "CLOSURE_ON")}
                    for arm in ARMS}
    # Design-time refusal: no precondition may be structurally unsatisfiable for an arm.
    assert_no_structurally_unsatisfiable_gate(specs, list(arm_contexts.values()),
                                              arm_id_key="arm")

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            env_probe = CausalGridWorldV2(
                size=GRID_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
                max_episode_steps=MAX_EPISODE_STEPS, seed=seed,
            )
            config_slice = _arm_config_slice(env_probe, arm)
            with arm_cell(
                seed,
                config_slice=config_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,   # cross-driver reusable mint
            ) as cell:
                row = _run_cell(arm, seed, waking_steps)
                cell.stamp(row)
            rows.append(row)

    # --- per-arm readiness gates (never AND the whole run -- V3-EXQ-785) --------------
    arm_gates = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arm_gates.append(evaluate_arm_gate(
            arm, arm_contexts[arm], specs, _arm_measured(arm_rows, self_test),
        ))
    aggregate = aggregate_arm_gates(arm_gates)

    # --- pre-registered criteria -----------------------------------------------------
    worst_mutations = max(r["authority_stores_mutated"] for r in rows)
    worst_lineage_less = max(r["lineage_less_fraction"] for r in rows)
    c1 = bool(worst_mutations <= MAX_AUTHORITY_MUTATIONS)
    c2 = bool(worst_lineage_less <= MAX_LINEAGE_LESS_FRACTION)

    gate_green = bool(aggregate["any_green"])
    passed = bool(gate_green and c1 and c2)
    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif passed:
        label = "offline_online_write_locus_isolation_holds"
    else:
        label = "offline_online_write_locus_isolation_violated"
    outcome = "PASS" if passed else "FAIL"

    if not gate_green:
        evidence_direction = "non_contributory"
    elif passed:
        evidence_direction = "supports"
    else:
        evidence_direction = "weakens"

    per_arm_summary = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        per_arm_summary[arm] = {
            "n_cells": len(arm_rows),
            "authority_stores_mutated_max": max(r["authority_stores_mutated"] for r in arm_rows),
            "lineage_less_fraction_max": max(r["lineage_less_fraction"] for r in arm_rows),
            "lineage_less_fraction_mean": float(
                sum(r["lineage_less_fraction"] for r in arm_rows) / len(arm_rows)
            ),
            "durable_writes_total": sum(r["durable_writes"] for r in arm_rows),
            "lineage_less_writes_total": sum(r["lineage_less_writes"] for r in arm_rows),
            "closure_entry_ticks_total": sum(r["closure_entry_ticks"] for r in arm_rows),
            "closure_operator_present": all(r["closure_operator_present"] for r in arm_rows),
        }

    # Flat scalar readout -- booleans as ints, no non-finite values (standard sec 3b).
    readout: Dict[str, float] = {
        "n_cells": float(len(rows)),
        "n_seeds": float(len(seeds)),
        "authority_stores_mutated_max": float(worst_mutations),
        "lineage_less_fraction_max": float(worst_lineage_less),
        "c1_offline_authority_stores_unmutated": 1.0 if c1 else 0.0,
        "c2_online_lineage_complete": 1.0 if c2 else 0.0,
        "readiness_gate_green": 1.0 if gate_green else 0.0,
        "hash_ledger_detects_mutation": float(self_test["detects_mutation"]),
        "durable_writes_total": float(sum(r["durable_writes"] for r in rows)),
        "lineage_less_writes_total": float(sum(r["lineage_less_writes"] for r in rows)),
        "lineage_less_writes_broad_total": float(
            sum(r["lineage_less_writes_broad_predicate"] for r in rows)
        ),
        "commit_entry_ticks_total": float(sum(r["commit_entry_ticks"] for r in rows)),
        "lineage_less_after_first_commit_total": float(
            sum(r["lineage_less_after_first_commit"] for r in rows)
        ),
        "offline_durable_writes_during_cycle_total": float(
            sum(r["offline_durable_writes_during_cycle"] for r in rows)
        ),
        "representational_delta_witness_min": float(
            min(r["representational_delta_witness"] for r in rows)
        ),
        "active_centers_pre_cycle_min": float(min(r["active_centers_pre_cycle"] for r in rows)),
        "mech018_neural_param_delta_norm_min": float(
            min(r["mech018_neural_param_delta_norm"] for r in rows)
        ),
        "mech018_rbf_weight_abs_sum_delta_max_abs": float(
            max(abs(r["mech018_rbf_weight_abs_sum_delta"]) for r in rows)
        ),
        "closure_off_lineage_less_fraction_max": float(
            per_arm_summary["CLOSURE_OFF"]["lineage_less_fraction_max"]
        ),
        "closure_on_lineage_less_fraction_max": float(
            per_arm_summary["CLOSURE_ON"]["lineage_less_fraction_max"]
        ),
    }

    criteria = [
        {
            "name": "C1_offline_authority_stores_unmutated",
            "load_bearing": True,
            "passed": c1,
            "measured": float(worst_mutations),
            "threshold": float(MAX_AUTHORITY_MUTATIONS),
            "comparator": "<=",
            "direction": "upper",
            "measured_note": "worst cell: max mutated authority stores across all cells",
        },
        {
            "name": "C2_online_lineage_complete",
            "load_bearing": True,
            "passed": c2,
            "measured": float(worst_lineage_less),
            "threshold": float(MAX_LINEAGE_LESS_FRACTION),
            "comparator": "<=",
            "direction": "upper",
            "measured_note": "worst cell: max lineage-less durable-write fraction across all cells",
        },
    ]

    interpretation: Dict[str, Any] = {
        "label": label,
        "preconditions": aggregate["adjudication_preconditions"],
        "criteria_non_degenerate": {
            # C1 discriminates iff there was authority state that COULD have been mutated
            # and the ledger could have seen it; C2 iff there were durable writes to audit
            # and commitment actually occurred.
            "C1_offline_authority_stores_unmutated": bool(
                min(r["active_centers_pre_cycle"] for r in rows) > 0.0
                and self_test["detects_mutation"] > 0.0
                and min(r["representational_delta_witness"] for r in rows) > 0.0
            ),
            "C2_online_lineage_complete": bool(
                min(r["durable_writes"] for r in rows) > 0
                and min(r["commit_entry_ticks"] for r in rows) > 0
            ),
        },
        "criteria": criteria,
        "combination_rule": (
            "PASS = per-arm readiness gate green (any arm green; never ANDed whole-run) "
            "AND C1_offline_authority_stores_unmutated AND C2_online_lineage_complete. "
            "Both criteria are worst-cell over all (arm x seed) cells, so a single "
            "violating cell fails the run -- INV-024's own 'one confirmed instance refutes "
            "the invariant AS IMPLEMENTED'."
        ),
        "dv_symmetry_note": (
            "ARM_CLOSURE_OFF: the DV is a content hash of the authority stores and a count "
            "of durable writes lacking a commit-entry snapshot. Its symmetry group is "
            "permutation of write ORDER within a tick; the manipulation (running a sleep "
            "cycle) is NOT invariant under it -- a cycle that wrote an authority store would "
            "change the hash regardless of ordering, and the hash ledger's sensitivity to a "
            "1e-6 perturbation is measured, not assumed. "
            "ARM_CLOSURE_ON: same DV and same symmetry group. The manipulation (enabling the "
            "closure commit plane) is not invariant under it either -- it adds a second, "
            "independently-observable lineage source (closure_entry_ticks is recorded "
            "separately), so an arm difference is a real change in the predicate's "
            "satisfiability rather than a relabelling. Neither arm's DV is a broadcast "
            "constant, a monotone rescaling, or a set-aggregate over interchangeable units."
        ),
        "arm_difference_caveat": (
            "The closure plane cannot be enabled in isolation: agent.py:1538-1560 enforces a "
            "mandatory six-flag precondition chain, and use_natural_commit_latch_hold in "
            "particular changes commit occupancy. A CROSS-ARM DIFFERENCE is therefore "
            "attributable to the closure commit plane AS A WHOLE, not to the closure "
            "disjunct alone. The pre-registered criteria are per-arm ABSOLUTE, not a "
            "cross-arm delta, so each arm independently answers the isolation question."
        ),
    }

    evidence_asymmetry_per_half = {
        "offline_half": {
            "criterion": "C1_offline_authority_stores_unmutated",
            "pass_evidence_strength": "weak",
            "fail_evidence_strength": "very_strong",
            "why": (
                "ResidueField.integrate builds its optimizer over neural_field.parameters() "
                "only, rbf_field is a disjoint submodule, and targets are computed under "
                "torch.no_grad() (residue/field.py:1122-1170), so no-erasure holds BY "
                "CONSTRUCTION. A PASS confirms the current implementation honours the "
                "isolation, NOT that isolation is architecturally necessary -- read it as a "
                "contract regression-guard. A FAIL would breach a construction-level "
                "guarantee and is very strong evidence."
            ),
        },
        "online_half": {
            "criterion": "C2_online_lineage_complete",
            "pass_evidence_strength": "moderate",
            "fail_evidence_strength": "very_strong",
            "why": (
                "NOT construction-guaranteed. Two durable-write paths exist: "
                "e3_selector.py:4664 is commitment-gated, but agent.py:11051 "
                "(REEAgent.update_residue) is gated only on `owned` (a parameter defaulting "
                "to True, agent.py:10873) and consults no commit state. This half is "
                "genuinely discriminating and is where a real FAIL can come from."
            ),
        },
        "governance_note": (
            "Do NOT average the two halves into one evidence strength. An overall PASS is "
            "NOT confirmation of INV-024's necessity claim; it is a regression-guard on the "
            "offline half plus a moderate live confirmation on the online half."
        ),
    }

    summary_markdown = """# V3-EXQ-1072 -- INV-024 offline/online write-locus isolation audit

**Status:** {outcome} -- label: `{label}`
**Purpose:** evidence (INV-024). Two arms x {nseeds} seeds = {ncells} cells.

- C1 offline authority stores unmutated: **{c1}** (worst cell: {mut} mutated, threshold {c1t})
- C2 online lineage complete: **{c2}** (worst cell: {ll:.4f}, threshold {c2t})
- readiness gate green: {green} ({greenarms})
- durable writes audited: {dw} | lineage-less: {llw}
- CLOSURE_OFF lineage-less fraction (max): {off:.4f}
- CLOSURE_ON  lineage-less fraction (max): {on:.4f}

**Evidence asymmetry is registered PER HALF** -- see `evidence_asymmetry_per_half`. An
overall PASS is a contract regression-guard on the offline half (which is construction-
guaranteed) plus a moderate live confirmation on the online half (which is not). Do not
read it as confirmation that isolation is architecturally necessary.

See `interpretation` for the pre-registered acceptance rule, the per-arm readiness gate and
the arm-difference caveat, and `per_cell_results` for the full (arm x seed) table including
the pre/post authority-store hash snapshots.
""".format(
        outcome=outcome, label=label, nseeds=len(seeds), ncells=len(rows),
        c1=c1, mut=worst_mutations, c1t=MAX_AUTHORITY_MUTATIONS,
        c2=c2, ll=worst_lineage_less, c2t=MAX_LINEAGE_LESS_FRACTION,
        green=gate_green, greenarms=", ".join(aggregate["green_arms"]) or "none",
        dw=int(readout["durable_writes_total"]), llw=int(readout["lineage_less_writes_total"]),
        off=per_arm_summary["CLOSURE_OFF"]["lineage_less_fraction_max"],
        on=per_arm_summary["CLOSURE_ON"]["lineage_less_fraction_max"],
    )

    manifest: Dict[str, Any] = {
        "status": outcome,
        "outcome": outcome,
        "readout": readout,
        "metrics": dict(readout),
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "interpretation": interpretation,
        "evidence_asymmetry_per_half": evidence_asymmetry_per_half,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "per_arm_summary": per_arm_summary,
        "per_cell_results": rows,
        "arm_results": rows,
        "instrument_self_test": self_test,
        "diagnostics": {
            "hash_ledger_self_test": self_test,
            "lineage_predicate_preregistered": "e3_committed OR closure_committed",
            "lineage_predicate_broad_recorded": (
                "e3_committed OR closure_committed OR persistent_committed OR beta_elevated"
            ),
            "lineage_read_point": (
                "StepHarness on_action hook -- after select_action, before env.step and "
                "before update_residue (which tears down _committed_trajectory)"
            ),
        },
        "config": {
            "arms": ARMS,
            "waking_steps": waking_steps,
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "sleep_loop_episodes_K": 10_000_000,
            "use_sleep_aggregation_cluster": True,
            "use_cross_module_consolidation": True,
            "use_sleep_residue_integration": True,
            "use_offline_integration_gradient_step": True,
            "closure_on_flags": [
                "use_closure_operator", "use_lateral_pfc_analog",
                "use_closure_commit_entry", "use_closure_commit_beta_coupling",
                "use_natural_commit_latch_hold", "use_closure_commit_entry_trajectory",
            ],
            "max_authority_mutations": MAX_AUTHORITY_MUTATIONS,
            "max_lineage_less_fraction": MAX_LINEAGE_LESS_FRACTION,
        },
    }
    return manifest


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"] = CLAIM_IDS

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=(args.seeds if args.seeds is not None else DEFAULT_SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print("\nResult written to: %s" % out_path, flush=True)
    print("Status: %s" % result["status"], flush=True)
    print("final_outcome: %s" % result["outcome"], flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
