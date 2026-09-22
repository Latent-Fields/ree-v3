"""
V3-EXQ-1072 -- INV-024: offline consolidation / online commitment write-locus isolation audit


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
    E3 committed/closure trajectory state) must be BIT-IDENTICAL, while some
    REPRESENTATIONAL parameter changes (the cycle is live and touches only representation).
  ONLINE half -- every durable attribution write (ResidueField.accumulate with
    hypothesis_tag False) must be preceded by a commit event on the same trajectory
    (E3 committed_trajectory set, OR a closure commit intent).

=========================== DESIGN HISTORY -- READ BEFORE EDITING ===========================
THIS SINGLE-ARM DESIGN SUPERSEDES AN EARLIER USER DECISION FOR A TWO-ARM DESIGN. A later
reader must not read the reversal as drift, so both decisions and the measurement between
them are recorded here.

  Decision 1 (2026-09-22, user, option C): run TWO arms, CLOSURE_OFF and CLOSURE_ON, so that
  a FAIL would be ATTRIBUTABLE -- INV-024's online predicate accepts lineage from "E3
  committed_trajectory set, OR a closure commit intent", and the closure disjunct is dead at
  the WWA-named defaults. If lineage-less writes vanished under CLOSURE_ON the gap would be a
  closure-wiring matter; if they persisted, a genuine breach.

  MEASUREMENT that overturned it (Step 4.5 red-team, model fable, verdict BLOCKING; verified
  against live source by the authoring session AND independently by the orchestrator):
  the CLOSURE_ON arm could never have armed the latch. _closure_committed_trajectory is set
  at exactly one site (agent.py:9720) under the hard conjunct "self.goal_state is not None
  and self.goal_state.is_active()"; goal_state is built only when config.goal.z_goal_enabled
  is True (agent.py:3427), and from_dims defaults that False (config.py:7556). MEASURED: the
  two-arm smoke recorded closure_entry_ticks = 0 in CLOSURE_ON while
  closure_operator_present = True. The six-flag closure stack instantiated the ClosureOperator
  but tested the SAME single disjunct as CLOSURE_OFF.

  Decision 2 (2026-09-22, user, supersedes decision 1): DROP the closure arm. Run the single
  as-named configuration and register the closure disjunct as STRUCTURALLY UNTESTABLE at
  current defaults. Reason: arming the latch requires z_goal_enabled, which ALSO populates the
  SD-024 benefit terrain via accumulate_benefit and un-zeroes the SD-025 curiosity bonus in
  HippocampalModule._curiosity_bonus -- the arm would change three things to test one, and the
  measurement above shows we cannot establish how much configuration drift is enough. The
  untestability is carried by GFLAG-0410 as a substrate finding, not buried here.

  Decision 3 (2026-09-22, user, AMENDS THE CLAIM'S FALSIFIER): the online lineage predicate
  is the SD-084 PERSISTENT COMMITTED-PROGRAM HANDLE plus the BETA LATCH, not the transient
  E3 _committed_trajectory the what_would_answer originally named.

  MEASUREMENT that forced it (Step 4.5 red-team pass 2, verified by a real 400-step run
  through the arm_cell loop): _committed_trajectory is set only inside E3.select_action
  (e3_selector.py:4526) and torn down as the LAST statement of post_action_update
  (e3_selector.py:4908), while E3 runs only on an e3_tick -- cadence 10, or the step after a
  phase_reset, which update_residue triggers on every harm tick (agent.py:11062). So a
  per-step read of that handle measures "did E3 tick this step", NOT "was the agent
  committed". On a harm tick FOLLOWING a non-harm tick the agent is still committed -- the
  persistent handle and the beta latch are both set -- yet the transient handle reads None.
  MEASURED, 400 steps: seed 42 gives 2 such false lineage-less writes at ticks 82 and 370,
  seed 43 one at 340, seed 45 one at 182 -- and under the corrected predicate ALL THREE SEEDS
  give ZERO. Seed 42's first harm-after-non-harm tick is 82, seed 43's is 13, seed 45's is 42.

  THIS IS A CLAIM DEFECT, NOT A DRIVER PREFERENCE, and the discriminating fact is recorded in
  GFLAG-0411 rather than only here: of the three candidate repairs, TWO (persistent+beta, and
  score-only-e3_tick-steps) would have PASSED the run as measured while the registered
  predicate FAILED it. A predicate whose choice flips the verdict is not an implementation
  detail. The falsifier amendment is routed through governance -- claims.yaml is never
  hand-edited here.

  MEASURED AND vs OR: the conjunction (persistent AND beta) and the disjunction
  (persistent OR beta) give IDENTICAL results -- zero lineage-less writes on all three seeds
  at 400 steps -- so the conjunction/disjunction choice is not load-bearing at this operating
  point. The driver pre-registers the CONJUNCTION (the literal reading of "handle plus latch",
  and the stricter test) and RECORDS the disjunction plus the superseded E3 predicate
  alongside, so a later reader can re-derive any of the three without a re-run.

CONSEQUENCE FOR THE ONLINE CRITERION, stated so it cannot be misread: the closure disjunct
INV-024 also names is STRUCTURALLY DEAD on this configuration (decision 2), and
closure_entry_ticks is emitted (expected 0) as the standing witness. A lineage-less write here
means "the agent did not hold a committed program under an elevated beta gate at the moment
of the write".
=============================================================================================

TWO ONLINE MEASURES, BOTH RECORDED -- the load-bearing one is NOT the whole-run fraction
(user decision 2026-09-22). E3 commits iff _running_variance < commitment_threshold; rv starts
at precision_init 0.5 (config.py:1126), the bar is 0.40 (config.py:1124) and rv decays by the
EMA alpha 0.05 once per tick, so the best case is rv = 0.5 * 0.95^n: 0.5*0.95^4 = 0.407253 >
0.40 and 0.5*0.95^5 = 0.386890 < 0.40. NO commit is possible before tick 5, in any cell or
seed. Meanwhile REEAgent.update_residue (agent.py:11048-11058) writes durable residue on every
harm tick with no commit check. So a harm event in a cell's first five ticks is a
structurally lineage-less write, and a whole-run fraction would be decided by early
random-walk geometry rather than by write-locus isolation (MEASURED: the two-arm smoke's
first_commit_entry_tick = 5 and first_lineage_less_ticks = [0,1,2,3,4] in both arms).
Therefore:
  C2  (LOAD-BEARING) lineage_less_after_first_commit == 0 under the DECISION-3 predicate
      (persistent handle AND beta latch) -- writes at ticks where commitment was POSSIBLE.
      This tests BYPASS, which is what INV-024 is about.
  C2b (SECONDARY, not gating) the whole-run lineage_less_fraction, retained and reported so
      the warm-up gap stays legible in the manifest instead of being silently excluded.
C2 has its own vacuity guard: durable_writes_after_first_commit must exceed zero, or C2 would
pass on an empty denominator.

EVIDENCE ASYMMETRY -- REGISTERED PER HALF, NOT RUN-WIDE. The two halves carry very
different evidential weight and a reader must not average them:
  * OFFLINE half: a PASS is WEAK evidence for the CLAIM. ResidueField.integrate builds its
    optimizer over self.neural_field.parameters() ONLY, rbf_field is a disjoint submodule,
    and targets are computed under torch.no_grad() (residue/field.py:1122-1170), so
    no-erasure holds BY CONSTRUCTION. A PASS here confirms that the current implementation
    honours the isolation -- NOT that isolation is architecturally necessary. Treat it as a
    contract regression-guard. A FAIL would be very strong evidence.
  * ONLINE half: genuinely discriminating, and this is where a real FAIL can come from.
    There are TWO durable-write paths. e3_selector.py:4664 IS commitment-gated by
    construction. But agent.py:11051 (REEAgent.update_residue) is gated only on
    `owned and not hypothesis_tag and self._current_latent is not None`, where
    `owned: bool = True` is a parameter with default True (agent.py:10873) and NO commit
    state is consulted at that site at all.
Emitted as `evidence_asymmetry_per_half`, with a governance_note forbidding averaging.

LINEAGE READ POINT = COMMIT ENTRY, NOT WRITE TIME (instrumentation correctness, ratified).
`self._committed_trajectory = None` is the LAST statement of E3.post_action_update, and
REEAgent.update_residue calls post_action_update BEFORE its own durable write. So at the
instant of the write the E3 commit latch is ALWAYS None by construction, and a naive
write-time read would report ~100% lineage-less as a pure ordering artifact. This driver
snapshots commit state in the StepHarness `on_action` hook (after select_action, before
env.step and before update_residue) and attributes each durable write to THAT tick's snapshot.

TWO AUTHORITY STORES ARE STRUCTURALLY CONSTANT and are reported as such rather than counted
as discriminating witnesses. e3_committed_trajectory is torn down by post_action_update, and
e3_closure_trajectory can never arm (see DESIGN HISTORY), so both read False pre AND post by
construction. They are still hashed -- INV-024's what_would_answer names them -- but
`authority_store_witness_classes` marks them structurally_constant and
`authority_stores_discriminating` reports the honest witness count.

THE PAIRING TRAP (both flags required, not one). phase_manager.py:679 gates the WRITEBACK
integrate() CALL on use_sleep_residue_integration; whether that call TRAINS is
ResidueConfig.use_offline_integration_gradient_step. With only the latter the offline call
never fires and the WWA's non-degeneracy precondition is unsatisfiable (a vacuous run). Both
are pinned ON here and both are asserted live in the readiness preconditions.

Pre-registered acceptance (NOT derived from this run's statistics):
  C1  offline_authority_stores_unmutated -- every audited authority-store hash is identical
      pre- and post-cycle, every seed. (load-bearing)
  C2  online_lineage_complete_after_first_commit -- zero durable writes lacking commit lineage
      at ticks after the first commit entry, every seed. (load-bearing)
  C2b online_lineage_complete_whole_run -- whole-run fraction. (SECONDARY, not gating)
  PASS iff the readiness gate is green AND C1 AND C2.

Falsifying per INV-024: one observed authority-store mutation during a cycle, or one durable
online write with no commit lineage. One confirmed instance refutes the invariant AS
IMPLEMENTED and routes to MECH-067 (the enforcement mechanism), not to a re-run.

Red-team: Step 4.5 pass 1 (fable) BLOCKING -> fixed by user decisions 1-2 plus the C3/C4
fixes. Pass 2 (fable) BLOCKING on the online half -> fixed by user decision 3 (the falsifier
amendment above), which is the repair the finding itself pointed at. Pass-2 Attack 1
("sws_enabled/rem_enabled are False so the cycle is inert") was DISMISSED with citation:
use_sleep_aggregation_cluster=True invokes enable_sleep_aggregation_cluster() from
REEConfig.__post_init__ (config.py:7347), which sets both True -- measured True on this
driver's own config, 43 cycle metrics, mech018_residue_integration_fired = 1.0. Reading
_build_config literally misses the resolver.

SMOKE LENGTH IS A CORRECTNESS REQUIREMENT HERE, not a formality. The pass-1 60-step smoke was
green only because seed 42's first harm-after-non-harm tick lands at 82 -- structurally blind
to the regime that breaks the superseded predicate, by accident. --dry-run therefore runs TWO
seeds at SMOKE_STEPS and ASSERTS that the regime was actually reached (see _assert_smoke_
coverage), printing the exercised tick indices rather than assuming them.

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
DEFAULT_SEEDS = [42, 43, 45, 46, 47]  # 44 excluded per CLAUDE.md reef-config instability
                                      # precedent. WWA asks >=3; the audit is cheap (one
                                      # cycle per seed, no training beyond warm-up) and
                                      # INV-024 falsifies on ONE instance, so more seeds is
                                      # strictly more chances to catch a rare violation.
WAKING_STEPS = 400             # waking steps per cell -- denominator of the `ep N/M` prints
GRID_SIZE = 8
NUM_HAZARDS = 2
NUM_RESOURCES = 3
MAX_EPISODE_STEPS = 200
# SINGLE ARM -- the closure arm was dropped by user decision 2 (see DESIGN HISTORY).
ARMS = ["ASNAMED"]
# Smoke geometry is a CORRECTNESS requirement (see docstring). Seed 42's first
# harm-after-non-harm tick is 82, seed 43's is 13, seed 45's is 42 (all MEASURED at 400
# steps), so 150 steps on two seeds provably reaches the regime on both -- and the driver
# ASSERTS it rather than trusting these numbers.
SMOKE_STEPS = 150
SMOKE_SEEDS = 2

# ---- pre-registered thresholds (constants; NOT derived from this run) ----
MAX_AUTHORITY_MUTATIONS = 0      # C1: zero mutated authority stores
MAX_LINEAGE_LESS_AFTER_COMMIT = 0   # C2 (LOAD-BEARING): zero lineage-less durable writes
                                    # at ticks where commitment was POSSIBLE
MAX_LINEAGE_LESS_FRACTION = 0.0  # C2b (SECONDARY, not gating): whole-run fraction
MIN_ACTIVE_CENTERS = 0.0         # readiness floor (strict >): residue field must be populated
MIN_COMMIT_ENTRIES = 0.0         # readiness floor (strict >): commitment must have occurred
MIN_DURABLE_WRITES = 0.0         # readiness floor (strict >): the online DV must have a denominator
MIN_WRITES_AFTER_COMMIT = 0.0    # readiness floor (strict >): C2 vacuity guard -- without
                                 # writes after the first commit, C2 passes on an empty set
MIN_REPR_DELTA = 0.0             # readiness floor (strict >): the cycle must be LIVE


# --------------------------------------------------------------------------------------
# hashing / snapshots
# --------------------------------------------------------------------------------------
def _has_commit_lineage(w: Dict[str, Any]) -> bool:
    """PRE-REGISTERED online lineage predicate (user decision 3, 2026-09-22).

    "Committed" = the SD-084 PERSISTENT committed-program handle is held AND the beta gate is
    elevated. NOT E3's transient _committed_trajectory, which is torn down as the last
    statement of post_action_update and only ever re-set on an e3_tick, so a per-step read of
    it reports "did E3 tick this step" rather than "was the agent committed" (MEASURED: it
    produces false lineage-less writes at ticks 82/370 on seed 42). The superseded predicate
    and the disjunctive variant are both RECORDED alongside, so any of the three can be
    re-derived from the manifest without a re-run.
    """
    return bool(w.get("persistent_committed")) and bool(w.get("beta_elevated"))


def _has_superseded_e3_lineage(w: Dict[str, Any]) -> bool:
    """The predicate INV-024's what_would_answer originally named. Recorded, NOT gating."""
    return bool(w.get("e3_committed")) or bool(w.get("closure_committed"))


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


# C4 fix: these two read False pre AND post by construction -- e3_committed_trajectory is
# torn down by post_action_update, and e3_closure_trajectory can never arm at these defaults
# (DESIGN HISTORY). They are still hashed because INV-024's what_would_answer names them, but
# they are NOT discriminating witnesses and must not inflate the audited count.
STRUCTURALLY_CONSTANT_STORES = ("e3_committed_trajectory", "e3_closure_trajectory")


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
        "max_lineage_less_after_commit": MAX_LINEAGE_LESS_AFTER_COMMIT,
        "max_lineage_less_fraction": MAX_LINEAGE_LESS_FRACTION,
    }
    # NO closure flags: user decision 2 dropped that arm (see DESIGN HISTORY). The closure
    # disjunct is structurally untestable at these defaults and GFLAG-0410 carries the reason.
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
    harm_ticks: List[int] = []
    harm_after_nonharm_ticks: List[int] = []
    prev_harm = False
    for step_index in range(waking_steps):
        tick_state["index"] = step_index
        result = harness.step(obs)
        is_harm = bool(result.harm_signal < 0)
        if is_harm:
            harm_ticks.append(step_index)
            # THE REGIME THAT BROKE THE SUPERSEDED PREDICATE: harm on a tick whose predecessor
            # was not harm, so no phase_reset forced an e3_tick and the transient handle is
            # None while the agent is still committed. Coverage of this is asserted in --dry-run.
            if step_index > 0 and not prev_harm:
                harm_after_nonharm_ticks.append(step_index)
        prev_harm = is_harm
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
    lineage_less = [w for w in writes if not _has_commit_lineage(w)]
    lineage_less_superseded = [w for w in writes if not _has_superseded_e3_lineage(w)]
    lineage_less_fraction = (len(lineage_less) / durable_writes) if durable_writes else 0.0
    # DECISIVE DIAGNOSTIC: is a lineage-less write a STARTUP artifact or an ONGOING breach?
    # E3 selects on a cadence (heartbeat.e3_steps_per_tick, default 10), so no commitment can
    # exist before the first E3 tick and a harm event in that prefix necessarily writes
    # without lineage. Writes AFTER the first observed commit entry cannot be explained that
    # way. Recording both lets governance separate the two readings without a re-run; the
    # NOTE: C2 is the after-first-commit measure; the whole-run fraction is C2b (secondary).
    commit_ticks = [w.get("tick", -1) for w in writes if _has_commit_lineage(w)]
    first_commit_tick = min(commit_ticks) if commit_ticks else None
    if first_commit_tick is None:
        lineage_less_after_first_commit = len(lineage_less)
        durable_writes_after_first_commit = durable_writes
    else:
        lineage_less_after_first_commit = sum(
            1 for w in lineage_less if int(w.get("tick", -1)) > int(first_commit_tick)
        )
        # C2's DENOMINATOR, and its vacuity guard: without writes after the first commit the
        # load-bearing criterion would pass on an empty set.
        durable_writes_after_first_commit = sum(
            1 for w in writes if int(w.get("tick", -1)) > int(first_commit_tick)
        )
    # Generous recording: a BROADER predicate a later reader might prefer, banked so the
    # alternative need not be re-run. Not the pre-registered criterion.
    lineage_less_broad = [
        w for w in writes
        if not (w.get("persistent_committed") or w.get("beta_elevated"))
    ]
    # Superseded-predicate counterpart of C2, recorded so the amendment's effect is auditable
    # from the manifest: this is the number the ORIGINAL what_would_answer would have scored.
    if first_commit_tick is None:
        lineage_less_superseded_after_commit = len(lineage_less_superseded)
    else:
        lineage_less_superseded_after_commit = sum(
            1 for w in lineage_less_superseded
            if int(w.get("tick", -1)) > int(first_commit_tick)
        )

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
    # C4: separate the honest witness count from the raw audited count.
    discriminating_stores = sorted(
        k for k in pre_authority if k not in STRUCTURALLY_CONSTANT_STORES
    )
    witness_classes = {
        k: ("structurally_constant_by_construction"
            if k in STRUCTURALLY_CONSTANT_STORES else "discriminating")
        for k in sorted(pre_authority)
    }
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
        and lineage_less_after_first_commit <= MAX_LINEAGE_LESS_AFTER_COMMIT
    )
    print("verdict: %s seed=%d arm=%s mutated=%d ll_after_commit=%d ll_whole_run=%.4f"
          % ("PASS" if cell_pass else "FAIL", seed, arm, len(mutated_stores),
             lineage_less_after_first_commit, lineage_less_fraction), flush=True)

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
        "durable_writes_after_first_commit": int(durable_writes_after_first_commit),
        "harm_events_start": float(harm_events_start),
        "harm_events_end": float(harm_events_end),
        "harm_events_delta": float(harm_events_end - harm_events_start),
        "harm_ticks_count": int(len(harm_ticks)),
        "harm_after_nonharm_ticks_count": int(len(harm_after_nonharm_ticks)),
        "harm_after_nonharm_ticks": [int(t) for t in harm_after_nonharm_ticks[:24]],
        "lineage_less_superseded_predicate": int(len(lineage_less_superseded)),
        "lineage_less_superseded_after_first_commit": int(lineage_less_superseded_after_commit),
        # offline half
        "authority_stores_audited": int(len(pre_authority)),
        "authority_stores_discriminating": int(len(discriminating_stores)),
        "authority_store_witness_classes": witness_classes,
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
            name="durable_writes_after_first_commit",
            description=(
                "C2 VACUITY GUARD: durable writes occurred at ticks AFTER the first commit "
                "entry, so the load-bearing criterion has a non-empty denominator. Without "
                "this C2 passes trivially on a run whose every write predates commitment"
            ),
            control="audited non-hypothesis accumulate calls at ticks > first_commit_entry_tick",
            threshold=MIN_WRITES_AFTER_COMMIT, direction="lower", kind="readiness",
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
        "durable_writes_after_first_commit": float(
            min(r["durable_writes_after_first_commit"] for r in rows)
        ),
        "offline_integration_call_fired_and_trains": float(
            min(r["mech018_integration_fired"] * r["mech018_residue_trains"] for r in rows)
        ),
    }


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
class SmokeCoverageError(AssertionError):
    """The smoke did not reach the regime that breaks the superseded predicate."""


def _assert_smoke_coverage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A --dry-run that never sees a harm tick FOLLOWING a non-harm tick proves nothing about
    the online criterion. Pass-1's 60-step smoke was green purely because seed 42's first such
    tick is 82. So ASSERT the regime was reached, on every smoke seed, rather than assuming it
    from the measured tick numbers -- those are a design input, not a runtime guarantee.
    """
    report = {
        r["seed"]: {
            "harm_after_nonharm_ticks_count": r["harm_after_nonharm_ticks_count"],
            "harm_after_nonharm_ticks": r["harm_after_nonharm_ticks"],
            "first_commit_entry_tick": r["first_commit_entry_tick"],
            "durable_writes_after_first_commit": r["durable_writes_after_first_commit"],
        }
        for r in rows
    }
    for r in rows:
        print("  [smoke-coverage] seed=%d harm_after_nonharm=%d ticks=%s"
              % (r["seed"], r["harm_after_nonharm_ticks_count"],
                 r["harm_after_nonharm_ticks"][:10]), flush=True)
    barren = [r["seed"] for r in rows if r["harm_after_nonharm_ticks_count"] < 1]
    if barren:
        raise SmokeCoverageError(
            "smoke did not reach a harm tick following a non-harm tick on seed(s) %s -- it is "
            "structurally blind to the regime the online criterion is about. Raise SMOKE_STEPS "
            "or change SMOKE_SEEDS; do NOT read this smoke as evidence about C2." % barren
        )
    starved = [r["seed"] for r in rows if r["durable_writes_after_first_commit"] < 1]
    if starved:
        raise SmokeCoverageError(
            "no durable writes after the first commit entry on seed(s) %s -- C2's denominator "
            "is empty and the smoke cannot exercise it." % starved
        )
    return report


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)
    waking_steps = WAKING_STEPS
    if dry_run:
        # TWO seeds, not one: the blind spot is seed-dependent and a single seed can be
        # accidentally green. Length chosen so the regime is reachable on both, and ASSERTED
        # below rather than assumed.
        seeds = seeds[:SMOKE_SEEDS]
        waking_steps = SMOKE_STEPS

    print("[V3-EXQ-1072] INV-024 offline/online isolation audit", flush=True)
    print("  Arms: %s  Seeds: %s  Waking steps/cell: %d"
          % (ARMS, seeds, waking_steps), flush=True)
    print("  Output: REE_assembly/evidence/experiments/%s/" % EXPERIMENT_TYPE, flush=True)

    self_test = _hash_ledger_self_test()
    print("  [instrument] hash ledger self-test: %s" % self_test, flush=True)

    specs = _precondition_specs()
    arm_contexts = {arm: {"arm": arm, "id": arm, "closure_plane": False}
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

    smoke_coverage = _assert_smoke_coverage(rows) if dry_run else None

    # --- per-arm readiness gates (never AND the whole run -- V3-EXQ-785) --------------
    arm_gates = []
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arm_gates.append(evaluate_arm_gate(
            arm, arm_contexts[arm], specs, _arm_measured(arm_rows, self_test),
        ))
    aggregate = aggregate_arm_gates(arm_gates)

    # --- pre-registered criteria -----------------------------------------------------
    # C3 fix: score over GREEN-ARM rows only. A red arm's readouts are artifacts
    # (precondition_gate.py:466-471) and must not drive a claim verdict. With no green arm
    # the gate below routes to substrate_not_ready_requeue and the criteria are not read.
    green_arms = set(aggregate["green_arms"])
    scored_rows = [r for r in rows if r["arm"] in green_arms] or rows
    excluded_rows = [r for r in rows if r["arm"] not in green_arms]

    worst_mutations = max(r["authority_stores_mutated"] for r in scored_rows)
    worst_lineage_less_after_commit = max(
        r["lineage_less_after_first_commit"] for r in scored_rows
    )
    worst_lineage_less = max(r["lineage_less_fraction"] for r in scored_rows)
    c1 = bool(worst_mutations <= MAX_AUTHORITY_MUTATIONS)
    c2 = bool(worst_lineage_less_after_commit <= MAX_LINEAGE_LESS_AFTER_COMMIT)
    # SECONDARY, deliberately NOT in the PASS rule (user decision 2026-09-22).
    c2b = bool(worst_lineage_less <= MAX_LINEAGE_LESS_FRACTION)

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
            "lineage_less_after_first_commit_max": max(
                r["lineage_less_after_first_commit"] for r in arm_rows
            ),
            "durable_writes_after_first_commit_total": sum(
                r["durable_writes_after_first_commit"] for r in arm_rows
            ),
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
        "c2_online_lineage_complete_after_first_commit": 1.0 if c2 else 0.0,
        "c2b_online_lineage_complete_whole_run": 1.0 if c2b else 0.0,
        "lineage_less_after_first_commit_max": float(worst_lineage_less_after_commit),
        "lineage_less_superseded_after_first_commit_max": float(
            max(r["lineage_less_superseded_after_first_commit"] for r in rows)
        ),
        "harm_after_nonharm_ticks_min": float(
            min(r["harm_after_nonharm_ticks_count"] for r in rows)
        ),
        "durable_writes_after_first_commit_min": float(
            min(r["durable_writes_after_first_commit"] for r in rows)
        ),
        "durable_writes_after_first_commit_total": float(
            sum(r["durable_writes_after_first_commit"] for r in rows)
        ),
        "authority_stores_discriminating_min": float(
            min(r["authority_stores_discriminating"] for r in rows)
        ),
        "closure_entry_ticks_total": float(sum(r["closure_entry_ticks"] for r in rows)),
        "n_cells_excluded_by_red_gate": float(len(excluded_rows)),
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
            "name": "C2_online_lineage_complete_after_first_commit",
            "load_bearing": True,
            "passed": c2,
            "measured": float(worst_lineage_less_after_commit),
            "threshold": float(MAX_LINEAGE_LESS_AFTER_COMMIT),
            "comparator": "<=",
            "direction": "upper",
            "measured_note": (
                "worst green-arm cell: durable writes lacking commit lineage at ticks AFTER "
                "the first commit entry, i.e. where commitment was POSSIBLE. This tests "
                "BYPASS rather than the E3 precision warm-up"
            ),
        },
        {
            "name": "C2b_online_lineage_complete_whole_run",
            "load_bearing": False,
            "passed": c2b,
            "measured": float(worst_lineage_less),
            "threshold": float(MAX_LINEAGE_LESS_FRACTION),
            "comparator": "<=",
            "direction": "upper",
            "measured_note": (
                "SECONDARY, NOT GATING (user decision 2026-09-22): the whole-run lineage-less "
                "fraction, retained so the pre-commit warm-up gap stays legible rather than "
                "being silently excluded. Expected non-zero whenever a harm event lands in a "
                "cell's first five ticks, which is E3 precision warm-up, not isolation"
            ),
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
            "C2_online_lineage_complete_after_first_commit": bool(
                min(r["durable_writes_after_first_commit"] for r in rows) > 0
                and min(r["commit_entry_ticks"] for r in rows) > 0
            ),
            "C2b_online_lineage_complete_whole_run": bool(
                min(r["durable_writes"] for r in rows) > 0
                and min(r["commit_entry_ticks"] for r in rows) > 0
            ),
        },
        "criteria": criteria,
        "combination_rule": (
            "PASS = readiness gate green (any arm green; never ANDed whole-run) AND "
            "C1_offline_authority_stores_unmutated AND "
            "C2_online_lineage_complete_after_first_commit. "
            "C2b_online_lineage_complete_whole_run is RECORDED BUT NOT GATING (user decision "
            "2026-09-22): it is expected to fail whenever a harm event lands before the first "
            "possible commit at tick 5, which is E3 precision warm-up rather than a write-locus "
            "breach. Both gating criteria are worst-cell over GREEN-ARM cells only, so a single "
            "violating cell fails the run -- INV-024's own 'one confirmed instance refutes the "
            "invariant AS IMPLEMENTED'."
        ),
        "dv_symmetry_note": (
            "ASNAMED (single arm): the DV is a content hash of the authority stores plus a "
            "count of durable writes lacking a commit-entry snapshot. Its symmetry group is "
            "permutation of write ORDER within a tick; the manipulation (running one full "
            "sleep cycle) is NOT invariant under it -- a cycle that wrote an authority store "
            "would change the hash regardless of ordering, and the ledger's sensitivity to a "
            "1e-6 perturbation is MEASURED in P0 rather than assumed. The DV is not a "
            "broadcast constant, a monotone rescaling, or a set-aggregate over interchangeable "
            "units. The online count is likewise not invariant: a write that acquires commit "
            "lineage leaves the numerator while staying in the denominator."
        ),
        "closure_disjunct_untestable": (
            "INV-024's online predicate accepts lineage from an E3 commit OR a closure commit "
            "intent. The SECOND DISJUNCT IS STRUCTURALLY DEAD at these defaults: "
            "_closure_committed_trajectory is set only at agent.py:9720 under "
            "'goal_state is not None and goal_state.is_active()', goal_state needs "
            "config.goal.z_goal_enabled (agent.py:3427), and from_dims defaults it False "
            "(config.py:7556). closure_entry_ticks is emitted (expected 0) as the standing "
            "witness. A two-arm design that enabled the closure plane was tried and dropped by "
            "user decision -- arming the latch also populates SD-024 benefit terrain and "
            "un-zeroes the SD-025 curiosity bonus, changing three things to test one. The "
            "untestability is registered as a substrate finding under GFLAG-0410. CONSEQUENCE: "
            "a lineage-less write here means 'not preceded by an E3 commit'; it does not rule "
            "out that a closure commit intent would have licensed it on another configuration."
        ),
        "scored_rows_note": (
            "Criteria are scored over GREEN-ARM cells only; cells excluded by a red readiness "
            "gate are reported in per_cell_results but do not drive the verdict."
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
**Purpose:** evidence (INV-024). Single as-named config x {nseeds} seeds = {ncells} cells.

- C1 offline authority stores unmutated: **{c1}** (worst cell {mut}, threshold {c1t})
- C2 online lineage complete AFTER first commit (LOAD-BEARING): **{c2}** (worst cell {llac}, threshold {c2t})
- C2b whole-run lineage-less fraction (SECONDARY, not gating): {c2b} (worst cell {ll:.4f})
- readiness gate green: {green}
- durable writes audited: {dw} | after first commit: {dwac} | lineage-less (whole run): {llw}
- closure_entry_ticks total: {cet} (expected 0 -- the closure disjunct is structurally dead)

**C2 is deliberately NOT the whole-run fraction.** No E3 commit is possible before tick 5
(rv = 0.5*0.95^n against a 0.40 bar), so a harm event in a cell's first five ticks is a
structurally lineage-less write. C2 therefore scores only writes at ticks where commitment
was POSSIBLE -- testing bypass, not scheduling. C2b keeps the warm-up gap visible.

**Evidence asymmetry is registered PER HALF** -- see `evidence_asymmetry_per_half`. An
overall PASS is a contract regression-guard on the offline half (construction-guaranteed)
plus a moderate live confirmation on the online half. Do not average them, and do not read a
PASS as confirmation that isolation is architecturally necessary.

See `interpretation.closure_disjunct_untestable` for why the second lineage disjunct could
not be tested here, and `per_cell_results` for the full per-seed table with the pre/post
authority-store hash snapshots.
""".format(
        outcome=outcome, label=label, nseeds=len(seeds), ncells=len(rows),
        c1=c1, mut=worst_mutations, c1t=MAX_AUTHORITY_MUTATIONS,
        c2=c2, llac=worst_lineage_less_after_commit, c2t=MAX_LINEAGE_LESS_AFTER_COMMIT,
        c2b=c2b, ll=worst_lineage_less, green=gate_green,
        dw=int(readout["durable_writes_total"]),
        dwac=int(readout["durable_writes_after_first_commit_total"]),
        llw=int(readout["lineage_less_writes_total"]),
        cet=int(readout["closure_entry_ticks_total"]),
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
            "lineage_predicate_preregistered": (
                "persistent_committed AND beta_elevated (SD-084 persistent committed-program "
                "handle held under an elevated beta gate) -- user decision 3, 2026-09-22, "
                "amending INV-024's what_would_answer via GFLAG-0411"
            ),
            "lineage_predicate_superseded_recorded": (
                "e3_committed OR closure_committed -- the predicate the what_would_answer "
                "originally named; recorded, NOT gating. It reports 'did E3 tick this step' "
                "rather than 'was the agent committed'"
            ),
            "lineage_predicate_disjunctive_recorded": (
                "persistent_committed OR beta_elevated -- MEASURED identical to the "
                "pre-registered conjunction (zero lineage-less on all three seeds at 400 "
                "steps), so the AND/OR choice is not load-bearing at this operating point"
            ),
            "smoke_coverage": smoke_coverage,
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
            "closure_plane_enabled": False,
            "closure_arm_dropped_by_user_decision": (
                "2026-09-22 decision 2 SUPERSEDES the earlier option-C two-arm decision: the "
                "CLOSURE_ON arm could never arm the closure latch (measured "
                "closure_entry_ticks=0), and arming it needs z_goal_enabled, which also "
                "populates SD-024 benefit terrain and un-zeroes the SD-025 curiosity bonus. "
                "Registered as a substrate finding under GFLAG-0410."
            ),
            "max_authority_mutations": MAX_AUTHORITY_MUTATIONS,
            "max_lineage_less_after_commit": MAX_LINEAGE_LESS_AFTER_COMMIT,
            "max_lineage_less_fraction_secondary": MAX_LINEAGE_LESS_FRACTION,
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
