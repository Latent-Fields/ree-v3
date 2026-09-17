#!/opt/local/bin/python3
"""
V3-EXQ-1052 -- MECH-066 / MECH-067: commit-boundary write-locus AUDIT SPIKE

SLEEP DRIVER: K=1 single-fire (SleepLoopManager, fires every episode)

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------------------------------------
MECH-067's own `what_would_answer` (claims.yaml, authored 2026-09-16) splits
its own test in two and says which half is buildable:

    "SUBSTRATE: the AUDIT half is buildable now as a driver-side instrument
     wrapping ResidueField.accumulate / discharge_domain, BetaGate.elevate /
     release, AnchorSet writes, and the SleepLoopManager._run_cycle parameter
     deltas (hash before / after), with phase read from SleepPhase and actor
     from the call site. The MATRIX itself (an explicit (phase, store, actor)
     default-deny table with typed store classes POL / ID / CAPS / residue /
     attribution ledger / scratch) is NOT built in ree_core.
     substrate_conditional for the comparator arm; the audit is the spike that
     decides whether the build is owed. Do not invent a matrix DV in the
     meantime."

    "Disposition 2026-09-16: complex (probe-gated) -> the audit spike first; if
     it finds violations, /implement-substrate the matrix and re-run as the
     comparator."

THIS SCRIPT IS THE AUDIT HALF ONLY. The matrix-enforced COMPARATOR arm is NOT
built and NOT queued: a pre-flight grep over all of `ree_core/` (2026-09-17)
confirmed the claim's SUBSTRATE note is still true -- there is no
(phase, store, actor) table, no store-class vocabulary (no POL / ID / CAPS /
attribution-ledger / scratch enum), and no actor vocabulary anywhere in the
tree. The only Enum classes in `ree_core` are `SleepPhase`, `ControlDemandType`
and `ChunkState`, none of which is a write-authority concept. Per the claim's
own instruction, this run invents no matrix DV.

WHY BOTH CLAIMS ARE TAGGED BY ONE RUN. MECH-066 is the general separation
principle ("pre-commit and post-commit channels may share representations but
must stay separated at durable write boundaries"); MECH-067 is the refinement
that a machine-checkable matrix is REQUIRED to enforce it. The audit produces
evidence for both from one instrument: the violation census speaks to whether
separation actually holds at the write boundary (MECH-066), and the
corruption-on-weakening leg speaks to whether local gates suffice or a matrix
is required (MECH-067). Queueing them separately would run the same instrument
twice; queueing MECH-066 alone is not possible -- it carries no falsifier of its
own (no `what_would_answer` at all).

FIRST V3 READ. MECH-066 has ZERO evidence entries of any kind in
claim_evidence.v1.json. MECH-067 has exactly two, both
architecture_epoch `ree_v1_minimal_genuine_v1` and both carrying
`scoring_excluded: "stale_epoch"`; the earlier of the two was additionally
adjudicated a measurement/test-design defect (underpowered 1-seed/5-episode
pilot) by the confirmed
failure_autopsy_grandfathered-r5-legacy-provenance-sweep_2026-08-08. So this is
a first real V3 read, not a re-run of banked V1 work. The V1 EXQ-005 PASS is
evidence for the write-locus DISTINCTION (MECH-060), not for the matrix being
required -- the claim says so itself.

THE ONE STRUCTURAL FINDING THAT SHAPED THE DESIGN
---------------------------------------------------------------------------
A pre-flight call-site trace (2026-09-17) found that NONE of
`ResidueField.accumulate` / `discharge_domain`, `BetaGate.elevate` / `release`,
or any `AnchorSet` writer is reachable from inside `SleepLoopManager._run_cycle`
-- every call site resolves to `agent.py`, `hippocampal/module.py` or
`governance/closure_operator.py`, all waking. So a call-site counter can NEVER
observe a sleep-phase write, and a coverage precondition written as "count
sleep-phase calls >= 1" would be structurally unsatisfiable and would vacate
the whole run (the V3-EXQ-785 shape).

That is exactly why the claim's SUBSTRATE clause says to instrument sleep with
"`_run_cycle` parameter deltas (hash before / after)" rather than call hooks.
This script therefore uses TWO instruments, and says which is which:

  * CALL-SITE WRAPPERS for the waking / simulation stores (residue, beta_gate,
    anchor_set), recording (phase, store, actor) per write; and
  * BEFORE/AFTER STATE HASHES around each sleep cycle, which is the only
    instrument that can detect an offline authority write at all.

A sleep cell reading "no call-site hits but hash unchanged" is therefore the
EXPECTED, non-violating reading, and is recorded as covered-by-hash, not as a
coverage failure.

ARMS
---------------------------------------------------------------------------
ARM_INTACT          every local gate as shipped. The census arm.
ARM_MECH094_WEAKENED  the MECH-094 refusal inside ResidueField.accumulate is
                    bypassed by the driver (hypothesis-tagged rollout content
                    is allowed to reach residue). This is the claim's
                    FALSIFYING clause made executable: "deliberately weakening
                    one local gate is caught by another (redundancy) or
                    produces no attribution corruption". The weakening is
                    driver-side only -- `ree_core` is not modified.

PRE-REGISTERED CRITERIA (thresholds are constants below, set before any run)
---------------------------------------------------------------------------
C1 (LOAD-BEARING, MECH-067's 'required' clause)
    ARM_INTACT violation count >= 1.
    PASS  -> separation leaks under the shipped gates; a default-deny matrix is
             CONFIRMED as required, not merely tidy.
    FAIL  -> zero violations under full coverage; the local gates are
             SUFFICIENT on this evidence and MECH-067's 'matrix is required'
             is over-strong on this run.
C2 (GRADING, not load-bearing -- see `criteria` and combination_rule)
    Under ARM_MECH094_WEAKENED, residue mass per real harm event rises by at
    least CORRUPTION_RATIO_FLOOR x relative to ARM_INTACT. This is the V1
    EXQ-005 statistic (which saw ~46x). It distinguishes "a leak that corrupts
    attribution" from "a leak nothing downstream cares about".
C3 (REDUNDANCY, not load-bearing)
    Whether any OTHER gate caught the weakened write (recorded as a count of
    blocked-elsewhere events). Informational: redundancy is one of the two
    FALSIFYING routes and must be recorded to be readable.

NON-DEGENERACY / COVERAGE PRECONDITION (the claim's own, verbatim in spirit):
    "the write audit must have COVERAGE -- it must observe at least one write
     in every (phase, store) class it claims to police".
Implemented as a per-cell coverage table with an explicit `applies_to` scoping
so that the structurally-unreachable sleep call-site cells are SCOPED OUT
rather than failed (see `experiments/_lib/precondition_gate.py` rationale).
A cell that is genuinely uncovered fails the gate and the run self-routes
`substrate_not_ready_requeue` -- never a substrate verdict.

red-team: see the queue entry note for the verdict and model.

=============================================================================
STATUS 2026-09-17: **BLOCKED AT /queue-experiment Step 4.5. DO NOT QUEUE.**
NOT queued, NO EXQ id consumed as a live queue entry. The reserved slot claim
for V3-EXQ-1052 was closed --not-landed.

Red-team adversarial design review (Opus; the Fable spawn hit a model spend
limit and was re-spawned once on the session model per the skill) returned
BLOCKING. Every finding below was VERIFIED against source by the authoring
session before this header was written -- these are confirmed defects, not
reviewer conjecture:

1. `_state_hash` hashes a CONSTANT for anchors. It reads
   `getattr(anchors, "_anchors", {})`, but AnchorSet stores its anchors in
   `self._active` / `self._all` (anchor_set.py:201,204) and defines no
   `_anchors` at all -- CONFIRMED by inspecting AnchorSet.__init__. The
   getattr default fires every call, so the anchor term is the literal '[]'
   in both arms on every cycle.
2. TWO sleep cycles run per episode and the instrument watches the wrong one.
   `run_episode` calls `agent.reset()` (_harness.py:44), which calls
   `sleep_loop.notify_episode_end(self)` (agent.py:3480) -- CONFIRMED -- and
   with sleep_loop_episodes_K=1 that fires a full cycle at episode end, while
   `audit.phase` is "waking" and outside the before/after hash window. The
   driver's own force_cycle is then a SECOND, extra cycle, so `sleep_cycles`
   under-counts 2x and the naturally-fired cycle is unobserved.
3. C1 has exactly one live route in the intact arm and it is blind. Every
   other `_flag_violation` site is dead by construction there, so C1 rests
   entirely on the hash -- which does not cover `e1.context_memory`, even
   though `run_sws_schema_pass` deliberately lifts the offline gate
   (agent.py:12547) and writes ContextMemory slots (agent.py:12674). That is
   plausibly the very violation shape MECH-067 describes, and it is invisible.
4. Both configurations MECH-067's FALSIFYING clause names are OFF:
   `use_sleep_aggregation_cluster` and `use_cross_module_consolidation` both
   default False and are never set, so offline_gradient_pass on e2_harm_s and
   E1/E2 cross-module consolidation -- the sleep writes the claim's coverage
   clause enumerates -- structurally never happen.
5. C2 is unreachable. Ceiling on the corruption ratio is ~1.4-1.8 given the
   injection rate and magnitude vs genuine waking writes, against a
   pre-registered floor of 2.0, so `evidence_direction: "supports"` cannot be
   produced under any outcome.
6. Coverage certifies the HOOKS, not the stores: only 1 of 4 residue writers
   is wrapped (`accumulate_benefit` / `accumulate_safety` / `update_valence`
   are not), and two of five in-scope cells are driver-manufactured.

Recording `0 violations` as `weakens` against two claims that have NO prior V3
manifests would bank an instrument artefact as a demotion signal. That is why
this refuses rather than ships.
=============================================================================
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiment_protocol import emit_outcome

EXPERIMENT_PURPOSE = "diagnostic"
QUEUE_ID = "V3-EXQ-1052"
EXPERIMENT_TYPE = "v3_exq_1052_mech066_mech067_write_locus_audit_spike"
CLAIM_IDS = ["MECH-066", "MECH-067"]

# The one anchor-kind readiness precondition here is `sleep_cycles_observed >= 1`,
# whose control is an UNCONDITIONAL `SleepLoopManager.force_cycle(agent)` call
# once per episode -- it is reachable by construction, not by a hand-written
# predicate that could be narrower than the state it anchors to. Measured
# end-to-end at full scale before queueing: 12 cycles over the 2-arm x 1-seed
# pilot (2026-09-17). There is no instrument-specification gap for this gate to
# mislabel, which is the condition the exemption is written for.
ANCHOR_REACHABILITY_EXEMPT = (
    "sleep_cycles_observed anchors on an unconditional force_cycle call, one per "
    "episode; reachable by construction and measured at 12 in the pre-queue pilot")

# ---- pre-registered constants (set before any run; never derived from data) ----
SEEDS = [0, 1, 2]
EPISODES = 6
STEPS_PER_EPISODE = 120
EPISODES_PER_RUN = EPISODES  # denominator for the [train] ep N/M prints

VIOLATION_FLOOR = 1            # C1: >=1 violation confirms the matrix is required
CORRUPTION_RATIO_FLOOR = 2.0   # C2: weakened/intact residue-mass-per-harm ratio
COVERAGE_MIN_WRITES = 1        # every in-scope (phase, store) cell needs >=1
# One hypothesis-tagged rollout write every N env steps. NOT decorative: at the
# original one-per-EPISODE rate the injected leak was ~6 writes against ~375
# waking writes, so C2's corruption ratio was pinned at ~1.05 and could not
# discriminate under ANY outcome (measured 2026-09-17 before this constant
# existed). V1 EXQ-005 saw ~46x because it contaminated the whole write locus;
# a leak has to be a realistic fraction of the write stream to corrupt anything.
SIM_PROBE_EVERY_STEPS = 5

ARM_INTACT = "ARM_INTACT"
ARM_WEAKENED = "ARM_MECH094_WEAKENED"
ARMS = [ARM_INTACT, ARM_WEAKENED]

# The (phase, store) cells the audit claims to police. `call_site_reachable`
# records the pre-flight structural finding: sleep cells are covered by the
# before/after hash instrument, never by a call-site hit, so scoring them on
# call-site hits would be structurally unsatisfiable.
COVERAGE_CELLS = [
    {"phase": "waking", "store": "residue", "call_site_reachable": True},
    {"phase": "waking", "store": "beta_gate", "call_site_reachable": True},
    {"phase": "waking", "store": "anchor_set", "call_site_reachable": True},
    {"phase": "simulation", "store": "residue", "call_site_reachable": True},
    {"phase": "sleep", "store": "residue", "call_site_reachable": False},
    {"phase": "sleep", "store": "beta_gate", "call_site_reachable": False},
    {"phase": "sleep", "store": "anchor_set", "call_site_reachable": False},
    {"phase": "closure", "store": "residue", "call_site_reachable": True},
]

_ZG = ZGoalStreamAccumulator()

ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=6,
    use_proxy_fields=True,
    harm_history_len=10,
    env_drift_interval=5,
    env_drift_prob=0.1,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flat_scalar(value: Any) -> Optional[float]:
    """Coerce to a finite float for the flat readout block, else None.

    Booleans are emitted as 0/1 ints by the caller; non-finite values are
    DROPPED rather than emitted (a nan is numeric to the indexer and would
    pollute a delta, while an absent key correctly reads as unmeasured).
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _state_hash(agent) -> str:
    """sha256 over the three durable authority stores the audit polices.

    This is the sleep instrument: `_run_cycle` never CALLS the write methods
    (pre-flight finding in the module docstring), so an offline authority write
    is only detectable as a change in this digest across the cycle.
    """
    h = hashlib.sha256()
    rf = getattr(agent, "residue_field", None)
    if rf is not None:
        for key, value in sorted(rf.state_dict().items()):
            h.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
    bg = getattr(agent, "beta_gate", None)
    if bg is not None:
        h.update(repr(bool(getattr(bg, "is_elevated", False))).encode("utf-8"))
        h.update(repr(int(getattr(bg, "_committed_run_length", 0) or 0)).encode("utf-8"))
    hip = getattr(agent, "hippocampal", None)
    anchors = getattr(hip, "anchor_set", None) if hip is not None else None
    if anchors is not None:
        h.update(repr(sorted(str(k) for k in getattr(anchors, "_anchors", {}))).encode("utf-8"))
    return h.hexdigest()


class WriteAudit:
    """Driver-side instrument. Wraps the write sites; restores them on exit.

    `ree_core` is never modified -- every wrapper is bound as an INSTANCE
    attribute and removed in `restore()`, so the class method is untouched.
    """

    def __init__(self, agent, weaken_mech094: bool = False):
        self.agent = agent
        self.weaken_mech094 = weaken_mech094
        self.writes: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []
        self.blocked_elsewhere = 0
        self.phase = "waking"
        self._originals: List[Any] = []

    # -- bookkeeping -------------------------------------------------------
    def _record(self, store: str, actor: str, hypothesis_tag: bool) -> None:
        self.writes.append(
            {"phase": self.phase, "store": store, "actor": actor,
             "hypothesis_tag": bool(hypothesis_tag)}
        )

    def _flag_violation(self, store: str, actor: str, reason: str) -> None:
        self.violations.append(
            {"phase": self.phase, "store": store, "actor": actor, "reason": reason}
        )

    # -- wrappers ----------------------------------------------------------
    def install(self) -> None:
        agent = self.agent

        rf = getattr(agent, "residue_field", None)
        if rf is not None:
            orig_acc = rf.accumulate

            def wrapped_accumulate(z_world, harm_magnitude=1.0, world_delta=None,
                                   hypothesis_tag=False, _orig=orig_acc):
                self._record("residue", "ResidueField.accumulate", hypothesis_tag)
                if hypothesis_tag:
                    # MECH-094: hypothesis-tagged content must NOT reach residue.
                    if self.weaken_mech094:
                        # Deliberate driver-side weakening of this ONE local gate
                        # (the claim's FALSIFYING clause). Let it through.
                        self._flag_violation(
                            "residue", "ResidueField.accumulate",
                            "hypothesis_tagged_content_reached_residue_gate_weakened")
                        return _orig(z_world, harm_magnitude, world_delta, False)
                    # Shipped behaviour: refused here. Record that a DIFFERENT
                    # gate did the catching -- this is the redundancy leg (C3).
                    self.blocked_elsewhere += 1
                return _orig(z_world, harm_magnitude, world_delta, hypothesis_tag)

            rf.accumulate = wrapped_accumulate
            self._originals.append((rf, "accumulate"))

            orig_disc = rf.discharge_domain

            def wrapped_discharge(z_world, factor=0.5, radius=1.5, _orig=orig_disc):
                prev, self.phase = self.phase, "closure"
                try:
                    self._record("residue", "ResidueField.discharge_domain", False)
                    if prev == "sleep":
                        self._flag_violation(
                            "residue", "ResidueField.discharge_domain",
                            "closure_discharge_fired_during_sleep_phase")
                    return _orig(z_world, factor, radius)
                finally:
                    self.phase = prev

            rf.discharge_domain = wrapped_discharge
            self._originals.append((rf, "discharge_domain"))

        bg = getattr(agent, "beta_gate", None)
        if bg is not None:
            for name in ("elevate", "release"):
                orig = getattr(bg, name, None)
                if orig is None:
                    continue

                def make(nm, _orig):
                    def wrapped():
                        self._record("beta_gate", f"BetaGate.{nm}", False)
                        if self.phase == "sleep":
                            self._flag_violation(
                                "beta_gate", f"BetaGate.{nm}",
                                "beta_gate_latch_mutated_during_sleep_phase")
                        return _orig()
                    return wrapped

                setattr(bg, name, make(name, orig))
                self._originals.append((bg, name))

        hip = getattr(agent, "hippocampal", None)
        anchors = getattr(hip, "anchor_set", None) if hip is not None else None
        if anchors is not None:
            for name in ("write_anchor", "mark_inactive", "reset_region",
                         "consume_boundary_events"):
                orig = getattr(anchors, name, None)
                if orig is None:
                    continue

                def make(nm, _orig):
                    def wrapped(*a, **kw):
                        self._record("anchor_set", f"AnchorSet.{nm}", False)
                        if self.phase == "sleep":
                            self._flag_violation(
                                "anchor_set", f"AnchorSet.{nm}",
                                "anchor_active_flag_written_during_sleep_phase")
                        return _orig(*a, **kw)
                    return wrapped

                setattr(anchors, name, make(name, orig))
                self._originals.append((anchors, name))

    def restore(self) -> None:
        for obj, name in self._originals:
            try:
                delattr(obj, name)   # fall back through to the class method
            except AttributeError:
                pass
        self._originals = []

    # -- census ------------------------------------------------------------
    def cell_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for w in self.writes:
            out[f"{w['phase']}::{w['store']}"] = out.get(f"{w['phase']}::{w['store']}", 0) + 1
        return out


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_config(env, obs) -> Dict[str, Any]:
    """The config slice the audit runs under. Sleep ON so the sleep cells are
    reachable at all; MECH-094 replay ON so simulation-phase writes exist."""
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_obs_a_dim=int(obs["harm_obs_a"].numel()),
        harm_history_len=10,
        use_sleep_loop=True,
        sws_enabled=True,
        rem_enabled=True,
        sleep_loop_episodes_K=1,
        # Coverage-driven, not decorative: without these two the
        # (waking, anchor_set) and (closure, residue) cells have NO reachable
        # call site and the coverage precondition fails the run. The smoke test
        # confirmed both were missing before they were added.
        use_anchor_sets=True,
        use_event_segmenter=True,   # boundary events are what DRIVE AnchorSet writes
        use_lateral_pfc_analog=True,   # required by use_closure_operator
        use_closure_operator=True,
        # notify_env_completion returns None immediately without this flag
        # (agent.py:10319), so the (closure, residue) cell stays unreachable.
        use_closure_env_completion_hook=True,
    )


def run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    env = _make_env(seed)
    _flat, obs = env.reset()
    cfg_slice = _make_config(env, obs)

    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
        config = REEConfig.from_dims(**cfg_slice)
        agent = REEAgent(config)
        audit = WriteAudit(agent, weaken_mech094=(arm == ARM_WEAKENED))
        audit.install()

        n_harm_events = 0
        sleep_cycles = 0
        sleep_hash_changed = 0
        episodes = 1 if dry_run else EPISODES
        steps = 8 if dry_run else STEPS_PER_EPISODE

        try:
            for ep in range(episodes):
                env_ep = _make_env(seed * 1000 + ep)
                env_ep.reset()
                harness = StepHarness(agent, env_ep, train_mode=True, seed=seed)

                step_n = {"i": 0}

                def _on_step(result):
                    nonlocal n_harm_events
                    hs = getattr(result, "harm_signal", None)
                    if hs is not None and float(hs) < 0:
                        n_harm_events += 1
                    step_n["i"] += 1
                    # Simulation-phase leak probe, at a realistic RATE (see
                    # SIM_PROBE_EVERY_STEPS). Rollout content is produced every
                    # E3 tick, so a per-episode injection understates the leak
                    # by ~2 orders of magnitude and starves C2.
                    if step_n["i"] % SIM_PROBE_EVERY_STEPS == 0:
                        zw_s = getattr(
                            getattr(agent, "_current_latent", None), "z_world", None)
                        if zw_s is not None:
                            prev_phase = audit.phase
                            audit.phase = "simulation"
                            try:
                                agent.residue_field.accumulate(
                                    zw_s, harm_magnitude=1.0, world_delta=None,
                                    hypothesis_tag=True)
                            finally:
                                audit.phase = prev_phase

                audit.phase = "waking"
                harness.run_episode(max_steps=steps, on_step=_on_step)

                # --- closure-phase probe ----------------------------------
                # SD-034's explicit env-completion hook is what fires the
                # ClosureOperator, which is the ONLY ree_core caller of
                # discharge_domain. Without this call the (closure, residue)
                # coverage cell is unreachable -- confirmed empirically at full
                # scale before this line was added.
                zw0 = getattr(getattr(agent, "_current_latent", None), "z_world", None)
                if zw0 is not None:
                    try:
                        agent.notify_env_completion(action_class=0, z_world=zw0,
                                                    bypass_mode_conditioning=True)
                    except Exception:
                        pass

                # --- sleep-phase probe ------------------------------------
                audit.phase = "sleep"
                before = _state_hash(agent)
                loop = getattr(agent, "sleep_loop", None)
                if loop is not None:
                    loop.force_cycle(agent)
                    sleep_cycles += 1
                after = _state_hash(agent)
                if before != after:
                    sleep_hash_changed += 1
                    audit._flag_violation(
                        "multi", "SleepLoopManager._run_cycle",
                        "durable_authority_store_mutated_across_sleep_cycle")
                audit.phase = "waking"

                print(f"  [train] {arm} seed={seed} ep {ep + 1}/{episodes} "
                      f"writes={len(audit.writes)} violations={len(audit.violations)}",
                      flush=True)

            residue_mass = 0.0
            try:
                stats = agent.get_residue_statistics()
                total = stats.get("total_residue")
                residue_mass = float(total.sum().item()) if total is not None else 0.0
            except Exception:
                residue_mass = 0.0
        finally:
            audit.restore()
            _ZG.observe(agent)

        row = {
            "arm": arm,
            "seed": seed,
            "n_writes": len(audit.writes),
            "n_violations": len(audit.violations),
            "violations": audit.violations[:50],
            "blocked_elsewhere": audit.blocked_elsewhere,
            "cell_counts": audit.cell_counts(),
            "n_harm_events": n_harm_events,
            "residue_mass": residue_mass,
            "residue_mass_per_harm_event": (
                residue_mass / n_harm_events if n_harm_events > 0 else None),
            "sleep_cycles": sleep_cycles,
            "sleep_hash_changed": sleep_hash_changed,
        }
        cell.stamp(row)
    return row


def _coverage_preconditions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One precondition per (phase, store) cell the audit claims to police.

    Cells whose call site is structurally unreachable from `_run_cycle` are
    SCOPED OUT (disposition (a) in the precondition-gate rule) rather than
    failed: the sleep instrument for those cells is the before/after hash, and
    that is recorded separately as `sleep_cycles_observed`.
    """
    merged: Dict[str, int] = {}
    for r in rows:
        for key, count in (r.get("cell_counts") or {}).items():
            merged[key] = merged.get(key, 0) + count

    out: List[Dict[str, Any]] = []
    for spec in COVERAGE_CELLS:
        key = f"{spec['phase']}::{spec['store']}"
        observed = merged.get(key, 0)
        if not spec["call_site_reachable"]:
            out.append({
                "name": f"coverage_{key}",
                "kind": "readiness",
                "description": (
                    "SCOPED OUT: no ree_core call site for this store is reachable "
                    "from SleepLoopManager._run_cycle (pre-flight trace 2026-09-17), "
                    "so a call-site count here is structurally unsatisfiable. This "
                    "cell is covered by the before/after state hash instead."),
                "scoped_out": True,
                "applies_note": "call_site_unreachable_covered_by_state_hash",
                "measured": observed,
                "met": True,
            })
            continue
        out.append({
            "name": f"coverage_{key}",
            "kind": "readiness",
            "description": f"audit observed >=1 write in the {key} cell",
            "control": "the audit claims to police this (phase, store) class",
            "measured": observed,
            "threshold": COVERAGE_MIN_WRITES,
            "direction": "lower",
            "met": observed >= COVERAGE_MIN_WRITES,
        })

    sleep_cycles = sum(int(r.get("sleep_cycles") or 0) for r in rows)
    out.append({
        "name": "sleep_cycles_observed",
        "kind": "readiness",
        "description": (
            "at least one sleep cycle actually ran, so the before/after hash "
            "instrument had something to measure"),
        "control": "SleepLoopManager.force_cycle on an agent with sws+rem enabled",
        "measured": sleep_cycles,
        "threshold": 1,
        "direction": "lower",
        "met": sleep_cycles >= 1,
    })
    return out


def analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    intact = [r for r in rows if r["arm"] == ARM_INTACT]
    weakened = [r for r in rows if r["arm"] == ARM_WEAKENED]

    intact_violations = sum(int(r["n_violations"]) for r in intact)
    c1_pass = intact_violations >= VIOLATION_FLOOR

    def _mass(rs):
        vals = [r["residue_mass_per_harm_event"] for r in rs
                if r.get("residue_mass_per_harm_event") is not None]
        return sum(vals) / len(vals) if vals else None

    m_intact, m_weak = _mass(intact), _mass(weakened)
    ratio = (m_weak / m_intact) if (m_intact and m_weak and m_intact > 0) else None
    c2_pass = bool(ratio is not None and ratio >= CORRUPTION_RATIO_FLOOR)

    preconditions = _coverage_preconditions(rows)
    gate_green = all(p.get("met") for p in preconditions)

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif c1_pass and c2_pass:
        label = "separation_leaks_and_corrupts_matrix_required"
    elif c1_pass and not c2_pass:
        label = "separation_leaks_without_measurable_corruption"
    else:
        label = "no_violations_under_full_coverage_local_gates_sufficient"

    return {
        "c1_intact_violations": intact_violations,
        "c1_pass": c1_pass,
        "c2_corruption_ratio": ratio,
        "c2_pass": c2_pass,
        "c3_blocked_elsewhere": sum(int(r["blocked_elsewhere"]) for r in rows),
        "residue_mass_per_harm_intact": m_intact,
        "residue_mass_per_harm_weakened": m_weak,
        "preconditions": preconditions,
        "gate_green": gate_green,
        "label": label,
    }


def run(dry_run: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = run_cell(arm, seed, dry_run=dry_run)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_writes'] > 0 else 'FAIL'}", flush=True)

    summary = analyse(rows)
    gate_green = summary["gate_green"]
    outcome = "PASS" if (gate_green and summary["c1_pass"]) else "FAIL"

    criteria = [
        {"name": "C1_intact_violations_ge_floor", "load_bearing": True,
         "passed": bool(summary["c1_pass"]),
         "measured": summary["c1_intact_violations"], "threshold": VIOLATION_FLOOR,
         "direction": "lower"},
        # NOT load_bearing, deliberately: C2 grades the leak's CONSEQUENCE and
        # selects the interpretation label, but it does not gate the outcome
        # (only C1 does, per combination_rule). Marking it load_bearing would
        # make a C1-PASS / C2-FAIL run read as `vacuous_pass` to the indexer,
        # which is the opposite of what that flag means here.
        {"name": "C2_corruption_ratio_ge_floor", "load_bearing": False,
         "passed": bool(summary["c2_pass"]),
         "measured": summary["c2_corruption_ratio"],
         "threshold": CORRUPTION_RATIO_FLOOR, "direction": "lower"},
        {"name": "C3_blocked_elsewhere_redundancy", "load_bearing": False,
         "passed": True, "measured": summary["c3_blocked_elsewhere"],
         "threshold_not_applicable":
             "count-based redundancy census; a bar is not the right shape"},
    ]

    direction = "unknown"
    if gate_green:
        if summary["c1_pass"] and summary["c2_pass"]:
            direction = "supports"
        elif not summary["c1_pass"]:
            direction = "weakens"
        else:
            direction = "mixed"

    readout = {}
    for key, value in (
        ("intact_violations", summary["c1_intact_violations"]),
        ("corruption_ratio", summary["c2_corruption_ratio"]),
        ("blocked_elsewhere", summary["c3_blocked_elsewhere"]),
        ("residue_mass_per_harm_intact", summary["residue_mass_per_harm_intact"]),
        ("residue_mass_per_harm_weakened", summary["residue_mass_per_harm_weakened"]),
        ("c1_pass", 1 if summary["c1_pass"] else 0),
        ("c2_pass", 1 if summary["c2_pass"] else 0),
        ("gate_green", 1 if gate_green else 0),
        ("n_cells_covered", sum(1 for p in summary["preconditions"] if p.get("met"))),
    ):
        coerced = _flat_scalar(value)
        if coerced is not None:
            readout[key] = coerced

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{_utc_stamp()}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "MECH-066": direction,
            "MECH-067": direction,
        },
        "outcome": outcome,
        "timestamp_utc": _utc_stamp(),
        "sleep_driver_pattern": "K=1 single-fire (SleepLoopManager, fires every episode)",
        "comparator_arm_deferred": {
            "reason": (
                "MECH-067's matrix-enforced comparator requires a (phase, store, "
                "actor) default-deny table that does not exist anywhere in ree_core "
                "(pre-flight grep 2026-09-17: no store-class enum, no actor "
                "vocabulary, no permission matrix). The claim marks it "
                "substrate_conditional and instructs 'do not invent a matrix DV in "
                "the meantime'. This run is the audit half only."),
        },
        "arm_results": rows,
        "criteria": criteria,
        "criteria_non_degenerate": {
            "C1_intact_violations_ge_floor": bool(gate_green),
            "C2_corruption_ratio_ge_floor": bool(
                gate_green and summary["residue_mass_per_harm_intact"] is not None),
            "C3_blocked_elsewhere_redundancy": bool(gate_green),
        },
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria_non_degenerate": {
                "C1_intact_violations_ge_floor": bool(gate_green),
                "C2_corruption_ratio_ge_floor": bool(gate_green),
            },
        },
        "combination_rule": (
            "outcome PASS requires the coverage gate green AND C1 (>=1 violation "
            "in the intact arm). C2 grades the leak's consequence and separates "
            "'leaks and corrupts' from 'leaks harmlessly'; it does not gate PASS."),
        "readout": readout,
        "summary": summary,
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config={"env": ENV_KWARGS, "episodes": EPISODES,
                "steps_per_episode": STEPS_PER_EPISODE, "arms": ARMS},
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=_t_start,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
