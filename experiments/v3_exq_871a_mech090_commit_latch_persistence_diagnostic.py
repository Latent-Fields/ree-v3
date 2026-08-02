#!/usr/bin/env python3
"""V3-EXQ-871a -- MECH-090 commit-latch cross-tick persistence: H1(wiring) vs
H2(readout artifact) discrimination for the V3-EXQ-841 finding.

SUPERSEDES V3-EXQ-871 (`v3_exq_855_mech090_commit_latch_persistence_diagnostic_
20260802T104333Z_v3`) -- see "STATUS UPDATE (2026-08-02, V3-EXQ-871a fix)"
below and `failure_autopsy_V3-EXQ-871_2026-08-02.json`/`.md` for the confirmed
root cause and routing.

SLEEP DRIVER: n/a -- no sleep loop in this diagnostic (use_sleep_loop /
sws_enabled / rem_enabled / use_sleep_aggregation_cluster all False).

GOV-FANOUT-1 discrimination probe, per
`failure_autopsy_V3-EXQ-841_2026-07-31.json` `targets[0].fanout_recommendation`.
Both suggested axes (H1 instrumentation of the real latch; H2 an independent
readout of the same state) are read from ONE execution trace, per the
autopsy's own note that this is diagnostic instrumentation, not two ablation
arms.

WHAT V3-EXQ-841 FOUND
----------------------
Wherever `use_chunk_proposal_injection=True` is live-tested, the experiment's
own realised-committed-length readout (`_committed_chunk_state()` +
`_segment_commitments()`'s "idx <= prev_idx => new segment" rule) pins at
EXACTLY 1 step in 100% of committing cells, independent of `chunk_max_size`.
Two live hypotheses:

  H1 (wiring) -- the MECH-090 commit latch's step-index advance /
      re-deliberation suppression does not hold across ticks in this harness
      configuration; select_action re-evaluates and re-commits from scratch
      every tick regardless of an existing commitment.
  H2 (readout artifact) -- the true internal commitment state persists
      correctly across ticks, but the experiment's own read path
      (`agent.e3._committed_trajectory` / `agent._committed_step_idx`, or the
      `metadata['source'] != 'arc071_chunk'` check) misreads it.

STATIC-SOURCE FINDING THIS PROBE IS DESIGNED TO CONFIRM EMPIRICALLY
---------------------------------------------------------------------
Read directly against ree-v3 HEAD (2026-08-01), by exact line -- NOT taken on
trust from the autopsy's prose attribute names, matching the SD-076 probes'
verification pattern earlier this session:

  * `agent.py:5965-5987` -- the MECH-090 "Layer 1" between-E3-tick stepping
    fast path (`if not ticks["e3_tick"] and self._last_action is not None:
    ... step through committed trajectory`) reads `self.e3._committed_trajectory`,
    falling back only to `self.e3._closure_committed_trajectory` -- NEVER to
    `_persistent_committed_trajectory`.
  * `e3_selector.py:3910` -- `post_action_update`'s LAST statement
    unconditionally sets `self._committed_trajectory = None`, on EVERY tick.
    `e3_selector.py:345-346,365-366` document this explicitly ("torn down
    every tick by post_action_update").
  * `agent.py:8701` (inside `update_residue`) calls `e3.post_action_update`
    unconditionally on every env step (ARC-016 comment at `agent.py:8695-8698`:
    "update running variance on EVERY step"; `arc071_chunking.py`'s own
    docstring: "update_residue ... drives e3.post_action_update").
  * So `_committed_trajectory` is non-None ONLY within the SAME env-step it
    was set (immediately after `e3.select()`); by the NEXT env-step it is
    already `None` again. The between-tick fast path at 5965-5987 can
    therefore never see a live `_committed_trajectory` on any step OTHER than
    the exact E3-tick step that set it -- absent
    `use_closure_commit_entry_trajectory` (a separate, unused opt-in),
    `agent._committed_step_idx` structurally CANNOT be incremented by the
    trajectory-stepping mechanism outside that one tick. It freezes at
    whatever value it reached on the last E3-tick step until the next one.
  * `agent.e3._persistent_committed_trajectory` (SD-084) is the ONE handle
    NOT torn down every tick (`e3_selector.py:374`: "NOT torn down by
    post_action_update, so it persists across ticks") -- but
    `agent.py:5972-5975`'s between-tick fast path does not consult it.
  * `_committed_chunk_state()` (V3-EXQ-841's OWN read function, reproduced
    verbatim below as `_committed_chunk_state`) DOES fall back to
    `_persistent_committed_trajectory` for the chunk-sequence identity, so
    that half of its reading is accurate across ticks -- but it reads
    `_committed_step_idx` from the agent directly, which is the value that
    FREEZES per the above. A frozen (non-increasing) idx satisfies 841's own
    `_segment_commitments` "new commitment" rule (`idx <= prev_idx`) on every
    tick after the first, producing "new segment, length 1" on every single
    step even while the SAME chunk is genuinely still committed underneath.
  * Also confirmed: `beta_gate_bistable` defaults to False
    (`ree_core/utils/config.py:2387`) and is not set True anywhere in this
    experiment's flag chain (`arc071_chunking.off_arm_flags()` /
    `v3_exq_841`'s own `_arm_flags` / this script's `_arm_flags`), so the
    parent run used the "legacy" MECH-090 path (`agent.py:7975-8048`), which
    ADDITIONALLY force-resets `_committed_step_idx = 0` on every E3-committed
    tick (`agent.py:7991-7998`, comment: "In non-bistable mode E3 re-selects a
    trajectory on every E3 tick, so the counter must restart from 0 each
    time") -- a second, independent contributor layered on the freeze above.

If this reading is right, the mechanism is closer to H2 in SPIRIT (the real
commitment -- by object identity -- persists correctly; the SD-084
`_persistent_committed_trajectory` handle is stable) but the CAUSE is a
genuine, specific substrate gap: the Layer-1 between-tick stepping fast path
was wired against the handle that CANNOT survive a tick
(`_committed_trajectory`) rather than the one the substrate itself built for
cross-tick survival (`_persistent_committed_trajectory`, SD-084). This probe
exists to CONFIRM that empirically rather than resting on static reading
alone, and to quantify it: does the persistent trajectory's OBJECT IDENTITY
genuinely change every tick (true H1: the substrate really does re-commit
from scratch constantly), or does it persist across many ticks while the
OFFICIALLY-READ step-index/segmentation reports a reset every tick anyway
(the frozen-index mechanism above, H2-flavoured)?

STATUS UPDATE (2026-08-02) -- H1/H2 ALREADY RESOLVED; THIS RUN IS NOW A
POST-FIX VALIDATION, NOT A LIVE DISCRIMINATION
-------------------------------------------------------------------------
Between this script's original drafting and its queuing, an ad hoc (non-EXQ)
diagnostic run by a parallel session -- `quirky-mayer-ee5ad2`, recorded in
`REE_assembly/evidence/planning/diagnostic_arc071_commit_latch_h1h2_probe_
2026-07-31.md` -- already answered the H1-vs-H2 question directly, using a
monkeypatched guaranteed-selectable chunk rather than organic formation:
**H1 CONFIRMED (real ree_core wiring defect), H2 REFUTED (not a readout
artifact)**. The fix for one of the two independent wiring gaps that diagnosis
found landed on ree-v3 main the same day: commit `278599a` ("ARC-071
commit-latch persistence: fallback to persistent trajectory handle") adds a
fallback to `agent.e3._persistent_committed_trajectory` in the between-E3-tick
stepping branch (`agent.py`, `select_action`), gated behind a NEW flag
`use_persistent_committed_program_handle` (default False,
`ree_core/utils/config.py`). This addresses gap (a) from the diagnosis (the
between-tick fast path never consulting the persistent handle); gap (b) (E3
unconditionally re-committing a brand-new trajectory object on every E3 tick,
regardless of an existing unexpired commitment) is a SEPARATE, still-open
question the diagnosis surfaced but did not resolve.

Because H1/H2 are already decided, running this script with the flag at its
lineage default (False, matching `experiments/_lib/baselines/
arc071_chunking.py`'s `off_arm_flags()`, which the fix's gap (b) deliberately
leaves unchanged pending a governance-aware review of the arm-reuse
fingerprint blast radius) would only reproduce the already-known pre-fix
result under organic formation -- informative as a from-scratch replication,
but not the open question. So `_arm_flags()` below sets
`use_persistent_committed_program_handle: True` as a SCRIPT-LOCAL override
(mirroring the precedent at `experiments/_lib/baselines/
sd084_midexec_reachability.py:271`, which does the same without touching the
shared baseline module) -- this run instead asks: does the landed fix (a)
restore meaningful multi-step commit-latch persistence under REAL organic
chunk formation (as opposed to the ad hoc probe's forced/monkeypatched
setup)? That is exactly the evidence gap (b)'s own governance review needs:
whether flipping the flag on lineage-wide for the `arc071_chunking` family is
warranted.

STATUS UPDATE (2026-08-02, V3-EXQ-871a fix) -- FLAG RE-SCOPED TO THE PROBE
PHASE ONLY; V3-EXQ-871 CONFIRMED FAIL, NOT A SUBSTRATE FINDING
-------------------------------------------------------------------------
V3-EXQ-871 (the direct predecessor of this script, `v3_exq_855_...` on disk)
set `use_persistent_committed_program_handle: True` in `_arm_flags()`, which
is folded into the single `flags` dict passed to `base.build_agent(env,
flags)` at the top of `_run_cell` -- i.e. baked into the agent's REEConfig at
CONSTRUCTION time, applied identically through BOTH `_run_formation` and
`_run_instrumented_probe`. Per `failure_autopsy_V3-EXQ-871_2026-08-02.json`
(confirmed), this invalidated the "readiness inherited by construction" claim
above: the flag also changes `select_action`'s between-E3-tick fast path
DURING formation (not just during the probe), which -- via RNG-coupled
environment dynamics (RNG reset confirmed deterministic per seed,
`arm_fingerprint.py:457-480`) -- cascaded into materially fewer ARC-071
chunks forming. Direct same-seed comparison against V3-EXQ-841 (flag OFF,
lineage default) on the identical 120ep/72-step schedule: seeds 101/505
formed 2/2 chunks under 841 vs only 1/0 chunks under 871's whole-cell flag.
That starved the probe's own readiness precondition
(`arc071_chunk_commitments_observed_supra_floor`, floor 10, measured 0.0 on
both seeds) and forced a `substrate_not_ready_requeue` FAIL -- a pure
probe-construction defect (autopsy: `measurement_test_design_defect`), not a
substrate gap. Neither MECH-090 nor ARC-071 evidence changed
(`non_contributory`, `precondition_unmet`); both claims remain intact.

**The fix, per the autopsy's confirmed routing (Section 6):** `_arm_flags()`
below no longer sets `use_persistent_committed_program_handle` at all, so
`_run_formation` runs under the agent's constructed default (False, same as
V3-EXQ-841's lineage-default construction) and replays 841's proven 2/2-chunk
dynamics UNPERTURBED. The flag is instead flipped directly on the live
agent's config object -- `agent.config.use_persistent_committed_program_handle
= True` -- in `_run_cell`, AFTER `_run_formation()` returns and BEFORE
`_run_instrumented_probe()` begins. This is safe because `agent.py:5609-5612`
and `agent.py:6009-6012` (the two between-E3-tick fast-path branches this
flag gates) both read it LIVE via `getattr(self.config,
"use_persistent_committed_program_handle", False)` on EVERY tick -- the value
is never cached or baked in at agent-construction time -- so mutating
`agent.config` after formation takes effect immediately for the probe phase,
with no need to rebuild the agent (which would also discard the just-formed
chunk library). All existing instrumentation, thresholds, and verdict logic
(`_analyze_cell`, `PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR`,
`PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL`, etc.) are unchanged and remain
the correct readouts for the same question -- only the flag's temporal SCOPE
changed, from "whole cell" to "probe phase only". This makes the
`ANCHOR_REACHABILITY_EXEMPT` claim below actually TRUE by construction now
(previously it was asserted but silently violated by the whole-cell flag
application).

Reading the existing instrumentation under this flipped flag: the underlying
measurements (`mean_true_dwell_ticks`, `frac_between_tick_steps_idx_frozen`,
`official_readout_pct_length_one`) are unchanged and remain exactly the right
readouts for this revised question -- they just now answer "does persistence
hold post-fix" rather than "is this H1 or H2". The `cell_verdict` labels below
(`h1_wiring_defect_dominant` / `h2_readout_artifact_dominant`) are historical
names for the same underlying statistical signatures and should now be read
as, respectively, "still pinned near length 1 even with the fix enabled --
fix (a) alone is insufficient, gap (b) is the dominant remaining defect" and
"real multi-step persistence restored -- fix (a) is doing its job under
organic formation". `h1_h2_note` in the manifest states this reading
explicitly so a later reader does not mistake this for a live H1-vs-H2 result.

DESIGN -- ONE INSTRUMENTED RUN, NOT A RE-RUN OF V3-EXQ-841
------------------------------------------------------------
Per the autopsy's own `fanout_recommendation.suggested_probes`: a single
instrumented run on the two seeds where the effect reproduces most cleanly
(A_HIER_S2, seeds 101 and 505, chunk_max_size=2), reading BOTH signals from
the SAME execution trace. This is diagnostic INSTRUMENTATION -- it reads
existing ree_core attributes from this script's own driving loop, via
`_harness.StepHarness`'s `on_action` / `on_post_step` hooks (which fire
immediately after `select_action()` and after `update_residue()`
respectively -- see `_harness.py` step() docstring, steps 8 and 11). IT DOES
NOT MODIFY ree_core IN ANY WAY -- no new instrumentation hooks are added to
the substrate itself. No devaluation manipulation, no dose sweep, no
persistence DV: those belong to MECH-163/Q-085 and are out of scope here.

The formation phase is IDENTICAL to `arc071_chunking`'s proven 120ep x
72-step schedule (the same module V3-EXQ-810a passed), so chunk
formation/crystallisation readiness is INHERITED, not re-derived. The probe
phase is a plain post-formation run (10 episodes; no devaluation) with
per-tick instrumentation attached via `StepHooks`.

CLAIM_IDS -- MECH-090 and ARC-071, NOT MECH-163/Q-085
-------------------------------------------------------
V3-EXQ-841 excluded ARC-071 and MECH-323 as PRECONDITIONS of its MECH-163/
Q-085 measurement (chunk formation/injection exercised, not tested; see 841's
own `CLAIM_IDS` comment). This probe inverts that: it does not touch
MECH-163's hierarchical-vs-flat question or Q-085's dose-response question at
all (no devaluation, no dose sweep, no persistence DV), so neither claim
belongs here. What IS directly under test is whether the MECH-090 commit
latch holds ACROSS TICKS under live ARC-071 chunk injection -- i.e. MECH-090
itself (the mechanism whose cross-tick persistence semantics this probe
interrogates) and ARC-071 (the chunk-injection pathway whose LIVE use is what
exposes the failure -- 841's own finding is scoped to "wherever the ARC-071
chunk-injection pathway is live-tested"). MECH-323 (the accumulator/
crystallisation side) is exercised as a precondition here too (chunk
formation must succeed before there is anything to commit) but is not itself
in question -- 841 already found accumulation behaves sensibly in isolation
-- so it is excluded, matching 841's own precedent for the same reason.

ETHICS PREFLIGHT (descriptive, non-enforced; SENT-0 -- V3 is not claimed
sentient): identical posture to V3-EXQ-841 -- involves_negative_valence False
(the num_hazards=2 harm stream is pre-ethical instrumentation, unchanged from
810a/841), involves_suffering_like_state False (MECH-219 accumulator off),
involves_self_model False, involves_inescapability False,
involves_offline_replay_over_harm False (sleep off), involves_social_mind
False, involves_human_data False. decision: allow.
"""

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.baselines import arc071_chunking as base  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# z_goal liveness (code-review requirement -- see /queue-experiment skill Step
# 3.5 "Output contract": every driver that steps an agent must record the
# z_goal_stream block or a dead-writer defect is invisible). Each seed cell
# below builds a FRESH agent inside `_run_cell`, so the accumulator (not a
# single `agent=` kwarg, and not a run-level list of live agents) is the
# correct wiring per the module's own docstring.
_ZG = ZGoalStreamAccumulator()

EXPERIMENT_TYPE = "v3_exq_871a_mech090_commit_latch_persistence_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"

# Supersession (CLAUDE.md EXQ lettered-iteration policy): this run corrects a
# pure probe-construction defect in its predecessor (V3-EXQ-871, script on
# disk as v3_exq_855_mech090_commit_latch_persistence_diagnostic.py) -- see
# module docstring "STATUS UPDATE (2026-08-02, V3-EXQ-871a fix)" and
# failure_autopsy_V3-EXQ-871_2026-08-02.json (confirmed). Same question, same
# claims, same thresholds -- only the flag's temporal scope changed.
SUPERSEDES_QUEUE_ID = "V3-EXQ-871"
SUPERSEDES_RUN_ID = (
    "v3_exq_855_mech090_commit_latch_persistence_diagnostic_"
    "20260802T104333Z_v3"
)

# See module docstring "CLAIM_IDS" section for the full reasoning: this probe
# tests MECH-090's cross-tick persistence directly (the hypothesis under
# test), exercised via ARC-071's live chunk-injection pathway (the vehicle
# that exposes the failure). MECH-163/Q-085 (841's own claims) are NOT tested
# here -- no devaluation, no dose sweep. MECH-323 is exercised as a
# precondition (formation must succeed) but not itself in question, matching
# 841's own precedent for the same exclusion.
CLAIM_IDS = ["MECH-090", "ARC-071"]

SEEDS = [101, 505]  # the two seeds where V3-EXQ-841's effect reproduced most
                     # cleanly, per the autopsy's own suggested_probes
ARM_ID = "A_HIER_S2"
CHUNK_MAX_SIZE = 2  # smallest dose in 841's sweep -- effect reproduced cleanly here

N_EPISODES_TRAIN = base.N_EPISODES          # 120 -- the 810a-proven formation schedule
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72
N_EPISODES_PROBE = 10                       # instrumentation only; no DV needs
                                             # statistical power, so this is far
                                             # smaller than 841's 30-episode probe

# validate_experiments.py readiness-anchor-reachability advisory: the
# arc071_chunk_commitments_observed_supra_floor gate below is reachable BY
# CONSTRUCTION at the real (non-dry-run) N_EPISODES_TRAIN=120 schedule -- this
# is the exact schedule + config V3-EXQ-810a proved forms chunks (8/8 seeds)
# and V3-EXQ-841 proved COMMITS them (32/32 chunking cells registered >=1
# commitment at this schedule, chunk_max_size in {2,3,5}). This script does
# not re-derive that readiness; it inherits it unchanged (module docstring
# "Formation phase is IDENTICAL to arc071_chunking's proven ... schedule").
# UNLIKE its predecessor V3-EXQ-871, this claim is now actually TRUE by
# construction: _arm_flags() below no longer perturbs
# use_persistent_committed_program_handle during formation (see "STATUS
# UPDATE (2026-08-02, V3-EXQ-871a fix)" above), so _run_formation runs under
# the exact same flag configuration V3-EXQ-841 used. The gate is UNREACHABLE
# only at toy --dry-run scale (n_train=2), which is the expected, self-routing
# substrate_not_ready_requeue smoke-test outcome, not a defect -- confirmed
# directly by this script's own --dry-run run.
ANCHOR_REACHABILITY_EXEMPT = (
    "reachability already established by V3-EXQ-810a (8/8 seeds formed "
    "chunks) and V3-EXQ-841 (32/32 chunking cells committed >=1 chunk) on "
    "the IDENTICAL inherited formation schedule/config (base.N_EPISODES=120, "
    "base.STEPS_PER_EPISODE=72, same ENV_KWARGS/CHUNK_* constants), AND (fixed "
    "in this iteration, V3-EXQ-871a) under the IDENTICAL flag configuration -- "
    "use_persistent_committed_program_handle is no longer set during "
    "_run_formation, only flipped on the live agent config afterward, before "
    "_run_instrumented_probe. This script changes nothing about formation, "
    "only what happens to a chunk after it is committed AND only during the "
    "probe phase, so re-deriving reachability here would just re-run "
    "810a/841's own already-recorded result at extra compute cost."
)

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
# P0 readiness: SOME commitment must register on the probe agent, or there is
# nothing to discriminate (mirrors 841's own gate (a): zero commitments is a
# readiness failure that scores nothing, not a null).
MIN_PERSISTENT_PRESENT_TICKS = 10

# Supplementary identity-based diagnostics (retained from the original H1/H2
# framing -- see module docstring "STATUS UPDATE"). mean_true_dwell_ticks =
# (probe ticks with a live persistent-trajectory handle) / (distinct
# persistent-trajectory OBJECT IDENTITIES observed), taken OUTSIDE
# _committed_chunk_state's own read path. NOT the verdict-routing statistic
# below -- see the STATUS UPDATE note on why identity-dwell and the official
# step-index counter are two DIFFERENT substrate mechanisms (SD-084 handle
# survival vs the MECH-090 step-index advance) and conflating them was the
# original script's own H1/H2 mislabelling risk, caught in code review.
H2_MEAN_TRUE_DWELL_FLOOR = 3.0   # true commit dwell >= this -> real persistence
H2_FROZEN_FRACTION_FLOOR = 0.8   # frac of eligible between-tick steps whose
                                  # OFFICIAL step-idx did not advance
H1_MEAN_TRUE_DWELL_CEIL = 1.5    # true commit dwell <= this -> genuine
                                  # near-every-tick re-commitment

# PRIMARY verdict-routing thresholds (post-fix validation, added 2026-08-02).
# Routes directly on official_readout_mean_realised_length -- the SAME
# statistic V3-EXQ-841 itself used to establish the "pinned at 1" finding --
# rather than the true-dwell/frozen-fraction pair above. Reusing that pair as
# the verdict criterion would conflate two independent substrate mechanisms:
# the SD-084 persistent-handle's OBJECT IDENTITY can survive across many
# ticks (a fact about `_persistent_committed_trajectory`) while the SEPARATE
# `_committed_step_idx` counter still fails to advance (a fact about the
# MECH-090 latch's step-index wiring) -- the diagnostic doc's own ad hoc probe
# hit exactly this combination (true dwell high, idx frozen) and correctly
# read it as H1 (real wiring defect), not H2 (readout misread the true state)
# -- because the official read function was independently confirmed to
# FAITHFULLY report the frozen idx (0/160 raw-attribute mismatches), so a
# frozen idx alongside a surviving handle identity is a genuine counter bug,
# not evidence the reader is lying. Routing on the counter's own
# officially-read realised length sidesteps that conflation entirely.
PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR = 3.0   # realised committed length
                                                 # clears this -> fix (a)
                                                 # restores meaningful
                                                 # multi-step persistence
PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL = 1.5  # realised length stays near
                                                    # 1 -> still pinned; fix
                                                    # (a) alone is
                                                    # insufficient (gap (b),
                                                    # E3 re-committing every
                                                    # E3 tick, still dominates)

RELEASES_PINNED_OFF = {
    "use_maintenance_release": False,
    "use_natural_commit_urgency_release": False,
    "use_vs_commit_release": False,
    "use_closure_decommit": False,
}


def _arm_flags() -> Dict[str, Any]:
    """A_HIER_S2 flags, built on top of the lineage OFF-path flags -- same
    construction as V3-EXQ-841's `_arm_flags(A_HIER_S2)`.

    V3-EXQ-871a fix (see module docstring "STATUS UPDATE (2026-08-02,
    V3-EXQ-871a fix)"): UNLIKE its predecessor V3-EXQ-871, this function does
    NOT set `use_persistent_committed_program_handle` -- the agent is
    constructed with it at its REEConfig default (False), so `_run_formation`
    in `_run_cell` below replays V3-EXQ-841's proven lineage-default (flag
    OFF) formation dynamics unperturbed. The flag is instead flipped directly
    on the live agent's config object in `_run_cell`, AFTER `_run_formation`
    completes and BEFORE `_run_instrumented_probe` begins -- scoping ree-v3
    main commit 278599a's fallback (`agent.e3._persistent_committed_trajectory`
    in the between-E3-tick stepping branch of `agent.py`'s `select_action`)
    to the probe phase only, matching the biological/experimental-design
    reading of "does the fix restore persistence GIVEN organically-formed
    chunks" rather than "does the fix also change what forms".
    """
    flags = base.off_arm_flags()
    flags.update(RELEASES_PINNED_OFF)
    flags.update({
        "use_policy_chunking": True,
        "use_chunk_maintenance": True,
        "use_chunk_proposal_injection": True,
        "use_chunk_all_position_credit": True,
        "chunk_max_size": CHUNK_MAX_SIZE,
    })
    return flags


def _config_slice() -> Dict[str, Any]:
    slice_ = base.arm_config_slice(_arm_flags())
    slice_["probe"] = {
        "n_episodes_probe": N_EPISODES_PROBE,
        "instrumented": True,
        "diagnostic": "mech090_commit_latch_persistence_h1_h2",
        # V3-EXQ-871a fix: the flag is applied AFTER _arm_flags()/build_agent()
        # -- see _run_cell -- scoped to the probe phase only. Recorded here so
        # a later reader of the config slice sees the scoping explicitly
        # rather than inferring it from the absence of the flag above.
        "use_persistent_committed_program_handle_probe_only": True,
    }
    return slice_


def _traj_chunk_seq(traj: Any) -> Optional[Tuple[int, ...]]:
    """Chunk-sequence identity iff `traj` is a live arc071_chunk trajectory."""
    meta = getattr(traj, "metadata", None) if traj is not None else None
    if not meta or meta.get("source") != "arc071_chunk":
        return None
    seq = tuple(int(a) for a in meta.get("chunk_sequence", ()))
    return seq or None


def _committed_chunk_state(agent: Any) -> Tuple[Optional[Tuple[int, ...]], int]:
    """VERBATIM reproduction of V3-EXQ-841's own read path.

    This IS "the existing read path" the autopsy's H2 probe sketch asks the
    independent reading to be compared against -- attribute names verified
    against ree-v3 HEAD (see module docstring), not copied on trust.
    """
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return None, 0
    traj = getattr(e3, "_committed_trajectory", None)
    if traj is None:
        traj = getattr(e3, "_persistent_committed_trajectory", None)
    return _traj_chunk_seq(traj), int(getattr(agent, "_committed_step_idx", 0))


def _run_formation(harness: Any, agent: Any, n_episodes: int, *,
                   progress_every: int, label: str) -> Dict[str, Any]:
    """The formation phase. Reproduced from V3-EXQ-841's `_run_formation`
    (same lineage rationale: `_lib/**` loop-sharing must not edit
    `arc071_chunking.run_cell`, which builds+discards its own agent -- this
    experiment's probe phase must run on the SAME agent that formed the
    chunks, or gate (a) reads ~0 (V3-EXQ-834's outcome))."""
    episode_outcomes: List[float] = []
    n_steps_total = 0
    for ep in range(n_episodes):
        results = harness.run_episode(STEPS_PER_EPISODE)
        ep_reward = sum(r.harm_signal for r in results)
        n_steps_total += len(results)
        agent.note_chunk_outcome(ep_reward)
        episode_outcomes.append(ep_reward)
        if (ep + 1) % progress_every == 0:
            st_now = agent.get_chunking_state()
            print(f"  [train] chunk {label} ep {ep+1}/{n_episodes} "
                  f"formed={st_now.get('chunk_acc_n_formed', 0)} "
                  f"cryst={st_now.get('chunk_lib_n_crystallised', 0)}", flush=True)
    st = agent.get_chunking_state()
    return {
        "chunking_state": st,
        "episode_outcomes": episode_outcomes,
        "n_steps_total": n_steps_total,
        "formed_chunk_lengths": base.formed_chunk_lengths(agent),
        "chunking_instantiated": bool(agent.policy_chunking is not None),
    }


def _run_instrumented_probe(harness: Any, n_probe: int) -> Dict[str, Any]:
    """The instrumentation phase. Attaches StepHooks to the SAME harness/agent
    that just formed chunks (no devaluation, no dose sweep -- this is not a
    re-run of V3-EXQ-841's DV).

    Two hook points give the head-to-head comparison the autopsy's H2 probe
    sketch asks for, on the SAME tick:
      * on_action  -- fires immediately after select_action(), BEFORE env.step
        and BEFORE update_residue()/post_action_update() run. This is the one
        moment `agent.e3._committed_trajectory` can be observed non-None on a
        fresh-commit tick, before it is torn down.
      * on_post_step -- fires AFTER update_residue()/post_action_update(). By
        this point `_committed_trajectory` should ALREADY be None again (the
        e3_selector.py:3910 teardown) on every tick, if the static-source
        reading in the module docstring is right -- this hook directly tests
        that empirically.

    `n_genuine_commits` is the INDEPENDENT identity-based reading: it counts
    every tick on which `id(agent.e3._persistent_committed_trajectory)`
    CHANGES from the previous tick -- read purely from Python object identity,
    entirely OUTSIDE `_committed_chunk_state()`'s own read path (which reads
    `_committed_step_idx`, not object identity).
    """
    records: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {"prev_persistent_id": None, "n_genuine_commits": 0}

    def on_action(*, agent: Any, ticks: Dict[str, bool], **_kw: Any) -> None:
        e3 = agent.e3
        raw_traj = getattr(e3, "_committed_trajectory", None)
        pers_traj = getattr(e3, "_persistent_committed_trajectory", None)
        pers_id = id(pers_traj) if pers_traj is not None else None
        is_new_persistent = (
            pers_id is not None and pers_id != state["prev_persistent_id"]
        )
        if is_new_persistent:
            state["n_genuine_commits"] += 1
        state["prev_persistent_id"] = pers_id
        existing_seq, existing_idx = _committed_chunk_state(agent)
        records.append({
            "e3_tick": bool(ticks.get("e3_tick", False)),
            "beta_elevated": bool(agent.beta_gate.is_elevated),
            "raw_committed_trajectory_present": raw_traj is not None,
            "raw_committed_trajectory_id": (
                id(raw_traj) if raw_traj is not None else None),
            "raw_chunk_seq": _traj_chunk_seq(raw_traj),
            "persistent_trajectory_present": pers_traj is not None,
            "persistent_trajectory_id": pers_id,
            "persistent_chunk_seq": _traj_chunk_seq(pers_traj),
            "persistent_is_new_object": is_new_persistent,
            "n_genuine_commits_so_far": state["n_genuine_commits"],
            "committed_step_idx": int(getattr(agent, "_committed_step_idx", 0)),
            "existing_readout_chunk_seq": existing_seq,
            "existing_readout_step_idx": existing_idx,
        })

    def on_post_step(*, agent: Any, **_kw: Any) -> None:
        e3 = agent.e3
        raw_traj_post = getattr(e3, "_committed_trajectory", None)
        records[-1].update({
            "post_raw_committed_trajectory_present": raw_traj_post is not None,
            "post_committed_step_idx": int(
                getattr(agent, "_committed_step_idx", 0)),
        })

    harness.hooks.on_action = on_action
    harness.hooks.on_post_step = on_post_step

    probe_every = max(1, min(5, n_probe // 2))
    for ep in range(n_probe):
        harness.run_episode(STEPS_PER_EPISODE)
        if (ep + 1) % probe_every == 0:
            print(f"  [probe] instrumented episode {ep+1}/{n_probe} "
                  f"records={len(records)}", flush=True)

    # Detach the hooks so a caller reusing this harness elsewhere is not
    # surprised by them still being wired.
    harness.hooks.on_action = None
    harness.hooks.on_post_step = None

    return {"records": records, "n_genuine_commits": state["n_genuine_commits"]}


def _segment_official_readout(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], int]:
    """Reproduces V3-EXQ-841's `_segment_commitments` rule EXACTLY (a new
    segment begins when the sequence changes, or when the step index fails to
    advance from the previous step -- `idx <= prev_idx`), applied to THIS
    probe's `existing_readout_*` fields. This is the replication check: does
    the "pinned at 1" signature reproduce at this much smaller scale before we
    call it explained?

    Returns (realised_lengths, n_segments).
    """
    lengths: List[int] = []
    cur_len = 0
    prev_seq: Optional[Tuple[int, ...]] = None
    prev_idx = -1
    for r in records:
        seq = r["existing_readout_chunk_seq"]
        idx = int(r["existing_readout_step_idx"])
        if seq is None:
            if cur_len > 0:
                lengths.append(cur_len)
            cur_len = 0
            prev_seq, prev_idx = None, -1
            continue
        new = (cur_len == 0) or (seq != prev_seq) or (idx <= prev_idx)
        if new:
            if cur_len > 0:
                lengths.append(cur_len)
            cur_len = 0
        cur_len += 1
        prev_seq, prev_idx = seq, idx
    if cur_len > 0:
        lengths.append(cur_len)
    return lengths, len(lengths)


def _analyze_cell(records: List[Dict[str, Any]], agent: Any) -> Dict[str, Any]:
    """SCOPED TO GENUINE ARC-071 CHUNK COMMITMENTS ONLY.

    `agent.e3._persistent_committed_trajectory` is set on ANY committed
    selection (e3_selector.py:3527-3535's `if committed:` block does not
    check candidate provenance), so an un-scoped identity count would mix in
    ordinary (non-chunk) MECH-090 commitments -- confirmed by the --dry-run
    smoke test: at the toy 2-episode formation schedule ZERO chunks form (as
    834/841 already established a short schedule cannot form chunks), yet the
    probe still registered ordinary commitments, so an un-scoped reading
    produced a confident-looking verdict from data that never touched ARC-071
    injection at all. Every statistic below is therefore restricted to ticks
    where `persistent_chunk_seq`/`existing_readout_chunk_seq` is non-None --
    i.e. the persistent handle currently held IS an arc071_chunk trajectory
    (`meta["source"] == "arc071_chunk"`, per `_traj_chunk_seq`) -- which is
    what "wherever the ARC-071 chunk-injection pathway is live-tested" (the
    autopsy's own scoping of the finding) actually means.
    """
    n_ticks = len(records)
    n_e3_ticks = sum(1 for r in records if r["e3_tick"])
    chunk_ticks = [r for r in records if r["persistent_chunk_seq"] is not None]
    n_persistent_present = len(chunk_ticks)
    n_raw_present_on_action = sum(
        1 for r in records if r["raw_chunk_seq"] is not None)
    n_raw_present_post_teardown = sum(
        1 for r in records if r.get("post_raw_committed_trajectory_present"))

    # Independent identity-based reading: average ticks-per-genuine-commit,
    # counting a "genuine commit" as a change in `persistent_trajectory_id`
    # WHILE the persistent handle is chunk-sourced. This is read purely from
    # object identity, entirely outside `_committed_chunk_state()`'s own read
    # path (which reads `_committed_step_idx`, never identity).
    n_genuine_commits = 0
    prev_chunk_id: Optional[int] = None
    for r in chunk_ticks:
        if r["persistent_trajectory_id"] != prev_chunk_id:
            n_genuine_commits += 1
        prev_chunk_id = r["persistent_trajectory_id"]
    mean_true_dwell_ticks = (
        n_persistent_present / n_genuine_commits if n_genuine_commits else 0.0)

    # Frozen-index diagnostic: among BETWEEN-tick steps (e3_tick == False)
    # where beta is elevated and the persistent handle is chunk-sourced --
    # i.e. exactly the steps the MECH-090 Layer-1 fast path is SUPPOSED to
    # advance committed_step_idx on -- did the OFFICIAL step index actually
    # change from the previous tick's reading?
    n_eligible_between_tick = 0
    n_frozen_between_tick = 0
    for i in range(1, n_ticks):
        r, prev = records[i], records[i - 1]
        if (r["e3_tick"] or not r["beta_elevated"]
                or r["persistent_chunk_seq"] is None):
            continue
        n_eligible_between_tick += 1
        if r["committed_step_idx"] == prev["committed_step_idx"]:
            n_frozen_between_tick += 1
    frac_between_tick_frozen = (
        n_frozen_between_tick / n_eligible_between_tick
        if n_eligible_between_tick else None)

    official_lengths, official_n_segments = _segment_official_readout(records)
    official_mean_realised_length = (
        statistics.fmean(official_lengths) if official_lengths else 0.0)
    official_pct_length_one = (
        sum(1 for L in official_lengths if L == 1) / len(official_lengths)
        if official_lengths else 0.0)

    frac_post_teardown_still_present = (
        n_raw_present_post_teardown / n_ticks if n_ticks else 0.0)

    # PRIMARY verdict: routes on official_readout_mean_realised_length (see
    # the PERSISTENCE_RESTORED_* threshold block's comment for why this is
    # the correct statistic post-fix, not the true-dwell/frozen-fraction
    # pair -- those remain in the returned dict below as supplementary
    # diagnostics only, not verdict inputs.
    if n_persistent_present < MIN_PERSISTENT_PRESENT_TICKS:
        cell_verdict = "insufficient_commitments"
    elif official_mean_realised_length >= PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR:
        cell_verdict = "persistence_restored_post_fix"
    elif official_mean_realised_length <= PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL:
        cell_verdict = "persistence_still_broken_post_fix"
    else:
        cell_verdict = "inconclusive"

    return {
        "n_probe_ticks": n_ticks,
        "n_e3_ticks": n_e3_ticks,
        "e3_tick_rate": (n_e3_ticks / n_ticks) if n_ticks else 0.0,
        "n_persistent_present_ticks": n_persistent_present,
        "n_raw_committed_present_on_action_ticks": n_raw_present_on_action,
        "n_raw_committed_present_post_teardown_ticks": n_raw_present_post_teardown,
        "frac_post_teardown_still_present": frac_post_teardown_still_present,
        "n_genuine_commits_identity_based": n_genuine_commits,
        "mean_true_dwell_ticks": mean_true_dwell_ticks,
        "n_eligible_between_tick_steps": n_eligible_between_tick,
        "n_frozen_between_tick_steps": n_frozen_between_tick,
        "frac_between_tick_steps_idx_frozen": frac_between_tick_frozen,
        "official_readout_realised_lengths": official_lengths,
        "official_readout_n_segments": official_n_segments,
        "official_readout_mean_realised_length": official_mean_realised_length,
        "official_readout_pct_length_one": official_pct_length_one,
        "beta_gate_bistable": bool(
            getattr(agent.config.heartbeat, "beta_gate_bistable", False)),
        "cell_verdict": cell_verdict,
    }


def _run_cell(seed: int, *, n_train: int, n_probe: int) -> Dict[str, Any]:
    """One seed's formation phase, then instrumented probe phase, on ONE
    agent (same "one agent, two phases" pattern as V3-EXQ-841 -- the probe
    must read the commit-latch state of the agent that actually formed the
    chunks)."""
    flags = _arm_flags()
    train_every = max(1, min(30, n_train // 2))
    # include_driver_script_in_hash left at its default (True): this is a
    # one-off diagnostic probe, not a baseline/mint candidate for the
    # arc071_chunking lineage's arm-reuse economy (it has no OFF arm and is
    # excluded from confidence scoring as EXPERIMENT_PURPOSE="diagnostic"),
    # so cross-driver reuse is not a goal here -- the driver script's own
    # content (including the H1/H2 threshold constants below) is correctly
    # folded into the fingerprint instead of needing separate config_slice
    # declaration.
    with arm_cell(seed,
                  config_slice=_config_slice(),
                  config_slice_declared=True,
                  script_path=Path(__file__)) as cell:
        print(f"Seed {seed} Condition {ARM_ID} (commit-latch diagnostic)", flush=True)

        env = base.build_env(seed)
        agent = base.build_agent(env, flags)
        harness = StepHarness(agent, env, train_mode=False, seed=seed)

        # V3-EXQ-871a fix: verify the flag is genuinely at the lineage
        # default (False) going INTO formation -- this is the exact assumption
        # V3-EXQ-871 violated (it baked the flag into `flags` above, so it was
        # already True here). Fail loudly rather than silently repeat 871's bug
        # if `_arm_flags()` or `base.build_agent()` ever changes underneath us.
        assert agent.config.use_persistent_committed_program_handle is False, (
            "V3-EXQ-871a precondition violated: "
            "use_persistent_committed_program_handle must be False (lineage "
            "default) entering _run_formation, or formation is perturbed "
            "exactly as it was in V3-EXQ-871. Check _arm_flags()."
        )

        train = _run_formation(harness, agent, n_train,
                               progress_every=train_every,
                               label=f"{ARM_ID} seed={seed}")

        # V3-EXQ-871a fix: scope the flag to the PROBE PHASE ONLY. Flipped on
        # the live agent config object here -- AFTER formation, BEFORE the
        # probe -- rather than baked into `flags`/build_agent() as V3-EXQ-871
        # did. Safe because agent.py:5609-5612 / 6009-6012 read this flag LIVE
        # via getattr(self.config, ...) on every tick, never cached at
        # construction, so this mutation takes effect immediately without
        # rebuilding the agent (which would discard the chunks `train` just
        # formed). See module docstring "STATUS UPDATE (2026-08-02, V3-EXQ-871a
        # fix)" for the full mechanism and the failure this corrects.
        agent.config.use_persistent_committed_program_handle = True

        probe = _run_instrumented_probe(harness, n_probe)
        analysis = _analyze_cell(probe["records"], agent)

        row: Dict[str, Any] = {
            "arm": ARM_ID,
            "seed": seed,
            "chunk_max_size": CHUNK_MAX_SIZE,
            "chunks_formed": float(
                train["chunking_state"].get("chunk_acc_n_formed", 0)),
            "chunks_crystallised": float(
                train["chunking_state"].get("chunk_lib_n_crystallised", 0)),
            "formed_chunk_lengths": train["formed_chunk_lengths"],
            "chunking_instantiated": train["chunking_instantiated"],
            # V3-EXQ-871a audit trail: confirms the flag really was False
            # throughout formation and True throughout the probe for this
            # cell (belt-and-braces alongside the assert above, which would
            # already have raised if this were ever False here).
            "use_persistent_committed_program_handle_scoped_to_probe": bool(
                agent.config.use_persistent_committed_program_handle),
            **analysis,
            "tick_records": probe["records"],
        }
        cell.stamp(row)
        _ZG.observe(agent)  # AFTER stepping -- reads the counters at call time
        print(f"verdict: {analysis['cell_verdict']}", flush=True)
    return row


def run_experiment(*, n_train: int, n_probe: int,
                   seeds: List[int]) -> Dict[str, Any]:
    rows = [_run_cell(seed, n_train=n_train, n_probe=n_probe) for seed in seeds]

    verdicts = [r["cell_verdict"] for r in rows]
    min_persistent_present = min(r["n_persistent_present_ticks"] for r in rows)
    readiness_met = bool(
        min_persistent_present >= MIN_PERSISTENT_PRESENT_TICKS)

    definitive = {"persistence_restored_post_fix", "persistence_still_broken_post_fix"}
    if not readiness_met:
        label = "substrate_not_ready_requeue"
    elif all(v == "persistence_restored_post_fix" for v in verdicts):
        label = "persistence_restored_confirmed"
    elif all(v == "persistence_still_broken_post_fix" for v in verdicts):
        label = "persistence_still_broken_confirmed"
    elif all(v in definitive for v in verdicts) and len(set(verdicts)) > 1:
        label = "mixed_across_seeds"
    else:
        label = "inconclusive"

    c1_sufficient_commitments = readiness_met
    c2_discrimination_reached = bool(
        label in ("persistence_restored_confirmed",
                   "persistence_still_broken_confirmed"))
    criteria = [
        {"name": "C1_sufficient_commitments_observed",
         "load_bearing": True, "passed": c1_sufficient_commitments},
        {"name": "C2_discrimination_reached",
         "load_bearing": True, "passed": c2_discrimination_reached},
    ]
    overall_pass = c1_sufficient_commitments and c2_discrimination_reached

    preconditions = [{
        "name": "arc071_chunk_commitments_observed_supra_floor",
        "description": (
            "Gate (a)-analog: the probe agent must register some genuine "
            "ARC-071 CHUNK-SOURCED commitment (agent.e3."
            "_persistent_committed_trajectory present AND its metadata "
            "source == 'arc071_chunk') during the probe, or there is "
            "nothing ARC-071-specific to discriminate H1 from H2 with. "
            "Scoped to chunk-sourced commitments specifically (not any "
            "MECH-090 commitment) because _persistent_committed_trajectory "
            "is set on ANY committed selection regardless of candidate "
            "provenance -- confirmed by this script's own --dry-run smoke "
            "test, which at the toy 2-episode formation schedule formed "
            "zero chunks yet still registered ordinary commitments."),
        "control": (
            "Count of probe ticks with agent.e3._persistent_committed_trajectory "
            "not None AND metadata['source'] == 'arc071_chunk', worst seed."),
        "measured": float(min_persistent_present),
        "threshold": float(MIN_PERSISTENT_PRESENT_TICKS),
        "direction": "lower",
        "met": readiness_met,
    }]

    return {
        "rows": rows,
        "label": label,
        "criteria": criteria,
        "overall_pass": overall_pass,
        "preconditions": preconditions,
        "readiness_met": readiness_met,
    }


def main() -> Tuple[str, Path, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.dry_run:
        n_train, n_probe, seeds = 2, 2, SEEDS
    else:
        n_train, n_probe, seeds = N_EPISODES_TRAIN, N_EPISODES_PROBE, SEEDS

    res = run_experiment(n_train=n_train, n_probe=n_probe, seeds=seeds)
    outcome = "PASS" if res["overall_pass"] else "FAIL"
    label = res["label"]

    if label == "persistence_restored_confirmed":
        # Fix (a) restores meaningful multi-step commit-latch persistence
        # under REAL organic chunk formation -- supports MECH-090 (the latch
        # holds, once correctly wired to the SD-084 persistent handle).
        direction_mech090 = "supports"
        direction_arc071 = "unknown"
    elif label == "persistence_still_broken_confirmed":
        # Fix (a) alone is insufficient under organic formation -- gap (b)
        # (E3 unconditionally re-committing a new trajectory object every E3
        # tick) still dominates -- weakens MECH-090 as currently wired.
        direction_mech090 = "weakens"
        direction_arc071 = "unknown"
    else:
        direction_mech090 = "unknown"
        direction_arc071 = "unknown"

    manifest: Dict[str, Any] = {
        "run_id": (f"{EXPERIMENT_TYPE}_"
                   f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"),
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(args.dry_run),
        "supersedes": SUPERSEDES_RUN_ID,
        "supersedes_queue_id": SUPERSEDES_QUEUE_ID,
        "supersession_note": (
            "V3-EXQ-871 applied use_persistent_committed_program_handle=True "
            "for the WHOLE cell (formation + probe), perturbing formation and "
            "collapsing chunks_formed on seeds 101/505 from V3-EXQ-841's 2/2 "
            "to 1/0, starving the readiness precondition and forcing a "
            "substrate_not_ready_requeue FAIL -- confirmed pure "
            "probe-construction defect (measurement_test_design_defect), not "
            "a substrate finding (failure_autopsy_V3-EXQ-871_2026-08-02.json). "
            "This run scopes the flag to the probe phase only (flipped on the "
            "live agent config after _run_formation, before "
            "_run_instrumented_probe); formation now replays 841's proven "
            "lineage-default dynamics unperturbed."
        ),
        "discrimination_target": "failure_autopsy_V3-EXQ-841_2026-07-31.json "
                                  "targets[0].fanout_recommendation "
                                  "(H1/H2 already resolved by an ad hoc probe -- "
                                  "diagnostic_arc071_commit_latch_h1h2_probe_"
                                  "2026-07-31.md, H1 confirmed/H2 refuted; this run "
                                  "validates whether ree-v3 main commit 278599a "
                                  "restores persistence under organic formation, "
                                  "under the V3-EXQ-871a probe-phase-only flag "
                                  "scoping -- see supersession_note)",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "arm_results": res["rows"],
        "evidence_direction": "mixed" if res["overall_pass"] else "non_contributory",
        "evidence_direction_per_claim": {
            "MECH-090": direction_mech090,
            "ARC-071": direction_arc071,
        },
        "interpretation": {
            "label": label,
            "preconditions": res["preconditions"],
            "criteria": res["criteria"],
            "criteria_non_degenerate": {
                "C1_sufficient_commitments_observed": True,
                "C2_discrimination_reached": res["readiness_met"],
            },
        },
        "h1_h2_note": (
            "H1/H2 are NOT live in this run -- already resolved by an ad hoc "
            "probe (diagnostic_arc071_commit_latch_h1h2_probe_2026-07-31.md): "
            "H1 (real ree_core wiring defect) CONFIRMED, H2 (experiment "
            "readout artifact) REFUTED -- the official read path was shown to "
            "faithfully report the substrate's true (broken) state, 0 "
            "mismatches across 160 ticks. This run instead validates whether "
            "the landed fix for one of the two wiring gaps that diagnosis "
            "found (ree-v3 main commit 278599a, the between-E3-tick stepping "
            "branch's fallback to agent.e3._persistent_committed_trajectory, "
            "forced on here via use_persistent_committed_program_handle=True, "
            "flipped on the live agent config AFTER formation and BEFORE the "
            "probe -- V3-EXQ-871a's fix to its predecessor V3-EXQ-871, which "
            "applied the flag to the whole cell and thereby perturbed "
            "formation itself; see supersession_note) "
            "restores meaningful multi-step persistence under REAL organic "
            "chunk formation, as opposed to the ad hoc probe's monkeypatched "
            "forced-chunk setup. label='persistence_restored_confirmed' means "
            "official_readout_mean_realised_length now clears "
            f"{PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR} in every cell -- the "
            "fix works under organic formation and the still-open gap (b) "
            "(E3 unconditionally re-committing a new trajectory object every "
            "E3 tick) does not, in practice, prevent meaningful persistence "
            "between E3 ticks. label='persistence_still_broken_confirmed' "
            "means realised length stays pinned near 1 even with the fix "
            "enabled -- gap (b) dominates and fix (a) alone does not resolve "
            "the substrate finding. mean_true_dwell_ticks / "
            "frac_between_tick_steps_idx_frozen are retained as supplementary "
            "identity-based diagnostics (see module docstring for why they "
            "are NOT the verdict-routing statistic): a high true-dwell "
            "combined with a still-high frozen-fraction would indicate the "
            "SD-084 persistent handle survives (as it always did) while the "
            "separate step-index counter still fails to advance -- exactly "
            "the H1 signature the ad hoc probe found, not evidence of a "
            "readout misread."),
    }

    full_config: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {"n_episodes_train": n_train, "steps_per_episode":
                     STEPS_PER_EPISODE, "n_episodes_probe": n_probe},
        "arm": ARM_ID,
        "arm_config_slice": _config_slice(),
        "thresholds": {
            "MIN_PERSISTENT_PRESENT_TICKS": MIN_PERSISTENT_PRESENT_TICKS,
            "PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR":
                PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR,
            "PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL":
                PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL,
            "H2_MEAN_TRUE_DWELL_FLOOR_supplementary": H2_MEAN_TRUE_DWELL_FLOOR,
            "H2_FROZEN_FRACTION_FLOOR_supplementary": H2_FROZEN_FRACTION_FLOOR,
            "H1_MEAN_TRUE_DWELL_CEIL_supplementary": H1_MEAN_TRUE_DWELL_CEIL,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=bool(args.dry_run),
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"label: {label}", flush=True)
    for row in res["rows"]:
        print(f"  seed={row['seed']} verdict={row['cell_verdict']} "
              f"true_dwell={row['mean_true_dwell_ticks']:.2f} "
              f"official_pct_len1={row['official_readout_pct_length_one']:.2f} "
              f"frozen_frac={row['frac_between_tick_steps_idx_frozen']}", flush=True)
    for c in res["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} "
              f"load_bearing={c.get('load_bearing', False)}", flush=True)
    print(f"outcome: {outcome}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return outcome, out_path, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _out_path, _dry_run = main()
    _outcome_raw = str(_outcome).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
