#!/usr/bin/env python3
"""V3-EXQ-871b -- MECH-090/ARC-071 AT-tick commit-latch persistence: owed
validation of the E3-reselection short-circuit fix (ree-v3 main 31ed78e).

SLEEP DRIVER: n/a -- no sleep loop in this diagnostic (use_sleep_loop /
sws_enabled / rem_enabled / use_sleep_aggregation_cluster all False).

LINEAGE
-------
`failure_autopsy_V3-EXQ-871a_2026-08-03.md` diagnosed "gap (b)": under real
organic ARC-071 chunk formation, with the between-tick fallback
(`use_persistent_committed_program_handle=True`, ree-v3 main `278599a`)
already in effect, E3 UNCONDITIONALLY recreates a brand-new committed-
trajectory object on literally every E3 tick, regardless of an existing
unexpired commitment. Exact arithmetic, both seeds, zero exceptions:
`n_genuine_commits_identity_based == n_e3_ticks + n_episodes_probe`. The
autopsy also found the OLD verdict statistic
(`official_readout_mean_realised_length` / `mean_true_dwell_ticks`) confounded
by `e3_tick_rate` (varies ~4.4x by seed for reasons unrelated to the latch)
and explicitly prescribed the fix: "A future validation of this fix should
not reuse the current DV ... the correct discriminating statistic is
something like `n_genuine_commits / n_e3_ticks` ... not raw
`official_readout_mean_realised_length`" (Section 6, point 5).

The autopsy routed `/implement-substrate create
mech090-arc071-attick-persistent-handle-fix`. `substrate_queue.json`'s sd_id
`arc071_e3_reselection_on_committed_program` records that build as
`implemented_validated`: an "already committed to this unexpired program,
skip re-selection" short-circuit in `select_action`'s E3-tick branch
(`agent.py`), gated behind NEW flag `REEConfig.use_e3_reselection_shortcircuit`
(default False), landed ree-v3 main `31ed78e` on 2026-08-01. 5 contract tests
pass (`tests/contracts/test_arc071_e3_reselection_shortcircuit.py`). That
entry's own `implementation_note.not_yet_done` states explicitly: no existing
experiment enables the flag today, and a future validating driver needs its
own run enabling BOTH `use_persistent_committed_program_handle=True` AND
`use_e3_reselection_shortcircuit=True` together on chunking-ON arms. THIS RUN
IS THAT VALIDATION -- the one remaining piece of debt the routing left open
(GFLAG-0079).

WIRING VERIFIED AGAINST ree-v3 HEAD (2026-09-01, this session), not taken on
trust from the substrate_queue.json prose:
  * `agent.py:7202-7222` -- the short-circuit fires only when
    `ticks["e3_tick"]` AND `getattr(self.config,
    "use_e3_reselection_shortcircuit", False)` AND `self.beta_gate.is_elevated`
    all hold, AND the live `_persistent_committed_trajectory` is still
    `arc071_chunk`-sourced with `_committed_step_idx < chunk_length - 1`
    (`chunk_length` from `metadata['chunk_sequence']`, never
    `actions.shape[1]` -- matches this driver's own `_traj_chunk_seq`/
    `_committed_chunk_state` reads below, unchanged from V3-EXQ-871a).
  * `beta_gate.is_elevated` is NOT an extra precondition this run must
    engineer separately: it is the substrate's own invariant that an active
    MECH-090 commitment implies an elevated beta gate (confirmed by the
    short-circuit's own contract test fixture,
    `test_arc071_e3_reselection_shortcircuit.py:270-287`, which asserts
    `agent.beta_gate.is_elevated` as a fixture-validity check on "an active
    commitment"). This run's own per-tick instrumentation already records
    `beta_elevated` (inherited from V3-EXQ-871a's `on_action` hook,
    unchanged), so the assumption is verified from THIS run's own data, not
    merely cited.
  * Placement is AFTER every principled-release site (MECH-091 urgency
    interrupt, MECH-342 readiness release, the rung-6 duration release, the
    MECH-321 mid-execution decomposition hook, the VS/relief/safety releases,
    SD-084 reap-on-not-elevated) -- a genuine release this same tick already
    invalidates the short-circuit's own precondition, so MECH-091 can never
    be swallowed by it. Nothing about this run's design interacts with that
    ordering; it is a substrate property, not a driver concern.

Both flags are wired LIVE via `getattr(self.config, ..., False)` on every
tick (`agent.py:6749`, `agent.py:7155`, `agent.py:7204`), never cached at
agent-construction time -- confirmed identical to `use_persistent_
committed_program_handle`'s own live-read pattern, which V3-EXQ-871a already
relied on for its probe-phase-only flag scoping. So the SAME "flip after
formation, before probe" idiom V3-EXQ-871a used for the first flag applies
unchanged to the second.

SUBSTRATE-PATH OVERLAP CHECK (Step 2.5c, `/queue-experiment` skill) -- two
`corrupting` `substrate_queue.json` entries name bare `ree_core/agent.py`
(`mode-governance-engagement`, `SD-082`), which on a literal whole-file match
would overlap this driver's `agent.py` import. Both are confirmed PROVABLY
INERT for this run: `mode-governance-engagement`'s defect lives in the
`_et_commit`/`_et_engagement` external-task-drive salience term
(`agent.py:7567-7584`), gated on `use_external_task_drive=True` (its own
substrate_queue.json text: "this whole lineage is inert in production" when
that flag is False, which is this driver's unmodified default via
`base.off_arm_flags()`). `SD-082`'s defect lives in the SD-078 rule-pool ->
action-bias consumer, gated on `use_lateral_pfc_analog=True`
(`ree_core/utils/config.py:3842` default False, `agent.py:915-951`,
`lateral_pfc_analog.py::compute_bias`) -- also never set True here. Neither
flag appears anywhere in `_arm_flags()` below. A third, `degrading`-severity
entry (`SD-MECH303-THRESHOLD-SOURCING`) also names bare `agent.py` but
concerns `contextual_safety_harm_threshold`, unrelated to this driver's
chunk-injection / E3-tick causal chain; no note needed per the gate's own
`degrading` handling (only worth noting when the causal chain actually
touches it, which it does not here).

RE-DERIVE BRAKE CHECK (Step 2.5b) -- MECH-090 carries 11 counted
`substrate_ceiling` autopsy hits at time of writing, clearing the >=2
threshold. ALL 11 are the SD-034 cluster (delayed-reward persistence /
decommit-hold behavioural, a V2-era sub-question about held commitments
across a devaluation/reward-timing manipulation) -- a DIFFERENT mechanism at
a DIFFERENT granularity from the AT-tick re-selection-cadence question this
lineage (V3-EXQ-841/871/871a/871b) tests. V3-EXQ-871a's OWN autopsy already
made and recorded this exact distinction ("Already carries 7 confirmed
substrate_ceiling autopsies from an unrelated V2-era sub-question ... not the
same question as this run; the re-derive brake does not apply here") -- this
run is the direct continuation of that already-adjudicated lineage, not a new
circling of the SD-034 ceiling, so the brake does not apply here either, by
the same precedent.

DESIGN -- chunk_max_size=5 (A_HIER_S5), NOT 871a's chunk_max_size=2 --
CORRECTED AFTER AN ADVERSARIAL DESIGN REVIEW (Step 4.5, `/queue-experiment`
skill, model fable) FOUND chunk_max_size=2 STRUCTURALLY BLIND TO THE FIX
--------------------------------------------------------------------------
The original draft of this script replayed V3-EXQ-871a's exact design point
(A_HIER_S2, chunk_max_size=2), reasoning that 871a's own finding -- gap (b)
firing on literally EVERY E3 tick, zero exceptions, in both seeds -- was
already maximal at that size and so a same-size retest would be a clean
single-variable comparison. THE RED-TEAM REVIEW FOUND THIS WRONG: with
`chunk_max_size=2`, `chunk_length = 2`, so the short-circuit's own
continuation condition (`agent.py:7214`, `self._committed_step_idx <
chunk_length - 1` = `< 1`, i.e. `== 0`) can NEVER hold on any tick after the
very commit tick itself. Confirmed by direct trace of `agent.py:9591`
(`_committed_step_idx = 0` on new commitment) followed, within the SAME
tick, by the "Layer 1" step-through-committed-trajectory branch
(`agent.py:9757-9759`) which steps `actions[:, 0, :]` and increments
`_committed_step_idx` to 1 before the tick even ends. So on every
SUBSEQUENT E3 tick, `_committed_step_idx >= 1` and the short-circuit's own
expiry check judges a length-2 chunk already exhausted, falling through to
full re-deliberation regardless of whether the fix is engaged or not --
reproducing the pre-fix `ratio ~= 1.0` signature EVEN WHEN THE FIX WORKS
CORRECTLY. At `chunk_max_size=2` the "shortcircuit_working_post_fix" outcome
is mechanically unreachable, so the run could only ever conclude "inert" --
a manipulation-cannot-reach-DV defect, not a genuine null result. This
matches the fix's OWN contract tests, which deliberately use chunk lengths
5 and 15 (`tests/contracts/test_arc071_e3_reselection_shortcircuit.py`,
SEQ5/SEQ15), never 2.

**Fix applied**: `chunk_max_size=5` (`ARM_ID = "A_HIER_S5"`) -- one of
V3-EXQ-841's own three tested dose arms (alongside S2/S3), on the IDENTICAL
120ep/72-step formation schedule 841 proved commits >=1 chunk in 32/32
chunking cells at this size too (not just S2). A length-5 chunk gives idx
values 1, 2, 3 all satisfying `< chunk_length - 1 = 4` -- three genuine
opportunities per chunk for the short-circuit to fire, versus zero at
length 2. Seeds remain [101, 505] (871a's own seeds, chosen for a different
reason -- cleanest S2 reproduction of 841's effect -- but nothing in that
choice depends on chunk_max_size, and `arm_cell`'s per-seed full RNG reset,
`experiments/_lib/arm_fingerprint.py`, means this script (unlike the ad hoc
`diagnostic_arc071_e3_reselection_probe_2026-08-01.md`, which lacked that
reset and suffered a cross-seed torch-RNG-carryover artifact at exactly
chunk_max_size=5/seed-overlap) does not inherit that artifact -- each seed
cell's RNG state is independent by construction).

**Compounding gate added**: the red-team review also found the original
readiness precondition (some chunk-sourced commitment merely EXISTS) does
not verify the short-circuit ever had an opportunity to fire at all -- it
would pass on data with zero such opportunities and let the run confidently
certify "inert" on a vacuous input. `_analyze_cell` below now also computes
`n_shortcircuit_opportunities` (an E3 tick where, going INTO that tick, an
unexpired arc071_chunk-sourced commitment already existed with room to
continue) and gates on it explicitly -- see `_analyze_cell` and
`run_experiment`.

A SECOND design flaw, found by a re-spawned Step 4.5 review (fable) verifying
the chunk_max_size fix above: even with genuine opportunities present, the
autopsy's own literal DV (`n_genuine_commits_ticks_ratio =
n_genuine_commits_identity_based / n_e3_ticks`) denominates over ALL E3
ticks, most of which are NOT opportunity ticks (fresh commits at chunk
boundaries, natural completions, non-chunk selections) -- ticks the fix
cannot possibly affect. The fix's own motivating probe
(`diagnostic_arc071_e3_reselection_probe_2026-08-01.md`) measured only a
36.8% premature-reselection rate among chunk-committed decisions at
chunk_max_size=5, meaning opportunity ticks are a MINORITY even among
chunk-committed ticks -- so even a perfectly working short-circuit could
leave the unconditional ratio well inside the "working" band, making
"fix is inert" and "fix works but opportunities are structurally rare"
print identically. **Fix applied**: the verdict now routes on
`opportunity_continuation_rate` -- conditioned on exactly the ticks
`n_shortcircuit_opportunities` already isolates, so it cannot be diluted by
ticks the fix cannot touch. Pre-fix this rate is EXACTLY 0.0 by construction
(871a's own exact-arithmetic finding: every opportunity, and every tick, was
squandered) -- see `_analyze_cell` and the threshold block above for the
full reasoning and the retained supplementary statistics.

WHAT CHANGED FROM V3-EXQ-871a
-------------------------------
1. `agent.config.use_e3_reselection_shortcircuit = True` is set alongside
   `agent.config.use_persistent_committed_program_handle = True` in
   `_run_cell`, AFTER `_run_formation()` returns and BEFORE
   `_run_instrumented_probe()` begins -- mirroring 871a's own probe-phase-only
   scoping for the first flag, for the identical reason (formation must
   replay V3-EXQ-841's proven organic dynamics unperturbed; V3-EXQ-871's own
   whole-cell-flag defect is exactly the mistake this scoping avoids).
2. THE DV IS REPLACED, TWICE OVER (see "DESIGN" above for the full story).
   First pass: this run adds the autopsy's own prescribed ratio,
   `n_genuine_commits_ticks_ratio = n_genuine_commits_identity_based /
   n_e3_ticks` (both terms already computed by 871a's own `_analyze_cell` as
   the ingredients of `e3_tick_rate`/`mean_true_dwell_ticks`). Second pass,
   after a SECOND Step 4.5 red-team review: that ratio's denominator (ALL E3
   ticks) is diluted by ticks the fix cannot possibly affect, so even a
   perfectly working short-circuit could leave it well inside the "working"
   band -- an unattributable result. THE ACTUAL VERDICT-ROUTING STATISTIC IS
   `opportunity_continuation_rate` (conditioned only on E3 ticks where
   continuation was structurally possible -- see `_analyze_cell`).
   `n_genuine_commits_ticks_ratio_supplementary` / `official_readout_mean_
   realised_length_supplementary` / `mean_true_dwell_ticks` /
   `frac_between_tick_steps_idx_frozen_supplementary` are STILL COMPUTED
   (useful supplementary diagnostics, and needed to detect the between-tick
   fallback (fix a) regressing) but are explicitly NOT criteria inputs.
3. Module docstring, `EXPERIMENT_TYPE`, `CLAIM_IDS` (unchanged: MECH-090,
   ARC-071 -- see 871a's own "CLAIM_IDS" reasoning, which applies unchanged),
   thresholds, and verdict labels updated for this run's own question. No
   `supersedes` -- 871a's own result stands (it correctly diagnosed gap (b));
   this run does not invalidate it, it validates the fix gap (b) motivated.
4. `chunk_max_size` is 5 (A_HIER_S5), NOT 871a's 2 (A_HIER_S2) -- see
   "DESIGN" above for why chunk_max_size=2 makes the short-circuit's own
   firing condition unreachable, found by the mandatory Step 4.5 adversarial
   design review. A new `n_shortcircuit_opportunities` readiness precondition
   was added for the same reason (see `_analyze_cell`).

CLAIM_IDS -- MECH-090 and ARC-071, unchanged from V3-EXQ-871a
-----------------------------------------------------------------
Same reasoning as 871a's own docstring: this tests whether the MECH-090
commit latch holds ACROSS TICKS (now with BOTH landed fixes engaged) under
live ARC-071 chunk injection (the vehicle exposing the mechanism). MECH-163/
Q-085 (dose-response, devaluation) are not touched. MECH-323 (chunk
formation/crystallisation) is exercised as a precondition only, not itself
in question -- same exclusion 841/871a already applied.

GOVERNANCE NOTE (not applied by this script or this session): once this run
lands, MECH-090's `pending_retest_after_substrate` (claims.yaml) and
GFLAG-0079 (governance_flags.v1.json) are both governance dispositions for
`/governance` to apply, not this queueing session's call.

ETHICS PREFLIGHT (descriptive, non-enforced; SENT-0 -- V3 is not claimed
sentient): identical posture to V3-EXQ-871a -- involves_negative_valence
False (num_hazards=2 harm stream is pre-ethical instrumentation, unchanged),
involves_suffering_like_state False, involves_self_model False,
involves_inescapability False, involves_offline_replay_over_harm False
(sleep off), involves_social_mind False, involves_human_data False.
decision: allow.

RED-TEAM (fable): TWO passes, per the skill's "re-spawn exactly once, only
when a BLOCKING finding forces a change to the causal chain" rule -- pass 1
found the manipulation (chunk_max_size=2) unreachable; the fix (chunk_max_size
-> 5) changed the manipulation, licensing the one allowed re-spawn. Pass 2
found the CRITERION (unconditional n_genuine_commits_ticks_ratio) diluted and
unable to discriminate even with opportunities present; fixed by routing on
opportunity_continuation_rate instead (a different criterion). Both findings
verified against ree_core/agent.py source directly by this session before
being fixed, per the skill's "verify before acting" instruction -- not
applied on the reviewer's say-so alone. Per the skill's one-pass rule, this
second fix is NOT re-verified by a third spawn (it was accepted as sound by
direct source verification); see queue entry `note` for the final
disposition summary.
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

# z_goal liveness -- each seed cell builds a FRESH agent inside `_run_cell`,
# so the accumulator (not a single `agent=` kwarg) is correct, same pattern
# as V3-EXQ-871a.
_ZG = ZGoalStreamAccumulator()

EXPERIMENT_TYPE = "v3_exq_871b_mech090_e3_reselection_shortcircuit_retest"
EXPERIMENT_PURPOSE = "diagnostic"

CLAIM_IDS = ["MECH-090", "ARC-071"]

SEEDS = [101, 505]  # 871a's own seeds -- see module docstring "DESIGN"
ARM_ID = "A_HIER_S5"
CHUNK_MAX_SIZE = 5  # NOT 871a's 2 -- see module docstring "DESIGN" for why
                     # chunk_max_size=2 makes the short-circuit unreachable

N_EPISODES_TRAIN = base.N_EPISODES          # 120 -- the 810a-proven formation schedule
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE  # 72
N_EPISODES_PROBE = 10                       # identical to V3-EXQ-871a

# Readiness-anchor reachability -- identical inheritance argument as
# V3-EXQ-871a: formation is run under the agent's constructed default for
# BOTH new-and-old flags (both False during _run_formation, flipped True on
# the live config only afterward), so this run's formation replays 841's
# proven 2/2-chunk dynamics on seeds 101/505 unperturbed. Confirmed by this
# script's own --dry-run smoke test, whose readiness gate correctly reports
# substrate_not_ready_requeue at the toy 2-episode schedule (unreachable only
# at that scale, exactly as in 871a).
ANCHOR_REACHABILITY_EXEMPT = (
    "reachability already established by V3-EXQ-810a (8/8 seeds formed "
    "chunks), V3-EXQ-841 (32/32 chunking cells committed >=1 chunk), and "
    "V3-EXQ-871a (same schedule/config, same probe-phase-only flag-scoping "
    "idiom, readiness met on both seeds) on the IDENTICAL inherited formation "
    "schedule/config. This run changes nothing about formation, only which "
    "flags are live during the probe phase, so re-deriving reachability here "
    "would just re-run 810a/841/871a's own already-recorded result at extra "
    "compute cost."
)

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
# P0 readiness: identical floor to V3-EXQ-871a -- some genuine ARC-071
# chunk-sourced commitment must register, or there is nothing to discriminate.
MIN_PERSISTENT_PRESENT_TICKS = 10

# n_genuine_commits_ticks_ratio = n_genuine_commits_identity_based / n_e3_ticks
# is the autopsy's OWN literally-prescribed statistic (Section 6/7). It is
# STILL COMPUTED below and reported (module docstring point 2), but a SECOND
# Step 4.5 red-team review (fable) found it CANNOT be the verdict-routing
# statistic: the denominator (ALL E3 ticks) is diluted by ticks where no
# continuation was ever structurally possible (a fresh commit at a chunk
# boundary, a natural chunk completion, a non-chunk-sourced selection) --
# ticks the short-circuit cannot possibly affect regardless of whether it
# works. Because "opportunity" ticks (an unexpired, room-remaining,
# beta-elevated chunk commitment already existed going in) are a MINORITY of
# all E3 ticks even at chunk_max_size=5 (the fix's own motivating probe,
# diagnostic_arc071_e3_reselection_probe_2026-08-01.md, measured a 36.8%
# premature-reselection rate among chunk-committed decisions at this size --
# i.e. most chunk-committed E3 ticks are NOT opportunity ticks at all), even
# a PERFECTLY working short-circuit could only reduce
# n_genuine_commits_ticks_ratio by removing re-commits on the minority
# opportunity subset, leaving it well inside (or above) the working ceiling
# regardless of ground truth -- an unattributable result: "inert" and
# "fix works but opportunities are rare" would print identically.
#
# PRIMARY verdict-routing statistic (added after this second review):
# `opportunity_continuation_rate = n_opportunity_ticks_continued /
# n_shortcircuit_opportunities` -- conditioned ONLY on ticks where continuing
# was structurally possible (the same population the readiness precondition
# below already isolates), so it is never diluted by ticks the fix cannot
# touch. Pre-fix, by construction (V3-EXQ-871a: literally every E3 tick
# reinstalls a fresh object, zero exceptions), this rate is EXACTLY 0.0 --
# every opportunity is squandered. Post-fix "working" means the short-circuit
# continues the SAME object on most of its own opportunities.
OPPORTUNITY_CONTINUATION_RATE_WORKING_FLOOR = 0.5   # rate >= this -> short-
                                                       # circuit measurably
                                                       # continues on its own
                                                       # opportunities
OPPORTUNITY_CONTINUATION_RATE_INERT_CEIL = 0.1      # rate <= this ->
                                                       # statistically
                                                       # indistinguishable
                                                       # from the pre-fix 0.0
                                                       # baseline

# Supplementary only (NOT verdict-routing -- see above): the autopsy's own
# literal statistic, retained and reported because it is what the autopsy
# text asked for, but demonstrated by the second red-team pass to have a
# diluted-denominator ceiling that makes it unsuitable as the discriminator.
RATIO_SHORTCIRCUIT_WORKING_CEIL_supplementary = 0.5
RATIO_SHORTCIRCUIT_INERT_FLOOR_supplementary = 0.9

# Supplementary diagnostics (retained from V3-EXQ-871a -- NOT verdict-routing
# in this run; see module docstring point 2 for why the autopsy prescribes
# against reusing these as criteria).
PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR_supplementary = 3.0
PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL_supplementary = 1.5

RELEASES_PINNED_OFF = {
    "use_maintenance_release": False,
    "use_natural_commit_urgency_release": False,
    "use_vs_commit_release": False,
    "use_closure_decommit": False,
}


def _arm_flags() -> Dict[str, Any]:
    """A_HIER_S2 flags, identical to V3-EXQ-871a's `_arm_flags()`. Neither new
    nor old persistence/short-circuit flag is set here -- both are False at
    agent construction (lineage default), so `_run_formation` in `_run_cell`
    below replays V3-EXQ-841/871a's proven formation dynamics unperturbed.
    Both flags are instead flipped directly on the live agent's config object
    in `_run_cell`, AFTER `_run_formation` completes and BEFORE
    `_run_instrumented_probe` begins -- see module docstring "WHAT CHANGED
    FROM V3-EXQ-871a" point 1.
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
        "diagnostic": "mech090_e3_reselection_shortcircuit_retest",
        "use_persistent_committed_program_handle_probe_only": True,
        "use_e3_reselection_shortcircuit_probe_only": True,
    }
    return slice_


def _traj_chunk_seq(traj: Any) -> Optional[Tuple[int, ...]]:
    """Chunk-sequence identity iff `traj` is a live arc071_chunk trajectory.
    Identical to V3-EXQ-871a's function of the same name."""
    meta = getattr(traj, "metadata", None) if traj is not None else None
    if not meta or meta.get("source") != "arc071_chunk":
        return None
    seq = tuple(int(a) for a in meta.get("chunk_sequence", ()))
    return seq or None


def _committed_chunk_state(agent: Any) -> Tuple[Optional[Tuple[int, ...]], int]:
    """VERBATIM reproduction of V3-EXQ-841/871a's own read path."""
    e3 = getattr(agent, "e3", None)
    if e3 is None:
        return None, 0
    traj = getattr(e3, "_committed_trajectory", None)
    if traj is None:
        traj = getattr(e3, "_persistent_committed_trajectory", None)
    return _traj_chunk_seq(traj), int(getattr(agent, "_committed_step_idx", 0))


def _run_formation(harness: Any, agent: Any, n_episodes: int, *,
                   progress_every: int, label: str) -> Dict[str, Any]:
    """The formation phase. Identical to V3-EXQ-871a's `_run_formation` (same
    lineage rationale: the probe phase must run on the SAME agent that formed
    the chunks)."""
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
    """The instrumentation phase. Identical hook wiring to V3-EXQ-871a's
    `_run_instrumented_probe` -- `on_action` fires immediately after
    `select_action()` (this is the moment the short-circuit, if it fired,
    already returned an action stepping the EXISTING persistent trajectory
    rather than a freshly-installed one -- so `persistent_trajectory_id`
    changing, or not, on an E3 tick is exactly what distinguishes the
    short-circuit engaging from a fresh recommit).

    `n_genuine_commits` counts every tick on which
    `id(agent.e3._persistent_committed_trajectory)` CHANGES from the previous
    tick, read purely from Python object identity -- unchanged from 871a.
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

    harness.hooks.on_action = None
    harness.hooks.on_post_step = None

    return {"records": records, "n_genuine_commits": state["n_genuine_commits"]}


def _segment_official_readout(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], int]:
    """Reproduces V3-EXQ-841/871a's `_segment_commitments` rule EXACTLY.
    Retained for the supplementary official-readout diagnostic only -- NOT
    the verdict-routing statistic in this run (see module docstring point 2).
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
    """SCOPED TO GENUINE ARC-071 CHUNK COMMITMENTS ONLY -- identical scoping
    rationale to V3-EXQ-871a (an un-scoped identity count mixes in ordinary,
    non-chunk MECH-090 commitments; confirmed by this script's own --dry-run
    smoke test).

    THE PRIMARY VERDICT STATISTIC IN THIS RUN is
    `opportunity_continuation_rate` -- see the threshold block's comment
    above for why the autopsy's own literal `n_genuine_commits/n_e3_ticks`
    (still computed here as `n_genuine_commits_ticks_ratio_supplementary`)
    is retained only as a supplementary diagnostic, not the router.
    """
    n_ticks = len(records)
    n_e3_ticks = sum(1 for r in records if r["e3_tick"])
    chunk_ticks = [r for r in records if r["persistent_chunk_seq"] is not None]
    n_persistent_present = len(chunk_ticks)
    n_raw_present_on_action = sum(
        1 for r in records if r["raw_chunk_seq"] is not None)
    n_raw_present_post_teardown = sum(
        1 for r in records if r.get("post_raw_committed_trajectory_present"))

    # Identity-based genuine-commit count, scoped to chunk-sourced ticks --
    # identical method to V3-EXQ-871a.
    n_genuine_commits = 0
    prev_chunk_id: Optional[int] = None
    for r in chunk_ticks:
        if r["persistent_trajectory_id"] != prev_chunk_id:
            n_genuine_commits += 1
        prev_chunk_id = r["persistent_trajectory_id"]
    mean_true_dwell_ticks = (
        n_persistent_present / n_genuine_commits if n_genuine_commits else 0.0)

    # PRIMARY: tick-rate-normalised commit rate. Scoped to n_e3_ticks (the
    # ONLY ticks a re-commit could possibly occur on -- between-tick steps
    # never call select() at all), matching the autopsy's own denominator.
    n_genuine_commits_ticks_ratio = (
        n_genuine_commits / n_e3_ticks if n_e3_ticks else None)

    # Supplementary diagnostics -- frozen-index and official-readout
    # statistics, retained unchanged from V3-EXQ-871a but NOT used for the
    # verdict in this run (see module docstring point 2).
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

    # Fraction of E3 ticks (while a chunk-sourced commitment was live) on
    # which beta_gate was elevated -- verifies the module docstring's claim
    # that an active commitment implies elevation, i.e. that the
    # short-circuit's `beta_gate.is_elevated` precondition is not starving
    # the run of opportunities to fire.
    e3_chunk_ticks = [r for r in records if r["e3_tick"]
                      and r["persistent_chunk_seq"] is not None]
    frac_e3_chunk_ticks_beta_elevated = (
        sum(1 for r in e3_chunk_ticks if r["beta_elevated"])
        / len(e3_chunk_ticks) if e3_chunk_ticks else None)

    # READINESS-KIND PRECONDITION (added after the Step 4.5 red-team review
    # found the original chunk_max_size=2 design gave the short-circuit ZERO
    # opportunities to ever fire -- see module docstring "DESIGN"). An
    # "opportunity" is an E3 tick where, going INTO that tick (i.e. per the
    # PREVIOUS record -- `on_action` fires immediately after select_action(),
    # so a tick's own record already reflects that tick's outcome), an
    # unexpired arc071_chunk-sourced commitment already existed with room
    # left to continue (`existing_readout_step_idx < chunk_length - 1`) AND
    # beta was elevated (the short-circuit's own extra precondition,
    # `agent.py:7205`). Without at least one such opportunity, a
    # "shortcircuit_inert_still_broken" verdict is indistinguishable from a
    # design that never gave the fix a chance to act -- exactly the
    # compounding gate defect the review found in 871a's own readiness
    # precondition (existence of a commitment, not existence of an
    # opportunity to continue one).
    # PRIMARY: opportunity-conditional continuation rate (added after the
    # SECOND Step 4.5 red-team review found the unconditional
    # n_genuine_commits_ticks_ratio diluted by non-opportunity ticks -- see
    # the threshold block's comment above). `n_opportunity_ticks_continued`
    # counts, among exactly the same opportunity ticks
    # `n_shortcircuit_opportunities` identifies, how many did NOT install a
    # new persistent-trajectory object (id() unchanged from the immediately
    # preceding tick) -- i.e. the short-circuit actually fired.
    n_shortcircuit_opportunities = 0
    n_opportunity_ticks_continued = 0
    for i in range(1, n_ticks):
        r, prev = records[i], records[i - 1]
        if not r["e3_tick"]:
            continue
        prev_seq = prev["existing_readout_chunk_seq"]
        if prev_seq is None or not prev["beta_elevated"]:
            continue
        if prev["existing_readout_step_idx"] < len(prev_seq) - 1:
            n_shortcircuit_opportunities += 1
            if r["persistent_trajectory_id"] == prev["persistent_trajectory_id"]:
                n_opportunity_ticks_continued += 1
    shortcircuit_opportunity_present = n_shortcircuit_opportunities > 0
    opportunity_continuation_rate = (
        n_opportunity_ticks_continued / n_shortcircuit_opportunities
        if shortcircuit_opportunity_present else None)

    if n_persistent_present < MIN_PERSISTENT_PRESENT_TICKS:
        cell_verdict = "insufficient_commitments"
    elif not shortcircuit_opportunity_present:
        cell_verdict = "no_shortcircuit_opportunity"
    elif opportunity_continuation_rate >= OPPORTUNITY_CONTINUATION_RATE_WORKING_FLOOR:
        cell_verdict = "shortcircuit_working_post_fix"
    elif opportunity_continuation_rate <= OPPORTUNITY_CONTINUATION_RATE_INERT_CEIL:
        cell_verdict = "shortcircuit_inert_still_broken"
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
        "n_genuine_commits_ticks_ratio_supplementary": n_genuine_commits_ticks_ratio,
        "mean_true_dwell_ticks": mean_true_dwell_ticks,
        "n_shortcircuit_opportunities": n_shortcircuit_opportunities,
        "n_opportunity_ticks_continued": n_opportunity_ticks_continued,
        "opportunity_continuation_rate": opportunity_continuation_rate,
        "shortcircuit_opportunity_present": shortcircuit_opportunity_present,
        "frac_e3_chunk_ticks_beta_elevated": frac_e3_chunk_ticks_beta_elevated,
        "n_eligible_between_tick_steps": n_eligible_between_tick,
        "n_frozen_between_tick_steps": n_frozen_between_tick,
        "frac_between_tick_steps_idx_frozen_supplementary": frac_between_tick_frozen,
        "official_readout_realised_lengths_supplementary": official_lengths,
        "official_readout_n_segments_supplementary": official_n_segments,
        "official_readout_mean_realised_length_supplementary":
            official_mean_realised_length,
        "official_readout_pct_length_one_supplementary": official_pct_length_one,
        "beta_gate_bistable": bool(
            getattr(agent.config.heartbeat, "beta_gate_bistable", False)),
        "cell_verdict": cell_verdict,
    }


def _run_cell(seed: int, *, n_train: int, n_probe: int) -> Dict[str, Any]:
    """One seed's formation phase, then instrumented probe phase, on ONE
    agent -- identical structure to V3-EXQ-871a's `_run_cell`."""
    flags = _arm_flags()
    train_every = max(1, min(30, n_train // 2))
    with arm_cell(seed,
                  config_slice=_config_slice(),
                  config_slice_declared=True,
                  script_path=Path(__file__)) as cell:
        print(f"Seed {seed} Condition {ARM_ID} (e3-reselection-shortcircuit retest)",
              flush=True)

        env = base.build_env(seed)
        agent = base.build_agent(env, flags)
        harness = StepHarness(agent, env, train_mode=False, seed=seed)

        # Verify BOTH flags are genuinely at their lineage default (False)
        # going INTO formation -- this is the exact assumption V3-EXQ-871
        # violated for the first flag; fail loudly rather than silently
        # perturb formation if _arm_flags() or base.build_agent() ever
        # changes underneath us.
        assert agent.config.use_persistent_committed_program_handle is False, (
            "V3-EXQ-871b precondition violated: "
            "use_persistent_committed_program_handle must be False (lineage "
            "default) entering _run_formation. Check _arm_flags()."
        )
        assert agent.config.use_e3_reselection_shortcircuit is False, (
            "V3-EXQ-871b precondition violated: "
            "use_e3_reselection_shortcircuit must be False (lineage default) "
            "entering _run_formation. Check _arm_flags()."
        )

        train = _run_formation(harness, agent, n_train,
                               progress_every=train_every,
                               label=f"{ARM_ID} seed={seed}")

        # Scope BOTH fix flags to the PROBE PHASE ONLY -- flipped on the live
        # agent config object here, AFTER formation, BEFORE the probe. Safe
        # because both flags are read LIVE via getattr(self.config, ...) on
        # every tick (agent.py:6749/7155/7204), never cached at construction
        # -- see module docstring "WIRING VERIFIED AGAINST ree-v3 HEAD".
        agent.config.use_persistent_committed_program_handle = True
        agent.config.use_e3_reselection_shortcircuit = True

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
            # Audit trail: confirms both flags really were False throughout
            # formation and True throughout the probe for this cell
            # (belt-and-braces alongside the asserts above).
            "use_persistent_committed_program_handle_scoped_to_probe": bool(
                agent.config.use_persistent_committed_program_handle),
            "use_e3_reselection_shortcircuit_scoped_to_probe": bool(
                agent.config.use_e3_reselection_shortcircuit),
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
    min_opportunities = min(r["n_shortcircuit_opportunities"] for r in rows)
    commitments_readiness_met = bool(
        min_persistent_present >= MIN_PERSISTENT_PRESENT_TICKS)
    opportunity_readiness_met = bool(min_opportunities > 0)
    readiness_met = commitments_readiness_met and opportunity_readiness_met

    definitive = {"shortcircuit_working_post_fix", "shortcircuit_inert_still_broken"}
    if not commitments_readiness_met:
        label = "substrate_not_ready_requeue"
    elif not opportunity_readiness_met:
        # The compounding gate defect the Step 4.5 red-team review found:
        # commitments existed but the short-circuit never had a chance to
        # fire (e.g. every commitment expired before the next E3 tick, as
        # was mechanically guaranteed at the pre-review chunk_max_size=2
        # design point). This is a design-readiness failure, not a
        # substrate-verdict label -- never route it to
        # shortcircuit_inert_still_broken, which would misattribute a
        # zero-opportunity design as a confirmed-inert substrate.
        label = "no_shortcircuit_opportunity_requeue"
    elif all(v == "shortcircuit_working_post_fix" for v in verdicts):
        label = "shortcircuit_working_confirmed"
    elif all(v == "shortcircuit_inert_still_broken" for v in verdicts):
        label = "shortcircuit_inert_confirmed"
    elif all(v in definitive for v in verdicts) and len(set(verdicts)) > 1:
        label = "mixed_across_seeds"
    else:
        label = "inconclusive"

    c1_sufficient_commitments = readiness_met
    c2_discrimination_reached = bool(
        label in ("shortcircuit_working_confirmed",
                   "shortcircuit_inert_confirmed"))
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
            "The probe agent must register some genuine ARC-071 "
            "CHUNK-SOURCED commitment during the probe, or there is nothing "
            "to compute the ratio DV over. Identical scoping rationale to "
            "V3-EXQ-871a's own precondition of the same name."),
        "control": (
            "Count of probe ticks with agent.e3._persistent_committed_trajectory "
            "not None AND metadata['source'] == 'arc071_chunk', worst seed."),
        "measured": float(min_persistent_present),
        "threshold": float(MIN_PERSISTENT_PRESENT_TICKS),
        "direction": "lower",
        "met": commitments_readiness_met,
    }, {
        "name": "shortcircuit_opportunity_present",
        "description": (
            "ADDED after the Step 4.5 red-team review found the "
            "pre-review chunk_max_size=2 design gave the short-circuit "
            "ZERO opportunities to ever fire (the commit-tick's own "
            "bookkeeping advances _committed_step_idx to 1 within the "
            "SAME tick, so a length-2 chunk always reads as already "
            "exhausted on every later tick). At least one E3 tick must "
            "occur while an unexpired chunk-sourced commitment with room "
            "left to continue AND beta elevated already existed, or a "
            "shortcircuit_inert reading would be indistinguishable from "
            "a design that never gave the fix a chance to act."),
        "control": (
            "Count of E3 ticks where the PRIOR tick already held a "
            "chunk-sourced commitment with existing_readout_step_idx < "
            "chunk_length - 1 and beta_elevated, worst seed."),
        "measured": float(min_opportunities),
        "threshold": 0.0,
        "comparator": ">",
        "direction": "lower",
        "met": opportunity_readiness_met,
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

    if label == "shortcircuit_working_confirmed":
        # The short-circuit measurably reduces the E3-tick-normalised
        # re-commit rate under REAL organic chunk formation -- supports
        # MECH-090 (the AT-tick gap is closed; the commit latch now holds
        # across E3's own internal re-deliberation cycles, not only between
        # ticks).
        direction_mech090 = "supports"
        direction_arc071 = "unknown"
    elif label == "shortcircuit_inert_confirmed":
        # Both fix flags enabled and the ratio is still ~1.0 -- the
        # short-circuit is not engaging in practice at this design point,
        # meaning MECH-090's own AT-tick gate-propagation architecture is
        # still violated at this interaction (narrow implementation gap, not
        # a challenge to the core mechanism -- see V3-EXQ-871a's own claim-
        # layer mapping, which applies unchanged).
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
        "discrimination_target": (
            "failure_autopsy_V3-EXQ-871a_2026-08-03.md Section 6 point 5 / "
            "Section 7 failure_record_entry -- validates ree-v3 main commit "
            "31ed78e (use_e3_reselection_shortcircuit). Routes on "
            "opportunity_continuation_rate (conditioned on E3 ticks where "
            "continuing was structurally possible), NOT the confounded "
            "official_readout_mean_realised_length V3-EXQ-871a used, NOR "
            "the autopsy's own literal unconditional n_genuine_commits/ "
            "n_e3_ticks (retained as a supplementary diagnostic only -- a "
            "second Step 4.5 red-team pass found that statistic diluted by "
            "non-opportunity ticks the fix cannot affect; see module "
            "docstring 'DESIGN')."
        ),
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
            "H1/H2 (the between-tick fast-path discrimination) is not live "
            "in this run either -- resolved by an ad hoc probe in 2026-07, "
            "and confirmed working post-fix by V3-EXQ-871a. This run tests "
            "the SEPARATE 'gap (b)' question V3-EXQ-871a diagnosed and "
            "the routed /implement-substrate fixed: does E3's AT-TICK "
            "re-selection now preferentially continue an existing unexpired "
            "ARC-071-chunk-sourced commitment, instead of unconditionally "
            "recreating a fresh trajectory object every E3 tick. "
            "label='shortcircuit_working_confirmed' means "
            "opportunity_continuation_rate clears "
            f"{OPPORTUNITY_CONTINUATION_RATE_WORKING_FLOOR} (well above the "
            "pre-fix 0.0 baseline -- every opportunity was squandered "
            "pre-fix, by V3-EXQ-871a's own exact-arithmetic finding) in "
            "every cell. label='shortcircuit_inert_confirmed' means the "
            f"rate stays at or below "
            f"{OPPORTUNITY_CONTINUATION_RATE_INERT_CEIL} in every cell -- "
            "statistically indistinguishable from the pre-fix 0.0 baseline, "
            "meaning the short-circuit is not engaging in practice despite "
            "both flags being live AND genuine opportunities being present "
            "(see the shortcircuit_opportunity_present precondition). "
            "n_genuine_commits_ticks_ratio_supplementary / "
            "official_readout_mean_realised_length_supplementary / "
            "mean_true_dwell_ticks / "
            "frac_between_tick_steps_idx_frozen_supplementary are retained "
            "as supplementary diagnostics only -- a second Step 4.5 "
            "red-team pass (fable) found the unconditional ratio's "
            "denominator (ALL E3 ticks) diluted by ticks the fix cannot "
            "possibly affect (fresh commits, natural completions, "
            "non-chunk selections), so even a perfectly working "
            "short-circuit could leave it well inside the working band; "
            "opportunity_continuation_rate is conditioned on exactly the "
            "ticks the fix CAN affect and is not subject to that dilution. "
            "These supplementary stats remain useful to confirm fix (a), "
            "the between-tick fallback, has not regressed."),
    }

    full_config: Dict[str, Any] = {
        "env": dict(base.ENV_KWARGS),
        "schedule": {"n_episodes_train": n_train, "steps_per_episode":
                     STEPS_PER_EPISODE, "n_episodes_probe": n_probe},
        "arm": ARM_ID,
        "arm_config_slice": _config_slice(),
        "thresholds": {
            "MIN_PERSISTENT_PRESENT_TICKS": MIN_PERSISTENT_PRESENT_TICKS,
            "OPPORTUNITY_CONTINUATION_RATE_WORKING_FLOOR":
                OPPORTUNITY_CONTINUATION_RATE_WORKING_FLOOR,
            "OPPORTUNITY_CONTINUATION_RATE_INERT_CEIL":
                OPPORTUNITY_CONTINUATION_RATE_INERT_CEIL,
            "RATIO_SHORTCIRCUIT_WORKING_CEIL_supplementary":
                RATIO_SHORTCIRCUIT_WORKING_CEIL_supplementary,
            "RATIO_SHORTCIRCUIT_INERT_FLOOR_supplementary":
                RATIO_SHORTCIRCUIT_INERT_FLOOR_supplementary,
            "PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR_supplementary":
                PERSISTENCE_RESTORED_MEAN_LENGTH_FLOOR_supplementary,
            "PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL_supplementary":
                PERSISTENCE_STILL_PINNED_MEAN_LENGTH_CEIL_supplementary,
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
        rate = row["opportunity_continuation_rate"]
        print(f"  seed={row['seed']} verdict={row['cell_verdict']} "
              f"opportunity_continuation_rate="
              f"{rate if rate is None else round(rate, 3)} "
              f"n_shortcircuit_opportunities={row['n_shortcircuit_opportunities']} "
              f"n_e3_ticks={row['n_e3_ticks']}", flush=True)
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
