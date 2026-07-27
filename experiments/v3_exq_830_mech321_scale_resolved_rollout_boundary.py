"""V3-EXQ-830 -- MECH-321 SCALE-RESOLVED ROLLOUT BOUNDARY DIAGNOSTIC.

DIAGNOSTIC. Executes the probe named in section 5b of
`REE_assembly/evidence/planning/mech321_decomposition_scale_scoping_spike_2026-07-27.md`,
on the EXISTING V3-EXQ-816 harness (816d's dose-escalated harshened env, verbatim).
No new environment work -- this reads the stream the 816 campaign already generates.
This probe does NOT weight any claim -- it BEARS ON MECH-321 / MECH-288 / ARC-070
(experiment_purpose=diagnostic, claim_ids=[]).

THE QUESTION. MECH-288 ships two qualitatively heterogeneous segmentation scales:
`fast` (PE-threshold z-score over z_world + z_self, per-tick) and `slow`
(BOCPD-Gaussian over z_goal, hazard 1/40). MECH-321 consumes exactly one of them
and collapses even that to a boolean. Do the fast and slow scales fire at
DISSOCIABLE rollout positions once the slow scale is given its stream?

WHY NOW -- THIS ATTACKS THE SATURATED END NOBODY HAS TRIED. The 816 campaign's
numbers: `decomp_fired_frac_arm1 = 1.0` (816d) with `vs_trigger_fires_total = 0`,
`pe_trigger_fires_total = 13`, `cofire_total = 0` (816c). MECH-321's declared
PRIMARY trigger (V_s drop, R1) has NEVER fired across the whole campaign; every
decomposition observed was driven by the fast scale alone, firing on essentially
every evaluation. A trigger that never fires and a trigger that always fires carry
the same discriminative signal: none. The campaign has spent two env doses on the
never-fires end (816b, 816d) and its two open axes are trigger SENSITIVITY
(H-env-underdrives-uncertainty) and trigger VALIDITY (H-vs-proxy-saturation).
This is a THIRD axis -- trigger COMPOSITION -- and the only one that attacks the
always-fires end. `decomp_fired_frac = 1.0` is opaque; "fast fires on 100% of
sweeps, slow on 3%" is a measurement.

SUBSTRATE (already built and landed; ree-v3 9a6e7f3976 on main -- NOT rebuilt here):
  * REEConfig flag `use_decomposition_scale_resolved_probe` (default False,
    bit-identical when OFF), wired at four sites in ree_core/utils/config.py.
    When True, HippocampalModule._evaluate_decomposition_ticks adds z_goal to the
    rollout latent_signature, giving MECH-288's slow BOCPD scale its stream.
  * PolicyDecomposition.get_state() counters decomp_n_boundary_fires_{fast,slow}
    and _cofire. Exact identity: fires_fast + fires_slow - cofire == n_boundary_fires.

THREE PRE-REGISTERED READINGS (spike section 5b decision table). ALL THREE ARE
RESULTS -- see PASS/FAIL SEMANTICS below for why none of them is a failure:
  scales_dissociable_on_rollout  -- low cofire, distinct positions. REE's substrate
      exhibits heterogeneous distinct-scale segmentation on the imagination stream
      -> register option (b), the MECH-level scale-differentiated-decomposition
      extension sketched in spike section 5c.
  slow_never_fires_on_rollout    -- z_goal does not vary informatively within a
      rollout; one effective scale on the imagination stream -> option (a) stands,
      close the design question.
  slow_fires_only_with_fast      -- two detectors, one signal; the heterogeneity is
      nominal -> option (a) stands, record the null.

PASS/FAIL SEMANTICS (deliberate, so the middle reading is INFORMATIVE not a failure).
The LOAD-BEARING criterion of this diagnostic is C_DECIDABLE -- did the instrument
work and return a reading -- NOT "did the slow scale fire". PASS = a decidable
reading was produced (any of the three above). FAIL = substrate_not_ready_requeue,
i.e. the instrument was not actually switched on. Making "slow fires" load-bearing
would have made spike outcome 2 -- a genuine, pre-named, publishable finding --
score as a vacuous_pass or a failure, which is exactly the mis-scoring the spike
warns against ("the middle one is a genuine possibility worth naming in advance:
z_goal is an integrator, so it may simply not move inside a single short rollout.
If so, that is itself the answer").

THE FAILURE MODE THIS DESIGN EXISTS TO AVOID (and its readiness gate).
The slow scale can be dead for FOUR independent reasons, and the substrate commit
fixed only the first. Each of the other three produces the SAME clean, confident
"slow never fires on rollout" reading -- spike outcome 2, which CLOSES the design
question -- while the truth is that the instrument was never switched on. All four
were traced during authoring; the last two were found only because the readiness
gate refused the smoke twice.
  (1) z_goal absent from the rollout latent_signature. Fixed in ree-v3 9a6e7f3976;
      this is what the probe flag switches on.
  (2) GoalState never CONSTRUCTED. `REEConfig.from_dims(z_goal_enabled=...)`
      DEFAULTS TO FALSE, so REEAgent.goal_state is None and `_current_z_goal` is
      None on every proposal tick. The probe would add `z_goal: None`, and
      MECH-288's BOCPD skips a None value exactly as if the key were absent
      (`if not sources: return False, 0.0, []`).
  (3) GoalState constructed but never SEEDED, because the seeding gate is out of
      range. GoalState.update() pulls z_goal toward z_world only when
      `benefit_exposure > benefit_threshold`; until that first pull z_goal is the
      zero vector, `is_active()` is False, and the agent hands `current_z_goal=None`
      anyway -- (2) again, one layer down.
  (4) THE SEEDING HOOK IS NEVER CALLED AT ALL -- the actual root cause here.
      z_goal is updated by exactly one entry point, the explicit
      `agent.update_z_goal(...)`. NOTHING in sense(), generate_trajectories(),
      select_action() or update_residue() touches it. The V3-EXQ-816 harness --
      which this probe reuses per the spike's "no new environment work" constraint
      -- hand-rolls its inner loop and OMITS that call entirely, because nothing
      in the 816 campaign reads z_goal. So on the inherited loop z_goal is
      unconditionally the zero vector for the whole run, at ANY threshold.
(2), (3) and (4) are all closed here, arm-symmetrically.

HOW (4) WAS FOUND, AND WHY IT MATTERS THAT THE GATE FOUND IT. Smoke 1
(z_goal_enabled=True) returned zgoal_present_frac = 0.0, gate red,
substrate_not_ready_requeue. That looked like cause (3), so benefit_threshold was
measured on this env dose (3 seeds x 60 episodes x 24 steps, random policy, 2941
steps): benefit_exposure is a hedonic EMA with p50=0.014, p90=0.045, p95=0.059,
MAX=0.1018 -- it exceeds from_dims's benefit_threshold=0.1 on 0.03% of steps, 1
step in 2941, so that default is indeed unreachable and was replaced (see below).
Smoke 2 STILL returned zgoal_present_frac = 0.0. That second refusal is what
exposed (4): under a random policy seed 23 crosses the 0.05 seeding level in
episode 0, so a live goal stream should have activated immediately, and its failure
to do so could no longer be a threshold problem. Without the gate, this probe would
have consumed a ~5-hour cloud run and reported a wiring artefact as a scientific
finding that CLOSES the design question -- twice over.

THE FIX, AND WHAT IT COSTS.
  * `agent.update_z_goal(...)` is now called once per env step in BOTH arms, in the
    canonical position (after generate_trajectories, before select_action) and
    KWARGS-ONLY. Kwargs-only is load-bearing, not style: a positional call collides
    with `latent` and raises TypeError every tick, which is the documented
    EXQ-471/475/483/490/524 cohort bug that `experiments/_harness.py` StepHarness
    exists to prevent (invariant 2). This probe keeps the 816 loop rather than
    migrating to StepHarness, so the invariant is honoured explicitly here.
  * SIDE EFFECT, DECLARED: update_z_goal is also the SD-024 benefit-attractor
    producer (it calls ResidueField.accumulate_benefit), so calling it populates
    benefit_rbf_field and therefore un-zeroes the SD-025 curiosity bonus in
    HippocampalModule._curiosity_bonus. That is a real behavioural change relative
    to 816d, it is IDENTICAL IN BOTH ARMS, and it is therefore not a confound of
    the probe-flag contrast -- but it does mean ARM_PROBE_OFF is NOT bit-identical
    to 816d's ARM_1 and must not be cited as a reproduction of it.
  * benefit_threshold=0.05 is retained: it is the substrate's OWN default in
    `REEConfig.enable_goal_stream()` (config.py:5095) -- the value the substrate
    uses whenever the goal stream is deliberately switched on, which is exactly
    what this run does -- and on the measured distribution it seeds on 8.1% of
    steps. Intermittent is what the slow BOCPD needs: the norm is pulled toward
    z_world on seeding steps and decays between them (decay_goal=0.005), so it
    genuinely MOVES. A threshold firing on every step would saturate the norm and
    be as dead as one that never fires. Cost, stated plainly: the goal is seeded by
    high-benefit EXPOSURE rather than only by resource CONTACT, so a slow-scale
    fire means "the goal latent shifted", not "a resource was consumed" -- which is
    the correct object here, since MECH-288's slow scale is a goal-SHIFT detector.

THE GATE IS KEPT REGARDLESS, and it asserts VARIATION not presence. The ON arm's
readiness precondition is the STD of the z_goal norm ACROSS SWEEPS -- the same
statistic the slow BOCPD actually routes on, since it change-points on shifts in
the summed per-stream `z.norm()`. A present-but-frozen z_goal therefore trips the
gate rather than masquerading as a refuted slow scale (the 643 magnitude-vs-range
defect). Per-cell benefit_exposure mean/max and benefit_seeding_step_frac are also
recorded, so if a slow-scale silence ever recurs its upstream cause is diagnosable
from the manifest without another run.

POSITION INSTRUMENTATION IS DRIVER-SIDE (no substrate increment). The get_state()
counters are RUN TOTALS -- they answer "how often", not "where", and the question
is a joint distribution over rollout position. `_evaluate_decomposition_ticks`
sweeps up to 8 ticks per candidate and BREAKS on the first trigger, so position is
recoverable without touching the substrate: this driver wraps
`HippocampalModule._evaluate_decomposition_ticks`, and inside one sweep wraps
`source.evaluate` to read the per-scale counter DELTAS across each call. The tick
index at which each delta lands is the rollout position. Nothing in ree_core is
modified; the wrapper is installed per cell and removed in a finally block.

  STRUCTURAL PROPERTY DECLARED IN ADVANCE (so it is a prediction, not a post-hoc
  discovery). Two facts constrain what "position" can mean here. (i) In
  vs_boundary trigger mode a boundary fire sets should_decompose, which BREAKS the
  sweep -- so at most ONE tick per sweep carries a fire, and it is the terminal
  tick. (ii) z_goal is CONSTANT across the ticks of one rollout (Trajectory carries
  no goal track), so the slow detector sees the identical value at every tick of a
  sweep and can therefore only change-point on the FIRST tick of a sweep in which
  the value shifted. Expect slow to be pinned near tick 0 while fast spreads over
  ticks 0..7. The load-bearing dissociation readout is therefore the SWEEP-LEVEL
  JOINT (fast-only / slow-only / cofire / neither), which is robust to that
  pinning; the within-sweep position histograms are co-recorded as the concrete
  secondary, and their asymmetry is expected rather than surprising.

KNOWN ASYMMETRY, BOUNDED NOT ASSUMED AWAY. The flag covers the PRE-COMMITMENT
sweep only (ree_core/hippocampal/module.py). MECH-321's MID-EXECUTION hook in
ree_core/agent.py builds its own {z_world, z_self} signature and is NOT extended,
so mid-execution ticks advance the fast rollout detector but can never advance the
slow one, diluting the slow fraction. get_state() reports
decomp_n_evaluated_precommit and decomp_n_evaluated_midexec separately, so this run
records BOTH the naive slow fraction (over all boundary fires) and the
dilution-corrected one (over precommit evaluations only, the only ticks on which
the slow scale is even reachable), plus midexec_dilution_frac itself. Extending the
mid-execution hook is a separate possible substrate increment -- NAMED HERE, NOT
DONE INLINE.

ARMS. Both arms run MECH-321 (use_policy_decomposition=True); the manipulation is
the probe flag alone.
  ARM_PROBE_OFF -- control. Reproduces the current single-scale behaviour; the slow
    scale is structurally dead so its per-scale counters read
    (n_boundary_fires, 0, 0) by construction.
  ARM_PROBE_ON  -- the manipulation.
The control arm is NOT optional and is not merely a fingerprint anchor: the flag is
NOT a pure diagnostic. With z_goal present the slow scale can newly reach
boundary.fired and therefore CHANGE DECISIONS, so ARM_PROBE_OFF is what allows any
behavioural delta (executed action sequence, net harm, decomposition rate) to be
attributed to the added scale rather than to run-to-run variation.

DV-SYMMETRY (Step 3 mandatory per-arm declaration -- name the symmetry group of
each arm's DV and state that the manipulation is not invariant under it):
  ARM_PROBE_OFF. No manipulation (it is the control), hence no symmetry trap. Its
    slow-scale zero is structural-by-construction and is USED AS SUCH -- criterion
    C3 asserts it, so a non-zero reading would indict the instrumentation rather
    than count as a finding.
  ARM_PROBE_ON. Load-bearing DV = the sweep-level joint over
    {fast-only, slow-only, cofire, neither} plus the per-scale terminal-tick
    histograms. Symmetry group = permutation of sweeps x permutation of seeds
    (the joint is a count over sweeps, symmetric in their order) x relabelling of
    the two scales' names. The manipulation -- adding a z_goal ENTRY to the
    latent_signature DICT -- is NOT invariant under it: MECH-288's BOCPD detector
    branches on whether ANY of its declared streams RESOLVES to a tensor
    (`if not sources: return False, 0.0, []`), so adding the key moves that
    detector's source set from empty to non-empty and changes which counts the
    joint receives. This is categorically NOT the 604c broadcast-scalar hazard:
    the manipulation is not a constant added uniformly across candidates, the DV
    is not an argmax / rank / order statistic over a candidate pool, and no
    monotone rescaling or unit permutation can annihilate a detector's source-set
    membership test.
  Statistic-survival check: the added stream survives into a COUNT of detector
    fires (the routed statistic), which is exactly what the readiness gate asserts
    variation in. It would NOT survive into a scale-blind fired:bool -- which is
    precisely the collapse the spike identifies as the defect, and precisely why
    this run reads the per-scale counters rather than boundary.fired.

NULLS DECLARED (GOV-FANOUT-1). A `slow_never_fires_on_rollout` reading does NOT
mean the slow scale is broken and does NOT weaken MECH-288: it means z_goal does
not vary informatively WITHIN the rollout window at this env dose and schedule, so
the imagination stream carries one effective timescale. It would close the spike's
heterogeneity design question in favour of option (a) and retire the section 5c
extension sketch. It would NOT license any statement about the observation stream,
where the slow scale is separately contracted. A `slow_fires_only_with_fast`
reading likewise leaves MECH-288 untouched and records the heterogeneity as
nominal at this grain.

SLEEP DRIVER: not applicable -- no sleep phase is entered in this run.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.module import HippocampalModule  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.baselines import mech321_scale_resolved_probe as base  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_830_mech321_scale_resolved_rollout_boundary"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []                       # diagnostic: bears on, does not weight
BEARS_ON = ["MECH-321", "MECH-288", "ARC-070"]

# No E3 last_* diagnostic is read in any per-env-step loop: the per-scale scale
# attribution comes from PolicyDecomposition counter DELTAS taken around each
# source.evaluate() call (a genuinely fresh evaluation every time, by
# construction), and the only per-step reads are region_vs, harm and the argmax
# action index -- none of them a latched E3 selector diagnostic.
E3_DIAGNOSTICS_STALENESS_EXEMPT = (
    "No agent.e3.last_* read anywhere. Scale attribution is read as counter deltas "
    "bracketing each PolicyDecomposition.evaluate() call, so every recorded observation "
    "corresponds to exactly one fresh evaluation; the driver additionally asserts "
    "recorder tick total == decomp_n_evaluated_precommit as a coverage precondition."
)
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "Load-bearing DV is a count over PolicyDecomposition evaluate() calls (sweep-level "
    "scale joint), not an E3 selection readout; the a_idx read feeds only a "
    "threshold-invariant action-sequence divergence check between arms."
)
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness preconditions are single-scalar reachable-by-construction floors "
    "(z_goal norm variation, z_goal presence fraction, instrumentation coverage, "
    "fast-scale positive-control fire count), each met by the dry-run smoke; none is "
    "a composite/narrow predicate that could be sub-maximal by construction."
)

SEEDS = [11, 23, 47, 71, 97]
ARM_OFF = "ARM_PROBE_OFF"
ARM_ON = "ARM_PROBE_ON"
ARMS = [ARM_OFF, ARM_ON]

# Schedule / env / decomposition params from the canonical baseline module
# (single source of truth so ARM_PROBE_OFF is minted for THIS sub-lineage).
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE
DECOMPOSITION_VS_THRESHOLD = base.DECOMPOSITION_VS_THRESHOLD
DECOMPOSITION_DEPTH_CAP = base.DECOMPOSITION_DEPTH_CAP

# Sweep bound in HippocampalModule._evaluate_decomposition_ticks (min(len(states), 8)).
MAX_SWEEP_TICKS = 8

# --- Pre-registered thresholds (defined HERE, never inferred post-hoc). ---
# READINESS (all four gate DECIDABILITY, the load-bearing criterion).
# z_goal norm VARIATION across sweeps -- the SAME statistic the slow BOCPD routes
# on (it change-points on shifts in the summed per-stream z.norm()). A present but
# frozen z_goal must trip this rather than read as a refuted slow scale.
ZGOAL_NORM_STD_FLOOR = 1e-4
# Fraction of ON-arm sweeps at which current_z_goal was a tensor at all.
ZGOAL_PRESENT_FRAC_FLOOR = 0.10
# Driver recorder must have observed essentially every precommit evaluation, else
# the position histogram is over an unknown subsample.
INSTRUMENTATION_COVERAGE_FLOOR = 0.99
# Positive control: the always-fires end must reproduce (fast scale fires at all).
FAST_FIRE_MIN = 1

# REPORTING criteria (non-load-bearing; these ROUTE among the three readings).
SLOW_FIRE_MIN_SWEEPS = 5          # pooled ON-arm sweeps carrying a slow fire
SLOW_FIRE_MIN_SEEDS = 2           # ... spread over at least this many seeds
COFIRE_MAX_FRAC = 0.50            # of slow fires, at most this fraction co-fire with fast
SLOW_ONLY_MIN_SWEEPS = 3          # pooled ON-arm sweeps where slow fired and fast did not


# ---------------------------------------------------------------------------
# Driver-side per-sweep scale/position recorder (no substrate change).
# ---------------------------------------------------------------------------
class SweepRecorder:
    """Records, per pre-commitment decomposition sweep, which MECH-288 scale
    fired and at which rollout tick index.

    Reads PolicyDecomposition's own per-scale counters as DELTAS bracketing each
    source.evaluate() call, so the semantics (including MECH-288's cross-scale
    rule, under which a slow fire SUPPRESSES the same-tick fast event yet still
    counts as a fast fire via suppressed_scales) are inherited from the substrate
    rather than re-derived here.
    """

    def __init__(self) -> None:
        self.n_sweeps = 0
        self.n_ticks = 0
        self.n_fast_only = 0
        self.n_slow_only = 0
        self.n_cofire = 0
        self.n_neither = 0
        self.fast_positions: Counter = Counter()
        self.slow_positions: Counter = Counter()
        self.sweep_lengths: Counter = Counter()
        self.zgoal_norms: List[float] = []
        self.n_sweeps_zgoal_present = 0
        self._tick = -1
        self._fast_tick: Optional[int] = None
        self._slow_tick: Optional[int] = None

    # -- sweep lifecycle -------------------------------------------------
    def begin_sweep(self, current_z_goal: Optional[torch.Tensor]) -> None:
        self._tick = -1
        self._fast_tick = None
        self._slow_tick = None
        if current_z_goal is not None:
            self.n_sweeps_zgoal_present += 1
            try:
                self.zgoal_norms.append(float(current_z_goal.detach().norm().item()))
            except Exception:
                pass

    def note_tick(self, fast: bool, slow: bool) -> None:
        self._tick += 1
        self.n_ticks += 1
        if fast and self._fast_tick is None:
            self._fast_tick = self._tick
        if slow and self._slow_tick is None:
            self._slow_tick = self._tick

    def end_sweep(self) -> None:
        self.n_sweeps += 1
        self.sweep_lengths[self._tick + 1] += 1
        f, s = self._fast_tick, self._slow_tick
        if f is not None:
            self.fast_positions[f] += 1
        if s is not None:
            self.slow_positions[s] += 1
        if f is not None and s is not None:
            self.n_cofire += 1
        elif f is not None:
            self.n_fast_only += 1
        elif s is not None:
            self.n_slow_only += 1
        else:
            self.n_neither += 1

    # -- readout ---------------------------------------------------------
    def _hist(self, counter: Counter) -> List[int]:
        return [int(counter.get(i, 0)) for i in range(MAX_SWEEP_TICKS)]

    def summary(self) -> Dict[str, Any]:
        norms = self.zgoal_norms
        return {
            "n_sweeps": int(self.n_sweeps),
            "n_recorder_ticks": int(self.n_ticks),
            "n_sweeps_fast_only": int(self.n_fast_only),
            "n_sweeps_slow_only": int(self.n_slow_only),
            "n_sweeps_cofire": int(self.n_cofire),
            "n_sweeps_neither": int(self.n_neither),
            "fast_position_hist": self._hist(self.fast_positions),
            "slow_position_hist": self._hist(self.slow_positions),
            "sweep_length_hist": [int(self.sweep_lengths.get(i, 0))
                                  for i in range(MAX_SWEEP_TICKS + 1)],
            "n_sweeps_zgoal_present": int(self.n_sweeps_zgoal_present),
            "zgoal_present_frac": (self.n_sweeps_zgoal_present / self.n_sweeps)
                                  if self.n_sweeps else 0.0,
            "zgoal_norm_mean": float(statistics.fmean(norms)) if norms else 0.0,
            "zgoal_norm_std": float(statistics.pstdev(norms)) if len(norms) > 1 else 0.0,
            "zgoal_norm_min": float(min(norms)) if norms else 0.0,
            "zgoal_norm_max": float(max(norms)) if norms else 0.0,
            "zgoal_norm_n": len(norms),
        }


_ORIG_EVAL_TICKS = HippocampalModule._evaluate_decomposition_ticks


def _install_recorder(recorder: SweepRecorder) -> None:
    """Patch HippocampalModule._evaluate_decomposition_ticks for this cell.

    Wraps the per-sweep entry point, and inside it shadows `source.evaluate`
    with an instance attribute so each call's per-scale counter delta can be
    attributed to a tick index. Both patches are removed in finally blocks.
    """
    def patched(self, *args, **kwargs):
        source = kwargs.get("source", args[0] if args else None)
        z_goal = kwargs.get("current_z_goal")
        if z_goal is None and len(args) >= 7:
            z_goal = args[6]
        if source is None:
            return _ORIG_EVAL_TICKS(self, *args, **kwargs)

        recorder.begin_sweep(z_goal)
        orig_evaluate = source.evaluate

        def wrapped_evaluate(*a, **kw):
            before_fast = int(getattr(source, "_n_boundary_fires_fast", 0))
            before_slow = int(getattr(source, "_n_boundary_fires_slow", 0))
            decision = orig_evaluate(*a, **kw)
            after_fast = int(getattr(source, "_n_boundary_fires_fast", 0))
            after_slow = int(getattr(source, "_n_boundary_fires_slow", 0))
            recorder.note_tick(after_fast > before_fast, after_slow > before_slow)
            return decision

        source.evaluate = wrapped_evaluate
        try:
            return _ORIG_EVAL_TICKS(self, *args, **kwargs)
        finally:
            try:
                del source.evaluate
            except AttributeError:
                pass
            recorder.end_sweep()

    HippocampalModule._evaluate_decomposition_ticks = patched


def _remove_recorder() -> None:
    HippocampalModule._evaluate_decomposition_ticks = _ORIG_EVAL_TICKS


# ---------------------------------------------------------------------------
def _arm_flags(arm: str) -> Dict[str, Any]:
    return dict(base.off_arm_flags() if arm == ARM_OFF else base.on_arm_flags())


def _config_slice(arm: str) -> Dict[str, Any]:
    if arm == ARM_OFF:
        # Single source of truth -- this sub-lineage's own minted control closure.
        return base.off_path_config_slice()
    slice_ = base.off_path_config_slice()
    slice_.update(_arm_flags(arm))
    return slice_


def _build_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=base.ALPHA_WORLD,
        **_arm_flags(arm),
    ))
    # Seed the identical crystallised chunk in BOTH arms (both can decompose it).
    agent.policy_chunking.library.register(ChunkedPrimitive(
        sequence=SEEDED_CHUNK_SEQUENCE, depth=1,
        state=ChunkState.CRYSTALLISED, selection_weight=1.0,
    ))
    return agent


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    total_eps = WARMUP_EPISODES + MEASURE_EPISODES
    # ARM_PROBE_OFF is minted reuse-ELIGIBLE (driver excluded from the hash) so a
    # future same-config successor matches by construction; ARM_PROBE_ON is a
    # treatment arm and is never reused as-is.
    fold_driver = arm != ARM_OFF
    recorder = SweepRecorder()
    with arm_cell(seed, config_slice=_config_slice(arm), script_path=Path(__file__),
                  include_driver_script_in_hash=fold_driver) as cell:
        env = CausalGridWorldV2(**base.env_kwargs(seed))
        agent = _build_agent(env, arm)
        wd = agent.config.latent.world_dim

        region_vs_samples: List[float] = []
        benefit_samples: List[float] = []
        action_seq: List[int] = []
        net_harm = 0.0
        n_measure_steps = 0
        low_vs_steps = 0
        max_n_streams = 0

        _install_recorder(recorder)
        try:
            for ep in range(total_eps):
                _, obs = env.reset()
                agent.reset()
                measuring = ep >= WARMUP_EPISODES
                for _ in range(STEPS_PER_EPISODE):
                    latent = agent.sense(obs["body_state"], obs["world_state"])
                    ticks = agent.clock.advance()
                    e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                          else torch.zeros(1, wd, device=agent.device))
                    cands = agent.generate_trajectories(latent, e1, ticks)

                    # --- THE CALL THE 816 HARNESS OMITS, AND WITHOUT WHICH THIS
                    # PROBE CANNOT MEASURE ANYTHING (see module docstring cause 4).
                    # z_goal is updated ONLY by this explicit hook -- nothing in
                    # sense() / generate_trajectories / update_residue touches it --
                    # so a loop that never calls it leaves z_goal at the zero vector
                    # forever, is_active() False, and current_z_goal None on every
                    # sweep. Canonical position (after generate_trajectories, before
                    # select_action) and KWARGS-ONLY, per experiments/_harness.py
                    # StepHarness invariant 2: a positional call collides with
                    # `latent` and raises TypeError every tick, which is the
                    # documented EXQ-471/475/483 cohort bug. Called IDENTICALLY in
                    # both arms, so it cannot confound the probe-flag contrast. ---
                    obs_body = obs["body_state"]
                    benefit_raw = obs.get("benefit_exposure", None)
                    if benefit_raw is None and torch.is_tensor(obs_body):
                        if obs_body.shape[-1] > 11:
                            benefit_raw = (obs_body[0, 11] if obs_body.dim() == 2
                                           else obs_body[11])
                    benefit_exposure = (0.0 if benefit_raw is None
                                        else max(0.0, float(benefit_raw)))
                    _rtype_raw = obs.get("resource_type_at_agent", None)
                    resource_type = None
                    if _rtype_raw is not None:
                        try:
                            resource_type = int(
                                _rtype_raw[0] if hasattr(_rtype_raw, "__len__")
                                else _rtype_raw)
                        except (TypeError, ValueError):
                            resource_type = None
                    agent.update_z_goal(
                        benefit_exposure=benefit_exposure,
                        drive_level=REEAgent.compute_drive_level(obs_body),
                        resource_type=resource_type,
                    )
                    if measuring:
                        benefit_samples.append(benefit_exposure)

                    action = agent.select_action(cands, ticks)
                    a_idx = int(action.argmax(dim=-1).item())
                    _flat, harm_signal, done, _info, obs = env.step(a_idx)
                    agent.update_residue(harm_signal)

                    if measuring:
                        region_vs = float(agent.hippocampal._region_vs())
                        region_vs_samples.append(region_vs)
                        n_streams = len(agent.hippocampal.per_stream_vs)
                        if n_streams > max_n_streams:
                            max_n_streams = n_streams
                        if region_vs < DECOMPOSITION_VS_THRESHOLD:
                            low_vs_steps += 1
                        net_harm += float(harm_signal)
                        n_measure_steps += 1
                        action_seq.append(a_idx)
                    if done:
                        break
                # Every 10 episodes AND on the last one. The final-episode clause
                # is not cosmetic: without it a short run (any --dry-run, or any
                # schedule not a multiple of 10) emits NO "ep N/M" line at all, so
                # the runner's progress contract goes unverified by the smoke --
                # exactly the case the skill's post-smoke instrumentation check is
                # meant to catch. M is the loop bound variable, never a constant.
                if (ep + 1) % 10 == 0 or (ep + 1) == total_eps:
                    st = agent.get_policy_decomposition_state()
                    print(f"  [train] scale seed={seed} arm={arm} ep {ep+1}/{total_eps} "
                          f"sweeps={recorder.n_sweeps} "
                          f"fast={st.get('decomp_n_boundary_fires_fast', 0)} "
                          f"slow={st.get('decomp_n_boundary_fires_slow', 0)} "
                          f"cofire={st.get('decomp_n_boundary_cofire', 0)}", flush=True)
        finally:
            _remove_recorder()

        st = agent.get_policy_decomposition_state()
        n_pre = int(st.get("decomp_n_evaluated_precommit", 0))
        n_mid = int(st.get("decomp_n_evaluated_midexec", 0))
        rec = recorder.summary()

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "n_measure_steps": int(n_measure_steps),
            # --- Scale-resolved rollout boundary readouts (the question). ---
            **rec,
            "decomp_n_boundary_fires": int(st.get("decomp_n_boundary_fires", 0)),
            "decomp_n_boundary_fires_fast": int(st.get("decomp_n_boundary_fires_fast", 0)),
            "decomp_n_boundary_fires_slow": int(st.get("decomp_n_boundary_fires_slow", 0)),
            "decomp_n_boundary_cofire": int(st.get("decomp_n_boundary_cofire", 0)),
            # --- Pre-commit vs mid-exec split: BOUNDS the known dilution rather
            # than assuming it away (the mid-execution hook is not extended, so
            # midexec ticks can advance fast but never slow). ---
            "decomp_n_evaluated_precommit": n_pre,
            "decomp_n_evaluated_midexec": n_mid,
            "midexec_dilution_frac": (n_mid / (n_pre + n_mid)) if (n_pre + n_mid) else 0.0,
            # Instrumentation coverage self-check: the recorder must have seen
            # essentially every precommit evaluation.
            "instrumentation_coverage": (rec["n_recorder_ticks"] / n_pre) if n_pre else 0.0,
            # --- Mechanism firing / campaign continuity context. ---
            "decomp_n_decomposed_precommit": int(st.get("decomp_n_decomposed_precommit", 0)),
            "decomp_n_decomposed_midexec": int(st.get("decomp_n_decomposed_midexec", 0)),
            "decomp_n_marked_unreliable": int(st.get("decomp_n_marked_unreliable", 0)),
            "decomp_n_vs_trigger": int(st.get("decomp_n_vs_trigger", 0)),
            "decomp_trigger_mode": st.get("decomp_trigger_mode"),
            # --- Region-V_s context (ladder continuity with 816/816b/816d). ---
            "low_vs_steps": int(low_vs_steps),
            "max_n_streams_tracked": int(max_n_streams),
            "region_vs_mean": (float(statistics.fmean(region_vs_samples))
                               if region_vs_samples else None),
            "region_vs_min": float(min(region_vs_samples)) if region_vs_samples else None,
            "region_vs_max": float(max(region_vs_samples)) if region_vs_samples else None,
            "region_vs_samples": [round(x, 6) for x in region_vs_samples],
            # --- Benefit exposure: the quantity that gates z_goal seeding, and
            # therefore the upstream cause of any slow-scale silence. Recorded so
            # a null is diagnosable without a re-run. ---
            "benefit_exposure_mean": (float(statistics.fmean(benefit_samples))
                                      if benefit_samples else None),
            "benefit_exposure_max": float(max(benefit_samples)) if benefit_samples else None,
            "benefit_seeding_step_frac": (
                sum(1 for b in benefit_samples if b > base.GOAL_BENEFIT_THRESHOLD)
                / len(benefit_samples)) if benefit_samples else 0.0,
            "goal_benefit_threshold": base.GOAL_BENEFIT_THRESHOLD,
            # --- Behavioural-delta attribution (why the control arm exists). ---
            "net_harm_per_step": (net_harm / n_measure_steps) if n_measure_steps else 0.0,
            "measure_action_seq": list(action_seq),
        }
        cell.stamp(row)

    # A cell "verdict" is a local readiness proxy for the runner progress bar,
    # not a claim verdict: did this cell exercise the instrument as intended?
    cell_ok = row["n_sweeps"] > 0 and row["instrumentation_coverage"] >= INSTRUMENTATION_COVERAGE_FLOOR
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Precondition specs. `applies_to` is what keeps a control-arm-meaningless gate
# from vacating the treatment arm (and vice versa) -- see the 785 regime-
# conditioning rule.
# ---------------------------------------------------------------------------
def _specs() -> List[PreconditionSpec]:
    on_only = lambda ctx: bool(ctx.get("probe_on"))  # noqa: E731
    return [
        PreconditionSpec(
            name="instrumentation_coverage",
            description=("READINESS: the driver-side sweep recorder must have observed "
                         "essentially every pre-commitment evaluation "
                         "(n_recorder_ticks / decomp_n_evaluated_precommit), else the "
                         "position histograms are over an unknown subsample."),
            control="worst cell in this arm",
            threshold=INSTRUMENTATION_COVERAGE_FLOOR,
            direction="lower",
        ),
        PreconditionSpec(
            name="fast_scale_fires",
            description=("READINESS positive control: the always-fires end must reproduce "
                         "-- the fast scale must fire at all in this arm, else there is no "
                         "saturated end to resolve and a slow-scale null is uninterpretable."),
            control="pooled fast-scale fires in this arm",
            threshold=float(FAST_FIRE_MIN) - 0.5,
            direction="lower",
        ),
        PreconditionSpec(
            name="zgoal_present_on_rollout",
            description=("READINESS: current_z_goal must actually be a tensor on the "
                         "rollout side. With z_goal_enabled False (the substrate default) "
                         "it is None, the BOCPD detector skips it exactly as if absent, "
                         "and 'slow never fires' would be a wiring artefact."),
            control="pooled fraction of ARM_PROBE_ON sweeps carrying a z_goal tensor",
            threshold=ZGOAL_PRESENT_FRAC_FLOOR,
            direction="lower",
            applies_to=on_only,
            applies_note=("ARM_PROBE_OFF never passes z_goal to boundary_on by design, so "
                          "its presence fraction is not the statistic that arm's readout "
                          "depends on; scoped out rather than failed."),
        ),
        PreconditionSpec(
            name="zgoal_norm_varies",
            description=("READINESS, SAME STATISTIC THE LOAD-BEARING DETECTOR ROUTES ON: "
                         "MECH-288's slow BOCPD change-points on shifts in the summed "
                         "per-stream z.norm(), so the readiness check asserts the STD of "
                         "the z_goal norm ACROSS SWEEPS, not its magnitude or presence. A "
                         "present-but-frozen z_goal must trip this gate rather than read "
                         "as a refuted slow scale (the 643 magnitude-vs-range defect)."),
            control="pooled ARM_PROBE_ON sweeps (positive control: z_goal is an integrator "
                    "and is expected to drift across agent ticks once benefit is received)",
            threshold=ZGOAL_NORM_STD_FLOOR,
            direction="lower",
            applies_to=on_only,
            applies_note=("ARM_PROBE_OFF's slow scale is structurally dead whatever z_goal "
                          "does, so z_goal variation is not meaningful for that arm's "
                          "readout; scoped out rather than failed."),
        ),
    ]


def _arm_ctx(arm: str) -> Dict[str, Any]:
    return {"id": arm, "probe_on": arm == ARM_ON}


def _pooled(rows: List[Dict[str, Any]], key: str) -> int:
    return int(sum(int(r.get(key, 0) or 0) for r in rows))


def _pool_hist(rows: List[Dict[str, Any]], key: str) -> List[int]:
    out = [0] * MAX_SWEEP_TICKS
    for r in rows:
        for i, v in enumerate(r.get(key) or []):
            if i < MAX_SWEEP_TICKS:
                out[i] += int(v)
    return out


def _hist_mean_position(hist: List[int]) -> Optional[float]:
    n = sum(hist)
    if n <= 0:
        return None
    return float(sum(i * c for i, c in enumerate(hist)) / n)


def run_experiment() -> Dict[str, Any]:
    # Design-time refusal BEFORE compute: no arm may carry a structurally
    # unsatisfiable gate (785). Both arms are scorable here -- the two z_goal
    # preconditions are scoped OUT of the control arm (disposition (a)), never
    # failed by it, so no arm is acknowledged vacuous.
    assert_no_structurally_unsatisfiable_gate(_specs(), [_arm_ctx(a) for a in ARMS])

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            rows.append(_run_cell(arm, seed))
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    off_rows, on_rows = by_arm[ARM_OFF], by_arm[ARM_ON]

    # --- Per-arm measured values for the gate (worst cell for coverage; pooled
    # for the count/fraction statistics). ---
    def _measured(arm: str) -> Dict[str, float]:
        arm_rows = by_arm[arm]
        zg_norms_present = _pooled(arm_rows, "n_sweeps_zgoal_present")
        n_sweeps = _pooled(arm_rows, "n_sweeps")
        # Pooled z_goal norm std across cells: take the max over cells of the
        # per-cell std. The routed statistic is WITHIN-cell drift (the detector
        # state is per-agent), so the positive control is "at least one cell's
        # z_goal genuinely moved"; a per-cell worst would gate on the least
        # informative agent rather than on whether the stream can move at all.
        stds = [float(r.get("zgoal_norm_std", 0.0) or 0.0) for r in arm_rows]
        return {
            "instrumentation_coverage": min(
                (float(r.get("instrumentation_coverage", 0.0) or 0.0) for r in arm_rows),
                default=0.0),
            "fast_scale_fires": float(_pooled(arm_rows, "decomp_n_boundary_fires_fast")),
            "zgoal_present_on_rollout": (zg_norms_present / n_sweeps) if n_sweeps else 0.0,
            "zgoal_norm_varies": max(stds) if stds else 0.0,
        }

    arm_gates = [
        evaluate_arm_gate(arm, _arm_ctx(arm), _specs(), _measured(arm))
        for arm in ARMS
    ]
    gate_by_arm = {g["arm"]: g for g in arm_gates}
    off_green = gate_by_arm[ARM_OFF]["gate_green"]
    on_green = gate_by_arm[ARM_ON]["gate_green"]

    # --- Control-arm structural check: the probe flag must be the ONLY thing
    # that lets the slow scale reach the rollout stream. A non-zero slow count
    # in ARM_PROBE_OFF indicts the instrumentation, it is not a finding. ---
    off_slow_total = _pooled(off_rows, "decomp_n_boundary_fires_slow")
    control_slow_silent = off_slow_total == 0

    # --- The sweep-level joint (LOAD-BEARING DISSOCIATION READOUT). ---
    on_sweeps = _pooled(on_rows, "n_sweeps")
    on_fast_only = _pooled(on_rows, "n_sweeps_fast_only")
    on_slow_only = _pooled(on_rows, "n_sweeps_slow_only")
    on_cofire = _pooled(on_rows, "n_sweeps_cofire")
    on_neither = _pooled(on_rows, "n_sweeps_neither")
    on_slow_sweeps = on_slow_only + on_cofire
    on_seeds_with_slow = sum(
        1 for r in on_rows if (r["n_sweeps_slow_only"] + r["n_sweeps_cofire"]) > 0)
    cofire_frac_of_slow = (on_cofire / on_slow_sweeps) if on_slow_sweeps else 0.0

    # --- Within-sweep position histograms (the concrete secondary; see the
    # module docstring's declared structural pinning of the slow scale). ---
    on_fast_hist = _pool_hist(on_rows, "fast_position_hist")
    on_slow_hist = _pool_hist(on_rows, "slow_position_hist")
    off_fast_hist = _pool_hist(off_rows, "fast_position_hist")

    # --- Dilution bound: the slow scale is reachable ONLY on precommit ticks. ---
    on_pre = _pooled(on_rows, "decomp_n_evaluated_precommit")
    on_mid = _pooled(on_rows, "decomp_n_evaluated_midexec")
    on_fires = _pooled(on_rows, "decomp_n_boundary_fires")
    on_slow_fires = _pooled(on_rows, "decomp_n_boundary_fires_slow")
    slow_frac_naive = (on_slow_fires / on_fires) if on_fires else 0.0
    slow_frac_precommit_corrected = (on_slow_fires / on_pre) if on_pre else 0.0

    # --- Pre-registered REPORTING criteria (non-load-bearing; they route the
    # label among the three publishable readings). ---
    c_slow_fires = (on_slow_sweeps >= SLOW_FIRE_MIN_SWEEPS
                    and on_seeds_with_slow >= SLOW_FIRE_MIN_SEEDS)
    c_dissociable = (c_slow_fires
                     and cofire_frac_of_slow <= COFIRE_MAX_FRAC
                     and on_slow_only >= SLOW_ONLY_MIN_SWEEPS)

    # --- LOAD-BEARING criterion: did the instrument work and return a reading? ---
    c_decidable = bool(on_green and off_green and control_slow_silent)

    # --- Behavioural delta (why the control arm is not optional). ---
    action_divergence_seeds = sum(
        1 for o, n in zip(sorted(off_rows, key=lambda r: r["seed"]),
                          sorted(on_rows, key=lambda r: r["seed"]))
        if o["measure_action_seq"] != n["measure_action_seq"])

    # --- Self-route (a diagnostic HYPOTHESIS; adjudicated by /failure-autopsy
    # or /governance, never self-applied to a claim). ---
    if not c_decidable:
        label = "substrate_not_ready_requeue"
        if not on_green:
            degeneracy_reason = (
                "ARM_PROBE_ON readiness gate red ("
                + ", ".join(gate_by_arm[ARM_ON]["failed_preconditions"])
                + "); the slow scale's stream was not demonstrably live, so a "
                  "slow-never-fires reading here would be a wiring artefact, not a finding")
        elif not off_green:
            degeneracy_reason = (
                "ARM_PROBE_OFF readiness gate red ("
                + ", ".join(gate_by_arm[ARM_OFF]["failed_preconditions"])
                + "); without a sound control arm no behavioural delta is attributable")
        else:
            degeneracy_reason = (
                f"control arm recorded {off_slow_total} slow-scale fires with the probe "
                "flag OFF, where the flag is the only path z_goal has to the rollout "
                "stream -- this indicts the instrumentation, it is not a finding")
    elif not c_slow_fires:
        label = "slow_never_fires_on_rollout"
        degeneracy_reason = None
    elif c_dissociable:
        label = "scales_dissociable_on_rollout"
        degeneracy_reason = None
    else:
        label = "slow_fires_only_with_fast"
        degeneracy_reason = None

    outcome = "PASS" if c_decidable else "FAIL"

    metrics = {
        # LOAD-BEARING decidability.
        "decidable_reading_produced": c_decidable,
        "arm_probe_on_gate_green": on_green,
        "arm_probe_off_gate_green": off_green,
        "control_arm_slow_silent": control_slow_silent,
        "control_arm_slow_fires_total": off_slow_total,
        # The sweep-level joint -- the dissociation readout.
        "on_n_sweeps": on_sweeps,
        "on_n_sweeps_fast_only": on_fast_only,
        "on_n_sweeps_slow_only": on_slow_only,
        "on_n_sweeps_cofire": on_cofire,
        "on_n_sweeps_neither": on_neither,
        "on_n_sweeps_with_slow": on_slow_sweeps,
        "on_n_seeds_with_slow": on_seeds_with_slow,
        "cofire_frac_of_slow": cofire_frac_of_slow,
        "slow_sweep_frac": (on_slow_sweeps / on_sweeps) if on_sweeps else 0.0,
        "fast_sweep_frac": ((on_fast_only + on_cofire) / on_sweeps) if on_sweeps else 0.0,
        # Within-sweep position (secondary; slow is expected pinned near tick 0).
        "on_fast_position_hist": on_fast_hist,
        "on_slow_position_hist": on_slow_hist,
        "off_fast_position_hist": off_fast_hist,
        "on_fast_mean_position": _hist_mean_position(on_fast_hist),
        "on_slow_mean_position": _hist_mean_position(on_slow_hist),
        # Mid-execution dilution, bounded rather than assumed away.
        "on_n_evaluated_precommit": on_pre,
        "on_n_evaluated_midexec": on_mid,
        "on_midexec_dilution_frac": (on_mid / (on_pre + on_mid)) if (on_pre + on_mid) else 0.0,
        "on_slow_frac_naive": slow_frac_naive,
        "on_slow_frac_precommit_corrected": slow_frac_precommit_corrected,
        # Readiness measurements (also carried as preconditions).
        "on_zgoal_present_frac": _measured(ARM_ON)["zgoal_present_on_rollout"],
        "on_zgoal_norm_std_best_cell": _measured(ARM_ON)["zgoal_norm_varies"],
        "on_instrumentation_coverage_worst": _measured(ARM_ON)["instrumentation_coverage"],
        "off_instrumentation_coverage_worst": _measured(ARM_OFF)["instrumentation_coverage"],
        "on_fast_fires_total": _pooled(on_rows, "decomp_n_boundary_fires_fast"),
        # Campaign continuity (the always-fires end this probe attacks).
        "on_decomp_fired_frac": (
            sum(1 for r in on_rows
                if (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1)
            / len(on_rows)) if on_rows else 0.0,
        "off_decomp_fired_frac": (
            sum(1 for r in off_rows
                if (r["decomp_n_decomposed_precommit"] + r["decomp_n_marked_unreliable"]) >= 1)
            / len(off_rows)) if off_rows else 0.0,
        "on_vs_trigger_total": _pooled(on_rows, "decomp_n_vs_trigger"),
        "off_vs_trigger_total": _pooled(off_rows, "decomp_n_vs_trigger"),
        # Behavioural delta attribution (the flag is NOT a pure diagnostic).
        "n_seeds_action_seq_differs": action_divergence_seeds,
        "off_net_harm_per_step_mean": statistics.fmean(
            [r["net_harm_per_step"] for r in off_rows]) if off_rows else 0.0,
        "on_net_harm_per_step_mean": statistics.fmean(
            [r["net_harm_per_step"] for r in on_rows]) if on_rows else 0.0,
        # Pre-registered thresholds, echoed for audit.
        "slow_fire_min_sweeps": SLOW_FIRE_MIN_SWEEPS,
        "slow_fire_min_seeds": SLOW_FIRE_MIN_SEEDS,
        "cofire_max_frac": COFIRE_MAX_FRAC,
        "slow_only_min_sweeps": SLOW_ONLY_MIN_SWEEPS,
    }

    # interpretation.preconditions is the FLAT UNION of both arms' applied
    # preconditions, deliberately NOT the green-arms-only adjudication list.
    # In this design ARM_PROBE_ON is the sole load-bearing arm: a green control
    # arm answers nothing on its own, so a red ON arm must vacate the WHOLE run
    # rather than leave a scorable remainder. Scoped-out entries are carried
    # separately in per_arm_gate so the exclusions stay auditable.
    flat_preconditions: List[Dict[str, Any]] = []
    for g in arm_gates:
        flat_preconditions.extend(g["preconditions"])

    interpretation = {
        "label": label,
        "preconditions": flat_preconditions,
        "preconditions_scope_note": (
            "Flat union of BOTH arms' applied preconditions. ARM_PROBE_ON is the sole "
            "load-bearing arm (the control arm's slow-scale zero is structural), so a red "
            "ON gate vacates the whole run rather than leaving a scorable remainder. "
            "Per-arm detail, including preconditions scoped out of the control arm, is at "
            "top level under per_arm_gate."),
        "criteria": [
            {"name": "C_DECIDABLE_instrument_returned_a_reading", "load_bearing": True,
             "passed": c_decidable},
            {"name": "C_SLOW_FIRES_on_rollout", "load_bearing": False,
             "passed": c_slow_fires},
            {"name": "C_DISSOCIABLE_low_cofire_distinct_positions", "load_bearing": False,
             "passed": c_dissociable},
            {"name": "C_CONTROL_slow_silent_with_flag_off", "load_bearing": False,
             "passed": control_slow_silent},
        ],
        "criteria_non_degenerate": {
            # Decidability is non-degenerate iff both arms actually ran sweeps
            # (there was something to gate on at all).
            "C_DECIDABLE": bool(on_sweeps > 0 and _pooled(off_rows, "n_sweeps") > 0),
            # The routing criteria are only non-degenerate once the instrument is
            # certified live -- otherwise a zero slow count is a wiring artefact.
            "C_SLOW_FIRES": c_decidable,
            "C_DISSOCIABLE": bool(c_decidable and c_slow_fires),
            "C_CONTROL": bool(_pooled(off_rows, "n_sweeps") > 0),
        },
        "null_reading_guide": {
            "scales_dissociable_on_rollout":
                "REE's substrate exhibits heterogeneous distinct-scale segmentation on the "
                "imagination stream -> register option (b), the MECH-level "
                "scale-differentiated-decomposition extension sketched in spike section 5c "
                "(fast/PE boundaries re-segment at fine grain, slow/goal-BOCPD boundaries "
                "trigger a coarser response). MECH-321 would become the fast-scale member.",
            "slow_never_fires_on_rollout":
                "z_goal does not vary informatively WITHIN a rollout; one effective scale on "
                "the imagination stream -> option (a) stands, close the design question and "
                "retire the section 5c sketch. Says nothing about the OBSERVATION stream, "
                "where the slow scale is separately contracted, and does not weaken MECH-288.",
            "slow_fires_only_with_fast":
                "Two detectors, one signal; the heterogeneity is nominal at this grain -> "
                "option (a) stands, record the null. Also does not weaken MECH-288.",
            "substrate_not_ready_requeue":
                "The instrument was not demonstrably switched on (z_goal absent or frozen on "
                "the rollout side, recorder coverage short, or the control arm's slow scale "
                "non-silent). Re-queue with the failing precondition repaired -- do NOT read "
                "this as any of the three scientific outcomes.",
        },
        "follow_on_named_not_done": (
            "MECH-321's MID-EXECUTION hook (ree_core/agent.py) builds its own "
            "{z_world, z_self} signature and is NOT extended by this flag, so midexec ticks "
            "advance the fast rollout detector but never the slow one. That dilution is "
            "BOUNDED here (on_midexec_dilution_frac, on_slow_frac_precommit_corrected), not "
            "removed. Extending the mid-execution hook is a separate /implement-substrate "
            "increment and is deliberately NOT done inline."),
    }

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": interpretation,
        "per_arm_gate": {
            "green": [g["arm"] for g in arm_gates if g["gate_green"]],
            "red": [g["arm"] for g in arm_gates if not g["gate_green"]],
            "failed_preconditions_by_arm": {
                g["arm"]: g["failed_preconditions"] for g in arm_gates},
            "scoped_out_by_arm": {g["arm"]: g["scoped_out"] for g in arm_gates},
            "load_bearing_arm": ARM_ON,
        },
        "non_degenerate": c_decidable,
        "degeneracy_reason": degeneracy_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()

    global SEEDS, WARMUP_EPISODES, MEASURE_EPISODES
    if args.dry_run:
        SEEDS = [11, 23]
        WARMUP_EPISODES = 2
        MEASURE_EPISODES = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS,
        "warmup_episodes": WARMUP_EPISODES, "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "max_sweep_ticks": MAX_SWEEP_TICKS,
        "decomposition_vs_threshold": DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": DECOMPOSITION_DEPTH_CAP,
        "goal_benefit_threshold": base.GOAL_BENEFIT_THRESHOLD,
        # Provenance of that value, recorded so no reader has to re-derive it:
        # from_dims defaults to 0.1, which this env dose exceeds on 0.03% of
        # steps (measured, 2941 random-policy steps, max benefit_exposure
        # 0.1018) -- unreachable. 0.05 is enable_goal_stream()'s own default and
        # seeds on 8.1% of steps. See the baseline module docstring.
        "goal_benefit_threshold_provenance": {
            "value_source": "REEConfig.enable_goal_stream default (config.py:5095)",
            "from_dims_default_rejected": 0.1,
            "measured_steps": 2941,
            "measured_max_benefit_exposure": 0.10177,
            "measured_frac_steps_above_0p1": 0.0003,
            "measured_frac_steps_above_0p05": 0.0813,
        },
        "harsh_env_drift_interval": base.HARSH_ENV_DRIFT_INTERVAL,
        "harsh_world_rule_shift_enabled": base.HARSH_WORLD_RULE_SHIFT_ENABLED,
        "harsh_world_rule_shift_interval": base.HARSH_WORLD_RULE_SHIFT_INTERVAL,
        "harsh_world_rule_shift_depth": base.HARSH_WORLD_RULE_SHIFT_DEPTH,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
        "zgoal_norm_std_floor": ZGOAL_NORM_STD_FLOOR,
        "zgoal_present_frac_floor": ZGOAL_PRESENT_FRAC_FLOOR,
        "instrumentation_coverage_floor": INSTRUMENTATION_COVERAGE_FLOOR,
        "fast_fire_min": FAST_FIRE_MIN,
        "slow_fire_min_sweeps": SLOW_FIRE_MIN_SWEEPS,
        "slow_fire_min_seeds": SLOW_FIRE_MIN_SEEDS,
        "cofire_max_frac": COFIRE_MAX_FRAC,
        "slow_only_min_sweeps": SLOW_ONLY_MIN_SWEEPS,
        "arm_config_slices": {a: _config_slice(a) for a in ARMS},
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "design_question": "mech321_decomposition_scale_heterogeneity",
        "evidence_direction": result["evidence_direction"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "per_arm_gate": result["per_arm_gate"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": None,
        "cites": {
            "scoping_spike": "mech321_decomposition_scale_scoping_spike_2026-07-27 (section 5b)",
            "substrate_commit": "ree-v3 9a6e7f3976",
            "harness_predecessors": ["V3-EXQ-816", "V3-EXQ-816b", "V3-EXQ-816c", "V3-EXQ-816d"],
            "cluster_autopsy": "failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26",
        },
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"direction: {result['evidence_direction']} non_degenerate: {result['non_degenerate']}",
          flush=True)
    print(f"decidable={m['decidable_reading_produced']} on_gate={m['arm_probe_on_gate_green']} "
          f"off_gate={m['arm_probe_off_gate_green']} control_slow_silent={m['control_arm_slow_silent']}",
          flush=True)
    print(f"joint: sweeps={m['on_n_sweeps']} fast_only={m['on_n_sweeps_fast_only']} "
          f"slow_only={m['on_n_sweeps_slow_only']} cofire={m['on_n_sweeps_cofire']} "
          f"neither={m['on_n_sweeps_neither']}", flush=True)
    print(f"slow: sweeps_with_slow={m['on_n_sweeps_with_slow']} seeds={m['on_n_seeds_with_slow']} "
          f"cofire_frac={m['cofire_frac_of_slow']:.4g} "
          f"frac_precommit_corrected={m['on_slow_frac_precommit_corrected']:.4g}", flush=True)
    print(f"position: fast_hist={m['on_fast_position_hist']} slow_hist={m['on_slow_position_hist']}",
          flush=True)
    print(f"zgoal: present_frac={m['on_zgoal_present_frac']:.4g} "
          f"norm_std_best={m['on_zgoal_norm_std_best_cell']:.6g} "
          f"coverage={m['on_instrumentation_coverage_worst']:.4g}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
